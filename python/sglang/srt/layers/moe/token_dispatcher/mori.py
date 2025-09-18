from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

#from sglang.srt.layers.moe.token_dispatcher.base_dispatcher import BaseDispatcher
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.utils import MoRIEPMode, get_deepep_config, is_tbo_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
    is_hip,
    is_npu,
    load_json_config,
)
import torch
_is_npu = is_npu()

# TODO: 
# 1. Find subsitution of deep_ep Buffer and Config
# 2. Change the deep_ep conditions into mori_ep conditions
try: 
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig, 
        EpDispatchCombineOp, 
        EpDispatchCombineKernelType,
    )
    
    # NOTE: Could we need to support 'npu' devices?
    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_mori = True
except ImportError:
    use_mori = False


from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


# --------------------------- MoRI Dispatch Output ---------------------------------
# TODO: Change the output format to meet MoRI Dispatch output
# * [ ] MoRINormalOutput
# * [V] MoRILLOutput

class MoRINormalOutput(NamedTuple):
    """MoRI normal dispatch output."""

    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    # hidden_states_scale
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_NORMAL
    
class MoRILLOutput(NamedTuple):
    """MoRI low latency dispatch output."""

    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    scales: torch.Tensor
    num_recv_tokens_per_expert: List[int] 

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_LL

assert isinstance(MoRINormalOutput, DispatchOutput)
assert isinstance(MoRILLOutput, DispatchOutput)


# ----------------------------- MoRI Combine Input ----------------------------------
# TODO: Change the input format to meet MoRI Combine input

class MoRINormalCombineInput(NamedTuple):
    """MoRI normal combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORI_NORMAL
    
class MoRILLCombineInput(NamedTuple):
    """MoRI low latency combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORI_LL

assert isinstance(MoRINormalCombineInput, CombineInput)
assert isinstance(MoRILLCombineInput, CombineInput)


class MoRIDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()

# NOTE: Actually... not needed, only skeleton code
#       actual implementation should be done inside of MoRIDispatcher
class MoRIBuffer:
    pass

# NOTE: Actually... not needed, only skeleton code
#       actual implementation should be done inside of MoRIDispatcher
class MoRIConfig(BaseDispatcherConfig):
    pass
       
        
"""
NOTE & TODO:
We should modify the interface of dispatch & combine following the mori's.
Check the below function interface.

def dispatch(
    self,
    input: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    block_num: int = -1,
    warp_per_block: int = -1,
):
    ...

def combine(
    self,
    input: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    block_num: int = -1,
    warp_per_block: int = -1,
    call_reset: bool = True,
):
    ...
"""
class _MoRIDispatcherImplBase:
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
    ):
        if not use_mori:
            raise ImportError(
                "MoRI is not installed. Pleas install MoRI package from "
                "https://github.com/ROCm/mori."
            )
        self.config = config
        self._ops_handle = EpDispatchCombineOp(config)
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        use_fp8_w8a8: bool = False,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        raise NotImplementedError

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

class _MoRIDispatcherImplNormal(_MoRIDispatcherImplBase):
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
    ):
        super().__init__(config, moriep_mode)
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError("mori normal mode is currently not supported.")
    
    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError("mori normal mode is currently not supported.")

class _MoRIDispatcherImplLowLatency(_MoRIDispatcherImplBase):
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
    ):
        super().__init__(config, moriep_mode)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        use_fp8_w8a8: bool = False,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        topk_idx = topk_idx.to(torch.int64)
        scales = None
        if use_fp8_w8a8:
            from aiter import (
                get_hip_quant,
                QuantType,
            )
            quant_dtype = fp8_dtype
            quant_type = QuantType.per_128x128
            quant_func = get_hip_quant(quant_type)
            hidden_states, scales = quant_func(
                hidden_states,
                quant_dtype=quant_dtype,
            )            
            
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = self._ops_handle.dispatch(
            input=hidden_states,
            weights=topk_weights,
            scales=scales,
            indicies=topk_idx
        )
    
        return MoRILLOutput(
            hidden_states=dispatch_output, 
            topk_idx=dispatch_indices,
            topk_weights=dispatch_weights,
            scales=dispatch_scales,
            num_recv_tokens_per_expert=dispatch_recv_num_token
        )
    
    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        use_fp8_w8a8: bool = False,
    ):
        # NOTE: Question: Does combine process not need original number of tokens?
        #       I can't find any reorder and slicing procedure to get original-sized 
        #       'hidden_states' with using aiter.
        
        if use_fp8_w8a8 or _use_aiter:
            output = hidden_states
        else:
            raise RuntimeError(f"We currently supports aiter kernel only.")

        try:
            hidden_states, combined_weights = self._ops_handle.combine(
                input=hidden_states,
                weights=topk_weights,
                indices=topk_idx,
            )
            
            return hidden_states

        except Exception as e:
            logger.error(f"mori combine failed: {e}")
            raise RuntimeError(f"mori combine failed: {e}") from e


# TODO: should Implement MORI dispatcher, below is nonsense code for removal
# NOTE: to implement TBO with MoRI, first use same function name and logic flow
#       of DeepEPDispatcher (check _use_aiter or _use_hip in deepep.py)
class MoRIDispatcher(BaseDispatcher):   
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        use_fp8_w8a8: bool = False,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        moriep_mode: MoRIEPMode = MoRIEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        if not use_mori:
            raise ImportError(
                "MoRI is not installed. Pleas install MoRI package from "
                "https://github.com/ROCm/mori."
            )
        
        # TODO: Clean the unused APIs
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_world_group,
            get_tp_group,
            get_moe_expert_parallel_world_size,
            in_the_same_node_as,
        )
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            is_dp_attention_enabled,
        )

        self.group = group # We may need change the group to _EP_MOE
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.quant_dtype = quant_dtype
        self._internode = False
        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), f"NCCL backend not support inter-node communication"
        if not all(in_the_same_node_as(group, source_rank=0)):
            self._internode = True          

        self.rank = (
            get_attention_tp_rank()
            if is_dp_attention_enabled()
            else get_tp_group().rank_in_group
        )
        self.world_size = get_tp_group().world_size
        
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128
        )
        
        self.config = self._make_mori_config(
            data_type=params_dtype,
            hidden_dim=hidden_size,
            rank=self.rank,
            world_size=self.world_size,
            max_num_tokens=self.num_max_dispatch_tokens_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=router_topk,
        )
        
        # NOTE:
        # 1. mori does not use recv hook like deepep. So there is no distinguish 
        #    of each stages
        # 2. Currently, mori has low-latency mode only but high-throughput 
        #    mode will be added later. So, the 'normal' mode should be change 
        #    to 'ht' (high-throughput) mode in the future.
        self.moriep_mode = moriep_mode
        if self.moriep_mode.enable_normal():
            self._normal_dispatcher = _MoRIDispatcherImplNormal(
                self.config,
                self.moriep_mode
            )
        if self.moriep_mode.enable_low_latency():
            self._low_latency_dispatcher = _MoRIDispatcherImplLowLatency(
                self.config,
                self.moriep_mode
            )
        logger.debug(
            f"[rank:{self.rank}] mori dispatcher created with configs: "
            f"max_num_tokens={self.max_dispatch_tokens_per_rank}, "
            f"num_local_experts={num_local_experts}, "
            f"num_experts_per_token={router_topk}, "
            f"hidden_size={hidden_size}"
        )
        
    # NOTE: Moved to MoRIEPMoE class
    # def _init_mori_shmem():
    #     pass
    
    def _make_mori_config(
        self,
        data_type: torch.dtype = torch.bfloat16,
        hidden_dim: int,
        rank: int,
        world_size: int,
        max_num_tokens: int,
        num_experts_per_rank: int,
        num_experts_per_token: int,
        scale_dim: int = 0,
        scale_type_size: int = 0,        
    ):
        
        # Determine data type size
        dtype_to_size = {
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.float16: 2,
        }
        max_token_type_size = dtype_to_size.get(data_type, 2)

        config = EpDispatchCombineConfig(
            data_type=data_type,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_tokens,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,

            # Performance tuning parameters (can be optimized later)
            warp_num_per_block=8,  # Good default for MI300X
            block_num=80,          # Good default for MI300X
            max_token_type_size=max_token_type_size,

            # Quantization support (disabled for now)
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,

            # Use internal buffer management
            use_external_inp_buf=False,

            # Determine kernel type based on topology
            kernel_type=(EpDispatchCombineKernelType.InterNode
                        if self._internode
                        else EpDispatchCombineKernelType.IntraNode)
        )

        return config
          
    #def dispatch(self, *args, **kwargs) -> DispatchOutput:
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> DispatchOutput:
        ret = self._get_impl().dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            use_fp8_w8a8=self.use_fp8_w8a8,
            quant_dtype=self.quant_dtype,
        )
        return ret
    
    #def conbine(self, *args, **kwargs) -> Tuple:
    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple:
        ret = self._get_impl(forward_batch).combine(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            use_fp8_w8a8=self.use_fp8_w8a8,
        )
        return ret
        
    def _get_impl(self, forward_batch: ForwardBatch) -> _MoRIDispatcherImplBase:
        resolved_moriep_mode = self.moriep_mode.resolve(
            forward_batch.is_extend_in_batch
        )
        if resolved_moriep_mode == MoRIEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_moriep_mode == MoRIEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid moriep_mode: {self.moriep_mode}")
    
    
    