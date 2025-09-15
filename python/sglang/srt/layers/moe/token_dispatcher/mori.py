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
from sglang.srt.layers.moe.utils import DeepEPMode, get_deepep_config, is_tbo_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
    is_hip,
    is_npu,
    load_json_config,
)
_is_npu = is_npu()

# TODO: 
# 1. Find subsitution of deep_ep Buffer and Config
# 2. Change the deep_ep conditions into mori_ep conditions
try: 
    from mori.ops.dispatch_combine import EpDispatchCombineConfig, EpDispatchCombineOp, EpDispatchCombineKernelType
    
    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


# --------------------------- MoRI Dispatch Output ---------------------------------
# TODO: Change the output format to meet MoRI Dispatch output

class MoRINormalOutput(NamedTuple):
    """MoRI normal dispatch output."""

    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    # hidden_states_scale
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORIEP_NORMAL
    
class MoRILLOutput(NamedTuple):
    """MoRI low latency dispatch output."""

    hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORIEP_LL

assert isinstance(MoRINormalOutput, DispatchOutput)
assert isinstance(MoRILLOutput, DispatchOutput)


# ----------------------------- MoRI Combine Input ----------------------------------
# TODO: Change the input format to meet MoRI Combine input

class MoRINormalCombineInput(NamedTuple):
    """MoRI normal combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORIEP_NORMAL
    
class MoRILLCombineInput(NamedTuple):
    """MoRI low latency combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORIEP_LL

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
        

class _MoRIDispatcherImplBase:
    pass

class _MoRIDispatcherImplNormal(_MoRIDispatcherImplBase):
    pass

class _MoRIDispatcherImplLowLatency(_MoRIDispatcherImplBase):
    pass




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
        # deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        # TODO: should Implement MORI dispatcher, below is nonsense code for removal
        # NOTE: to implement TBO with MoRI, first use same function name and logic flow
        #       of DeepEPDispatcher (check _use_aiter or _use_hip in deepep.py)
        self.config = _make_mori_config(...)
        
        pass

    def _init_mori_shmem():
        pass
    
    def _make_mori_config(
        self,
        data_type: torch.dtype = torch.bfloat16,
        rank: int,
        world_size: int,
        hidden_dim: int,
        scale_dim: int,
        scale_type_size: int,
        max_token_type_size: int,
        max_num_inp_token_per_rank: int,
        num_experts_per_rank: int,
        num_experts_per_token: int,
        # warp_num_per_block: int = 8,
        # block_num: int = 80,
        # use_external_inp_buf: bool = True,
        # kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
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
            rank=self.dp_rank,  # Use dp_rank for expert parallelism
            world_size=self.dp_world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_tokens,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,

            # Performance tuning parameters (can be optimized later)
            warp_num_per_block=8,  # Good default for MI300X
            block_num=80,          # Good default for MI300X
            max_token_type_size=max_token_type_size,

            # Quantization support (disabled for now)
            scale_dim=0,
            scale_type_size=0,

            # Use internal buffer management
            use_external_inp_buf=False,

            # Determine kernel type based on topology
            kernel_type=(EpDispatchCombineKernelType.InterNode
                        if self.internode
                        else EpDispatchCombineKernelType.IntraNode)
        )

        return config
    