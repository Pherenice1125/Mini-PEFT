import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypeAlias, Union

import torch

Tokens: TypeAlias = List[int]
Labels: TypeAlias = List[int]
Masks: TypeAlias = List[bool]

@dataclass
class Prompt:
    instruction: str = None
    input: str = None
    label: str = None


@dataclass
class InputData:
    inputs: List[Union[Prompt, List[str], str]] = None
    tokens: Optional[Tokens] = None
    labels: Optional[Labels] = None


@dataclass
class LLMModelConfig:
    name_or_path_: str = None
    device_: str = None
    dim_: int = None
    head_dim_: int = None
    intermediate_: int = None
    n_heads_: int = None
    n_kv_heads_: int = None
    n_layers_: int = None
    hidden_act_: str = None
    hidden_dropout_: float = None
    vocab_size_: int = None
    pad_token_id_: int = None
    rope_theta_: float = None
    partial_rotary_factor_: float = None
    max_seq_len_: int = None
    # eager or flash_attn
    attn_implementation_: str = "eager"
    # data type
    dtype_: torch.dtype = None


@dataclass
class LLMModelOutput:
    adapter_name: str = None
    logits: torch.Tensor = None
    router_logits: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None
    # for internal use
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    loss_fn_: Callable = None


@dataclass
class LLMBatchConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1

# They're not a efficient_operator_factory anymore...

@dataclass
class LLMModelInput:
    batch_configs_: List[LLMBatchConfig] = None
    batch_tokens_: List[Tokens] = None
    batch_labels_: List[Labels] = None
    batch_masks_: List[Masks] = None

    output_router_logits_: bool = True

    gradient_checkpoint_: str = "none"
    # efficient_operator_: bool = field(default_factory=_efficient_operator_factory)
    inference_mode_: bool = False


@dataclass
class AdapterConfig:
    adapter_name: str = ""
    task_name: str = "casual"

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            adapter_name=config.get("name", None),
            task_name=config.get("task_name", None),
        )
    

lora_target_modules = {
# Normally, we use Llama-2-7b or Gemma-2b as the pretrained models in this demo. They have a similar structure.
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
}