import math
from typing import Optional, Tuple

import torch

from .config import LLMModelConfig


def _compute_default_rope_parameters(
    config: Optional[LLMModelConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta_
        partial_rotary_factor = (
            config.partial_rotary_factor_
            if config.partial_rotary_factor_ is not None
            else 1.0
        )
        head_dim = (
            config.dim_ // config.n_heads_
            if config.head_dim_ is None
            else config.head_dim_
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}