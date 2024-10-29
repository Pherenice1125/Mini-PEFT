from typing import Dict, List, Tuple

import torch

from moe_peft.executors import executor

from .abstracts import LLMFeedForward, LLMMoeBlock
from .config import LLMModelInput
from .lora_linear import Linear


class FeedForward(torch.nn.Module):
    def __init__(self, mlp: LLMFeedForward) -> None:
        super().__init__()
        self.mlp_: LLMFeedForward = mlp
        # mix of experts
        self.moes_: Dict[str, LLMMoeBlock] = {}

    def state_dict(self) -> Dict[str, Linear]:
        return self.mlp_.state_dict()

    def forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> Tuple[torch.Tensor, List]:
        if len(self.moes_) == 0:
            return self.mlp_._batch_forward(data, input_args), []
        else:
            return self._moe_forward(data, input_args)