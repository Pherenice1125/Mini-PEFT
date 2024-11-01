import copy
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
# from mini_peft.adapters import ...
from mini_peft.common import (
    CHECKPOINT_CLASSES,
    AdapterConfig,
    Linear,
    LLMCache,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    LLMModelOutput,
    LLMMoeBlock,
    LLMOutput,
    LoraConfig,
    unpack_router_logits,
)
from mini_peft.executors import executor
from mini_peft.models import from_pretrained
from mini_peft.tasks import SequenceClassificationTask, task_dict
from mini_peft.utils import is_package_available