import logging
import random
import sys
from abc import abstractmethod
from typing import Callable, Dict, List

import datasets

from .common import InputData, LLMBatchConfig, LLMModelInput, Masks, Tokens
from .tokenizer import Tokenizer