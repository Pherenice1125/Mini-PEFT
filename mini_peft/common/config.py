import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypeAlias, Union

import torch

Tokens: TypeAlias = List[int]
Labels: TypeAlias = List[int]
Masks: TypeAlias = List[bool]