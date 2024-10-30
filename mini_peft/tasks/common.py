import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import evaluate as hf_evaluate

import torch

from moe_peft.common import InputData


class BasicMetric:
    def __init__(self) -> None:
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


class AutoMetric(BasicMetric):
    def __init__(self, task_name: str) -> None:
        super().__init__()
        path_prefix = os.getenv("MOE_PEFT_METRIC_PATH")
        if path_prefix is None:
            path_prefix = ""
        elif not path_prefix.endswith(os.sep):
            path_prefix += os.sep

        if ":" in task_name:
            split = task_name.split(":")
            self.metric_ = hf_evaluate.load(path_prefix + split[0], split[1])
        else:
            self.metric_ = hf_evaluate.load(path_prefix + task_name)

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        self.metric_.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, Any]:
        return self.metric_.compute()


class BasicTask:
    def __init__(self) -> None:
        pass

    @property
    def peft_task_type(self) -> str:
        pass

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        pass

    def loading_metric(self) -> BasicMetric:
        pass

    def init_kwargs(self) -> Dict:
        return {}


# Common Sense
class CommonSenseTask(BasicTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_type_ = "common_sense"
        self.label_dtype_ = None

    @property
    def peft_task_type(self) -> str:
        return "QUESTION_ANS"

    def label_list(self) -> List[str]:
        pass

