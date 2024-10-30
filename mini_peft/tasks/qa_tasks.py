import logging
from typing import List, Optional

import datasets as hf_datasets
import torch

from moe_peft.common import InputData

from .common import AutoMetric, BasicMetric, CommonSenseTask


class QuestionAnswerTask(CommonSenseTask):
    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels_ = labels
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def label_list(self) -> List[str]:
        return self.labels_

    def loading_metric(self) -> BasicMetric:
        return AutoMetric("accuracy")


class BoolQ(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["true", "false"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset(
            "google/boolq" if path is None else path,
            trust_remote_code=True,
        )["train" if is_train else "validation"]
        logging.info("Preparing data for BoolQ")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please answer the following question with true or false: "
                + f"{data_point['question']}?\nAnswer:"
            )
            answer = "true" if data_point["answer"] else "false"
            if is_train:
                prompt += f" {answer}"
                labels = None
            else:
                labels = [self.labels2id_[answer]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret
    

def update_task_dict(task_dict):
    task_dict.update(
        {
            "boolq": BoolQ(),
        }
    )