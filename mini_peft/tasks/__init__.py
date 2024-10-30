from . import qa_tasks
from .common import(
    AutoMetric,
    BasicMetric,
    BasicTask,
    CommonSenseTask
)
from .qa_tasks import QuestionAnswerTask

#qa_tasks.update_task_dict(task_dict)

__all__ = [
    "BasicMetric",
    "AutoMetric",
    "BasicTask",
    "QuestionAnswerTask",
]