"""2.4 自动化测评包。对外暴露裁判、评测集与运行器。"""
from .judge import Judge
from .dataset import EVAL_DATASET
from .runner import run_evaluation, main as run_eval_main

__all__ = ["Judge", "EVAL_DATASET", "run_evaluation", "run_eval_main"]
