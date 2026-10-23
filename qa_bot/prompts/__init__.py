"""prompts 包：提示词模板集中地（2.3 起）。"""
from .base import (
    SYSTEM_TEMPLATE,
    TASK_BOOK_TEMPLATE,
    build_system_prompt,
    build_task_book,
)

__all__ = [
    "SYSTEM_TEMPLATE",
    "TASK_BOOK_TEMPLATE",
    "build_system_prompt",
    "build_task_book",
]
