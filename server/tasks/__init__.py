"""
SupplyMind Task Definitions

Provides the task registry and individual task registration functions.
"""

from server.tasks.registry import TaskDefinition, TaskRegistry
from server.tasks.task_easy import register_easy_task
from server.tasks.task_medium import register_medium_task
from server.tasks.task_hard import register_hard_task

__all__ = [
    "TaskDefinition",
    "TaskRegistry",
    "register_easy_task",
    "register_medium_task",
    "register_hard_task",
]
