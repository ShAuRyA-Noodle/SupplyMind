"""
SupplyMind Task Registry

Central registry for all task definitions. Tasks are registered at startup
and looked up by task_id when the environment is reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable definition of a single SupplyMind task."""

    task_id: str
    name: str
    difficulty: str  # easy, medium, hard
    description: str
    episode_length: int  # max steps
    budget: float  # USD
    graph_file: str  # path to supply chain graph JSON
    disruption_file: str  # path to disruption scenario JSON
    min_episode_days: int  # minimum days before early termination allowed


class TaskRegistry:
    """
    Singleton-style class registry for SupplyMind tasks.

    All methods are classmethods so the registry is shared across the process.
    Tasks are registered once at startup via register_all().
    """

    _tasks: dict[str, TaskDefinition] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, task: TaskDefinition) -> None:
        """Register a task definition. Overwrites if task_id already exists."""
        cls._tasks[task.task_id] = task

    @classmethod
    def get(cls, task_id: str) -> TaskDefinition:
        """
        Retrieve a task definition by ID.

        Raises:
            ValueError: If task_id is not registered.
        """
        if task_id not in cls._tasks:
            available = list(cls._tasks.keys())
            raise ValueError(
                f"Unknown task: '{task_id}'. Available tasks: {available}"
            )
        return cls._tasks[task_id]

    @classmethod
    def list_tasks(cls) -> list[TaskDefinition]:
        """Return all registered task definitions, ordered by difficulty."""
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        return sorted(
            cls._tasks.values(),
            key=lambda t: difficulty_order.get(t.difficulty, 99),
        )

    @classmethod
    def register_all(cls) -> None:
        """
        Register all built-in tasks. Safe to call multiple times
        (idempotent after first registration).
        """
        if cls._initialized:
            return
        from server.tasks.task_easy import register_easy_task
        from server.tasks.task_medium import register_medium_task
        from server.tasks.task_hard import register_hard_task

        register_easy_task()
        register_medium_task()
        register_hard_task()
        cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._tasks.clear()
        cls._initialized = False
