"""
Tests for SupplyMind task loading and registry.

Validates that all 3 tasks are registered correctly with the expected
configurations (IDs, episode lengths, budgets, graph files).
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.tasks.registry import TaskRegistry, TaskDefinition


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_registry():
    """Reset and re-register tasks for each test to ensure clean state."""
    TaskRegistry.reset()
    TaskRegistry.register_all()
    yield
    TaskRegistry.reset()


# ──────────────────────────────────────────────
# Registry: registration and listing
# ──────────────────────────────────────────────

class TestTaskRegistry:
    """Test that all tasks are registered and retrievable."""

    def test_register_all_registers_3_tasks(self) -> None:
        """TaskRegistry.register_all() should register exactly 3 tasks."""
        tasks = TaskRegistry.list_tasks()
        assert len(tasks) == 3

    def test_task_ids_are_valid(self) -> None:
        """All expected task IDs should be retrievable."""
        expected_ids = [
            "easy_typhoon_response",
            "medium_multi_front",
            "hard_cascading_crisis",
        ]
        for tid in expected_ids:
            task = TaskRegistry.get(tid)
            assert task.task_id == tid

    def test_unknown_task_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            TaskRegistry.get("nonexistent_task")

    def test_list_tasks_ordered_by_difficulty(self) -> None:
        """Tasks should be sorted easy -> medium -> hard."""
        tasks = TaskRegistry.list_tasks()
        difficulties = [t.difficulty for t in tasks]
        assert difficulties == ["easy", "medium", "hard"]

    def test_register_all_is_idempotent(self) -> None:
        """Calling register_all() multiple times should not duplicate tasks."""
        TaskRegistry.register_all()
        TaskRegistry.register_all()
        tasks = TaskRegistry.list_tasks()
        assert len(tasks) == 3


# ──────────────────────────────────────────────
# Easy task configuration
# ──────────────────────────────────────────────

class TestEasyTask:
    """Test the easy Typhoon Response task definition."""

    def test_episode_length(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        assert task.episode_length == 30

    def test_budget(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        assert task.budget == 5_000_000.0

    def test_difficulty(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        assert task.difficulty == "easy"

    def test_name(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        assert task.name == "Typhoon Response"

    def test_graph_file_exists(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        path = os.path.join(PROJECT_ROOT, task.graph_file)
        assert os.path.isfile(path), f"Graph file not found: {path}"

    def test_disruption_file_exists(self) -> None:
        task = TaskRegistry.get("easy_typhoon_response")
        path = os.path.join(PROJECT_ROOT, task.disruption_file)
        assert os.path.isfile(path), f"Disruption file not found: {path}"


# ──────────────────────────────────────────────
# Medium task configuration
# ──────────────────────────────────────────────

class TestMediumTask:
    """Test the medium Multi-Front Crisis task definition."""

    def test_episode_length(self) -> None:
        task = TaskRegistry.get("medium_multi_front")
        assert task.episode_length == 45

    def test_budget(self) -> None:
        task = TaskRegistry.get("medium_multi_front")
        assert task.budget == 8_000_000.0

    def test_difficulty(self) -> None:
        task = TaskRegistry.get("medium_multi_front")
        assert task.difficulty == "medium"

    def test_graph_file_exists(self) -> None:
        task = TaskRegistry.get("medium_multi_front")
        path = os.path.join(PROJECT_ROOT, task.graph_file)
        assert os.path.isfile(path), f"Graph file not found: {path}"

    def test_disruption_file_exists(self) -> None:
        task = TaskRegistry.get("medium_multi_front")
        path = os.path.join(PROJECT_ROOT, task.disruption_file)
        assert os.path.isfile(path), f"Disruption file not found: {path}"


# ──────────────────────────────────────────────
# Hard task configuration
# ──────────────────────────────────────────────

class TestHardTask:
    """Test the hard Cascading Crisis task definition."""

    def test_episode_length(self) -> None:
        task = TaskRegistry.get("hard_cascading_crisis")
        assert task.episode_length == 60

    def test_budget(self) -> None:
        task = TaskRegistry.get("hard_cascading_crisis")
        assert task.budget == 10_000_000.0

    def test_difficulty(self) -> None:
        task = TaskRegistry.get("hard_cascading_crisis")
        assert task.difficulty == "hard"

    def test_graph_file_exists(self) -> None:
        task = TaskRegistry.get("hard_cascading_crisis")
        path = os.path.join(PROJECT_ROOT, task.graph_file)
        assert os.path.isfile(path), f"Graph file not found: {path}"

    def test_disruption_file_exists(self) -> None:
        task = TaskRegistry.get("hard_cascading_crisis")
        path = os.path.join(PROJECT_ROOT, task.disruption_file)
        assert os.path.isfile(path), f"Disruption file not found: {path}"
