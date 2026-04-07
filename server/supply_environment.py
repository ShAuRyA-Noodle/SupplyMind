"""
SupplyMind Environment

High-level environment class that ties together the simulation engine,
task registry, and graders. This is the main interface used by the
FastAPI application -- all game logic lives here, not in the HTTP layer.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4
from typing import Optional

from models import SupplyMindAction, SupplyMindObservation, SupplyMindState
from server.engine.simulation import SimulationEngine
from server.tasks.registry import TaskRegistry, TaskDefinition
from server.graders.grader import EpisodeGrader


class SupplyMindEnvironment:
    """
    OpenEnv-compliant environment for supply chain risk management.

    Wraps SimulationEngine with episode management, task selection,
    and grading. The FastAPI app.py delegates all logic to this class.

    Lifecycle:
        1. __init__() -- registers tasks
        2. reset(task_id) -- creates engine, returns initial observation
        3. step(action) -- advances simulation, returns observation
        4. grade() -- scores the completed episode
        5. Repeat from 2 for next episode
    """

    def __init__(self) -> None:
        """Initialize the environment and register all built-in tasks."""
        TaskRegistry.register_all()
        self.engine: Optional[SimulationEngine] = None
        self.current_task: Optional[TaskDefinition] = None
        self._state: SupplyMindState = SupplyMindState()
        self._episode_history: list[tuple[SupplyMindAction, SupplyMindObservation]] = []

    def reset(
        self,
        task_id: str = "easy_typhoon_response",
        seed: Optional[int] = None,
    ) -> SupplyMindObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to run. Must be one of the registered task IDs.
            seed: Optional episode seed. When provided, enables scenario jitter
                  for episode variation (different seeds = different episodes).
                  When None, uses deterministic seed from task_id for backward-
                  compatible reproducible behavior.

        Returns:
            Initial observation of the supply chain state.

        Raises:
            ValueError: If task_id is not registered.
        """
        task = TaskRegistry.get(task_id)
        self.current_task = task

        # Seed logic:
        # - No seed provided: derive deterministically from task_id (backward compat)
        # - Seed provided: use it directly AND enable scenario jitter
        episode_id = str(uuid4())
        if seed is not None:
            episode_seed = seed % (2**31)
            jitter_enabled = True
        else:
            episode_seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**31)
            jitter_enabled = False

        # Create a fresh simulation engine for this episode
        self.engine = SimulationEngine(
            graph_file=task.graph_file,
            disruption_file=task.disruption_file,
            budget=task.budget,
            max_steps=task.episode_length,
            min_episode_days=task.min_episode_days,
            seed=episode_seed,
            jitter_enabled=jitter_enabled,
        )

        # Initialize episode state tracking
        self._state = SupplyMindState(
            episode_id=episode_id,
            step_count=0,
            task_id=task.task_id,
            task_name=task.name,
            task_difficulty=task.difficulty,
            total_steps=task.episode_length,
            is_done=False,
            cumulative_reward=0.0,
        )

        # Clear history for the new episode
        self._episode_history = []

        # Get the initial observation from the engine
        initial_obs = self.engine.get_initial_observation()
        return initial_obs

    def step(self, action: SupplyMindAction) -> SupplyMindObservation:
        """
        Execute one step in the environment.

        Args:
            action: The action to take this step.

        Returns:
            Observation after the action is applied and the simulation advances.

        Raises:
            RuntimeError: If the engine has not been initialized (call reset first).
            RuntimeError: If the episode is already done.
        """
        if self.engine is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() before step()."
            )
        if self._state.is_done:
            # Return the last observation with done=True instead of crashing.
            # This is graceful behavior: calling step() after done is a no-op.
            from models import SupplyMindObservation, FinancialSnapshot, ActionResult
            return SupplyMindObservation(
                current_day=self._state.step_count,
                days_remaining=0,
                financials=FinancialSnapshot(
                    budget_remaining=self.engine.financial.budget_remaining,
                    budget_total=self.engine.financial.budget_total,
                ),
                last_action_result=ActionResult(
                    success=False,
                    message="Episode is already done. Call reset() to start a new episode.",
                    cost=0.0,
                ),
                reward=0.0,
                done=True,
                info={"post_done": True},
            )

        # Execute the step in the simulation engine
        obs = self.engine.step(action)

        # Update episode state
        self._state.step_count += 1
        self._state.cumulative_reward += obs.reward
        self._state.is_done = obs.done

        # Record in history for grading
        self._episode_history.append((action, obs))

        return obs

    @property
    def state(self) -> SupplyMindState:
        """Return the current episode state metadata."""
        return self._state

    def grade(self) -> dict:
        """
        Grade the completed (or in-progress) episode.

        Runs the task-specific grader over the full episode history and
        returns a detailed score breakdown.

        Returns:
            Dict with keys: task_id, task_name, difficulty, score,
            steps_taken, cumulative_reward, breakdown.

        Raises:
            RuntimeError: If no episode has been run.
        """
        if self.engine is None:
            raise RuntimeError(
                "No episode to grade. Call reset() and run an episode first."
            )

        grader = EpisodeGrader(self._state.task_id)
        score = grader.grade(self._episode_history, self.engine)

        return {
            "task_id": self._state.task_id,
            "task_name": self._state.task_name,
            "difficulty": self._state.task_difficulty,
            "score": score,
            "steps_taken": self._state.step_count,
            "total_steps": self._state.total_steps,
            "cumulative_reward": round(self._state.cumulative_reward, 4),
            "is_done": self._state.is_done,
            "breakdown": grader.get_breakdown(),
        }

    @property
    def episode_history(self) -> list[tuple[SupplyMindAction, SupplyMindObservation]]:
        """Return the episode history (read-only access for testing)."""
        return list(self._episode_history)
