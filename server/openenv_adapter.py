"""
OpenEnv SDK Integration Layer

Wraps SupplyMindEnvironment in the official openenv.core.Environment base
class, uses TrajectoryRubric for grading, and exposes create_app() which
provides both REST and WebSocket (/ws, /mcp) endpoints automatically.

This module is imported by app.py to register the OpenEnv-native app.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from uuid import uuid4

from openenv.core import Environment
from openenv.core.rubrics import TrajectoryRubric, RubricDict

from models import SupplyMindAction, SupplyMindObservation, SupplyMindState
from server.engine.simulation import SimulationEngine
from server.tasks.registry import TaskRegistry, TaskDefinition
from server.graders.grader import EpisodeGrader


# ---------------------------------------------------------------------------
# Rubric: wraps our existing EpisodeGrader in the OpenEnv TrajectoryRubric
# ---------------------------------------------------------------------------


class SupplyMindRubric(TrajectoryRubric):
    """
    OpenEnv-compliant rubric that delegates to our existing EpisodeGrader.

    TrajectoryRubric accumulates the full trajectory and evaluates at the end.
    We use this to bridge our grader (which needs the full episode history and
    engine state) into the OpenEnv rubric framework.
    """

    def __init__(self) -> None:
        super().__init__(intermediate_reward=0.0)
        self._task_id: str = "easy_typhoon_response"
        self._engine: Optional[SimulationEngine] = None

        # Register task-specific sub-rubrics for introspection
        self.tasks = RubricDict({})

    def set_context(self, task_id: str, engine: SimulationEngine) -> None:
        """Set the current task and engine (called by the environment on reset)."""
        self._task_id = task_id
        self._engine = engine

    def score_trajectory(self, trajectory: list[tuple[Any, Any]]) -> float:
        """Score the full trajectory using our existing EpisodeGrader."""
        if self._engine is None:
            return 0.0
        grader = EpisodeGrader(self._task_id)
        score = grader.grade(trajectory, self._engine)
        return score

    def compute_step_rewards(self) -> list[float]:
        """Equal credit assignment across all steps."""
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score / len(self._trajectory)] * len(self._trajectory)


# ---------------------------------------------------------------------------
# Environment: OpenEnv Environment[ActT, ObsT, StateT] subclass
# ---------------------------------------------------------------------------


class OpenEnvSupplyMind(Environment[SupplyMindAction, SupplyMindObservation, SupplyMindState]):
    """
    OpenEnv-compliant Environment subclass for SupplyMind.

    Implements the official Environment[ActT, ObsT, StateT] generic protocol
    with reset(), step(), state(), and close() methods. Uses SupplyMindRubric
    for grading via the OpenEnv rubric framework.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        rubric = SupplyMindRubric()
        super().__init__(rubric=rubric)
        TaskRegistry.register_all()
        self.engine: Optional[SimulationEngine] = None
        self.current_task: Optional[TaskDefinition] = None
        self._state: SupplyMindState = SupplyMindState()
        self._episode_history: list[tuple[SupplyMindAction, SupplyMindObservation]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplyMindObservation:
        """Reset environment for a new episode."""
        task_id = kwargs.get("task_id", "easy_typhoon_response")
        task = TaskRegistry.get(task_id)
        self.current_task = task

        ep_id = episode_id or str(uuid4())
        if seed is not None:
            episode_seed = seed % (2**31)
            jitter_enabled = True
        else:
            episode_seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**31)
            jitter_enabled = False

        self.engine = SimulationEngine(
            graph_file=task.graph_file,
            disruption_file=task.disruption_file,
            budget=task.budget,
            max_steps=task.episode_length,
            min_episode_days=task.min_episode_days,
            seed=episode_seed,
            jitter_enabled=jitter_enabled,
        )

        self._state = SupplyMindState(
            episode_id=ep_id,
            step_count=0,
            task_id=task.task_id,
            task_name=task.name,
            task_difficulty=task.difficulty,
            total_steps=task.episode_length,
            is_done=False,
            cumulative_reward=0.0,
        )

        self._episode_history = []

        # Reset rubric and set context
        self._reset_rubric()
        if isinstance(self.rubric, SupplyMindRubric):
            self.rubric.set_context(task_id, self.engine)

        return self.engine.get_initial_observation()

    def step(
        self,
        action: SupplyMindAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupplyMindObservation:
        """Execute one step in the environment."""
        if self.engine is None:
            raise RuntimeError("Call reset() before step().")

        obs = self.engine.step(action)

        self._state.step_count += 1
        self._state.cumulative_reward += obs.reward
        self._state.is_done = obs.done

        self._episode_history.append((action, obs))

        return obs

    def state(self, **kwargs: Any) -> SupplyMindState:
        """Return current episode metadata."""
        return self._state

    def close(self) -> None:
        """Cleanup resources."""
        self.engine = None

    def grade(self) -> dict:
        """Grade the current episode (bridge method for compatibility)."""
        if self.engine is None:
            raise RuntimeError("No episode to grade.")
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


# ---------------------------------------------------------------------------
# App factory: creates the OpenEnv-native FastAPI app with WebSocket support
# ---------------------------------------------------------------------------


def register_openenv_routes(app) -> None:
    """
    Register OpenEnv SDK routes (/ws, /mcp WebSocket) on an existing FastAPI app.

    This adds WebSocket support to our custom app.py while keeping all existing
    REST endpoints intact.
    """
    from openenv.core.env_server import HTTPEnvServer

    server = HTTPEnvServer(
        env=OpenEnvSupplyMind,
        action_cls=SupplyMindAction,
        observation_cls=SupplyMindObservation,
        max_concurrent_envs=10,
    )
    # Register only the WebSocket routes on our existing app
    # mode="simulation" enables /reset, /step, /state + /ws + /mcp
    server.register_routes(app, mode="simulation")
