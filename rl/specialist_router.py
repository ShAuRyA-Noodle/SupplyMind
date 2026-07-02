"""
Task-Specialist Router — Meta-agent that picks the best model per task.

Instead of one model for all tasks, maintains a roster of task-specific
specialists and routes each task to the best performer.

This is production-grade architecture: companies don't use one model for
everything — they use specialized models per use case.

Usage:
    from rl.specialist_router import SpecialistRouter
    router = SpecialistRouter()
    router.load_specialists()
    action = router.predict(task_id, obs, action_mask)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
AUTORESEARCH_DIR = Path(__file__).resolve().parent / "autoresearch_results"


class SpecialistRouter:
    """Routes each task to the best specialist model.

    Maintains a registry of task -> model mappings, automatically
    selecting the best checkpoint based on AutoResearch results.
    Falls back to scripted agent if no specialist available.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.specialists: dict[str, Any] = {}
        self.specialist_scores: dict[str, float] = {}
        self.scripted_fallback = True

    def load_specialists(self) -> None:
        """Load best specialist for each task from checkpoints."""
        from rl.distributional.qr_dqn import QRDQNNetwork

        # Check AutoResearch results for best configs
        task_map = {
            "easy_typhoon_response": ["autoresearch_best_200k.pt", "autoresearch_best.pt",
                                       "autoresearch_experiment_003.pt", "qrdqn_best_easy.pt"],
            "medium_multi_front": ["autoresearch_medium_specialist.pt",
                                    "qrdqn_best_medium.pt", "autoresearch_best.pt"],
            "hard_cascading_crisis": ["autoresearch_hard_specialist.pt",
                                       "autoresearch_best_200k.pt", "qrdqn_best_hard.pt"],
        }

        for task_id, checkpoint_priority in task_map.items():
            loaded = False
            for ckpt_name in checkpoint_priority:
                ckpt_path = CHECKPOINT_DIR / ckpt_name
                if ckpt_path.exists():
                    try:
                        from rl.checkpoint_security import safe_load
                        ckpt = safe_load(ckpt_path, map_location=self.device)
                        cfg = {k: v for k, v in ckpt["config"].items()
                               if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
                        model = QRDQNNetwork(**cfg)
                        model.load_state_dict(ckpt["state_dict"])
                        model.eval()
                        self.specialists[task_id] = {
                            "model": model,
                            "checkpoint": ckpt_name,
                            "grade": ckpt.get("grade_avg", 0),
                        }
                        logger.info("Loaded specialist for %s: %s (grade=%.3f)",
                                    task_id, ckpt_name, ckpt.get("grade_avg", 0))
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", ckpt_name, str(e)[:60])

            if not loaded:
                logger.warning("No specialist found for %s — will use scripted fallback", task_id)

    @torch.no_grad()
    def predict(
        self,
        task_id: str,
        obs: np.ndarray,
        action_mask: np.ndarray,
        cvar_alpha: float = 0.5,
    ) -> int:
        """Predict action using the best specialist for this task.

        Args:
            task_id:     Task identifier.
            obs:         (408,) state vector.
            action_mask: (280,) boolean mask.
            cvar_alpha:  Risk level for CVaR policy.

        Returns:
            flat_action: int (0-279).
        """
        if task_id in self.specialists:
            model = self.specialists[task_id]["model"]
            state_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)
            return model.cvar_policy(state_t, alpha=cvar_alpha, action_mask=mask_t).item()

        # Fallback: scripted agent
        return self._scripted_fallback(obs, action_mask)

    def _scripted_fallback(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Use scripted heuristic as fallback."""
        valid = np.where(action_mask)[0]
        # Prefer non-alert, non-do_nothing actions
        real_actions = [a for a in valid if a // 40 not in (0, 6)]
        if real_actions:
            return real_actions[0]
        return valid[0] if len(valid) > 0 else 0

    def get_roster(self) -> dict[str, dict[str, Any]]:
        """Get current specialist roster for dashboard display."""
        roster = {}
        for task_id, spec in self.specialists.items():
            roster[task_id] = {
                "checkpoint": spec["checkpoint"],
                "grade": spec.get("grade", 0),
                "type": "QR-DQN (CVaR)",
            }
        return roster

    def evaluate_all(self, n_episodes: int = 10, seeds: list[int] | None = None) -> dict[str, float]:
        """Evaluate specialist roster on all tasks with grader.

        Returns {task_id: avg_grade_score}.
        """
        from rl.gym_env import SupplyMindGymnasiumEnv
        from server.supply_environment import SupplyMindEnvironment

        if seeds is None:
            seeds = [42, 99, 7, 123, 256]

        results = {}
        tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]

        for task_id in tasks:
            grades = []
            for seed in seeds:
                for ep in range(n_episodes // len(seeds)):
                    env = SupplyMindGymnasiumEnv(task_id=task_id)
                    env_core = SupplyMindEnvironment()
                    obs, info = env.reset(seed=seed * 1000 + ep)
                    env_core.reset(task_id=task_id, seed=seed * 1000 + ep)

                    while True:
                        flat = self.predict(task_id, obs, info["action_masks"])
                        action = np.array([flat // 40, flat % 40], dtype=np.int64)
                        obs, r, term, trunc, info = env.step(action)
                        sm = env._decode_action(action)
                        obs_c = env_core.step(sm)
                        if term or trunc or obs_c.done:
                            break

                    grades.append(env_core.grade()["score"])
                    env.close()

            results[task_id] = float(np.mean(grades))
            logger.info("  %s: grade=%.3f (n=%d)", task_id, results[task_id], len(grades))

        return results
