"""
Constrained PPO with Lagrangian relaxation for SupplyMind.

Supply chain managers have fixed risk budgets. The RL agent must never
exceed them. This module extends PPO with a learnable penalty multiplier
lambda that self-tunes until the budget constraint is satisfied on average.

Policy optimizes: reward - lambda * budget_violation
Lambda increases whenever budget constraint is violated.

Lagrangian relaxation drives the constraint toward satisfaction *on average*
over training; it is NOT a hard per-episode guarantee. Individual episodes can
still exceed the budget.

Usage:
    python -m rl.constrained_ppo --task easy --steps 1000000
    python -m rl.constrained_ppo --task hard --budget-limit 0.7
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

TASK_MAP = {
    "easy": "SupplyMind-Easy-v1",
    "medium": "SupplyMind-Medium-v1",
    "hard": "SupplyMind-Hard-v1",
}


class ConstrainedPPO:
    """PPO with Lagrangian relaxation for budget constraints.

    The key idea: instead of hard-constraining the budget, we add a
    learned penalty term to the loss:

        L = L_ppo + lambda_ * max(0, mean_budget_used - budget_limit)

    Lambda starts at 0 and increases whenever the agent violates the
    constraint. Over training, the agent learns to stay within budget.

    Args:
        policy:       The PPO policy network.
        lambda_lr:    Learning rate for lambda updates.
        budget_limit: Maximum allowed budget usage fraction (0-1).
        initial_lambda: Starting penalty multiplier.
    """

    def __init__(
        self,
        lambda_lr: float = 0.01,
        budget_limit: float = 0.7,
        initial_lambda: float = 0.0,
    ) -> None:
        self.lambda_: float = initial_lambda
        self.lambda_lr = lambda_lr
        self.budget_limit = budget_limit
        self._lambda_history: list[float] = []
        self._violation_history: list[float] = []

    def update_lambda(self, mean_budget_used: float, budget_limit: float | None = None) -> float:
        """Update Lagrangian multiplier based on constraint violation.

        Args:
            mean_budget_used:  Average budget usage ratio across rollout (0-1).
            budget_limit:      Override budget limit (optional).

        Returns:
            Updated lambda value.
        """
        limit = budget_limit if budget_limit is not None else self.budget_limit
        violation = mean_budget_used - limit

        # Gradient ascent on lambda (dual variable)
        self.lambda_ = max(0.0, self.lambda_ + self.lambda_lr * violation)

        self._lambda_history.append(self.lambda_)
        self._violation_history.append(violation)

        return self.lambda_

    def compute_loss(
        self,
        base_ppo_loss: torch.Tensor,
        budget_usage_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Augment PPO loss with Lagrangian budget penalty.

        Args:
            base_ppo_loss:     Standard PPO clipped surrogate loss.
            budget_usage_batch: (batch,) budget usage ratios for each sample.

        Returns:
            augmented_loss: base_loss + lambda * mean(max(0, usage - limit))
        """
        violations = torch.clamp(budget_usage_batch - self.budget_limit, min=0.0)
        penalty = self.lambda_ * violations.mean()
        return base_ppo_loss + penalty

    @property
    def constraint_satisfied(self) -> bool:
        """Check if recent violations are within tolerance."""
        if len(self._violation_history) < 10:
            return False
        recent = self._violation_history[-10:]
        return np.mean(recent) <= 0.0

    def get_stats(self) -> dict[str, float]:
        """Get current constraint stats for logging."""
        return {
            "lambda": self.lambda_,
            "mean_violation": np.mean(self._violation_history[-100:]) if self._violation_history else 0.0,
            "constraint_satisfied": self.constraint_satisfied,
        }


def train_constrained_ppo(
    task: str = "easy",
    total_timesteps: int = 1_000_000,
    n_envs: int = 16,
    budget_limit: float = 0.7,
    lambda_lr: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    """Train Constrained PPO with budget constraint."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_id = TASK_MAP[task]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Constrained PPO Training (Lagrangian Relaxation)")
    logger.info("  Task: %s | Steps: %s | Budget limit: %.1f%%",
                task, f"{total_timesteps:,}", budget_limit * 100)
    logger.info("  Lambda LR: %.3f | Device: %s", lambda_lr, device)
    logger.info("=" * 60)

    constraint = ConstrainedPPO(
        lambda_lr=lambda_lr,
        budget_limit=budget_limit,
    )

    def make_env(rank: int) -> Callable:
        def _init():
            import gymnasium as gym
            import numpy as _np
            import rl  # noqa: F401
            from sb3_contrib.common.wrappers import ActionMasker

            def _get_masks(env):
                unwrapped = env.unwrapped
                full_mask = unwrapped._compute_action_mask()
                type_mask = _np.zeros(7, dtype=_np.bool_)
                for i in range(7):
                    type_mask[i] = full_mask[i * 40:(i + 1) * 40].any()
                node_mask = _np.zeros(40, dtype=_np.bool_)
                for j in range(40):
                    node_mask[j] = full_mask[j::40].any()
                return _np.concatenate([type_mask, node_mask])

            from rl.gym_env import SupplyMindGymnasiumEnv
            task_map = {"SupplyMind-Easy-v1": "easy_typhoon_response", "SupplyMind-Medium-v1": "medium_multi_front", "SupplyMind-Hard-v1": "hard_cascading_crisis"}
            env = SupplyMindGymnasiumEnv(task_id=task_map.get(env_id, "easy_typhoon_response"), training_mode=True)
            env.reset(seed=seed + rank)
            env = ActionMasker(env, _get_masks)
            return env
        return _init

    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    class ConstraintCallback(BaseCallback):
        """Track budget usage and update lambda after each rollout."""

        def __init__(self):
            super().__init__()
            self._budget_usages: list[float] = []

        def _on_step(self) -> bool:
            # Extract budget usage from info
            for info in self.locals.get("infos", []):
                raw = info.get("raw_obs", None)
                if raw is not None:
                    fin = raw.financials
                    usage = 1.0 - (fin.budget_remaining / max(fin.budget_total, 1.0))
                    self._budget_usages.append(usage)
            return True

        def _on_rollout_end(self) -> None:
            if self._budget_usages:
                mean_usage = np.mean(self._budget_usages[-1000:])
                constraint.update_lambda(mean_usage)
                stats = constraint.get_stats()
                if self.num_timesteps % 50_000 < self.model.n_steps * self.model.n_envs:
                    logger.info(
                        "[%dk] lambda=%.4f | mean_violation=%.4f | satisfied=%s",
                        self.num_timesteps // 1000,
                        stats["lambda"], stats["mean_violation"],
                        stats["constraint_satisfied"],
                    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[ConstraintCallback()],
        progress_bar=True,
    )
    elapsed = time.time() - start

    save_path = CHECKPOINT_DIR / f"constrained_ppo_{task}.zip"
    model.save(str(save_path))

    # Save constraint stats
    stats_path = CHECKPOINT_DIR / f"constrained_ppo_stats_{task}.npz"
    np.savez(
        str(stats_path),
        lambda_history=np.array(constraint._lambda_history),
        violation_history=np.array(constraint._violation_history),
    )

    logger.info("=" * 60)
    logger.info("Constrained PPO done in %.1f min", elapsed / 60)
    logger.info("  Final lambda: %.4f | Constraint satisfied: %s",
                constraint.lambda_, constraint.constraint_satisfied)
    logger.info("  Model: %s", save_path)
    logger.info("=" * 60)

    train_env.close()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return save_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train Constrained PPO")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--budget-limit", type=float, default=0.7)
    parser.add_argument("--lambda-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_constrained_ppo(
        task=args.task, total_timesteps=args.steps, budget_limit=args.budget_limit,
        lambda_lr=args.lambda_lr, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
