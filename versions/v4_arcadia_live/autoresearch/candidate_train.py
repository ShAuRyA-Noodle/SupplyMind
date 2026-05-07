"""
candidate_train.py — The mutable RL training script.

This is the ONLY file the autoresearch agent modifies. Everything between
the `# --- SAFE TO MODIFY BELOW ---` and `# --- SAFE TO MODIFY ABOVE ---`
markers is fair game. Everything outside is frozen contract.

Adapted from Karpathy's train.py pattern: single file, clear modification zone,
stable signature, structured output dict.

Contract:
    def run_experiment(seed: int, total_steps: int) -> dict:
        returns {
            "grader_scores": list[float],          # length-9: 3 tasks * 3 seeds
            "wall_clock_s": float,
            "total_steps": int,
            "architecture_summary": str,
            "final_checkpoint": str,               # path
            "training_seed": int,
        }
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# FROZEN IMPORTS — agent cannot remove these, but may add more.
from rl.gym_env import SupplyMindGymnasiumEnv  # noqa: E402
from server.supply_environment import SupplyMindEnvironment  # noqa: E402

# Eval seeds are frozen. Training must not use any of these.
EVAL_SEEDS = (42, 99, 7)
EVAL_TASKS = ("easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis")


class FlatDiscreteEnv(gym.Wrapper):
    """Flatten MultiDiscrete([7,40]) to Discrete(280) so MaskablePPO's 280-dim
    action mask aligns. Matches the v3 Gethsemane pattern."""

    def __init__(self, base_env):
        super().__init__(base_env)
        n_type, n_target = base_env.action_space.nvec
        self._n_target = int(n_target)
        self.action_space = spaces.Discrete(int(n_type) * int(n_target))

    def step(self, action):
        flat = int(np.asarray(action).item())
        a_type, a_target = divmod(flat, self._n_target)
        return self.env.step(np.array([a_type, a_target]))


def _safe_predict(model: Any, obs: np.ndarray, action_masks) -> int:
    """Call model.predict; swallow the action_masks kwarg if unsupported."""
    try:
        out = model.predict(obs, deterministic=True, action_masks=action_masks)
    except TypeError:
        out = model.predict(obs, deterministic=True)
    action = out[0] if isinstance(out, tuple) else out
    return int(np.asarray(action).item())


def _evaluate_policy(model: Any, device: str = "cuda") -> list[float]:
    """Run 3 tasks x 3 seeds = 9 episodes, return grader scores.

    This function is FROZEN. Agent cannot modify the eval loop.
    """
    scores: list[float] = []
    for task_id in EVAL_TASKS:
        for seed in EVAL_SEEDS:
            base_env = SupplyMindGymnasiumEnv(task_id=task_id)
            eval_env = FlatDiscreteEnv(base_env)
            eval_core = SupplyMindEnvironment()
            obs, info = eval_env.reset(seed=seed)
            core_obs = eval_core.reset(task_id=task_id, seed=seed)
            done = False
            steps = 0
            while not done and steps < 200:
                mask = info.get("action_masks")
                mask_np = np.asarray(mask) if mask is not None else None
                flat = _safe_predict(model, obs, mask_np)
                obs, _, terminated, truncated, info = eval_env.step(flat)
                a_type, a_target = divmod(flat, 40)
                sm_action = base_env._decode_action(np.array([a_type, a_target], dtype=np.int64))
                core_obs = eval_core.step(sm_action)
                done = terminated or truncated or getattr(core_obs, "done", False)
                steps += 1
            score = eval_core.grade()["score"]
            scores.append(float(score))
            eval_env.close()
    return scores


# --- SAFE TO MODIFY BELOW ---

def build_policy_and_env(seed: int) -> tuple[Any, Any]:
    """Build the policy and training environment.

    Default: MaskablePPO with standard 64-64 MLP on easy_typhoon_response.
    Agent should mutate THIS function plus the training loop below.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _env_fn():
        env = SupplyMindGymnasiumEnv(
            task_id="easy_typhoon_response",
            training_mode=True,
            grade_reward=False,
        )
        env = FlatDiscreteEnv(env)
        return ActionMasker(env, lambda e: e.unwrapped._compute_action_mask())

    env = DummyVecEnv([_env_fn])
    env.seed(seed)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=0,
    )
    return model, env


def train_policy(model: Any, env: Any, total_steps: int) -> None:
    """Train for `total_steps` environment steps.

    Agent may swap in curriculum learning, learning-rate schedule, callbacks,
    reward shaping via wrappers, etc. — as long as the total_steps budget is
    respected.
    """
    model.learn(total_timesteps=total_steps, progress_bar=False)


def architecture_summary() -> str:
    """One-line human-readable summary for the lab notebook."""
    return "MaskablePPO MlpPolicy[256,256]+ReLU, lr=3e-4, n_steps=2048, gamma=0.99"

# --- SAFE TO MODIFY ABOVE ---


def run_experiment(seed: int, total_steps: int) -> dict:
    """Contract entrypoint. FROZEN signature.

    Args:
        seed: Training seed. MUST NOT be in EVAL_SEEDS (42, 99, 7).
        total_steps: Fixed step budget from program.md (default 50_000).

    Returns:
        dict with keys: grader_scores, wall_clock_s, total_steps,
        architecture_summary, final_checkpoint, training_seed
    """
    if seed in EVAL_SEEDS:
        raise ValueError(
            f"Training seed {seed} overlaps with EVAL_SEEDS {EVAL_SEEDS}. "
            "Holdout leakage forbidden (program.md rule 2)."
        )

    start = time.time()
    model, env = build_policy_and_env(seed)
    train_policy(model, env, total_steps)
    env.close()

    ckpt_dir = Path(__file__).resolve().parent / "experiments" / f"seed{seed}_candidate"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "policy.zip"
    try:
        if hasattr(model, "save"):
            model.save(str(ckpt_path))
        else:
            torch.save(model.state_dict(), str(ckpt_path).replace(".zip", ".pt"))
    except Exception as e:  # noqa: BLE001
        ckpt_path = Path("<save_failed>")
        print(f"[warn] checkpoint save failed: {e}", file=sys.stderr)

    scores = _evaluate_policy(model)
    wall_clock = time.time() - start

    return {
        "grader_scores": scores,
        "wall_clock_s": round(wall_clock, 2),
        "total_steps": total_steps,
        "architecture_summary": architecture_summary(),
        "final_checkpoint": str(ckpt_path),
        "training_seed": seed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one autoresearch candidate experiment.")
    parser.add_argument("--seed", type=int, default=123, help="Training seed (must not be in 42,99,7).")
    parser.add_argument("--steps", type=int, default=50_000, help="Fixed training step budget.")
    parser.add_argument("--out", type=str, default="candidate_result.json", help="Output JSON path.")
    args = parser.parse_args()

    result = run_experiment(seed=args.seed, total_steps=args.steps)
    Path(args.out).write_text(json.dumps(result, indent=2))
    scores = result["grader_scores"]
    print(f"grader_scores mean: {np.mean(scores):.3f}  min: {np.min(scores):.3f}  max: {np.max(scores):.3f}")
    print(f"wrote {args.out}")
