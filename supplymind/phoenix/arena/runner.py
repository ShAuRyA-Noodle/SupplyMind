"""runner.py — evaluate a PyTorch policy on 3 SupplyMind tasks with CI95 reward.

The contract for "a PyTorch policy":

- A file named `policy.pt` (or any .pt / .zip / .pth), loadable via one of:
    (a) `stable_baselines3.PPO.load(path, env=None)`
    (b) `sb3_contrib.MaskablePPO.load(path, env=None)`
    (c) `torch.load(path)` returning an nn.Module with a
        `forward(obs_tensor) -> action_logits` method
- For (c), we feed a `torch.FloatTensor(obs)` of shape (1, 408) and take
    `argmax` as the Discrete(280) action.

For each of 3 tasks (easy_typhoon_response, medium_multi_front,
hard_cascading_crisis), we roll out `n_episodes` episodes with frozen seeds
(42, 99, 7 rotating), record episode reward and violation count, and bootstrap
a 95% confidence interval.

Output schema (`ArenaResult`):

    {
        "policy_name": "...",
        "submitted_at": "2026-04-22T...",
        "per_task": {
            "easy_typhoon_response": {"reward_mean": 1.20, "reward_std": 0.21,
                                      "ci95": [1.18, 1.22], "violations_mean": 0.0,
                                      "n_episodes": 50},
            ...
        },
        "overall_reward_mean": 2.15,
        "overall_ci95": [1.98, 2.33],
        "total_violations": 0,
        "rank_against_baseline": "beats random, ties greedy, 0.85x MaskablePPO",
    }
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TASKS = ("easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis")
DEFAULT_SEEDS = (42, 99, 7)
DEFAULT_EPISODES_PER_TASK = 50
MAX_STEPS_PER_EPISODE = 200


@dataclass
class TaskResult:
    task_id: str
    reward_mean: float
    reward_std: float
    reward_ci95_lower: float
    reward_ci95_upper: float
    violations_mean: float
    n_episodes: int
    episode_rewards: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "reward_mean": round(self.reward_mean, 4),
            "reward_std": round(self.reward_std, 4),
            "ci95": [round(self.reward_ci95_lower, 4), round(self.reward_ci95_upper, 4)],
            "violations_mean": round(self.violations_mean, 4),
            "n_episodes": self.n_episodes,
        }


@dataclass
class ArenaResult:
    policy_name: str
    submitted_at: str
    per_task: dict[str, TaskResult]
    overall_reward_mean: float
    overall_ci95_lower: float
    overall_ci95_upper: float
    total_violations: int
    rank_against_baseline: str

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "submitted_at": self.submitted_at,
            "per_task": {k: v.to_dict() for k, v in self.per_task.items()},
            "overall_reward_mean": round(self.overall_reward_mean, 4),
            "overall_ci95": [round(self.overall_ci95_lower, 4), round(self.overall_ci95_upper, 4)],
            "total_violations": self.total_violations,
            "rank_against_baseline": self.rank_against_baseline,
        }


def _bootstrap(rewards: np.ndarray, n: int = 1000, seed: int = 12345) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(rewards, size=len(rewards), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_policy(policy_path: Path) -> tuple[Any, str]:
    """Try sb3_contrib.MaskablePPO, then stable_baselines3.PPO, then torch.load.

    Returns (policy, loader_name).
    """
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    try:
        from sb3_contrib import MaskablePPO
        p = MaskablePPO.load(str(policy_path), env=None, device="auto")
        return p, "sb3_contrib.MaskablePPO"
    except Exception as e1:
        logger.debug("MaskablePPO load failed: %s", e1)

    try:
        from stable_baselines3 import PPO
        p = PPO.load(str(policy_path), env=None, device="auto")
        return p, "stable_baselines3.PPO"
    except Exception as e2:
        logger.debug("PPO load failed: %s", e2)

    try:
        import torch
        obj = torch.load(str(policy_path), map_location="cpu")
        if hasattr(obj, "forward"):
            return obj, "torch.nn.Module"
        raise ValueError("torch.load returned non-Module")
    except Exception as e3:
        raise ValueError(f"Could not load policy from {policy_path}: "
                         f"sb3={e1}; ppo={e2}; torch={e3}")


def _predict(policy: Any, obs: np.ndarray, mask: np.ndarray | None) -> int:
    """Robust prediction dispatch."""
    import numpy as np
    if hasattr(policy, "predict"):
        try:
            out = policy.predict(obs, deterministic=True, action_masks=mask)
        except TypeError:
            out = policy.predict(obs, deterministic=True)
        act = out[0] if isinstance(out, tuple) else out
        arr = np.asarray(act).flatten()
        return int(arr[0])
    # Raw torch.nn.Module path
    import torch
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        if isinstance(logits, tuple):
            logits = logits[0]
        if mask is not None:
            m = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits = logits.masked_fill(~m, float("-inf"))
        return int(torch.argmax(logits, dim=-1).item())


def _run_one_episode(policy: Any, task_id: str, seed: int) -> tuple[float, int]:
    """Run one episode; return (grade_score, violation_count)."""
    from rl.gym_env import SupplyMindGymnasiumEnv
    from server.supply_environment import SupplyMindEnvironment
    from gymnasium.spaces import Discrete
    import gymnasium as gym

    class Flat(gym.Wrapper):
        def __init__(self, base):
            super().__init__(base)
            _, n_t = base.action_space.nvec
            self._nt = int(n_t)
            self.action_space = Discrete(int(base.action_space.nvec[0]) * self._nt)

        def step(self, a):
            flat = int(np.asarray(a).flatten()[0])
            at, ag = divmod(flat, self._nt)
            return self.env.step(np.array([at, ag]))

    base = SupplyMindGymnasiumEnv(task_id=task_id)
    env = Flat(base)
    core = SupplyMindEnvironment()
    obs, info = env.reset(seed=seed)
    core.reset(task_id=task_id, seed=seed)
    violations = 0
    for _ in range(MAX_STEPS_PER_EPISODE):
        mask = info.get("action_masks")
        mask_np = np.asarray(mask) if mask is not None else None
        flat = _predict(policy, obs, mask_np)
        if mask_np is not None and not bool(mask_np[flat]):
            violations += 1
        obs, _, term, trunc, info = env.step(flat)
        at, ag = divmod(flat, 40)
        core.step(base._decode_action(np.array([at, ag], dtype=np.int64)))
        if term or trunc:
            break
    return float(core.grade()["score"]), violations


def evaluate_policy(
    policy_path: Path | str,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    n_episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    policy_name: str | None = None,
) -> ArenaResult:
    policy_path = Path(policy_path)
    policy_name = policy_name or policy_path.stem
    policy, loader = _load_policy(policy_path)
    logger.info("[arena] loaded %s via %s", policy_name, loader)

    per_task: dict[str, TaskResult] = {}
    all_rewards: list[float] = []
    total_violations = 0

    for task in tasks:
        rewards, vios = [], []
        for i in range(n_episodes_per_task):
            seed = DEFAULT_SEEDS[i % len(DEFAULT_SEEDS)] + (i // len(DEFAULT_SEEDS))
            try:
                r, v = _run_one_episode(policy, task, seed)
                rewards.append(r)
                vios.append(v)
            except Exception as e:  # noqa: BLE001
                logger.warning("[arena] %s ep %d failed: %s", task, i, e)
                rewards.append(0.0)
                vios.append(0)
        arr = np.asarray(rewards, dtype=np.float64)
        lo, hi = _bootstrap(arr)
        per_task[task] = TaskResult(
            task_id=task,
            reward_mean=float(arr.mean()),
            reward_std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            reward_ci95_lower=lo,
            reward_ci95_upper=hi,
            violations_mean=float(np.mean(vios)),
            n_episodes=len(rewards),
            episode_rewards=[round(float(x), 4) for x in rewards],
        )
        all_rewards.extend(rewards)
        total_violations += int(sum(vios))

    overall = np.asarray(all_rewards, dtype=np.float64)
    olo, ohi = _bootstrap(overall)
    result = ArenaResult(
        policy_name=policy_name,
        submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        per_task=per_task,
        overall_reward_mean=float(overall.mean()),
        overall_ci95_lower=olo,
        overall_ci95_upper=ohi,
        total_violations=total_violations,
        rank_against_baseline=_rank_note(overall.mean()),
    )
    return result


def _rank_note(overall_mean: float) -> str:
    """Compare to our known baselines (from v3_arcadia R6 results)."""
    # Baselines from R6_EUCLIDIAN.json (10800 eps): random avg ~0.0, greedy ~-0.75, MaskablePPO ~2.21
    if overall_mean < 0.0:
        return "below random baseline"
    if overall_mean < 0.8:
        return "between random and greedy"
    if overall_mean < 1.8:
        return "between greedy and MaskablePPO"
    if overall_mean < 2.3:
        return "near MaskablePPO baseline"
    return "exceeds MaskablePPO baseline"


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run a policy through the OpenEnv Arena.")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    res = evaluate_policy(args.policy, tuple(args.tasks), args.episodes, args.name)
    out_path = args.out or ROOT / "versions/v5_phoenix" / "experiments" / "arena" / f"{res.policy_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res.to_dict(), indent=2))
    print(json.dumps(res.to_dict(), indent=2))
    print(f"[arena] wrote {out_path}")
