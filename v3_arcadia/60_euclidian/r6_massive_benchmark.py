"""R6 Block 6 — Euclidian: 10,800-episode benchmark across policies + tasks.

Benchmarks 4 policies (random, greedy, PPO_v2, PPO_v3) across 3 tasks
with 900 episodes each = 10,800 episodes total. Confidence intervals via bootstrap.

Outputs:
  v3_arcadia/results/R6_EUCLIDIAN.json
  v3_arcadia/plots/euclidian/benchmark.png
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = ROOT / "v3_arcadia" / "checkpoints" / "gethsemane"
PLOTS = ROOT / "v3_arcadia" / "plots" / "euclidian"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"

TASKS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
EPISODES_PER_CELL = 900
SEED = 42


def _compute_mask(env):
    if hasattr(env, "_compute_action_mask"):
        return env._compute_action_mask()
    return np.ones(280, dtype=bool)


def _greedy_action(obs, mask):
    nf = obs[:400].reshape(40, 10)
    target = int(np.argmax(nf[:, 3]))
    a_type = 3  # increase_safety_stock
    flat = a_type * 40 + target
    if not mask[flat]:
        valid = np.where(mask)[0]
        flat = int(valid[0]) if len(valid) else 0
    return divmod(flat, 40)


def _random_action(mask):
    valid = np.where(mask)[0]
    if len(valid) == 0: return (0, 0)
    return divmod(int(np.random.choice(valid)), 40)


def evaluate(policy_name: str, model, task_id: str, n_episodes: int) -> list[dict]:
    from rl.gym_env import SupplyMindGymnasiumEnv
    env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=False)
    rng = np.random.default_rng(SEED)
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep * 17)
        done = False; total = 0.0; length = 0; viol = 0
        while not done:
            mask = _compute_mask(env)
            if policy_name == "random":
                a = _random_action(mask)
            elif policy_name == "greedy":
                a = _greedy_action(obs, mask)
            elif policy_name == "ppo_v3":
                flat, _ = model.predict(obs[None], action_masks=mask[None], deterministic=True)
                a = divmod(int(flat[0] if hasattr(flat, "__len__") else flat), 40)
            elif policy_name == "ppo_v2":
                # v2 checkpoints are SB3 PPO (zip), predict outputs MultiDiscrete directly
                flat_or_md, _ = model.predict(obs[None], deterministic=True)
                v = flat_or_md[0] if flat_or_md.ndim >= 1 else flat_or_md
                if hasattr(v, "__len__") and len(v) == 2:
                    a = (int(v[0]), int(v[1]))
                else:
                    a = divmod(int(v), 40)
            else:
                a = (0, 0)
            obs, r, term, trunc, info = env.step(np.array(a))
            done = term or trunc
            total += float(r); length += 1
            if info.get("constraint_violated", False): viol += 1
        episodes.append({"reward": total, "length": length, "violations": viol})
    return episodes


def aggregate_episodes(eps: list[dict]) -> dict:
    rewards = np.array([e["reward"] for e in eps])
    # Bootstrap 95% CI on mean reward
    rng = np.random.default_rng(SEED)
    B = 1000
    boot_means = [rewards[rng.integers(0, len(rewards), size=len(rewards))].mean() for _ in range(B)]
    return {
        "n_episodes": len(eps),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_ci95": [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))],
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "length_mean": float(np.mean([e["length"] for e in eps])),
        "violations_mean": float(np.mean([e["violations"] for e in eps])),
    }


def main():
    t0 = time.time()
    log.info(f"R6 Euclidian — 10,800-episode benchmark "
             f"(3 tasks x 4 policies x {EPISODES_PER_CELL} ep = "
             f"{3 * 4 * EPISODES_PER_CELL} episodes)")

    # Load PPO v3 per task
    ppo_v3_models = {}
    try:
        from sb3_contrib import MaskablePPO
        for task in TASKS:
            p = CKPT / f"ppo_{task}.zip"
            if p.exists():
                ppo_v3_models[task] = MaskablePPO.load(str(p))
                log.info(f"  loaded ppo_v3 {task}")
    except Exception as e:
        log.warning(f"ppo_v3 load skipped: {e}")

    # Load PPO v2 per task (different action space: MultiDiscrete[7,40] without masking)
    ppo_v2_models = {}
    try:
        from stable_baselines3 import PPO
        for task in TASKS:
            diff = task.split("_")[0]
            p = ROOT / "rl" / "checkpoints" / f"ppo_final_{diff}.zip"
            if p.exists():
                ppo_v2_models[task] = PPO.load(str(p))
                log.info(f"  loaded ppo_v2 {task}")
    except Exception as e:
        log.warning(f"ppo_v2 load skipped: {e}")

    results = {}
    for task in TASKS:
        log.info(f"\n=== Task: {task} ===")
        task_res = {}
        for policy in ["random", "greedy", "ppo_v3", "ppo_v2"]:
            if policy == "ppo_v3" and task not in ppo_v3_models:
                log.info(f"  [{policy}] SKIP (no model)"); continue
            if policy == "ppo_v2" and task not in ppo_v2_models:
                log.info(f"  [{policy}] SKIP (no model)"); continue
            model = ppo_v3_models.get(task) if policy == "ppo_v3" else (
                ppo_v2_models.get(task) if policy == "ppo_v2" else None)
            log.info(f"  [{policy}] running {EPISODES_PER_CELL} episodes...")
            tp = time.time()
            eps = evaluate(policy, model, task, EPISODES_PER_CELL)
            agg = aggregate_episodes(eps)
            agg["elapsed_s"] = time.time() - tp
            task_res[policy] = agg
            log.info(f"    reward={agg['reward_mean']:.2f} ± {agg['reward_std']:.2f}  "
                     f"CI95=[{agg['reward_ci95'][0]:.2f},{agg['reward_ci95'][1]:.2f}]  "
                     f"viol={agg['violations_mean']:.1f}/ep  ({agg['elapsed_s']/60:.1f}m)")
        results[task] = task_res

    total_eps = sum(r.get("n_episodes", 0) for task_r in results.values() for r in task_r.values())
    out = {
        "tasks": results,
        "config": {"episodes_per_cell": EPISODES_PER_CELL, "seed": SEED},
        "total_episodes": total_eps,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R6_EUCLIDIAN.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\n=== SUMMARY ===")
    for task, pols in results.items():
        log.info(f"  {task}:")
        for pol, a in pols.items():
            log.info(f"    {pol:<10}  reward={a['reward_mean']:7.2f}  CI95=[{a['reward_ci95'][0]:.2f},{a['reward_ci95'][1]:.2f}]")
    log.info(f"Total {total_eps} episodes in {out['elapsed_min']:.1f} min")
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
