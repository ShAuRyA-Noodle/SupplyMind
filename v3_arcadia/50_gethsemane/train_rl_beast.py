"""R6 Block 5 — Gethsemane: Beast-mode RL stack.

MaskablePPO (sb3-contrib) with action masking, trained on all 3 SupplyMind tasks.
Benchmark vs random + greedy baselines. Resume-safe checkpointing.

Outputs:
  v3_arcadia/checkpoints/gethsemane/ppo_<task>.zip
  v3_arcadia/results/R6_GETHSEMANE.json
  v3_arcadia/plots/gethsemane/training_curves.png
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
CKPT.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "v3_arcadia" / "plots" / "gethsemane"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"

TASKS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
TIMESTEPS_PER_TASK = 100_000
EVAL_EPISODES = 50
SEED = 42


class FlattenMultiDiscrete(np.ndarray):
    """Marker class for unused shape-check."""
    pass


def _build_flatten_wrapper():
    """Wrapper that converts MultiDiscrete([7, 40]) -> Discrete(280) so MaskablePPO's flat mask works."""
    import gymnasium as gym
    from gymnasium import spaces

    class FlatDiscreteEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            nvec = env.action_space.nvec
            self._n_action_type = int(nvec[0])
            self._n_target = int(nvec[1])
            self.action_space = spaces.Discrete(self._n_action_type * self._n_target)

        def step(self, action):
            a_type, a_target = divmod(int(action), self._n_target)
            return self.env.step(np.array([a_type, a_target]))

    return FlatDiscreteEnv


def make_masked_env(task_id: str, seed: int = 0):
    """Factory: SupplyMind env -> flatten MultiDiscrete to Discrete(280) -> ActionMasker."""
    from rl.gym_env import SupplyMindGymnasiumEnv
    from sb3_contrib.common.wrappers import ActionMasker

    FlatDiscreteEnv = _build_flatten_wrapper()

    def _init():
        env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=True)
        env = FlatDiscreteEnv(env)

        def mask_fn(env_inner):
            inner = env_inner.env  # unwrap FlatDiscreteEnv
            if hasattr(inner, "_compute_action_mask"):
                return inner._compute_action_mask()
            return np.ones(env_inner.action_space.n, dtype=bool)

        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init


def train_task(task_id: str, total_timesteps: int) -> dict:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    log.info(f"\n=== Training {task_id}  ({total_timesteps:,} steps) ===")
    ckpt_path = CKPT / f"ppo_{task_id}.zip"

    # Build vec env (1 env — env is stateful complex simulation, multi-proc not worth it)
    env_fn = make_masked_env(task_id, seed=SEED)
    vec_env = DummyVecEnv([env_fn])
    eval_env = DummyVecEnv([make_masked_env(task_id, seed=SEED + 100)])

    if ckpt_path.exists():
        log.info(f"  Loading existing checkpoint {ckpt_path.name}")
        model = MaskablePPO.load(str(ckpt_path), env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
            tensorboard_log=None,
            seed=SEED,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(CKPT / f"best_{task_id}"),
        eval_freq=max(total_timesteps // 10, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    train_time = time.time() - t0

    model.save(str(ckpt_path))
    log.info(f"  Saved {ckpt_path.name}  ({train_time/60:.1f} min)")

    # Evaluate trained policy
    stats = evaluate_policy(model, task_id, n_episodes=EVAL_EPISODES, policy_name="ppo_v3")
    stats["train_time_s"] = train_time
    stats["total_timesteps"] = total_timesteps
    return stats


def evaluate_policy(model, task_id: str, n_episodes: int, policy_name: str) -> dict:
    """Evaluate a policy over N episodes. Returns aggregate metrics."""
    from rl.gym_env import SupplyMindGymnasiumEnv
    env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=False)

    ep_rewards = []
    ep_lengths = []
    ep_constraint_violations = []  # count of steps below safety threshold

    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + ep * 17)
        done = False
        total_r = 0.0
        length = 0
        violations = 0
        while not done:
            mask = env._compute_action_mask() if hasattr(env, "_compute_action_mask") else None
            if model == "random":
                if mask is not None and mask.any():
                    valid = np.where(mask)[0]
                    idx = int(np.random.choice(valid))
                    a_type, a_target = divmod(idx, 40)
                    action = np.array([a_type, a_target])
                else:
                    action = env.action_space.sample()
            elif model == "greedy":
                action = greedy_action(obs, env, mask)
            else:
                # Model was trained on flattened Discrete(280). Predict -> decode to MultiDiscrete([7,40]).
                obs_batch = obs[None] if obs.ndim == 1 else obs
                mask_batch = mask[None] if mask is not None and mask.ndim == 1 else mask
                if mask_batch is not None:
                    flat_action, _ = model.predict(obs_batch, action_masks=mask_batch, deterministic=True)
                else:
                    flat_action, _ = model.predict(obs_batch, deterministic=True)
                flat = int(flat_action[0] if hasattr(flat_action, "__len__") else flat_action)
                a_type, a_target = divmod(flat, 40)
                action = np.array([a_type, a_target])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += float(reward)
            length += 1
            # Constraint violation: any node-level risk > 0.9 in observation
            if info.get("constraint_violated", False):
                violations += 1
        ep_rewards.append(total_r)
        ep_lengths.append(length)
        ep_constraint_violations.append(violations)

    return {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "reward_min": float(np.min(ep_rewards)),
        "reward_max": float(np.max(ep_rewards)),
        "length_mean": float(np.mean(ep_lengths)),
        "violations_mean": float(np.mean(ep_constraint_violations)),
        "violations_max": int(np.max(ep_constraint_violations)),
    }


def greedy_action(obs, env, mask=None):
    """Greedy heuristic: pick action that targets the highest-risk node."""
    node_feats = obs[:400].reshape(40, 10)
    risk_col = 3
    target_node = int(np.argmax(node_feats[:, risk_col]))
    action_type = 3
    action = np.array([action_type, target_node])
    if mask is not None:
        flat_idx = action_type * 40 + target_node
        if not mask[flat_idx]:
            valid = np.where(mask)[0]
            if len(valid) > 0:
                idx = int(valid[0])
                action_type, target_node = divmod(idx, 40)
                action = np.array([action_type, target_node])
    return action


def main():
    t0 = time.time()
    log.info("R6 Gethsemane — RL BEAST")

    results = {"tasks": {}, "baselines": {}, "config": {
        "timesteps_per_task": TIMESTEPS_PER_TASK,
        "eval_episodes": EVAL_EPISODES,
        "seed": SEED,
    }}

    # Train + eval PPO on each task
    for task in TASKS:
        stats = train_task(task, TIMESTEPS_PER_TASK)
        results["tasks"][task] = {"ppo_v3": stats}

    # Baselines: random + greedy on each task
    for task in TASKS:
        log.info(f"\n=== Baselines on {task} ===")
        random_stats = evaluate_policy("random", task, EVAL_EPISODES, "random")
        greedy_stats = evaluate_policy("greedy", task, EVAL_EPISODES, "greedy")
        results["tasks"][task]["random"] = random_stats
        results["tasks"][task]["greedy"] = greedy_stats
        log.info(f"  random:  reward={random_stats['reward_mean']:.2f} ± {random_stats['reward_std']:.2f}")
        log.info(f"  greedy:  reward={greedy_stats['reward_mean']:.2f} ± {greedy_stats['reward_std']:.2f}")
        ppo = results["tasks"][task]["ppo_v3"]
        log.info(f"  ppo_v3:  reward={ppo['reward_mean']:.2f} ± {ppo['reward_std']:.2f}")

    results["elapsed_min"] = (time.time() - t0) / 60
    out = RESULTS / "R6_GETHSEMANE.json"
    out.write_text(json.dumps(results, indent=2, default=str))

    log.info("\n=== SUMMARY ===")
    for task in TASKS:
        log.info(f"  {task}:")
        for pol in ["random", "greedy", "ppo_v3"]:
            s = results["tasks"][task][pol]
            log.info(f"    {pol:<10}  reward={s['reward_mean']:7.2f}  violations={s['violations_mean']:.1f}/ep")
    log.info(f"\nSaved: {out}  ({results['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
