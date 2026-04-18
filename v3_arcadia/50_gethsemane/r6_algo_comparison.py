"""R6-ζ — MaskablePPO vs PPO vs A2C vs RecurrentPPO on easy_typhoon_response.

Four RL algorithms trained identically (same env, same steps, same seed, same
net arch) and evaluated on the same 50-episode held-out suite. Publishes a
clean win-margin table.

Output:
  v3_arcadia/results/R6_ALGO_COMPARISON.json
  v3_arcadia/checkpoints/gethsemane/{maskable_ppo,ppo,a2c,recppo}_easy.zip
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
RESULTS = ROOT / "v3_arcadia" / "results"

TASK = "easy_typhoon_response"
TIMESTEPS = 100_000
EVAL_EPISODES = 50
SEED = 42


def make_flat_env(mask_fn=None):
    import gymnasium as gym
    from gymnasium import spaces

    class FlatDiscreteEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            nvec = env.action_space.nvec
            self._n_target = int(nvec[1])
            self.action_space = spaces.Discrete(int(nvec[0]) * self._n_target)

        def step(self, action):
            a_type, a_target = divmod(int(action), self._n_target)
            return self.env.step(np.array([a_type, a_target]))

    def _init():
        from rl.gym_env import SupplyMindGymnasiumEnv
        env = SupplyMindGymnasiumEnv(task_id=TASK, training_mode=True)
        env = FlatDiscreteEnv(env)
        if mask_fn is not None:
            from sb3_contrib.common.wrappers import ActionMasker
            env = ActionMasker(env, mask_fn)
        env.reset(seed=SEED)
        return env

    return _init


def _mask_fn(env_inner):
    inner = env_inner.env
    if hasattr(inner, "_compute_action_mask"):
        return inner._compute_action_mask()
    return np.ones(env_inner.action_space.n, dtype=bool)


def train_and_save(algo_name: str):
    from stable_baselines3.common.vec_env import DummyVecEnv
    log.info(f"Training {algo_name} ({TIMESTEPS:,} steps)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_kwargs = {"net_arch": [256, 256], "activation_fn": torch.nn.Tanh}
    common = dict(learning_rate=3e-4, gamma=0.99, seed=SEED, device=device,
                   policy_kwargs=policy_kwargs, verbose=0)

    if algo_name == "MaskablePPO":
        from sb3_contrib import MaskablePPO
        vec = DummyVecEnv([make_flat_env(mask_fn=_mask_fn)])
        m = MaskablePPO("MlpPolicy", vec,
                         n_steps=2048, batch_size=128, n_epochs=10,
                         gae_lambda=0.95, clip_range=0.2,
                         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                         **common)
    elif algo_name == "PPO":
        from stable_baselines3 import PPO
        vec = DummyVecEnv([make_flat_env(mask_fn=None)])
        m = PPO("MlpPolicy", vec,
                 n_steps=2048, batch_size=128, n_epochs=10,
                 gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 **common)
    elif algo_name == "A2C":
        from stable_baselines3 import A2C
        vec = DummyVecEnv([make_flat_env(mask_fn=None)])
        m = A2C("MlpPolicy", vec,
                 n_steps=5, gae_lambda=0.95,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 **common)
    elif algo_name == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        vec = DummyVecEnv([make_flat_env(mask_fn=None)])
        m = RecurrentPPO("MlpLstmPolicy", vec,
                          n_steps=1024, batch_size=128, n_epochs=10,
                          gae_lambda=0.95, clip_range=0.2,
                          ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                          **common)
    else:
        raise ValueError(algo_name)

    t0 = time.time()
    m.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    dt = time.time() - t0
    tag = algo_name.lower().replace("ppo", "ppo")
    ckpt = CKPT / f"{tag}_easy.zip"
    m.save(str(ckpt))
    log.info(f"  {algo_name}: trained in {dt/60:.1f} min")
    return m, dt


def eval_policy(model, algo_name: str, n_eps: int):
    from rl.gym_env import SupplyMindGymnasiumEnv
    from sb3_contrib import MaskablePPO, RecurrentPPO
    env = SupplyMindGymnasiumEnv(task_id=TASK, training_mode=False)
    is_maskable = isinstance(model, MaskablePPO)
    is_recurrent = isinstance(model, RecurrentPPO)
    rs, ls, vs, iv = [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=SEED + ep * 17)
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False; total = 0.0; L = 0; v = 0; inv = 0
        while not done:
            mask = env._compute_action_mask() if hasattr(env, "_compute_action_mask") else None
            if is_recurrent:
                flat, lstm_states = model.predict(obs[None], state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = np.zeros((1,), dtype=bool)
            elif is_maskable and mask is not None:
                flat, _ = model.predict(obs[None], action_masks=mask[None], deterministic=True)
            else:
                flat, _ = model.predict(obs[None], deterministic=True)
            flat = int(flat[0] if hasattr(flat, "__len__") else flat)
            if mask is not None and not mask[flat]:
                inv += 1
            at, ag = divmod(flat, 40)
            obs, r, term, trunc, info = env.step(np.array([at, ag]))
            done = term or trunc
            total += float(r); L += 1
            if info.get("constraint_violated", False): v += 1
        rs.append(total); ls.append(L); vs.append(v); iv.append(inv)
    return {
        "algorithm": algo_name, "n_episodes": n_eps,
        "reward_mean": float(np.mean(rs)), "reward_std": float(np.std(rs)),
        "reward_min": float(np.min(rs)), "reward_max": float(np.max(rs)),
        "length_mean": float(np.mean(ls)),
        "violations_mean": float(np.mean(vs)),
        "invalid_action_picks_mean_per_ep": float(np.mean(iv)),
    }


def main():
    t0 = time.time()
    log.info("R6-ζ — RL algorithm comparison on easy_typhoon_response")

    algos = ["MaskablePPO", "PPO", "A2C", "RecurrentPPO"]
    results = {}
    train_times = {}
    for a in algos:
        try:
            m, dt = train_and_save(a)
            train_times[a] = dt
            results[a] = eval_policy(m, a, EVAL_EPISODES)
            r = results[a]
            log.info(f"  {a}: reward={r['reward_mean']:.3f}±{r['reward_std']:.3f}, invalid={r['invalid_action_picks_mean_per_ep']:.1f}")
        except Exception as e:
            log.warning(f"  {a} failed: {str(e)[:200]}")
            results[a] = {"status": "FAILED", "error": str(e)[:300]}

    # Compute margins relative to MaskablePPO
    baseline = results.get("MaskablePPO", {}).get("reward_mean")
    comparison = {}
    if baseline is not None:
        for a, r in results.items():
            if a == "MaskablePPO" or "reward_mean" not in r:
                continue
            comparison[a] = {
                "reward_delta": r["reward_mean"] - baseline,
                "maskable_lift_pct": (baseline - r["reward_mean"]) / max(abs(r["reward_mean"]), 1e-6) * 100,
            }

    out = {
        "task": TASK, "training_timesteps": TIMESTEPS, "eval_episodes": EVAL_EPISODES,
        "per_algorithm": results,
        "train_times_min": {a: t / 60 for a, t in train_times.items()},
        "maskable_vs_others": comparison,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R6_ALGO_COMPARISON.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
