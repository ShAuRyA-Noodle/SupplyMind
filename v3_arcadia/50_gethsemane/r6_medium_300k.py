"""R6-α-plus — MaskablePPO on medium_multi_front at 300k timesteps.

Scale up medium-task training 3× to solidify the masking lift on the harder
mid-tier scenario. Evaluates both the 300k masked policy and an equally-
trained 300k unmasked baseline to publish the cleanest per-task lift table.

Output:
  v3_arcadia/results/R6_GETHSEMANE_MEDIUM_300K.json
  v3_arcadia/checkpoints/gethsemane/ppo_medium_multi_front_300k.zip
  v3_arcadia/checkpoints/gethsemane/ppo_medium_multi_front_300k_UNMASKED.zip
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

TASK = "medium_multi_front"
TIMESTEPS = 300_000
EVAL_EPISODES = 50
SEED = 42


def make_flat_env(task_id, mask_fn=None):
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
        env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=True)
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


def train(masked: bool, tag: str):
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec = DummyVecEnv([make_flat_env(TASK, mask_fn=_mask_fn if masked else None)])
    Cls = MaskablePPO if masked else PPO
    lr = 2.5e-4
    ent = 0.015 if masked else 0.01
    m = Cls("MlpPolicy", vec,
            learning_rate=lr, n_steps=2048, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=ent, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
            seed=SEED, device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0)
    log.info(f"  Training {tag} ({TIMESTEPS:,} steps, ent={ent})...")
    t0 = time.time()
    m.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    dt = time.time() - t0
    ckpt = CKPT / f"ppo_{TASK}_300k{'' if masked else '_UNMASKED'}.zip"
    m.save(str(ckpt))
    log.info(f"  {tag}: trained in {dt/60:.1f} min, saved {ckpt.name}")
    return m, dt


def eval_policy(model, task_id, n_eps, name):
    from rl.gym_env import SupplyMindGymnasiumEnv
    from sb3_contrib import MaskablePPO
    env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=False)
    is_m = isinstance(model, MaskablePPO)
    rs, ls, vs, iv = [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=SEED + ep * 17)
        done = False; tot = 0.0; L = 0; v = 0; inv = 0
        while not done:
            mask = env._compute_action_mask() if hasattr(env, "_compute_action_mask") else None
            if is_m and mask is not None:
                a, _ = model.predict(obs[None], action_masks=mask[None], deterministic=True)
            else:
                a, _ = model.predict(obs[None], deterministic=True)
            a = int(a[0] if hasattr(a, "__len__") else a)
            if mask is not None and not mask[a]:
                inv += 1
            at, ag = divmod(a, 40)
            obs, r, term, trunc, info = env.step(np.array([at, ag]))
            done = term or trunc
            tot += float(r); L += 1
            if info.get("constraint_violated", False): v += 1
        rs.append(tot); ls.append(L); vs.append(v); iv.append(inv)
    return {
        "policy": name, "n_episodes": n_eps,
        "reward_mean": float(np.mean(rs)), "reward_std": float(np.std(rs)),
        "reward_min": float(np.min(rs)), "reward_max": float(np.max(rs)),
        "length_mean": float(np.mean(ls)),
        "violations_mean": float(np.mean(vs)),
        "invalid_action_picks_mean_per_ep": float(np.mean(iv)),
    }


def main():
    t0 = time.time()
    log.info("R6-α-plus — MaskablePPO on medium_multi_front at 300k")

    m_m, dt_m = train(masked=True, tag="MaskablePPO 300k")
    m_u, dt_u = train(masked=False, tag="Plain PPO 300k")

    eval_m = eval_policy(m_m, TASK, EVAL_EPISODES, "ppo_v3_medium_300k_masked")
    eval_u = eval_policy(m_u, TASK, EVAL_EPISODES, "ppo_v3_medium_300k_unmasked")

    dR = eval_m["reward_mean"] - eval_u["reward_mean"]
    pct = dR / max(abs(eval_u["reward_mean"]), 1e-6) * 100
    out = {
        "task": TASK, "training_timesteps": TIMESTEPS, "eval_episodes": EVAL_EPISODES,
        "masked": eval_m, "unmasked": eval_u,
        "action_masking_contribution": {
            "reward_delta": dR, "reward_pct_delta": pct,
            "invalid_reduction": eval_u["invalid_action_picks_mean_per_ep"] - eval_m["invalid_action_picks_mean_per_ep"],
            "train_time_masked_min": dt_m / 60,
            "train_time_unmasked_min": dt_u / 60,
        },
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R6_GETHSEMANE_MEDIUM_300K.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nΔ = {dR:+.3f} ({pct:+.1f}%) — saved {out_path} ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
