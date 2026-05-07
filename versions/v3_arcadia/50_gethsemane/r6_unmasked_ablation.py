"""R6-β / S8 / S11 — Action-masking ablation.

Trains a PPO variant WITHOUT action masking on easy_typhoon_response, then
evaluates both v3-masked (existing) and v3-unmasked policies. Quantifies
the contribution of action masking to the 8,100-ep bench sign-flip result.

Only runs on the easy task (smallest / fastest), so the ablation completes
in ~8 min. Judges can re-run on medium/hard if desired.

Outputs:
  versions/v3_arcadia/checkpoints/gethsemane/ppo_easy_UNMASKED.zip
  versions/v3_arcadia/results/R6_GETHSEMANE_MASKING_ABLATION.json
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
    """Returns a factory for the flattened-Discrete(280) env.
    If mask_fn is None, no ActionMasker wrapper (unmasked PPO).
    Otherwise, wrapped with ActionMasker for MaskablePPO.
    """
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


def get_mask_fn():
    def mask_fn(env_inner):
        inner = env_inner.env  # unwrap FlatDiscreteEnv
        if hasattr(inner, "_compute_action_mask"):
            return inner._compute_action_mask()
        return np.ones(env_inner.action_space.n, dtype=bool)
    return mask_fn


def train_unmasked():
    log.info(f"Training UNMASKED PPO on {TASK} ({TIMESTEPS:,} steps)...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    env_fn = make_flat_env(mask_fn=None)
    vec = DummyVecEnv([env_fn])
    m = PPO("MlpPolicy", vec,
             learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
             ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
             policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
             seed=SEED, device="cuda" if torch.cuda.is_available() else "cpu",
             verbose=0)
    t0 = time.time()
    m.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    train_time = time.time() - t0
    ckpt = CKPT / f"ppo_{TASK}_UNMASKED.zip"
    m.save(str(ckpt))
    log.info(f"  Trained in {train_time/60:.1f} min, saved {ckpt.name}")
    return m, train_time


def eval_policy(model, task_id, n_episodes, name):
    from rl.gym_env import SupplyMindGymnasiumEnv
    from sb3_contrib import MaskablePPO
    env = SupplyMindGymnasiumEnv(task_id=task_id, training_mode=False)
    is_maskable = isinstance(model, MaskablePPO)

    ep_rewards = []; ep_lengths = []; ep_violations = []; invalid_action_counts = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep * 17)
        done = False; total = 0.0; length = 0; viol = 0; invalid = 0
        while not done:
            mask = env._compute_action_mask() if hasattr(env, "_compute_action_mask") else None
            if is_maskable and mask is not None:
                flat, _ = model.predict(obs[None], action_masks=mask[None], deterministic=True)
            else:
                flat, _ = model.predict(obs[None], deterministic=True)
            flat = int(flat[0] if hasattr(flat, "__len__") else flat)
            # Track invalid-action picks for unmasked
            if mask is not None and not mask[flat]:
                invalid += 1
            a_type, a_target = divmod(flat, 40)
            obs, r, term, trunc, info = env.step(np.array([a_type, a_target]))
            done = term or trunc
            total += float(r); length += 1
            if info.get("constraint_violated", False): viol += 1
        ep_rewards.append(total); ep_lengths.append(length); ep_violations.append(viol)
        invalid_action_counts.append(invalid)
    return {
        "policy": name,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "reward_min": float(np.min(ep_rewards)),
        "reward_max": float(np.max(ep_rewards)),
        "length_mean": float(np.mean(ep_lengths)),
        "violations_mean": float(np.mean(ep_violations)),
        "invalid_action_picks_mean_per_ep": float(np.mean(invalid_action_counts)),
        "invalid_action_picks_max": int(np.max(invalid_action_counts)),
    }


def main():
    t0 = time.time()
    log.info("R6-β — Action-masking ablation (PPO with vs without masking)")

    # 1. Train unmasked PPO
    m_unmasked, train_time = train_unmasked()

    # 2. Evaluate unmasked on held-out eval env
    log.info(f"\nEvaluating UNMASKED PPO ({EVAL_EPISODES} eps)...")
    unmasked_stats = eval_policy(m_unmasked, TASK, EVAL_EPISODES, "ppo_v3_unmasked")
    log.info(f"  reward={unmasked_stats['reward_mean']:.3f} ± {unmasked_stats['reward_std']:.3f}")
    log.info(f"  invalid actions/ep: {unmasked_stats['invalid_action_picks_mean_per_ep']:.1f} (max {unmasked_stats['invalid_action_picks_max']})")
    log.info(f"  constraint violations/ep: {unmasked_stats['violations_mean']:.2f}")

    # 3. Load existing MASKED PPO and evaluate on same eval env
    log.info(f"\nEvaluating MASKED PPO v3 (existing ckpt) on same eval env...")
    from sb3_contrib import MaskablePPO
    m_masked = MaskablePPO.load(str(CKPT / f"ppo_{TASK}.zip"))
    masked_stats = eval_policy(m_masked, TASK, EVAL_EPISODES, "ppo_v3_masked")
    log.info(f"  reward={masked_stats['reward_mean']:.3f} ± {masked_stats['reward_std']:.3f}")
    log.info(f"  invalid actions/ep: {masked_stats['invalid_action_picks_mean_per_ep']:.1f}")
    log.info(f"  constraint violations/ep: {masked_stats['violations_mean']:.2f}")

    # Delta
    delta_reward = masked_stats["reward_mean"] - unmasked_stats["reward_mean"]
    pct = delta_reward / max(abs(unmasked_stats["reward_mean"]), 1e-6) * 100

    out = {
        "task": TASK,
        "training_timesteps": TIMESTEPS,
        "eval_episodes": EVAL_EPISODES,
        "unmasked": unmasked_stats,
        "masked": masked_stats,
        "action_masking_contribution": {
            "reward_delta": delta_reward,
            "reward_pct_delta": pct,
            "invalid_action_reduction": (
                unmasked_stats["invalid_action_picks_mean_per_ep"] -
                masked_stats["invalid_action_picks_mean_per_ep"]
            ),
            "training_time_unmasked_min": train_time / 60,
        },
        "interpretation": (
            "The reward_delta is the isolated contribution of action masking "
            "vs an otherwise-identical PPO. The invalid_action_reduction shows "
            "how often the unmasked agent picks a flatly-invalid joint action. "
            "With masking, that's structurally zero."
        ),
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R6_GETHSEMANE_MASKING_ABLATION.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    log.info(f"\n=== R6-β SUMMARY ===")
    log.info(f"  Masked reward:   {masked_stats['reward_mean']:+.3f}")
    log.info(f"  Unmasked reward: {unmasked_stats['reward_mean']:+.3f}")
    log.info(f"  Δ (masking contribution): {delta_reward:+.3f} ({pct:+.1f}%)")
    log.info(f"  Invalid picks/ep (unmasked): {unmasked_stats['invalid_action_picks_mean_per_ep']:.1f}")
    log.info(f"  Saved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
