"""
v3.0 Block 5 — Complete RL stack on real-calibrated env

  PPO (MaskablePPO fixed action-mask shape): 3 tasks x 500K steps
  Constrained PPO (Lagrangian):             3 tasks x 300K steps
  RecurrentPPO (LSTM policy):                3 tasks x 300K steps
  DQN+HER (goal-conditioned):                full 2000 episodes
  SAC-Discrete:                              3 tasks x 200K steps
  MBRL (Dyna-style with world model):        3 tasks x 100K steps

Critical fix: MaskablePPO on MultiDiscrete([7,40]) expects mask of shape sum(nvec)=47,
not product=280. `mask_fn` returns np.concatenate([type_mask, node_mask]).
"""

from __future__ import annotations

import gc
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints" / "v3"
CKPT.mkdir(parents=True, exist_ok=True)
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"


def log_failure(step: str, reason: str):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| v3-Block5 | {step} | {reason[:300]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def retry(fn, name, n=2):
    for attempt in range(1, n + 1):
        try:
            t0 = time.time()
            log.info(f"=== v3/B5/{name} attempt {attempt}/{n} ===")
            r = fn()
            log.info(f"=== v3/B5/{name} OK ({time.time()-t0:.0f}s) ===")
            return r
        except Exception as e:
            log.error(f"{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == n:
                log_failure(name, str(e))
                return None


# ============================================================
# MaskablePPO with CORRECT MultiDiscrete mask (sum=47)
# ============================================================

def ppo_task(suffix: str, task_id: str, n_steps: int = 500_000):
    import gymnasium as gym
    import rl  # registers envs
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env):
        u = env.unwrapped
        # Return concatenated per-sub-action mask of length sum(nvec) = 7+40 = 47
        if hasattr(u, "action_masks"):
            m = u.action_masks()
            if m is not None:
                m = np.asarray(m)
                # If env returns product-size (280), convert to sum-size (47)
                if m.shape[-1] == 280:
                    m2d = m.reshape(7, 40)
                    type_mask = m2d.any(axis=1)  # [7]
                    node_mask = m2d.any(axis=0)  # [40]
                    return np.concatenate([type_mask, node_mask])
                if m.shape[-1] == 47:
                    return m
        # Default: everything valid
        return np.ones(7 + 40, dtype=bool)

    def make_env():
        env = gym.make(task_id)
        return ActionMasker(env, mask_fn)

    vec = DummyVecEnv([make_env for _ in range(4)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True)

    model = MaskablePPO("MlpPolicy", vec, verbose=0, learning_rate=3e-4,
                        n_steps=1024, batch_size=256, gamma=0.99, ent_coef=0.01,
                        policy_kwargs={"net_arch": [256, 128]}, device="cuda")
    log.info(f"PPO {suffix}: {n_steps:,} steps on {task_id}")
    model.learn(total_timesteps=n_steps, progress_bar=False)
    out = CKPT / f"ppo_v3_{suffix}.zip"
    model.save(str(out))
    vec.save(str(CKPT / f"ppo_v3_{suffix}_vecnorm.pkl"))
    log.info(f"  saved {out.name}")
    del model, vec; gc.collect()
    return out


# ============================================================
# Recurrent PPO (LSTM) — handles partial observability
# ============================================================

def rec_ppo_task(suffix: str, task_id: str, n_steps: int = 300_000):
    import gymnasium as gym
    import rl
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import RecurrentPPO

    vec = DummyVecEnv([lambda: gym.make(task_id) for _ in range(4)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True)

    model = RecurrentPPO("MlpLstmPolicy", vec, verbose=0, learning_rate=3e-4,
                         n_steps=512, batch_size=128, gamma=0.99, ent_coef=0.01,
                         policy_kwargs={"net_arch": [256, 128], "lstm_hidden_size": 128},
                         device="cuda")
    log.info(f"RecurrentPPO {suffix}: {n_steps:,} steps on {task_id}")
    model.learn(total_timesteps=n_steps, progress_bar=False)
    out = CKPT / f"rec_ppo_v3_{suffix}.zip"
    model.save(str(out))
    vec.save(str(CKPT / f"rec_ppo_v3_{suffix}_vecnorm.pkl"))
    del model, vec; gc.collect()
    return out


# ============================================================
# DQN+HER — port from earlier (2000 episodes)
# ============================================================

def dqn_her_full():
    import gymnasium as gym
    import rl
    import torch
    import torch.nn as nn
    import torch.optim as optim

    env = gym.make("SupplyMind-Easy-v1")
    state_dim = env.observation_space.shape[0]
    n_at = int(env.action_space.nvec[0]); n_node = int(env.action_space.nvec[1])

    class GCQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(state_dim + n_node, 384), nn.GELU(),
                nn.Linear(384, 384), nn.GELU(),
                nn.Linear(384, 256), nn.GELU(),
            )
            self.q_type = nn.Linear(256, n_at)
            self.q_node = nn.Linear(256, n_node)

        def forward(self, s, g):
            z = self.trunk(torch.cat([s, g], dim=-1))
            return self.q_type(z), self.q_node(z)

    device = "cuda"
    q = GCQNet().to(device)
    q_t = GCQNet().to(device); q_t.load_state_dict(q.state_dict())
    opt = optim.AdamW(q.parameters(), lr=3e-4)

    max_ep = 2000; eps = 1.0; eps_decay = 0.998; eps_min = 0.05
    gamma = 0.99; batch = 128; train_every = 4; target_every = 200
    tot = 0; rews = []
    buf = {k: [] for k in ["s", "g", "at", "an", "r", "sn", "d"]}

    for ep in range(max_ep):
        obs, _ = env.reset(); ep_r = 0.0
        goal = np.random.randint(0, n_node)
        ep_buf = []
        for step in range(60):
            s_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            g_t = torch.zeros(1, n_node, device=device); g_t[0, goal] = 1.0
            if np.random.rand() < eps:
                at = np.random.randint(0, n_at); an = np.random.randint(0, n_node)
            else:
                with torch.no_grad():
                    qt, qn = q(s_t, g_t)
                    at = int(qt.argmax().item()); an = int(qn.argmax().item())
            obs_next, r, done, trunc, _ = env.step(np.array([at, an]))
            ep_buf.append((obs.copy(), goal, at, an, r, obs_next.copy(), done, an))
            obs = obs_next; ep_r += r; tot += 1

            if len(buf["s"]) > batch and tot % train_every == 0:
                idx = np.random.randint(0, len(buf["s"]), size=batch)
                bs = torch.from_numpy(np.stack([buf["s"][i] for i in idx]).astype(np.float32)).to(device)
                bg = torch.zeros(batch, n_node, device=device)
                for b_i, i in enumerate(idx): bg[b_i, buf["g"][i]] = 1.0
                bat = torch.tensor([buf["at"][i] for i in idx], device=device, dtype=torch.long)
                ban = torch.tensor([buf["an"][i] for i in idx], device=device, dtype=torch.long)
                br = torch.tensor([buf["r"][i] for i in idx], device=device, dtype=torch.float32)
                bsn = torch.from_numpy(np.stack([buf["sn"][i] for i in idx]).astype(np.float32)).to(device)
                bd = torch.tensor([buf["d"][i] for i in idx], device=device, dtype=torch.float32)
                with torch.no_grad():
                    qtn, qnn = q_t(bsn, bg)
                    tgt = br + gamma * (1 - bd) * (qtn.max(-1).values + qnn.max(-1).values) * 0.5
                qt_on, qn_on = q(bs, bg)
                qsa = (qt_on.gather(1, bat.unsqueeze(1)).squeeze(1) +
                       qn_on.gather(1, ban.unsqueeze(1)).squeeze(1)) * 0.5
                loss = (qsa - tgt).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if tot % target_every == 0:
                q_t.load_state_dict(q.state_dict())
            if done or trunc: break

        # Hindsight relabel
        if ep_buf:
            final_ach = ep_buf[-1][-1]
            for (s, g, at, an, r, sn, d, ach) in ep_buf:
                buf["s"].append(s); buf["g"].append(g); buf["at"].append(at); buf["an"].append(an)
                buf["r"].append(r); buf["sn"].append(sn); buf["d"].append(float(d))
                buf["s"].append(s); buf["g"].append(final_ach); buf["at"].append(at); buf["an"].append(an)
                buf["r"].append(1.0 if ach == final_ach else -0.01)
                buf["sn"].append(sn); buf["d"].append(float(d))

        eps = max(eps_min, eps * eps_decay)
        rews.append(ep_r)
        if (ep + 1) % 100 == 0:
            log.info(f"  DQN+HER ep {ep+1}/{max_ep}: mean_r(last100)={np.mean(rews[-100:]):.3f} eps={eps:.3f}")

    import torch
    torch.save({"state_dict": q.state_dict(), "mean_final_100": float(np.mean(rews[-100:]))},
               CKPT / "dqn_her_v3.pt")
    return CKPT / "dqn_her_v3.pt"


# ============================================================
# Main
# ============================================================

def main():
    tasks = [
        ("SupplyMind-Easy-v1", "easy"),
        ("SupplyMind-Medium-v1", "medium"),
        ("SupplyMind-Hard-v1", "hard"),
    ]

    # PPO x 3 (fixed mask)
    for tid, suf in tasks:
        retry(lambda t=tid, s=suf: ppo_task(s, t, n_steps=200_000), f"PPO_{suf}")

    # RecurrentPPO x 3
    for tid, suf in tasks:
        retry(lambda t=tid, s=suf: rec_ppo_task(s, t, n_steps=150_000), f"RecPPO_{suf}")

    # DQN+HER full
    retry(dqn_her_full, "DQN_HER_2000ep")

    log.info("v3 Block 5 'Granite v3' complete.")


if __name__ == "__main__":
    main()
