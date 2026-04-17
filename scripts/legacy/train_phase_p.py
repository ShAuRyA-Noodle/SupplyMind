"""
Phase P "Granite" — online RL on real-calibrated env.
  U17 PPO 200K x 3 tasks, U18 QR-DQN 80K x 3 tasks, U19 Constrained PPO 3 tasks,
  U20 DQN+HER port with discrete goal conditioning + hindsight relabeling.
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
CKPT = ROOT / "rl" / "checkpoints"
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"


def log_failure(step, reason):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| P Granite | {step} | {reason[:300]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def retry(fn, name, n=2):
    for attempt in range(1, n + 1):
        try:
            t0 = time.time()
            log.info(f"=== P/{name} attempt {attempt}/{n} ===")
            r = fn()
            log.info(f"=== P/{name} OK ({time.time()-t0:.0f}s) ===")
            return r
        except Exception as e:
            log.error(f"{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == n:
                log_failure(name, str(e))
                return None


def ppo_task(task_id="SupplyMind-Easy-v1", n_steps=200_000, suffix="easy"):
    import gymnasium as gym
    import rl
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env):
        u = env.unwrapped
        if hasattr(u, "action_masks"):
            return u.action_masks()
        # Build a fully-open mask matching MultiDiscrete flat size
        import numpy as np
        n = int(np.prod(u.action_space.nvec)) if hasattr(u.action_space, "nvec") else u.action_space.n
        return np.ones(n, dtype=bool)

    def make_env():
        env = gym.make(task_id)
        env = ActionMasker(env, mask_fn)
        return env

    vec = DummyVecEnv([make_env for _ in range(4)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True)

    model = MaskablePPO("MlpPolicy", vec, verbose=0, learning_rate=3e-4,
                        n_steps=1024, batch_size=256, gamma=0.99, ent_coef=0.01,
                        policy_kwargs={"net_arch": [256, 128]}, device="cuda")
    log.info(f"PPO {suffix}: {n_steps:,} steps on {task_id}")
    model.learn(total_timesteps=n_steps, progress_bar=False)
    out = CKPT / f"ppo_v2_{suffix}.zip"
    model.save(str(out))
    vec.save(str(CKPT / f"ppo_v2_{suffix}_vecnorm.pkl"))
    log.info(f"  saved {out.name}")
    del model, vec; gc.collect()
    return out


def qrdqn_task(task_id="SupplyMind-Easy-v1", n_steps=80_000, suffix="easy"):
    from rl.distributional.train import train_qrdqn
    path = train_qrdqn(task=suffix, total_steps=n_steps)
    # Rename to v2
    import shutil
    src = CKPT / f"qrdqn_best_{suffix}.pt"
    dst = CKPT / f"qrdqn_v2_{suffix}.pt"
    if src.exists():
        shutil.copy(src, dst)
        log.info(f"  QR-DQN {suffix} saved to {dst.name}")
    return dst


def dqn_her():
    import gymnasium as gym
    import rl
    import torch, torch.nn as nn, torch.optim as optim

    env = gym.make("SupplyMind-Easy-v1")
    state_dim = env.observation_space.shape[0]
    n_action_types = int(env.action_space.nvec[0])
    n_nodes = int(env.action_space.nvec[1])

    class GCQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + n_nodes, 256), nn.GELU(),
                nn.Linear(256, 256), nn.GELU(),
            )
            self.q_type = nn.Linear(256, n_action_types)
            self.q_node = nn.Linear(256, n_nodes)

        def forward(self, s, g):
            z = self.net(torch.cat([s, g], dim=-1))
            return self.q_type(z), self.q_node(z)

    device = "cuda"
    q = GCQNet().to(device)
    q_target = GCQNet().to(device); q_target.load_state_dict(q.state_dict())
    opt = optim.AdamW(q.parameters(), lr=3e-4)

    buf = {"s": [], "g": [], "at": [], "an": [], "r": [], "sn": [], "d": [], "achieved": []}
    max_episodes = 300; eps = 1.0; eps_decay = 0.99; eps_min = 0.05
    gamma = 0.99; batch_size = 64; train_interval = 4; target_update = 100
    tot_steps = 0; episode_rewards = []

    for ep in range(max_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        goal_idx = np.random.randint(0, n_nodes)
        ep_buf = []
        for step in range(50):
            s_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            g_t = torch.zeros(1, n_nodes, device=device); g_t[0, goal_idx] = 1.0
            if np.random.rand() < eps:
                at = np.random.randint(0, n_action_types); an = np.random.randint(0, n_nodes)
            else:
                with torch.no_grad():
                    qt, qn = q(s_t, g_t)
                    at = int(qt.argmax().item()); an = int(qn.argmax().item())
            obs_next, r, done, trunc, info = env.step(np.array([at, an]))
            ep_buf.append((obs.copy(), goal_idx, at, an, r, obs_next.copy(), done, an))
            obs = obs_next; ep_r += r; tot_steps += 1

            if len(buf["s"]) > batch_size and tot_steps % train_interval == 0:
                idx = np.random.randint(0, len(buf["s"]), size=batch_size)
                bs = torch.from_numpy(np.stack([buf["s"][i] for i in idx]).astype(np.float32)).to(device)
                bg = torch.zeros(batch_size, n_nodes, device=device)
                for bi, i in enumerate(idx): bg[bi, buf["g"][i]] = 1.0
                bat = torch.tensor([buf["at"][i] for i in idx], device=device, dtype=torch.long)
                ban = torch.tensor([buf["an"][i] for i in idx], device=device, dtype=torch.long)
                br = torch.tensor([buf["r"][i] for i in idx], device=device, dtype=torch.float32)
                bsn = torch.from_numpy(np.stack([buf["sn"][i] for i in idx]).astype(np.float32)).to(device)
                bd = torch.tensor([buf["d"][i] for i in idx], device=device, dtype=torch.float32)
                with torch.no_grad():
                    qtn, qnn = q_target(bsn, bg)
                    target = br + gamma * (1 - bd) * (qtn.max(-1).values + qnn.max(-1).values) * 0.5
                qt_on, qn_on = q(bs, bg)
                q_sa = (qt_on.gather(1, bat.unsqueeze(1)).squeeze(1) + qn_on.gather(1, ban.unsqueeze(1)).squeeze(1)) * 0.5
                loss = (q_sa - target).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            if tot_steps % target_update == 0:
                q_target.load_state_dict(q.state_dict())
            if done or trunc: break

        if ep_buf:
            final_achieved = ep_buf[-1][-1]
            for (s, g, at, an, r, sn, d, ach) in ep_buf:
                buf["s"].append(s); buf["g"].append(g); buf["at"].append(at); buf["an"].append(an)
                buf["r"].append(r); buf["sn"].append(sn); buf["d"].append(float(d)); buf["achieved"].append(ach)
                buf["s"].append(s); buf["g"].append(final_achieved); buf["at"].append(at); buf["an"].append(an)
                buf["r"].append(1.0 if ach == final_achieved else -0.01)
                buf["sn"].append(sn); buf["d"].append(float(d)); buf["achieved"].append(ach)

        eps = max(eps_min, eps * eps_decay)
        episode_rewards.append(ep_r)
        if (ep + 1) % 25 == 0:
            log.info(f"  DQN+HER ep {ep+1}/{max_episodes}: mean_r={np.mean(episode_rewards[-25:]):.3f} eps={eps:.3f}")

    torch.save({"state_dict": q.state_dict(), "mean_final": float(np.mean(episode_rewards[-25:]))},
               CKPT / "dqn_her_v2.pt")
    return CKPT / "dqn_her_v2.pt"


def main():
    for task, suffix in [("SupplyMind-Easy-v1", "easy"), ("SupplyMind-Medium-v1", "medium"), ("SupplyMind-Hard-v1", "hard")]:
        retry(lambda t=task, s=suffix: ppo_task(t, n_steps=150_000, suffix=s), f"PPO_{suffix}")
    for task, suffix in [("SupplyMind-Easy-v1", "easy"), ("SupplyMind-Medium-v1", "medium"), ("SupplyMind-Hard-v1", "hard")]:
        retry(lambda t=task, s=suffix: qrdqn_task(t, n_steps=60_000, suffix=s), f"QRDQN_{suffix}")
    retry(dqn_her, "DQN_HER")
    log.info("Phase P 'Granite' complete.")


if __name__ == "__main__":
    main()
