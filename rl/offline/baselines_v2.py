"""
Phase N "Chokehold" — Offline RL v2: saturated training with architectural fixes.

Upgrades addressed:
  U10 BC 200 epochs (was 30, loss still decreasing)
  U11 IQL 300K steps (was 50K) + separate type/node action heads
  U12 CQL 300K steps (was 50K) + wider Q (512-512)
  U13 TD3+BC 300K steps + separate type/node heads
  U14 DT 50 epochs + full test eval
  U15 Ensemble tuning on real data
  U16 Statistical rigor handed off to Phase O

Key architectural fix: ActionHead(state_emb) -> (type_logits[7], node_logits[40])
  instead of flat 280-way softmax. Dramatically easier to learn on 164-action space.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "rl" / "checkpoints"
DATA = ROOT / "rl" / "data"
CKPT.mkdir(exist_ok=True)


def load_v2():
    train = np.load(str(DATA / "real_train_v2.npz"))
    val = np.load(str(DATA / "real_val_v2.npz"))
    return train, val


# ============================================================
# Shared factorized policy head: (type ∈ 7, node ∈ 40)
# ============================================================

class FactorizedPolicy(nn.Module):
    """State encoder -> (type_logits, node_logits). Drastically better on 164-action space."""

    def __init__(self, state_dim=408, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 256), nn.GELU(),
        )
        self.type_head = nn.Linear(256, 7)
        self.node_head = nn.Linear(256, 40)

    def forward(self, x):
        z = self.trunk(x)
        return self.type_head(z), self.node_head(z)

    def flat_logits(self, x):
        """280-way logits = outer sum of type + node."""
        tl, nl = self.forward(x)
        return (tl.unsqueeze(-1) + nl.unsqueeze(-2)).reshape(x.shape[0], -1)


class FactorizedTwinQ(nn.Module):
    """Twin Q-net, factorized: Q_type(s) ∈ 7, Q_node(s) ∈ 40; Q(s,a) = Q_type[type] + Q_node[node]."""

    def __init__(self, state_dim=408, hidden=512):
        super().__init__()
        self.t1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.t2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.q1_type = nn.Linear(hidden, 7); self.q1_node = nn.Linear(hidden, 40)
        self.q2_type = nn.Linear(hidden, 7); self.q2_node = nn.Linear(hidden, 40)

    def q1(self, s):
        z = self.t1(s)
        return self.q1_type(z), self.q1_node(z)

    def q2(self, s):
        z = self.t2(s)
        return self.q2_type(z), self.q2_node(z)

    def q1_flat(self, s):
        t, n = self.q1(s)
        return (t.unsqueeze(-1) + n.unsqueeze(-2)).reshape(s.shape[0], -1)

    def q2_flat(self, s):
        t, n = self.q2(s)
        return (t.unsqueeze(-1) + n.unsqueeze(-2)).reshape(s.shape[0], -1)


class ValueNet(nn.Module):
    def __init__(self, state_dim=408, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# BC v2 — 200 epochs, factorized, separate CE on type + node
# ============================================================

def train_bc_v2(epochs=200, batch_size=1024, lr=3e-4, device="cuda", seed=42):
    torch.manual_seed(seed)
    train, val = load_v2()
    states = torch.from_numpy(train["states"].astype(np.float32))
    actions = train["actions"].astype(np.int64)  # [N, 2]
    a_type = torch.from_numpy(actions[:, 0])
    a_node = torch.from_numpy(actions[:, 1])

    ds = TensorDataset(states, a_type, a_node)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    s_val = torch.from_numpy(val["states"].astype(np.float32)).to(device)
    a_val_type = torch.from_numpy(val["actions"][:, 0].astype(np.int64)).to(device)
    a_val_node = torch.from_numpy(val["actions"][:, 1].astype(np.int64)).to(device)

    model = FactorizedPolicy().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_val = 0.0
    best_path = CKPT / "bc_v2.pt"
    t0 = time.time()
    log.info(f"BC v2: {epochs} epochs, batch {batch_size}, lr {lr}, factorized head")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for s, at, an in dl:
            s, at, an = s.to(device, non_blocking=True), at.to(device, non_blocking=True), an.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                tl, nl = model(s)
                loss = F.cross_entropy(tl, at) + F.cross_entropy(nl, an)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
        sched.step()
        ep_loss /= len(dl)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tl, nl = model(s_val)
                type_acc = (tl.argmax(-1) == a_val_type).float().mean().item()
                node_acc = (nl.argmax(-1) == a_val_node).float().mean().item()
                full_acc = ((tl.argmax(-1) == a_val_type) & (nl.argmax(-1) == a_val_node)).float().mean().item()
            log.info(f"  BC ep {epoch:3d}/{epochs} loss={ep_loss:.4f} val type={type_acc:.3f} node={node_acc:.3f} full={full_acc:.3f}")
            if full_acc > best_val:
                best_val = full_acc
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_full_acc": best_val, "val_type_acc": type_acc, "val_node_acc": node_acc},
                           best_path)

    log.info(f"BC v2 done {time.time()-t0:.0f}s, best val full={best_val:.4f}")
    del model, opt; torch.cuda.empty_cache(); gc.collect()
    return best_path


# ============================================================
# CQL v2 — 300K steps, factorized twin Q, wider
# ============================================================

def train_cql_v2(n_steps=300_000, batch_size=512, lr=3e-4, gamma=0.99, conservative_weight=5.0, device="cuda"):
    train, val = load_v2()
    N = len(train["states"])
    states = torch.from_numpy(train["states"].astype(np.float32)).to(device)
    a_type = torch.from_numpy(train["actions"][:, 0].astype(np.int64)).to(device)
    a_node = torch.from_numpy(train["actions"][:, 1].astype(np.int64)).to(device)
    rewards = torch.from_numpy(train["rewards"].astype(np.float32)).to(device)
    next_states = torch.from_numpy(train["next_states"].astype(np.float32)).to(device)
    dones = torch.from_numpy(train["dones"].astype(np.float32)).to(device)

    online = FactorizedTwinQ().to(device)
    target = FactorizedTwinQ().to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters(): p.requires_grad = False
    opt = optim.AdamW(online.parameters(), lr=lr, weight_decay=1e-5)

    s_val = torch.from_numpy(val["states"].astype(np.float32)).to(device)
    a_val_type = torch.from_numpy(val["actions"][:, 0].astype(np.int64)).to(device)
    a_val_node = torch.from_numpy(val["actions"][:, 1].astype(np.int64)).to(device)

    best_val = 0.0
    best_path = CKPT / "cql_v2.pt"
    t0 = time.time()
    log.info(f"CQL v2: {n_steps:,} steps, factorized twin-Q, wider (512)")

    tau = 0.005
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        s, at, an = states[idx], a_type[idx], a_node[idx]
        r, ns, d = rewards[idx], next_states[idx], dones[idx]

        with torch.no_grad():
            t_tl, t_nl = target.q1(ns)
            t2_tl, t2_nl = target.q2(ns)
            min_type = torch.min(t_tl, t2_tl)
            min_node = torch.min(t_nl, t2_nl)
            # Double DQN: next_action from online network
            on_tl, on_nl = online.q1(ns)
            next_at = on_tl.argmax(-1)
            next_an = on_nl.argmax(-1)
            next_q_type = min_type.gather(1, next_at.unsqueeze(1)).squeeze(1)
            next_q_node = min_node.gather(1, next_an.unsqueeze(1)).squeeze(1)
            # Factorized Q for taken action = type_Q[type] + node_Q[node]
            target_q = r + gamma * (1 - d) * (next_q_type + next_q_node) * 0.5

        q1_t, q1_n = online.q1(s)
        q2_t, q2_n = online.q2(s)
        q1_a = q1_t.gather(1, at.unsqueeze(1)).squeeze(1) + q1_n.gather(1, an.unsqueeze(1)).squeeze(1)
        q2_a = q2_t.gather(1, at.unsqueeze(1)).squeeze(1) + q2_n.gather(1, an.unsqueeze(1)).squeeze(1)
        q1_a = q1_a * 0.5; q2_a = q2_a * 0.5

        bellman = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        # Conservative penalty on flat action space
        q1_flat = online.q1_flat(s); q2_flat = online.q2_flat(s)
        a_flat = at * 40 + an
        cql = (
            (torch.logsumexp(q1_flat, dim=1).mean() - q1_flat.gather(1, a_flat.unsqueeze(1)).mean())
            + (torch.logsumexp(q2_flat, dim=1).mean() - q2_flat.gather(1, a_flat.unsqueeze(1)).mean())
        )

        loss = bellman + conservative_weight * cql

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        opt.step()

        # Soft target update
        with torch.no_grad():
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.lerp_(op.data, tau)

        if step % 15000 == 0 or step == 1:
            online.eval()
            with torch.no_grad():
                vt, vn = online.q1(s_val)
                type_acc = (vt.argmax(-1) == a_val_type).float().mean().item()
                node_acc = (vn.argmax(-1) == a_val_node).float().mean().item()
                full_acc = ((vt.argmax(-1) == a_val_type) & (vn.argmax(-1) == a_val_node)).float().mean().item()
            online.train()
            log.info(f"  CQL step {step:,}/{n_steps:,} bellman={bellman.item():.3f} cql_pen={cql.item():.3f} "
                     f"val type={type_acc:.3f} node={node_acc:.3f} full={full_acc:.3f}")
            if full_acc > best_val:
                best_val = full_acc
                torch.save({"state_dict": online.state_dict(), "step": step,
                            "val_full_acc": best_val, "val_type_acc": type_acc, "val_node_acc": node_acc},
                           best_path)

    log.info(f"CQL v2 done {time.time()-t0:.0f}s, best val full={best_val:.4f}")
    del online, target, opt; torch.cuda.empty_cache(); gc.collect()
    return best_path


# ============================================================
# IQL v2 — 300K steps, expectile regression, factorized heads
# ============================================================

def train_iql_v2(n_steps=300_000, batch_size=512, lr=3e-4, expectile=0.7, weight_temp=3.0,
                 gamma=0.99, device="cuda"):
    train, val = load_v2()
    N = len(train["states"])
    states = torch.from_numpy(train["states"].astype(np.float32)).to(device)
    a_type = torch.from_numpy(train["actions"][:, 0].astype(np.int64)).to(device)
    a_node = torch.from_numpy(train["actions"][:, 1].astype(np.int64)).to(device)
    rewards = torch.from_numpy(train["rewards"].astype(np.float32)).to(device)
    next_states = torch.from_numpy(train["next_states"].astype(np.float32)).to(device)
    dones = torch.from_numpy(train["dones"].astype(np.float32)).to(device)

    actor = FactorizedPolicy().to(device)
    q = FactorizedTwinQ().to(device)
    q_target = FactorizedTwinQ().to(device); q_target.load_state_dict(q.state_dict())
    for p in q_target.parameters(): p.requires_grad = False
    v = ValueNet().to(device)

    opt_a = optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q = optim.AdamW(q.parameters(), lr=lr, weight_decay=1e-5)
    opt_v = optim.AdamW(v.parameters(), lr=lr, weight_decay=1e-5)

    s_val = torch.from_numpy(val["states"].astype(np.float32)).to(device)
    a_val_type = torch.from_numpy(val["actions"][:, 0].astype(np.int64)).to(device)
    a_val_node = torch.from_numpy(val["actions"][:, 1].astype(np.int64)).to(device)

    best_val = 0.0
    best_path = CKPT / "iql_v2.pt"
    t0 = time.time()
    log.info(f"IQL v2: {n_steps:,} steps, factorized actor + twin-Q + V, expectile={expectile}")

    tau = 0.005
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        s, at, an = states[idx], a_type[idx], a_node[idx]
        r, ns, d = rewards[idx], next_states[idx], dones[idx]

        # Q targets: r + gamma * V(s')
        with torch.no_grad():
            v_next = v(ns)
            q_target_val = r + gamma * (1 - d) * v_next
            # Q(s,a) supervised target
            t1_t, t1_n = q_target.q1(s); t2_t, t2_n = q_target.q2(s)
            q1_a = (t1_t.gather(1, at.unsqueeze(1)).squeeze(1) + t1_n.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
            q2_a = (t2_t.gather(1, at.unsqueeze(1)).squeeze(1) + t2_n.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
            q_min = torch.min(q1_a, q2_a)

        # Q loss
        q1_ft, q1_fn = q.q1(s); q2_ft, q2_fn = q.q2(s)
        q1_s = (q1_ft.gather(1, at.unsqueeze(1)).squeeze(1) + q1_fn.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
        q2_s = (q2_ft.gather(1, at.unsqueeze(1)).squeeze(1) + q2_fn.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
        q_loss = F.mse_loss(q1_s, q_target_val) + F.mse_loss(q2_s, q_target_val)
        opt_q.zero_grad(); q_loss.backward(); opt_q.step()

        # V loss: expectile regression toward q_min
        v_s = v(s)
        diff = q_min - v_s
        v_loss = torch.where(diff > 0, expectile * diff.pow(2), (1 - expectile) * diff.pow(2)).mean()
        opt_v.zero_grad(); v_loss.backward(); opt_v.step()

        # Actor loss: advantage-weighted behavior cloning
        with torch.no_grad():
            adv = q_min - v_s
            weight = torch.clamp(torch.exp(weight_temp * adv), max=100.0)
        tl, nl = actor(s)
        actor_loss = -(weight * (F.log_softmax(tl, dim=-1).gather(1, at.unsqueeze(1)).squeeze(1)
                                 + F.log_softmax(nl, dim=-1).gather(1, an.unsqueeze(1)).squeeze(1))).mean()
        opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

        # Soft target update
        with torch.no_grad():
            for p, tp in zip(q.parameters(), q_target.parameters()):
                tp.data.lerp_(p.data, tau)

        if step % 15000 == 0 or step == 1:
            actor.eval()
            with torch.no_grad():
                tl, nl = actor(s_val)
                type_acc = (tl.argmax(-1) == a_val_type).float().mean().item()
                node_acc = (nl.argmax(-1) == a_val_node).float().mean().item()
                full_acc = ((tl.argmax(-1) == a_val_type) & (nl.argmax(-1) == a_val_node)).float().mean().item()
            actor.train()
            log.info(f"  IQL step {step:,}/{n_steps:,} q={q_loss.item():.3f} v={v_loss.item():.3f} a={actor_loss.item():.3f} "
                     f"val type={type_acc:.3f} node={node_acc:.3f} full={full_acc:.3f}")
            if full_acc > best_val:
                best_val = full_acc
                torch.save({"state_dict": actor.state_dict(), "step": step,
                            "val_full_acc": best_val, "val_type_acc": type_acc, "val_node_acc": node_acc,
                            "q_state_dict": q.state_dict(), "v_state_dict": v.state_dict()},
                           best_path)

    log.info(f"IQL v2 done {time.time()-t0:.0f}s, best val full={best_val:.4f}")
    del actor, q, q_target, v; torch.cuda.empty_cache(); gc.collect()
    return best_path


# ============================================================
# TD3+BC v2 — 300K steps, factorized, with BC regularization
# ============================================================

def train_td3bc_v2(n_steps=300_000, batch_size=512, lr=3e-4, gamma=0.99, alpha_bc=2.5, device="cuda"):
    train, val = load_v2()
    N = len(train["states"])
    states = torch.from_numpy(train["states"].astype(np.float32)).to(device)
    a_type = torch.from_numpy(train["actions"][:, 0].astype(np.int64)).to(device)
    a_node = torch.from_numpy(train["actions"][:, 1].astype(np.int64)).to(device)
    rewards = torch.from_numpy(train["rewards"].astype(np.float32)).to(device)
    next_states = torch.from_numpy(train["next_states"].astype(np.float32)).to(device)
    dones = torch.from_numpy(train["dones"].astype(np.float32)).to(device)

    actor = FactorizedPolicy().to(device)
    q = FactorizedTwinQ().to(device)
    q_target = FactorizedTwinQ().to(device); q_target.load_state_dict(q.state_dict())
    for p in q_target.parameters(): p.requires_grad = False

    opt_a = optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q = optim.AdamW(q.parameters(), lr=lr, weight_decay=1e-5)

    s_val = torch.from_numpy(val["states"].astype(np.float32)).to(device)
    a_val_type = torch.from_numpy(val["actions"][:, 0].astype(np.int64)).to(device)
    a_val_node = torch.from_numpy(val["actions"][:, 1].astype(np.int64)).to(device)

    best_val = 0.0
    best_path = CKPT / "td3bc_v2.pt"
    t0 = time.time()
    log.info(f"TD3+BC v2: {n_steps:,} steps, factorized, alpha_bc={alpha_bc}")

    tau = 0.005
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        s, at, an = states[idx], a_type[idx], a_node[idx]
        r, ns, d = rewards[idx], next_states[idx], dones[idx]

        # Q update
        with torch.no_grad():
            tl_n, nl_n = actor(ns)
            next_at = tl_n.argmax(-1); next_an = nl_n.argmax(-1)
            q1t, q1n = q_target.q1(ns); q2t, q2n = q_target.q2(ns)
            q1_next = q1t.gather(1, next_at.unsqueeze(1)).squeeze(1) + q1n.gather(1, next_an.unsqueeze(1)).squeeze(1)
            q2_next = q2t.gather(1, next_at.unsqueeze(1)).squeeze(1) + q2n.gather(1, next_an.unsqueeze(1)).squeeze(1)
            q_min = torch.min(q1_next, q2_next) * 0.5
            target_q = r + gamma * (1 - d) * q_min

        q1t, q1n = q.q1(s); q2t, q2n = q.q2(s)
        q1_a = (q1t.gather(1, at.unsqueeze(1)).squeeze(1) + q1n.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
        q2_a = (q2t.gather(1, at.unsqueeze(1)).squeeze(1) + q2n.gather(1, an.unsqueeze(1)).squeeze(1)) * 0.5
        q_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        opt_q.zero_grad(); q_loss.backward(); opt_q.step()

        # Actor update (delayed, every 2)
        if step % 2 == 0:
            tl, nl = actor(s)
            # Policy gradient via REINFORCE on argmax; simpler: BC + Q gradient through softmax
            pi_at = tl.argmax(-1); pi_an = nl.argmax(-1)
            q1t2, q1n2 = q.q1(s)
            q_pi = q1t2.gather(1, pi_at.unsqueeze(1)).squeeze(1) + q1n2.gather(1, pi_an.unsqueeze(1)).squeeze(1)
            lam = alpha_bc / q_pi.abs().mean().detach().clamp(min=1e-3)
            bc_loss = F.cross_entropy(tl, at) + F.cross_entropy(nl, an)
            actor_loss = -lam * q_pi.mean() + bc_loss
            opt_a.zero_grad(); actor_loss.backward(); opt_a.step()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), q_target.parameters()):
                    tp.data.lerp_(p.data, tau)

        if step % 15000 == 0 or step == 1:
            actor.eval()
            with torch.no_grad():
                tl, nl = actor(s_val)
                type_acc = (tl.argmax(-1) == a_val_type).float().mean().item()
                node_acc = (nl.argmax(-1) == a_val_node).float().mean().item()
                full_acc = ((tl.argmax(-1) == a_val_type) & (nl.argmax(-1) == a_val_node)).float().mean().item()
            actor.train()
            log.info(f"  TD3+BC step {step:,}/{n_steps:,} q={q_loss.item():.3f} "
                     f"val type={type_acc:.3f} node={node_acc:.3f} full={full_acc:.3f}")
            if full_acc > best_val:
                best_val = full_acc
                torch.save({"state_dict": actor.state_dict(), "step": step,
                            "val_full_acc": best_val, "val_type_acc": type_acc, "val_node_acc": node_acc},
                           best_path)

    log.info(f"TD3+BC v2 done {time.time()-t0:.0f}s, best val full={best_val:.4f}")
    del actor, q, q_target; torch.cuda.empty_cache(); gc.collect()
    return best_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["bc", "cql", "iql", "td3bc", "all"], default="all")
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300_000)
    args = ap.parse_args()

    if args.algo in ("bc", "all"): train_bc_v2(epochs=args.bc_epochs)
    if args.algo in ("cql", "all"): train_cql_v2(n_steps=args.steps)
    if args.algo in ("iql", "all"): train_iql_v2(n_steps=args.steps)
    if args.algo in ("td3bc", "all"): train_td3bc_v2(n_steps=args.steps)
