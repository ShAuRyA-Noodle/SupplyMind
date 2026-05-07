"""
Phase C — Retrain world models on real unified buffer.

Swaps offline_buffer.npz -> real_unified.npz for training, restores after.
Trains:
  - Neural surrogate (WorldModel): 20 epochs
  - DreamerV3 RSSM: 30 epochs (first real training — was code-only)

Saves: world_model_real.pt, rssm_real.pt
Counterfactual + GPU Monte Carlo auto-rebind via load_world_model() helper.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rl" / "data"
CKPT = ROOT / "rl" / "checkpoints"

SIM = DATA / "offline_buffer.npz"
SIM_BACKUP = DATA / "offline_buffer_simulated_backup.npz"
REAL = DATA / "real_unified.npz"
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"


def log_failure(phase, step, reason):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| {phase} | {step} | {reason[:200]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def swap_to_real():
    if not SIM_BACKUP.exists():
        shutil.copy(SIM, SIM_BACKUP)
    shutil.copy(REAL, SIM)
    log.info("Swapped offline_buffer.npz <- real_unified.npz")


def restore():
    if SIM_BACKUP.exists():
        shutil.copy(SIM_BACKUP, SIM)
        log.info("Restored simulated offline_buffer.npz")


def retry(fn, name, max_attempts=2):
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            log.info(f"=== {name} attempt {attempt}/{max_attempts} ===")
            r = fn()
            log.info(f"=== {name} OK ({time.time()-t0:.1f}s) ===")
            return r
        except Exception as e:
            log.error(f"{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == max_attempts:
                log_failure("C", name, str(e))
                return None


def train_world_model_real():
    from rl.surrogate.world_model import train_world_model
    path = train_world_model(epochs=20, batch_size=1024, device="cuda")
    v2 = CKPT / "world_model_real.pt"
    shutil.copy(path, v2)
    log.info(f"Saved {v2.name}")
    return v2


def train_rssm_real():
    """DreamerV3 RSSM — train on real buffer, 10 epochs. First real training."""
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from rl.surrogate.rssm import SupplyChainRSSM, compute_rssm_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(str(REAL))
    # Use single-step sequences (seq_len=1) since DataCo episodes are single-step
    n = min(40_000, len(data["states"]))
    states = torch.from_numpy(data["states"][:n].astype(np.float32)).unsqueeze(1)  # [N,1,408]
    actions = data["actions"][:n]
    next_states = torch.from_numpy(data["next_states"][:n].astype(np.float32))
    rewards = torch.from_numpy(data["rewards"][:n].astype(np.float32)).unsqueeze(1).unsqueeze(1)  # [N,1,1]
    dones = torch.from_numpy(data["dones"][:n].astype(np.float32)).unsqueeze(1).unsqueeze(1)

    action_onehot = np.zeros((n, 280), dtype=np.float32)
    for i in range(n):
        flat = int(actions[i, 0]) * 40 + int(actions[i, 1])
        action_onehot[i, min(flat, 279)] = 1.0
    action_t = torch.from_numpy(action_onehot).unsqueeze(1)  # [N,1,280]
    target_states = next_states.unsqueeze(1)  # [N,1,408]

    ds = TensorDataset(states, action_t, target_states, rewards, dones)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    model = SupplyChainRSSM(state_dim=408, action_dim=280, latent_dim=64, hidden_dim=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    log.info(f"RSSM params: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    best_path = CKPT / "rssm_real.pt"
    for epoch in range(1, 11):
        model.train()
        ep_loss = 0.0
        ep_parts = {"state": 0.0, "reward": 0.0, "kl": 0.0}
        n_batches = 0
        for s, a, s_next, r, d in dl:
            s, a, s_next, r, d = [x.to(device) for x in [s, a, s_next, r, d]]
            opt.zero_grad()
            out = model(s, a)
            losses = compute_rssm_loss(out, s_next, r, d, kl_weight=0.1)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += losses["total"].item()
            ep_parts["state"] += losses["state"].item()
            ep_parts["reward"] += losses["reward"].item()
            ep_parts["kl"] += losses["kl"].item()
            n_batches += 1
        ep_loss /= max(n_batches, 1)
        for k in ep_parts:
            ep_parts[k] /= max(n_batches, 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "loss": best_loss}, best_path)
        log.info(f"  RSSM epoch {epoch:2d}: total={ep_loss:.4f} state={ep_parts['state']:.4f} "
                 f"reward={ep_parts['reward']:.4f} kl={ep_parts['kl']:.4f} best={best_loss:.4f}")

    return best_path


def main():
    try:
        swap_to_real()

        results = {}
        results["world_model"] = retry(train_world_model_real, "world_model_real")
        results["rssm"] = retry(train_rssm_real, "rssm_real")

        log.info(f"Phase C results: {results}")
    finally:
        restore()
        log.info("Phase C done.")


if __name__ == "__main__":
    main()
