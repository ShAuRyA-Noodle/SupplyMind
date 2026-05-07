"""calibrate_conformal_from_harvest.py — fit conformal action filter on
real harvested PPO/scripted trajectories.

Pipeline:
  1. Load versions/v5_phoenix/experiments/rap_xc_v1/transitions.npz
  2. Split into 80/20 train/calibration sets
  3. Train a small reference policy on train (BC on action targets)
  4. Compute action-NLL quantile on the calibration set with target
     coverage 1-alpha (default alpha=0.1 → 90%% coverage)
  5. Save ConformalActionFilter to action_v2/conformal_calibrated.pt
     and a JSON receipt to tests/receipts/conformal_calibration.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from versions.v5_phoenix.action_v2.conformal import calibrate_conformal

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANS_NPZ = REPO_ROOT / "versions/v5_phoenix" / "experiments" / "rap_xc_v1" / "transitions.npz"
OUT_PT = REPO_ROOT / "versions/v5_phoenix" / "action_v2" / "conformal_calibrated.pt"
RECEIPT = REPO_ROOT / "tests" / "receipts" / "conformal_calibration.json"


class _RefPolicy(nn.Module):
    """Tiny MLP reference policy used purely for conformal calibration."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, n_actions: int = 280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main(alpha: float = 0.1, epochs: int = 8, batch_size: int = 256):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not TRANS_NPZ.exists():
        logger.error("[conformal] no harvest npz at %s", TRANS_NPZ)
        return

    npz = np.load(TRANS_NPZ)
    state = torch.from_numpy(npz["state_feats"]).float()
    actions = torch.from_numpy(npz["actions"]).long()
    n = state.size(0)
    logger.info("[conformal] loaded %d transitions, state_dim=%d", n, state.size(-1))

    # 80/20 split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    tr_idx = perm[:split]
    cal_idx = perm[split:]

    Xtr, Ytr = state[tr_idx], actions[tr_idx]
    Xcal, Ycal = state[cal_idx], actions[cal_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _RefPolicy(in_dim=state.size(-1), hidden=128, n_actions=280).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    ds = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    train_losses: list[float] = []
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); n_batches += 1
        train_losses.append(ep_loss / max(1, n_batches))
        logger.info("[conformal] ep %d  train_loss=%.4f", ep, train_losses[-1])

    # Compute logits on calibration set
    model.eval()
    Xcal_d = Xcal.to(device)
    with torch.no_grad():
        cal_logits = model(Xcal_d).cpu()

    cf = calibrate_conformal(cal_logits, Ycal, alpha=alpha)
    logger.info("[conformal] calibrated: nll_quantile=%.4f, alpha=%.2f, n=%d",
                cf.nll_quantile, cf.alpha, cf.n_calibration)

    # Empirical coverage check on calibration set itself
    filtered_logits = cf.filter_logits(cal_logits)
    accept_mask = filtered_logits != float("-inf")
    expert_in_accepted = accept_mask.gather(1, Ycal.unsqueeze(-1)).squeeze(-1).float().mean()
    n_accepted_per_row = accept_mask.sum(dim=-1).float()
    logger.info("[conformal] empirical coverage on calibration set: %.4f",
                float(expert_in_accepted))

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ref_policy_state_dict": model.state_dict(),
        "conformal_filter": cf.to_dict(),
        "in_dim": int(state.size(-1)),
        "n_actions": 280,
    }, OUT_PT)
    logger.info("[conformal] saved %s", OUT_PT)

    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_transitions_total": int(n),
        "n_train": int(len(tr_idx)),
        "n_calibration": int(len(cal_idx)),
        "ref_policy_train_losses": train_losses,
        "conformal_filter": cf.to_dict(),
        "empirical_coverage_on_cal": float(expert_in_accepted),
        "n_accepted_actions_per_row_mean": float(n_accepted_per_row.mean()),
        "n_accepted_actions_per_row_median": float(n_accepted_per_row.median()),
        "n_accepted_actions_per_row_min": float(n_accepted_per_row.min()),
        "n_accepted_actions_per_row_max": float(n_accepted_per_row.max()),
        "alpha": alpha,
        "expected_coverage_1_minus_alpha": 1.0 - alpha,
        "elapsed_s": round(time.time() - t0, 2),
        "weights_path": str(OUT_PT.relative_to(REPO_ROOT)),
        "transitions_source": str(TRANS_NPZ.relative_to(REPO_ROOT)),
        "method": "split_conformal_NLL_on_real_harvested_trajectories",
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("[conformal] receipt: %s", RECEIPT)
    print(json.dumps({k: v for k, v in receipt.items()
                       if k != "ref_policy_train_losses"}, indent=2))


if __name__ == "__main__":
    main()
