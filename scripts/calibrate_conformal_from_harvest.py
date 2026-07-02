"""calibrate_conformal_from_harvest.py — REAL split-conformal action filter
calibrated + VALIDATED on real harvested RAP-XC trajectories.

This is R11 of the rebuild backlog. The old fake used Gaussian noise labelled as
model NLL. An interim version used real transitions but reported empirical
coverage on the SAME calibration set used to pick the quantile — which is
guaranteed to be ~= 1-alpha by construction and therefore not a real test.

This version does honest Vovk split-conformal with a genuine held-out test set:

  1. Load versions/v5_phoenix/experiments/rap_xc_v1/transitions.npz (real,
     40k harvested (state, expert_action) transitions).
  2. Three-way split: 60% train / 20% calibration / 20% TEST (seeded, disjoint).
  3. Train a small reference policy on train (behavioural cloning).
  4. For each target alpha, compute the (1-alpha) NLL quantile on the calibration
     set (finite-sample corrected), then measure EMPIRICAL COVERAGE on the
     held-out TEST set the quantile never saw. That test coverage vs the nominal
     1-alpha is the honest calibration number.
  5. Save the alpha=0.1 ConformalActionFilter and a JSON receipt.

Coverage is measured as the raw conformal-set membership P[NLL(expert) <= q],
NOT the product 'always-accept-argmax' mask (which would inflate coverage).

Receipt: tests/receipts/conformal_REAL.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
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

from supplymind.phoenix.action_v2.conformal import calibrate_conformal  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _ROOT
TRANS_NPZ = REPO_ROOT / "versions" / "v5_phoenix" / "experiments" / "rap_xc_v1" / "transitions.npz"
OUT_PT = REPO_ROOT / "supplymind" / "phoenix" / "action_v2" / "conformal_calibrated.pt"
RECEIPT = REPO_ROOT / "tests" / "receipts" / "conformal_REAL.json"

N_ACTIONS = 280
ALPHAS = [0.05, 0.10, 0.20]     # nominal miscoverage levels to validate


class _RefPolicy(nn.Module):
    """Tiny MLP reference policy used purely for conformal calibration."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, n_actions: int = N_ACTIONS):
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _expert_nll(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """NLL of the expert action under the policy: -log_softmax(logits)[a]."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)


def main(epochs: int = 12, batch_size: int = 256):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not TRANS_NPZ.exists():
        # DATA-BLOCKED: fail loud with an honest, machine-readable receipt.
        msg = (f"transitions.npz not found at {TRANS_NPZ}. R11 is DATA-BLOCKED: "
               f"restore the 1.16GB real harvest (repo or Sleep-Token-ARCHIVE) "
               f"before running. No synthetic fallback.")
        logger.error("[conformal] %s", msg)
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps({
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "DATA_BLOCKED",
            "reason": msg,
            "transitions_source": str(TRANS_NPZ.relative_to(REPO_ROOT)).replace("\\", "/"),
        }, indent=2), encoding="utf-8")
        raise SystemExit(2)

    npz = np.load(TRANS_NPZ)
    state = torch.from_numpy(npz["state_feats"]).float()
    actions = torch.from_numpy(npz["actions"]).long()
    n = state.size(0)
    n_unique_actions = int(torch.unique(actions).numel())
    logger.info("[conformal] loaded %d real transitions, state_dim=%d, "
                "unique_actions=%d", n, state.size(-1), n_unique_actions)

    # 60/20/20 train / calibration / TEST split (disjoint, seeded)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(n * 0.6)
    n_cal = int(n * 0.2)
    tr_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

    Xtr, Ytr = state[tr_idx], actions[tr_idx]
    Xcal, Ycal = state[cal_idx], actions[cal_idx]
    Xtest, Ytest = state[test_idx], actions[test_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _RefPolicy(in_dim=state.size(-1), hidden=128, n_actions=N_ACTIONS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)

    t0 = time.time()
    train_losses: list[float] = []
    for ep in range(epochs):
        model.train()
        ep_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); n_batches += 1
        train_losses.append(ep_loss / max(1, n_batches))
        logger.info("[conformal] ep %2d  train_loss=%.4f", ep, train_losses[-1])

    # Logits on calibration + held-out test sets
    model.eval()
    with torch.no_grad():
        cal_logits = model(Xcal.to(device)).cpu()
        test_logits = model(Xtest.to(device)).cpu()

    test_expert_nll = _expert_nll(test_logits, Ytest)   # (n_test,)

    # For each nominal alpha: calibrate quantile on CAL, validate on TEST.
    per_alpha = []
    filters = {}
    for alpha in ALPHAS:
        cf = calibrate_conformal(cal_logits, Ycal, alpha=alpha)
        filters[alpha] = cf

        # Honest coverage on the held-out TEST set: raw conformal-set membership.
        test_cover = float((test_expert_nll <= cf.nll_quantile).float().mean())
        # Circular coverage on the calibration set (for contrast only).
        cal_expert_nll = _expert_nll(cal_logits, Ycal)
        cal_cover = float((cal_expert_nll <= cf.nll_quantile).float().mean())
        # Mean conformal set size on the test set.
        test_log_probs = F.log_softmax(test_logits, dim=-1)
        test_nll_all = -test_log_probs                    # (n_test, n_actions)
        set_sizes = (test_nll_all <= cf.nll_quantile).sum(dim=-1).float()

        per_alpha.append({
            "alpha": alpha,
            "nominal_coverage": round(1.0 - alpha, 4),
            "nll_quantile": round(cf.nll_quantile, 5),
            "empirical_coverage_TEST": round(test_cover, 5),
            "coverage_gap_TEST": round(test_cover - (1.0 - alpha), 5),
            "empirical_coverage_calibration_circular": round(cal_cover, 5),
            "mean_set_size_TEST": round(float(set_sizes.mean()), 3),
            "median_set_size_TEST": round(float(set_sizes.median()), 3),
            "min_set_size_TEST": int(set_sizes.min()),
            "max_set_size_TEST": int(set_sizes.max()),
        })
        logger.info("[conformal] alpha=%.2f nominal=%.3f  TEST coverage=%.4f "
                    "(gap=%+.4f)  cal(circular)=%.4f  mean|set|=%.2f",
                    alpha, 1.0 - alpha, test_cover, test_cover - (1.0 - alpha),
                    cal_cover, float(set_sizes.mean()))

    # Save the default (alpha=0.1) filter + ref policy.
    default_cf = filters[0.10]
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ref_policy_state_dict": model.state_dict(),
        "conformal_filter": default_cf.to_dict(),
        "in_dim": int(state.size(-1)),
        "n_actions": N_ACTIONS,
    }, OUT_PT)
    logger.info("[conformal] saved default (alpha=0.1) filter -> %s", OUT_PT)

    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "status": "OK",
        "transitions_source": str(TRANS_NPZ.relative_to(REPO_ROOT)).replace("\\", "/"),
        "transitions_sha256": _sha256(TRANS_NPZ),
        "n_transitions_total": int(n),
        "n_unique_actions_in_data": n_unique_actions,
        "n_action_slots": N_ACTIONS,
        "split": {"train": int(len(tr_idx)), "calibration": int(len(cal_idx)),
                  "test_heldout": int(len(test_idx)), "seed": 42},
        "ref_policy": {
            "arch": "MLP(in->128->128->280), dropout 0.1, AdamW lr=3e-4",
            "epochs": epochs,
            "train_losses": [round(x, 4) for x in train_losses],
            "final_train_loss": round(train_losses[-1], 4),
        },
        "conformal_coverage_by_alpha": per_alpha,
        "headline": {
            "target_alpha": 0.10,
            "nominal_coverage": 0.90,
            "empirical_coverage_on_heldout_test": per_alpha[1]["empirical_coverage_TEST"],
            "coverage_gap": per_alpha[1]["coverage_gap_TEST"],
        },
        "weights_path": str(OUT_PT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "elapsed_s": round(time.time() - t0, 2),
        "method": (
            "Split-conformal (Vovk 2005) with a genuine held-out test set. Real "
            "harvested RAP-XC transitions split 60/20/20 into train/calibration/"
            "test. A small MLP reference policy is behaviour-cloned on train; the "
            "(1-alpha) NLL quantile of the expert action is fit on calibration "
            "(finite-sample corrected); empirical coverage is then measured on "
            "the disjoint held-out TEST set the quantile never saw. Reported "
            "coverage is raw conformal-set membership P[NLL(expert)<=q], not the "
            "product always-accept-argmax mask. Marginal coverage assumes "
            "exchangeability; transitions are shuffled across trajectories before "
            "splitting to approximate it (autocorrelation within a trajectory is "
            "a known caveat)."
        ),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("[conformal] receipt: %s", RECEIPT)
    print(json.dumps({
        "status": receipt["status"],
        "split": receipt["split"],
        "headline": receipt["headline"],
        "conformal_coverage_by_alpha": per_alpha,
    }, indent=2))


if __name__ == "__main__":
    main()
