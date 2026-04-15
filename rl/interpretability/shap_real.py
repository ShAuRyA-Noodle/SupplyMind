"""
Phase F — SHAP feature-importance analysis on the REAL-trained BC policy.

Uses DeepExplainer only (no GradientExplainer fallback).
Background dataset = 1,000 real states from real_train.npz.
Explains 500 held-out real_test.npz states.

Reports global top-20 feature importances + per-group (NOAA/FRED/USGS/node) aggregates.

Output: rl/checkpoints/shap_real.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "rl" / "checkpoints" / "bc_best_real_v2.pt"
TRAIN = ROOT / "rl" / "data" / "real_train.npz"
TEST = ROOT / "rl" / "data" / "real_test.npz"
OUT = ROOT / "rl" / "checkpoints" / "shap_real.json"


class BCNetwork(nn.Module):
    def __init__(self, state_dim=408, action_dim=280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=False),
            nn.Linear(256, 128), nn.ReLU(inplace=False),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def feature_group(idx: int) -> str:
    if idx < 350:
        node = idx // 10
        feat = idx % 10
        feat_name = ["operational", "risk", "inv_days", "backup", "t0", "t1", "t2", "t3", "type", "revenue"][feat]
        return f"node{node}_{feat_name}"
    if idx < 380:
        return f"NOAA_{idx - 350}"
    if idx < 400:
        return f"USGS_{idx - 380}"
    if idx < 407:
        fred_names = ["oil_wti", "copper", "twd_fx", "krw_fx", "jpy_fx", "eur_fx", "cny_fx"]
        return f"FRED_{fred_names[idx - 400]}"
    return "status"


def main():
    import shap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BCNetwork().to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("BC real model loaded.")

    train = np.load(TRAIN)
    test = np.load(TEST)
    log.info(f"Train states: {train['states'].shape}, test: {test['states'].shape}")

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(train["states"]), size=500, replace=False)
    bg = torch.from_numpy(train["states"][bg_idx].astype(np.float32)).to(device)

    test_idx = rng.choice(len(test["states"]), size=200, replace=False)
    te = torch.from_numpy(test["states"][test_idx].astype(np.float32)).to(device)

    log.info("Building DeepExplainer...")
    explainer = shap.DeepExplainer(model, bg)

    log.info("Computing SHAP values...")
    shap_values = explainer.shap_values(te)  # list of [N, F] for each class, or [N, F, C]
    if isinstance(shap_values, list):
        # Stack into [N, F, C]
        arr = np.stack(shap_values, axis=-1)
    else:
        arr = shap_values

    # Global importance = mean |shap| over samples and classes
    if arr.ndim == 3:
        importance = np.mean(np.abs(arr), axis=(0, 2))
    else:
        importance = np.mean(np.abs(arr), axis=0)

    # Top-20 features
    top20 = np.argsort(importance)[-20:][::-1]
    top20_list = [{"feature_idx": int(i), "name": feature_group(int(i)), "importance": float(importance[i])}
                  for i in top20]

    # Aggregate by group
    groups = {}
    for i, imp in enumerate(importance):
        g = feature_group(i)
        # Coarse group
        if g.startswith("FRED_"):
            key = "FRED"
        elif g.startswith("NOAA_"):
            key = "NOAA"
        elif g.startswith("USGS_"):
            key = "USGS"
        elif g.startswith("node"):
            key = "node_features"
        else:
            key = g
        groups[key] = groups.get(key, 0.0) + float(imp)

    total = sum(groups.values())
    group_shares = {k: v / total for k, v in groups.items()}

    log.info("Top-20 features by SHAP importance:")
    for t in top20_list[:10]:
        log.info(f"  {t['name']:<30} idx={t['feature_idx']:<4} importance={t['importance']:.5f}")
    log.info(f"Group shares: {group_shares}")

    out = {
        "n_background": int(bg.shape[0]),
        "n_explained": int(te.shape[0]),
        "top20": top20_list,
        "group_importance": groups,
        "group_shares": group_shares,
        "checkpoint": str(CKPT),
    }
    OUT.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {OUT}")


if __name__ == "__main__":
    main()
