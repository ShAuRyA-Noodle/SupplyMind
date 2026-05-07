"""
Phase S "Aqua Regia" — Full MC Dropout on v2 agents + reliability plot.

Upgrades:
  U28 MC Dropout on CQL (best agent) + BC + IQL + TD3BC
  U29 Full 27K test set (not sample)
  U30 Reliability diagram PNG committed

Calibration: 10-bin reliability plot per agent, report ECE.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints"
DATA = ROOT / "rl" / "data"
RESULTS = ROOT / "benchmark" / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


class FactorizedPolicyDropout(nn.Module):
    """Policy matching baselines_v2 FactorizedPolicy with dropout injected for MC sampling."""
    def __init__(self, state_dim=408, hidden=512, p=0.1):
        super().__init__()
        self.trunk_mods = nn.ModuleList([
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(p),
        ])
        self.type_head = nn.Linear(256, 7)
        self.node_head = nn.Linear(256, 40)

    def forward(self, x):
        for m in self.trunk_mods:
            x = m(x)
        return self.type_head(x), self.node_head(x)


def load_as_mc_policy(path: Path, device: str, has_q_state=False):
    """Load saved FactorizedPolicy weights into MC dropout wrapper.
    Original trunk is Sequential(Linear, GELU, Linear, GELU, Linear, GELU).
    Our dropout version injects Dropout after each GELU; indices line up:
      trunk.0 -> trunk_mods.0
      trunk.2 -> trunk_mods.3
      trunk.4 -> trunk_mods.6
    """
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    m = FactorizedPolicyDropout().to(device)
    mapped = {}
    pos_map = {"trunk.0": "trunk_mods.0", "trunk.2": "trunk_mods.3", "trunk.4": "trunk_mods.6"}
    for k, v in sd.items():
        # Key format: trunk.0.weight, trunk.0.bias, trunk.2.weight, ...
        for old_pref, new_pref in pos_map.items():
            if k.startswith(old_pref + "."):
                mapped[k.replace(old_pref, new_pref)] = v
                break
        else:
            mapped[k] = v
    missing, unexpected = m.load_state_dict(mapped, strict=False)
    log.info(f"  {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    return m


def reliability_diagram(probs: np.ndarray, correct: np.ndarray, n_bins=10) -> tuple:
    """Compute reliability curve + ECE.
    probs: max posterior probability per sample, correct: 0/1."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_conf, bin_acc, bin_n = [], [], [], []
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        n = mask.sum()
        if n > 0:
            conf = probs[mask].mean()
            acc = correct[mask].mean()
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_conf.append(float(conf))
            bin_acc.append(float(acc))
            bin_n.append(int(n))
            ece += n / N * abs(acc - conf)
    return bin_centers, bin_conf, bin_acc, bin_n, float(ece)


def mc_eval(name, path, s_te, a_type_gt, a_node_gt, device, n_passes=50):
    log.info(f"MC Dropout on {name}...")
    model = load_as_mc_policy(path, device)
    model.train()  # keep dropout active
    probs_accum = None
    with torch.no_grad():
        for _ in range(n_passes):
            tl, nl = model(s_te)
            p_t = F.softmax(tl, dim=-1)
            p_n = F.softmax(nl, dim=-1)
            if probs_accum is None:
                probs_accum = {"t": torch.zeros_like(p_t), "n": torch.zeros_like(p_n),
                               "t_sq": torch.zeros_like(p_t), "n_sq": torch.zeros_like(p_n)}
            probs_accum["t"] += p_t / n_passes
            probs_accum["n"] += p_n / n_passes
    mean_t = probs_accum["t"].cpu().numpy()
    mean_n = probs_accum["n"].cpu().numpy()
    pred_t = mean_t.argmax(-1)
    pred_n = mean_n.argmax(-1)
    correct_full = ((pred_t == a_type_gt) & (pred_n == a_node_gt)).astype(np.int32)
    correct_type = (pred_t == a_type_gt).astype(np.int32)

    # Max probability = confidence
    max_p = np.maximum(mean_t.max(-1), mean_n.max(-1))  # Rough confidence proxy
    conf_full = mean_t.max(-1) * mean_n.max(-1)  # joint confidence
    bc, bconf, bacc, bn, ece_full = reliability_diagram(conf_full, correct_full)
    _, bconf_t, bacc_t, _, ece_type = reliability_diagram(mean_t.max(-1), correct_type)

    # Uncertainty buckets (epistemic proxy: 1 - max_prob)
    unc = 1 - conf_full
    q = np.quantile(unc, [0.25, 0.5, 0.75])
    buckets = {}
    for label, mask in [("Q1_low_unc", unc <= q[0]), ("Q2", (unc > q[0]) & (unc <= q[1])),
                         ("Q3", (unc > q[1]) & (unc <= q[2])), ("Q4_high_unc", unc > q[2])]:
        buckets[label] = {"n": int(mask.sum()), "accuracy": float(correct_full[mask].mean()) if mask.sum() > 0 else None,
                          "mean_confidence": float(conf_full[mask].mean())}
    log.info(f"  {name}: acc={correct_full.mean():.4f}, ECE_full={ece_full:.4f}, ECE_type={ece_type:.4f}")

    return {
        "accuracy": float(correct_full.mean()),
        "type_accuracy": float(correct_type.mean()),
        "ECE_full": ece_full,
        "ECE_type": ece_type,
        "reliability_full": {"confidence": bconf, "accuracy": bacc, "n_per_bin": bn, "bin_centers": bc},
        "reliability_type": {"confidence": bconf_t, "accuracy": bacc_t},
        "uncertainty_buckets": buckets,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = np.load(str(DATA / "real_test_v2.npz"))
    s_te = torch.from_numpy(test["states"].astype(np.float32)).to(device)
    a_type_gt = test["actions"][:, 0].astype(np.int64)
    a_node_gt = test["actions"][:, 1].astype(np.int64)
    log.info(f"Test set: {len(s_te):,}")

    results = {}
    for name, fname in [("BC_v2", "bc_v2.pt"), ("CQL_v2", "cql_v2.pt"),
                         ("IQL_v2", "iql_v2.pt"), ("TD3BC_v2", "td3bc_v2.pt")]:
        p = CKPT / fname
        if not p.exists():
            log.warning(f"  {fname} missing")
            continue
        # IQL checkpoint stores Q + V state too; only need policy
        results[name] = mc_eval(name, p, s_te, a_type_gt, a_node_gt, device)

    # Save results + reliability plot
    (CKPT / "mc_dropout_v2.json").write_text(json.dumps(results, indent=2))
    log.info("Saved mc_dropout_v2.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for name, r in results.items():
            rel = r["reliability_full"]
            ax[0].plot(rel["confidence"], rel["accuracy"], "o-", label=f"{name} (ECE={r['ECE_full']:.3f})")
            rel2 = r["reliability_type"]
            ax[1].plot(rel2["confidence"], rel2["accuracy"], "o-", label=f"{name} (ECE={r['ECE_type']:.3f})")
        for a, title in [(ax[0], "Full-action calibration"), (ax[1], "Action-type calibration")]:
            a.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
            a.set_xlabel("Predicted confidence"); a.set_ylabel("Empirical accuracy")
            a.set_title(title); a.legend(); a.grid(alpha=0.3); a.set_xlim(0, 1); a.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(PLOTS / "reliability_v2.png", dpi=120, bbox_inches="tight")
        log.info(f"Saved reliability_v2.png")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    log.info("Phase S 'Aqua Regia' complete.")


if __name__ == "__main__":
    main()
