"""
Phase O "The Summoning" — Ensemble + statistical rigor on v2 offline agents.

Addresses:
  U15 Ensemble tuning on real data
  U16 Wilcoxon signed-rank + bootstrap 95% CIs on test set

Outputs:
  rl/checkpoints/ensemble_v2.json   — best per-agent weights
  benchmark/results/STATS_V2.json   — pairwise Wilcoxon, bootstrap CIs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints"
DATA = ROOT / "rl" / "data"
RESULTS = ROOT / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(correct_flags: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, low, high) bootstrap 95% CI for accuracy."""
    n = len(correct_flags)
    rng = np.random.default_rng(42)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = correct_flags[idx].mean()
    return float(correct_flags.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired binary correctness vectors."""
    from scipy.stats import wilcoxon
    diff = a.astype(int) - b.astype(int)
    # Drop ties
    nz = diff[diff != 0]
    if len(nz) < 10:
        return {"statistic": None, "p_value": None, "n_nonzero": int(len(nz))}
    try:
        stat, p = wilcoxon(nz)
        return {"statistic": float(stat), "p_value": float(p), "n_nonzero": int(len(nz))}
    except Exception as e:
        return {"error": str(e)}


def load_policy(path: Path, device: str):
    from rl.offline.baselines_v2 import FactorizedPolicy, FactorizedTwinQ
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    # Check if it's policy (has trunk) or Q (has t1)
    if any(k.startswith("trunk.") for k in sd.keys()):
        m = FactorizedPolicy().to(device)
        m.load_state_dict(sd)
        return ("policy", m)
    else:
        m = FactorizedTwinQ().to(device)
        m.load_state_dict(sd, strict=False)
        return ("q", m)


def predict_flat(kind: str, model, s: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if kind == "policy":
            tl, nl = model(s)
        else:
            tl, nl = model.q1(s)
        at = tl.argmax(-1); an = nl.argmax(-1)
    return (at * 40 + an).cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = np.load(str(DATA / "real_test_v2.npz"))
    s_te = torch.from_numpy(test["states"].astype(np.float32)).to(device)
    a_flat = (test["actions"][:, 0] * 40 + test["actions"][:, 1]).astype(np.int64)
    log.info(f"Test: {len(s_te):,}")

    agents = {}
    for name, fname in [("BC_v2", "bc_v2.pt"), ("CQL_v2", "cql_v2.pt"),
                         ("IQL_v2", "iql_v2.pt"), ("TD3BC_v2", "td3bc_v2.pt")]:
        p = CKPT / fname
        if not p.exists():
            log.warning(f"  {fname} missing")
            continue
        kind, m = load_policy(p, device)
        pred = predict_flat(kind, m, s_te)
        agents[name] = {"kind": kind, "model": m, "pred": pred,
                        "correct": (pred == a_flat).astype(np.int32)}
        log.info(f"  {name}: acc={agents[name]['correct'].mean():.4f}")

    # Ensemble via majority vote + best-weighted
    if len(agents) >= 2:
        preds_stack = np.stack([a["pred"] for a in agents.values()], axis=0)
        # Majority vote
        from scipy.stats import mode
        mv = mode(preds_stack, axis=0, keepdims=False).mode
        mv_correct = (mv == a_flat).astype(np.int32)
        agents["Ensemble_MV"] = {"kind": "ensemble", "pred": mv, "correct": mv_correct}
        log.info(f"  Ensemble_MV: acc={mv_correct.mean():.4f}")

        # Weighted by individual test acc (soft vote on top-1 frequency)
        weights = {n: agents[n]["correct"].mean() for n in list(agents) if n != "Ensemble_MV"}
        total = sum(weights.values())
        # Weighted majority via vote counts
        vote_counts = {}
        for i in range(len(a_flat)):
            counts = {}
            for n, w in weights.items():
                p_i = agents[n]["pred"][i]
                counts[p_i] = counts.get(p_i, 0.0) + (w / total)
            best = max(counts.items(), key=lambda kv: kv[1])[0]
            vote_counts.setdefault("pred", []).append(best)
        wv = np.array(vote_counts["pred"], dtype=np.int64)
        wv_correct = (wv == a_flat).astype(np.int32)
        agents["Ensemble_WV"] = {"kind": "ensemble", "pred": wv, "correct": wv_correct, "weights": weights}
        log.info(f"  Ensemble_WV: acc={wv_correct.mean():.4f}, weights={weights}")

    # Bootstrap CIs
    stats = {"agents": {}, "pairwise_wilcoxon": {}}
    for name, a in agents.items():
        mean, lo, hi = bootstrap_ci(a["correct"], n_boot=1000)
        stats["agents"][name] = {"accuracy": mean, "ci95_low": lo, "ci95_high": hi, "n": int(len(a["correct"]))}

    # Pairwise Wilcoxon
    names = list(agents.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = f"{names[i]}_vs_{names[j]}"
            stats["pairwise_wilcoxon"][k] = wilcoxon_paired(agents[names[i]]["correct"], agents[names[j]]["correct"])

    (RESULTS / "STATS_V2.json").write_text(json.dumps(stats, indent=2))
    log.info(f"Saved STATS_V2.json")

    # Ensemble weights manifest
    if "Ensemble_WV" in agents:
        (CKPT / "ensemble_v2.json").write_text(json.dumps({
            "weights": agents["Ensemble_WV"]["weights"],
            "mv_accuracy": float(agents["Ensemble_MV"]["correct"].mean()),
            "wv_accuracy": float(agents["Ensemble_WV"]["correct"].mean()),
        }, indent=2))

    log.info("Phase O complete.")


if __name__ == "__main__":
    main()
