"""
Phase Z "The Offering" — grand benchmark with statistical rigor.

All agents x test set, plus Wilcoxon + bootstrap CI.
Lightweight version focusing on classification accuracy on real_test_v2 (27K transitions),
because full 10,800-episode env rollout is prohibitively expensive in one session.

Output:
  benchmark/results/GRAND_BENCHMARK_V2.csv + .json
  benchmark/results/PAIRWISE_WILCOXON_V2.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "rl" / "checkpoints"
DATA = ROOT / "rl" / "data"
RESULTS = ROOT / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(correct, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(correct)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = correct[idx].mean()
    return float(correct.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as wx
        diff = a.astype(int) - b.astype(int)
        nz = diff[diff != 0]
        if len(nz) < 10:
            return {"stat": None, "p": None, "n": int(len(nz))}
        s, p = wx(nz)
        return {"stat": float(s), "p": float(p), "n": int(len(nz))}
    except Exception as e:
        return {"error": str(e)}


def predict_agent(name, path, s_te, device):
    from rl.offline.baselines_v2 import FactorizedPolicy, FactorizedTwinQ
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    # Detect: is it policy (trunk.*) or twin-Q (t1.*)?
    if any(k.startswith("trunk.") for k in sd):
        m = FactorizedPolicy().to(device); m.load_state_dict(sd, strict=False)
        kind = "policy"
    else:
        m = FactorizedTwinQ().to(device); m.load_state_dict(sd, strict=False)
        kind = "q"
    m.eval()
    with torch.no_grad():
        if kind == "policy":
            tl, nl = m(s_te)
        else:
            tl, nl = m.q1(s_te)
    return tl.argmax(-1).cpu().numpy(), nl.argmax(-1).cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = np.load(str(DATA / "real_test_v2.npz"))
    s_te = torch.from_numpy(test["states"].astype(np.float32)).to(device)
    a_type_gt = test["actions"][:, 0].astype(np.int64)
    a_node_gt = test["actions"][:, 1].astype(np.int64)
    a_flat_gt = a_type_gt * 40 + a_node_gt
    N = len(s_te)
    log.info(f"Test: {N:,} transitions")

    # Random baseline
    rng = np.random.default_rng(42)
    rnd_t = rng.integers(0, 7, size=N)
    rnd_n = rng.integers(0, 40, size=N)
    agents_pred = {"Random": (rnd_t, rnd_n)}

    # Scripted baseline (always "alert" = 1, target node 0)
    agents_pred["Scripted_Alert"] = (np.ones(N, dtype=np.int64), np.zeros(N, dtype=np.int64))

    # All v2 agents
    for name, fname in [("BC_v2", "bc_v2.pt"), ("CQL_v2", "cql_v2.pt"),
                         ("IQL_v2", "iql_v2.pt"), ("TD3BC_v2", "td3bc_v2.pt"),
                         ("Federated_v2", "federated_v2.pt")]:
        p = CKPT / fname
        if not p.exists():
            log.warning(f"  {fname} missing")
            continue
        try:
            agents_pred[name] = predict_agent(name, p, s_te, device)
        except Exception as e:
            log.error(f"  {name}: {e}")

    # Also legacy v1 for comparison
    for name, fname in [("BC_v1", "bc_best_real_v2.pt"), ("CQL_v1", "cql_best_real_v2.pt")]:
        p = CKPT / fname
        if not p.exists():
            continue
        try:
            import torch.nn as nn
            class BCv1(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(408, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 280))
                def forward(self, x): return self.net(x)
            class CQLv1(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q1 = nn.Sequential(nn.Linear(408, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 280))
                    self.q2 = nn.Sequential(nn.Linear(408, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 280))
                def forward(self, x): return torch.min(self.q1(x), self.q2(x))
            m = (BCv1() if "BC" in name else CQLv1()).to(device)
            ck = torch.load(p, map_location=device)
            m.load_state_dict(ck.get("state_dict", ck), strict=False)
            m.eval()
            with torch.no_grad():
                logits = m(s_te)
            pred = logits.argmax(-1).cpu().numpy()
            agents_pred[name] = (pred // 40, pred % 40)
        except Exception as e:
            log.warning(f"  {name}: {e}")

    # Aggregate results
    summary = []
    correct_vectors = {}
    for name, (pt, pn) in agents_pred.items():
        correct_full = ((pt == a_type_gt) & (pn == a_node_gt)).astype(np.int32)
        correct_type = (pt == a_type_gt).astype(np.int32)
        correct_node = (pn == a_node_gt).astype(np.int32)
        correct_vectors[name] = correct_full

        full_mean, full_lo, full_hi = bootstrap_ci(correct_full)
        type_mean, type_lo, type_hi = bootstrap_ci(correct_type)
        node_mean, node_lo, node_hi = bootstrap_ci(correct_node)

        summary.append({
            "agent": name,
            "full_acc": full_mean, "full_ci95_lo": full_lo, "full_ci95_hi": full_hi,
            "type_acc": type_mean, "type_ci95_lo": type_lo, "type_ci95_hi": type_hi,
            "node_acc": node_mean, "node_ci95_lo": node_lo, "node_ci95_hi": node_hi,
        })
        log.info(f"  {name}: full={full_mean:.4f} [{full_lo:.4f},{full_hi:.4f}] type={type_mean:.4f}")

    # Pairwise Wilcoxon
    pw = {}
    names = list(agents_pred.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pw[f"{names[i]}_vs_{names[j]}"] = wilcoxon(correct_vectors[names[i]], correct_vectors[names[j]])

    # Save CSV + JSON
    fields = list(summary[0].keys())
    with open(RESULTS / "GRAND_BENCHMARK_V2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in summary: w.writerow(r)
    (RESULTS / "GRAND_BENCHMARK_V2.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "PAIRWISE_WILCOXON_V2.json").write_text(json.dumps(pw, indent=2))

    log.info("Phase Z 'The Offering' complete.")


if __name__ == "__main__":
    main()
