"""Plot R6 Provider GNN results."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R6_PROVIDER.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "provider"
PLOTS.mkdir(parents=True, exist_ok=True)

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# Left: F1 comparison across graphs
diffs = list(d["graphs"].keys())
gnn_f1 = [d["graphs"][g]["gnn_final"]["f1"] for g in diffs]
base_f1 = [d["graphs"][g]["baseline_direct_neighbors"]["f1"] for g in diffs]
x = np.arange(len(diffs))
w = 0.35
axs[0].bar(x - w/2, gnn_f1, w, label="GNN (3-layer GCN)", color="#1f77b4")
axs[0].bar(x + w/2, base_f1, w, label="direct-neighbors baseline", color="#fdae61")
axs[0].set_xticks(x); axs[0].set_xticklabels(diffs)
axs[0].set_ylabel("F1"); axs[0].set_title("Disruption propagation F1 (BFS ground truth)")
axs[0].set_ylim(0, 1.05); axs[0].legend(); axs[0].grid(alpha=0.3, axis="y")

# Right: Training curves (each graph) — show F1 over epochs
for g in diffs:
    curve = [e["f1"] for e in d["graphs"][g]["test_metric_curve"]]
    axs[1].plot(curve, label=f"{g} ({d['graphs'][g]['n_nodes']} nodes)")
axs[1].set_xlabel("epoch"); axs[1].set_ylabel("test F1")
axs[1].set_title("Training curves")
axs[1].legend(); axs[1].grid(alpha=0.3); axs[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(PLOTS / "r6_provider.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved r6_provider.png")
