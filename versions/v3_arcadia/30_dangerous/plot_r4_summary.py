"""R4 Dangerous summary plot: judge agreement + per-judge latency + risk distribution."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "dangerous"
PLOTS.mkdir(parents=True, exist_ok=True)

judges = d["judges"]
scenarios = list(d["per_scenario"].keys())
short = [s.replace("_", " ")[:25] for s in scenarios]

# Panel 1: risk_level per judge per scenario (heatmap 3 x N)
risk_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
mat = np.full((len(judges), len(scenarios)), np.nan)
for ji, j in enumerate(judges):
    for si, s in enumerate(scenarios):
        pj = d["per_scenario"][s]["per_judge"].get(j, {})
        p = pj.get("parsed") or {}
        rl = str(p.get("risk_level", "")).upper()
        if rl in risk_map:
            mat[ji, si] = risk_map[rl]

fig, axs = plt.subplots(2, 2, figsize=(16, 9))

ax = axs[0, 0]
im = ax.imshow(mat, cmap="YlOrRd", vmin=1, vmax=4, aspect="auto")
ax.set_yticks(range(len(judges))); ax.set_yticklabels(judges, fontsize=9)
ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
for ji in range(len(judges)):
    for si in range(len(scenarios)):
        v = mat[ji, si]
        if np.isnan(v):
            ax.text(si, ji, "x", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
        else:
            lbl = {1: "L", 2: "M", 3: "H", 4: "C"}[int(v)]
            ax.text(si, ji, lbl, ha="center", va="center", color="white" if v >= 3 else "black", fontsize=9)
cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4])
cbar.ax.set_yticklabels(["LOW", "MED", "HIGH", "CRIT"])
ax.set_title("Risk level per judge  (x = parse failure)")

# Panel 2: per-scenario consensus alpha
ax = axs[0, 1]
alphas = [d["per_scenario"][s]["consensus"]["risk_alpha_ordinal"] for s in scenarios]
alphas = [0.0 if a is None or np.isnan(a) else a for a in alphas]
bars = ax.barh(range(len(scenarios)), alphas, color="#2ca02c", alpha=0.85)
ax.set_yticks(range(len(scenarios))); ax.set_yticklabels(short, fontsize=8)
ax.axvline(0.80, color="black", linestyle="--", alpha=0.5, label="high-agreement threshold")
ax.set_xlabel("ordinal alpha (1=unanimous, 0=chance)")
ax.set_title(f"Risk-level agreement per scenario  (mean={d['summary']['mean_risk_alpha']:.3f})")
ax.set_xlim(0, 1.05); ax.grid(alpha=0.3, axis="x")
ax.legend(fontsize=8)

# Panel 3: per-judge success rate + latency
ax = axs[1, 0]
succ = d["summary"]["parse_success_rate_per_judge"]
lat = d["summary"]["mean_latency_s_per_judge"]
x = np.arange(len(judges))
ax.bar(x - 0.2, [succ[j] for j in judges], 0.4, label="parse success", color="#1f77b4")
ax2 = ax.twinx()
ax2.bar(x + 0.2, [lat[j] for j in judges], 0.4, label="mean latency (s)", color="#ff7f0e", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(judges, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("success rate", color="#1f77b4")
ax2.set_ylabel("latency (s)", color="#ff7f0e")
ax.set_ylim(0, 1.1)
ax.set_title("Per-judge parse success + latency")

# Panel 4: risk level distribution (majority per scenario)
ax = axs[1, 1]
majorities = [d["per_scenario"][s]["consensus"]["risk_majority"] for s in scenarios]
counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "UNKNOWN": 0}
for m in majorities: counts[m] = counts.get(m, 0) + 1
order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
vals = [counts[o] for o in order]
ax.bar(order, vals, color=["#66c2a5", "#fdae61", "#f46d43", "#a50026"])
for i, v in enumerate(vals): ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)
ax.set_title("Majority-vote risk distribution across 10 scenarios")
ax.set_ylabel("count"); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = PLOTS / "r4_summary.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
