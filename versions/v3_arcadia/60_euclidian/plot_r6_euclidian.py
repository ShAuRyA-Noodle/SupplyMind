"""Plot R6 Euclidian 10,800-episode benchmark with bootstrap CIs."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R6_EUCLIDIAN.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "euclidian"
PLOTS.mkdir(parents=True, exist_ok=True)

tasks = list(d["tasks"].keys())
policies = set()
for t in tasks:
    policies.update(d["tasks"][t].keys())
policies = sorted(policies, key=lambda p: ["random", "greedy", "ppo_v2", "ppo_v3"].index(p)
                  if p in ["random", "greedy", "ppo_v2", "ppo_v3"] else 99)
colors = {"random": "#888", "greedy": "#fdae61", "ppo_v2": "#9467bd", "ppo_v3": "#1f77b4"}

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(tasks))
w = 0.8 / len(policies)
for i, pol in enumerate(policies):
    means = []
    lo = []
    hi = []
    for t in tasks:
        s = d["tasks"][t].get(pol, {})
        m = s.get("reward_mean", 0)
        ci = s.get("reward_ci95", [m, m])
        means.append(m)
        lo.append(m - ci[0])
        hi.append(ci[1] - m)
    offset = (i - (len(policies) - 1) / 2) * w
    ax.bar(x + offset, means, w, yerr=[lo, hi], capsize=4, label=pol,
           color=colors.get(pol, "#333"), alpha=0.88)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=8)
ax.set_ylabel("mean episode reward (bootstrap 95% CI)")
ax.set_title(f"R6 Euclidian — {d.get('total_episodes', 0):,}-episode benchmark")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS / "r6_euclidian.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved r6_euclidian.png")
