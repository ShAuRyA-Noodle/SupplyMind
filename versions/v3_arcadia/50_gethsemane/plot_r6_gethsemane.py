"""Plot R6 Gethsemane RL benchmark results."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R6_GETHSEMANE.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "gethsemane"
PLOTS.mkdir(parents=True, exist_ok=True)

tasks = list(d["tasks"].keys())
policies = ["random", "greedy", "ppo_v3"]
colors = {"random": "#888", "greedy": "#fdae61", "ppo_v3": "#1f77b4"}

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(tasks))
w = 0.26
for i, pol in enumerate(policies):
    means = [d["tasks"][t].get(pol, {}).get("reward_mean", 0) for t in tasks]
    stds = [d["tasks"][t].get(pol, {}).get("reward_std", 0) for t in tasks]
    ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, label=pol, color=colors[pol])
ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=10)
ax.set_ylabel("mean episode reward")
ax.set_title("R6 Gethsemane — PPO v3 vs random vs greedy")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS / "r6_gethsemane.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved r6_gethsemane.png")
