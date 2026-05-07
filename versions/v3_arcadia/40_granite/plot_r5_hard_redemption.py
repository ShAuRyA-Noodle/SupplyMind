"""Plot R5 hard-query reranker redemption."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
easy = json.loads((ROOT / "v3_arcadia" / "results" / "R5_GRANITE.json").read_text())
hard = json.loads((ROOT / "v3_arcadia" / "results" / "R5_GRANITE_HARD.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "granite"
PLOTS.mkdir(parents=True, exist_ok=True)

# Pairs: (bi, rerank)
pairs = [
    ("P1_bge_m3_bi", "P4_bge_m3_rerank", "BGE-M3"),
    ("P2_mxbai_bi", "P5_mxbai_rerank", "mxbai"),
    ("P3_snowflake_bi", "P6_snowflake_rerank", "Snowflake"),
]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Left: P@1 — easy bi / easy rerank / hard bi / hard rerank
x = np.arange(len(pairs))
w = 0.2
colors = ["#1f77b4", "#6495ed", "#d62728", "#ff7f7f"]
labels = ["easy bi-encoder", "easy +reranker", "hard bi-encoder", "hard +reranker"]

easy_bi = [easy["pipelines"][p[0]]["p1"] for p in pairs]
easy_rr = [easy["pipelines"][p[1]]["p1"] for p in pairs]
hard_bi = [hard["pipelines"][p[0]]["p1"] for p in pairs]
hard_rr = [hard["pipelines"][p[1]]["p1"] for p in pairs]

axs[0].bar(x - 1.5*w, easy_bi, w, label=labels[0], color=colors[0])
axs[0].bar(x - 0.5*w, easy_rr, w, label=labels[1], color=colors[1])
axs[0].bar(x + 0.5*w, hard_bi, w, label=labels[2], color=colors[2])
axs[0].bar(x + 1.5*w, hard_rr, w, label=labels[3], color=colors[3])
axs[0].set_xticks(x); axs[0].set_xticklabels([p[2] for p in pairs])
axs[0].set_ylabel("P@1")
axs[0].set_title("Reranker redemption: lifts on HARD queries, flat/negative on EASY")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3, axis="y")
axs[0].set_ylim(0, 1.0)

# Right: Δ P@1 (rerank minus bi) — positive on hard, flat/negative on easy
easy_delta = [easy["pipelines"][p[1]]["p1"] - easy["pipelines"][p[0]]["p1"] for p in pairs]
hard_delta = [hard["pipelines"][p[1]]["p1"] - hard["pipelines"][p[0]]["p1"] for p in pairs]
axs[1].bar(x - 0.2, easy_delta, 0.4, label="easy queries", color="#1f77b4")
axs[1].bar(x + 0.2, hard_delta, 0.4, label="hard queries", color="#d62728")
axs[1].axhline(0, color="black", lw=0.5)
axs[1].set_xticks(x); axs[1].set_xticklabels([p[2] for p in pairs])
axs[1].set_ylabel("Δ P@1 (reranker − bi-encoder)")
axs[1].set_title("Reranker's P@1 contribution: negative on easy, positive on hard")
axs[1].legend()
axs[1].grid(alpha=0.3, axis="y")
for i, v in enumerate(easy_delta):
    axs[1].text(i - 0.2, v + (0.005 if v >= 0 else -0.015), f"{v:+.3f}", ha="center", fontsize=8)
for i, v in enumerate(hard_delta):
    axs[1].text(i + 0.2, v + (0.005 if v >= 0 else -0.015), f"{v:+.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS / "r5_hard_redemption.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved r5_hard_redemption.png")
