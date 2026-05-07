"""Plot R6 action-masking ablation: masked vs unmasked PPO on easy_typhoon_response."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "gethsemane"
PLOTS.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads((RESULTS / "R6_GETHSEMANE_MASKING_ABLATION.json").read_text())
    u, m = d["unmasked"], d["masked"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    labels = ["Unmasked PPO", "Masked PPO"]
    means = [u["reward_mean"], m["reward_mean"]]
    stds = [u["reward_std"], m["reward_std"]]
    colors = ["#c05555", "#3a7d3a"]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black")
    for b, mv in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mv + 0.02, f"{mv:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean reward (50 eval episodes)")
    ax.set_title(f"Action masking contribution: +{d['action_masking_contribution']['reward_pct_delta']:.1f}%")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.3)

    ax = axes[1]
    labels2 = ["Unmasked PPO", "Masked PPO"]
    inv = [u["invalid_action_picks_mean_per_ep"], m["invalid_action_picks_mean_per_ep"]]
    bars = ax.bar(labels2, inv, color=colors, alpha=0.85, edgecolor="black")
    for b, v in zip(bars, inv):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Invalid action picks per episode")
    ax.set_title("Invalid action picks: 13.64 → 0 (structurally)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("R6-α — Action masking ablation (easy_typhoon_response, 100k timesteps)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOTS / "r6_masking_ablation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
