"""Plot R3-β TimesFM-CP vs Chronos-native quantile coverage deviation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "past_self"
PLOTS.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads((RESULTS / "R3_TIMESFM_QUANTILE.json").read_text())
    targets = list(d["targets"].keys())
    confs = [0.8, 0.9, 0.95]

    fig, axes = plt.subplots(1, len(targets), figsize=(13, 4.2), sharey=True)
    x = np.arange(len(confs))
    w = 0.38

    for ax, t in zip(axes, targets):
        v = d["targets"][t]
        tf = [v[f"timesfm_conf={c}"]["dev_from_nominal"] for c in confs]
        ch = [v[f"chronos_native_conf={c}"]["dev_from_nominal"] for c in confs]
        ax.bar(x - w / 2, tf, w, label="TimesFM-CP (split-conformal)", color="#2d6e9e", alpha=0.9, edgecolor="black")
        ax.bar(x + w / 2, ch, w, label="Chronos native", color="#c28850", alpha=0.9, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.0%}" for c in confs])
        ax.set_title(t)
        ax.set_xlabel("Nominal coverage")
        ax.grid(axis="y", alpha=0.3)
        for i, (a, b) in enumerate(zip(tf, ch)):
            ax.text(i - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(i + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)

    axes[0].set_ylabel("|empirical − nominal| (lower = better)")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("R3-β — TimesFM residual-quantile wrapper vs Chronos-native quantiles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOTS / "r3_timesfm_quantile.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
