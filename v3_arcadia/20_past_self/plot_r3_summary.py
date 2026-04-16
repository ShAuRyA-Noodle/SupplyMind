"""R3 Past Self summary plots: backtest MAE + PICP80 + DirAcc per target x horizon."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "v3_arcadia" / "results" / "R3_PAST_SELF.json"
PLOTS = ROOT / "v3_arcadia" / "plots" / "past_self"
PLOTS.mkdir(parents=True, exist_ok=True)

d = json.loads(RESULTS.read_text())
models = ["chronos", "timesfm", "arima", "prophet"]
colors = {"chronos": "#1f77b4", "timesfm": "#ff7f0e", "arima": "#2ca02c", "prophet": "#d62728"}
targets = [t for t in d["per_target"] if any(d["per_target"][t].get(f"h{h}", {}).get("backtest_agg") for h in [7, 14, 28])]
horizons = [7, 14, 28]

fig, axs = plt.subplots(3, 3, figsize=(18, 12))
# Row 1: normalized MAE (vs target median)
# Row 2: PICP80 (target is 0.80)
# Row 3: DirAcc (target is >0.5)
for ci, h in enumerate(horizons):
    # MAE
    ax = axs[0, ci]
    x = np.arange(len(targets))
    w = 0.2
    for mi, m in enumerate(models):
        vals = []
        for t in targets:
            bt = d["per_target"][t].get(f"h{h}", {}).get("backtest_agg", {})
            mae = bt.get(m, {}).get("mean_mae", np.nan)
            # normalize by min across models for this target/h
            vals.append(mae)
        vals = np.array(vals, dtype=float)
        # normalize by max to put all on 0-1
        norm = np.array([v / np.nanmax([d["per_target"][t].get(f"h{h}", {}).get("backtest_agg", {}).get(mm, {}).get("mean_mae", np.nan) for mm in models]) if not np.isnan(v) else 0 for t, v in zip(targets, vals)])
        ax.bar(x + (mi - 1.5) * w, norm, w, label=m, color=colors[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Relative MAE (normalized) h={h}")
    ax.set_ylabel("MAE / worst")
    if ci == 0: ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # PICP80
    ax = axs[1, ci]
    for mi, m in enumerate(models):
        vals = [d["per_target"][t].get(f"h{h}", {}).get("backtest_agg", {}).get(m, {}).get("mean_picp80") for t in targets]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + (mi - 1.5) * w, vals, w, label=m, color=colors[m], alpha=0.85)
    ax.axhline(0.80, color="black", linestyle="--", alpha=0.7, label="nominal 0.80")
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"PICP@80% h={h}  (closer to 0.80 = better calibration)")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1)
    if ci == 0: ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    # DirAcc
    ax = axs[2, ci]
    for mi, m in enumerate(models):
        vals = [d["per_target"][t].get(f"h{h}", {}).get("backtest_agg", {}).get(m, {}).get("mean_dir_acc") for t in targets]
        vals = [np.nan if v is None else v for v in vals]
        ax.bar(x + (mi - 1.5) * w, vals, w, label=m, color=colors[m], alpha=0.85)
    ax.axhline(0.50, color="black", linestyle="--", alpha=0.7, label="chance 0.50")
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Direction Accuracy h={h}")
    ax.set_ylabel("acc")
    ax.set_ylim(0, 1)
    if ci == 0: ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = PLOTS / "r3_summary.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
