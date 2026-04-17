"""Plot R6 Aqua Regia conformal coverage results."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R6_AQUA_REGIA.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "aqua_regia"
PLOTS.mkdir(parents=True, exist_ok=True)

targets = list(d["results"].keys())
forecasters = ["arima", "chronos"]
alphas = [0.2, 0.1, 0.05]  # error rates
nominals = [1 - a for a in alphas]

fig, axs = plt.subplots(len(forecasters), 2, figsize=(12, 4 * len(forecasters)))
if len(forecasters) == 1: axs = [axs]

for fi, forecaster in enumerate(forecasters):
    # Left: coverage vs nominal
    ax_cov = axs[fi][0]
    for ti, target in enumerate(targets):
        r = d["results"][target].get(forecaster, {})
        if "error" in r: continue
        bare = [r.get(f"alpha={a}", {}).get("bare_coverage_mean") for a in alphas]
        conf = [r.get(f"alpha={a}", {}).get("conformal_coverage_mean") for a in alphas]
        bare = [b if b is not None else np.nan for b in bare]
        conf = [c if c is not None else np.nan for c in conf]
        ax_cov.plot(nominals, bare, "o-", label=f"{target} bare", alpha=0.6)
        ax_cov.plot(nominals, conf, "s--", label=f"{target} conformal", alpha=0.6)
    ax_cov.plot([0.7, 1.0], [0.7, 1.0], "k:", alpha=0.4, label="perfect")
    ax_cov.set_xlabel("nominal coverage"); ax_cov.set_ylabel("empirical coverage")
    ax_cov.set_title(f"{forecaster.upper()} — coverage vs nominal")
    ax_cov.grid(alpha=0.3); ax_cov.legend(fontsize=7, loc="lower right")
    ax_cov.set_xlim(0.7, 1.0); ax_cov.set_ylim(0.4, 1.05)

    # Right: width comparison
    ax_w = axs[fi][1]
    for ti, target in enumerate(targets):
        r = d["results"][target].get(forecaster, {})
        if "error" in r: continue
        bare_w = [r.get(f"alpha={a}", {}).get("bare_width_mean") for a in alphas]
        conf_w = [r.get(f"alpha={a}", {}).get("conformal_width_mean") for a in alphas]
        bare_w = [b if b is not None else np.nan for b in bare_w]
        conf_w = [c if c is not None else np.nan for c in conf_w]
        ax_w.plot(nominals, bare_w, "o-", label=f"{target} bare", alpha=0.6)
        ax_w.plot(nominals, conf_w, "s--", label=f"{target} conformal", alpha=0.6)
    ax_w.set_xlabel("nominal coverage"); ax_w.set_ylabel("mean interval width")
    ax_w.set_title(f"{forecaster.upper()} — interval width")
    ax_w.grid(alpha=0.3); ax_w.legend(fontsize=7, loc="upper left")
    ax_w.set_yscale("log")

plt.tight_layout()
plt.savefig(PLOTS / "r6_aqua_regia.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved r6_aqua_regia.png")
