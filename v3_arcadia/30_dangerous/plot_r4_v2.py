"""R4 V2 Beast plots: heatmap, calibration, confusion, latency, escalation + markdown report."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
d = json.loads((ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json").read_text())
PLOTS = ROOT / "v3_arcadia" / "plots" / "dangerous"
PLOTS.mkdir(parents=True, exist_ok=True)

judges = d["judges"]
critic = d["critic"]
scenarios = list(d["per_scenario"].keys())
RISK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 0}
LABELS = ["LOW", "MED", "HIGH", "CRIT"]

# ============================================================
# 1. Agreement heatmap (judges + majority + ground truth) x scenarios
# ============================================================
rows = judges + ["majority_vote", "GROUND_TRUTH"]
mat = np.full((len(rows), len(scenarios)), np.nan)
for si, s in enumerate(scenarios):
    sc = d["per_scenario"][s]
    for ji, j in enumerate(judges):
        p = sc["per_judge"].get(j, {}).get("parsed")
        if p and isinstance(p, dict):
            rl = str(p.get("risk_level", "")).upper()
            if rl in RISK and RISK[rl] > 0:
                mat[ji, si] = RISK[rl]
    maj = sc.get("risk_majority", "UNKNOWN")
    if maj in RISK and RISK[maj] > 0:
        mat[len(judges), si] = RISK[maj]
    gt = sc.get("ground_truth", "UNKNOWN")
    if gt in RISK and RISK[gt] > 0:
        mat[len(judges) + 1, si] = RISK[gt]

fig, ax = plt.subplots(figsize=(22, 6))
im = ax.imshow(mat, cmap="YlOrRd", vmin=1, vmax=4, aspect="auto")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels([s.replace("_", " ")[:28] for s in scenarios], rotation=55, ha="right", fontsize=7)
for ri in range(len(rows)):
    for si in range(len(scenarios)):
        v = mat[ri, si]
        if np.isnan(v):
            ax.text(si, ri, "x", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
        else:
            lbl = LABELS[int(v) - 1][0]
            ax.text(si, ri, lbl, ha="center", va="center",
                     color="white" if v >= 3 else "black", fontsize=8)
cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], shrink=0.8)
cbar.ax.set_yticklabels(LABELS)
ax.set_title(f"R4 V2: Risk-level assignments across 26 scenarios  (x = parse failure)\n"
             f"α={d['agreement']['krippendorff_alpha_ordinal']:.3f}, "
             f"κ={d['agreement']['fleiss_kappa_nominal']:.3f}")
plt.tight_layout()
plt.savefig(PLOTS / "r4v2_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved r4v2_heatmap.png")

# ============================================================
# 2. Calibration / reliability per judge
# ============================================================
fig, axs = plt.subplots(1, len(judges), figsize=(5 * len(judges), 4))
for i, j in enumerate(judges):
    ax = axs[i] if len(judges) > 1 else axs
    stats = d["calibration_ece"].get(j, {}).get("bins", [])
    ece = d["calibration_ece"].get(j, {}).get("ece")
    xs = [s["mean_conf"] for s in stats if s.get("mean_conf") is not None]
    ys = [s["accuracy"] for s in stats if s.get("mean_conf") is not None]
    sizes = [s["n"] * 60 for s in stats if s.get("mean_conf") is not None]
    ax.scatter(xs, ys, s=sizes, alpha=0.7, color="#1f77b4", edgecolors="k")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{j}\nECE={ece:.3f}" if ece is not None and not np.isnan(ece) else f"{j}\nECE=n/a")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS / "r4v2_calibration.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved r4v2_calibration.png")

# ============================================================
# 3. Confusion matrices (GT rows × Pred cols)
# ============================================================
targets = judges + ["majority_vote"]
fig, axs = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.2))
for i, t in enumerate(targets):
    ax = axs[i] if len(targets) > 1 else axs
    cm = np.array(d["confusion_matrices"].get(t, np.zeros((4, 4))))
    im = ax.imshow(cm, cmap="Blues")
    for ri in range(4):
        for ci in range(4):
            ax.text(ci, ri, int(cm[ri, ci]), ha="center", va="center",
                     color="white" if cm[ri, ci] > cm.max() * 0.5 else "black", fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    acc = d["accuracy_vs_ground_truth"].get(t, {}).get("accuracy", 0)
    ax.set_title(f"{t}\nacc={acc:.3f}")
plt.tight_layout()
plt.savefig(PLOTS / "r4v2_confusion.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved r4v2_confusion.png")

# ============================================================
# 4. Latency per judge (mean + distribution)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
data = []
labels = []
for j in judges + [critic]:
    lats = []
    if j in judges:
        for s in scenarios:
            lats.append(d["per_scenario"][s]["per_judge"].get(j, {}).get("latency_s", 0))
    else:
        for s in scenarios:
            lats.append(d["per_scenario"][s]["critic"].get("latency_s", 0))
    data.append(lats)
    labels.append(j)
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("latency (s)")
ax.set_title("R4 V2: Latency per judge (incl critic)")
ax.grid(alpha=0.3, axis="y")
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS / "r4v2_latency.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved r4v2_latency.png")

# ============================================================
# 5. Escalation distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
esc = d["escalation_distribution"]
order = ["C_SUITE_IMMEDIATE", "C_SUITE_REVIEW", "OPS_DIRECTOR_4H", "OPS_DIRECTOR_24H",
         "REGIONAL_MANAGER", "FYI_DASHBOARD"]
vals = [esc.get(o, 0) for o in order]
colors_esc = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#66c2a5"]
ax.barh(order, vals, color=colors_esc)
for i, v in enumerate(vals): ax.text(v + 0.05, i, str(v), va="center", fontsize=10)
ax.set_xlabel("# scenarios")
ax.set_title(f"R4 V2: Escalation routing across {d['n_scenarios']} scenarios")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS / "r4v2_escalation.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved r4v2_escalation.png")

# ============================================================
# 6. Markdown report
# ============================================================
md = []
md.append("# R4 Dangerous V2 — BEAST Mode Results\n")
md.append(f"- **Scenarios**: {d['n_scenarios']} real Wikipedia crisis articles")
md.append(f"- **Judges**: {', '.join(judges)}")
md.append(f"- **Critic**: {critic}")
md.append(f"- **Extractor (for DeepSeek 2-pass)**: {d['extractor']}")
md.append(f"- **Total runtime**: {d['summary']['total_elapsed_min']:.1f} min\n")
md.append("## Agreement\n")
md.append(f"- Krippendorff α (ordinal): **{d['agreement']['krippendorff_alpha_ordinal']:.3f}**")
md.append(f"- Fleiss κ (nominal): **{d['agreement']['fleiss_kappa_nominal']}**")
md.append("- Pairwise weighted κ:")
for k, v in d['agreement']['pairwise_cohen_weighted_kappa'].items():
    md.append(f"  - {k}: {v:.3f}")
md.append("\n## Accuracy vs Ground Truth\n")
md.append("| Judge | Correct / Total | Accuracy |")
md.append("|-------|-----------------|----------|")
for j in judges + ["majority_vote"]:
    a = d["accuracy_vs_ground_truth"].get(j, {})
    md.append(f"| {j} | {a.get('correct',0)} / {a.get('total',0)} | {a.get('accuracy',0):.3f} |")
md.append("\n## Calibration (ECE)\n")
for j in judges:
    e = d["calibration_ece"].get(j, {})
    md.append(f"- {j}: ECE = **{e.get('ece',0):.4f}** (n={e.get('n_predictions',0)})")
md.append("\n## Semantic Agreement (mxbai-embed-large-v1 cosine > 0.65)\n")
md.append(f"- Vulnerabilities: mean Jaccard = **{d['summary']['mean_vulnerabilities_semantic_jaccard']:.3f}**")
md.append(f"- Mitigations: mean Jaccard = **{d['summary']['mean_mitigations_semantic_jaccard']:.3f}**")
md.append("\n## Parse Success + Latency\n")
for j in judges:
    md.append(f"- {j}: {d['summary']['parse_success_rate_per_judge'][j]*100:.0f}% parse OK, "
              f"{d['summary']['mean_latency_s_per_judge'][j]:.1f}s avg")
md.append(f"- Critic ({critic}): {d['summary']['critic_success_rate']*100:.0f}% parse OK")
md.append("\n## Escalation Distribution\n")
for k, v in d["escalation_distribution"].items():
    md.append(f"- {k}: {v}")
md.append("\n## Per-scenario detail\n")
md.append("| Scenario | GT | Majority | α | Escal. |")
md.append("|----------|----|----------|----|--------|")
for s in scenarios:
    sc = d["per_scenario"][s]
    a = sc.get("scenario_ordinal_alpha")
    a_s = f"{a:.2f}" if isinstance(a, (int, float)) and not (isinstance(a, float) and np.isnan(a)) else "n/a"
    md.append(f"| {s} | {sc.get('ground_truth','?')} | {sc.get('risk_majority','?')} | {a_s} | {sc.get('escalation','?')} |")

out_md = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2_REPORT.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"saved {out_md}")
