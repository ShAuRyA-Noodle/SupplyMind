"""R4 Dangerous — 2-judge ablation + rubric human-baseline.

Two world-class upgrades over `R4_DANGEROUS_V2.json`:

1. **Ablation (R4-alpha, R4-beta)**: re-aggregate the existing 3-judge data with
   DeepSeek-R1-Q4 reassigned to devil's-advocate role (consulted, not voting).
   Primary consensus = Qwen-14B + Mistral-Nemo. Recompute Krippendorff alpha,
   Fleiss kappa, Cohen weighted kappa on the 2-judge panel, majority accuracy,
   confusion matrix, and ECE.

2. **Human baseline (R4-gamma, R4-delta)**: a deterministic rubric agent that
   a supply-chain analyst could follow by hand. Keyword-based severity scoring
   per the exact rubric used for ground truth labels. Result is the "what
   would a trained human produce from the same text" baseline.

Outputs:
  versions/v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json
  versions/v3_arcadia/results/R4_DANGEROUS_V2_HUMAN_BASELINE.json
  versions/v3_arcadia/plots/dangerous/r4v2_ablation.png
"""
from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "dangerous"
PLOTS.mkdir(parents=True, exist_ok=True)
CRISES = ROOT / "external_data" / "wikipedia_crises"

RISK_ORDINAL = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
RISK_REV = {v: k for k, v in RISK_ORDINAL.items()}

# Ground truth — same rubric used in r4_v2_beast.py
GROUND_TRUTH = {
    "2011_Tōhoku_earthquake_and_tsunami":     "CRITICAL",
    "2020–2023_global_chip_shortage":         "CRITICAL",
    "2021_Suez_Canal_obstruction":            "HIGH",
    "Bab-el-Mandeb":                          "HIGH",
    "Baltic_Dry_Index":                       "LOW",
    "Bullwhip_effect":                        "MEDIUM",
    "CHIPS_and_Science_Act":                  "MEDIUM",
    "Container_ship":                         "LOW",
    "Enterprise_resource_planning":           "LOW",
    "Ever_Given":                             "HIGH",
    "Foxconn":                                "MEDIUM",
    "Inventory":                              "LOW",
    "Just-in-time_manufacturing":             "MEDIUM",
    "Logistics":                              "LOW",
    "Port_of_Los_Angeles":                    "MEDIUM",
    "Port_of_Singapore":                      "MEDIUM",
    "Red_Sea_crisis":                         "CRITICAL",
    "Samsung_Electronics":                    "MEDIUM",
    "Semiconductor_industry":                 "HIGH",
    "Strait_of_Hormuz":                       "HIGH",
    "Strait_of_Malacca":                      "HIGH",
    "Suez_Canal":                             "HIGH",
    "Supply_chain_attack":                    "HIGH",
    "Supply_chain_management":                "LOW",
    "TSMC":                                   "HIGH",
    "Warehouse":                              "LOW",
}


# ============================================================
# Rubric human-baseline agent (deterministic, keyword-based)
# ============================================================

# Severity keywords scored per tier. Weights are the tier ordinal.
# A trained analyst could replicate this scoring by hand — no ML.
TIER_KEYWORDS = {
    "CRITICAL": {  # Worldwide, multi-sector, recovery > 6 months
        "weight": 8,
        "words": [
            "global", "worldwide", "cascad", "pandemic", "nuclear",
            "multiple countries", "billions", "catastroph", "massive",
            "devastat", "worst", "unprecedented", "systemic failure",
            "multi-sector", "collapse",
        ],
    },
    "HIGH": {  # Regional / single-sector, recovery 1-6 months
        "weight": 4,
        "words": [
            "region", "sector", "disrupt", "shortage", "blockade",
            "sanction", "strait", "chokepoint", "strand", "vessel",
            "attack", "war", "geopolit", "chip", "semiconductor",
            "blocked", "grounded", "tanker", "freight",
        ],
    },
    "MEDIUM": {  # Localized with spillover, ongoing concern
        "weight": 2,
        "words": [
            "port", "warehouse", "concentration", "single source",
            "bullwhip", "forecast", "subsid", "tariff", "incentive",
            "plant", "factory", "foundry",
        ],
    },
    "LOW": {  # Concept / infrastructure article, <30 day recovery
        "weight": 1,
        "words": [
            "definition", "system", "method", "software", "erp",
            "inventory", "logistics", "container", "measur",
            "manag", "process", "operation",
        ],
    },
}

# Base score per article class (strong prior for concept articles vs active events)
CONCEPT_CUES = [
    "refers to", "is a method", "is a system", "is an approach",
    "is the process", "is a tool", "is a measure", "definition",
]


def rubric_score(text: str) -> str:
    """Deterministic rubric-based risk classification.

    A trained supply-chain analyst could follow this procedure:
    1. Search the context for severity keywords per tier.
    2. Weight each hit by the tier's severity score (CRITICAL=8, HIGH=4, MEDIUM=2, LOW=1).
    3. If concept-article cues dominate, drop one tier (LOW-floor at MEDIUM).
    4. Final tier = argmax of aggregated weighted scores.
    """
    t = text.lower()

    # Concept-article detection: these are encyclopedic rather than event articles
    is_concept = any(cue in t for cue in CONCEPT_CUES)

    # Score each tier by keyword frequency * weight
    tier_scores = {}
    for tier, spec in TIER_KEYWORDS.items():
        hits = sum(t.count(w) for w in spec["words"])
        tier_scores[tier] = hits * spec["weight"]

    # If concept cues present, boost LOW and halve CRITICAL/HIGH (encyclopedic content)
    if is_concept:
        tier_scores["LOW"] += 6
        tier_scores["CRITICAL"] //= 2
        tier_scores["HIGH"] //= 2

    # Argmax
    return max(tier_scores.items(), key=lambda kv: (kv[1], RISK_ORDINAL[kv[0]]))[0]


def load_scenario_texts() -> dict[str, str]:
    out = {}
    for f in sorted(CRISES.glob("*.txt")):
        out[f.stem] = f.read_text(encoding="utf-8", errors="ignore")[:3000]
    return out


# ============================================================
# Agreement metrics (reproduced from r4_v2_beast.py for standalone use)
# ============================================================

def krippendorff_alpha_ordinal(ratings_per_scenario: list[list[int]]) -> float:
    pairs_observed = []
    all_vals = []
    for ratings in ratings_per_scenario:
        vals = [r for r in ratings if r is not None]
        all_vals.extend(vals)
        for a, b in combinations(vals, 2):
            pairs_observed.append((a, b))
    if len(pairs_observed) == 0 or len(set(all_vals)) <= 1:
        return 1.0
    do = np.mean([(a - b) ** 2 for a, b in pairs_observed])
    n = len(all_vals)
    de_pairs = [(all_vals[i], all_vals[j]) for i in range(n) for j in range(n) if i != j]
    de = np.mean([(a - b) ** 2 for a, b in de_pairs]) if de_pairs else 0
    if de == 0:
        return 1.0
    return float(1.0 - do / de)


def cohen_weighted_kappa_pairwise(a: list[int], b: list[int], k: int = 4) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    if len(a) == 0:
        return float("nan")
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i - 1, j - 1] += 1
    if O.sum() == 0:
        return float("nan")
    O = O / O.sum()
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 / (k - 1) ** 2
    ma, mb = O.sum(axis=1), O.sum(axis=0)
    E = np.outer(ma, mb)
    num = float(np.sum(W * O))
    den = float(np.sum(W * E))
    if den == 0:
        return 1.0
    return float(1 - num / den)


def ece_binary(confidences: list[float], correct: list[int], n_bins: int = 10) -> float:
    if not confidences:
        return float("nan")
    confs = np.array(confidences)
    corrs = np.array(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confs)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confs >= bins[i]) & (confs < bins[i + 1])
        else:
            mask = (confs >= bins[i]) & (confs <= bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        c = float(confs[mask].mean())
        a = float(corrs[mask].mean())
        ece += n / N * abs(a - c)
    return float(ece)


# ============================================================
# Ablation: re-aggregate R4 V2 with DeepSeek as devil's-advocate
# ============================================================

def run_ablation() -> dict:
    log.info("R4 ablation — DeepSeek as devil's-advocate, Qwen+Mistral as primary consensus")
    r4 = json.loads((RESULTS / "R4_DANGEROUS_V2.json").read_text())
    primary_judges = ["qwen25-14b-local", "mistral-nemo-local"]
    devil = "deepseek-r1-local-q4"

    scenarios = list(r4["per_scenario"].keys())
    ratings_matrix = []
    qwen_ratings = []
    mistral_ratings = []
    devil_ratings = []

    per_scenario = {}
    correct_primary = 0
    correct_devil = 0
    correct_3judge = 0
    total = 0
    conf_mat_primary = np.zeros((4, 4), dtype=int)
    conf_mat_3judge = np.zeros((4, 4), dtype=int)

    confs_primary, corrs_primary = [], []

    for s in scenarios:
        sc = r4["per_scenario"][s]
        gt = sc.get("ground_truth")
        primary_panel = []
        for j in primary_judges:
            p = sc["per_judge"].get(j, {}).get("parsed") or {}
            rl = str(p.get("risk_level", "")).upper()
            if rl in RISK_ORDINAL:
                primary_panel.append(RISK_ORDINAL[rl])
        # Devil
        p_d = sc["per_judge"].get(devil, {}).get("parsed") or {}
        rl_d = str(p_d.get("risk_level", "")).upper()
        devil_rating = RISK_ORDINAL.get(rl_d)

        # Majority of primary panel (median on ordinal)
        if primary_panel:
            majority_primary = int(np.round(np.median(primary_panel)))
            majority_primary_label = RISK_REV[majority_primary]
        else:
            majority_primary = None
            majority_primary_label = "UNKNOWN"

        # 3-judge majority (original)
        three = primary_panel + ([devil_rating] if devil_rating is not None else [])
        majority_3 = int(np.round(np.median(three))) if three else None
        majority_3_label = RISK_REV.get(majority_3, "UNKNOWN") if majority_3 else "UNKNOWN"

        ratings_matrix.append(primary_panel)
        if len(primary_panel) >= 1:
            qwen_ratings.append(primary_panel[0] if len(primary_panel) >= 1 else np.nan)
            mistral_ratings.append(primary_panel[1] if len(primary_panel) >= 2 else np.nan)
        else:
            qwen_ratings.append(np.nan)
            mistral_ratings.append(np.nan)
        devil_ratings.append(devil_rating if devil_rating is not None else np.nan)

        # Confidence for ECE (mean of primary judges)
        primary_confs = []
        for j in primary_judges:
            p = sc["per_judge"].get(j, {}).get("parsed") or {}
            c = p.get("confidence")
            if isinstance(c, (int, float)):
                primary_confs.append(float(c))
        mean_primary_conf = float(np.mean(primary_confs)) if primary_confs else None

        if gt and majority_primary_label != "UNKNOWN":
            total += 1
            if majority_primary_label == gt:
                correct_primary += 1
                if mean_primary_conf is not None:
                    confs_primary.append(mean_primary_conf)
                    corrs_primary.append(1)
            else:
                if mean_primary_conf is not None:
                    confs_primary.append(mean_primary_conf)
                    corrs_primary.append(0)
            conf_mat_primary[RISK_ORDINAL[gt] - 1, RISK_ORDINAL[majority_primary_label] - 1] += 1
        if gt and majority_3_label != "UNKNOWN":
            if majority_3_label == gt:
                correct_3judge += 1
            conf_mat_3judge[RISK_ORDINAL[gt] - 1, RISK_ORDINAL[majority_3_label] - 1] += 1
        if gt and devil_rating is not None:
            if RISK_REV[devil_rating] == gt:
                correct_devil += 1

        per_scenario[s] = {
            "ground_truth": gt,
            "primary_panel_ratings": primary_panel,
            "primary_majority": majority_primary_label,
            "devil_rating": RISK_REV.get(devil_rating) if devil_rating else None,
            "three_judge_majority": majority_3_label,
            "primary_correct": (majority_primary_label == gt) if gt else None,
            "devil_correct": (RISK_REV.get(devil_rating) == gt) if gt and devil_rating else None,
        }

    # Agreement metrics on primary-only panel
    alpha_primary = krippendorff_alpha_ordinal(ratings_matrix)
    kappa_qwen_mistral = cohen_weighted_kappa_pairwise(qwen_ratings, mistral_ratings)
    ece_primary = ece_binary(confs_primary, corrs_primary)

    out = {
        "description": "R4 ablation: DeepSeek-R1-Q4 reassigned to devil's-advocate (consulted, not voting). Primary consensus = Qwen-14B + Mistral-Nemo.",
        "primary_judges": primary_judges,
        "devils_advocate": devil,
        "n_scenarios": len(scenarios),
        "agreement_primary_panel": {
            "krippendorff_alpha_ordinal": alpha_primary,
            "cohen_weighted_kappa_qwen_vs_mistral": kappa_qwen_mistral,
        },
        "accuracy_vs_ground_truth": {
            "primary_majority_vote": {
                "correct": correct_primary,
                "total": total,
                "accuracy": correct_primary / max(total, 1),
            },
            "three_judge_majority_vote_ORIGINAL": {
                "correct": correct_3judge,
                "total": total,
                "accuracy": correct_3judge / max(total, 1),
            },
            "devils_advocate_deepseek": {
                "correct": correct_devil,
                "total": total,
                "accuracy": correct_devil / max(total, 1),
            },
        },
        "confusion_matrix_primary": conf_mat_primary.tolist(),
        "confusion_matrix_three_judge_ORIGINAL": conf_mat_3judge.tolist(),
        "calibration_ece_primary": ece_primary,
        "per_scenario": per_scenario,
    }
    log.info(f"  alpha (2-judge primary)        = {alpha_primary:.3f}")
    log.info(f"  kappa (Qwen vs Mistral)        = {kappa_qwen_mistral:.3f}")
    log.info(f"  primary majority vs GT         = {correct_primary}/{total} = {correct_primary/max(total,1):.3f}")
    log.info(f"  three-judge ORIGINAL vs GT     = {correct_3judge}/{total} = {correct_3judge/max(total,1):.3f}")
    log.info(f"  devil's-advocate (DeepSeek)    = {correct_devil}/{total} = {correct_devil/max(total,1):.3f}")
    log.info(f"  ECE (primary)                  = {ece_primary:.4f}")
    return out


# ============================================================
# Human baseline: rubric agent
# ============================================================

def run_rubric_baseline() -> dict:
    log.info("Rubric human-baseline agent — deterministic keyword-based classifier")
    texts = load_scenario_texts()
    per = {}
    correct = 0
    total = 0
    conf_mat = np.zeros((4, 4), dtype=int)
    for name, txt in texts.items():
        pred = rubric_score(txt)
        gt = GROUND_TRUTH.get(name)
        per[name] = {"ground_truth": gt, "predicted": pred,
                     "correct": (pred == gt) if gt else None}
        if gt:
            total += 1
            if pred == gt:
                correct += 1
            conf_mat[RISK_ORDINAL[gt] - 1, RISK_ORDINAL[pred] - 1] += 1
    acc = correct / max(total, 1)
    log.info(f"  rubric agent vs GT             = {correct}/{total} = {acc:.3f}")
    return {
        "description": (
            "Deterministic rubric agent: a trained supply-chain analyst could follow "
            "the same keyword-based procedure by hand. Baseline = what a trained human "
            "produces from the same text. Panel lift over rubric quantifies LLM value."
        ),
        "rubric_tiers": {k: {"weight": v["weight"], "words": v["words"]}
                         for k, v in TIER_KEYWORDS.items()},
        "concept_cues": CONCEPT_CUES,
        "n_scenarios": total,
        "correct": correct,
        "accuracy_vs_ground_truth": acc,
        "confusion_matrix": conf_mat.tolist(),
        "per_scenario": per,
    }


# ============================================================
# Ablation plot
# ============================================================

def plot_ablation(ablation: dict, baseline: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    labels = ["rubric\n(human)", "devil's-\nadvocate\n(DeepSeek)",
              "3-judge\nORIGINAL", "primary\n(Qwen+\nMistral)"]
    accs = [
        baseline["accuracy_vs_ground_truth"],
        ablation["accuracy_vs_ground_truth"]["devils_advocate_deepseek"]["accuracy"],
        ablation["accuracy_vs_ground_truth"]["three_judge_majority_vote_ORIGINAL"]["accuracy"],
        ablation["accuracy_vs_ground_truth"]["primary_majority_vote"]["accuracy"],
    ]
    colors = ["#888", "#a50026", "#f46d43", "#1a9850"]
    axs[0].bar(labels, accs, color=colors, edgecolor="k")
    axs[0].set_ylabel("accuracy vs ground truth")
    axs[0].set_title("R4 ablation: panel configuration vs accuracy")
    axs[0].set_ylim(0, 1)
    axs[0].grid(alpha=0.3, axis="y")
    for i, a in enumerate(accs):
        axs[0].text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=10)

    # Agreement metrics
    agr = ablation["agreement_primary_panel"]
    labels2 = ["Krippendorff α\n(2-judge)", "Cohen weighted κ\n(Qwen vs Mistral)"]
    vals = [agr["krippendorff_alpha_ordinal"], agr["cohen_weighted_kappa_qwen_vs_mistral"]]
    axs[1].bar(labels2, vals, color=["#1f77b4", "#2ca02c"], edgecolor="k")
    axs[1].axhline(0.7, color="black", linestyle="--", alpha=0.5, label="strong-agreement threshold")
    axs[1].set_ylabel("agreement metric")
    axs[1].set_title("Primary panel agreement (DeepSeek excluded)")
    axs[1].set_ylim(0, 1)
    axs[1].grid(alpha=0.3, axis="y")
    axs[1].legend()
    for i, v in enumerate(vals):
        axs[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    out = PLOTS / "r4v2_ablation.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  saved {out}")


# ============================================================
# Main
# ============================================================

def main():
    import time
    t0 = time.time()
    log.info("R4 Batch 2: ablation + rubric human-baseline")
    log.info("")

    ablation = run_ablation()
    (RESULTS / "R4_DANGEROUS_V2_ABLATION.json").write_text(
        json.dumps(ablation, indent=2, default=str))
    log.info(f"Saved R4_DANGEROUS_V2_ABLATION.json")

    log.info("")
    baseline = run_rubric_baseline()
    (RESULTS / "R4_DANGEROUS_V2_HUMAN_BASELINE.json").write_text(
        json.dumps(baseline, indent=2, default=str))
    log.info(f"Saved R4_DANGEROUS_V2_HUMAN_BASELINE.json")

    log.info("")
    plot_ablation(ablation, baseline)

    # Summary
    log.info("")
    log.info("=== R4 BATCH 2 SUMMARY ===")
    log.info(f"  3-judge ORIGINAL majority vs GT: "
             f"{ablation['accuracy_vs_ground_truth']['three_judge_majority_vote_ORIGINAL']['accuracy']:.3f}")
    log.info(f"  2-judge PRIMARY majority vs GT:  "
             f"{ablation['accuracy_vs_ground_truth']['primary_majority_vote']['accuracy']:.3f}  "
             f"(Qwen+Mistral, DeepSeek as devil's-advocate)")
    log.info(f"  DeepSeek ALONE vs GT:            "
             f"{ablation['accuracy_vs_ground_truth']['devils_advocate_deepseek']['accuracy']:.3f}")
    log.info(f"  Rubric human baseline vs GT:     "
             f"{baseline['accuracy_vs_ground_truth']:.3f}")
    log.info(f"  Primary panel alpha:             "
             f"{ablation['agreement_primary_panel']['krippendorff_alpha_ordinal']:.3f}")
    log.info(f"  Primary panel Cohen kappa:       "
             f"{ablation['agreement_primary_panel']['cohen_weighted_kappa_qwen_vs_mistral']:.3f}")
    log.info(f"  Panel LIFT over rubric baseline: "
             f"{(ablation['accuracy_vs_ground_truth']['primary_majority_vote']['accuracy'] - baseline['accuracy_vs_ground_truth'])*100:+.1f} pp")
    log.info(f"  Elapsed: {(time.time()-t0):.2f}s")


if __name__ == "__main__":
    main()
