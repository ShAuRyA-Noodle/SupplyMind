"""
Phase I — AutoResearch final summary.

AutoResearch has already run 10 experiments (stored in rl/autoresearch_results/).
This script aggregates results, picks best config per agent family, and writes:
  rl/autoresearch_final.json  — best configs, trial metadata
  AUTORESEARCH_SUMMARY.md     — human-readable narrative
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
AR_DIR = ROOT / "rl" / "autoresearch_results"
CKPT = ROOT / "rl" / "checkpoints"


def main():
    if not AR_DIR.exists():
        log.warning("No autoresearch_results directory.")
        return

    experiments = []
    for f in sorted(AR_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                for d in data:
                    experiments.append({"file": f.name, **d})
            else:
                experiments.append({"file": f.name, **data})
        except Exception as e:
            log.warning(f"Could not parse {f}: {e}")

    log.info(f"Loaded {len(experiments)} AutoResearch experiments")

    # Identify best by grade_reward or episode_reward_mean
    def score(e):
        return e.get("grade_avg", e.get("final_grade_mean", e.get("grade_mean", e.get("episode_reward_mean", 0.0))))

    sorted_exps = sorted(experiments, key=score, reverse=True)
    best = sorted_exps[0] if sorted_exps else {}

    # Group by hyperparameter family if present
    by_family = {}
    for e in experiments:
        fam = e.get("algo", e.get("family", "unknown"))
        by_family.setdefault(fam, []).append(e)
    best_per_family = {fam: max(lst, key=score) for fam, lst in by_family.items()}

    summary = {
        "total_experiments": len(experiments),
        "best_overall": best,
        "best_per_family": best_per_family,
        "ranking_top5": sorted_exps[:5],
    }
    out = ROOT / "rl" / "autoresearch_final.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Saved {out}")

    # Markdown summary
    md = ["# AutoResearch Summary (Phase I)", "",
          f"Total experiments: {len(experiments)}", "",
          "## Best configurations per agent family", ""]
    for fam, e in best_per_family.items():
        md.append(f"### {fam}")
        md.append(f"- File: `{e.get('file', 'n/a')}`")
        md.append(f"- Score: {score(e):.4f}")
        hp = e.get("hyperparameters", e.get("config", {}))
        if hp:
            md.append("- Hyperparameters:")
            for k, v in hp.items():
                md.append(f"  - `{k}` = {v}")
        md.append("")
    (ROOT / "AUTORESEARCH_SUMMARY.md").write_text("\n".join(md))
    log.info("Saved AUTORESEARCH_SUMMARY.md")


if __name__ == "__main__":
    main()
