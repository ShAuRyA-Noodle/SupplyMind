"""
Ablation study for SupplyMind.

Compares the agents that have a real evaluator here (random baseline vs the
scripted heuristics). Trained-agent configurations (PPO / +CVaR / +DT /
+Ensemble ...) are intentionally omitted: they require loaded checkpoints and
are evaluated by benchmark/run_full_benchmark.py. Including them here would
mean re-scoring the scripted agent under different labels, which is fabricated
component-contribution evidence and is not done.

5 seeds x 20 episodes per configuration.

Usage:
    python -m benchmark.ablation
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Ablation configurations. Only agents with a real evaluator in run_benchmark
# are included; adding trained-agent rows here without loaded checkpoints would
# produce byte-identical scripted scores under different labels.
CONFIGURATIONS = [
    {"name": "Random agent", "agent": "random", "description": "Uniform random action selection"},
    {"name": "Scripted (no ML)", "agent": "scripted", "description": "Hand-crafted heuristics"},
]


def run_ablation(
    seeds: list[int] | None = None,
    n_episodes: int = 20,
) -> Path:
    """Run ablation study across all configurations."""
    if seeds is None:
        seeds = [42, 99, 7, 123, 256]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("  Configurations: %d", len(CONFIGURATIONS))
    logger.info("  Seeds: %s | Episodes: %d", seeds, n_episodes)
    logger.info("=" * 60)

    from benchmark.run_benchmark import evaluate_agent

    task_ids = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    task_short = {"easy_typhoon_response": "Easy", "medium_multi_front": "Medium", "hard_cascading_crisis": "Hard"}

    all_results = []
    start = time.time()

    for config in CONFIGURATIONS:
        config_scores = {"Easy": [], "Medium": [], "Hard": []}

        for task_id in task_ids:
            task_scores = []
            for seed in seeds:
                scores = evaluate_agent(config["agent"], task_id, seed, n_episodes)
                task_scores.extend(scores)
            config_scores[task_short[task_id]] = task_scores

        # Compute means
        all_task_scores = []
        for task, scores in config_scores.items():
            all_task_scores.extend(scores)

        result = {
            "configuration": config["name"],
            "description": config["description"],
            "easy_mean": np.mean(config_scores["Easy"]),
            "easy_std": np.std(config_scores["Easy"]),
            "medium_mean": np.mean(config_scores["Medium"]),
            "medium_std": np.std(config_scores["Medium"]),
            "hard_mean": np.mean(config_scores["Hard"]),
            "hard_std": np.std(config_scores["Hard"]),
            "avg_mean": np.mean(all_task_scores),
            "avg_std": np.std(all_task_scores),
        }
        all_results.append(result)
        logger.info("  %s: avg=%.4f +/- %.4f", config["name"], result["avg_mean"], result["avg_std"])

    # Save
    output_path = RESULTS_DIR / "ablation_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Ablation complete in %.1f min. Results: %s", elapsed / 60, output_path)
    logger.info("=" * 60)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_ablation()


if __name__ == "__main__":
    main()
