"""
Simulation backtesting against historical crises.

Measures how the environment's simulated disruption metrics (episode duration
in days, peak disruption severity) compare to reference values from a library
of real historical supply-chain crises. This is a plausibility / calibration
check against public reference data -- it is NOT a proof that any task
reproduces a specific crisis, and it will honestly report low credibility when
the simulation diverges from the reference.

Ground truth is loaded from benchmark/crisis_library/*.json (curated public
data). Each crisis is mapped to the closest-matching environment task.

Compute: mean_relative_error = avg(abs(sim - real) / real) over the metrics
that both the simulation and the reference define.

Usage:
    python -m benchmark.backtesting
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CRISIS_DIR = Path(__file__).resolve().parent / "crisis_library"

# The environment advances exactly one simulated day per step (current_day ==
# current_step in server/engine/simulation.py), so duration in days equals the
# number of steps taken.
STEPS_PER_DAY = 1

# Each historical crisis is assigned to the environment task whose disruption
# profile most closely resembles it. This mapping is an approximation, not a
# claim that the task reproduces the crisis.
CRISIS_TASK_MAP = {
    "chip_shortage_2020": "easy_typhoon_response",
    "suez_2021": "medium_multi_front",
    "red_sea_2023": "medium_multi_front",
    "tohoku_2011": "hard_cascading_crisis",
    "ukraine_neon_2022": "hard_cascading_crisis",
}


def _load_historical_crises() -> dict[str, dict[str, Any]]:
    """Load crisis ground truth from benchmark/crisis_library/*.json.

    Only the fields the simulation can actually measure are extracted into
    `ground_truth` (duration in days, peak severity); the rich narrative data
    in the JSON files is left in the library and not fabricated into metrics.
    """
    crises: dict[str, dict[str, Any]] = {}
    for crisis_id, task_id in CRISIS_TASK_MAP.items():
        path = CRISIS_DIR / f"{crisis_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Crisis library file missing: {path}. "
                f"Expected one JSON per crisis in {CRISIS_DIR}."
            )
        data = json.loads(path.read_text())
        meta = data["metadata"]
        gt = data["ground_truth"]
        crises[crisis_id] = {
            "name": meta["name"],
            "source": meta["source"],
            "task_id": task_id,
            "description": meta["description"],
            "ground_truth": {
                "disruption_duration_days": float(gt["duration_days"]),
                "peak_severity": float(gt["peak_severity"]),
            },
        }
    return crises


# Loaded from the crisis library at import time (fail loud if the data is
# missing). Replaces the previously hand-duplicated dict.
HISTORICAL_CRISES = _load_historical_crises()


def simulate_crisis(crisis_id: str, n_runs: int = 50) -> dict[str, list[float]]:
    """Run environment simulation for the task mapped to a historical crisis.

    Returns per-run values for the metrics that are directly comparable to the
    crisis reference data, using the same keys as the ground truth.
    """
    from server.supply_environment import SupplyMindEnvironment

    from scripted_agent import choose_action

    crisis = HISTORICAL_CRISES[crisis_id]
    task_id = crisis["task_id"]

    metrics: dict[str, list[float]] = {
        "disruption_duration_days": [],
        "peak_severity": [],
    }

    env = SupplyMindEnvironment()

    for run in range(n_runs):
        obs = env.reset(task_id=task_id, seed=run)
        step = 0
        peak_severity = max((s.severity for s in obs.active_signals), default=0.0)

        while not obs.done:
            action = choose_action(obs, step)
            obs = env.step(action)
            step += 1
            if obs.active_signals:
                peak_severity = max(peak_severity, max(s.severity for s in obs.active_signals))

        # Duration: number of simulated days (1 step == 1 day).
        metrics["disruption_duration_days"].append(step / STEPS_PER_DAY)
        metrics["peak_severity"].append(peak_severity)

    return metrics


def compute_calibration_error(
    simulated: dict[str, list[float]],
    ground_truth: dict[str, float],
) -> dict[str, Any]:
    """Compute mean relative error between simulation and reference values.

    Only metrics present in BOTH `simulated` and `ground_truth` are compared.
    If no metric can be compared, this returns is_credible=False with a maximal
    (100%) error and a reason -- it never reports a misleading 0.0 error.
    """
    errors = {}
    overall_errors = []

    for metric, gt_value in ground_truth.items():
        if metric in simulated and simulated[metric]:
            sim_mean = float(np.mean(simulated[metric]))
            sim_std = float(np.std(simulated[metric]))
            abs_error = abs(sim_mean - gt_value)
            rel_error = abs_error / max(abs(gt_value), 1e-2)  # Prevent tiny denominators

            errors[metric] = {
                "ground_truth": gt_value,
                "simulated_mean": round(sim_mean, 4),
                "simulated_std": round(sim_std, 4),
                "absolute_error": round(abs_error, 4),
                "relative_error_pct": round(rel_error * 100, 1),
            }
            overall_errors.append(rel_error)

    if not overall_errors:
        return {
            "mean_relative_error_pct": 100.0,
            "per_metric": {},
            "n_metrics": 0,
            "is_credible": False,
            "reason": "no metric was comparable between simulation and reference data",
        }

    mean_rel_error = float(np.mean(overall_errors))
    return {
        "mean_relative_error_pct": round(mean_rel_error * 100, 1),
        "per_metric": errors,
        "n_metrics": len(errors),
        "is_credible": 10 <= mean_rel_error * 100 <= 40,
    }


def run_backtesting(n_runs: int = 50) -> Path:
    """Run backtesting against all historical crises in the library."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SIMULATION BACKTESTING")
    logger.info("  Crises: %d | Runs per crisis: %d", len(HISTORICAL_CRISES), n_runs)
    logger.info("=" * 60)

    all_results = {}
    start = time.time()

    for crisis_id, crisis in HISTORICAL_CRISES.items():
        logger.info("  Simulating: %s...", crisis["name"])
        sim_metrics = simulate_crisis(crisis_id, n_runs)
        calibration = compute_calibration_error(sim_metrics, crisis["ground_truth"])

        all_results[crisis_id] = {
            "name": crisis["name"],
            "source": crisis["source"],
            "task_id": crisis["task_id"],
            "calibration": calibration,
        }

        logger.info("    Metrics compared: %d | mean relative error: %.1f%% (%s)",
                     calibration["n_metrics"],
                     calibration["mean_relative_error_pct"],
                     "CREDIBLE" if calibration["is_credible"] else "NOT CREDIBLE")

    # Save
    output_path = RESULTS_DIR / "backtesting_results.json"
    output_path.write_text(json.dumps(all_results, indent=2))

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Backtesting done in %.1f min. Results: %s", elapsed / 60, output_path)
    logger.info("=" * 60)

    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_backtesting()


if __name__ == "__main__":
    main()
