"""
evaluator.py — Single-metric accept/reject decision.

metric = bootstrap_ci95_lower(grader_scores_across(3 tasks x 3 seeds))

Accept if new_ci95_lower > best_ci95_lower + eps, else reject.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(__file__).resolve().parent
STATE_PATH = AUTORESEARCH_DIR / "state.json"

ACCEPT_EPSILON = 0.005  # program.md convention
BOOTSTRAP_N = 1000
RNG = np.random.default_rng(12345)


@dataclass
class MetricEval:
    mean: float
    std: float
    ci95_lower: float
    ci95_upper: float
    n: int

    def to_json(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci95_lower": round(self.ci95_lower, 4),
            "ci95_upper": round(self.ci95_upper, 4),
            "n": self.n,
        }


def bootstrap_ci95_lower(scores: list[float], n_boot: int = BOOTSTRAP_N) -> MetricEval:
    """Compute bootstrap CI95 lower bound as the metric.

    Args:
        scores: array-like of grader scores in [0, 1].
        n_boot: number of bootstrap resamples.
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return MetricEval(mean=0.0, std=0.0, ci95_lower=0.0, ci95_upper=0.0, n=0)

    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(arr, size=n, replace=True)
        means[i] = sample.mean()

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return MetricEval(mean=mean, std=std, ci95_lower=lo, ci95_upper=hi, n=n)


@dataclass
class Decision:
    accept: bool
    reason: str
    metric_new: MetricEval
    metric_best: Optional[MetricEval]
    delta: float

    def to_json(self) -> dict:
        return {
            "accept": self.accept,
            "reason": self.reason,
            "metric_new": self.metric_new.to_json(),
            "metric_best": self.metric_best.to_json() if self.metric_best else None,
            "delta_ci95_lower": round(self.delta, 4),
        }


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"best": None, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def decide(
    new_scores: list[float],
    new_name: str,
    status: str = "ok",
) -> Decision:
    """Compare new experiment to current best. Return Decision."""
    state = _load_state()
    best = state.get("best")

    if status != "ok" or not new_scores:
        # Any non-ok status = automatic reject, but log in history for provenance
        return Decision(
            accept=False,
            reason=f"status={status}; no valid scores",
            metric_new=MetricEval(0.0, 0.0, 0.0, 0.0, 0),
            metric_best=(MetricEval(**best["metric"]) if best else None),
            delta=-1.0,
        )

    new_metric = bootstrap_ci95_lower(new_scores)

    if best is None:
        # First successful experiment becomes the baseline.
        return Decision(
            accept=True,
            reason="first accepted experiment — seeding baseline",
            metric_new=new_metric,
            metric_best=None,
            delta=new_metric.ci95_lower,
        )

    best_metric = MetricEval(**{k: best["metric"][k] for k in ("mean", "std", "ci95_lower", "ci95_upper", "n")})
    delta = new_metric.ci95_lower - best_metric.ci95_lower

    if delta > ACCEPT_EPSILON:
        return Decision(
            accept=True,
            reason=f"CI95 lower +{delta:.4f} > {ACCEPT_EPSILON:.4f} threshold",
            metric_new=new_metric,
            metric_best=best_metric,
            delta=delta,
        )
    return Decision(
        accept=False,
        reason=f"CI95 lower delta {delta:+.4f} <= {ACCEPT_EPSILON:.4f} threshold",
        metric_new=new_metric,
        metric_best=best_metric,
        delta=delta,
    )


def commit(
    experiment_name: str,
    hypothesis: dict,
    scores: list[float],
    decision: Decision,
    wall_clock_s: float,
    architecture: str,
    checkpoint_path: str,
    stdout_path: str,
) -> None:
    """Append the experiment to state.history and update best if accepted."""
    state = _load_state()

    entry = {
        "experiment_name": experiment_name,
        "hypothesis": hypothesis,
        "grader_scores": scores,
        "metric": decision.metric_new.to_json() if decision.metric_new.n > 0 else None,
        "accepted": decision.accept,
        "reason": decision.reason,
        "delta_ci95_lower": decision.delta,
        "metric_ci95_lower": decision.metric_new.ci95_lower,
        "metric_mean": decision.metric_new.mean,
        "architecture_summary": architecture,
        "wall_clock_s": wall_clock_s,
        "stdout_path": stdout_path,
        "checkpoint_path": checkpoint_path,
        "status": "accepted" if decision.accept else "rejected",
    }

    state["history"].append(entry)

    if decision.accept:
        state["best"] = {
            "experiment_name": experiment_name,
            "metric": decision.metric_new.to_json(),
            "architecture_summary": architecture,
            "checkpoint_path": checkpoint_path,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        logger.info("[commit] accepted %s -> new best ci95_lower=%.4f",
                    experiment_name, decision.metric_new.ci95_lower)
    else:
        logger.info("[commit] rejected %s (%s)", experiment_name, decision.reason)

    _save_state(state)


# Time import for commit()
import time  # noqa: E402


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", type=float, required=True,
                        help="9 grader scores (3 tasks x 3 seeds)")
    parser.add_argument("--name", default="manual_decide")
    args = parser.parse_args()

    d = decide(args.scores, args.name)
    print(json.dumps(d.to_json(), indent=2))
