"""
Statistical significance tests for SupplyMind benchmark.

- Wilcoxon signed-rank (pairwise, one-sided): "Agent A > Agent B?"
- Friedman test (multi-agent): "Any agent significantly different?"
- Bootstrap confidence intervals (n=1000)
- Effect size calculation

Every result in README gets a p-value footnote.

Usage:
    python -m benchmark.statistics
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_benchmark_scores() -> dict[str, dict[str, list[float]]]:
    """Load benchmark CSV into {agent: {task: [scores]}} structure."""
    path = RESULTS_DIR / "benchmark_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark results CSV not found at {path}.\n"
            f"Generate it first by running the full benchmark: "
            f"`python benchmark/run_full_benchmark.py` "
            f"(or `python -m benchmark.run_benchmark` for the random/scripted subset)."
        )

    results: dict[str, dict[str, list[float]]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row["agent"]
            task = row["task"]
            score = float(row["score"])
            results.setdefault(agent, {}).setdefault(task, []).append(score)
    return results


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    alternative: str = "greater",
) -> dict[str, Any]:
    """Wilcoxon signed-rank test: is A > B?"""
    from scipy.stats import wilcoxon

    n = min(len(scores_a), len(scores_b))
    a = np.array(scores_a[:n])
    b = np.array(scores_b[:n])

    # Handle identical arrays (Wilcoxon is undefined when all diffs are zero)
    diff = a - b
    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "z_statistic": 0.0,
            "n_pairs": n,
            "n_nonzero_pairs": 0,
            "mean_diff": 0.0,
        }

    # method="approx" gives the normal-approximation z-statistic directly, which
    # is what the matched-pairs rank-biserial effect size r = Z / sqrt(N) needs.
    res = wilcoxon(a, b, alternative=alternative, method="approx")
    stat = float(res.statistic)
    p = float(res.pvalue)
    z = float(res.zstatistic)
    # r = Z / sqrt(N), N = number of non-zero-difference pairs entering the test.
    effect_size = z / np.sqrt(n_nonzero)

    return {
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05,
        "effect_size": round(float(effect_size), 3),
        "z_statistic": round(z, 4),
        "n_pairs": n,
        "n_nonzero_pairs": n_nonzero,
        "mean_diff": round(float(np.mean(diff)), 4),
    }


def friedman_test(all_scores: dict[str, list[float]]) -> dict[str, Any]:
    """Friedman test: any agent significantly different?"""
    from scipy.stats import friedmanchisquare

    agents = list(all_scores.keys())
    if len(agents) < 3:
        return {"error": "Need at least 3 agents for Friedman test"}

    # Align to same length
    min_n = min(len(v) for v in all_scores.values())
    arrays = [np.array(all_scores[a][:min_n]) for a in agents]

    stat, p = friedmanchisquare(*arrays)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
        "agents": agents,
        "n_per_agent": min_n,
    }


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence intervals."""
    rng = np.random.default_rng(42)
    arr = np.array(scores)
    n = len(arr)

    boot_means = [float(np.mean(rng.choice(arr, n, replace=True))) for _ in range(n_bootstrap)]

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def run_all_tests() -> dict[str, Any]:
    """Run all statistical tests on benchmark results."""
    data = load_benchmark_scores()
    agents = list(data.keys())

    results: dict[str, Any] = {"pairwise": {}, "friedman": {}, "bootstrap": {}}

    # Bootstrap CIs for each agent
    for agent in agents:
        all_scores = []
        for task_scores in data[agent].values():
            all_scores.extend(task_scores)
        results["bootstrap"][agent] = bootstrap_ci(all_scores)

    # Pairwise Wilcoxon: each agent vs scripted baseline
    if "scripted" in data:
        scripted_scores = []
        for task_scores in data["scripted"].values():
            scripted_scores.extend(task_scores)

        for agent in agents:
            if agent == "scripted":
                continue
            agent_scores = []
            for task_scores in data[agent].values():
                agent_scores.extend(task_scores)
            results["pairwise"][f"{agent}_vs_scripted"] = wilcoxon_test(agent_scores, scripted_scores)

    # Friedman across all agents
    all_agent_scores = {}
    for agent in agents:
        scores = []
        for task_scores in data[agent].values():
            scores.extend(task_scores)
        all_agent_scores[agent] = scores
    results["friedman"] = friedman_test(all_agent_scores)

    # Save
    import json
    output_path = RESULTS_DIR / "statistical_tests.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    logger.info("Statistical tests saved to %s", output_path)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    results = run_all_tests()

    print("\n" + "=" * 60)
    print("STATISTICAL TEST RESULTS")
    print("=" * 60)

    print("\nBootstrap 95% CIs:")
    for agent, ci in results["bootstrap"].items():
        print(f"  {agent}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    print("\nPairwise Wilcoxon vs Scripted:")
    for pair, test in results["pairwise"].items():
        sig = "*" if test["significant"] else ""
        print(f"  {pair}: p={test['p_value']:.4f}{sig} effect_size={test['effect_size']}")

    if "statistic" in results["friedman"]:
        print(f"\nFriedman test: stat={results['friedman']['statistic']:.2f}, "
              f"p={results['friedman']['p_value']:.4f}")


if __name__ == "__main__":
    main()
