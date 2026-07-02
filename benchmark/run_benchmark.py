"""
Benchmark suite for SupplyMind.

Evaluates the agents that have a real evaluator here (random, scripted) across
3 tasks x 5 seeds x 20 episodes each and writes CSVs to benchmark/results/.

Trained agents (bc, td3bc, cql, iql, ppo, qrdqn, dt, ensemble) require loaded
checkpoints and are evaluated by benchmark/run_full_benchmark.py against
rl/checkpoints/. This module does NOT substitute a placeholder for them: any
trained agent requested here is skipped loudly and recorded as 'not_evaluated'.

Usage:
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --agents ppo scripted --seeds 42 99
"""

from __future__ import annotations

import argparse
import csv
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
TASK_IDS = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
TASK_SHORT = {"easy_typhoon_response": "Easy", "medium_multi_front": "Medium", "hard_cascading_crisis": "Hard"}

ALL_AGENTS = [
    "random", "bc", "td3bc", "cql", "scripted", "iql", "ppo", "qrdqn", "dt", "ensemble",
]

# Agents with a real evaluator in this module. Everything else needs a trained
# checkpoint and is handled by run_full_benchmark.py.
REAL_AGENTS = {"random", "scripted"}


class AgentNotAvailable(RuntimeError):
    """Raised when an agent has no real evaluator in this module."""


def evaluate_random(task_id: str, seed: int, n_episodes: int = 20) -> list[float]:
    """Random agent baseline."""
    import gymnasium as gym
    import rl  # noqa: F401
    env_map = {"easy_typhoon_response": "SupplyMind-Easy-v1", "medium_multi_front": "SupplyMind-Medium-v1", "hard_cascading_crisis": "SupplyMind-Hard-v1"}
    scores = []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        env = gym.make(env_map[task_id])
        obs, info = env.reset(seed=seed * 1000 + ep)
        total_r = 0.0
        while True:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        scores.append(total_r)
        env.close()
    return scores


def evaluate_scripted(task_id: str, seed: int, n_episodes: int = 20) -> list[float]:
    """Scripted agent."""
    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action
    scores = []
    env = SupplyMindEnvironment()
    for ep in range(n_episodes):
        obs = env.reset(task_id=task_id, seed=seed * 1000 + ep if seed else None)
        step = 0
        total_r = 0.0
        while not obs.done:
            action = choose_action(obs, step)
            obs = env.step(action)
            total_r += obs.reward
            step += 1
        result = env.grade()
        scores.append(result["score"])
    return scores


def evaluate_agent(agent_name: str, task_id: str, seed: int, n_episodes: int = 20) -> list[float]:
    """Dispatch to the right evaluation function.

    Raises:
        AgentNotAvailable: If the agent needs a trained checkpoint that this
            module does not load. Callers must exclude it from results rather
            than substitute a placeholder.
    """
    if agent_name == "random":
        return evaluate_random(task_id, seed, n_episodes)
    elif agent_name == "scripted":
        return evaluate_scripted(task_id, seed, n_episodes)
    else:
        raise AgentNotAvailable(
            f"Agent '{agent_name}' has no real evaluator in run_benchmark.py "
            f"(only {sorted(REAL_AGENTS)} do). Trained checkpoints are evaluated "
            f"by benchmark/run_full_benchmark.py against rl/checkpoints/. "
            f"Refusing to substitute a scripted placeholder."
        )


def run_benchmark(
    agents: list[str] | None = None,
    seeds: list[int] | None = None,
    n_episodes: int = 20,
) -> Path:
    """Run full benchmark suite."""
    if agents is None:
        agents = ALL_AGENTS
    if seeds is None:
        seeds = [42, 99, 7, 123, 256]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "benchmark_results.csv"

    logger.info("=" * 70)
    logger.info("SUPPLYMIND BENCHMARK SUITE")
    logger.info("  Agents: %s", agents)
    logger.info("  Tasks: %s", [TASK_SHORT[t] for t in TASK_IDS])
    logger.info("  Seeds: %s | Episodes per config: %d", seeds, n_episodes)
    logger.info("  Total runs: %d", len(agents) * len(TASK_IDS) * len(seeds) * n_episodes)
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []
    evaluated_agents: list[str] = []
    not_evaluated: list[str] = []
    start = time.time()

    for agent in agents:
        try:
            agent_rows: list[dict[str, Any]] = []
            for task_id in TASK_IDS:
                agent_task_scores = []
                for seed in seeds:
                    scores = evaluate_agent(agent, task_id, seed, n_episodes)
                    agent_task_scores.extend(scores)
                    for s in scores:
                        agent_rows.append({
                            "agent": agent,
                            "task": TASK_SHORT[task_id],
                            "task_id": task_id,
                            "seed": seed,
                            "score": s,
                        })

                mean = np.mean(agent_task_scores)
                std = np.std(agent_task_scores)
                logger.info("  %s x %s: %.4f +/- %.4f (n=%d)",
                            agent, TASK_SHORT[task_id], mean, std, len(agent_task_scores))
        except AgentNotAvailable as e:
            logger.warning("NOT EVALUATED — excluding agent '%s' from results: %s", agent, e)
            not_evaluated.append(agent)
            continue

        all_results.extend(agent_rows)
        evaluated_agents.append(agent)

    # Write per-episode CSV (evaluated agents only)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "task", "task_id", "seed", "score"])
        writer.writeheader()
        writer.writerows(all_results)

    # Write summary table. Evaluated agents get real means; agents that could
    # not be evaluated are written with an explicit 'not_evaluated' marker so
    # they are never mistaken for a real (or scripted) score.
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Agent", "Easy", "Medium", "Hard", "Average"])
        for agent in agents:
            if agent in not_evaluated:
                writer.writerow([agent, "not_evaluated", "not_evaluated", "not_evaluated", "not_evaluated"])
                continue
            row = [agent]
            task_avgs = []
            for task_id in TASK_IDS:
                task_scores = [r["score"] for r in all_results if r["agent"] == agent and r["task_id"] == task_id]
                mean = np.mean(task_scores) if task_scores else 0
                std = np.std(task_scores) if task_scores else 0
                row.append(f"{mean:.3f}+/-{std:.3f}")
                task_avgs.append(mean)
            row.append(f"{np.mean(task_avgs):.3f}")
            writer.writerow(row)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("Benchmark complete in %.1f min", elapsed / 60)
    logger.info("  Evaluated: %s", evaluated_agents)
    if not_evaluated:
        logger.warning("  NOT evaluated (no checkpoint / no real evaluator here): %s", not_evaluated)
    logger.info("  Results: %s", output_path)
    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 70)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run SupplyMind benchmark suite")
    parser.add_argument("--agents", nargs="+", default=None, choices=ALL_AGENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    run_benchmark(agents=args.agents, seeds=args.seeds, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
