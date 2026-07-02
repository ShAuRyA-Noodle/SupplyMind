"""
Benchmark the Fast Monte Carlo Engine vs Python original.

Measures:
1. Speed: how many calls/sec each engine can do
2. Output similarity: do they produce comparable distributions?

If fast engine is 5x+ faster AND output stays in same order of magnitude,
we use it for training.

Usage:
    python -m rl.fast_engine.benchmark
"""

import sys
import os
import time
import logging
from pathlib import Path

os.chdir("c:/Users/Dell/Desktop/Sleep-Token")
sys.path.insert(0, ".")

import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fast_mc_bench")


def run_benchmark():
    from server.supply_environment import SupplyMindEnvironment
    from server.engine.monte_carlo import MonteCarloEngine
    from rl.fast_engine import FastMonteCarloEngine, FAST_MC_AVAILABLE

    if not FAST_MC_AVAILABLE:
        print("Fast MC not available — Numba compile failed")
        return

    # Set up environment with some active disruptions
    env = SupplyMindEnvironment()
    env.reset(task_id="medium_multi_front", seed=42)
    # Step until we hit the active phase of a disruption
    from supplymind.contracts import SupplyMindAction
    active_signals = []
    for _ in range(20):
        obs = env.step(SupplyMindAction(action_type="do_nothing"))
        active_signals = [s for s in obs.active_signals if s.lifecycle_phase in ("active", "warning")]
        if active_signals:
            break

    graph = env.engine.graph

    if not active_signals:
        print("Could not get active disruptions — creating synthetic for benchmark")
        from supplymind.contracts import DisruptionSignal
        # Create a realistic synthetic for benchmarking only (not for saving)
        active_signals = [
            DisruptionSignal(
                signal_id="bench_syn_1",
                disruption_type="tropical_cyclone",
                severity=0.7,
                confidence=0.85,
                affected_region="Taiwan",
                affected_node_ids=[list(graph.G.nodes())[i] for i in range(min(3, len(graph.G.nodes())))],
                time_to_impact_hours=24.0,
                estimated_duration_days=14.0,
                description="Benchmark-only synthetic signal",
                lifecycle_phase="active",
            ),
            DisruptionSignal(
                signal_id="bench_syn_2",
                disruption_type="port_congestion",
                severity=0.5,
                confidence=0.7,
                affected_region="Singapore",
                affected_node_ids=[list(graph.G.nodes())[i] for i in range(min(2, len(graph.G.nodes())))],
                time_to_impact_hours=0.0,
                estimated_duration_days=21.0,
                description="Benchmark-only synthetic signal",
                lifecycle_phase="active",
            ),
        ]

    logger.info("Active disruptions: %d", len(active_signals))
    logger.info("Graph nodes: %d", len(graph.G.nodes()))

    py_engine = MonteCarloEngine(seed=42)
    fast_engine = FastMonteCarloEngine(seed=42)

    # Warmup (Numba compiles on first call)
    logger.info("Warming up Numba JIT...")
    _ = fast_engine.run_simulation(graph, active_signals, n_simulations=100)
    logger.info("JIT compiled.")

    N_CALLS = 20
    N_SIMS = 1000

    # Benchmark Python
    t_start = time.perf_counter()
    py_results = []
    for _ in range(N_CALLS):
        r = py_engine.run_simulation(graph, active_signals, n_simulations=N_SIMS)
        py_results.append(r["p50_loss"])
    py_time = time.perf_counter() - t_start

    # Benchmark Fast
    t_start = time.perf_counter()
    fast_results = []
    for _ in range(N_CALLS):
        r = fast_engine.run_simulation(graph, active_signals, n_simulations=N_SIMS)
        fast_results.append(r["p50_loss"])
    fast_time = time.perf_counter() - t_start

    speedup = py_time / fast_time if fast_time > 0 else 0

    print("\n" + "=" * 60)
    print("MONTE CARLO ENGINE BENCHMARK")
    print("=" * 60)
    print(f"  Calls: {N_CALLS}, Simulations each: {N_SIMS}")
    print(f"  Python engine: {py_time*1000:.1f} ms total ({py_time/N_CALLS*1000:.2f} ms/call)")
    print(f"  Fast engine:   {fast_time*1000:.1f} ms total ({fast_time/N_CALLS*1000:.2f} ms/call)")
    print(f"  SPEEDUP: {speedup:.1f}x")
    print()
    print("  Output comparison (P50 loss):")
    print(f"    Python mean: {np.mean(py_results):,.0f}")
    print(f"    Fast mean:   {np.mean(fast_results):,.0f}")
    py_mean = np.mean(py_results) if np.mean(py_results) > 0 else 1
    print(f"    Ratio:       {np.mean(fast_results) / py_mean:.3f}x")
    print(f"  (They won't match exactly — different RNG streams —")
    print(f"   but should be in same order of magnitude)")
    print()

    # Save results
    result = {
        "python_ms_per_call": py_time / N_CALLS * 1000,
        "fast_ms_per_call": fast_time / N_CALLS * 1000,
        "speedup_x": round(speedup, 2),
        "python_p50_mean": float(np.mean(py_results)),
        "fast_p50_mean": float(np.mean(fast_results)),
        "n_calls": N_CALLS,
        "n_sims_per_call": N_SIMS,
        "fast_mc_available": FAST_MC_AVAILABLE,
    }
    Path("benchmark/results").mkdir(parents=True, exist_ok=True)
    Path("benchmark/results/fast_mc_benchmark.json").write_text(json.dumps(result, indent=2))
    print(f"Results saved to benchmark/results/fast_mc_benchmark.json")


if __name__ == "__main__":
    run_benchmark()
