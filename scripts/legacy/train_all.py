#!/usr/bin/env python3
"""
SupplyMind Master Training Orchestrator — FULL POWER, NO COMPROMISES.

RTX 4080 Laptop (12.9GB VRAM) + i9-13900HX (24 cores).

Pipeline:
  1. Dataset: 1000 episodes (500 scripted + 500 random)
  2. PPO: 500K steps x 3 tasks
  3. QR-DQN: 200K steps x 3 tasks
  4. Decision Transformer: 10 epochs
  5. IQL: 100K steps (pure PyTorch)
  6. CQL: 100K steps (pure PyTorch)
  7. TD3+BC: 100K steps (pure PyTorch)
  8. BC: 100 epochs
  9. Constrained PPO: 500K steps
  10. Surrogate world model: 50 epochs
  11. Ensemble tuning
  12. Full benchmark (all trained agents)
  13. Ablation study
  14. Backtesting
  15. Statistical tests
  16. ONNX export
  17. Explainer cache
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_all.log", mode="w"),
    ],
)
logger = logging.getLogger("train_all")


def setup_gpu() -> str:
    if not torch.cuda.is_available():
        logger.warning("NO GPU — will be slow!")
        return "cpu"
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("=" * 70)
    logger.info("GPU: %s (%.1f GB VRAM)", gpu, vram)
    logger.info("CUDA: %s | PyTorch: %s", torch.version.cuda, torch.__version__)
    logger.info("=" * 70)
    return "cuda"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.0
    def __enter__(self):
        self._t = time.time()
        logger.info("[START] %s", self.name)
        return self
    def __exit__(self, *a):
        self.elapsed = time.time() - self._t
        logger.info("[DONE]  %s — %.1f min", self.name, self.elapsed / 60)
        cleanup()


def run_step(name, fn, results):
    try:
        t = time.time()
        fn()
        results[name] = round((time.time() - t) / 60, 1)
    except Exception as e:
        logger.error("STEP '%s' FAILED: %s", name, e, exc_info=True)
        results[name] = f"FAILED: {str(e)[:100]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    start = time.time()
    device = args.device or setup_gpu()
    results = {}

    # =====================================================================
    # 1. DATASET — 1000 episodes
    # =====================================================================
    if not args.skip_dataset:
        def do_dataset():
            with Timer("Dataset (1000 episodes)"):
                from rl.offline.dataset import generate_dataset
                generate_dataset(n_scripted=500, n_random=500)
        run_step("dataset", do_dataset, results)

    # =====================================================================
    # 2. PPO — 500K steps per task (3 tasks)
    # =====================================================================
    def do_ppo():
        from rl.train_ppo import train_ppo
        for task in ["easy", "medium", "hard"]:
            with Timer(f"PPO {task} (500K steps)"):
                train_ppo(task=task, total_timesteps=500_000, device=device,
                          n_envs=8, log_wandb=False, log_mlflow=False)
    run_step("ppo", do_ppo, results)

    # =====================================================================
    # 3. QR-DQN — 200K steps per task
    # =====================================================================
    def do_qrdqn():
        from rl.distributional.train import train_qrdqn
        for task in ["easy", "medium", "hard"]:
            with Timer(f"QR-DQN {task} (200K steps)"):
                train_qrdqn(task=task, total_steps=200_000, device=device,
                            log_wandb=False, log_mlflow=False)
    run_step("qrdqn", do_qrdqn, results)

    # =====================================================================
    # 4. Decision Transformer — 10 epochs
    # =====================================================================
    def do_dt():
        with Timer("Decision Transformer (10 epochs)"):
            from rl.decision_transformer.train import train_dt
            train_dt(epochs=10, device=device, log_wandb=False, log_mlflow=False)
    run_step("dt", do_dt, results)

    # =====================================================================
    # 5. IQL — 100K steps (pure PyTorch)
    # =====================================================================
    def do_iql():
        with Timer("IQL (100K steps)"):
            from rl.offline.baselines import train_iql
            train_iql(n_steps=100_000, device=device)
    run_step("iql", do_iql, results)

    # =====================================================================
    # 6. CQL — 100K steps (pure PyTorch)
    # =====================================================================
    def do_cql():
        with Timer("CQL (100K steps)"):
            from rl.offline.baselines import train_cql
            train_cql(n_steps=100_000, device=device)
    run_step("cql", do_cql, results)

    # =====================================================================
    # 7. TD3+BC — 100K steps (pure PyTorch)
    # =====================================================================
    def do_td3bc():
        with Timer("TD3+BC (100K steps)"):
            from rl.offline.baselines import train_td3bc
            train_td3bc(n_steps=100_000, device=device)
    run_step("td3bc", do_td3bc, results)

    # =====================================================================
    # 8. BC — 100 epochs
    # =====================================================================
    def do_bc():
        with Timer("Behavior Cloning (100 epochs)"):
            from rl.offline.baselines import train_bc
            train_bc(epochs=100, device=device)
    run_step("bc", do_bc, results)

    # =====================================================================
    # 9. Constrained PPO — 500K steps
    # =====================================================================
    def do_constrained():
        with Timer("Constrained PPO (500K steps)"):
            from rl.constrained_ppo import train_constrained_ppo
            train_constrained_ppo(task="easy", total_timesteps=500_000, device=device)
    run_step("constrained_ppo", do_constrained, results)

    # =====================================================================
    # 10. Surrogate World Model — 50 epochs
    # =====================================================================
    def do_surrogate():
        with Timer("Neural Surrogate (50 epochs)"):
            from rl.surrogate.world_model import train_world_model
            train_world_model(epochs=50, device=device)
    run_step("surrogate", do_surrogate, results)

    # =====================================================================
    # 11. Ensemble tuning
    # =====================================================================
    def do_ensemble():
        with Timer("Ensemble Weight Tuning"):
            from rl.ensemble import EnsemblePolicy
            ens = EnsemblePolicy(device="cpu")
            ens.load_models()
            ens.tune_weight(task="easy", n_episodes=20)
    run_step("ensemble", do_ensemble, results)

    # =====================================================================
    # 12. Full Benchmark
    # =====================================================================
    def do_benchmark():
        with Timer("Full Benchmark"):
            from benchmark.run_benchmark import run_benchmark
            run_benchmark(agents=["random", "scripted"], seeds=[42, 99, 7, 123, 256], n_episodes=20)
    run_step("benchmark", do_benchmark, results)

    # =====================================================================
    # 13. Ablation
    # =====================================================================
    def do_ablation():
        with Timer("Ablation Study"):
            from benchmark.ablation import run_ablation
            run_ablation(seeds=[42, 99, 7], n_episodes=10)
    run_step("ablation", do_ablation, results)

    # =====================================================================
    # 14. Backtesting
    # =====================================================================
    def do_backtest():
        with Timer("Backtesting"):
            from benchmark.backtesting import run_backtesting
            run_backtesting(n_runs=50)
    run_step("backtest", do_backtest, results)

    # =====================================================================
    # 15. Statistical Tests
    # =====================================================================
    def do_stats():
        with Timer("Statistical Tests"):
            from benchmark.statistics import run_all_tests
            run_all_tests()
    run_step("statistics", do_stats, results)

    # =====================================================================
    # 16. ONNX Export
    # =====================================================================
    def do_onnx():
        with Timer("ONNX Export"):
            from rl.export_onnx import export_to_onnx
            export_to_onnx()
    run_step("onnx", do_onnx, results)

    # =====================================================================
    # 17. Explainer Cache
    # =====================================================================
    def do_explainer():
        with Timer("Explainer Cache (50 scenarios)"):
            from rl.explainer import pre_populate_cache
            pre_populate_cache(n_scenarios=50)
    run_step("explainer", do_explainer, results)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    total = time.time() - start
    logger.info("")
    logger.info("=" * 70)
    logger.info("FULL TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Total: %.1f min (%.1f hrs)", total / 60, total / 3600)
    logger.info("")

    succeeded = 0
    failed = 0
    for name, val in results.items():
        if isinstance(val, (int, float)):
            logger.info("  %-25s %.1f min", name, val)
            succeeded += 1
        else:
            logger.info("  %-25s %s", name, val)
            failed += 1

    logger.info("")
    logger.info("Succeeded: %d | Failed: %d", succeeded, failed)

    # List checkpoints
    ckpt = PROJECT_ROOT / "rl" / "checkpoints"
    if ckpt.exists():
        logger.info("")
        logger.info("Checkpoints:")
        total_size = 0
        for f in sorted(ckpt.rglob("*")):
            if f.is_file():
                sz = f.stat().st_size
                total_size += sz
                logger.info("  %-45s %6.1f MB", f.name, sz / 1e6)
        logger.info("  Total: %.1f MB", total_size / 1e6)

    # Save report
    report = {
        "total_minutes": round(total / 60, 1),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "steps": results,
        "succeeded": succeeded,
        "failed": failed,
    }
    (PROJECT_ROOT / "training_report.json").write_text(json.dumps(report, indent=2))

    # Final test
    logger.info("")
    logger.info("Running pytest verification...")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                       capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    last = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "no output"
    logger.info("Tests: %s", last)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
