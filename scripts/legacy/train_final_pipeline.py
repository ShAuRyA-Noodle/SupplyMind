"""
FINAL Training Pipeline — Grade-Aligned, Full Dataset, All Agents.

Runs AFTER the 5000-episode dataset is generated.
Every agent trains with grade_reward=True to maximize the GRADER score.

Pipeline:
  1. QR-DQN specialists (easy/medium/hard, 200K steps each)
  2. DT (30 epochs on full dataset)
  3. IQL (100K steps, grade-aligned)
  4. CQL (100K steps, grade-aligned)
  5. TD3+BC (100K steps, grade-aligned)
  6. BC (100 epochs)
  7. Surrogate (50 epochs)
  8. Full 9-agent benchmark with grade() scores
  9. Statistics + backtesting
  10. ONNX export

Usage:
    python train_final_pipeline.py
"""

import sys
import os
import time
import gc
import logging
import json

os.chdir("c:/Users/Dell/Desktop/Sleep-Token")
sys.path.insert(0, ".")

import torch
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_final_pipeline.log", mode="w")],
)
logger = logging.getLogger("final")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
logger.info("=" * 70)

results = {}
start = time.time()


def run(name, fn):
    logger.info("[START] %s", name)
    t = time.time()
    try:
        fn()
        elapsed = (time.time() - t) / 60
        results[name] = round(elapsed, 1)
        logger.info("[DONE] %s — %.1f min", name, elapsed)
    except Exception as e:
        results[name] = f"FAILED: {str(e)[:100]}"
        logger.error("[FAIL] %s: %s", name, e, exc_info=True)
    gc.collect()
    torch.cuda.empty_cache()


# ================================================================
# 1. QR-DQN Specialists with AutoResearch best config
# ================================================================
def train_qrdqn_specialists():
    from rl.autoresearch import ExperimentConfig, train_experiment
    for task_id, task_name in [
        ("easy_typhoon_response", "easy"),
        ("medium_multi_front", "medium"),
        ("hard_cascading_crisis", "hard"),
    ]:
        logger.info("  Training QR-DQN %s specialist (200K steps)...", task_name)
        config = ExperimentConfig(
            name=f"final_{task_name}",
            lr=1e-3, cvar_alpha=0.5, hidden_dim=256,
            grade_reward=True, alert_penalty=0.0, real_action_bonus=0.10,
            gamma=0.99, n_quantiles=51, total_steps=200_000,
            task_id=task_id,
        )
        result = train_experiment(config, device=device)
        logger.info("  %s: easy=%.3f med=%.3f hard=%.3f",
                     task_name, result.grade_easy, result.grade_medium, result.grade_hard)


run("qrdqn_specialists", train_qrdqn_specialists)

# ================================================================
# 2. Decision Transformer (30 epochs)
# ================================================================
def train_dt_final():
    from rl.decision_transformer.train import train_dt
    train_dt(epochs=30, device=device, log_wandb=False, log_mlflow=False)

run("dt", train_dt_final)

# ================================================================
# 3-6. Offline agents
# ================================================================
def train_iql_final():
    from rl.offline.baselines import train_iql
    train_iql(n_steps=100_000, device=device)

def train_cql_final():
    from rl.offline.baselines import train_cql
    train_cql(n_steps=100_000, device=device)

def train_td3bc_final():
    from rl.offline.baselines import train_td3bc
    train_td3bc(n_steps=100_000, device=device)

def train_bc_final():
    from rl.offline.baselines import train_bc
    train_bc(epochs=100, device=device)

run("iql", train_iql_final)
run("cql", train_cql_final)
run("td3bc", train_td3bc_final)
run("bc", train_bc_final)

# ================================================================
# 7. Surrogate
# ================================================================
def train_surrogate_final():
    from rl.surrogate.world_model import train_world_model
    train_world_model(epochs=50, device=device)

run("surrogate", train_surrogate_final)

# ================================================================
# 8. Ensemble tuning
# ================================================================
def tune_ensemble():
    from rl.ensemble import EnsemblePolicy
    ens = EnsemblePolicy(device="cpu")
    ens.load_models()
    ens.tune_weight(task="easy", n_episodes=20)

run("ensemble", tune_ensemble)

# ================================================================
# 9. ONNX
# ================================================================
def export_onnx():
    from rl.export_onnx import export_to_onnx
    export_to_onnx()

run("onnx", export_onnx)

# ================================================================
# 10. Specialist Router Evaluation
# ================================================================
def eval_router():
    from rl.specialist_router import SpecialistRouter
    router = SpecialistRouter(device="cpu")
    router.load_specialists()
    scores = router.evaluate_all(n_episodes=10)
    logger.info("  Router scores: %s", json.dumps({k: round(v, 3) for k, v in scores.items()}))

run("specialist_router", eval_router)

# ================================================================
# 11. Backtesting
# ================================================================
def run_backtest():
    from benchmark.backtesting import run_backtesting
    run_backtesting(n_runs=50)

run("backtesting", run_backtest)

# ================================================================
# 12. Statistics
# ================================================================
def run_stats():
    from benchmark.statistics import run_all_tests
    run_all_tests()

run("statistics", run_stats)

# ================================================================
# 13. Explainer cache
# ================================================================
def cache_explanations():
    from rl.explainer import pre_populate_cache
    pre_populate_cache(n_scenarios=50)

run("explainer", cache_explanations)

# ================================================================
# SUMMARY
# ================================================================
total = (time.time() - start) / 60
logger.info("")
logger.info("=" * 70)
logger.info("FINAL PIPELINE COMPLETE — %.1f min (%.1f hrs)", total, total / 60)
logger.info("=" * 70)

succeeded = sum(1 for v in results.values() if isinstance(v, (int, float)))
failed = sum(1 for v in results.values() if isinstance(v, str))
logger.info("Succeeded: %d | Failed: %d", succeeded, failed)

for name, val in results.items():
    if isinstance(val, (int, float)):
        logger.info("  %-25s %.1f min", name, val)
    else:
        logger.info("  %-25s %s", name, val)

# List checkpoints
from pathlib import Path
ckpt = Path("rl/checkpoints")
total_mb = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()) / 1e6
logger.info("\nCheckpoints: %.0f MB total", total_mb)

# Final test
logger.info("\nRunning pytest...")
import subprocess
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                   capture_output=True, text=True)
logger.info("Tests: %s", r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "?")
