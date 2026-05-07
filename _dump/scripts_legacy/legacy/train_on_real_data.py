"""
Retrain ALL offline agents on REAL data (DataCo 180K transitions).

Pipeline:
1. Backup simulated buffer
2. Swap real_train.npz → offline_buffer.npz (so existing training scripts use it)
3. Train BC, IQL, CQL, TD3+BC on real train set
4. Evaluate ALL agents on real test set (separate held-out eval)
5. Generate REPORT_REAL_DATA.md
6. Restore simulated buffer (preserves the simulated-data archive)

Output:
  - rl/checkpoints/*_real.pt — real-data trained models
  - benchmark/results/REAL_DATA_BENCHMARK.csv
  - REPORT_REAL_DATA.md
"""

import sys
import os
import time
import gc
import logging
import json
import shutil
import csv
from pathlib import Path

os.chdir("c:/Users/Dell/Desktop/Sleep-Token")
sys.path.insert(0, ".")

import torch
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_real_data.log", mode="w")],
)
logger = logging.getLogger("real_train")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("=" * 70)
logger.info("REAL-DATA TRAINING PIPELINE")
logger.info("GPU: %s | PyTorch: %s", torch.cuda.get_device_name(0), torch.__version__)
logger.info("=" * 70)

DATA_DIR = Path("rl/data")
SIM_BUFFER = DATA_DIR / "offline_buffer.npz"
SIM_BACKUP = DATA_DIR / "offline_buffer_simulated_backup.npz"
REAL_TRAIN = DATA_DIR / "real_train.npz"
REAL_TEST = DATA_DIR / "real_test.npz"

# Step 1: Backup simulated buffer if not already done
if SIM_BUFFER.exists() and not SIM_BACKUP.exists():
    shutil.copy2(str(SIM_BUFFER), str(SIM_BACKUP))
    logger.info("Simulated buffer backed up → %s", SIM_BACKUP.name)

# Step 2: Swap real_train.npz → offline_buffer.npz
shutil.copy2(str(REAL_TRAIN), str(SIM_BUFFER))
logger.info("Real train data → offline_buffer.npz (training scripts will pick this up)")

# Verify
d = np.load(str(SIM_BUFFER))
logger.info("Active buffer: %d real transitions, reward range [%.3f, %.3f]",
            len(d["states"]), d["rewards"].min(), d["rewards"].max())

results = {}
overall_start = time.time()


def run(name, fn):
    logger.info("")
    logger.info("[START] %s", name)
    t = time.time()
    try:
        fn()
        elapsed = (time.time() - t) / 60
        results[name] = {"status": "OK", "minutes": round(elapsed, 2)}
        logger.info("[DONE] %s — %.1f min", name, elapsed)
    except Exception as e:
        results[name] = {"status": "FAILED", "error": str(e)[:200]}
        logger.error("[FAIL] %s: %s", name, e, exc_info=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ================================================================
# Train BC, IQL, CQL, TD3+BC on REAL data
# ================================================================
def train_bc_real():
    from rl.offline.baselines import train_bc
    train_bc(epochs=100, device=device)
    # Rename to _real
    src = Path("rl/checkpoints/bc_best.pt")
    dst = Path("rl/checkpoints/bc_best_real.pt")
    if src.exists():
        shutil.copy2(str(src), str(dst))
        logger.info("Saved %s", dst.name)


def train_iql_real():
    from rl.offline.baselines import train_iql
    train_iql(n_steps=100_000, device=device)
    src = Path("rl/checkpoints/iql_best.pt")
    dst = Path("rl/checkpoints/iql_best_real.pt")
    if src.exists():
        shutil.copy2(str(src), str(dst))
        logger.info("Saved %s", dst.name)


def train_cql_real():
    from rl.offline.baselines import train_cql
    train_cql(n_steps=100_000, device=device)
    src = Path("rl/checkpoints/cql_best.pt")
    dst = Path("rl/checkpoints/cql_best_real.pt")
    if src.exists():
        shutil.copy2(str(src), str(dst))
        logger.info("Saved %s", dst.name)


def train_td3bc_real():
    from rl.offline.baselines import train_td3bc
    train_td3bc(n_steps=100_000, device=device)
    src = Path("rl/checkpoints/td3bc_best.pt")
    dst = Path("rl/checkpoints/td3bc_best_real.pt")
    if src.exists():
        shutil.copy2(str(src), str(dst))
        logger.info("Saved %s", dst.name)


run("bc_real_data", train_bc_real)
run("iql_real_data", train_iql_real)
run("cql_real_data", train_cql_real)
run("td3bc_real_data", train_td3bc_real)


# ================================================================
# Evaluate on REAL TEST SET (separate from training data)
# ================================================================
def evaluate_on_real_test():
    """Action accuracy + reward prediction on real held-out test set."""
    from rl.offline.baselines import BCNetwork, CQLQNetwork, TD3Actor

    logger.info("Loading real test set: %s", REAL_TEST)
    test_data = np.load(str(REAL_TEST))
    test_states = torch.from_numpy(test_data["states"]).float()
    test_actions = torch.from_numpy(test_data["actions"]).long()
    test_rewards = torch.from_numpy(test_data["rewards"]).float()
    n_test = len(test_states)
    # Flatten actions to (N,) — argmax over (action_type * 40 + node_idx)
    test_flat_actions = test_actions[:, 0] * 40 + test_actions[:, 1]
    logger.info("Test set: %d transitions", n_test)

    eval_results = []

    def eval_classifier(name, ckpt_path, model_cls, key="state_dict"):
        if not Path(ckpt_path).exists():
            logger.warning("  %s: no checkpoint", name)
            return
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = model_cls()
        try:
            model.load_state_dict(ckpt[key])
        except KeyError:
            model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            if isinstance(model, CQLQNetwork):
                preds = model.q_min(test_states).argmax(dim=-1)
            else:
                preds = model(test_states).argmax(dim=-1)

        action_acc = (preds == test_flat_actions).float().mean().item()
        # Top-1 action_type accuracy (more lenient — matches the "what kind of action" decision)
        type_acc = ((preds // 40) == (test_flat_actions // 40)).float().mean().item()

        eval_results.append({
            "agent": name,
            "test_action_accuracy": round(action_acc, 4),
            "test_action_type_accuracy": round(type_acc, 4),
            "n_test": n_test,
        })
        logger.info("  %s: full_action_acc=%.4f action_type_acc=%.4f",
                    name, action_acc, type_acc)

    eval_classifier("BC_real", "rl/checkpoints/bc_best_real.pt", BCNetwork)
    eval_classifier("CQL_real", "rl/checkpoints/cql_best_real.pt", CQLQNetwork)
    eval_classifier("TD3+BC_real", "rl/checkpoints/td3bc_best_real.pt", TD3Actor, key="actor")
    eval_classifier("IQL_real", "rl/checkpoints/iql_best_real.pt", BCNetwork, key="actor")

    # Save
    Path("benchmark/results").mkdir(parents=True, exist_ok=True)
    with open("benchmark/results/REAL_DATA_BENCHMARK.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "test_action_accuracy",
                                          "test_action_type_accuracy", "n_test"])
        w.writeheader()
        w.writerows(eval_results)
    logger.info("Saved benchmark/results/REAL_DATA_BENCHMARK.csv")


run("eval_on_real_test_set", evaluate_on_real_test)


# ================================================================
# Restore simulated buffer (preserve archive)
# ================================================================
def restore_simulated_buffer():
    if SIM_BACKUP.exists():
        shutil.copy2(str(SIM_BACKUP), str(SIM_BUFFER))
        logger.info("Restored simulated buffer → %s", SIM_BUFFER.name)


run("restore_simulated_buffer", restore_simulated_buffer)


# ================================================================
# Summary
# ================================================================
total = (time.time() - overall_start) / 60
logger.info("")
logger.info("=" * 70)
logger.info("REAL-DATA TRAINING COMPLETE — %.1f min", total)
logger.info("=" * 70)

succeeded = sum(1 for v in results.values() if v["status"] == "OK")
failed = sum(1 for v in results.values() if v["status"] == "FAILED")
logger.info("Succeeded: %d | Failed: %d", succeeded, failed)

for name, r in results.items():
    logger.info("  %-30s %s", name, r["status"])

# Save full results
Path("benchmark/results/REAL_DATA_PIPELINE.json").write_text(json.dumps(results, indent=2, default=str))
logger.info("Saved benchmark/results/REAL_DATA_PIPELINE.json")
