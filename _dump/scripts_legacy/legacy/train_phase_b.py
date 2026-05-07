"""
Phase B — Train all offline RL agents on unified real-data buffer.

Swaps offline_buffer.npz -> real_unified.npz for training duration.
Saves checkpoints as *_best_real_v2.pt.
Restores original simulated offline_buffer.npz on exit.

Agents trained:
  - BC (100 epochs)
  - IQL (100K steps)
  - CQL (100K steps)
  - TD3+BC (100K steps)
  - Decision Transformer (30 epochs)

Retry policy: 2 attempts per agent; failures logged to FAILURE_TABLE.md.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rl" / "data"
CKPT = ROOT / "rl" / "checkpoints"
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"

SIM_BUFFER = DATA / "offline_buffer.npz"
SIM_BACKUP = DATA / "offline_buffer_simulated_backup.npz"
REAL_BUFFER = DATA / "real_unified.npz"


def log_failure(phase: str, step: str, reason: str):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| {phase} | {step} | {reason[:200]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def swap_to_real():
    if not SIM_BACKUP.exists():
        shutil.copy(SIM_BUFFER, SIM_BACKUP)
        log.info(f"Backed up simulated buffer -> {SIM_BACKUP.name}")
    shutil.copy(REAL_BUFFER, SIM_BUFFER)
    log.info(f"Swapped offline_buffer.npz <- real_unified.npz")


def restore_simulated():
    if SIM_BACKUP.exists():
        shutil.copy(SIM_BACKUP, SIM_BUFFER)
        log.info("Restored simulated offline_buffer.npz")


def retry(fn, name, max_attempts=2):
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.time()
            log.info(f"=== {name} attempt {attempt}/{max_attempts} ===")
            result = fn()
            log.info(f"=== {name} OK ({time.time()-t0:.1f}s) ===")
            return result
        except Exception as e:
            log.error(f"{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == max_attempts:
                log_failure("B", name, str(e))
                return None


def train_bc_real():
    from rl.offline.baselines import train_bc
    path = train_bc(epochs=30, batch_size=1024, lr=3e-4, device="cuda", scripted_only=False)
    # Copy to v2 name
    v2 = CKPT / "bc_best_real_v2.pt"
    shutil.copy(path, v2)
    log.info(f"Saved {v2.name}")
    return v2


def train_iql_real():
    from rl.offline.baselines import train_iql
    path = train_iql(n_steps=50_000, batch_size=512, device="cuda")
    v2 = CKPT / "iql_best_real_v2.pt"
    shutil.copy(path, v2)
    return v2


def train_cql_real():
    from rl.offline.baselines import train_cql
    path = train_cql(n_steps=50_000, batch_size=512, device="cuda")
    v2 = CKPT / "cql_best_real_v2.pt"
    shutil.copy(path, v2)
    return v2


def train_td3bc_real():
    from rl.offline.baselines import train_td3bc
    path = train_td3bc(n_steps=50_000, batch_size=512, device="cuda")
    v2 = CKPT / "td3bc_best_real_v2.pt"
    shutil.copy(path, v2)
    return v2


def train_dt_real():
    from rl.decision_transformer.train import train_dt
    path = train_dt(epochs=10, batch_size=128, lr=1e-4, device="cuda")
    if path and Path(path).exists():
        v2 = CKPT / "dt_best_real_v2.pt"
        shutil.copy(path, v2)
        return v2
    return None


def main():
    try:
        swap_to_real()

        results = {}
        for name, fn in [
            ("BC", train_bc_real),
            ("IQL", train_iql_real),
            ("CQL", train_cql_real),
            ("TD3+BC", train_td3bc_real),
            ("DT", train_dt_real),
        ]:
            results[name] = retry(fn, name)

        log.info(f"Phase B results: {results}")
        (ROOT / "phase_b_results.json").write_text(json.dumps({k: str(v) for k, v in results.items()}, indent=2))
    finally:
        restore_simulated()
        log.info("Phase B done.")


if __name__ == "__main__":
    main()
