"""Phase N 'Chokehold' — orchestrator for full offline retraining."""
from __future__ import annotations
import logging, time, traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FAILURE_TABLE = ROOT / "FAILURE_TABLE.md"


def log_failure(step, reason):
    header = "| Phase | Step | Reason | Timestamp |\n|---|---|---|---|\n"
    if not FAILURE_TABLE.exists():
        FAILURE_TABLE.write_text("# Failure Table\n\n" + header)
    with FAILURE_TABLE.open("a") as f:
        f.write(f"| N Chokehold | {step} | {reason[:300]} | {time.strftime('%Y-%m-%d %H:%M')} |\n")


def retry(fn, name, n=2):
    for attempt in range(1, n + 1):
        try:
            t0 = time.time()
            log.info(f"=== N/{name} attempt {attempt}/{n} ===")
            r = fn()
            log.info(f"=== N/{name} OK ({time.time()-t0:.0f}s) ===")
            return r
        except Exception as e:
            log.error(f"{name} attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == n:
                log_failure(name, str(e))
                return None


def main():
    from rl.offline.baselines_v2 import train_bc_v2, train_cql_v2, train_iql_v2, train_td3bc_v2

    retry(lambda: train_bc_v2(epochs=200, batch_size=1024), "BC_v2")
    retry(lambda: train_cql_v2(n_steps=300_000, batch_size=512), "CQL_v2")
    retry(lambda: train_iql_v2(n_steps=300_000, batch_size=512), "IQL_v2")
    retry(lambda: train_td3bc_v2(n_steps=300_000, batch_size=512), "TD3BC_v2")

    log.info("Phase N 'Chokehold' complete.")


if __name__ == "__main__":
    main()
