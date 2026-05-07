"""train_dpo_roll.py — DPO via Alibaba ROLL's DPO pipeline.

Equivalent to train_dpo_trl.py but uses ROLL's production DPO pipeline. Only
reach for this if ROLL is installed (Phase A: pip install -e ROLL-main[hf],
or Phase B: WSL2 + CUDA passthrough). Otherwise prefer train_dpo_trl.

Advantages over trl fallback:
    - Async reward computation (for ongoing RL loops)
    - Drop-in 5D parallelism if ever promoted to multi-GPU
    - Same config lives in configs/dpo_qwen25_3b_supplymind.yaml
    - Identical checkpointing format as ROLL upstream (useful for env PR)

Usage:
    pip install -e ../../vendor/ROLL[hf]
    python -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_roll \\
        --config versions/v5_phoenix/roll_integration/configs/dpo_qwen25_3b_supplymind.yaml

Outputs the same adapter/metrics shape as train_dpo_trl, so downstream code
(train_dpo_judge receipt, evaluate_delta) is format-agnostic.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dpo_qwen25_3b_supplymind.yaml"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    try:
        from roll.pipeline.dpo import DPOPipeline  # type: ignore
    except ImportError as e:
        logger.error("[dpo-roll] ROLL not importable: %s", e)
        logger.error("[dpo-roll] Fall back to train_dpo_trl (identical scientific result).")
        sys.exit(2)

    if not args.config.exists():
        raise FileNotFoundError(args.config)
    logger.info("[dpo-roll] launching ROLL DPOPipeline with %s", args.config)

    if args.dry_run:
        logger.info("[dpo-roll] dry-run OK — ROLL importable, config present.")
        return

    pipeline = DPOPipeline.from_config(str(args.config))  # type: ignore[attr-defined]
    pipeline.run()
    logger.info("[dpo-roll] pipeline.run() complete.")


if __name__ == "__main__":
    main()
