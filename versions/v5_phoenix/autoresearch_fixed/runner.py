"""
runner.py — Fixed-budget subprocess executor for candidate_train.py.

Spawns candidate_train.py in an isolated subprocess with:
    - hard 10-min wall-clock timeout (SIGTERM then SIGKILL)
    - stdout/stderr captured to log file
    - VRAM pre-check (abort if < 2 GB free)
    - NaN detection (scrapes training log)
    - Test gate (pytest tests/ -q after training must pass)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESEARCH_DIR.parents[1]
CANDIDATE_PATH = AUTORESEARCH_DIR / "candidate_train.py"
EXPERIMENTS_DIR = AUTORESEARCH_DIR / "experiments"

WALL_CLOCK_MAX_S = 600  # 10 minutes
MIN_VRAM_GB = 2.0
TRAINING_SEED_DEFAULT = 1000  # agent-provided, but 1000 is the seed for seed_experiments


def _check_vram() -> tuple[float, float]:
    """Return (total_gb, free_gb). If no CUDA, returns (0, inf)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, float("inf")
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        free = (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9
        return total, free
    except Exception:  # noqa: BLE001
        return 0.0, float("inf")


def _has_nan(log_text: str) -> bool:
    """Scrape training log for NaN indicators."""
    patterns = ("loss is nan", "nan detected", "inf loss", "ValueError: NaN")
    low = log_text.lower()
    return any(p.lower() in low for p in patterns)


def run_candidate(
    training_seed: int = TRAINING_SEED_DEFAULT,
    total_steps: int = 50_000,
    experiment_name: str = "candidate",
    timeout_s: int = WALL_CLOCK_MAX_S,
) -> dict:
    """Execute candidate_train.py as subprocess with guards.

    Returns:
        {
            "status": "ok" | "timeout" | "crash" | "nan" | "oom",
            "grader_scores": list[float] | None,
            "wall_clock_s": float,
            "total_steps": int,
            "architecture_summary": str,
            "stdout_path": str,
            "stderr_path": str,
            "result_json_path": str,
            "error": str | None,
        }
    """
    # Eval seed overlap sanity check
    if training_seed in (42, 99, 7):
        raise ValueError(f"training_seed {training_seed} collides with EVAL_SEEDS; program.md rule 2")

    # Pre-flight VRAM
    total_vram, free_vram = _check_vram()
    if free_vram < MIN_VRAM_GB:
        logger.warning("skipping experiment %s: only %.1f GB free VRAM < %.1f min",
                       experiment_name, free_vram, MIN_VRAM_GB)
        return {
            "status": "oom",
            "error": f"VRAM {free_vram:.1f} GB < {MIN_VRAM_GB} min",
            "grader_scores": None,
            "wall_clock_s": 0.0,
            "total_steps": 0,
            "architecture_summary": "",
            "stdout_path": "",
            "stderr_path": "",
            "result_json_path": "",
        }

    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = exp_dir / "train.stdout.log"
    stderr_path = exp_dir / "train.stderr.log"
    result_json = exp_dir / "result.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Disable tokenizer parallelism warnings in subprocess
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        str(CANDIDATE_PATH),
        "--seed", str(training_seed),
        "--steps", str(total_steps),
        "--out", str(result_json),
    ]

    start = time.time()
    stdout_f = stdout_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        try:
            proc.wait(timeout=timeout_s)
            return_code = proc.returncode
            status = "ok" if return_code == 0 else "crash"
        except subprocess.TimeoutExpired:
            logger.warning("experiment %s exceeded %ds, killing", experiment_name, timeout_s)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return_code = -signal.SIGTERM
            status = "timeout"
    finally:
        stdout_f.close()
        stderr_f.close()

    wall = time.time() - start

    # NaN scrape
    if status == "ok":
        try:
            log_text = stdout_path.read_text(encoding="utf-8", errors="ignore") + \
                       stderr_path.read_text(encoding="utf-8", errors="ignore")
            if _has_nan(log_text):
                status = "nan"
        except Exception:  # noqa: BLE001
            pass

    # Parse result JSON
    grader_scores = None
    arch = ""
    if status == "ok" and result_json.exists():
        try:
            r = json.loads(result_json.read_text())
            grader_scores = r.get("grader_scores")
            arch = r.get("architecture_summary", "")
        except Exception as e:  # noqa: BLE001
            status = "crash"
            logger.error("failed to parse result.json for %s: %s", experiment_name, e)

    result = {
        "status": status,
        "grader_scores": grader_scores,
        "wall_clock_s": round(wall, 2),
        "total_steps": total_steps,
        "architecture_summary": arch,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_json_path": str(result_json),
        "error": None if status == "ok" else f"status={status} rc={return_code}",
    }

    logger.info(
        "[runner] %s status=%s wall=%.1fs scores=%s",
        experiment_name, status, wall,
        "None" if grader_scores is None else f"mean={sum(grader_scores)/len(grader_scores):.3f}",
    )

    return result


def test_gate() -> bool:
    """Run `pytest tests/ -q` and return True if all pass."""
    logger.info("[test_gate] running pytest tests/ -q ...")
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
            cwd=str(PROJECT_ROOT),
            timeout=300,
            capture_output=True,
            text=True,
        )
        passed = res.returncode == 0
        logger.info("[test_gate] %s", "PASS" if passed else f"FAIL: {res.stdout[-500:]}")
        return passed
    except Exception as e:  # noqa: BLE001
        logger.error("[test_gate] crashed: %s", e)
        return False


def apply_mutation(new_code: str, backup: bool = True) -> Path:
    """Write new_code to candidate_train.py, optionally backing up the old."""
    if backup:
        bak = CANDIDATE_PATH.with_suffix(".py.bak")
        shutil.copy2(CANDIDATE_PATH, bak)
    CANDIDATE_PATH.write_text(new_code, encoding="utf-8")
    return CANDIDATE_PATH


def revert_mutation() -> bool:
    """Restore candidate_train.py from .bak."""
    bak = CANDIDATE_PATH.with_suffix(".py.bak")
    if not bak.exists():
        logger.error("[revert] no .bak file found")
        return False
    shutil.copy2(bak, CANDIDATE_PATH)
    logger.info("[revert] restored candidate_train.py from .bak")
    return True


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--name", type=str, default="manual_run")
    parser.add_argument("--timeout", type=int, default=WALL_CLOCK_MAX_S)
    parser.add_argument("--test-gate", action="store_true")
    args = parser.parse_args()

    if args.test_gate:
        ok = test_gate()
        sys.exit(0 if ok else 1)

    res = run_candidate(
        training_seed=args.seed,
        total_steps=args.steps,
        experiment_name=args.name,
        timeout_s=args.timeout,
    )
    print(json.dumps(res, indent=2))
