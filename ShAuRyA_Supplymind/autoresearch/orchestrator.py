"""
orchestrator.py — Main autoresearch loop.

propose -> apply -> run -> evaluate -> accept/reject -> log -> loop.

Usage:
    python -m ShAuRyA_Supplymind.autoresearch.orchestrator --budget 6h
    python -m ShAuRyA_Supplymind.autoresearch.orchestrator --seeds-only
    python -m ShAuRyA_Supplymind.autoresearch.orchestrator --agent claude --budget 12h
    touch ShAuRyA_Supplymind/autoresearch/stop_autoresearch.flag  # graceful halt
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

from . import evaluator, lab_notebook, runner, seed_experiments
from .hypothesis_engine import Hypothesis, propose_hypothesis

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(__file__).resolve().parent
STATE_PATH = AUTORESEARCH_DIR / "state.json"
STOP_FLAG = AUTORESEARCH_DIR / "stop_autoresearch.flag"
CANDIDATE_PATH = AUTORESEARCH_DIR / "candidate_train.py"
MAX_CONSECUTIVE_REJECTS = 50


def _parse_budget(s: str) -> float:
    """'6h' -> 21600, '30m' -> 1800, '3600' -> 3600."""
    m = re.match(r"^(\d+(?:\.\d+)?)([smhd]?)$", s.strip().lower())
    if not m:
        raise ValueError(f"invalid budget: {s}")
    n, unit = float(m.group(1)), m.group(2)
    return n * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"best": None, "history": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _history_summaries(history: list[dict]) -> list[dict]:
    """Reduce history to the lightweight form the hypothesis engine expects."""
    out = []
    for h in history[-20:]:  # last 20 only to fit context
        out.append({
            "experiment_name": h.get("experiment_name", "?"),
            "metric_ci95_lower": h.get("metric_ci95_lower", 0),
            "metric_mean": h.get("metric_mean", 0),
            "status": h.get("status", "?"),
            "architecture_summary": h.get("architecture_summary", "?"),
        })
    return out


def run_seed_phase(training_seed: int = 1000, total_steps: int = 50_000) -> None:
    """Apply each of the 5 hand-crafted seeds in order, run + log."""
    logger.info("=" * 70)
    logger.info("SEED PHASE: running %d hand-crafted hypotheses", len(seed_experiments.SEEDS))
    logger.info("=" * 70)

    for seed_hyp in seed_experiments.SEEDS:
        if STOP_FLAG.exists():
            logger.info("stop flag detected, halting seed phase")
            return

        logger.info("")
        logger.info("--- SEED: %s ---", seed_hyp.name)
        logger.info("hypothesis: %s", seed_hyp.hypothesis)

        old_code = CANDIDATE_PATH.read_text(encoding="utf-8")
        try:
            new_code = seed_experiments.apply_seed(seed_hyp.name)
        except Exception as e:  # noqa: BLE001
            logger.error("[seed %s] apply failed: %s", seed_hyp.name, e)
            continue

        runner.apply_mutation(new_code)

        # Run
        result = runner.run_candidate(
            training_seed=training_seed,
            total_steps=total_steps,
            experiment_name=seed_hyp.name,
        )

        scores = result.get("grader_scores") or []
        status = result.get("status", "crash")

        # Decide
        decision = evaluator.decide(scores, seed_hyp.name, status=status)

        hyp_dict = {
            "hypothesis": seed_hyp.hypothesis,
            "expected_metric_delta": seed_hyp.expected,
            "justification": seed_hyp.justification,
            "references": seed_hyp.references,
        }

        if decision.accept:
            # Check tests still pass (test gate)
            if not runner.test_gate():
                logger.warning("[seed %s] accepted by metric but test gate FAILED — reverting", seed_hyp.name)
                runner.revert_mutation()
                lab_notebook.log_rejected(
                    experiment_name=seed_hyp.name,
                    hypothesis=hyp_dict,
                    status="test_gate_failed",
                    reason="pytest tests/ failed after mutation — reverted",
                    metric_before=_best_metric(),
                    metric_after=decision.metric_new.to_json(),
                    delta=decision.delta,
                    wall_clock_s=result.get("wall_clock_s", 0),
                    architecture=result.get("architecture_summary", ""),
                )
                continue

            lab_notebook.log_accepted(
                experiment_name=seed_hyp.name,
                hypothesis=hyp_dict,
                metric_before=_best_metric(),
                metric_after=decision.metric_new.to_json(),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                old_code=old_code,
                new_code=new_code,
            )
            evaluator.commit(
                experiment_name=seed_hyp.name,
                hypothesis=hyp_dict,
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )
        else:
            runner.revert_mutation()
            lab_notebook.log_rejected(
                experiment_name=seed_hyp.name,
                hypothesis=hyp_dict,
                status=status,
                reason=decision.reason,
                metric_before=_best_metric(),
                metric_after=(decision.metric_new.to_json() if decision.metric_new.n > 0 else None),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
            )
            evaluator.commit(
                experiment_name=seed_hyp.name,
                hypothesis=hyp_dict,
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )


def _best_metric() -> Optional[dict]:
    state = _load_state()
    best = state.get("best")
    return best["metric"] if best else None


def run_llm_phase(
    budget_s: float,
    agent: str = "ollama",
    model: Optional[str] = None,
    training_seed_base: int = 2000,
    total_steps: int = 50_000,
) -> None:
    """Loop: ask LLM agent for hypothesis, run, evaluate, log. Repeat until budget or max rejects."""
    logger.info("=" * 70)
    logger.info("LLM PHASE: agent=%s budget=%.1fh", agent, budget_s / 3600)
    logger.info("=" * 70)

    start = time.time()
    consecutive_rejects = 0
    iter_count = 0

    while time.time() - start < budget_s:
        if STOP_FLAG.exists():
            logger.info("stop flag detected, halting LLM phase")
            return
        if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
            logger.info("hit %d consecutive rejects, stopping", MAX_CONSECUTIVE_REJECTS)
            return

        iter_count += 1
        training_seed = training_seed_base + iter_count
        state = _load_state()
        history = _history_summaries(state.get("history", []))

        logger.info("")
        logger.info("--- LLM iter %d (wall %.1fs) ---", iter_count, time.time() - start)

        try:
            hyp: Hypothesis = propose_hypothesis(history, agent=agent, model=model)
        except Exception as e:  # noqa: BLE001
            logger.error("hypothesis generation failed: %s", e)
            time.sleep(30)  # backoff before retry
            continue

        logger.info("[proposed] %s", hyp.experiment_name)
        logger.info("  hypothesis: %s", hyp.hypothesis)
        logger.info("  expected:   %s", hyp.expected_metric_delta)

        old_code = CANDIDATE_PATH.read_text(encoding="utf-8")
        try:
            runner.apply_mutation(hyp.proposed_code)
        except Exception as e:  # noqa: BLE001
            logger.error("apply_mutation failed: %s", e)
            consecutive_rejects += 1
            continue

        result = runner.run_candidate(
            training_seed=training_seed,
            total_steps=total_steps,
            experiment_name=hyp.experiment_name,
        )

        scores = result.get("grader_scores") or []
        status = result.get("status", "crash")
        decision = evaluator.decide(scores, hyp.experiment_name, status=status)

        if decision.accept:
            if not runner.test_gate():
                logger.warning("[iter %d] test gate FAILED, reverting", iter_count)
                runner.revert_mutation()
                lab_notebook.log_rejected(
                    experiment_name=hyp.experiment_name,
                    hypothesis=hyp.to_json(),
                    status="test_gate_failed",
                    reason="pytest tests/ failed after mutation — reverted",
                    metric_before=_best_metric(),
                    metric_after=decision.metric_new.to_json(),
                    delta=decision.delta,
                    wall_clock_s=result.get("wall_clock_s", 0),
                    architecture=result.get("architecture_summary", ""),
                )
                consecutive_rejects += 1
                continue

            consecutive_rejects = 0
            lab_notebook.log_accepted(
                experiment_name=hyp.experiment_name,
                hypothesis=hyp.to_json(),
                metric_before=_best_metric(),
                metric_after=decision.metric_new.to_json(),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                old_code=old_code,
                new_code=hyp.proposed_code,
            )
            evaluator.commit(
                experiment_name=hyp.experiment_name,
                hypothesis=hyp.to_json(),
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )
        else:
            runner.revert_mutation()
            consecutive_rejects += 1
            lab_notebook.log_rejected(
                experiment_name=hyp.experiment_name,
                hypothesis=hyp.to_json(),
                status=status,
                reason=decision.reason,
                metric_before=_best_metric(),
                metric_after=(decision.metric_new.to_json() if decision.metric_new.n > 0 else None),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
            )
            evaluator.commit(
                experiment_name=hyp.experiment_name,
                hypothesis=hyp.to_json(),
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )

    logger.info("LLM phase finished: %d iterations in %.1fh", iter_count, (time.time() - start) / 3600)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="SupplyMind Karpathy-style autoresearch loop")
    parser.add_argument("--budget", type=str, default="6h", help="LLM-phase budget (e.g. 6h, 30m, 3600s)")
    parser.add_argument("--agent", type=str, default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seeds-only", action="store_true", help="Run only the 5 seed hypotheses, skip LLM phase")
    parser.add_argument("--skip-seeds", action="store_true", help="Skip seeds, go straight to LLM loop")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true", help="Resume: do NOT re-run seeds even if they exist")
    args = parser.parse_args()

    if STOP_FLAG.exists():
        logger.warning("stop flag exists at start — removing so we can run")
        STOP_FLAG.unlink()

    budget_s = _parse_budget(args.budget)

    # Seed phase
    if not args.skip_seeds and not args.resume:
        run_seed_phase(training_seed=1000, total_steps=args.steps)

    if args.seeds_only:
        logger.info("seeds-only mode, exiting")
        return

    # LLM phase
    run_llm_phase(
        budget_s=budget_s,
        agent=args.agent,
        model=args.model,
        training_seed_base=2000,
        total_steps=args.steps,
    )

    # Final leaderboard
    print("")
    print("=" * 70)
    print("AUTORESEARCH COMPLETE")
    print("=" * 70)
    print(lab_notebook.render_leaderboard(STATE_PATH))


if __name__ == "__main__":
    main()
