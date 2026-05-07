"""
rerun_seeds.py — Rerun specific seeds (e.g. the ones that crashed before the
FlatDiscreteEnv fix). Preserves state.json history.
"""
from __future__ import annotations

import argparse
import logging

from . import evaluator, lab_notebook, runner, seed_experiments


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+",
                        default=["s3_curriculum_learning", "s4_recurrent_ppo",
                                 "s5_action_diversity_bonus"])
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--training-seed", type=int, default=1001)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    for name in args.seeds:
        logger.info("=" * 70)
        logger.info("--- RERUN SEED: %s ---", name)
        try:
            seed_hyp = seed_experiments.get_seed(name)
        except ValueError as e:
            logger.error("unknown seed: %s", e)
            continue

        old_code = (runner.CANDIDATE_PATH.read_text(encoding="utf-8"))
        try:
            new_code = seed_experiments.apply_seed(name)
        except Exception as e:  # noqa: BLE001
            logger.error("apply failed: %s", e)
            continue

        runner.apply_mutation(new_code)
        result = runner.run_candidate(
            training_seed=args.training_seed,
            total_steps=args.steps,
            experiment_name=name + "_rerun",
        )

        scores = result.get("grader_scores") or []
        status = result.get("status", "crash")
        decision = evaluator.decide(scores, name + "_rerun", status=status)

        hyp_dict = {
            "hypothesis": seed_hyp.hypothesis,
            "expected_metric_delta": seed_hyp.expected,
            "justification": seed_hyp.justification,
            "references": seed_hyp.references,
        }

        # Always revert so next seed starts clean
        runner.revert_mutation()

        if decision.accept and runner.test_gate():
            runner.apply_mutation(new_code)  # re-apply (test_gate reverted via our revert above)
            lab_notebook.log_accepted(
                experiment_name=name + "_rerun",
                hypothesis=hyp_dict,
                metric_before=(
                    {k: evaluator._load_state().get("best", {}).get("metric", {}).get(k)
                     for k in ("mean", "std", "ci95_lower", "ci95_upper", "n")}
                    if evaluator._load_state().get("best") else None
                ),
                metric_after=decision.metric_new.to_json(),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                old_code=old_code,
                new_code=new_code,
            )
            evaluator.commit(
                experiment_name=name + "_rerun",
                hypothesis=hyp_dict,
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )
            runner.revert_mutation()  # back to baseline for next seed
        else:
            lab_notebook.log_rejected(
                experiment_name=name + "_rerun",
                hypothesis=hyp_dict,
                status=status,
                reason=decision.reason,
                metric_before=None,
                metric_after=(decision.metric_new.to_json() if decision.metric_new.n > 0 else None),
                delta=decision.delta,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
            )
            evaluator.commit(
                experiment_name=name + "_rerun",
                hypothesis=hyp_dict,
                scores=scores,
                decision=decision,
                wall_clock_s=result.get("wall_clock_s", 0),
                architecture=result.get("architecture_summary", ""),
                checkpoint_path=result.get("result_json_path", ""),
                stdout_path=result.get("stdout_path", ""),
            )


if __name__ == "__main__":
    main()
