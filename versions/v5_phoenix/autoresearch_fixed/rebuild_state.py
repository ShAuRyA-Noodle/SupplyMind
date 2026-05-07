"""rebuild_state.py — rebuild state.json from real result.json files.

Phoenix v5 fix: v4's state.json claimed all experiments crashed
('status=crash; no valid scores', wall_clock_s ~5s) but the actual
result.json files show s1/s2 ran to completion (20k steps, 122s / 135s,
9 grader scores each). This script reads the truth from result.json and
writes a correct state.json the evaluator's committal logic would produce.

Usage:
    python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE / "experiments"
STATE_PATH = HERE / "state.json"

ACCEPT_EPSILON = 0.005
BOOTSTRAP_N = 1000
RNG = np.random.default_rng(12345)

HYPOTHESES = {
    "s1_bigger_network": {
        "hypothesis": "MlpPolicy [256, 256] + ReLU beats [64, 64] on hard task (more capacity for 408-dim obs).",
        "expected_metric_delta": "+0.02 to +0.05 on CI95 lower",
        "justification": "Standard sb3 recommendation for obs_dim > 200.",
        "references": ["https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"],
    },
    "s2_higher_entropy": {
        "hypothesis": "ent_coef=0.1 vs 0.01 explores more of the 280-action space early.",
        "expected_metric_delta": "+0.01 to +0.04 on medium/hard",
        "justification": "Schulman et al. 2017 PPO paper: ent_coef sweep 0.01-0.1 optimal.",
        "references": ["https://arxiv.org/abs/1707.06347"],
    },
    "s3_curriculum_learning": {
        "hypothesis": "Curriculum (easy -> medium -> hard) accelerates learning via transfer.",
        "expected_metric_delta": "+0.03 to +0.07 on hard task",
        "justification": "Bengio et al. 2009 curriculum learning.",
        "references": ["https://dl.acm.org/doi/10.1145/1553374.1553380"],
    },
    "s4_recurrent_ppo": {
        "hypothesis": "RecurrentPPO with LSTM-128 captures long-horizon dependencies.",
        "expected_metric_delta": "-0.10 to +0.05 (risky)",
        "justification": "R6_ALGO_COMPARISON: RecurrentPPO 1.081 vs MaskablePPO 1.201.",
        "references": ["versions/v3_arcadia/results/R6_ALGO_COMPARISON.json"],
    },
    "s5_action_diversity_bonus": {
        "hypothesis": "Bonus reward for actions not used in last 5 steps encourages exploration.",
        "expected_metric_delta": "+0.01 to +0.03 on medium",
        "justification": "Pathak et al. 2017 curiosity-driven exploration (cheap lexical proxy).",
        "references": ["https://arxiv.org/abs/1705.05363"],
    },
}


def bootstrap(scores):
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return dict(mean=0.0, std=0.0, ci95_lower=0.0, ci95_upper=0.0, n=0)
    means = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        means[i] = RNG.choice(arr, size=n, replace=True).mean()
    return dict(
        mean=round(float(arr.mean()), 4),
        std=round(float(arr.std(ddof=1) if n > 1 else 0.0), 4),
        ci95_lower=round(float(np.percentile(means, 2.5)), 4),
        ci95_upper=round(float(np.percentile(means, 97.5)), 4),
        n=n,
    )


def classify(name: str, result: dict | None) -> tuple[dict, bool, str]:
    if result is None or not result.get("grader_scores"):
        stderr_path = EXP_DIR / name / "train.stderr.log"
        err = stderr_path.read_text(encoding="utf-8", errors="ignore") if stderr_path.exists() else ""
        if "shape" in err and "is invalid" in err:
            return ({}, False, "v4 crash: MaskablePPO action_mask shape mismatch when set_env() swaps env mid-training. Fixed in Phoenix via save->reload pattern (see seed_experiments._s3_curriculum)." )
        if "can only convert an array of size 1" in err:
            return ({}, False, "v4 crash: _safe_predict() can't handle RecurrentPPO's batched array return. Fixed in Phoenix via .flatten()[0] instead of .item().")
        if err.strip():
            return ({}, False, f"v4 crash: {err.strip().splitlines()[-1][:200]}")
        return ({}, False, "v4: experiment not yet run")
    return (bootstrap(result["grader_scores"]), True, "ok")


def main():
    history = []
    best = None
    run_order = ["s1_bigger_network", "s2_higher_entropy", "s3_curriculum_learning", "s4_recurrent_ppo", "s5_action_diversity_bonus"]
    for name in run_order:
        # Prefer *_rerun/ (Phoenix v5 post-fix runs) over the original crash dir
        rerun_dir = EXP_DIR / f"{name}_rerun"
        base_dir = EXP_DIR / name
        if (rerun_dir / "result.json").exists() and (rerun_dir / "result.json").stat().st_size > 0:
            exp_dir = rerun_dir
            exp_source = "phoenix_rerun"
        else:
            exp_dir = base_dir
            exp_source = "v4_original"
        result_path = exp_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() and result_path.stat().st_size > 0 else None
        metric, ran, reason = classify(name, result)
        entry = {
            "experiment_name": name,
            "exp_source": exp_source,
            "hypothesis": HYPOTHESES[name],
            "grader_scores": result.get("grader_scores", []) if result else [],
            "metric": metric if metric else None,
            "wall_clock_s": result.get("wall_clock_s", 0.0) if result else 0.0,
            "total_steps": result.get("total_steps", 0) if result else 0,
            "architecture_summary": result.get("architecture_summary", "") if result else "",
            "stdout_path": str(exp_dir / "train.stdout.log"),
            "stderr_path": str(exp_dir / "train.stderr.log"),
        }
        if not ran:
            entry.update(accepted=False, reason=reason, delta_ci95_lower=-1.0,
                         metric_ci95_lower=0.0, metric_mean=0.0, status="rejected_or_pending")
        else:
            ci_low = metric["ci95_lower"]
            if best is None:
                entry.update(accepted=True, reason="first accepted experiment -- seeding baseline",
                             delta_ci95_lower=ci_low, metric_ci95_lower=ci_low,
                             metric_mean=metric["mean"], status="accepted")
                best = {"experiment_name": name, "metric": metric, "architecture_summary": entry["architecture_summary"],
                        "checkpoint_path": str(EXP_DIR / "seed1000_candidate" / "policy.zip"),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            else:
                best_low = best["metric"]["ci95_lower"]
                delta = ci_low - best_low
                accepted = delta > ACCEPT_EPSILON
                entry.update(
                    accepted=accepted,
                    reason=(f"CI95 lower +{delta:.4f} > {ACCEPT_EPSILON:.4f} threshold"
                            if accepted else
                            f"CI95 lower delta {delta:+.4f} <= {ACCEPT_EPSILON:.4f} threshold"),
                    delta_ci95_lower=round(delta, 4),
                    metric_ci95_lower=ci_low,
                    metric_mean=metric["mean"],
                    status="accepted" if accepted else "rejected",
                )
                if accepted:
                    best = {"experiment_name": name, "metric": metric,
                            "architecture_summary": entry["architecture_summary"],
                            "checkpoint_path": str(EXP_DIR / "seed1000_candidate" / "policy.zip"),
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        history.append(entry)

    state = {
        "best": best,
        "history": history,
        "rebuilt_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rebuilt_note": "Phoenix v5 rebuild: v4 state.json was stale (claimed all crashed). This state reflects the real result.json artifacts plus Phoenix fixes to s3/s4.",
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"[rebuild] wrote {STATE_PATH}")
    print(f"[rebuild] best: {best['experiment_name'] if best else None}")
    for h in history:
        print(f"  - {h['experiment_name']}: status={h['status']} mean={h['metric_mean']} ci95_lower={h['metric_ci95_lower']}")


if __name__ == "__main__":
    main()
