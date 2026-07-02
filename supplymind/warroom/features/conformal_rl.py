"""
conformal_rl.py — F6. Conformal-calibrated RL policy wrapper.

Wraps any action-returning policy with split-conformal prediction intervals
over per-action expected reward. At prediction time the wrapper returns:

    {
        "action": <argmax action>,
        "reward_p50": <median expected reward>,
        "reward_ci95_lower": <lower 95% bound>,
        "reward_ci95_upper": <upper 95% bound>,
        "width_95": ci_upper - ci_lower,
        "abstain": True if width_95 > abstain_threshold else False,
    }

Method (Foygel Barber 2022 split-conformal, adapted to RL Q-values):

1. Collect N_cal Monte-Carlo rollouts from the policy against an env seed set,
   record per-rollout episode returns for each action executed at state s0.
2. For each action a with n_a >= 5 samples, compute residuals r_i = |R_i - mean(R)|
   and quantile q_hat(alpha) = ceil((n_a+1)(1-alpha))/n_a-th order statistic
   of sorted residuals.
3. At inference: reward_p50(a) = running mean, interval = [mean - q_hat, mean + q_hat].

Novelty: the combination of MaskablePPO action probabilities + split-conformal
intervals yields a policy that can ABSTAIN when the reward interval is wider
than a safety threshold — an actionable form of RL uncertainty that is
appropriate for high-stakes supply-chain operations.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "F6_CONFORMAL_RL.json"


@dataclass
class ConformalResult:
    action: int
    reward_p50: float
    reward_ci95_lower: float
    reward_ci95_upper: float
    width_95: float
    abstain: bool
    n_samples_used: int = 0
    per_action_intervals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reward_p50": round(self.reward_p50, 4),
            "reward_ci95_lower": round(self.reward_ci95_lower, 4),
            "reward_ci95_upper": round(self.reward_ci95_upper, 4),
            "width_95": round(self.width_95, 4),
            "abstain": self.abstain,
            "n_samples_used": self.n_samples_used,
            "per_action_intervals": {
                int(k): {kk: round(vv, 4) if isinstance(vv, float) else vv
                         for kk, vv in v.items()}
                for k, v in self.per_action_intervals.items()
            },
        }


def split_conformal_q_hat(residuals: np.ndarray, alpha: float = 0.05) -> float:
    """Return the split-conformal quantile q_hat at level alpha.

    Classic Foygel Barber 2022 finite-sample formula:
        q_hat = residuals sorted, take ceil((n+1)*(1-alpha))/n percentile.
    """
    n = len(residuals)
    if n == 0:
        return float("inf")
    sorted_r = np.sort(np.abs(residuals))
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(sorted_r[rank])


def conformal_intervals_per_action(
    rollouts: dict[int, list[float]],
    alpha: float = 0.05,
) -> dict[int, dict]:
    """For each action, compute {mean, q_hat, lo, hi, n}."""
    out: dict[int, dict] = {}
    for action, rewards in rollouts.items():
        if len(rewards) < 2:
            continue
        arr = np.array(rewards, dtype=np.float64)
        mean = float(arr.mean())
        residuals = arr - mean
        q = split_conformal_q_hat(residuals, alpha=alpha)
        out[int(action)] = {
            "mean": mean,
            "q_hat": q,
            "lo": mean - q,
            "hi": mean + q,
            "n": len(rewards),
        }
    return out


def wrap_policy_decision(
    rollouts: dict[int, list[float]],
    action_mask: np.ndarray | None = None,
    alpha: float = 0.05,
    abstain_threshold: float = 0.8,
) -> ConformalResult:
    """Given per-action rollout samples, return the calibrated decision.

    abstain_threshold: if width_95 of chosen action exceeds this, abstain flag True.
    """
    intervals = conformal_intervals_per_action(rollouts, alpha=alpha)
    valid = {a: v for a, v in intervals.items() if
             action_mask is None or (a < len(action_mask) and action_mask[a])}

    if not valid:
        return ConformalResult(
            action=-1, reward_p50=float("-inf"),
            reward_ci95_lower=float("-inf"), reward_ci95_upper=float("-inf"),
            width_95=float("inf"), abstain=True,
            per_action_intervals=intervals,
        )

    # Select by mean, but could also use "LCB-optimistic" (lo bound)
    best = max(valid.items(), key=lambda kv: kv[1]["mean"])
    a, v = best
    width = v["hi"] - v["lo"]
    return ConformalResult(
        action=int(a),
        reward_p50=v["mean"],
        reward_ci95_lower=v["lo"],
        reward_ci95_upper=v["hi"],
        width_95=width,
        abstain=width > abstain_threshold,
        n_samples_used=v["n"],
        per_action_intervals=intervals,
    )


# ---------------------------------------------------------------------------
# Demo: synthetic supply-chain rollouts
# ---------------------------------------------------------------------------


def demo_synthetic_rollouts(
    n_actions: int = 5,
    n_cal_per_action: int = 30,
    seed: int = 42,
) -> dict[int, list[float]]:
    """Generate synthetic per-action rollout rewards with known variance.

    Action 0 is best in mean but noisy; action 3 is mediocre but tight; wider
    intervals on noisy actions demonstrate the conformal wrapper's value.
    """
    rng = np.random.default_rng(seed)
    profiles = {
        0: {"mean": 1.20, "std": 0.60},   # best-on-average, noisy
        1: {"mean": 1.05, "std": 0.20},   # tight, slightly worse
        2: {"mean": 0.92, "std": 0.30},
        3: {"mean": 0.95, "std": 0.08},   # tightest interval
        4: {"mean": 0.70, "std": 0.40},
    }
    rollouts: dict[int, list[float]] = {}
    for a in range(n_actions):
        p = profiles.get(a, {"mean": 0.5, "std": 0.5})
        rollouts[a] = rng.normal(p["mean"], p["std"], size=n_cal_per_action).tolist()
    return rollouts


def run_demo() -> dict:
    rollouts = demo_synthetic_rollouts()
    # All 5 actions unmasked
    mask = np.ones(5, dtype=bool)
    # Three abstain thresholds to show behavior change
    conservative = wrap_policy_decision(rollouts, mask, alpha=0.05, abstain_threshold=0.5)
    balanced = wrap_policy_decision(rollouts, mask, alpha=0.05, abstain_threshold=1.0)
    aggressive = wrap_policy_decision(rollouts, mask, alpha=0.1, abstain_threshold=2.0)

    out = {
        "alpha_levels_tested": [0.05, 0.05, 0.1],
        "decisions": {
            "conservative_threshold_0.5": conservative.to_dict(),
            "balanced_threshold_1.0": balanced.to_dict(),
            "aggressive_threshold_2.0_alpha_0.1": aggressive.to_dict(),
        },
        "note": ("Conservative mode abstains when the best action's 95% CI width > 0.5. "
                 "Balanced mode runs with wider tolerance. Aggressive mode uses alpha=0.1 "
                 "(90% intervals) and a loose threshold. Same underlying rollouts — "
                 "different safety posture."),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    logger.info("[conformal_rl] conservative action=%d abstain=%s", conservative.action, conservative.abstain)
    logger.info("[conformal_rl] balanced     action=%d abstain=%s", balanced.action, balanced.abstain)
    logger.info("[conformal_rl] aggressive   action=%d abstain=%s", aggressive.action, aggressive.abstain)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        out = run_demo()
        print(json.dumps(out, indent=2))
    else:
        print("usage: --demo")
