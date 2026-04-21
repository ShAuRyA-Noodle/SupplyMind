"""test_conformal_rl.py — F6 regression."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features.conformal_rl import (
    conformal_intervals_per_action, demo_synthetic_rollouts, run_demo,
    split_conformal_q_hat, wrap_policy_decision,
)


def test_q_hat_empty_returns_inf():
    assert split_conformal_q_hat(np.array([])) == float("inf")


def test_q_hat_monotone_with_sample_spread():
    tight = np.array([0.01, 0.02, -0.01, 0.005, -0.008])
    wide = np.array([1.0, -1.0, 0.5, -0.5, 0.8])
    assert split_conformal_q_hat(wide) > split_conformal_q_hat(tight)


def test_per_action_intervals_structure():
    rollouts = demo_synthetic_rollouts(n_actions=3, n_cal_per_action=20, seed=1)
    intervals = conformal_intervals_per_action(rollouts, alpha=0.05)
    assert len(intervals) == 3
    for a, v in intervals.items():
        assert v["lo"] <= v["mean"] <= v["hi"]
        assert v["n"] == 20


def test_action_mask_restricts_choice():
    rollouts = demo_synthetic_rollouts(seed=2)
    mask = np.array([False, True, False, True, False])
    decision = wrap_policy_decision(rollouts, action_mask=mask)
    assert decision.action in (1, 3), f"masked selection must pick valid action (1 or 3), got {decision.action}"


def test_abstain_flag_triggers_on_wide_interval():
    rollouts = {
        0: [0.0, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5, 0.5, -0.5, 2.5],   # wide
        1: [1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0, 1.01, 1.0, 0.99],  # tight
    }
    mask = np.array([True, True])
    res_tight = wrap_policy_decision(rollouts, mask, abstain_threshold=0.5)
    # tight action 1 chosen because higher mean + narrower band
    assert res_tight.action == 1
    assert res_tight.abstain is False

    # Force threshold below even the tight interval -> abstain
    res_force = wrap_policy_decision(rollouts, mask, abstain_threshold=0.01)
    assert res_force.abstain is True


def test_run_demo_end_to_end():
    out = run_demo()
    assert "decisions" in out
    for key, d in out["decisions"].items():
        assert "action" in d and "reward_p50" in d and "abstain" in d
