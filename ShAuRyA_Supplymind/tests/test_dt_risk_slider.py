"""test_dt_risk_slider.py — G6+F4 regression."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features.dt_risk_slider import (
    ACTION_TYPES, SLIDER_POSITIONS, SliderPolicy, benchmark_slider,
)


def test_slider_positions_are_well_defined():
    assert set(SLIDER_POSITIONS) == {"conservative", "balanced", "aggressive"}
    for pos, cfg in SLIDER_POSITIONS.items():
        assert 0.0 <= cfg["target_return"] <= 1.0
        assert len(cfg["preferred_action_types"]) >= 2


def test_slider_policy_respects_action_mask():
    policy = SliderPolicy("balanced", seed=0)
    obs = np.zeros(408, dtype=np.float32)
    mask = np.zeros(280, dtype=bool)
    mask[100:110] = True   # only 10 actions valid
    for _ in range(5):
        a = policy.act(obs, mask)
        assert 100 <= a < 110, "policy must pick from masked subset only"


def test_aggressive_prefers_hedge_and_backup():
    policy = SliderPolicy("aggressive", seed=1)
    obs = np.zeros(408, dtype=np.float32)
    mask = np.ones(280, dtype=bool)
    counts = {at: 0 for at in ACTION_TYPES}
    for _ in range(400):
        a = policy.act(obs, mask)
        counts[ACTION_TYPES[a // 40]] += 1
    # Aggressive should pick backup, hedge, reroute, expedite most often
    preferred = {"activate_backup_supplier", "hedge_commodity", "reroute_shipment", "expedite_order"}
    non_preferred = {"do_nothing", "issue_supplier_alert"}
    pref_total = sum(counts[a] for a in preferred)
    non_pref_total = sum(counts[a] for a in non_preferred)
    assert pref_total > non_pref_total


def test_benchmark_quick_path_completes():
    # Runs 1 task x 1 seed x 3 slider positions = 3 rollouts; fast
    out = benchmark_slider(tasks=("easy_typhoon_response",), seeds=(42,))
    assert set(out["summary_by_position"]) == {"conservative", "balanced", "aggressive"}
    for pos, s in out["summary_by_position"].items():
        assert s["n_rollouts"] == 1
        assert 0.0 <= s["mean_return"] <= 1.0
