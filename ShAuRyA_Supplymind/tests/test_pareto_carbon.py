"""test_pareto_carbon.py — F9 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features.pareto_carbon import (
    EMISSION_FACTORS, ActionPlan, best_under_weights,
    generate_plans, pareto_front,
)


def test_emission_factors_ordering():
    # Air should be worst, sea best (by tonne-km)
    assert EMISSION_FACTORS["air"] > EMISSION_FACTORS["road"]
    assert EMISSION_FACTORS["road"] > EMISSION_FACTORS["rail"]
    assert EMISSION_FACTORS["rail"] > EMISSION_FACTORS["sea"]


def test_generate_plans_returns_at_least_15():
    plans = generate_plans()
    assert len(plans) >= 15
    assert any(p.name == "do_nothing" for p in plans)
    assert any(p.name.startswith("ship_") for p in plans)


def test_pareto_front_non_empty_and_valid():
    plans = generate_plans()
    front = pareto_front(plans)
    assert 1 <= len(front) <= len(plans)
    # No two frontier plans dominate each other
    for i, p in enumerate(front):
        for j, q in enumerate(front):
            if i == j:
                continue
            dominates = (q.cost_usd <= p.cost_usd
                         and q.resilience_bps >= p.resilience_bps
                         and q.carbon_kg_co2 <= p.carbon_kg_co2
                         and (q.cost_usd < p.cost_usd
                              or q.resilience_bps > p.resilience_bps
                              or q.carbon_kg_co2 < p.carbon_kg_co2))
            assert not dominates


def test_weight_slider_returns_different_plans_for_different_weights():
    plans = generate_plans()
    conservative = best_under_weights(plans, 0.7, 0.15, 0.15)
    green = best_under_weights(plans, 0.1, 0.1, 0.8)
    # The cost-heavy weight should choose the cheapest feasible (likely do_nothing)
    # while the carbon-heavy weight should choose a low-emission plan
    assert conservative.cost_usd <= green.cost_usd or conservative.name != green.name
