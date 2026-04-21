"""test_counterfactual_explainer.py — F3 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features import counterfactual_explainer as ce


def test_template_counterfactual_structure():
    cf = ce._template_counterfactual(
        state={"severity": 0.6, "duration_days": 14, "scenario_text": "Red Sea Houthi"},
        action={"action_type": "reroute_shipment", "via": ["cape"]},
    )
    assert cf.no_action_delta_usd > 0
    assert cf.opposite_action_delta_usd >= 0
    assert cf.opposite_action_delta_usd < cf.no_action_delta_usd, \
        "opposite-of-reroute should save less than full reroute"
    assert "reroute" in cf.rationale.lower()


def test_explain_counterfactual_uses_cache_second_call():
    state = {"severity": 0.5, "duration_days": 20, "scenario_text": "test"}
    action = {"action_type": "hedge_commodity", "commodity": "oil", "hedge_amount_usd": 1_000_000}
    # First call (no LLM -> template)
    cf1 = ce.explain_counterfactual(state, action, use_cache=True, use_llm=False)
    assert cf1.source in ("template", "cache")
    # Second call (should hit cache)
    cf2 = ce.explain_counterfactual(state, action, use_cache=True, use_llm=False)
    assert cf2.source == "cache"
    assert cf2.no_action_delta_usd == cf1.no_action_delta_usd


def test_six_demo_scenarios_defined():
    assert len(ce.DEMO_SCENARIOS) >= 6
    for sc in ce.DEMO_SCENARIOS:
        assert "state" in sc and "action" in sc and "name" in sc
        assert "action_type" in sc["action"]


def test_action_save_factors_ordering():
    """reroute should save more than issue_supplier_alert, do_nothing saves nothing."""
    sf = ce._template_counterfactual.__code__
    # Inspect via template runs
    state = {"severity": 0.7, "duration_days": 30}
    cf_reroute = ce._template_counterfactual(state, {"action_type": "reroute_shipment"})
    cf_alert = ce._template_counterfactual(state, {"action_type": "issue_supplier_alert"})
    cf_nothing = ce._template_counterfactual(state, {"action_type": "do_nothing"})
    # Reroute (save 60%) -> opposite loss 40% of base, smaller than alert (opp 95%) and nothing (100%)
    assert cf_reroute.opposite_action_delta_usd < cf_alert.opposite_action_delta_usd
    assert cf_alert.opposite_action_delta_usd < cf_nothing.opposite_action_delta_usd
