"""
Tests for SupplyMind Pydantic models.

Validates serialization, deserialization, field validation, and default values
for all core models in the agent-environment contract.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    DisruptionSignal,
    SupplierStatus,
    FinancialSnapshot,
    ActionResult,
    SupplyMindAction,
    SupplyMindObservation,
    SupplyMindState,
)


# ──────────────────────────────────────────────
# SupplyMindAction: all 7 action types
# ──────────────────────────────────────────────

class TestSupplyMindAction:
    """Test all 7 action types with correct parameters."""

    def test_do_nothing(self) -> None:
        action = SupplyMindAction(action_type="do_nothing")
        assert action.action_type == "do_nothing"
        assert action.target_node_id is None
        data = action.model_dump()
        assert data["action_type"] == "do_nothing"

    def test_activate_backup_supplier(self) -> None:
        action = SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        )
        assert action.action_type == "activate_backup_supplier"
        assert action.target_node_id == "SUP_TSMC"
        assert action.backup_supplier_id == "SUP_SAMSUNG"

    def test_reroute_shipment(self) -> None:
        action = SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_KAOHSIUNG",
            reroute_via=["PORT_LONG_BEACH", "PORT_OAKLAND"],
        )
        assert action.reroute_via == ["PORT_LONG_BEACH", "PORT_OAKLAND"]

    def test_increase_safety_stock(self) -> None:
        action = SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id="WH_US_WEST",
            additional_stock_days=14,
        )
        assert action.additional_stock_days == 14

    def test_increase_safety_stock_validation_bounds(self) -> None:
        """additional_stock_days must be in [1, 90]."""
        with pytest.raises(ValidationError):
            SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id="WH_US_WEST",
                additional_stock_days=0,
            )
        with pytest.raises(ValidationError):
            SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id="WH_US_WEST",
                additional_stock_days=91,
            )

    def test_expedite_order(self) -> None:
        action = SupplyMindAction(
            action_type="expedite_order",
            target_node_id="SUP_TSMC",
            expedite_mode="air",
        )
        assert action.expedite_mode == "air"

    def test_expedite_order_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            SupplyMindAction(
                action_type="expedite_order",
                target_node_id="SUP_TSMC",
                expedite_mode="teleport",
            )

    def test_hedge_commodity(self) -> None:
        action = SupplyMindAction(
            action_type="hedge_commodity",
            commodity="semiconductors",
            hedge_amount_usd=500_000.0,
        )
        assert action.commodity == "semiconductors"
        assert action.hedge_amount_usd == 500_000.0

    def test_hedge_commodity_amount_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SupplyMindAction(
                action_type="hedge_commodity",
                commodity="rare_earths",
                hedge_amount_usd=-100.0,
            )

    def test_issue_supplier_alert(self) -> None:
        action = SupplyMindAction(
            action_type="issue_supplier_alert",
            target_node_id="SUP_TSMC",
        )
        assert action.action_type == "issue_supplier_alert"

    def test_invalid_action_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SupplyMindAction(action_type="fly_to_moon")

    def test_round_trip_serialization(self) -> None:
        """model_dump -> parse should produce identical object."""
        action = SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_KAOHSIUNG",
            reroute_via=["PORT_LONG_BEACH"],
        )
        data = action.model_dump()
        restored = SupplyMindAction.model_validate(data)
        assert restored == action


# ──────────────────────────────────────────────
# DisruptionSignal
# ──────────────────────────────────────────────

class TestDisruptionSignal:
    """Test DisruptionSignal field ranges and validation."""

    def _make_signal(self, **overrides) -> DisruptionSignal:
        defaults = dict(
            signal_id="SIG_001",
            disruption_type="cyclone",
            severity=0.7,
            confidence=0.85,
            affected_region="Taiwan",
            affected_node_ids=["SUP_TSMC"],
            time_to_impact_hours=72.0,
            estimated_duration_days=10.0,
            description="Category 3 typhoon approaching Taiwan",
        )
        defaults.update(overrides)
        return DisruptionSignal(**defaults)

    def test_valid_signal(self) -> None:
        sig = self._make_signal()
        assert sig.severity == 0.7
        assert sig.confidence == 0.85
        assert sig.lifecycle_phase == "warning"

    def test_severity_range_lower_bound(self) -> None:
        sig = self._make_signal(severity=0.0)
        assert sig.severity == 0.0

    def test_severity_range_upper_bound(self) -> None:
        sig = self._make_signal(severity=1.0)
        assert sig.severity == 1.0

    def test_severity_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._make_signal(severity=1.5)
        with pytest.raises(ValidationError):
            self._make_signal(severity=-0.1)

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._make_signal(confidence=2.0)

    def test_lifecycle_phase_default(self) -> None:
        sig = self._make_signal()
        assert sig.lifecycle_phase == "warning"

    def test_lifecycle_phase_custom(self) -> None:
        sig = self._make_signal(lifecycle_phase="active")
        assert sig.lifecycle_phase == "active"

    def test_round_trip(self) -> None:
        sig = self._make_signal()
        data = sig.model_dump()
        restored = DisruptionSignal.model_validate(data)
        assert restored == sig


# ──────────────────────────────────────────────
# FinancialSnapshot
# ──────────────────────────────────────────────

class TestFinancialSnapshot:
    """Test FinancialSnapshot defaults and field validation."""

    def test_required_fields(self) -> None:
        snap = FinancialSnapshot(budget_remaining=5_000_000, budget_total=5_000_000)
        assert snap.budget_remaining == 5_000_000
        assert snap.budget_total == 5_000_000

    def test_default_values(self) -> None:
        snap = FinancialSnapshot(budget_remaining=1_000_000, budget_total=5_000_000)
        assert snap.total_revenue_at_risk == 0.0
        assert snap.cumulative_cost_incurred == 0.0
        assert snap.cumulative_revenue_lost == 0.0
        assert snap.cumulative_penalty_fees == 0.0
        assert snap.supply_chain_health_score == 100.0
        assert snap.monte_carlo_p50_loss == 0.0
        assert snap.monte_carlo_p95_loss == 0.0
        assert snap.commodity_price_changes == {}

    def test_health_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            FinancialSnapshot(
                budget_remaining=1_000_000,
                budget_total=5_000_000,
                supply_chain_health_score=101.0,
            )

    def test_round_trip(self) -> None:
        snap = FinancialSnapshot(
            budget_remaining=3_000_000,
            budget_total=5_000_000,
            cumulative_cost_incurred=200_000,
            commodity_price_changes={"semiconductors": 1.3},
        )
        data = snap.model_dump()
        restored = FinancialSnapshot.model_validate(data)
        assert restored == snap


# ──────────────────────────────────────────────
# SupplyMindObservation
# ──────────────────────────────────────────────

class TestSupplyMindObservation:
    """Test observation model round-trip serialization."""

    def test_minimal_observation(self) -> None:
        obs = SupplyMindObservation(
            current_day=0,
            days_remaining=30,
        )
        assert obs.current_day == 0
        assert obs.days_remaining == 30
        assert obs.active_signals == []
        assert obs.done is False
        assert obs.reward == 0.0

    def test_round_trip_with_nested_models(self) -> None:
        signal = DisruptionSignal(
            signal_id="SIG_001",
            disruption_type="cyclone",
            severity=0.7,
            confidence=0.85,
            affected_region="Taiwan",
            affected_node_ids=["SUP_TSMC"],
            time_to_impact_hours=72.0,
            estimated_duration_days=10.0,
            description="Typhoon approaching",
        )
        node = SupplierStatus(
            node_id="SUP_TSMC",
            name="TSMC Fab 14",
            node_type="supplier",
            tier=1,
            country="TW",
            is_operational=True,
            current_risk_score=0.7,
        )
        obs = SupplyMindObservation(
            current_day=5,
            days_remaining=25,
            active_signals=[signal],
            new_signals=[signal],
            node_statuses=[node],
            financials=FinancialSnapshot(budget_remaining=5_000_000, budget_total=5_000_000),
            reward=0.15,
            done=False,
            situation_summary="Typhoon warning day 5",
        )
        data = obs.model_dump()
        restored = SupplyMindObservation.model_validate(data)
        assert restored.current_day == 5
        assert len(restored.active_signals) == 1
        assert restored.active_signals[0].signal_id == "SIG_001"
        assert len(restored.node_statuses) == 1
        assert restored.node_statuses[0].node_id == "SUP_TSMC"
        assert restored.financials.budget_total == 5_000_000
        assert restored.reward == 0.15

    def test_observation_with_action_result(self) -> None:
        obs = SupplyMindObservation(
            current_day=3,
            days_remaining=27,
            last_action_result=ActionResult(
                success=True,
                message="Backup supplier activated",
                cost=50_000.0,
                effect_description="Samsung now active",
            ),
        )
        assert obs.last_action_result is not None
        assert obs.last_action_result.success is True
        assert obs.last_action_result.cost == 50_000.0


# ──────────────────────────────────────────────
# SupplyMindState
# ──────────────────────────────────────────────

class TestSupplyMindState:
    """Test SupplyMindState field validation and defaults."""

    def test_defaults(self) -> None:
        state = SupplyMindState()
        assert state.episode_id == ""
        assert state.step_count == 0
        assert state.task_id == ""
        assert state.task_difficulty == ""
        assert state.total_steps == 0
        assert state.is_done is False
        assert state.cumulative_reward == 0.0

    def test_with_values(self) -> None:
        state = SupplyMindState(
            episode_id="ep-001",
            step_count=10,
            task_id="easy_typhoon_response",
            task_name="Typhoon Response",
            task_difficulty="easy",
            total_steps=30,
            is_done=False,
            cumulative_reward=1.5,
        )
        assert state.task_id == "easy_typhoon_response"
        assert state.total_steps == 30
        assert state.cumulative_reward == 1.5

    def test_round_trip(self) -> None:
        state = SupplyMindState(
            episode_id="ep-002",
            step_count=5,
            task_id="medium_multi_front",
            task_name="Multi-Front Crisis",
            task_difficulty="medium",
            total_steps=45,
        )
        data = state.model_dump()
        restored = SupplyMindState.model_validate(data)
        assert restored == state
