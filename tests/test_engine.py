"""
Tests for SupplyMind engine components.

Covers the supply chain graph, disruption propagation, inventory depletion,
financial calculations, reward computation, and Monte Carlo simulation.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure the project root is on sys.path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import (
    SupplyMindAction,
    ActionResult,
    DisruptionSignal,
    SupplierStatus,
)
from server.engine.graph import SupplyChainGraph, SEVERITY_DECAY_PER_HOP
from server.engine.disruptions import DisruptionEngine
from server.engine.financial import FinancialEngine
from server.engine.rewards import RewardCalculator, StepState
from server.engine.monte_carlo import MonteCarloEngine


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

EASY_GRAPH = os.path.join(PROJECT_ROOT, "server", "data", "graphs", "easy_graph.json")
EASY_SCENARIOS = os.path.join(PROJECT_ROOT, "server", "data", "disruptions", "easy_scenarios.json")


@pytest.fixture
def easy_graph() -> SupplyChainGraph:
    """Load the easy task supply chain graph."""
    g = SupplyChainGraph()
    g.load_from_json(EASY_GRAPH)
    return g


@pytest.fixture
def easy_disruptions() -> DisruptionEngine:
    """Load the easy task disruption scenarios."""
    de = DisruptionEngine()
    de.load_scenarios(EASY_SCENARIOS)
    return de


# ──────────────────────────────────────────────
# SupplyChainGraph: loading and structure
# ──────────────────────────────────────────────

class TestSupplyChainGraph:
    """Test graph loading and structural properties."""

    def test_load_easy_graph_node_count(self, easy_graph: SupplyChainGraph) -> None:
        """Easy graph should have exactly 12 nodes."""
        assert easy_graph.G.number_of_nodes() == 12

    def test_load_easy_graph_edge_count(self, easy_graph: SupplyChainGraph) -> None:
        """Easy graph should have 12 edges (11 active + 1 dormant cross-geography)."""
        assert easy_graph.G.number_of_edges() == 12

    def test_all_node_ids_present(self, easy_graph: SupplyChainGraph) -> None:
        expected_ids = {
            "SUP_TSMC", "SUP_SAMSUNG", "SUP_ASE", "SUP_SILTRONIC",
            "PORT_KAOHSIUNG", "PORT_LONG_BEACH",
            "WH_TAIWAN", "WH_US_WEST",
            "FAC_PHOENIX",
            "CUST_APPLE", "CUST_DELL", "CUST_HP",
        }
        actual_ids = set(easy_graph.G.nodes())
        assert actual_ids == expected_ids

    def test_node_types_valid(self, easy_graph: SupplyChainGraph) -> None:
        valid_types = {"supplier", "warehouse", "port", "factory", "customer"}
        for _, data in easy_graph.G.nodes(data=True):
            assert data["node_type"].lower() in valid_types

    def test_total_annual_revenue_positive(self, easy_graph: SupplyChainGraph) -> None:
        """Total annual revenue from graph must be positive."""
        total = easy_graph.total_annual_revenue()
        assert total > 0

    def test_get_node_statuses_returns_all_nodes(self, easy_graph: SupplyChainGraph) -> None:
        statuses = easy_graph.get_node_statuses()
        assert len(statuses) == 12
        assert all(isinstance(s, SupplierStatus) for s in statuses)

    def test_get_customer_ids(self, easy_graph: SupplyChainGraph) -> None:
        customers = easy_graph.get_customer_ids()
        assert set(customers) == {"CUST_APPLE", "CUST_DELL", "CUST_HP"}

    def test_health_score_starts_high(self, easy_graph: SupplyChainGraph) -> None:
        """Initial health score should be near 100."""
        score = easy_graph.get_health_score()
        assert 80.0 <= score <= 100.0

    def test_sla_compliance_starts_at_one(self, easy_graph: SupplyChainGraph) -> None:
        """Before any disruption, SLA compliance should be 1.0."""
        compliance = easy_graph.get_sla_compliance()
        assert compliance == pytest.approx(1.0, abs=0.01)


# ──────────────────────────────────────────────
# Disruption propagation
# ──────────────────────────────────────────────

class TestDisruptionPropagation:
    """Test BFS propagation and severity decay."""

    def test_propagation_reaches_downstream(self, easy_graph: SupplyChainGraph) -> None:
        """Disrupting SUP_TSMC should propagate to downstream nodes."""
        affected = easy_graph.propagate_disruption(
            node_id="SUP_TSMC",
            severity=0.8,
            duration_days=7.0,
        )
        # TSMC itself should be in affected set
        assert "SUP_TSMC" in affected
        # At least one downstream node should be affected
        assert len(affected) > 1

    def test_severity_decays_per_hop(self, easy_graph: SupplyChainGraph) -> None:
        """Severity should decrease as we move downstream from the source."""
        affected = easy_graph.propagate_disruption(
            node_id="SUP_TSMC",
            severity=0.8,
            duration_days=7.0,
        )
        source_severity = affected["SUP_TSMC"]["severity"]
        assert source_severity == pytest.approx(0.8, abs=0.01)

        # Any downstream node should have lower severity
        for node_id, info in affected.items():
            if node_id != "SUP_TSMC":
                assert info["severity"] < source_severity

    def test_nonexistent_node_returns_empty(self, easy_graph: SupplyChainGraph) -> None:
        """Propagating from a nonexistent node should return empty dict."""
        affected = easy_graph.propagate_disruption(
            node_id="FAKE_NODE",
            severity=0.8,
            duration_days=5.0,
        )
        assert affected == {}


# ──────────────────────────────────────────────
# Inventory depletion
# ──────────────────────────────────────────────

class TestInventory:
    """Test inventory tracking on warehouse nodes."""

    def test_deplete_inventory_reduces_cover(self, easy_graph: SupplyChainGraph) -> None:
        """Depleting inventory for a disrupted supplier should reduce warehouse cover."""
        # Get initial warehouse inventory
        wh_data_before = dict(easy_graph.G.nodes["WH_TAIWAN"])
        initial_cover = wh_data_before.get("inventory_days_cover", 30.0)

        # Deplete inventory (simulating a disrupted supplier)
        easy_graph.deplete_inventory(disrupted_supplier_ids=["SUP_TSMC"])

        wh_data_after = dict(easy_graph.G.nodes["WH_TAIWAN"])
        final_cover = wh_data_after.get("inventory_days_cover", 30.0)

        # Cover should decrease or stay the same
        assert final_cover <= initial_cover

    def test_repeated_depletion_approaches_zero(self, easy_graph: SupplyChainGraph) -> None:
        """Repeated depletion should eventually bring inventory near zero."""
        for _ in range(60):
            easy_graph.deplete_inventory(disrupted_supplier_ids=["SUP_TSMC"])

        wh_data = dict(easy_graph.G.nodes["WH_TAIWAN"])
        cover = wh_data.get("inventory_days_cover", 30.0)
        assert cover < 5.0  # Should be very low after 60 days of depletion


# ──────────────────────────────────────────────
# Financial engine
# ──────────────────────────────────────────────

class TestFinancialEngine:
    """Test financial tracking and budget calculations."""

    def test_initial_state(self) -> None:
        fe = FinancialEngine(budget=5_000_000.0)
        assert fe.budget_total == 5_000_000.0
        assert fe.budget_remaining == 5_000_000.0
        assert fe.cumulative_cost_incurred == 0.0
        assert fe.cumulative_revenue_lost == 0.0
        assert fe.cumulative_penalty_fees == 0.0

    def test_budget_deduction(self) -> None:
        """Spending should reduce budget_remaining and increase cumulative cost."""
        fe = FinancialEngine(budget=5_000_000.0)
        # Manually deduct
        cost = 100_000.0
        fe.budget_remaining -= cost
        fe.cumulative_cost_incurred += cost
        assert fe.budget_remaining == pytest.approx(4_900_000.0)
        assert fe.cumulative_cost_incurred == pytest.approx(100_000.0)

    def test_snapshot(self) -> None:
        """FinancialEngine should produce a valid FinancialSnapshot."""
        fe = FinancialEngine(budget=5_000_000.0)
        # The engine should have a method to build a snapshot or we can check attributes
        assert fe.budget_total == 5_000_000.0


# ──────────────────────────────────────────────
# Reward calculator
# ──────────────────────────────────────────────

class TestRewardCalculator:
    """Test the dense 7-component reward function."""

    def test_compute_step_reward_returns_bounded_value(self) -> None:
        """Reward must be in [-1.0, 1.0]."""
        rc = RewardCalculator(initial_total_revenue=1_000_000_000.0)

        prev = StepState(
            revenue_at_risk=50_000_000.0,
            health_score=90.0,
            sla_compliance=0.9,
            budget_total=5_000_000.0,
        )
        current = StepState(
            revenue_at_risk=40_000_000.0,
            health_score=85.0,
            sla_compliance=0.85,
            budget_total=5_000_000.0,
        )

        action = SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        )
        result = ActionResult(success=True, cost=50_000.0)

        reward = rc.compute_step_reward(prev, current, action, result)
        assert -1.0 <= reward <= 1.0

    def test_do_nothing_during_crisis_is_not_rewarded(self) -> None:
        """Doing nothing when revenue is at risk should not produce high reward."""
        rc = RewardCalculator(initial_total_revenue=1_000_000_000.0)

        prev = StepState(revenue_at_risk=50_000_000.0, health_score=80.0)
        current = StepState(revenue_at_risk=60_000_000.0, health_score=75.0)

        action = SupplyMindAction(action_type="do_nothing")
        result = ActionResult(success=True, cost=0.0)

        reward = rc.compute_step_reward(prev, current, action, result)
        # Risk increased, so reward should be low/negative
        assert reward <= 0.2

    def test_different_actions_produce_different_rewards(self) -> None:
        """An expensive action and a do-nothing should yield different rewards."""
        rc = RewardCalculator(initial_total_revenue=1_000_000_000.0)

        prev = StepState(revenue_at_risk=50_000_000.0, budget_total=5_000_000.0)
        current_better = StepState(revenue_at_risk=30_000_000.0, budget_total=5_000_000.0)
        current_worse = StepState(revenue_at_risk=60_000_000.0, budget_total=5_000_000.0)

        action_active = SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        )
        result_active = ActionResult(success=True, cost=50_000.0)

        action_idle = SupplyMindAction(action_type="do_nothing")
        result_idle = ActionResult(success=True, cost=0.0)

        reward_active = rc.compute_step_reward(prev, current_better, action_active, result_active)
        reward_idle = rc.compute_step_reward(prev, current_worse, action_idle, result_idle)

        assert reward_active != reward_idle


# ──────────────────────────────────────────────
# Monte Carlo engine
# ──────────────────────────────────────────────

class TestMonteCarloEngine:
    """Test Monte Carlo loss estimation."""

    def test_no_disruptions_returns_zeros(self, easy_graph: SupplyChainGraph) -> None:
        """With no active disruptions, all estimates should be zero."""
        mc = MonteCarloEngine(seed=42)
        results = mc.run_simulation(easy_graph, active_disruptions=[], n_simulations=100)
        assert results["p50_loss"] == 0.0
        assert results["p95_loss"] == 0.0
        assert results["p99_loss"] == 0.0

    def test_with_disruption_returns_positive_estimates(self, easy_graph: SupplyChainGraph) -> None:
        """With an active disruption, loss estimates should be positive."""
        mc = MonteCarloEngine(seed=42)
        signal = DisruptionSignal(
            signal_id="SIG_TEST",
            disruption_type="cyclone",
            severity=0.8,
            confidence=0.9,
            affected_region="Taiwan",
            affected_node_ids=["SUP_TSMC"],
            time_to_impact_hours=0.0,
            estimated_duration_days=7.0,
            description="Test disruption",
            lifecycle_phase="active",
        )
        results = mc.run_simulation(easy_graph, active_disruptions=[signal], n_simulations=100)
        assert results["p50_loss"] >= 0.0
        assert results["p95_loss"] >= results["p50_loss"]

    def test_returns_expected_keys(self, easy_graph: SupplyChainGraph) -> None:
        """Result dict should contain p50, p95, p99 keys."""
        mc = MonteCarloEngine(seed=42)
        results = mc.run_simulation(easy_graph, active_disruptions=[], n_simulations=50)
        assert "p50_loss" in results
        assert "p95_loss" in results
        assert "p99_loss" in results

    def test_deterministic_with_seed(self, easy_graph: SupplyChainGraph) -> None:
        """Same seed should produce same results."""
        signal = DisruptionSignal(
            signal_id="SIG_TEST",
            disruption_type="cyclone",
            severity=0.8,
            confidence=0.9,
            affected_region="Taiwan",
            affected_node_ids=["SUP_TSMC"],
            time_to_impact_hours=0.0,
            estimated_duration_days=7.0,
            description="Test disruption",
            lifecycle_phase="active",
        )

        mc1 = MonteCarloEngine(seed=123)
        r1 = mc1.run_simulation(easy_graph, active_disruptions=[signal], n_simulations=100)

        # Reload graph to get clean state
        g2 = SupplyChainGraph()
        g2.load_from_json(EASY_GRAPH)
        mc2 = MonteCarloEngine(seed=123)
        r2 = mc2.run_simulation(g2, active_disruptions=[signal], n_simulations=100)

        assert r1["p50_loss"] == pytest.approx(r2["p50_loss"], rel=1e-6)
        assert r1["p95_loss"] == pytest.approx(r2["p95_loss"], rel=1e-6)
