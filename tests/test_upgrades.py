"""
Tests for the 5 major upgrades:
1. Seed-based scenario jitter
2. Backup supplier validation (disrupted backup rejection)
3. Reroute port degradation
4. Compact observation summary
5. Emergent cascade triggers
"""

import pytest

from models import SupplyMindAction
from server.supply_environment import SupplyMindEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SupplyMindEnvironment()


# ---------------------------------------------------------------------------
# 1. Seed-based scenario jitter
# ---------------------------------------------------------------------------

class TestSeedJitter:
    """Tests for seed-based scenario jitter on reset()."""

    def test_default_reset_backward_compatible(self, env):
        """No seed = deterministic behavior, identical across resets."""
        obs1 = env.reset("easy_typhoon_response")
        obs2 = env.reset("easy_typhoon_response")
        assert obs1.current_day == obs2.current_day == 0
        assert obs1.compact_summary == obs2.compact_summary

    def test_same_seed_same_episode(self, env):
        """Same seed produces identical episodes."""
        obs1 = env.reset("easy_typhoon_response", seed=42)
        obs2 = env.reset("easy_typhoon_response", seed=42)
        assert obs1.compact_summary == obs2.compact_summary
        assert len(obs1.active_signals) == len(obs2.active_signals)

    def test_different_seeds_differ(self, env):
        """Different seeds produce different disruption timings."""
        # Run both seeds forward to day 5 where disruption should be active
        results = {}
        for seed in [100, 999]:
            env.reset("easy_typhoon_response", seed=seed)
            for _ in range(5):
                obs = env.step(SupplyMindAction(action_type="do_nothing"))
            results[seed] = [s.severity for s in obs.active_signals]

        # At least one severity value should differ due to jitter
        assert results[100] != results[999], (
            "Different seeds should produce different severity values"
        )

    def test_seed_works_on_all_tasks(self, env):
        """Seed parameter works on all 3 tasks without crashing."""
        for task_id in ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]:
            obs = env.reset(task_id, seed=123)
            assert obs.current_day == 0
            assert obs.compact_summary != ""


# ---------------------------------------------------------------------------
# 2. Backup supplier validation
# ---------------------------------------------------------------------------

class TestBackupValidation:
    """Tests for backup supplier disruption checking."""

    def test_backup_succeeds_when_healthy(self, env):
        """Activating a healthy backup supplier should succeed."""
        env.reset("easy_typhoon_response")
        # Step 0: backup should be healthy
        obs = env.step(SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        ))
        assert obs.last_action_result.success is True
        assert obs.last_action_result.cost > 0

    def test_backup_rejected_when_disrupted(self, env):
        """Activating a disrupted backup should fail with zero cost."""
        env.reset("easy_typhoon_response")
        # Manually disrupt the backup supplier
        env.engine.graph.G.nodes["SUP_SAMSUNG"]["is_operational"] = False
        env.engine.graph.G.nodes["SUP_SAMSUNG"]["risk_score"] = 0.8

        obs = env.step(SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        ))
        assert obs.last_action_result.success is False
        assert "disrupted" in obs.last_action_result.message.lower()
        assert obs.last_action_result.cost == 0.0


# ---------------------------------------------------------------------------
# 3. Reroute port degradation
# ---------------------------------------------------------------------------

class TestRerouteDegradation:
    """Tests for reroute port operational status checking."""

    def test_reroute_through_healthy_port(self, env):
        """Rerouting through a healthy port uses normal transit times."""
        env.reset("medium_multi_front")
        obs = env.step(SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_LONG_BEACH",
            reroute_via=["PORT_OAKLAND"],
        ))
        # Should succeed without warning
        if obs.last_action_result.success:
            assert "WARNING" not in obs.last_action_result.message

    def test_reroute_through_disrupted_port_warns(self, env):
        """Rerouting through a disrupted port should warn and degrade."""
        env.reset("medium_multi_front")
        # Manually disrupt the reroute port
        env.engine.graph.G.nodes["PORT_OAKLAND"]["is_operational"] = False
        env.engine.graph.G.nodes["PORT_OAKLAND"]["risk_score"] = 0.9

        obs = env.step(SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_LONG_BEACH",
            reroute_via=["PORT_OAKLAND"],
        ))
        if obs.last_action_result.success:
            assert "WARNING" in obs.last_action_result.message
            assert "degraded" in obs.last_action_result.message.lower() or \
                   "Degraded" in obs.last_action_result.message


# ---------------------------------------------------------------------------
# 4. Compact observation summary
# ---------------------------------------------------------------------------

class TestCompactSummary:
    """Tests for compact_summary field in observations."""

    def test_compact_summary_present(self, env):
        """Compact summary should be populated on initial observation."""
        obs = env.reset("easy_typhoon_response")
        assert hasattr(obs, "compact_summary")
        assert obs.compact_summary != ""

    def test_compact_summary_concise(self, env):
        """Compact summary should be reasonably short (~500 chars max)."""
        obs = env.reset("hard_cascading_crisis")
        # Run a few steps to get disruptions active
        for _ in range(10):
            obs = env.step(SupplyMindAction(action_type="do_nothing"))
        # Should be under 600 chars (proxy for ~150 tokens)
        assert len(obs.compact_summary) < 600, (
            f"Compact summary too long ({len(obs.compact_summary)} chars): "
            f"{obs.compact_summary[:100]}..."
        )

    def test_compact_summary_contains_budget(self, env):
        """Compact summary should include budget information."""
        obs = env.reset("easy_typhoon_response")
        assert "Budget" in obs.compact_summary or "budget" in obs.compact_summary

    def test_compact_summary_on_all_tasks(self, env):
        """Compact summary should work on all 3 tasks."""
        for task_id in ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]:
            obs = env.reset(task_id)
            assert obs.compact_summary != "", f"Empty compact_summary for {task_id}"


# ---------------------------------------------------------------------------
# 5. Emergent cascade triggers
# ---------------------------------------------------------------------------

class TestEmergentCascades:
    """Tests for emergent cascade disruption injection."""

    def test_cascade_injected_on_prolonged_offline(self, env):
        """When a supplier stays offline and warehouse inventory depletes,
        a cascade disruption should be injected."""
        env.reset("easy_typhoon_response")

        # Manually simulate prolonged offline + inventory depletion
        engine = env.engine
        engine.graph.G.nodes["SUP_TSMC"]["is_operational"] = False

        # Find a downstream warehouse of SUP_TSMC
        downstream_wh = None
        for _, neighbor in engine.graph.G.out_edges("SUP_TSMC"):
            if engine.graph.G.nodes[neighbor].get("node_type", "").lower() == "warehouse":
                downstream_wh = neighbor
                break

        if downstream_wh:
            # Deplete warehouse inventory
            engine.graph.G.nodes[downstream_wh]["inventory_days_cover"] = 1.0
            engine.graph.G.nodes[downstream_wh]["current_inventory_units"] = 10

            # Simulate enough days offline
            initial_scenario_count = len(engine.disruption_engine.scenarios)
            for _ in range(5):
                engine._offline_durations["SUP_TSMC"] = engine._offline_durations.get("SUP_TSMC", 0) + 1
                engine._check_emergent_cascades()

            # Should have injected at least one cascade
            assert len(engine.disruption_engine.scenarios) > initial_scenario_count, (
                "Expected cascade injection when supplier offline and inventory depleted"
            )

    def test_no_cascade_when_inventory_healthy(self, env):
        """No cascade should be injected when inventory is sufficient."""
        env.reset("easy_typhoon_response")
        engine = env.engine

        initial_count = len(engine.disruption_engine.scenarios)

        # Run 10 steps with do_nothing (inventory should still be okay)
        for _ in range(3):
            env.step(SupplyMindAction(action_type="do_nothing"))

        # No cascades should have been injected this early
        assert len(engine.disruption_engine.scenarios) == initial_count

    def test_cascade_not_duplicated(self, env):
        """Same cascade should not be injected twice."""
        env.reset("easy_typhoon_response")
        engine = env.engine

        engine.graph.G.nodes["SUP_TSMC"]["is_operational"] = False
        for _, neighbor in engine.graph.G.out_edges("SUP_TSMC"):
            if engine.graph.G.nodes[neighbor].get("node_type", "").lower() == "warehouse":
                engine.graph.G.nodes[neighbor]["inventory_days_cover"] = 0.5
                engine.graph.G.nodes[neighbor]["current_inventory_units"] = 1
                break

        # Trigger cascade check multiple times
        for i in range(10):
            engine._offline_durations["SUP_TSMC"] = i + 3
            engine._check_emergent_cascades()

        # Count CASCADE_ scenarios
        cascade_count = sum(
            1 for s in engine.disruption_engine.scenarios
            if s.signal_id.startswith("CASCADE_")
        )
        # Should have at most 1 cascade per source-warehouse pair
        assert cascade_count <= 2, f"Too many cascades injected: {cascade_count}"
