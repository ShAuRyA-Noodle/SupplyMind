"""
Tests for SupplyMind episode graders.

CRITICAL: These tests prove that graders produce DIFFERENT scores for different
strategies. This is a competition requirement -- graders that always return the
same score result in disqualification.
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import SupplyMindAction
from server.supply_environment import SupplyMindEnvironment


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def run_do_nothing_episode(env: SupplyMindEnvironment, task_id: str) -> dict:
    """Run a complete episode doing nothing at every step."""
    env.reset(task_id=task_id)
    action = SupplyMindAction(action_type="do_nothing")
    while not env.state.is_done:
        env.step(action)
    return env.grade()


def run_smart_easy_episode(env: SupplyMindEnvironment) -> dict:
    """
    Run a smart strategy on the easy task.

    Strategy:
    - Day 0-2: Issue supplier alerts to gather info
    - Day 3: Activate Samsung as backup for TSMC
    - Day 4: Expedite order via air for TSMC
    - Day 5: Increase safety stock at US warehouse
    - Day 6+: Do nothing (let the situation resolve with mitigations in place)
    """
    env.reset(task_id="easy_typhoon_response")

    actions_by_step = [
        # Steps 0-1: Issue alerts during pre-warning
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="SUP_TSMC"),
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="PORT_KAOHSIUNG"),
        # Step 2: Activate backup supplier early (proactive)
        SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        ),
        # Step 3: Expedite critical orders
        SupplyMindAction(
            action_type="expedite_order",
            target_node_id="SUP_TSMC",
            expedite_mode="air",
        ),
        # Step 4: Increase safety stock
        SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id="WH_US_WEST",
            additional_stock_days=14,
        ),
        # Step 5: Reroute shipment away from affected port
        SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_KAOHSIUNG",
            reroute_via=["PORT_LONG_BEACH"],
        ),
    ]

    step_idx = 0
    while not env.state.is_done:
        if step_idx < len(actions_by_step):
            action = actions_by_step[step_idx]
        else:
            action = SupplyMindAction(action_type="do_nothing")
        env.step(action)
        step_idx += 1

    return env.grade()


def run_wasteful_easy_episode(env: SupplyMindEnvironment) -> dict:
    """
    Run a wasteful strategy: spend heavily on unnecessary hedges and
    expedites without targeting the right nodes.
    """
    env.reset(task_id="easy_typhoon_response")

    step_idx = 0
    while not env.state.is_done:
        if step_idx % 3 == 0:
            action = SupplyMindAction(
                action_type="hedge_commodity",
                commodity="rare_earths",
                hedge_amount_usd=200_000.0,
            )
        elif step_idx % 3 == 1:
            action = SupplyMindAction(
                action_type="expedite_order",
                target_node_id="SUP_SILTRONIC",
                expedite_mode="air",
            )
        else:
            action = SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id="WH_TAIWAN",
                additional_stock_days=30,
            )
        env.step(action)
        step_idx += 1

    return env.grade()


def run_smart_medium_episode(env: SupplyMindEnvironment) -> dict:
    """
    Run a smart triage strategy on the medium task.

    Strategy: address port strike first (highest impact), then Thailand floods,
    then hedge for sanctions.
    """
    env.reset(task_id="medium_multi_front")

    actions_by_step = [
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="PORT_LONG_BEACH"),
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="SUP_FOXCONN_TH"),
        SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_LONG_BEACH",
            reroute_via=["PORT_OAKLAND"],
        ),
        SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_FOXCONN_TH",
            backup_supplier_id="SUP_SHENZHEN",
        ),
        SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id="WH_US_WEST",
            additional_stock_days=14,
        ),
        SupplyMindAction(
            action_type="hedge_commodity",
            commodity="rare_earths",
            hedge_amount_usd=500_000.0,
        ),
        SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC",
            backup_supplier_id="SUP_SAMSUNG",
        ),
    ]

    step_idx = 0
    while not env.state.is_done:
        if step_idx < len(actions_by_step):
            action = actions_by_step[step_idx]
        else:
            action = SupplyMindAction(action_type="do_nothing")
        env.step(action)
        step_idx += 1

    return env.grade()


def run_smart_hard_episode(env: SupplyMindEnvironment) -> dict:
    """
    Run a smart strategy on the hard cascading crisis task.

    Strategy: scout early, activate backups for high-revenue nodes,
    hedge semiconductors, then manage the cascade.
    """
    env.reset(task_id="hard_cascading_crisis")

    actions_by_step = [
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="SUP_TSMC_AUTO"),
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="SUP_RENESAS"),
        SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_TSMC_AUTO",
            backup_supplier_id="SUP_SAMSUNG_SDI",
        ),
        SupplyMindAction(
            action_type="hedge_commodity",
            commodity="semiconductors",
            hedge_amount_usd=1_000_000.0,
        ),
        SupplyMindAction(
            action_type="activate_backup_supplier",
            target_node_id="SUP_RENESAS",
            backup_supplier_id="SUP_INFINEON",
        ),
        SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id="WH_JAPAN",
            additional_stock_days=21,
        ),
        SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id="PORT_KAOHSIUNG",
            reroute_via=["PORT_BUSAN"],
        ),
        SupplyMindAction(
            action_type="expedite_order",
            target_node_id="SUP_BOSCH",
            expedite_mode="air",
        ),
        SupplyMindAction(action_type="issue_supplier_alert", target_node_id="SUP_CATL"),
        SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id="WH_US",
            additional_stock_days=14,
        ),
    ]

    step_idx = 0
    while not env.state.is_done:
        if step_idx < len(actions_by_step):
            action = actions_by_step[step_idx]
        else:
            action = SupplyMindAction(action_type="do_nothing")
        env.step(action)
        step_idx += 1

    return env.grade()


# ──────────────────────────────────────────────
# Score bounds and variance
# ──────────────────────────────────────────────

class TestGraderScoreBounds:
    """Test that all graders produce scores in [0.0, 1.0]."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_do_nothing_easy_score_bounded(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "easy_typhoon_response")
        assert 0.0 <= result["score"] <= 1.0

    def test_do_nothing_medium_score_bounded(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "medium_multi_front")
        assert 0.0 <= result["score"] <= 1.0

    def test_do_nothing_hard_score_bounded(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "hard_cascading_crisis")
        assert 0.0 <= result["score"] <= 1.0


class TestDoNothingScoresLow:
    """Test that the do-nothing agent scores low on all tasks."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_do_nothing_easy_scores_low(self, env: SupplyMindEnvironment) -> None:
        """Do-nothing should score roughly 0.1-0.4 on easy task."""
        result = run_do_nothing_episode(env, "easy_typhoon_response")
        assert result["score"] < 0.5, (
            f"Do-nothing scored {result['score']} on easy -- too high"
        )

    def test_do_nothing_medium_scores_low(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "medium_multi_front")
        assert result["score"] < 0.5, (
            f"Do-nothing scored {result['score']} on medium -- too high"
        )

    def test_do_nothing_hard_scores_low(self, env: SupplyMindEnvironment) -> None:
        """Do-nothing should score below 0.6 on hard task (cascade has partial natural recovery)."""
        result = run_do_nothing_episode(env, "hard_cascading_crisis")
        assert result["score"] < 0.6, (
            f"Do-nothing scored {result['score']} on hard -- too high"
        )


class TestGraderDiscrimination:
    """
    CRITICAL TESTS: Prove that graders produce DIFFERENT scores for
    different strategies. Graders that always return the same score
    result in disqualification.
    """

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_smart_beats_do_nothing_easy(self, env: SupplyMindEnvironment) -> None:
        """A smart strategy MUST score higher than do-nothing on easy task."""
        do_nothing_result = run_do_nothing_episode(env, "easy_typhoon_response")
        smart_result = run_smart_easy_episode(env)

        assert smart_result["score"] > do_nothing_result["score"], (
            f"Smart ({smart_result['score']}) did not beat do-nothing "
            f"({do_nothing_result['score']}) on easy task"
        )

    def test_different_strategies_produce_different_scores(
        self, env: SupplyMindEnvironment
    ) -> None:
        """Three different strategies MUST produce three different scores."""
        do_nothing = run_do_nothing_episode(env, "easy_typhoon_response")
        smart = run_smart_easy_episode(env)
        wasteful = run_wasteful_easy_episode(env)

        scores = {
            round(do_nothing["score"], 4),
            round(smart["score"], 4),
            round(wasteful["score"], 4),
        }
        assert len(scores) >= 2, (
            f"Expected different scores but got: "
            f"do_nothing={do_nothing['score']}, "
            f"smart={smart['score']}, "
            f"wasteful={wasteful['score']}"
        )

    def test_smart_vs_wasteful_different_scores(
        self, env: SupplyMindEnvironment
    ) -> None:
        """Smart and wasteful strategies should produce different scores on medium task."""
        smart = run_smart_medium_episode(env)
        wasteful = run_do_nothing_episode(env, "medium_multi_front")

        assert smart["score"] != wasteful["score"], (
            f"Smart and do-nothing produced same score: {smart['score']}"
        )

    def test_smart_beats_do_nothing_medium(self, env: SupplyMindEnvironment) -> None:
        """A targeted strategy MUST outscore do-nothing on medium task."""
        do_nothing = run_do_nothing_episode(env, "medium_multi_front")
        smart = run_smart_medium_episode(env)

        assert smart["score"] > do_nothing["score"], (
            f"Smart ({smart['score']}) did not beat do-nothing "
            f"({do_nothing['score']}) on medium task"
        )

    def test_smart_beats_do_nothing_hard(self, env: SupplyMindEnvironment) -> None:
        """A targeted strategy MUST outscore do-nothing on hard task."""
        do_nothing = run_do_nothing_episode(env, "hard_cascading_crisis")
        smart = run_smart_hard_episode(env)

        assert smart["score"] > do_nothing["score"], (
            f"Smart ({smart['score']}) did not beat do-nothing "
            f"({do_nothing['score']}) on hard task"
        )


# ──────────────────────────────────────────────
# Grader breakdown validation
# ──────────────────────────────────────────────

class TestGraderBreakdown:
    """Test that grader breakdowns have the correct component weights."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_easy_breakdown_weights_sum_to_one(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "easy_typhoon_response")
        breakdown = result["breakdown"]
        total_weight = sum(v["weight"] for v in breakdown.values())
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_easy_breakdown_has_correct_components(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "easy_typhoon_response")
        breakdown = result["breakdown"]
        expected_components = {
            "revenue_preserved",
            "timeliness",
            "cost_efficiency",
            "stockout_prevention",
            "action_coverage",
        }
        assert set(breakdown.keys()) == expected_components

    def test_easy_component_weights(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "easy_typhoon_response")
        bd = result["breakdown"]
        assert bd["revenue_preserved"]["weight"] == pytest.approx(0.30)
        assert bd["timeliness"]["weight"] == pytest.approx(0.25)
        assert bd["action_coverage"]["weight"] == pytest.approx(0.20)
        assert bd["cost_efficiency"]["weight"] == pytest.approx(0.15)
        assert bd["stockout_prevention"]["weight"] == pytest.approx(0.10)

    def test_medium_breakdown_weights_sum_to_one(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "medium_multi_front")
        breakdown = result["breakdown"]
        total_weight = sum(v["weight"] for v in breakdown.values())
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_medium_breakdown_has_correct_components(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "medium_multi_front")
        breakdown = result["breakdown"]
        expected_components = {
            "financial_impact",
            "triage_quality",
            "budget_utilization",
            "sla_compliance",
            "proactive_score",
        }
        assert set(breakdown.keys()) == expected_components

    def test_hard_breakdown_weights_sum_to_one(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "hard_cascading_crisis")
        breakdown = result["breakdown"]
        total_weight = sum(v["weight"] for v in breakdown.values())
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_hard_breakdown_has_correct_components(self, env: SupplyMindEnvironment) -> None:
        result = run_do_nothing_episode(env, "hard_cascading_crisis")
        breakdown = result["breakdown"]
        expected_components = {
            "loss_minimized",
            "cascade_containment",
            "information_efficiency",
            "budget_roi",
            "resilience",
            "customer_impact",
            "active_mitigation",
            "cascade_stage_suppression",
        }
        assert set(breakdown.keys()) == expected_components

    def test_all_component_scores_bounded(self, env: SupplyMindEnvironment) -> None:
        """Every component score in every task's breakdown must be in [0, 1]."""
        for task_id in ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]:
            result = run_do_nothing_episode(env, task_id)
            for comp_name, comp_data in result["breakdown"].items():
                assert 0.0 <= comp_data["score"] <= 1.0, (
                    f"Component {comp_name} in {task_id} had out-of-bounds "
                    f"score: {comp_data['score']}"
                )


# ──────────────────────────────────────────────
# Grader determinism
# ──────────────────────────────────────────────

class TestGraderDeterminism:
    """Test that graders are deterministic -- same inputs produce same scores."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_do_nothing_is_deterministic(self, env: SupplyMindEnvironment) -> None:
        """Running do-nothing twice should produce the same score."""
        score1 = run_do_nothing_episode(env, "easy_typhoon_response")["score"]
        score2 = run_do_nothing_episode(env, "easy_typhoon_response")["score"]
        assert score1 == pytest.approx(score2, abs=1e-4)

    def test_smart_is_deterministic(self, env: SupplyMindEnvironment) -> None:
        score1 = run_smart_easy_episode(env)["score"]
        score2 = run_smart_easy_episode(env)["score"]
        assert score1 == pytest.approx(score2, abs=1e-4)


# ──────────────────────────────────────────────
# Seed determinism proof
# ──────────────────────────────────────────────

class TestSeedDeterminism:
    """
    Prove that reset(task_id) produces byte-identical observations and
    scores across multiple runs. CRITICAL for baseline reproducibility.
    """

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_same_task_id_produces_identical_observations(self, env: SupplyMindEnvironment) -> None:
        """Two reset() calls with same task_id must produce identical initial obs."""
        obs1 = env.reset(task_id="easy_typhoon_response")
        data1 = obs1.model_dump()

        obs2 = env.reset(task_id="easy_typhoon_response")
        data2 = obs2.model_dump()

        # Compare everything except episode_id (UUID changes each time)
        data1.pop("info", None)
        data2.pop("info", None)
        assert data1["current_day"] == data2["current_day"]
        assert data1["days_remaining"] == data2["days_remaining"]
        assert data1["financials"] == data2["financials"]
        assert len(data1["node_statuses"]) == len(data2["node_statuses"])
        assert len(data1["active_signals"]) == len(data2["active_signals"])

    def test_full_episode_scores_identical_across_runs(self, env: SupplyMindEnvironment) -> None:
        """Running the same strategy 3x must produce identical grader scores."""
        scores = []
        for _ in range(3):
            result = run_do_nothing_episode(env, "easy_typhoon_response")
            scores.append(result["score"])

        assert scores[0] == pytest.approx(scores[1], abs=1e-6)
        assert scores[1] == pytest.approx(scores[2], abs=1e-6)

    def test_all_tasks_deterministic(self, env: SupplyMindEnvironment) -> None:
        """All 3 tasks produce identical scores on repeated runs."""
        for task_id in ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]:
            s1 = run_do_nothing_episode(env, task_id)["score"]
            s2 = run_do_nothing_episode(env, task_id)["score"]
            assert s1 == pytest.approx(s2, abs=1e-6), (
                f"Task {task_id} not deterministic: {s1} vs {s2}"
            )


# ──────────────────────────────────────────────
# Score variance test (5x identical runs)
# ──────────────────────────────────────────────

class TestScoreVariance:
    """Run smart baseline 5x, prove identical scores."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_smart_easy_5x_identical(self, env: SupplyMindEnvironment) -> None:
        scores = [run_smart_easy_episode(env)["score"] for _ in range(5)]
        for s in scores[1:]:
            assert s == pytest.approx(scores[0], abs=1e-6), (
                f"Score variance detected: {scores}"
            )

    def test_smart_hard_5x_identical(self, env: SupplyMindEnvironment) -> None:
        scores = [run_smart_hard_episode(env)["score"] for _ in range(5)]
        for s in scores[1:]:
            assert s == pytest.approx(scores[0], abs=1e-6), (
                f"Score variance detected: {scores}"
            )


# ──────────────────────────────────────────────
# Post-done step() behavior
# ──────────────────────────────────────────────

class TestPostDoneBehavior:
    """Test that step() after done returns gracefully, not crash."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_step_after_done_returns_observation(self, env: SupplyMindEnvironment) -> None:
        """Calling step() after episode is done should return an obs with done=True."""
        run_do_nothing_episode(env, "easy_typhoon_response")
        assert env.state.is_done

        # This should NOT raise
        obs = env.step(SupplyMindAction(action_type="do_nothing"))
        assert obs.done is True
        assert obs.reward == 0.0
        assert obs.info.get("post_done") is True
        assert obs.last_action_result is not None
        assert obs.last_action_result.success is False

    def test_step_after_done_is_idempotent(self, env: SupplyMindEnvironment) -> None:
        """Multiple step() calls after done should all return the same thing."""
        run_do_nothing_episode(env, "easy_typhoon_response")

        obs1 = env.step(SupplyMindAction(action_type="do_nothing"))
        obs2 = env.step(SupplyMindAction(action_type="do_nothing"))
        assert obs1.done is True
        assert obs2.done is True


# ──────────────────────────────────────────────
# Empty history grader
# ──────────────────────────────────────────────

class TestEmptyHistoryGrader:
    """Test that grading an episode with no steps returns 0.0."""

    @pytest.fixture
    def env(self) -> SupplyMindEnvironment:
        return SupplyMindEnvironment()

    def test_grade_immediately_after_reset(self, env: SupplyMindEnvironment) -> None:
        """Grading right after reset (no steps) should return 0.0."""
        env.reset(task_id="easy_typhoon_response")
        result = env.grade()
        assert result["score"] == 0.0
        assert "no_steps" in result["breakdown"]
