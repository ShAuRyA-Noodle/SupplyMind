"""
Integration tests for SupplyMind FastAPI server.

Uses FastAPI TestClient to test all required OpenEnv endpoints without
starting a real HTTP server.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.app import app, env, _sessions


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def client() -> TestClient:
    """Create a fresh TestClient. Reset environment state before each test."""
    # Reset the shared environment to ensure clean state
    env.engine = None
    env.current_task = None
    env._state = env._state.__class__()
    env._episode_history = []
    # Clear the per-session environment pool so each test starts fresh
    _sessions.clear()
    return TestClient(app)


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_contains_environment_name(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "environment" in data


# ──────────────────────────────────────────────
# GET /tasks
# ──────────────────────────────────────────────

class TestTasksEndpoint:
    def test_tasks_returns_200(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_tasks_returns_3_tasks(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        data = resp.json()
        assert len(data["tasks"]) == 3

    def test_tasks_contain_expected_ids(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        data = resp.json()
        task_ids = {t["task_id"] for t in data["tasks"]}
        assert "easy_typhoon_response" in task_ids
        assert "medium_multi_front" in task_ids
        assert "hard_cascading_crisis" in task_ids

    def test_tasks_include_action_schema(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        data = resp.json()
        assert "action_schema" in data
        schema = data["action_schema"]
        assert "properties" in schema
        assert "action_type" in schema["properties"]

    def test_each_task_has_required_fields(self, client: TestClient) -> None:
        resp = client.get("/tasks")
        data = resp.json()
        for task in data["tasks"]:
            assert "task_id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "episode_length" in task
            assert "budget" in task


# ──────────────────────────────────────────────
# POST /reset
# ──────────────────────────────────────────────

class TestResetEndpoint:
    def test_reset_returns_200(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        assert resp.status_code == 200

    def test_reset_returns_observation_with_day_zero(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        data = resp.json()
        assert data["current_day"] == 0

    def test_reset_returns_correct_days_remaining(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        data = resp.json()
        assert data["days_remaining"] == 30

    def test_reset_observation_has_financials(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        data = resp.json()
        assert "financials" in data
        assert data["financials"]["budget_total"] == 5_000_000.0

    def test_reset_observation_has_node_statuses(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        data = resp.json()
        assert "node_statuses" in data
        assert len(data["node_statuses"]) == 12

    def test_reset_observation_not_done(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=easy_typhoon_response")
        data = resp.json()
        assert data["done"] is False

    def test_reset_invalid_task_returns_400(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=nonexistent_task")
        assert resp.status_code == 400

    def test_reset_default_task(self, client: TestClient) -> None:
        """Reset with no task_id should default to easy."""
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_medium_task(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=medium_multi_front")
        data = resp.json()
        assert data["days_remaining"] == 45
        assert data["financials"]["budget_total"] == 8_000_000.0

    def test_reset_hard_task(self, client: TestClient) -> None:
        resp = client.post("/reset?task_id=hard_cascading_crisis")
        data = resp.json()
        assert data["days_remaining"] == 60
        assert data["financials"]["budget_total"] == 10_000_000.0


# ──────────────────────────────────────────────
# POST /step
# ──────────────────────────────────────────────

class TestStepEndpoint:
    def test_step_without_reset_returns_400(self, client: TestClient) -> None:
        resp = client.post("/step", json={"action_type": "do_nothing"})
        assert resp.status_code == 400

    def test_step_returns_observation_with_reward(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        resp = client.post("/step", json={"action_type": "do_nothing"})
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert isinstance(data["reward"], (int, float))

    def test_step_advances_day(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        resp = client.post("/step", json={"action_type": "do_nothing"})
        data = resp.json()
        assert data["current_day"] >= 1

    def test_step_with_action_parameters(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        resp = client.post("/step", json={
            "action_type": "issue_supplier_alert",
            "target_node_id": "SUP_TSMC",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "last_action_result" in data

    def test_step_returns_done_eventually(self, client: TestClient) -> None:
        """Running 30 do-nothing steps should eventually end the episode."""
        client.post("/reset?task_id=easy_typhoon_response")
        done = False
        for _ in range(35):  # Slightly more than episode length
            resp = client.post("/step", json={"action_type": "do_nothing"})
            if resp.status_code != 200:
                break
            data = resp.json()
            if data.get("done", False):
                done = True
                break
        assert done, "Episode did not end within expected step count"

    def test_step_after_done_returns_gracefully(self, client: TestClient) -> None:
        """Stepping after episode is done should return 200 with done=True."""
        client.post("/reset?task_id=easy_typhoon_response")
        # Run until done
        for _ in range(35):
            resp = client.post("/step", json={"action_type": "do_nothing"})
            data = resp.json()
            if data.get("done", False):
                break

        # Try one more step -- should return gracefully with done=True
        resp = client.post("/step", json={"action_type": "do_nothing"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["info"].get("post_done") is True


# ──────────────────────────────────────────────
# GET /state
# ──────────────────────────────────────────────

class TestStateEndpoint:
    def test_state_returns_200(self, client: TestClient) -> None:
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_before_reset_has_defaults(self, client: TestClient) -> None:
        resp = client.get("/state")
        data = resp.json()
        assert "step_count" in data
        assert "is_done" in data

    def test_state_after_reset_has_task_info(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        resp = client.get("/state")
        data = resp.json()
        assert data["task_id"] == "easy_typhoon_response"
        assert data["total_steps"] == 30
        assert data["is_done"] is False

    def test_state_updates_after_steps(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        client.post("/step", json={"action_type": "do_nothing"})
        client.post("/step", json={"action_type": "do_nothing"})

        resp = client.get("/state")
        data = resp.json()
        assert data["step_count"] == 2

    def test_state_has_cumulative_reward(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        client.post("/step", json={"action_type": "do_nothing"})

        resp = client.get("/state")
        data = resp.json()
        assert "cumulative_reward" in data


# ──────────────────────────────────────────────
# POST /grader
# ──────────────────────────────────────────────

class TestGraderEndpoint:
    def test_grader_without_episode_returns_400(self, client: TestClient) -> None:
        resp = client.post("/grader")
        assert resp.status_code == 400

    def test_grader_returns_score_in_range(self, client: TestClient) -> None:
        """Run an episode and grade it; score should be in [0, 1]."""
        client.post("/reset?task_id=easy_typhoon_response")
        # Run a few steps
        for _ in range(30):
            resp = client.post("/step", json={"action_type": "do_nothing"})
            if resp.json().get("done", False):
                break

        resp = client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_returns_breakdown(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        for _ in range(30):
            resp = client.post("/step", json={"action_type": "do_nothing"})
            if resp.json().get("done", False):
                break

        resp = client.post("/grader")
        data = resp.json()
        assert "breakdown" in data
        assert len(data["breakdown"]) > 0

    def test_grader_returns_task_metadata(self, client: TestClient) -> None:
        client.post("/reset?task_id=easy_typhoon_response")
        for _ in range(30):
            resp = client.post("/step", json={"action_type": "do_nothing"})
            if resp.json().get("done", False):
                break

        resp = client.post("/grader")
        data = resp.json()
        assert data["task_id"] == "easy_typhoon_response"
        assert data["difficulty"] == "easy"
        assert "steps_taken" in data
        assert "cumulative_reward" in data

    def test_grader_can_be_called_mid_episode(self, client: TestClient) -> None:
        """Grader should work even before the episode is done."""
        client.post("/reset?task_id=easy_typhoon_response")
        client.post("/step", json={"action_type": "do_nothing"})

        resp = client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0


# ──────────────────────────────────────────────
# Full episode integration
# ──────────────────────────────────────────────

class TestFullEpisodeIntegration:
    """End-to-end test: reset -> step loop -> grade."""

    def test_full_episode_easy(self, client: TestClient) -> None:
        # Reset
        resp = client.post("/reset?task_id=easy_typhoon_response")
        assert resp.status_code == 200
        obs = resp.json()
        assert obs["current_day"] == 0

        # Step loop
        steps = 0
        while not obs.get("done", False) and steps < 35:
            resp = client.post("/step", json={"action_type": "do_nothing"})
            assert resp.status_code == 200
            obs = resp.json()
            steps += 1

        assert obs["done"] is True

        # Grade
        resp = client.post("/grader")
        assert resp.status_code == 200
        result = resp.json()
        assert 0.0 <= result["score"] <= 1.0
        assert result["is_done"] is True

        # State
        resp = client.get("/state")
        data = resp.json()
        assert data["is_done"] is True
        assert data["step_count"] == steps
