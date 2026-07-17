"""Formal OpenEnv compliance test.

Verifies SupplyMind satisfies the key points of the OpenEnv specification
(https://github.com/meta-llama/open-env).

Tests every contract item separately so judges can see exactly what compliance
means in practice.

Run:
  pytest tests/test_openenv_compliance.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from supplymind.contracts import (
    SupplyMindAction,
    SupplyMindObservation,
    SupplyMindState,
    DisruptionSignal,
    SupplierStatus,
    FinancialSnapshot,
    ActionResult,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# Contract 1: Typed Pydantic v2 models
# ============================================================

def test_action_is_pydantic_v2():
    """Actions must be validated Pydantic models."""
    a = SupplyMindAction(action_type="do_nothing")
    assert a.action_type == "do_nothing"
    # Pydantic v2 validates types
    with pytest.raises(Exception):
        SupplyMindAction(action_type="not_a_real_action", target_node_id="x")  # invalid type


def test_observation_has_dual_summaries():
    """OpenEnv spec: observations should support both structured + LLM-friendly forms."""
    fields = SupplyMindObservation.model_fields
    assert "situation_summary" in fields
    assert "compact_summary" in fields
    assert "active_signals" in fields
    assert "node_statuses" in fields
    assert "financials" in fields


def test_state_model():
    assert "current_day" in SupplyMindState.model_fields or "step_count" in SupplyMindState.model_fields


# ============================================================
# Contract 2: openenv.yaml exists and is valid
# ============================================================

def test_openenv_yaml_exists():
    path = ROOT / "openenv.yaml"
    assert path.exists(), "openenv.yaml must exist at repo root"


def test_openenv_yaml_has_tasks():
    import yaml
    y = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    assert "tasks" in y
    assert len(y["tasks"]) >= 3, "Must declare at least 3 tasks"
    task_ids = {t["id"] for t in y["tasks"]}
    assert "easy_typhoon_response" in task_ids
    assert "medium_multi_front" in task_ids
    assert "hard_cascading_crisis" in task_ids


def test_openenv_yaml_has_required_fields():
    import yaml
    y = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    for field in ["environment_id", "name", "version", "action", "observation",
                  "description", "author", "license"]:
        assert field in y, f"openenv.yaml missing required field: {field}"


# ============================================================
# Contract 3: Core HTTP endpoints
# ============================================================

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_tasks(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body, "/tasks must return a 'tasks' collection"
    tasks = body["tasks"]
    assert isinstance(tasks, list) and len(tasks) >= 3, "must list at least 3 tasks"
    assert all("task_id" in t for t in tasks), "every task must declare a task_id"


def test_reset_returns_observation(client):
    r = client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    assert r.status_code == 200
    obs = r.json()
    # Must match SupplyMindObservation shape
    for key in ["current_day", "days_remaining", "active_signals", "node_statuses",
                "financials", "situation_summary", "compact_summary"]:
        assert key in obs, f"/reset response missing: {key}"


def test_step_returns_observation(client):
    client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    r = client.post("/step", json={"action_type": "do_nothing"})
    assert r.status_code == 200
    obs = r.json()
    assert "reward" in obs
    assert "done" in obs


def test_state_endpoint(client):
    client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    r = client.get("/state")
    assert r.status_code == 200


def test_grader_endpoint(client):
    """Grader must return a bounded score."""
    client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    # Run a minimal trajectory
    for _ in range(3):
        client.post("/step", json={"action_type": "do_nothing"})
    r = client.post("/grader")
    assert r.status_code == 200, f"/grader must return 200 after a reset+steps, got {r.status_code}: {r.text}"
    score = r.json()["score"]
    assert 0.0 <= score <= 1.0, f"Grader score out of bounds [0,1]: {score}"


# ============================================================
# Contract 4: MCP JSON-RPC + WebSocket routes declared
# ============================================================

def test_mcp_route_declared():
    route_paths = [r.path for r in app.routes]
    assert "/mcp" in route_paths, "MCP JSON-RPC endpoint /mcp must exist"


def test_persistent_session_transport(client):
    """OpenEnv persistent-session / tool-discovery transport.

    SupplyMind does NOT currently expose a raw WebSocket; it serves the
    persistent-session + tool-discovery contract over MCP JSON-RPC on the
    HTTP POST /mcp route. This test asserts the REAL transport works — the
    route exists AND answers `initialize` with a well-formed JSON-RPC 2.0
    result. (If a raw WebSocket route is ever added, it also satisfies the
    contract.) This is deliberately falsifiable: a broken or missing /mcp
    transport turns it red, unlike the earlier vacuous `has_ws or /mcp-exists`
    check that could never fail while /mcp was a registered route.
    """
    from starlette.routing import WebSocketRoute
    has_ws = any(isinstance(r, WebSocketRoute) for r in app.routes)
    mcp_declared = any(getattr(r, "path", "") == "/mcp" for r in app.routes)
    assert has_ws or mcp_declared, "must expose a WebSocket or the /mcp transport"
    # The transport must actually respond, not merely be declared.
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r.status_code == 200, f"/mcp transport must answer: {r.status_code} {r.text}"
    body = r.json()
    assert body.get("jsonrpc") == "2.0" and body.get("id") == 1
    assert "result" in body and "serverInfo" in body["result"]


# ============================================================
# Contract 5: Reproducibility
# ============================================================

def test_reset_with_seed_deterministic(client):
    """Same seed must produce same initial observation."""
    r1 = client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42}).json()
    r2 = client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42}).json()
    # At minimum, top-level scalars should match exactly
    assert r1["current_day"] == r2["current_day"]
    assert r1["days_remaining"] == r2["days_remaining"]


def test_grader_zero_variance():
    """Grader must be deterministic: same trajectory -> same score 5x runs.
    (Integration-style; uses TestClient to avoid external processes.)
    """
    c = TestClient(app)
    scores = []
    for _ in range(3):  # 3 runs to keep test time reasonable; DEMO_SCRIPT states 5x
        c.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 123})
        for _ in range(5):
            c.post("/step", json={"action_type": "do_nothing"})
        r = c.post("/grader")
        assert r.status_code == 200, f"/grader failed on run {len(scores)}: {r.status_code} {r.text}"
        scores.append(r.json()["score"])
    # Determinism: identical seed + identical trajectory must yield identical scores.
    assert len(scores) == 3, f"Expected 3 grader results, got {len(scores)}"
    assert len(set(scores)) == 1, (
        f"Grader is not deterministic across identical trajectories: {scores}"
    )


# ============================================================
# Contract 6: Dense reward, not sparse binary
# ============================================================

def test_step_reward_is_float():
    c = TestClient(app)
    c.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    r = c.post("/step", json={"action_type": "do_nothing"})
    assert r.status_code == 200
    reward = r.json().get("reward")
    assert isinstance(reward, (int, float))


# ============================================================
# Contract 7: Action validation
# ============================================================

def test_invalid_action_rejected_or_graceful(client):
    """Invalid action (unknown target_node_id) must be handled, never crash, and
    never silently 'succeed'."""
    client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    r = client.post("/step", json={
        "action_type": "activate_backup_supplier",
        "target_node_id": "DOES_NOT_EXIST",
        "backup_supplier_id": "ALSO_FAKE",
    })
    # Either 200 with a truthful failure result, or a 4xx validation error —
    # but NOT a 500 crash, and NOT a fabricated success.
    assert r.status_code in (200, 400, 404, 422), f"must not crash: {r.status_code} {r.text}"
    if r.status_code == 200:
        lar = r.json()["last_action_result"]
        assert lar["success"] is False, "activating a nonexistent node must NOT report success"
        assert lar["cost"] == 0.0, "a failed action must incur zero cost"
        assert "not found" in lar["message"].lower()


# ============================================================
# Contract 8: Episode termination
# ============================================================

def test_episode_eventually_terminates(client):
    """Running to episode end must hit done=True."""
    client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    done = False
    for _ in range(200):  # episode_length=30 for easy; 200 is safe upper bound
        r = client.post("/step", json={"action_type": "do_nothing"})
        if r.status_code != 200:
            break
        if r.json().get("done"):
            done = True
            break
    assert done, "Easy task must terminate within 200 steps"
