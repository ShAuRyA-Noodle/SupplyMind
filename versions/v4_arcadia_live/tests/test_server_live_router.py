"""
test_server_live_router.py — Verify the v4 /live/* router is mounted on server/app.py
without breaking any existing v3 endpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app import app


def test_live_health_endpoint_mounted():
    client = TestClient(app)
    r = client.get("/live/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "ollama_available" in body
    assert "event_counts" in body


def test_live_hormuz_closure_endpoint_mounted():
    client = TestClient(app)
    r = client.post("/live/hormuz-closure", json={
        "scenario_text": "Iran threatens Hormuz closure. Brent spikes.",
        "region": "hormuz",
        "enable_llm_judges": False,
        "include_recent_signals": False,
        "k_analogs": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert len(body["recommended_actions"]) >= 1
    assert "counterfactual" in body
    assert body["ollama_available"] is False


def test_live_analog_match_endpoint():
    client = TestClient(app)
    r = client.post("/live/analog-match?query=Red+Sea+Houthi+attack&k=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body["analogs"]) == 2


def test_v3_endpoints_still_work():
    """Regression: v3 /health and /tasks must not break after adding the router."""
    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    r_tasks = client.get("/tasks")
    assert r_tasks.status_code == 200
    body = r_tasks.json()
    # Match whichever schema exposes tasks (some paths return list, some dict)
    if isinstance(body, dict):
        assert "tasks" in body or "task_ids" in body or "action_schema" in body
    else:
        assert isinstance(body, list)
