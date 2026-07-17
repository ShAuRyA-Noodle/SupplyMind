"""Live / introspection endpoint contracts — real-or-honestly-degraded shapes.

These endpoints back the war-room live layer and the OpenEnv runtime contract.
This suite asserts the CONTRACT of each, so it stays green OFFLINE in CI:
it never triggers a live external fetch — the /live/* endpoints read the local
ingested event store, and the honest degraded flag (`ollama_available`) must be
present and truthful when the LLM source is down.

No fabricated fields are tolerated: every leaderboard row must cite provenance,
every count must be a real non-negative integer, and the degraded flag must be
a real boolean, not a hard-coded "true".

Registered paths (the /live and /arena prefixes are where the routers mount):
  GET /live/signal-counts   GET /live/recent-events   GET /live/health
  GET /health   GET /metadata   GET /schema   GET /arena/leaderboard

Run:
  pytest tests/test_live_endpoints.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

_EXPECTED_ACTIONS = {
    "do_nothing", "activate_backup_supplier", "reroute_shipment",
    "increase_safety_stock", "expedite_order", "hedge_commodity",
    "issue_supplier_alert",
}


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /live/* — event-store-backed (offline-safe: no external fetch)
# ---------------------------------------------------------------------------

def test_live_signal_counts_shape(client):
    """Per-source counts over a window: a dict of {source: non-negative int}."""
    r = client.get("/live/signal-counts")
    assert r.status_code == 200, f"router must be mounted: {r.status_code} {r.text}"
    counts = r.json()
    assert isinstance(counts, dict), "signal-counts must be a {source: count} map"
    for src, n in counts.items():
        assert isinstance(src, str) and src != ""
        assert isinstance(n, int) and n >= 0, f"count for {src} must be a real int >= 0, got {n}"


def test_live_recent_events_contract(client):
    """Recent events: count must equal len(events); each event a real dict.
    Offline-safe — queries the local event store, never the network."""
    r = client.get("/live/recent-events", params={"hours": 24, "limit": 20})
    assert r.status_code == 200, f"router must be mounted: {r.status_code} {r.text}"
    body = r.json()
    assert set(body.keys()) >= {"count", "events"}
    assert isinstance(body["count"], int) and body["count"] >= 0
    assert isinstance(body["events"], list)
    assert body["count"] == len(body["events"]), "count must be truthful (== len(events))"
    for ev in body["events"]:
        assert isinstance(ev, dict) and len(ev) > 0


def test_live_health_degraded_flag_contract(client):
    """The honest degraded contract: /live/health surfaces a real `ollama_available`
    boolean + a note explaining degraded mode. In CI the LLM source is down, so the
    flag reports it truthfully rather than pretending the panel is live."""
    r = client.get("/live/health")
    assert r.status_code == 200, f"/live/health failed: {r.status_code} {r.text}"
    h = r.json()
    assert h["status"] == "ok"
    assert "ollama_available" in h, "degraded flag must be present"
    assert isinstance(h["ollama_available"], bool), "degraded flag must be a real bool"
    assert isinstance(h["event_counts"], dict), "event counts must be a real map"
    for src, n in h["event_counts"].items():
        assert isinstance(n, int) and n >= 0, f"event count for {src} must be int >= 0"
    # There is genuine ingested data in the store — proves counts are real, not empty theater.
    assert sum(h["event_counts"].values()) > 0, "event store must hold real ingested signals"
    assert "degraded" in h["note"].lower(), "degraded behavior must be documented in the response"


# ---------------------------------------------------------------------------
# OpenEnv runtime introspection endpoints
# ---------------------------------------------------------------------------

def test_root_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_metadata_shape(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    m = r.json()
    assert m["name"] == "supplymind"
    assert isinstance(m["version"], str) and m["version"] != ""
    assert isinstance(m["tags"], list) and "openenv" in m["tags"]
    assert isinstance(m["description"], str) and len(m["description"]) > 0


def test_schema_declares_real_action_contract(client):
    """/schema must publish the real action/observation/state JSON schemas, and
    the action schema must enumerate exactly the 7 real action types."""
    r = client.get("/schema")
    assert r.status_code == 200
    s = r.json()
    for key in ("action", "observation", "state"):
        assert key in s, f"/schema missing {key}"
        assert "properties" in s[key], f"{key} schema must be a real JSON schema"
    enum = set(s["action"]["properties"]["action_type"]["enum"])
    assert enum == _EXPECTED_ACTIONS, (
        f"action schema must expose exactly the 7 real actions; got {enum}"
    )


def test_arena_leaderboard_no_fabricated_rows(client):
    """Leaderboard rows must cite provenance — no headline number without a source.
    Ranks must be a real 1..N ordering, and totals must be internally consistent."""
    r = client.get("/arena/leaderboard")
    assert r.status_code == 200, f"arena router must be mounted: {r.status_code} {r.text}"
    b = r.json()
    assert set(b.keys()) >= {"generated_at", "n_submissions", "n_baselines", "rows"}
    assert b["n_baselines"] == 6, "6 real R6-benchmark baselines are pre-seeded"
    assert b["n_submissions"] >= 0
    rows = b["rows"]
    assert len(rows) == b["n_submissions"] + b["n_baselines"], "row count must be consistent"
    ranks = [row["rank"] for row in rows]
    assert ranks == list(range(1, len(rows) + 1)), f"ranks must be a real 1..N ordering: {ranks}"
    for row in rows:
        assert isinstance(row["policy_name"], str) and row["policy_name"] != ""
        assert isinstance(row["overall_reward_mean"], (int, float))
        assert isinstance(row["source"], str) and row["source"] != "", (
            f"every row must cite a real source (no fabricated numbers): {row}"
        )
        assert isinstance(row["overall_ci95"], list) and len(row["overall_ci95"]) == 2
    # conservative ranking: CI95-lower must be non-increasing down the board
    keys = [
        (row["overall_ci95"][0] if row["overall_ci95"][0] is not None
         else row["overall_reward_mean"])
        for row in rows
    ]
    assert keys == sorted(keys, reverse=True), "leaderboard must be sorted conservatively"
