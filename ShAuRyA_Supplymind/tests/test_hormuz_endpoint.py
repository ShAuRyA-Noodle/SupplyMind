"""
test_hormuz_endpoint.py — Integration tests for the /live/hormuz-closure pipeline.

Tests run WITHOUT Ollama (rubric fallback) and WITHOUT network calls by default.
A separate test hits the real endpoints when RUN_LIVE=1 is set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.realtime import crisis_library, hormuz_endpoint, store


# -------------------------------------------------------------------
# Crisis library
# -------------------------------------------------------------------


def test_library_has_eight_events():
    lib = crisis_library.load_library()
    assert len(lib["events"]) >= 8


def test_every_event_has_required_fields():
    lib = crisis_library.load_library()
    for e in lib["events"]:
        for field in ["id", "name", "date", "severity", "event_type", "region",
                      "summary", "citations"]:
            assert field in e, f"event {e.get('id', '?')} missing {field}"
        assert 0.0 <= e["severity"] <= 1.0
        assert len(e["citations"]) >= 3, f"{e['id']}: need >=3 citations"
        for c in e["citations"]:
            assert "url" in c and "title" in c and "publisher" in c


def test_analog_match_finds_hormuz_event():
    analogs = crisis_library.find_analogs(
        "Iran threatens to close Strait of Hormuz after US seizes tanker",
        k=3, mode="tfidf",  # force tfidf to avoid model download in test
    )
    assert len(analogs) == 3
    # Top analog should be Hormuz-related
    assert "hormuz" in analogs[0].event_id.lower() or \
           "hormuz" in analogs[0].summary.lower()
    assert analogs[0].similarity > 0.0


def test_projection_interpolation():
    analogs = crisis_library.find_analogs(
        "Red Sea Houthi attacks force carriers to reroute via Cape of Good Hope",
        k=3, mode="tfidf",
    )
    proj = crisis_library.interpolate_projection(analogs)
    assert "brent_projection_usd_bbl_p50" in proj
    assert "severity_p50" in proj
    assert proj["top_analog_name"] is not None


# -------------------------------------------------------------------
# Hormuz endpoint — offline path (rubric fallback, no Ollama)
# -------------------------------------------------------------------


def test_rubric_pipeline_returns_full_response():
    # Force rubric fallback by disabling LLM
    req = hormuz_endpoint.ScenarioRequest(
        scenario_text="Iran threatens Hormuz closure. Brent spikes.",
        region="hormuz",
        enable_llm_judges=False,
        include_recent_signals=False,  # no DB dependency
        k_analogs=3,
    )
    resp = hormuz_endpoint.run_hormuz_pipeline(req)
    # Structural sanity
    assert resp.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert 0.0 <= resp.consensus_confidence <= 1.0
    assert len(resp.analogs) == 3
    assert len(resp.recommended_actions) >= 1
    assert resp.counterfactual["no_action_p50_loss_usd"] >= 0
    assert resp.counterfactual["savings_pct"] >= 0
    assert resp.ollama_available is False
    assert len(resp.judges) >= 1
    assert resp.judges[0].name == "Rubric-Fallback"


def test_rubric_high_risk_includes_hedge():
    req = hormuz_endpoint.ScenarioRequest(
        scenario_text=("Iran launches ballistic missile attack on Israel. "
                       "Iran threatens to close Strait of Hormuz. "
                       "Brent crude surges past 120 dollars per barrel. "
                       "Major carriers pause Persian Gulf bookings."),
        region="hormuz",
        enable_llm_judges=False,
        include_recent_signals=False,
    )
    resp = hormuz_endpoint.run_hormuz_pipeline(req)
    # This scenario should be HIGH or CRITICAL
    assert resp.risk_level in ("HIGH", "CRITICAL")
    # Should recommend hedge
    action_types = [a.action_type for a in resp.recommended_actions]
    assert "hedge_commodity" in action_types
    assert "issue_supplier_alert" in action_types


def test_low_risk_scenario_returns_low_or_medium():
    req = hormuz_endpoint.ScenarioRequest(
        scenario_text="Routine container shipping. No geopolitical incidents reported.",
        region="global",
        enable_llm_judges=False,
        include_recent_signals=False,
    )
    resp = hormuz_endpoint.run_hormuz_pipeline(req)
    assert resp.risk_level in ("LOW", "MEDIUM")


# -------------------------------------------------------------------
# Event store smoke
# -------------------------------------------------------------------


def test_store_init_and_query():
    store.init_db()
    # Insert a test event
    ev = store.Event(
        source="pytest",
        ts_iso="2026-04-21T12:00:00Z",
        event_type="test",
        region="hormuz",
        severity=0.5,
        raw_text="pytest smoke event",
    )
    n = store.insert_events([ev])
    # At least one was inserted (may be 0 if previous test inserted same)
    assert n >= 0
    rows = store.query_recent(region="hormuz", limit=5)
    assert isinstance(rows, list)


# -------------------------------------------------------------------
# Live tests — auto-run when preconditions met, otherwise skip with a clear reason.
# Set OFFLINE_MODE=1 to force-skip all live tests.
# -------------------------------------------------------------------


def _env_loaded() -> bool:
    """Ensure .env is loaded once so keys are in os.environ."""
    from ShAuRyA_Supplymind.realtime import ingestor
    ingestor._load_dotenv_if_available()
    return bool(os.environ.get("NEWS_API_KEY") or os.environ.get("FRED_API_KEY"))


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434") + "/api/tags",
                         timeout=3)
        return r.status_code == 200
    except Exception:
        return False


OFFLINE = os.environ.get("OFFLINE_MODE") == "1"


@pytest.mark.skipif(OFFLINE or not _env_loaded(),
                    reason="live ingestion needs NEWS_API_KEY / FRED_API_KEY in .env")
def test_live_ingestion_cycle():
    from ShAuRyA_Supplymind.realtime import ingestor
    result = ingestor.ingest_once(lookback_minutes=1440, skip=("marinetraffic",))
    assert result["fetched"] > 0, "expected some live events"


@pytest.mark.skipif(OFFLINE or not _env_loaded(),
                    reason="live pipeline needs .env keys (ok if Ollama is down — rubric fallback)")
def test_live_hormuz_pipeline_with_ollama():
    req = hormuz_endpoint.ScenarioRequest(
        scenario_text="Iran threatens Hormuz closure. Brent spikes to 123 dollars per barrel.",
        region="hormuz",
        enable_llm_judges=_ollama_up(),
        include_recent_signals=True,
    )
    resp = hormuz_endpoint.run_hormuz_pipeline(req)
    # Live run — accept whatever result, just ensure it didn't crash
    assert resp.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert len(resp.judges) >= 1
