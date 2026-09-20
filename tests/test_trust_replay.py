"""Trust Layer + Historical Replay Proof — the honesty contract, as tests.

Covers the two builds of Issue Stream 5:

  BUILD 1  the Trust Graph (supplymind.warroom.trust + trust_router): every
           rendered number resolves to a real provenance chain; an unknown
           number is surfaced loudly, never given a fabricated chain.

  BUILD 2  the historical replay proof (supplymind.warroom.replay_proof): the
           no-lookahead guard is REAL — a future-dated datum is refused (the
           acceptance-mandated proof) — and a documented crisis replays
           end-to-end with an honest, untuned verdict.

Fully OFFLINE / CI-safe: live-source leaves are resolved in FORCE_REPLAY mode
(cache-only, no network); when no cache exists the leaf is honestly ``degraded``
and the test tolerates that (the honesty contract, not freshness, is asserted).

Run:
  pytest tests/test_trust_replay.py -q
"""
from __future__ import annotations

import pytest

from supplymind.warroom import replay_proof as rp
from supplymind.warroom import trust as trust_mod
from supplymind.warroom.replay_proof import NoLookaheadError

_HEX = set("0123456789abcdef")


# ==========================================================================
# BUILD 1 — Trust Graph
# ==========================================================================

def test_registry_lists_all_kinds():
    metrics = trust_mod.list_metrics()
    kinds = {m["kind"] for m in metrics}
    assert kinds == {trust_mod.KIND_LIVE, trust_mod.KIND_STATIC,
                     trust_mod.KIND_DERIVED, trust_mod.KIND_UNKNOWN}
    ids = {m["metric_id"] for m in metrics}
    assert "signal.fred.brent_price" in ids
    assert "warroom.brent_projection" in ids
    assert "warroom.operator_severity_input" in ids


def test_live_leaf_resolves_with_honest_provenance():
    """A live-source leaf resolves either RESOLVED (with a real 64-hex content
    sha256, raw URL and numeric age) or DEGRADED (with a real reason). Never a
    third, fabricated state."""
    node = trust_mod.resolve("signal.fred.brent_price", force_replay=True)
    assert node["kind"] == trust_mod.KIND_LIVE
    assert node["status"] in (trust_mod.RESOLVED, trust_mod.DEGRADED)
    assert node["code_path"] == "supplymind.data.live:_client_fred"
    if node["status"] == trust_mod.RESOLVED:
        sha = node["content_sha256"]
        assert isinstance(sha, str) and len(sha) == 64 and set(sha) <= _HEX, \
            "a resolved live leaf must carry a real content hash"
        assert node["raw_url"] and node["raw_url"].startswith("http")
        assert isinstance(node["age_seconds"], (int, float))
        assert node["provider"], "real provider must be named"
    else:
        assert node["degraded_reason"], "a degraded leaf must give the real reason"


def test_derived_number_resolves_to_a_real_recursive_chain():
    """A derived war-room number resolves to its code path + upstream metrics,
    and the chain bottoms out at a real source leaf carrying a content hash."""
    node = trust_mod.resolve("warroom.brent_projection", depth=3, force_replay=True)
    assert node["kind"] == trust_mod.KIND_DERIVED
    assert node["code_path"].endswith("interpolate_projection")
    assert node["method"], "a derived number must describe its transform"
    assert node["upstream"], "a derived number must name its upstream metrics"
    # Flatten the chain and require at least one real content-hashed leaf.
    def _leaves(n):
        if not n.get("chain"):
            return [n]
        out = []
        for c in n["chain"]:
            out.extend(_leaves(c))
        return out
    leaves = _leaves(node)
    hashed = [l for l in leaves if l.get("content_sha256")]
    assert hashed, "the derivation chain must reach at least one hashed real source"
    for l in hashed:
        assert len(l["content_sha256"]) == 64


def test_unknown_registered_number_is_surfaced_loudly_not_faked():
    """An operator-typed input has no external source. The resolver must say so
    LOUDLY and attach NO fabricated provenance chain."""
    node = trust_mod.resolve("warroom.operator_severity_input")
    assert node["status"] == trust_mod.UNKNOWN
    assert node["unknown_reason"] and "no external data source" in node["unknown_reason"].lower()
    assert node["content_sha256"] is None, "unknown provenance must NOT invent a hash"
    assert node["raw_url"] is None
    assert not node["chain"], "unknown provenance must NOT invent an upstream chain"


def test_unregistered_metric_is_unknown_not_an_error():
    """A metric we never measured resolves to unknown_provenance — an honest
    answer, not a fabricated chain and not a crash."""
    node = trust_mod.resolve("some.metric.we.never.measured")
    assert node["status"] == trust_mod.UNKNOWN
    assert node["content_sha256"] is None
    assert "not a registered" in node["unknown_reason"].lower()


def test_build_graph_summary_is_internally_consistent():
    g = trust_mod.build_graph(force_replay=True)
    s = g["summary"]
    assert s["n_metrics"] == len(g["nodes"])
    assert s["n_resolved"] + s["n_degraded"] + s["n_unknown_provenance"] == s["n_metrics"]
    assert s["n_unknown_provenance"] >= 1, "at least the operator-input metric is unknown"
    # Every derivation edge points from a real node to a real node.
    ids = {n["metric_id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["from"] in ids
        assert e["to"] in ids


# ==========================================================================
# BUILD 2 — No-lookahead guard (the acceptance-mandated proof)
# ==========================================================================

def test_guard_passes_strictly_prior_data():
    clean = [{"date": "2024-09-30", "x": 1}, {"date": "2024-01-01", "x": 2}]
    out = rp.guard_asof(clean, "2024-10-01", "date", context="unit")
    assert out is clean


def test_guard_refuses_a_future_dated_datum():
    """THE mandated proof: a datum dated on/after the as-of date is refused with
    a loud NoLookaheadError — the label-leakage guard is real, not decorative."""
    leaky = [{"date": "2024-09-30", "x": 1},
             {"date": "2024-10-02", "x": 2}]  # AFTER the event → must be refused
    with pytest.raises(NoLookaheadError) as ei:
        rp.guard_asof(leaky, "2024-10-01", "date", context="unit")
    assert "2024-10-02" in str(ei.value)
    assert "on/after" in str(ei.value)


def test_guard_refuses_a_same_dated_datum():
    """Even a datum dated exactly ON the event day is refused — a forecast may
    only use data strictly before the event."""
    same = [{"date": "2024-10-01", "x": 1}]
    with pytest.raises(NoLookaheadError):
        rp.guard_asof(same, "2024-10-01", "date", context="unit")


def test_reconstruction_never_leaks_the_future():
    """The real reconstruction of the featured crisis uses only strictly-prior
    FRED rows and strictly-prior analogs, and never includes the target event
    itself."""
    events = {e["id"]: e for e in rp._load_events()}
    ev = events["iran_true_promise_2_2024_10"]
    state = rp.reconstruct_information_state(ev)
    as_of = state.as_of
    assert all(r["date"] < as_of for r in state.fred_pre), "FRED leak into as-of"
    assert all(str(a["date"]) < as_of for a in state.prior_analogs), "analog leak"
    assert ev["id"] not in {a["id"] for a in state.prior_analogs}, \
        "the target event must never be its own analog"
    assert "0 violations" in state.lookahead_guard


# ==========================================================================
# BUILD 2 — Replay runs end-to-end with an honest verdict
# ==========================================================================

def test_replay_produces_a_real_untuned_verdict():
    """One documented crisis replays: real as-of assessment, real outcome from
    the FRED path, and a real boolean verdict (HIT or MISS — both acceptable;
    the point is it is measured, not asserted)."""
    r = rp.run_replay("iran_true_promise_2_2024_10")
    a, o, v = r["as_of_assessment"], r["actual_outcome"], r["verdict"]

    # as-of assessment is real + honest
    assert a["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert a["risk_inference_type"] == "rubric_fallback", \
        "BLOCKED-ON-KEY: the deterministic rubric path must run, honestly labeled"
    assert a["llm_hop"]["status"] == "pending-key", \
        "the LLM analyst hop must be honestly marked pending-key, never faked"
    assert a["pre_close_usd_bbl"] > 0
    # engine MC continuity channel is a real (possibly near-zero) number
    mc = a["engine_mc_continuity_channel"]
    assert mc["status"] in ("ok", "blocked")
    if mc["status"] == "ok":
        assert isinstance(mc["usd"], (int, float))

    # outcome is real ground truth from the FRED corpus
    assert o["realized_peak_usd_bbl"] is not None
    assert o["realized_direction"] in {"up", "down", "flat"}
    roc = o["realized_oil_channel_counterfactual"]
    assert roc["status"] in ("ok", "degraded", "blocked")

    # verdict is a genuine measured boolean, not a constant
    assert isinstance(v["directional_correct"], bool)
    assert v["verdict"], "an honest one-line verdict must be present"


def test_featured_event_direction_matches_documented_reality():
    """Falsifiability anchor: for the featured event, the OUTCOME the replay
    measures from FRED must agree with the independently-documented oil impact
    in the crisis library (pre < peak). This proves the outcome side reads real
    prices, not fabricated ones."""
    events = {e["id"]: e for e in rp._load_events()}
    ev = events["iran_true_promise_2_2024_10"]
    o = rp.observe_actual_outcome(ev)
    doc = ev["oil_impact_usd_bbl"]
    # documented: pre 71.8 -> peak 78.2 (a real rise); the measured FRED peak
    # must likewise exceed the measured pre-close.
    assert o["realized_peak_usd_bbl"] > o["pre_close_usd_bbl"], \
        "measured FRED peak must exceed pre-close, matching the documented rise"
    assert doc["peak"] > doc["pre"], "documented library rise sanity check"


def test_aggregate_sweep_reports_an_honest_hit_rate():
    """The full eligible sweep produces an aggregate directional hit rate in
    [0,1] over >=1 scored crisis — reported as-is (untuned)."""
    agg = rp.run_all_eligible()
    assert agg["n_eligible"] >= 1
    assert agg["n_scored"] >= 1
    assert 0 <= agg["n_directional_correct"] <= agg["n_scored"]
    assert agg["directional_hit_rate"] is None or 0.0 <= agg["directional_hit_rate"] <= 1.0
    assert "untuned" in agg["honest_reading"].lower()


# ==========================================================================
# Router (offline-safe) — the HTTP surface the war room calls
# ==========================================================================

@pytest.fixture
def trust_client():
    fastapi = pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from supplymind.warroom.trust_router import router
    app = FastAPI()
    app.include_router(router, prefix="/trust")
    return TestClient(app)


def test_router_metrics_and_health(trust_client):
    h = trust_client.get("/trust/health").json()
    assert h["status"] == "ok" and h["n_metrics"] >= 20
    m = trust_client.get("/trust/metrics").json()
    assert m["count"] == h["n_metrics"]


def test_router_resolve_live_and_unknown(trust_client):
    r = trust_client.get("/trust/resolve/signal.fred.brent_price",
                         params={"live": False})
    assert r.status_code == 200
    node = r.json()
    assert node["metric_id"] == "signal.fred.brent_price"
    assert node["status"] in (trust_mod.RESOLVED, trust_mod.DEGRADED)
    # unknown metric is a 200 honest answer, not an error
    u = trust_client.get("/trust/resolve/made.up.metric")
    assert u.status_code == 200
    assert u.json()["status"] == trust_mod.UNKNOWN


def test_router_replay_proof_one_event(trust_client):
    r = trust_client.get("/trust/replay-proof/iran_true_promise_2_2024_10")
    assert r.status_code == 200
    body = r.json()
    assert body["event_id"] == "iran_true_promise_2_2024_10"
    assert isinstance(body["verdict"]["directional_correct"], bool)
    # an unknown event id is an honest 404
    bad = trust_client.get("/trust/replay-proof/no_such_event")
    assert bad.status_code == 404
