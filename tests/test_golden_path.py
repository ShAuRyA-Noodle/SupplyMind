"""Golden-path end-to-end test — the SupplyMind product spine, as a test.

This is the ONE scripted journey from PRODUCT_THESIS.md / CLAUDE.md §12b,
executed against the REAL local FastAPI server (TestClient), fully OFFLINE —
no OpenRouter key, no Ollama daemon, no live external calls.

Every hop asserts REAL data: valid schemas, typed numbers, real provenance.
Where a hop is honestly DEGRADED without the OpenRouter/Ollama key (the live
LLM judge panel), the test asserts the degraded FLAG is present and TRUTHFUL
(and honestly labeled), never a fabricated LLM value.

Golden path (the 4-minute judge demo):
  1. SIGNAL       reset a seeded episode                          POST /reset
  2. DECISION     step a real action with real economic effect    POST /step
  3. STATE        real episode metadata                           GET  /state
  4. GRADE        deterministic grader score in [0,1]             POST /grader
  5. ASSESSMENT   keyword rubric vs real R4 ground truth (no key) POST /analyst/grade
  6. COUNTERFACT  4 real causal methods + honest consensus (R6)   POST /counterfactual/platinum
  7. WAR ROOM     renders it live; honest degraded-LLM flag       POST /demo/hormuz-war-room

Run:
  pytest tests/test_golden_path.py -q

If any hop returns a fabricated value where real data is required, or an
untruthful degraded flag, this test fails LOUD.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

# The 7 action types the environment contract exposes (mirrors SupplyMindAction).
_EXPECTED_ACTIONS = {
    "do_nothing", "activate_backup_supplier", "reroute_shipment",
    "increase_safety_stock", "expedite_order", "hedge_commodity",
    "issue_supplier_alert",
}
_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@pytest.fixture
def client():
    return TestClient(app)


def _far_tier(gt: str) -> str:
    """Return a risk tier >= 2 ordinal steps from `gt` (guaranteed match=0.0).

    Used to prove the analyst grader does REAL proximity scoring, not a canned
    constant: a tier two-or-more steps off must score strictly lower than exact.
    """
    idx = _RISK_ORDER[gt]
    return "LOW" if idx >= 2 else "CRITICAL"


def _long_completion() -> str:
    """A completion in the 30..400-token anti-hack bracket (real length reward)."""
    return (
        "The disruption materially threatens supply continuity across the "
        "exposed corridor. Rerouting, insurance premiums, and lead-time "
        "extension all compound the near-term financial impact, and the "
        "historical analog supports an elevated assessment for the planning "
        "horizon under consideration by the risk desk this quarter overall."
    )


# ==========================================================================
# THE SPINE — one ordered end-to-end journey
# ==========================================================================

def test_golden_path_spine(client, monkeypatch):
    """Walk the full product spine; assert real data (or truthful degradation)
    at every hop. Hops 1-4 share one episode; hops 5-7 are stateless."""
    # Force HuggingFace offline so the war-room analog matcher (hop 7) never
    # hits the network in CI — it degrades to its real TF-IDF fallback or a
    # locally-cached embedding model. This is the CI condition, made explicit.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # ---- HOP 1: SIGNAL — reset a seeded episode -------------------------
    r = client.post("/reset", params={"task_id": "easy_typhoon_response", "seed": 42})
    assert r.status_code == 200, f"/reset failed: {r.status_code} {r.text}"
    obs = r.json()
    for key in ("current_day", "days_remaining", "active_signals", "node_statuses",
                "financials", "situation_summary", "compact_summary"):
        assert key in obs, f"/reset observation missing real field: {key}"
    assert obs["current_day"] == 0
    assert obs["days_remaining"] == 30, "easy task episode length must be 30 days"
    assert isinstance(obs["active_signals"], list)
    assert isinstance(obs["node_statuses"], list) and len(obs["node_statuses"]) > 0
    assert isinstance(obs["financials"], dict) and len(obs["financials"]) > 0
    assert obs["compact_summary"].strip() != ""

    # ---- HOP 2: DECISION — a real action with real economic effect ------
    r = client.post("/step", json={
        "action_type": "activate_backup_supplier",
        "target_node_id": "SUP_TSMC",
        "backup_supplier_id": "SUP_SAMSUNG",
    })
    assert r.status_code == 200, f"/step failed: {r.status_code} {r.text}"
    step = r.json()
    assert isinstance(step["reward"], (int, float)), "reward must be a real number"
    assert isinstance(step["done"], bool)
    assert step["current_day"] == 1, "day must advance exactly one step"
    lar = step["last_action_result"]
    assert lar["success"] is True, f"healthy backup activation must succeed: {lar}"
    assert lar["cost"] > 0.0, "a real backup activation must incur a real cost"
    assert "Samsung" in lar["message"], "action message must describe the real effect"

    # ---- HOP 3: STATE — real episode metadata ---------------------------
    r = client.get("/state")
    assert r.status_code == 200, f"/state failed: {r.status_code} {r.text}"
    st = r.json()
    assert isinstance(st["episode_id"], str) and len(st["episode_id"]) > 0
    assert st["step_count"] == 1
    assert st["task_id"] == "easy_typhoon_response"
    assert isinstance(st["cumulative_reward"], (int, float))
    assert st["is_done"] is False

    # ---- HOP 4: GRADE — deterministic score in [0,1] --------------------
    for _ in range(3):
        client.post("/step", json={"action_type": "do_nothing"})
    r = client.post("/grader")
    assert r.status_code == 200, f"/grader failed: {r.status_code} {r.text}"
    g = r.json()
    assert 0.0 <= g["score"] <= 1.0, f"grader score out of bounds: {g['score']}"
    assert g["steps_taken"] >= 4
    weights = [c["weight"] for c in g["breakdown"].values()]
    assert abs(sum(weights) - 1.0) < 1e-6, f"grader component weights must sum to 1.0: {weights}"
    for comp in g["breakdown"].values():
        assert 0.0 <= comp["score"] <= 1.0

    # ---- HOP 5: ASSESSMENT — keyword rubric vs real R4 ground truth -----
    #     (works with NO OpenRouter key: this is the deterministic reward
    #      oracle scored against committed 3-judge ground truth.)
    sc = client.get("/analyst/scenarios")
    assert sc.status_code == 200, f"/analyst/scenarios failed: {sc.text}"
    scenarios = sc.json()["scenarios"]
    assert len(scenarios) > 0, "R4 ground-truth cache must expose scenarios"
    target = scenarios[0]
    sid, gt = target["scenario_id"], target["ground_truth"]
    assert gt in _RISK_ORDER, f"ground truth must be a real risk tier: {gt}"

    r = client.post("/analyst/grade", json={
        "scenario_id": sid,
        "assessment": {"risk_level": gt, "confidence": 0.8},
        "raw_completion": _long_completion(),
    })
    assert r.status_code == 200, f"/analyst/grade failed: {r.status_code} {r.text}"
    ag = r.json()
    assert ag["ground_truth_risk"] == gt, "ground truth must come from real R4 cache"
    assert ag["predicted_risk"] == gt
    assert ag["breakdown"]["match"] == 1.0, "exact-tier prediction must score match=1.0"
    assert ag["reward"] == pytest.approx(1.0), "exact match + valid format + ok length -> reward 1.0"
    assert ag["inference_type"] == "live_rubric_vs_r4_ground_truth"
    assert "R4_DANGEROUS_V2.json" in ag["scenario_source"], \
        "ground-truth provenance must point at the real R4 cache file"

    # ---- HOP 6: COUNTERFACTUAL — 4 real causal methods (R6) -------------
    r = client.post("/counterfactual/platinum", json={
        "task_id": "easy_typhoon_response", "severity_tier": "HIGH", "n_episodes_mc": 10,
    })
    assert r.status_code == 200, f"/counterfactual/platinum failed: {r.status_code} {r.text}"
    cf = r.json()
    for m in ("method_a_paired_bootstrap_mc", "method_b_synthetic_control",
              "method_c_bsts_lite", "method_d_scm_dowhy_proxy"):
        assert m in cf, f"counterfactual missing real method: {m}"
        assert "point_usd" in cf[m] and "ci95_usd" in cf[m]
    cons = cf["consensus"]
    assert cons["point_usd"] > 0.0, "consensus dollar estimate must be a real positive number"
    lo, hi = cons["ci95_usd"]
    assert lo <= hi, f"consensus CI must be ordered: [{lo}, {hi}]"
    assert cons["n_methods"] >= 2, "at least 2 methods must produce real estimates"
    assert 0.0 <= cons["method_agreement"] <= 1.0
    n_real = sum(1 for m in ("method_a_paired_bootstrap_mc", "method_b_synthetic_control",
                             "method_c_bsts_lite", "method_d_scm_dowhy_proxy")
                 if cf[m]["point_usd"] != 0.0)
    assert n_real >= 2, "counterfactual must be computed, not canned"
    assert isinstance(cf["paper_anchors"], list) and len(cf["paper_anchors"]) > 0
    assert cf["inference_type"] == "platinum_4method_consensus_no_magic_constants"

    # ---- HOP 7: WAR ROOM — renders live; honest degraded-LLM flag -------
    r = client.post("/demo/hormuz-war-room", json={"enable_openrouter_panel": False})
    assert r.status_code == 200, f"/demo/hormuz-war-room failed: {r.status_code} {r.text}"
    wr = r.json()
    # real cited chokepoint facts
    facts = wr["live_facts_chokepoint"]
    assert isinstance(facts, list) and len(facts) > 0
    assert all("source" in f and f["source"].startswith("http") for f in facts), \
        "every headline fact must cite a real source URL"
    # real deterministic sector tables (numbers, not placeholders)
    assert isinstance(wr["india_impact_table"], list) and len(wr["india_impact_table"]) > 0
    assert isinstance(wr["gulf_impact_table"], list) and len(wr["gulf_impact_table"]) > 0
    # real chokepoint graph
    assert len(wr["chokepoint_graph"]["nodes"]) > 0
    assert len(wr["chokepoint_graph"]["edges"]) > 0
    # sha256 receipt (64 hex chars)
    receipt = wr["receipt_sha256"]
    assert isinstance(receipt, str) and len(receipt) == 64
    assert all(ch in "0123456789abcdef" for ch in receipt)
    # HONEST DEGRADED LLM FLAG — truthful, not fabricated
    ollama_up = wr["live_pipeline"]["ollama_available"]
    assert isinstance(ollama_up, bool)
    flag = wr["data_source_flags"]["live_pipeline"]
    if ollama_up:
        assert flag == "live_llm_panel"
    else:
        # No OpenRouter key + no Ollama: the pipeline MUST label itself as the
        # deterministic rubric fallback (real, rule-based) — never pretend an
        # LLM produced the verdict.
        assert flag == "deterministic_rubric_fallback", \
            f"degraded LLM flag must be truthful, got: {flag}"
        judges = wr["live_pipeline"]["judges"]
        assert len(judges) >= 1
        assert any("rubric" in (j.get("name", "").lower()) for j in judges), \
            "degraded judge panel must be honestly labeled as a rubric fallback"


# ==========================================================================
# FALSIFIABILITY — prove the real hops are real, not canned
# ==========================================================================

def test_analyst_grade_is_real_scoring_not_canned(client):
    """A tier >=2 steps off must score STRICTLY lower than an exact match.
    A canned endpoint returning a constant reward would fail this."""
    scenarios = client.get("/analyst/scenarios").json()["scenarios"]
    target = scenarios[0]
    sid, gt = target["scenario_id"], target["ground_truth"]

    exact = client.post("/analyst/grade", json={
        "scenario_id": sid, "assessment": {"risk_level": gt, "confidence": 0.8},
        "raw_completion": _long_completion(),
    }).json()
    wrong = client.post("/analyst/grade", json={
        "scenario_id": sid,
        "assessment": {"risk_level": _far_tier(gt), "confidence": 0.8},
        "raw_completion": _long_completion(),
    }).json()

    assert exact["breakdown"]["match"] == 1.0
    assert wrong["breakdown"]["match"] == 0.0
    assert exact["reward"] > wrong["reward"], (
        "grader must reward proximity to the real ground truth, not return a "
        f"constant: exact={exact['reward']} wrong={wrong['reward']}"
    )


def test_counterfactual_methods_are_distinct_real_estimates(client):
    """The 4 causal methods estimate different scopes and must produce distinct
    dollar figures — a fabricated 'consensus' would collapse them to one number.
    The honest low agreement (orders-of-magnitude spread) must be surfaced."""
    cf = client.post("/counterfactual/platinum", json={
        "task_id": "easy_typhoon_response", "severity_tier": "HIGH", "n_episodes_mc": 10,
    }).json()
    points = sorted({
        cf[m]["point_usd"]
        for m in ("method_a_paired_bootstrap_mc", "method_b_synthetic_control",
                  "method_c_bsts_lite", "method_d_scm_dowhy_proxy")
        if cf[m]["point_usd"] != 0.0
    })
    assert len(points) >= 2, "distinct real methods must yield distinct estimates"
    # widely-scattered estimates -> honest low agreement, not a manufactured 1.0
    assert cf["consensus"]["method_agreement"] < 1.0, \
        "method agreement must be honestly computed, not asserted as perfect"
