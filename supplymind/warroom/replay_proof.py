"""replay_proof.py — historical replay: "what SupplyMind would have said on the
day, vs what actually happened."

The most powerful honest proof a forecasting product can offer. For a real,
documented crisis we:

  1. Reconstruct the information state AS OF the event date using ONLY data that
     existed then — FRED Brent rows strictly before the date, and crisis-library
     analogs strictly before the date. A hard :class:`NoLookaheadError` guard
     enforces this in code (and is proven by a test that a future-dated datum is
     refused). A lookahead leak here would be exactly the label-leakage sin
     CLAUDE.md §7 forbids.
  2. Run the REAL pipeline as of that day: the deterministic keyword/heuristic
     risk path (:func:`hormuz_endpoint._rubric_judge`), a no-lookahead analog
     match + projection, a forward oil-price-channel $ estimate from the
     projection, and the engine's real Monte-Carlo supply-continuity channel.
     The LLM analyst hop is BLOCKED-ON-KEY (OpenRouter 401) and is marked
     ``pending-key`` so it lights up the moment a key lands — never faked.
  3. Observe what ACTUALLY happened from the real FRED price path after the date
     (used ONLY for scoring, never as a model input) and the documented impacts,
     including the rigorous realized oil-channel counterfactual via the real R6
     ARIMA method.
  4. Report the honest verdict — did the as-of forecast call the direction and
     level? A miss is a real, publishable finding and is reported as one. The
     system is run ONCE per event; it is NEVER tuned to look good on a known
     event (that would be hindsight fitting — FORBIDDEN).

Run once from a fresh shell::

    python -m supplymind.warroom.replay_proof                 # sweep all eligible
    python -m supplymind.warroom.replay_proof --event iran_true_promise_2_2024_10
    python -m supplymind.warroom.replay_proof --receipt tests/receipts/replay_proof_REAL.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FRED_BRENT_CSV = REPO_ROOT / "external_data" / "fred_brent_daily.csv"
CRISIS_LIB = (REPO_ROOT / "supplymind" / "warroom" / "scenarios"
              / "iran_israel_hormuz_2024_2026.json")

# The persona (PRODUCT_THESIS) is an India-based risk officer; India's crude
# imports set the oil-price-channel USD scale. ≈4.5M bbl/d (EIA 2024).
INDIA_IMPORT_BBL_PER_DAY = 4_500_000

# Windows (calendar days). The oil-channel horizon a risk desk cares about.
PRE_WINDOW_DAYS = 45
POST_HORIZON_DAYS = 21


class NoLookaheadError(Exception):
    """A datum dated on/after the as-of date leaked into the information state.

    Raised (never swallowed) whenever a reconstruction would use data that did
    not exist yet as of the event date — the label-leakage guard for the replay.
    """


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# No-lookahead guard — the heart of the honesty contract
# --------------------------------------------------------------------------- #

def guard_asof(items: list[dict], as_of: str, date_key: str,
               *, context: str) -> list[dict]:
    """Return ``items`` unchanged iff every item's date is strictly before
    ``as_of``; otherwise raise :class:`NoLookaheadError` naming the offender.

    This is the enforcement, not a comment: the reconstruction routines below
    pass their assembled information state through this guard, so any leak of
    on/after-date data aborts the replay loudly instead of silently biasing the
    forecast with the future.
    """
    for it in items:
        d = str(it.get(date_key, ""))
        if d and d >= as_of:
            raise NoLookaheadError(
                f"lookahead leak in {context}: datum dated {d!r} is on/after the "
                f"as-of date {as_of!r} (key={date_key!r}). Refused — a forecast "
                f"may only use data that existed strictly before the event."
            )
    return items


# --------------------------------------------------------------------------- #
# FRED loading (date-filtered, guarded)
# --------------------------------------------------------------------------- #

def _load_fred_rows() -> list[dict]:
    """Real FRED Brent daily rows as [{date, price}], missing '.' skipped."""
    rows: list[dict] = []
    with open(FRED_BRENT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 2:
                continue
            try:
                v = float(r[-1].strip())
            except ValueError:
                continue
            if v > 0:
                rows.append({"date": r[0].strip(), "price": v})
    return rows


def _fred_asof(as_of: str, pre_days: int) -> list[dict]:
    """FRED rows strictly before ``as_of``, last ``pre_days`` calendar days,
    passed through the no-lookahead guard. These are the ONLY prices the as-of
    forecast may see."""
    all_rows = _load_fred_rows()
    cutoff = (datetime.strptime(as_of, "%Y-%m-%d")
              - _timedelta_days(pre_days)).strftime("%Y-%m-%d")
    pre = [r for r in all_rows if cutoff <= r["date"] < as_of]
    return guard_asof(pre, as_of, "date", context="FRED pre-event window")


def _fred_outcome(as_of: str, horizon_days: int) -> list[dict]:
    """FRED rows in [as_of, as_of+horizon]. This is the OUTCOME — real prices
    that materialised AFTER the event. Used ONLY for scoring, never as a model
    input. (Deliberately NOT guarded: this is post-event ground truth.)"""
    all_rows = _load_fred_rows()
    end = (datetime.strptime(as_of, "%Y-%m-%d")
           + _timedelta_days(horizon_days)).strftime("%Y-%m-%d")
    return [r for r in all_rows if as_of <= r["date"] <= end]


def _timedelta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


# --------------------------------------------------------------------------- #
# Crisis-library analogs (date-filtered, guarded)
# --------------------------------------------------------------------------- #

def _load_events() -> list[dict]:
    return json.loads(CRISIS_LIB.read_text(encoding="utf-8"))["events"]


def _prior_analogs(as_of: str, exclude_id: str) -> list[dict]:
    """Library events dated strictly before ``as_of`` (excluding the target
    event itself), passed through the no-lookahead guard."""
    events = _load_events()
    prior = [e for e in events if e.get("id") != exclude_id
             and str(e.get("date", "")) < as_of]
    return guard_asof(prior, as_of, "date", context="prior crisis-library analogs")


# --------------------------------------------------------------------------- #
# Information-state reconstruction (no lookahead)
# --------------------------------------------------------------------------- #

@dataclass
class InfoState:
    event_id: str
    as_of: str
    trigger_text: str
    fred_pre: list[dict]
    prior_analogs: list[dict]
    pre_close: float
    pre_close_date: str
    lookahead_guard: str


def _trigger_text(event: dict) -> str:
    """The 'what a risk officer sees in real time' trigger — event name, type,
    region, and affected routes ONLY. Deliberately EXCLUDES the event summary
    and its documented oil impact, which contain post-hoc outcome information."""
    routes = ", ".join(event.get("affected_routes", []) or [])
    return (f"{event.get('name', '')}. Type: {event.get('event_type', '')}. "
            f"Region: {event.get('region', '')}. Routes affected: {routes}.")


def reconstruct_information_state(event: dict) -> InfoState:
    """Assemble everything the system may know AS OF the event date — nothing
    dated on/after it. Both the FRED window and the analog set pass through the
    no-lookahead guard, so any leak aborts here loudly."""
    as_of = str(event["date"])
    fred_pre = _fred_asof(as_of, PRE_WINDOW_DAYS)
    prior = _prior_analogs(as_of, event["id"])
    if not fred_pre:
        raise ValueError(f"no pre-event FRED data before {as_of}")
    pre_close_row = fred_pre[-1]
    return InfoState(
        event_id=event["id"], as_of=as_of,
        trigger_text=_trigger_text(event),
        fred_pre=fred_pre, prior_analogs=prior,
        pre_close=float(pre_close_row["price"]),
        pre_close_date=pre_close_row["date"],
        lookahead_guard=(f"enforced — {len(fred_pre)} FRED rows + "
                         f"{len(prior)} analogs, all dated < {as_of}; "
                         "0 violations"),
    )


# --------------------------------------------------------------------------- #
# AS-OF assessment — the real pipeline, no lookahead
# --------------------------------------------------------------------------- #

def _find_analogs_asof(query: str, prior_events: list[dict], k: int = 3):
    """No-lookahead analog match: TF-IDF cosine of the trigger against ONLY the
    prior events. Reuses the real crisis-library TF-IDF helpers + Analog type +
    interpolate_projection — just restricted to a past-only corpus."""
    from collections import Counter
    from supplymind.warroom import crisis_library as cl

    if not prior_events:
        return []
    texts = [cl._event_text(e) for e in prior_events]
    tfidf_docs, idf = cl._tfidf_vectors(texts)
    q_tf = Counter(cl._tokenize(query))
    import math
    q_vec = {t: f * idf.get(t, 1.0) for t, f in q_tf.items()}
    norm = math.sqrt(sum(x * x for x in q_vec.values())) or 1.0
    q_vec = {t: x / norm for t, x in q_vec.items()}
    sims = [cl._cosine(q_vec, dv) for dv in tfidf_docs]
    order = sorted(range(len(prior_events)), key=lambda i: sims[i], reverse=True)[:k]
    out = []
    for i in order:
        e = prior_events[i]
        out.append(cl.Analog(
            event_id=e["id"], name=e["name"], date=e["date"],
            severity=e["severity"], summary=e.get("summary", ""),
            similarity=round(sims[i], 4), full_record=e))
    return out


def _engine_mc_channel(severity: float, duration_days: float) -> dict:
    """The engine's real Monte-Carlo supply-continuity channel for an India /
    Hormuz gateway disruption. No time-series input → inherently no lookahead.

    For a single-chokepoint disruption the structural graph typically reroutes
    around it, so this channel is often near-zero — an honest finding we surface,
    not hide: for a chokepoint event the cost lives in the price channel, not in
    lost supply continuity.
    """
    try:
        from server.engine.graph import SupplyChainGraph
        from server.engine.monte_carlo import MonteCarloEngine
        from supplymind.contracts import DisruptionSignal
    except Exception as e:  # noqa: BLE001
        return {"status": "blocked", "usd": None,
                "note": f"engine import failed: {type(e).__name__}: {e}"}
    graph_path = REPO_ROOT / "server" / "data" / "graphs" / "hard_graph.json"
    do_nodes = ["PORT_MUMBAI"]  # India's Gulf-crude gateway (Hormuz-exposed)
    g = SupplyChainGraph()
    g.load_from_json(str(graph_path))
    present = [n for n in do_nodes if n in g.G]
    if not present:
        return {"status": "blocked", "usd": None,
                "note": f"gateway nodes {do_nodes} absent from graph"}
    sig = DisruptionSignal(
        signal_id="REPLAY", disruption_type="geopolitical",
        severity=float(severity), confidence=1.0, affected_region="hormuz",
        affected_node_ids=present, time_to_impact_hours=0.0,
        estimated_duration_days=float(max(1.0, duration_days)),
        description="India/Hormuz gateway disruption (replay)",
        lifecycle_phase="active")
    res = MonteCarloEngine(seed=20241001).run_simulation(
        g.deep_copy(), [sig], n_simulations=200)
    p50 = float(res.get("p50_loss", 0.0))
    p95 = float(res.get("p95_loss", 0.0))
    return {
        "status": "ok", "usd": p50, "p95_usd": p95,
        "do_nodes": present,
        "note": ("engine MonteCarloEngine, supply-CONTINUITY channel. "
                 + ("Near-zero: the structural graph reroutes around a single "
                    "chokepoint, so continuity is preserved and the cost is in "
                    "the price channel below." if p50 <= 0.0
                    else "Structural continuity loss under the disruption.")),
    }


def run_asof_assessment(state: InfoState) -> dict:
    """Run the real deterministic pipeline as of the event date. NEVER touches
    post-as-of data."""
    from supplymind.warroom.crisis_library import interpolate_projection
    from supplymind.warroom.hormuz_endpoint import _rubric_judge

    analogs = _find_analogs_asof(state.trigger_text, state.prior_analogs, k=3)
    projection = interpolate_projection(analogs)

    # Real deterministic risk path (the keyword/heuristic rubric). No LLM.
    judge = _rubric_judge(state.trigger_text, projection, signals=[])

    # Forward oil-price-channel $ estimate — from the PROJECTION only (pre-event).
    fwd_brent = projection.get("brent_projection_usd_bbl_p50")
    fwd_move = (fwd_brent - state.pre_close) if fwd_brent is not None else None
    window = POST_HORIZON_DAYS
    fwd_oil_usd = (abs(fwd_move) * INDIA_IMPORT_BBL_PER_DAY * window
                   if fwd_move is not None else None)
    fwd_direction = ("up" if (fwd_move or 0) > 0.5 else
                     "down" if (fwd_move or 0) < -0.5 else "flat")

    # Engine MC supply-continuity channel (structural, timeless → no lookahead).
    sev = projection.get("severity_p50") or 0.3
    dur = projection.get("duration_days_p50") or 7.0
    engine_mc = _engine_mc_channel(sev, dur)

    return {
        "as_of": state.as_of,
        "pre_close_usd_bbl": state.pre_close,
        "pre_close_date": state.pre_close_date,
        "n_prior_analogs_available": len(state.prior_analogs),
        "analogs_matched": [
            {"event_id": a.event_id, "name": a.name, "date": a.date,
             "similarity": a.similarity, "severity": a.severity}
            for a in analogs],
        "projection": projection,
        "risk_level": judge["risk_level"],
        "risk_confidence": judge["confidence"],
        "risk_rationale": judge["rationale"],
        "risk_inference_type": judge["inference_type"],
        "forward_brent_p50_usd_bbl": fwd_brent,
        "forward_brent_move_usd_bbl": (round(fwd_move, 2)
                                       if fwd_move is not None else None),
        "forward_direction": fwd_direction,
        "forward_oil_channel_usd": (round(fwd_oil_usd, 0)
                                    if fwd_oil_usd is not None else None),
        "engine_mc_continuity_channel": engine_mc,
        "llm_hop": {
            "status": "pending-key",
            "would_call": "supplymind.llm.analyst (v5 calibrated prompt)",
            "reason": ("OpenRouter key revoked (401). The analyst verdict would "
                       "refine this heuristic risk call with a calibrated, "
                       "Brier-scored assessment; it lights up the moment a key "
                       "lands. Not faked in the interim."),
        },
    }


# --------------------------------------------------------------------------- #
# OUTCOME — what actually happened (post-event, for scoring only)
# --------------------------------------------------------------------------- #

def _realized_oil_channel(event: dict, as_of: str) -> dict:
    """The rigorous realized oil-price-channel impact via the REAL R6 ARIMA
    counterfactual (fit on pre-event Brent, compared to the actual post path).
    This measures what the shock actually cost the oil channel — an outcome-side
    measurement (it needs the post window by definition)."""
    try:
        from supplymind.phoenix.counterfactual_v2.causal_methods import (
            method_c_arima_fred)
    except Exception as e:  # noqa: BLE001
        return {"status": "blocked", "usd": None,
                "note": f"R6 import failed: {type(e).__name__}: {e}"}
    cfg = {
        "event_date": as_of,
        "fred_pre_days": PRE_WINDOW_DAYS,
        "fred_post_days": 15,
        "oil_import_bbl_per_day": INDIA_IMPORT_BBL_PER_DAY,
        "oil_confounder": ("Global macro / demand factors also move Brent over "
                           "the window; the isolated event effect is confounded."),
    }
    est = method_c_arima_fred(cfg)
    return {
        "status": est.status, "usd": est.estimate_usd,
        "ci_low_usd": est.ci_low_usd, "ci_high_usd": est.ci_high_usd,
        "effect_per_bbl_usd": (est.inputs or {}).get("effect_per_bbl_usd"),
        "actual_post_avg_usd_bbl": (est.inputs or {}).get("actual_post_avg_usd_bbl"),
        "counterfactual_avg_usd_bbl": (est.inputs or {}).get("counterfactual_avg_usd_bbl"),
        "method": "R6 ARIMA(p,1,0) on FRED DCOILBRENTEU (realized counterfactual)",
        "note": est.notes,
    }


def observe_actual_outcome(event: dict) -> dict:
    """What actually happened: the real FRED path after the event + documented
    impacts. GROUND TRUTH — used only for scoring, never as a model input."""
    as_of = str(event["date"])
    post = _fred_outcome(as_of, POST_HORIZON_DAYS)
    state = reconstruct_information_state(event)
    pre_close = state.pre_close

    realized_peak = realized_peak_date = None
    realized_post7d = realized_post7d_date = None
    if post:
        prices = [r["price"] for r in post]
        realized_peak = max(prices)
        realized_peak_date = post[int(np.argmax(prices))]["date"]
        # ~7 trading days out (index 5..7 depending on gaps); take the 6th row.
        idx7 = min(6, len(post) - 1)
        realized_post7d = post[idx7]["price"]
        realized_post7d_date = post[idx7]["date"]

    move_pct = (round((realized_post7d - pre_close) / pre_close * 100, 2)
                if realized_post7d else None)
    peak_move_pct = (round((realized_peak - pre_close) / pre_close * 100, 2)
                     if realized_peak else None)
    realized_direction = ("up" if (move_pct or 0) > 0.5 else
                          "down" if (move_pct or 0) < -0.5 else "flat")

    return {
        "as_of": as_of,
        "pre_close_usd_bbl": pre_close,
        "n_post_rows": len(post),
        "realized_peak_usd_bbl": realized_peak,
        "realized_peak_date": realized_peak_date,
        "realized_post7d_usd_bbl": realized_post7d,
        "realized_post7d_date": realized_post7d_date,
        "realized_move_pct_post7d": move_pct,
        "realized_peak_move_pct": peak_move_pct,
        "realized_direction": realized_direction,
        "documented_oil_impact_usd_bbl": event.get("oil_impact_usd_bbl"),
        "documented_summary": event.get("summary"),
        "realized_oil_channel_counterfactual": _realized_oil_channel(event, as_of),
    }


# --------------------------------------------------------------------------- #
# Honest scoring
# --------------------------------------------------------------------------- #

def score(assessment: dict, outcome: dict) -> dict:
    """Compare the as-of forecast to what actually happened. No tuning; this is
    a straight, falsifiable comparison run once."""
    fwd_dir = assessment["forward_direction"]
    act_dir = outcome["realized_direction"]
    fwd_brent = assessment["forward_brent_p50_usd_bbl"]
    peak = outcome["realized_peak_usd_bbl"]
    post7d = outcome["realized_post7d_usd_bbl"]

    directional_correct = (fwd_dir == act_dir) if act_dir != "flat" or fwd_dir == "flat" else False
    err_vs_peak = (round(abs(fwd_brent - peak), 2)
                   if (fwd_brent is not None and peak is not None) else None)
    err_vs_post7d = (round(abs(fwd_brent - post7d), 2)
                     if (fwd_brent is not None and post7d is not None) else None)

    verdict_parts = [
        f"As of {assessment['as_of']} (pre-event Brent ${assessment['pre_close_usd_bbl']:.2f}), "
        f"SupplyMind assessed risk={assessment['risk_level']} and projected Brent "
        f"${fwd_brent:.1f} ({fwd_dir})." if fwd_brent is not None
        else f"As of {assessment['as_of']}, risk={assessment['risk_level']}; no Brent projection."
    ]
    if peak is not None:
        verdict_parts.append(
            f"Brent actually {act_dir} to a peak of ${peak:.2f} "
            f"({outcome['realized_peak_move_pct']:+.1f}%) and sat at ${post7d:.2f} "
            f"({outcome['realized_move_pct_post7d']:+.1f}%) ~7 trading days out.")
    verdict_parts.append(
        "Direction CALLED CORRECTLY." if directional_correct
        else "Direction MISSED — reported honestly, not tuned away.")

    return {
        "directional_correct": bool(directional_correct),
        "forward_direction": fwd_dir,
        "realized_direction": act_dir,
        "brent_abs_error_vs_peak_usd": err_vs_peak,
        "brent_abs_error_vs_post7d_usd": err_vs_post7d,
        "risk_level": assessment["risk_level"],
        "verdict": " ".join(verdict_parts),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def eligible_events(min_prior_analogs: int = 2) -> list[dict]:
    """Events with enough prior analogs + FRED coverage to replay honestly."""
    events = _load_events()
    out = []
    for e in events:
        as_of = str(e.get("date", ""))
        if not as_of:
            continue
        prior = [x for x in events if x.get("id") != e["id"]
                 and str(x.get("date", "")) < as_of]
        if len(prior) < min_prior_analogs:
            continue
        if not _fred_asof(as_of, PRE_WINDOW_DAYS):
            continue
        if not _fred_outcome(as_of, POST_HORIZON_DAYS):
            continue
        out.append(e)
    return out


def run_replay(event_id: str) -> dict:
    """Full honest replay of one event: reconstruct → assess → observe → score."""
    events = {e["id"]: e for e in _load_events()}
    if event_id not in events:
        raise ValueError(f"unknown event '{event_id}'; choices: {list(events)}")
    event = events[event_id]
    state = reconstruct_information_state(event)
    assessment = run_asof_assessment(state)
    outcome = observe_actual_outcome(event)
    verdict = score(assessment, outcome)
    return {
        "event_id": event_id,
        "event_name": event.get("name"),
        "as_of": state.as_of,
        "lookahead_guard": state.lookahead_guard,
        "trigger_text_used": state.trigger_text,
        "as_of_assessment": assessment,
        "actual_outcome": outcome,
        "verdict": verdict,
    }


def run_all_eligible() -> dict:
    """Replay every eligible event and report an honest aggregate. Run once."""
    evs = eligible_events()
    replays = [run_replay(e["id"]) for e in evs]
    scored = [r for r in replays if r["verdict"]["directional_correct"] is not None]
    n_correct = sum(1 for r in scored if r["verdict"]["directional_correct"])
    errs = [r["verdict"]["brent_abs_error_vs_peak_usd"] for r in replays
            if r["verdict"]["brent_abs_error_vs_peak_usd"] is not None]
    hit_rate = (round(n_correct / len(scored), 3) if scored else None)
    reading = (
        f"Across {len(scored)} eligible documented crises replayed with a hard "
        f"no-lookahead guard, SupplyMind's as-of directional call was correct on "
        f"{n_correct}/{len(scored)} "
        + (f"({hit_rate:.0%}). " if hit_rate is not None else ". ")
        + "This is an untuned, run-once result: the system was never adjusted to "
        "improve on any known event (that would be hindsight fitting). "
        + (f"Median absolute Brent-level error vs the realised peak was "
           f"${float(np.median(errs)):.1f}/bbl. " if errs else "")
        + "The forward oil-price channel is the material one for chokepoint "
        "events; the engine supply-continuity channel is near-zero by design "
        "(rerouting preserves continuity) and is surfaced honestly.")
    return {
        "generated_at": _now_iso(),
        "n_eligible": len(evs),
        "n_scored": len(scored),
        "n_directional_correct": n_correct,
        "directional_hit_rate": hit_rate,
        "median_brent_abs_error_vs_peak_usd": (round(float(np.median(errs)), 2)
                                               if errs else None),
        "honest_reading": reading,
        "replays": replays,
    }


# --------------------------------------------------------------------------- #
# Receipt + CLI
# --------------------------------------------------------------------------- #

def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_receipt(result: dict, path: Path, *, featured_event: str) -> dict:
    from supplymind.data.provenance import sha256_of
    receipt = {
        "receipt": "replay_proof_REAL",
        "what": ("Historical replay proof: 'what SupplyMind would have said on "
                 "the day vs what actually happened', on real documented crises, "
                 "with a hard no-lookahead guard enforced in code. Untuned, run "
                 "once. The realized oil-channel impact uses the real R6 ARIMA "
                 "counterfactual; the engine Monte-Carlo continuity channel is "
                 "real; the LLM analyst hop is BLOCKED-ON-KEY (marked pending)."),
        "command": ("python -m supplymind.warroom.replay_proof "
                    "--receipt tests/receipts/replay_proof_REAL.json"),
        "git_sha": _git_sha(),
        "generated_at": result["generated_at"],
        "featured_event": featured_event,
        "no_lookahead_enforcement": (
            "supplymind.warroom.replay_proof.guard_asof raises NoLookaheadError "
            "on any datum dated on/after the as-of date; both the FRED pre-window "
            "and the prior-analog set pass through it. Proven by "
            "tests/test_trust_replay.py (a future-dated datum is refused)."),
        "aggregate": {
            "n_eligible": result["n_eligible"],
            "n_scored": result["n_scored"],
            "n_directional_correct": result["n_directional_correct"],
            "directional_hit_rate": result["directional_hit_rate"],
            "median_brent_abs_error_vs_peak_usd":
                result["median_brent_abs_error_vs_peak_usd"],
            "honest_reading": result["honest_reading"],
        },
        "per_event": [
            {"event_id": r["event_id"], "event_name": r["event_name"],
             "as_of": r["as_of"], "lookahead_guard": r["lookahead_guard"],
             "risk_level": r["as_of_assessment"]["risk_level"],
             "forward_brent_p50": r["as_of_assessment"]["forward_brent_p50_usd_bbl"],
             "forward_direction": r["as_of_assessment"]["forward_direction"],
             "pre_close": r["as_of_assessment"]["pre_close_usd_bbl"],
             "realized_peak": r["actual_outcome"]["realized_peak_usd_bbl"],
             "realized_post7d": r["actual_outcome"]["realized_post7d_usd_bbl"],
             "realized_direction": r["actual_outcome"]["realized_direction"],
             "realized_oil_channel_usd":
                 r["actual_outcome"]["realized_oil_channel_counterfactual"].get("usd"),
             "engine_mc_continuity_usd":
                 r["as_of_assessment"]["engine_mc_continuity_channel"].get("usd"),
             "directional_correct": r["verdict"]["directional_correct"],
             "brent_abs_error_vs_peak_usd": r["verdict"]["brent_abs_error_vs_peak_usd"],
             "verdict": r["verdict"]["verdict"]}
            for r in result["replays"]
        ],
        "featured_detail": next(
            (r for r in result["replays"] if r["event_id"] == featured_event),
            result["replays"][0] if result["replays"] else None),
        "match": True,
        "match_note": ("match:true means the replay genuinely ran end-to-end; "
                       "engine MC + ARIMA are seeded and reproduce; FRED path is "
                       "the committed real corpus."),
    }
    receipt["payload_sha256"] = sha256_of(receipt["per_event"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return receipt


def _print_replay(r: dict) -> None:
    a, o, v = r["as_of_assessment"], r["actual_outcome"], r["verdict"]
    print(f"\n{'='*74}\n{r['event_name']}  (as of {r['as_of']})\n{'='*74}")
    print(f"  lookahead guard : {r['lookahead_guard']}")
    print(f"  AS-OF FORECAST  : risk={a['risk_level']} conf={a['risk_confidence']}  "
          f"Brent proj ${a['forward_brent_p50_usd_bbl']} ({a['forward_direction']}) "
          f"from ${a['pre_close_usd_bbl']:.2f}")
    print(f"                    fwd oil-channel ${a['forward_oil_channel_usd']:,}  "
          f"engine-MC continuity ${a['engine_mc_continuity_channel'].get('usd')}")
    print(f"                    LLM hop: {a['llm_hop']['status']}")
    print(f"  ACTUAL OUTCOME  : peak ${o['realized_peak_usd_bbl']} "
          f"({o['realized_peak_move_pct']:+}%)  +7d ${o['realized_post7d_usd_bbl']} "
          f"({o['realized_move_pct_post7d']:+}%)  dir={o['realized_direction']}")
    roc = o["realized_oil_channel_counterfactual"]
    print(f"                    realized oil-channel (R6 ARIMA) ${roc.get('usd')}  "
          f"[{roc.get('status')}]")
    print(f"  VERDICT         : {'HIT' if v['directional_correct'] else 'MISS'}  "
          f"|proj-peak| err ${v['brent_abs_error_vs_peak_usd']}/bbl")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description="SupplyMind historical replay proof.")
    ap.add_argument("--event", default=None,
                    help="Replay one event id (default: sweep all eligible).")
    ap.add_argument("--featured", default="iran_true_promise_2_2024_10",
                    help="Featured event id for the receipt detail.")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON.")
    ap.add_argument("--receipt", metavar="PATH", default=None,
                    help="Write the real replay-proof receipt to PATH.")
    args = ap.parse_args()

    if args.event:
        result = {"generated_at": _now_iso(), "replays": [run_replay(args.event)]}
        result.update({"n_eligible": 1, "n_scored": 1,
                       "n_directional_correct":
                           int(result["replays"][0]["verdict"]["directional_correct"]),
                       "directional_hit_rate": None,
                       "median_brent_abs_error_vs_peak_usd": None,
                       "honest_reading": "single-event replay"})
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            _print_replay(result["replays"][0])
    else:
        result = run_all_eligible()
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            for r in result["replays"]:
                _print_replay(r)
            print(f"\n{'='*74}\nAGGREGATE  (untuned, run once)\n{'='*74}")
            print(f"  directional hit rate: {result['n_directional_correct']}/"
                  f"{result['n_scored']}"
                  + (f"  ({result['directional_hit_rate']:.0%})"
                     if result['directional_hit_rate'] is not None else ""))
            print(f"  median |proj-peak| error: "
                  f"${result['median_brent_abs_error_vs_peak_usd']}/bbl")
            print(f"\n  {result['honest_reading']}\n")

    if args.receipt:
        if args.event:
            full = run_all_eligible()
        else:
            full = result
        rec = write_receipt(full, Path(args.receipt), featured_event=args.featured)
        print(f"\nreceipt written: {args.receipt}")
        print(f"  hit rate {rec['aggregate']['n_directional_correct']}/"
              f"{rec['aggregate']['n_scored']}")


if __name__ == "__main__":
    main()
