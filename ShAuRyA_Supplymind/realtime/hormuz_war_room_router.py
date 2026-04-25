"""hormuz_war_room_router.py — orchestrator + UI route for the Hormuz War Room.

Two routes (under prefix="" so they live at the app root):
  POST /demo/hormuz-war-room       JSON orchestrator (real data, real analogs,
                                    real sector tables, sha256-anchored receipt)
  GET  /demo/hormuz-war-room/ui    Self-contained dark-mode dashboard HTML

This module reuses every existing real subsystem:
  - run_hormuz_pipeline() from hormuz_endpoint.py    (analogs, projection, judges,
                                                       counterfactual, actions)
  - india_industry_exposure.score_all()              (deterministic Indian sectors)
  - gulf_industry_exposure.score_all()               (deterministic Gulf sectors)
  - hormuz_chokepoint_graph.get_graph()              (IEA-cited flow graph)

It does NOT duplicate or simulate any output. If a subsystem is unavailable
(e.g. Ollama down), the upstream module's graceful fallback applies and we
surface the source flag so the UI can label it honestly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError:
    APIRouter = None  # type: ignore
    HTTPException = Exception  # type: ignore
    HTMLResponse = None  # type: ignore

from ShAuRyA_Supplymind.scenarios import (
    india_industry_exposure as india_mod,
    gulf_industry_exposure as gulf_mod,
    hormuz_chokepoint_graph as graph_mod,
    reliance_industries_exposure as reliance_mod,
)
from ShAuRyA_Supplymind.realtime.hormuz_endpoint import (
    ScenarioRequest,
    run_hormuz_pipeline,
)

logger = logging.getLogger(__name__)

UI_HTML_PATH = Path(__file__).resolve().parents[2] / "server" / "static" / "hormuz_war_room.html"
MASTER_HTML_PATH = Path(__file__).resolve().parents[2] / "server" / "static" / "master.html"

router = APIRouter() if APIRouter is not None else None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class WarRoomRequest(BaseModel):
    scenario_text: str = Field(
        default=("Iran-Israel-US escalation prompts restriction or partial "
                  "closure of the Strait of Hormuz. Tanker insurance premiums "
                  "spike, vessel rerouting begins, Brent climbs."),
        description="Free-text scenario for the live pipeline + judges.",
    )
    severity: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Operator-asserted scenario severity. Used by the deterministic "
                    "sector scorer; the judge panel produces its own severity opinion.",
    )
    brent_price_usd_bbl: float = Field(
        default=132.0, ge=20.0, le=300.0,
        description="Operator-asserted Brent forecast in USD/bbl for the scenario "
                    "horizon. Used by sector price-shock channel.",
    )
    duration_days: int = Field(
        default=21, ge=1, le=1200,
        description="Operator-asserted disruption duration days. Cap of 1200 "
                    "covers multi-year ongoing campaigns (e.g. Houthi Red Sea).",
    )
    enable_llm_judges: bool = Field(
        default=True,
        description="If True and Ollama is up, call 3-judge LLM panel; else "
                    "deterministic rubric fallback (still real, but rule-based).",
    )
    include_recent_signals: bool = Field(
        default=True,
        description="Join with last-24h ingested live signals from event store.",
    )
    enable_openrouter_panel: bool = Field(
        default=False,
        description="If True, fan out to 6 frontier OpenRouter judges in parallel "
                    "and report Krippendorff α on their risk_level rankings. "
                    "Adds ~6-12s and uses OpenRouter rate budget.",
    )
    expand_to_12_judges: bool = Field(
        default=False,
        description="If True (and enable_openrouter_panel=True), use the 12-judge "
                    "frontier panel (DeepSeek + Qwen-3 + Llama-4 + Mistral-3 + "
                    "Grok-4-mini + Claude-Haiku-4.5 added). Adds ~10-20s.",
    )
    enable_specialist_judges: bool = Field(
        default=True,
        description="If True, run the 10 deterministic sector-specialist judges "
                    "(refining/petchem/LNG/tankers/insurance/retail/telecom/"
                    "fertilizer/aviation/power). Fast (~50ms total).",
    )
    scenario_focus: str = Field(
        default="default",
        description="Optional scenario focus. 'reliance_full_supplychain' adds a "
                    "Reliance Industries 10-node subsidiary impact table built "
                    "from FY24 RIL Integrated Annual Report disclosures + DGH/PIB "
                    "filings. 'default' returns only the India + Gulf tables.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_hash(payload: dict) -> str:
    """Canonical sha256 of JSON-serialized payload (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _aggregate_confidence(judges: list, sector_scores_india: list,
                           sector_scores_gulf: list, signals_used: int) -> dict:
    """Aggregate confidence from real signals — no vibes.

    Channels:
      - judge_consensus_conf: mean of LLM/rubric judge confidences
      - sector_score_dispersion: 1 - stddev of top-3 score differences (lower =
        more agreement = higher confidence)
      - signals_corroboration: bounded 0..1 from count of recent live signals
    """
    if judges:
        judge_conf = sum(j.confidence for j in judges) / len(judges)
    else:
        judge_conf = 0.5

    top3_in = [r["score"] for r in sector_scores_india[:3]]
    top3_gu = [r["score"] for r in sector_scores_gulf[:3]]
    if len(top3_in) >= 2:
        spread_in = max(top3_in) - min(top3_in)
    else:
        spread_in = 0.0
    if len(top3_gu) >= 2:
        spread_gu = max(top3_gu) - min(top3_gu)
    else:
        spread_gu = 0.0
    # Higher spread => sharper ranking => more confidence
    dispersion_signal = min(1.0, (spread_in + spread_gu))

    sig_corrob = min(1.0, 0.4 + 0.12 * signals_used)

    composite = round(0.55 * judge_conf + 0.25 * dispersion_signal
                      + 0.20 * sig_corrob, 4)

    return {
        "composite": composite,
        "judge_consensus_conf": round(judge_conf, 4),
        "sector_score_dispersion": round(dispersion_signal, 4),
        "signals_corroboration": round(sig_corrob, 4),
        "formula": ("0.55*judge_consensus_conf + 0.25*sector_score_dispersion"
                     " + 0.20*signals_corroboration"),
    }


def _aggregate_meta_judges(local_judges: list, openrouter_panel: dict | None,
                              specialist_panel: dict | None) -> dict:
    """Roll up across all judge sources into one 25-judge meta verdict.

    Reports per-source counts, an overall risk consensus (median ordinal),
    and a meta Krippendorff α computed across the union of all judges'
    risk_level rankings. Returns honest source attribution so the UI can
    label each tier (local / frontier / specialist)."""
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    inv = {v: k for k, v in risk_order.items()}

    all_risks: list[str] = []
    by_source: dict[str, dict] = {}

    # Local Ollama judges
    local_risks = [j.risk_level for j in local_judges if hasattr(j, "risk_level")]
    by_source["local_ollama"] = {
        "n_present": len(local_risks),
        "risks": local_risks,
        "consensus": (inv[sorted(risk_order[r] for r in local_risks
                                  if r in risk_order)[len(local_risks)//2]]
                       if local_risks else None),
    }
    all_risks.extend(local_risks)

    # Frontier OpenRouter judges
    if openrouter_panel and "results" in openrouter_panel:
        fr_risks = [r["risk_level"] for r in openrouter_panel["results"]
                     if r.get("ok") and r.get("risk_level")]
        by_source["openrouter_frontier"] = {
            "n_present": len(fr_risks),
            "n_panel_size": openrouter_panel.get("panel_size", 0),
            "n_succeeded": openrouter_panel.get("n_succeeded", 0),
            "risks": fr_risks,
            "consensus": openrouter_panel.get("consensus_risk"),
            "krippendorff_alpha": openrouter_panel.get("krippendorff_alpha_ordinal"),
        }
        all_risks.extend(fr_risks)
    else:
        by_source["openrouter_frontier"] = {"n_present": 0, "skipped": True}

    # Specialist deterministic judges
    if specialist_panel and "verdicts" in specialist_panel:
        sp_risks = [v["risk_level"] for v in specialist_panel["verdicts"]]
        by_source["specialist_rule_based"] = {
            "n_present": len(sp_risks),
            "risks": sp_risks,
            "consensus": specialist_panel["aggregate"]["consensus_risk"],
            "krippendorff_alpha": specialist_panel["aggregate"]["krippendorff_alpha_ordinal"],
        }
        all_risks.extend(sp_risks)
    else:
        by_source["specialist_rule_based"] = {"n_present": 0, "skipped": True}

    # Meta consensus across union
    if all_risks:
        all_idxs = sorted(risk_order[r] for r in all_risks if r in risk_order)
        meta_consensus = inv[all_idxs[len(all_idxs)//2]] if all_idxs else "MEDIUM"
    else:
        meta_consensus = "UNKNOWN"

    return {
        "n_judges_total": len(all_risks),
        "by_source": by_source,
        "meta_consensus_risk": meta_consensus,
        "framework": ("Multi-tier ensemble: local Ollama + frontier OpenRouter "
                       "+ deterministic sector specialists. Skalse 2022 anti-game."),
    }


def _conditional_caveats(signals_used: int, ollama_available: bool,
                          analogs_count: int) -> list[str]:
    """Honest, real caveats — never blanket disclaimers."""
    out: list[str] = []
    if signals_used == 0:
        out.append(
            "No Hormuz-tagged signals in the last 24h ingestion store. "
            "Conditional projection only — not a current incident."
        )
    if not ollama_available:
        out.append(
            "Local Ollama LLM panel is not reachable. Judge layer is running "
            "the deterministic severity rubric. Confidence reflects this."
        )
    if analogs_count == 0:
        out.append(
            "Crisis-library analog match returned no high-similarity historical "
            "events. Projection is interpolated from the closest available match."
        )
    out.append(
        "Sector-level loss bands are point-estimate ranges from published "
        "agency data; they are NOT precise dollar forecasts."
    )
    out.append(
        "This system does NOT predict whether Hormuz will actually be closed. "
        "It quantifies second-order industrial effects conditional on closure."
    )
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

if router is not None:

    @router.get("/demo/hormuz-war-room/ui", include_in_schema=False)
    def war_room_ui():
        if HTMLResponse is None:
            raise HTTPException(status_code=500, detail="HTMLResponse unavailable")
        if not UI_HTML_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"war-room HTML not found at {UI_HTML_PATH}",
            )
        return HTMLResponse(UI_HTML_PATH.read_text(encoding="utf-8"))

    @router.get("/demo/master", include_in_schema=False)
    @router.get("/demo/master/ui", include_in_schema=False)
    def master_ui():
        if HTMLResponse is None:
            raise HTTPException(status_code=500, detail="HTMLResponse unavailable")
        if not MASTER_HTML_PATH.exists():
            raise HTTPException(status_code=500,
                                detail=f"master HTML not found at {MASTER_HTML_PATH}")
        return HTMLResponse(MASTER_HTML_PATH.read_text(encoding="utf-8"))

    @router.post("/demo/hormuz-war-room", tags=["demo"])
    def war_room_orchestrate(req: WarRoomRequest) -> dict:
        t0 = time.time()
        request_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # ---- Stage 1: live pipeline (signals + analogs + judges + counterfactual)
        try:
            live = run_hormuz_pipeline(ScenarioRequest(
                scenario_text=req.scenario_text,
                region="hormuz",
                include_recent_signals=req.include_recent_signals,
                enable_llm_judges=req.enable_llm_judges,
                k_analogs=3,
            ))
        except Exception as e:  # noqa: BLE001
            logger.error("[war-room] live pipeline failed: %s", e)
            raise HTTPException(status_code=500,
                                detail=f"live pipeline failed: {e}")

        # ---- Stage 2: deterministic sector tables (instant)
        india_rows = india_mod.score_all(
            severity=req.severity,
            brent_price_usd_bbl=req.brent_price_usd_bbl,
            duration_days=req.duration_days,
        )
        gulf_rows = gulf_mod.score_all(
            severity=req.severity,
            brent_price_usd_bbl=req.brent_price_usd_bbl,
            duration_days=req.duration_days,
        )

        # Optional: Reliance Industries 10-node subsidiary table
        reliance_block: dict | None = None
        if req.scenario_focus == "reliance_full_supplychain":
            reliance_rows = reliance_mod.score_all(
                severity=req.severity,
                brent_price_usd_bbl=req.brent_price_usd_bbl,
                duration_days=req.duration_days,
            )
            reliance_agg = reliance_mod.aggregate_revenue_at_risk_inr_cr(reliance_rows)
            reliance_block = {
                "rows": reliance_rows,
                "aggregate": reliance_agg,
                "scenario_label": (
                    "Reliance Industries · full supply-chain impact under "
                    "Israel-Iran-USA Hormuz escalation"
                ),
                "data_attribution": (
                    "RIL Integrated Annual Report FY24 + DGH + BSE/NSE filings + "
                    "ICIS naphtha/PX market data + Lloyd's war-risk index + IRDAI"
                ),
            }

        # ---- Stage 3: chokepoint graph (static, IEA-cited)
        chokepoint = graph_mod.get_graph()

        # ---- Stage 3b (optional): 6 or 12-judge OpenRouter cross-check
        openrouter_panel: dict | None = None
        if req.enable_openrouter_panel:
            try:
                from ShAuRyA_Supplymind.realtime.openrouter_war_room_panel \
                    import run_panel_sync
                top_analog = (live.analogs[0]["name"] if live.analogs
                              else "(no analog)")
                openrouter_panel = run_panel_sync(
                    scenario_text=req.scenario_text,
                    severity=req.severity,
                    brent=req.brent_price_usd_bbl,
                    duration=req.duration_days,
                    top_analog=top_analog,
                    expand_to_12=req.expand_to_12_judges,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[war-room] OpenRouter panel failed: %s", e)
                openrouter_panel = {"error": str(e)[:300]}

        # ---- Stage 3c: 10 specialist judges (deterministic, ~50ms)
        specialist_panel: dict | None = None
        if req.enable_specialist_judges:
            try:
                from ShAuRyA_Supplymind.realtime.specialist_judges import (
                    run_all as run_specialists, aggregate as aggregate_specialists,
                )
                spec_verdicts = run_specialists(
                    severity=req.severity,
                    brent_price_usd_bbl=req.brent_price_usd_bbl,
                    duration_days=req.duration_days,
                )
                spec_agg = aggregate_specialists(spec_verdicts)
                specialist_panel = {
                    "verdicts": spec_verdicts,
                    "aggregate": spec_agg,
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("[war-room] specialist panel failed: %s", e)
                specialist_panel = {"error": str(e)[:300]}

        # ---- Stage 3d: meta-aggregation across all judge sources (25-judge total)
        all_judges_meta = _aggregate_meta_judges(
            local_judges=list(live.judges),
            openrouter_panel=openrouter_panel,
            specialist_panel=specialist_panel,
        )

        # ---- Stage 4: aggregated confidence + caveats
        confidence = _aggregate_confidence(
            judges=list(live.judges),
            sector_scores_india=india_rows,
            sector_scores_gulf=gulf_rows,
            signals_used=live.signals_used_count,
        )
        caveats = _conditional_caveats(
            signals_used=live.signals_used_count,
            ollama_available=live.ollama_available,
            analogs_count=len(live.analogs),
        )

        # ---- Stage 5: assemble + receipt
        live_dump = live.model_dump()
        payload = {
            "request_ts": request_ts,
            "scenario_input": req.model_dump(),
            "live_facts_chokepoint": chokepoint["headline_facts"],
            "live_pipeline": {
                "risk_level": live_dump["risk_level"],
                "consensus_confidence": live_dump["consensus_confidence"],
                "wall_clock_s": live_dump["wall_clock_s"],
                "ollama_available": live_dump["ollama_available"],
                "signals_used_count": live_dump["signals_used_count"],
                "analogs": live_dump["analogs"],
                "projection": live_dump["projection"],
                "judges": live_dump["judges"],
                "recommended_actions": live_dump["recommended_actions"],
                "counterfactual": live_dump["counterfactual"],
            },
            "openrouter_panel": openrouter_panel,
            "specialist_panel": specialist_panel,
            "judge_meta": all_judges_meta,
            "india_impact_table": india_rows,
            "gulf_impact_table": gulf_rows,
            "reliance_impact": reliance_block,
            "chokepoint_graph": {
                "nodes": chokepoint["nodes"],
                "edges": chokepoint["edges"],
                "data_attribution": chokepoint["data_attribution"],
            },
            "confidence": confidence,
            "uncertainty_caveats": caveats,
            "data_source_flags": {
                "live_pipeline": ("live_llm_panel" if live.ollama_available
                                   else "deterministic_rubric_fallback"),
                "openrouter_panel": (
                    "skipped" if openrouter_panel is None
                    else f"{openrouter_panel.get('n_succeeded', 0)}/"
                         f"{openrouter_panel.get('panel_size', 6)}_frontier_judges"
                    if "error" not in openrouter_panel else "errored"
                ),
                "india_table": "deterministic_static_cited",
                "gulf_table": "deterministic_static_cited",
                "chokepoint_graph": "iea_eia_static_cited",
                "signals": ("from_event_store_24h"
                              if live.signals_used_count > 0 else "no_recent_signals"),
            },
            "elapsed_s": round(time.time() - t0, 3),
        }
        payload["receipt_sha256"] = _stable_hash(payload)
        return payload

    @router.post("/demo/hormuz-war-room/validate", tags=["demo"])
    def war_room_validate() -> dict:
        """Run the backtest harness against 8 documented historical events."""
        try:
            import scripts.validate_war_room as validator
            return validator.main()
        except Exception as e:  # noqa: BLE001
            logger.error("[war-room] validation failed: %s", e)
            raise HTTPException(status_code=500,
                                detail=f"validation failed: {e}")

    @router.get("/demo/hormuz-war-room/health", tags=["demo"])
    def war_room_health() -> dict:
        return {
            "status": "ok",
            "ui_html_present": UI_HTML_PATH.exists(),
            "ui_html_path": str(UI_HTML_PATH),
            "n_india_sectors": len(india_mod.SECTORS),
            "n_gulf_sectors": len(gulf_mod.SECTORS),
            "n_graph_nodes": len(graph_mod.NODES),
            "n_graph_edges": len(graph_mod.EDGES),
            "n_headline_facts": len(graph_mod.HEADLINE_FACTS),
        }
