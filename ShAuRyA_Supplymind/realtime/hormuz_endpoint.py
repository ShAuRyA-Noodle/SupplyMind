"""
hormuz_endpoint.py — FastAPI router for /live/hormuz-closure and sibling endpoints.

Mount into main server/app.py via:
    from ShAuRyA_Supplymind.realtime.hormuz_endpoint import router as hormuz_router
    app.include_router(hormuz_router, prefix="/live", tags=["live"])

Endpoints:
    GET  /live/health                  — subsystem availability check
    GET  /live/recent-events           — recent events from ingestion store
    GET  /live/signal-counts           — per-source counts
    POST /live/hormuz-closure          — main live assessment endpoint
    POST /live/analog-match            — match free-text against crisis library

The /live/hormuz-closure pipeline:
    1. Gather recent high-severity events (last 24h) from store.
    2. Match the incoming scenario text against crisis library analogs.
    3. Interpolate a quantitative projection (Brent $, duration, rerouting).
    4. Call 3-judge LLM panel (Ollama if up, else deterministic heuristic).
    5. Build recommended actions from the OpenEnv env action schema.
    6. Return structured JSON ready for a live demo.

All subsystems degrade gracefully — if Ollama is down, we use a rubric-based
judge; if Chronos isn't loaded, we use the analog-interpolated Brent projection.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException
except ImportError:  # allow module-level import even without fastapi (for unit tests)
    APIRouter = None  # type: ignore
    HTTPException = Exception  # type: ignore

from .crisis_library import find_analogs, interpolate_projection
from . import store

logger = logging.getLogger(__name__)

if APIRouter is not None:
    router = APIRouter()
else:
    router = None

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScenarioRequest(BaseModel):
    scenario_text: str = Field(..., description="Free-text description of the live event.")
    region: str = Field("hormuz", description="One of: hormuz, iran_israel, red_sea, taiwan_strait, global")
    include_recent_signals: bool = Field(True, description="Join with last-24h live signals.")
    enable_llm_judges: bool = Field(True, description="Call Ollama 3-judge panel if available.")
    k_analogs: int = Field(3, ge=1, le=5)


class JudgeResult(BaseModel):
    name: str
    risk_level: str       # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float
    rationale: str
    latency_s: float


class ActionRec(BaseModel):
    action_type: str
    target: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    reason: str
    estimated_cost_usd: Optional[float] = None
    estimated_loss_avoided_usd: Optional[float] = None


class ScenarioResponse(BaseModel):
    request_ts: str
    region: str
    risk_level: str
    consensus_confidence: float
    analogs: list[dict]
    projection: dict
    judges: list[JudgeResult]
    recommended_actions: list[ActionRec]
    counterfactual: dict
    signals_used_count: int
    wall_clock_s: float
    ollama_available: bool


# ---------------------------------------------------------------------------
# Judge plumbing
# ---------------------------------------------------------------------------


JUDGE_PROMPT_TEMPLATE = """You are a supply-chain risk assessor. Analyze the
scenario below and return a JSON object with risk_level (LOW/MEDIUM/HIGH/CRITICAL),
confidence (0.0-1.0 float), and rationale (1-2 sentence reason).

Scenario:
{scenario_text}

Context — top historical analog: {top_analog}
Recent signals (last 24h, up to 5): {signals_brief}

Return ONLY JSON: {{"risk_level": "...", "confidence": 0.XX, "rationale": "..."}}."""


def _check_ollama() -> bool:
    import requests
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama_judge(model: str, prompt: str) -> dict:
    import json as _json
    import requests
    start = time.time()
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 16384},
        },
        timeout=120,
    )
    r.raise_for_status()
    content = r.json()["message"]["content"]
    parsed = _json.loads(content)
    return {
        "risk_level": parsed.get("risk_level", "MEDIUM").upper(),
        "confidence": float(parsed.get("confidence", 0.5)),
        "rationale": parsed.get("rationale", "(no rationale)")[:500],
        "latency_s": round(time.time() - start, 2),
    }


def _rubric_judge(scenario_text: str, projection: dict, signals: list[dict]) -> dict:
    """Deterministic fallback when Ollama is unavailable.

    Simple rule: severity_p50 >= 0.80 -> CRITICAL; >= 0.60 -> HIGH;
    >= 0.35 -> MEDIUM; else LOW. Confidence proportional to signal count.
    """
    start = time.time()
    sev = (projection or {}).get("severity_p50") or 0.3
    if sev >= 0.80:
        level = "CRITICAL"
    elif sev >= 0.60:
        level = "HIGH"
    elif sev >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"
    conf = min(0.95, 0.5 + 0.05 * len(signals or []))
    return {
        "risk_level": level,
        "confidence": round(conf, 2),
        "rationale": (f"Rubric assessment (Ollama unavailable). "
                      f"Analog severity P50={sev:.2f} mapped to {level}. "
                      f"{len(signals or [])} recent signals corroborate."),
        "latency_s": round(time.time() - start, 2),
    }


def _consensus_risk(judges: list[dict]) -> tuple[str, float]:
    """Return (majority level, mean confidence)."""
    if not judges:
        return "MEDIUM", 0.5
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    idx = [order.index(j["risk_level"]) if j["risk_level"] in order else 1 for j in judges]
    median_idx = sorted(idx)[len(idx) // 2]
    mean_conf = sum(j["confidence"] for j in judges) / len(judges)
    return order[median_idx], round(mean_conf, 3)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _recommend_actions(region: str, risk_level: str, projection: dict) -> list[dict]:
    """Map risk level + projection to a ranked list of OpenEnv actions."""
    brent = projection.get("brent_projection_usd_bbl_p50") or 80.0
    rerouting = projection.get("vessel_rerouting_days_p50") or 3.0
    sev = projection.get("severity_p50") or 0.3

    actions: list[dict] = []

    if risk_level in ("HIGH", "CRITICAL"):
        # Hedge commodity (oil)
        actions.append({
            "action_type": "hedge_commodity",
            "target": None,
            "parameters": {"commodity": "oil", "hedge_amount_usd": round(4_200_000 * sev, 0)},
            "reason": f"Brent projection ${brent:.2f}/bbl under analog scenario; hedge sized to severity {sev:.2f}.",
            "estimated_cost_usd": round(4_200_000 * sev * 0.06, 0),  # 6% hedge premium
            "estimated_loss_avoided_usd": round(brent * 1_000_000 * sev * 0.3, 0),
        })
        # Reroute shipment away from affected chokepoint
        if region in ("hormuz", "red_sea"):
            actions.append({
                "action_type": "reroute_shipment",
                "target": "IN_TRANSIT_TANKERS",
                "parameters": {"via": ["cape_of_good_hope"], "delay_days": int(rerouting)},
                "reason": (f"{region} chokepoint at {risk_level}; reroute via Cape adds "
                           f"{rerouting:.0f} days but eliminates route-closure exposure."),
                "estimated_cost_usd": round(rerouting * 180_000, 0),  # ~$180K/day carrier premium
                "estimated_loss_avoided_usd": round(9_600_000_000 * rerouting / 86, 0),  # Suez-equivalent
            })
        # Activate backup supplier for affected region
        actions.append({
            "action_type": "activate_backup_supplier",
            "target": "TSMC" if region == "taiwan_strait" else "SUP_AFFECTED",
            "parameters": {"backup_supplier_id": "SUP_SAMSUNG"},
            "reason": "Activate pre-qualified backup under elevated risk.",
            "estimated_cost_usd": 350_000,
            "estimated_loss_avoided_usd": round(sev * 12_000_000, 0),
        })

    if risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
        actions.append({
            "action_type": "increase_safety_stock",
            "target": "WAREHOUSE_PRIMARY",
            "parameters": {"additional_stock_days": min(30, int(rerouting) + 7)},
            "reason": (f"Rebuild {int(rerouting) + 7}-day buffer to absorb "
                       f"potential {rerouting:.0f}-day rerouting delay."),
            "estimated_cost_usd": 280_000,
            "estimated_loss_avoided_usd": round(sev * 4_000_000, 0),
        })

    actions.append({
        "action_type": "issue_supplier_alert",
        "target": "ALL_TIER1_SUPPLIERS",
        "parameters": {},
        "reason": "Zero-cost information action; request supplier status update + continuity plan.",
        "estimated_cost_usd": 0,
        "estimated_loss_avoided_usd": None,
    })
    return actions


def _counterfactual(risk_level: str, projection: dict, actions: list[dict]) -> dict:
    """Estimate P50 loss with vs without recommended actions."""
    sev = projection.get("severity_p50") or 0.3
    brent = projection.get("brent_projection_usd_bbl_p50") or 80.0
    duration = projection.get("duration_days_p50") or 14.0

    # "Do nothing" P50: scales with severity + duration + brent delta
    baseline_delta_bbl = max(0, brent - 80.0)
    no_action_p50 = round(sev * duration * (1_500_000 + baseline_delta_bbl * 40_000), 0)

    # "With plan" P50: sum of estimated_loss_avoided, capped at 80% of no-action
    saved = sum((a.get("estimated_loss_avoided_usd") or 0) for a in actions)
    with_plan_p50 = max(no_action_p50 * 0.2, no_action_p50 - saved)

    return {
        "no_action_p50_loss_usd": no_action_p50,
        "with_plan_p50_loss_usd": round(with_plan_p50, 0),
        "savings_usd": round(no_action_p50 - with_plan_p50, 0),
        "savings_pct": round((no_action_p50 - with_plan_p50) / no_action_p50 * 100, 1)
                       if no_action_p50 > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Recent signals
# ---------------------------------------------------------------------------


def _recent_signals(region: str, hours: int = 24, limit: int = 5) -> list[dict]:
    since = time.time() - hours * 3600
    return store.query_recent(since_unix=since, region=region, limit=limit)


def _signals_brief(signals: list[dict]) -> str:
    if not signals:
        return "(no recent signals)"
    out = []
    for s in signals[:5]:
        out.append(f"[{s['source']}] {s['ts_iso'][:19]} sev={s['severity']:.2f} {s['raw_text'][:120]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Core pipeline (reusable from endpoint + CLI)
# ---------------------------------------------------------------------------


def run_hormuz_pipeline(req: ScenarioRequest) -> ScenarioResponse:
    from datetime import datetime, timezone
    start = time.time()

    # 1. Recent signals
    signals = _recent_signals(req.region, hours=24, limit=5) if req.include_recent_signals else []

    # 2. Analog matching
    analogs = find_analogs(
        req.scenario_text + " " + _signals_brief(signals),
        k=req.k_analogs,
    )
    projection = interpolate_projection(analogs)

    # 3. LLM judges
    ollama_up = _check_ollama() if req.enable_llm_judges else False
    judges: list[dict] = []
    if ollama_up and req.enable_llm_judges:
        top = analogs[0].name if analogs else "(none)"
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            scenario_text=req.scenario_text,
            top_analog=top,
            signals_brief=_signals_brief(signals),
        )
        for model_name, friendly in [
            ("qwen2.5:14b-instruct-q4_K_M", "Qwen-2.5-14B"),
            ("mistral-nemo:latest", "Mistral-Nemo"),
            ("deepseek-r1-local-q4:latest", "DeepSeek-R1-Q4"),
        ]:
            try:
                j = _call_ollama_judge(model_name, prompt)
                j["name"] = friendly
                judges.append(j)
            except Exception as e:  # noqa: BLE001
                logger.warning("[hormuz] judge %s failed: %s", friendly, e)
    if not judges:
        j = _rubric_judge(req.scenario_text, projection, signals)
        j["name"] = "Rubric-Fallback"
        judges = [j]

    risk_level, consensus_conf = _consensus_risk(judges)

    # 4. Actions + counterfactual
    action_dicts = _recommend_actions(req.region, risk_level, projection)
    cf = _counterfactual(risk_level, projection, action_dicts)

    request_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    wall = round(time.time() - start, 2)

    return ScenarioResponse(
        request_ts=request_ts,
        region=req.region,
        risk_level=risk_level,
        consensus_confidence=consensus_conf,
        analogs=[{
            "event_id": a.event_id, "name": a.name, "date": a.date,
            "severity": a.severity, "similarity": a.similarity,
            "summary": a.summary,
        } for a in analogs],
        projection=projection,
        judges=[JudgeResult(**j) for j in judges],
        recommended_actions=[ActionRec(**a) for a in action_dicts],
        counterfactual=cf,
        signals_used_count=len(signals),
        wall_clock_s=wall,
        ollama_available=ollama_up,
    )


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------


if router is not None:
    @router.get("/health")
    def live_health() -> dict:
        return {
            "status": "ok",
            "ollama_available": _check_ollama(),
            "event_store_db": str(store.DB_PATH),
            "event_counts": store.count_by_source(),
        }

    @router.get("/recent-events")
    def live_recent_events(region: Optional[str] = None, hours: int = 24,
                           limit: int = 20) -> dict:
        since = time.time() - hours * 3600
        rows = store.query_recent(since_unix=since, region=region, limit=limit)
        return {"count": len(rows), "events": rows}

    @router.get("/signal-counts")
    def live_signal_counts(hours: int = 24) -> dict:
        return store.count_by_source(since_unix=time.time() - hours * 3600)

    @router.post("/hormuz-closure", response_model=ScenarioResponse)
    def live_hormuz_closure(req: ScenarioRequest) -> ScenarioResponse:
        try:
            return run_hormuz_pipeline(req)
        except Exception as e:  # noqa: BLE001
            logger.error("[hormuz] pipeline failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/analog-match")
    def live_analog_match(query: str, k: int = 3) -> dict:
        analogs = find_analogs(query, k=k)
        return {
            "analogs": [{
                "event_id": a.event_id, "name": a.name, "date": a.date,
                "severity": a.severity, "similarity": a.similarity,
                "summary": a.summary,
            } for a in analogs],
            "projection": interpolate_projection(analogs),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Free-text scenario description.")
    parser.add_argument("--region", default="hormuz")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama judge calls.")
    parser.add_argument("--no-signals", action="store_true", help="Skip recent signals.")
    args = parser.parse_args()

    resp = run_hormuz_pipeline(ScenarioRequest(
        scenario_text=args.scenario,
        region=args.region,
        enable_llm_judges=not args.no_llm,
        include_recent_signals=not args.no_signals,
    ))
    print(_json.dumps(resp.model_dump(), indent=2, default=str))
