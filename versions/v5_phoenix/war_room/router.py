"""router.py — POST /demo/hormuz-india-war-room.

Builds a 5-layer response (Scene 1..6 from the demo spec):
    Scene 1 — Live Shock     (live signals, severity, brent, evidence)
    Scene 2 — Chokepoint Map (geographic anchors + curated chokepoint facts)
    Scene 3 — India Ranking  (5 sectors, severity-modulated, with citations)
    Scene 4 — Gulf Ranking   (5 sectors, same)
    Scene 5 — Recommended Actions (tied to top-3 in each ranking)
    Scene 6 — Receipt        (hash, command, runtime, evidence summary)

Every leaf-level claim ships with an _evidence drawer so a judge can click
the source URL or read the model_estimate derivation. Live vs cached vs
model-derived facts are explicitly distinguished.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import atlas_loader, live_signals, provenance, ranker

logger = logging.getLogger(__name__)

router = APIRouter(tags=["war_room"])


class WarRoomRequest(BaseModel):
    scenario_text: str = Field(
        "Hormuz closure / Iran-Israel-US escalation",
        description="Free-text description of the live shock scenario.",
    )
    severity: float = Field(0.85, ge=0.0, le=1.0,
                            description="Operator-assessed severity in [0, 1]. 0.85 = HIGH escalation.")
    brent_price_override: float | None = Field(None,
        description="Optional Brent USD/bbl override; if None we pull from FRED.")
    country_focus: str = Field("India", description="Currently 'India' supported as primary; 'Gulf' always included.")
    include_gulf: bool = Field(True)
    horizon_days: int = Field(30, ge=1, le=180)
    enable_live_signals: bool = Field(True, description="Pull live NewsAPI + FRED. Set false for offline-replay-only path.")
    enable_llm_judges: bool = Field(False, description="Stretch — call the 3-judge panel (slow). Off by default.")


@router.get("/health")
def health():
    try:
        atl = atlas_loader.load()
        return {
            "ok": True,
            "atlases": {
                "chokepoint_facts": len(atl.chokepoint["facts"]),
                "india_sectors": len(atl.india["sectors"]),
                "gulf_sectors": len(atl.gulf["sectors"]),
            },
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@router.post("/hormuz-india-war-room")
def war_room(req: WarRoomRequest):
    started = time.time()

    try:
        atl = atlas_loader.load()
    except atlas_loader.AtlasValidationError as e:
        raise HTTPException(500, f"atlas validation failed: {e}")

    # Scene 1 — Live shock
    live = live_signals.aggregate(req.scenario_text, enable_live=req.enable_live_signals)
    brent_used = req.brent_price_override if req.brent_price_override is not None else live["brent_usd"]
    brent_evidence = (
        provenance.Evidence(
            source_type="model_estimate",
            derivation=f"caller-supplied brent_price_override={req.brent_price_override}"
        ).to_dict()
        if req.brent_price_override is not None else live["brent_evidence"]
    )

    severity_label = (
        "CRITICAL" if req.severity >= 0.85
        else "HIGH" if req.severity >= 0.70
        else "MEDIUM" if req.severity >= 0.40
        else "LOW"
    )
    scene_1_live_shock = {
        "scenario_text": req.scenario_text,
        "severity_input": req.severity,
        "severity_label": severity_label,
        "brent_usd": {
            "value": brent_used,
            "_evidence": brent_evidence,
        },
        "headlines": live["headlines"],
        "served_from_replay": live["served_from_replay"],
        "live_query_failures": live["live_query_failures"],
        "_evidence": provenance.Evidence(
            source_type="live_api" if not live["served_from_replay"] else "internal_artifact",
            publisher="NewsAPI + FRED" if not live["served_from_replay"] else "versions/v5_phoenix replay_cache",
            url=("https://newsapi.org/, https://fred.stlouisfed.org/" if not live["served_from_replay"]
                 else "versions/v5_phoenix/realtime_v5/replay_cache_latest.json"),
        ).to_dict(),
    }

    # Scene 2 — Chokepoint map
    scene_2_chokepoint = {
        "facts": [
            {
                "id": f["id"],
                "claim": f["claim"],
                "value": f.get("value") or f.get("value_range"),
                "unit": f.get("unit"),
                "_evidence": provenance.evidence_from_atlas_fact(f).to_dict(),
            }
            for f in atl.chokepoint["facts"]
        ],
        "geographic_anchors": atl.chokepoint["geographic_anchors"],
        "_evidence": provenance.Evidence(
            source_type="internal_artifact",
            artifact_path="versions/v5_phoenix/scenarios/hormuz_chokepoint_atlas.json",
            derivation="Curated atlas of IEA + EIA primary-source facts. See file for per-fact citations.",
        ).to_dict(),
    }

    # Scene 3 — India ranking
    india_ranks = ranker.rank(atl.india, req.severity, live["concatenated_text_for_keyword_match"])
    scene_3_india = {
        "country": "India",
        "n_sectors": len(india_ranks),
        "ranking": [s.to_dict() for s in india_ranks],
        "_evidence": provenance.Evidence(
            source_type="internal_artifact",
            artifact_path="versions/v5_phoenix/scenarios/india_supply_chain_exposure.json",
            derivation="Curated India atlas. Ranking is severity-modulated per ranker.py; per-sector exposure_facts are primary/secondary citations.",
        ).to_dict(),
    }

    # Scene 4 — Gulf ranking (optional)
    scene_4_gulf = None
    gulf_ranks: list[ranker.RankedSector] = []
    if req.include_gulf:
        gulf_ranks = ranker.rank(atl.gulf, req.severity, live["concatenated_text_for_keyword_match"])
        scene_4_gulf = {
            "country_focus": "Gulf (UAE, Qatar, Saudi Arabia, Bahrain, Kuwait, Oman)",
            "n_sectors": len(gulf_ranks),
            "ranking": [s.to_dict() for s in gulf_ranks],
            "_evidence": provenance.Evidence(
                source_type="internal_artifact",
                artifact_path="versions/v5_phoenix/scenarios/gulf_supply_chain_exposure.json",
                derivation="Same model as the India ranking; over the Gulf curated atlas.",
            ).to_dict(),
        }

    # Scene 5 — Recommended actions
    scene_5_actions = ranker.recommended_actions(india_ranks, gulf_ranks)

    # Scene 6 — Top-level header summary (the JSON the user spec'd)
    risk_level_aggregate = severity_label
    india_top_risks = [s.sector_id for s in india_ranks[:3]]
    gulf_top_risks = [s.sector_id for s in gulf_ranks[:3]] if gulf_ranks else []
    confidence_estimate = 0.55 + 0.35 * req.severity   # see HORMUZ_DEMO_LIMITATIONS — calibration footnote
    summary_block = {
        "risk_level": risk_level_aggregate,
        "india_top_risks": india_top_risks,
        "gulf_top_risks": gulf_top_risks,
        "horizon_days": req.horizon_days,
        "confidence_estimate": round(confidence_estimate, 2),
        "_evidence": provenance.Evidence(
            source_type="model_estimate",
            derivation=(
                f"risk_level mapped from severity_input={req.severity:.2f} via the same step thresholds (0.40/0.70/0.85). "
                f"top_risks are the 3 highest current_risk_score sectors after severity modulation. "
                f"confidence_estimate = 0.55 + 0.35*severity; see HORMUZ_DEMO_LIMITATIONS.md §3 for calibration caveats "
                f"(R4 ECE 0.19-0.34)."
            ),
        ).to_dict(),
    }

    payload_for_hash = {
        "scene_1_live_shock": scene_1_live_shock,
        "scene_2_chokepoint": scene_2_chokepoint,
        "scene_3_india": scene_3_india,
        "scene_4_gulf": scene_4_gulf,
        "scene_5_actions": scene_5_actions,
        "summary": summary_block,
    }
    receipt = provenance.build_receipt(
        payload=payload_for_hash,
        command=f"POST /demo/hormuz-india-war-room scenario_text={req.scenario_text!r} severity={req.severity}",
        runtime_s=time.time() - started,
    )

    return {
        **payload_for_hash,
        "receipt": receipt,
        "framework_version": "phoenix_war_room_v1.0",
        "honest_limitations": [
            "Numeric values labelled source_type='model_estimate' are derived; only 'primary'/'secondary'/'live_api' carry external citations.",
            "Total endpoint runtime is typically 3-90s depending on whether live APIs are reachable; not 'within seconds' for the live path.",
            "Confidence estimate is heuristic; R4 ECE of 0.19-0.34 means panel confidences are not well-calibrated. See HORMUZ_DEMO_LIMITATIONS.md.",
            "Sector exposure tables are curated from public sources at last_curated date; they will go stale if shipping flows shift materially.",
            "We do NOT predict the political event itself; the endpoint conditions on operator-supplied severity.",
        ],
    }
