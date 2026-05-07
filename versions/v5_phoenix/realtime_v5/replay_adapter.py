"""replay_adapter.py — FastAPI router that serves /live/hormuz-closure from the
frozen replay cache instead of hitting live APIs.

Design: does NOT edit or monkey-patch v4's versions/v4_arcadia_live/realtime/
hormuz_endpoint.py. Instead we mount this adapter at a sibling path
(`/replay/hormuz-closure`) AND provide a switch that the Phoenix app can
flip at startup to have the main v4 `/live/hormuz-closure` fall through
to replay when the env var FORCE_REPLAY=1 is set.

For the demo: set FORCE_REPLAY=1 in the terminal before launching
phoenix_app.py, and the v5 router intercepts first. Unset to restore live.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
LATEST_CACHE = HERE / "replay_cache_latest.json"


class ReplayRequest(BaseModel):
    scenario_text: str
    region: str = "hormuz"
    event_id: str | None = Field(None, description="Optional — match to a cached event by ID")
    k_analogs: int = 3


def _load_cache() -> dict:
    if not LATEST_CACHE.exists():
        logger.warning("[replay] no cache at %s -- did you run freeze_cache.py?", LATEST_CACHE)
        return {"events": {}}
    return json.loads(LATEST_CACHE.read_text())


def _best_analog(scenario_text: str, events: dict) -> tuple[str, dict, float]:
    """Simple lexical match: token overlap as similarity (good enough for 8 events)."""
    if not events:
        return "", {}, 0.0
    q_tokens = set(scenario_text.lower().split())
    best_id, best_event, best_sim = "", {}, 0.0
    for eid, ev in events.items():
        ev_text = f"{ev.get('top_analog', {}).get('name', '')} {ev.get('scenario_input', {}).get('scenario_text', '')}"
        ev_tokens = set(ev_text.lower().split())
        if not ev_tokens:
            continue
        sim = len(q_tokens & ev_tokens) / max(1, len(q_tokens | ev_tokens))
        if sim > best_sim:
            best_id, best_event, best_sim = eid, ev, sim
    return best_id, best_event, best_sim


router = APIRouter(tags=["replay"])


@router.post("/hormuz-closure")
def replay_hormuz(req: ReplayRequest):
    cache = _load_cache()
    events = cache.get("events", {})
    if req.event_id and req.event_id in events:
        ev = events[req.event_id]
        ev["_served_from_replay"] = True
        ev["_served_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return ev
    best_id, best_ev, sim = _best_analog(req.scenario_text, events)
    if not best_ev:
        return {"error": "no cached events; run versions.v5_phoenix.realtime_v5.freeze_cache"}
    out = dict(best_ev)
    # overwrite similarity with the lexical score we just computed (not the cached 0.99)
    out.setdefault("top_analog", {})["similarity"] = round(sim, 3)
    out["_served_from_replay"] = True
    out["_served_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["_replay_matched_id"] = best_id
    return out


@router.get("/status")
def status():
    cache = _load_cache()
    return {
        "cache_path": str(LATEST_CACHE),
        "cache_exists": LATEST_CACHE.exists(),
        "build_mode": cache.get("build_mode"),
        "built_at": cache.get("built_at"),
        "n_events": len(cache.get("events", {})),
        "force_replay_env": os.environ.get("FORCE_REPLAY") == "1",
    }


def should_force_replay() -> bool:
    return os.environ.get("FORCE_REPLAY") == "1"
