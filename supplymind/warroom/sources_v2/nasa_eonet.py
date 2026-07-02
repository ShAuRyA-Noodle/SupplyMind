"""nasa_eonet.py — NASA Earth Observatory Natural Event Tracker.

Real active natural events (wildfires, storms, volcanoes, sea/lake ice).
No auth. Public API.

Docs: https://eonet.gsfc.nasa.gov/docs/v3
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://eonet.gsfc.nasa.gov/api/v3/events"


def fetch_open_events(days: int = 30, limit: int = 50) -> list[dict]:
    cache_key = f"open_d{days}_l{limit}_list"

    def _fetch():
        params = {"status": "open", "days": days, "limit": limit}
        data = http_get_json(BASE, params=params, timeout=25)
        events = data.get("events") or []
        logger.info("[nasa_eonet] returned %d open events", len(events))
        return [_normalize(e) for e in events]

    return cached_get("nasa_eonet", cache_key, _fetch, ttl=1800)


def _normalize(ev: dict) -> dict:
    eid = ev.get("id") or "?"
    title = ev.get("title") or ""
    cats = ", ".join(c.get("title", "") for c in ev.get("categories", []))
    geoms = ev.get("geometry") or []
    last = geoms[-1] if geoms else {}
    coords = last.get("coordinates") or [None, None]

    # Severity proxy from category type
    cat_low = cats.lower()
    sev = 0.4
    if "volcano" in cat_low: sev = 0.75
    elif "severe storm" in cat_low: sev = 0.7
    elif "wildfires" in cat_low: sev = 0.55
    elif "earthquakes" in cat_low: sev = 0.65

    sources = ", ".join((s.get("id") or "") for s in ev.get("sources", []))

    return standard_event(
        source="nasa_eonet", event_id=eid,
        title=f"{title} ({cats})"[:160],
        description=(f"NASA EONET event. Categories: {cats}. "
                     f"Sources: {sources}. "
                     f"Geometry samples: {len(geoms)}.")[:1500],
        occurred_at_utc=last.get("date"),
        lat=coords[1] if len(coords) >= 2 and isinstance(coords[1], (int, float)) else None,
        lon=coords[0] if coords and isinstance(coords[0], (int, float)) else None,
        raw_url=ev.get("link") or f"https://eonet.gsfc.nasa.gov/api/v3/events/{eid}",
        severity_proxy=sev,
        extra={
            "categories": cats,
            "n_geometry_samples": len(geoms),
            "sources": sources,
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_open_events(days=30, limit=20)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} active natural events")
