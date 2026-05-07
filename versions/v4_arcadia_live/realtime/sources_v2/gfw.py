"""gfw.py — Global Fishing Watch (GFW) v3 API.

Real AIS-derived vessel events: port visits, encounters, fishing activity,
loitering. Free token. Far better than free MarineTraffic tier.

Reads GFW_API_TOKEN from env.

Docs: https://globalfishingwatch.org/our-apis/documentation
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE = "https://gateway.api.globalfishingwatch.org/v3"

# Bounding boxes for chokepoint regions (lon_min, lat_min, lon_max, lat_max)
CHOKEPOINTS = {
    "strait_of_hormuz":  (54.0, 24.0, 58.0, 28.0),
    "suez_canal":        (32.0, 28.5, 33.5, 31.5),
    "bab_el_mandeb":     (42.5, 11.0, 44.5, 14.0),
    "panama_canal":      (-82.0, 8.0, -79.0, 10.0),
    "singapore_strait":  (103.0, 1.0, 105.0, 2.0),
    "english_channel":   (1.0, 50.0, 3.0, 51.5),
}


def _key() -> str | None:
    k = os.environ.get("GFW_API_TOKEN")
    if k: return k
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GFW_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_recent_port_visits(days_back: int = 7, limit_per_region: int = 5) -> list[dict]:
    """Pull recent port visits (last N days) — high-volume traffic signal."""
    cache_key = f"port_visits_d{days_back}_n{limit_per_region}_list"

    def _fetch():
        token = _key()
        if not token:
            logger.warning("[gfw] GFW_API_TOKEN not set; returning []")
            return []

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days_back)
        params = {
            "datasets[0]": "public-global-port-visits-events:latest",
            "start-date": start.isoformat(),
            "end-date":   end.isoformat(),
            "limit": limit_per_region * len(CHOKEPOINTS),
            "offset": 0,
        }
        try:
            data = http_get_json(
                f"{BASE}/events", params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[gfw] /events fetch failed: %s", str(e)[:120])
            return []
        entries = data.get("entries") or []
        out = [_normalize(e, "port_visit") for e in entries[:limit_per_region * len(CHOKEPOINTS)]]
        logger.info("[gfw] returned %d port-visit events (of %d total available)",
                    len(out), data.get("total", "?"))
        return out

    return cached_get("gfw", cache_key, _fetch, ttl=3600)


def fetch_loitering_events(days_back: int = 14, limit: int = 30) -> list[dict]:
    """Pull recent vessel loitering events — anomaly signal."""
    cache_key = f"loitering_d{days_back}_n{limit}_list"

    def _fetch():
        token = _key()
        if not token: return []

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days_back)
        params = {
            "datasets[0]": "public-global-loitering-events:latest",
            "start-date": start.isoformat(),
            "end-date":   end.isoformat(),
            "limit": limit, "offset": 0,
        }
        try:
            data = http_get_json(
                f"{BASE}/events", params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[gfw] /loitering fetch failed: %s", str(e)[:120])
            return []
        entries = data.get("entries") or []
        out = [_normalize(e, "loitering") for e in entries[:limit]]
        logger.info("[gfw] returned %d loitering events", len(out))
        return out

    return cached_get("gfw", cache_key, _fetch, ttl=3600)


def _normalize(e: dict, ev_type: str) -> dict:
    eid = e.get("id") or "?"
    pos = e.get("position") or {}
    lat = pos.get("lat")
    lon = pos.get("lon")
    start = e.get("start")
    end = e.get("end")
    vessel = (e.get("vessel") or {})
    name = vessel.get("name") or "?"
    flag = vessel.get("flag") or "?"

    # Severity proxy by event type
    sev = {"port_visit": 0.2, "loitering": 0.55, "encounter": 0.65}.get(ev_type, 0.3)

    region_label = _region_for(lat, lon) or "open_water"

    return standard_event(
        source="gfw", event_id=eid,
        title=f"{ev_type.replace('_', ' ').title()} — {name} ({flag}) at {region_label}",
        description=(f"GFW AIS-derived {ev_type} event from {start} to {end}. "
                     f"Vessel: {name}, flag {flag}. Position ({lat}, {lon}). "
                     f"Region: {region_label}.")[:1500],
        occurred_at_utc=start,
        lat=lat, lon=lon,
        raw_url=f"https://globalfishingwatch.org/map/?event={eid}",
        severity_proxy=sev,
        extra={
            "vessel_name": name, "vessel_flag": flag,
            "event_type": ev_type, "duration_to": end,
            "region_label": region_label,
        },
    )


def _region_for(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None: return None
    for name, (lo_min, la_min, lo_max, la_max) in CHOKEPOINTS.items():
        if lo_min <= lon <= lo_max and la_min <= lat <= la_max:
            return name
    return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_recent_port_visits(days_back=7)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} GFW port-visit events")
