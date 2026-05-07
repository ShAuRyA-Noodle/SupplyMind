"""
usgs.py — USGS live earthquake feed (M4.5+ or M2.5+ significant).

Free, no key, JSON feed refreshes every minute.
Docs: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from ..store import Event

logger = logging.getLogger(__name__)

USGS_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"
USGS_45_DAY = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
SOURCE_NAME = "usgs"

# Regions of interest — bounding boxes (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "hormuz": (50.0, 24.0, 58.0, 28.5),        # Persian Gulf
    "iran_israel": (33.0, 28.0, 60.0, 39.0),    # Iran + Israel + neighbors
    "red_sea": (32.0, 12.0, 44.0, 28.0),        # Red Sea + Bab-el-Mandeb
    "taiwan_strait": (118.0, 21.0, 123.0, 26.0),# Taiwan + Strait
    "japan_korea": (125.0, 30.0, 146.0, 45.0),  # NE Asia
    "gulf_mexico": (-100.0, 18.0, -80.0, 30.0), # Gulf of Mexico (US refineries)
}


def _region_of(lon: float, lat: float) -> str:
    for name, (mn_lo, mn_la, mx_lo, mx_la) in REGIONS.items():
        if mn_lo <= lon <= mx_lo and mn_la <= lat <= mx_la:
            return name
    return "other"


def _severity_from_mag(mag: Optional[float]) -> float:
    if mag is None:
        return 0.1
    # M4.5 -> 0.1, M6 -> 0.5, M7.5 -> 0.9, M8+ -> 1.0
    return float(max(0.0, min(1.0, (mag - 4.0) / 4.0)))


def fetch(url: str = USGS_45_DAY) -> list[Event]:
    """Pull current earthquake feed. Default: M4.5+ last 24 hours."""
    events: list[Event] = []
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.error("[usgs] fetch failed: %s", e)
        return events

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates") or [0, 0, 0]
        lon, lat = float(coords[0]), float(coords[1])
        mag = props.get("mag")
        place = props.get("place") or ""
        time_ms = props.get("time")
        if time_ms is None:
            ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            ts_iso = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc) \
                .isoformat().replace("+00:00", "Z")

        region = _region_of(lon, lat)
        ev = Event(
            source=SOURCE_NAME,
            ts_iso=ts_iso,
            event_type="earthquake",
            region=region,
            severity=_severity_from_mag(mag),
            raw_text=f"M{mag} earthquake — {place}",
            urls=[props.get("url") or ""],
            entities=[],
            meta={
                "mag": mag,
                "lon": lon, "lat": lat, "depth_km": coords[2] if len(coords) > 2 else None,
                "tsunami": props.get("tsunami", 0),
                "sig": props.get("sig"),
                "alert": props.get("alert"),
            },
        )
        events.append(ev)

    logger.info("[usgs] fetched %d earthquakes", len(events))
    return events


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", default=USGS_45_DAY)
    args = parser.parse_args()

    evs = fetch(args.feed)
    for e in evs[:10]:
        print(f"{e.ts_iso}  {e.region:15s}  mag={e.meta.get('mag')}  sev={e.severity:.2f}  {e.raw_text}")
    print(f"\ntotal: {len(evs)}")
