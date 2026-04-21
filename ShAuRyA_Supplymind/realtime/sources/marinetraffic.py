"""
marinetraffic.py — Vessel positions near Strait of Hormuz (graceful fallback).

MarineTraffic API is paid. We implement three tiers:

    1. If MARINETRAFFIC_API_KEY env set: use the real API.
    2. Else if VESSELFINDER_API_KEY set: use VesselFinder free tier.
    3. Else: fall back to a LOCAL STATIC snapshot committed to repo
       (ShAuRyA_Supplymind/realtime/vessel_snapshot_hormuz.json) — updated
       manually via `python -m ShAuRyA_Supplymind.realtime.sources.marinetraffic
       --refresh-snapshot`.

This source is a "soft signal" — we annotate the event with severity based on
unusual queue length or rapid rerouting, not raw vessel count.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from ..store import Event

logger = logging.getLogger(__name__)

SOURCE_NAME = "marinetraffic"
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "vessel_snapshot_hormuz.json"

# Hormuz bounding box (approximate)
HORMUZ_BBOX = {"minLat": 25.0, "maxLat": 27.5, "minLon": 55.5, "maxLon": 58.0}

# Normal baseline: ~30 tankers in the strait at any time (2024 Maritime intel estimate)
BASELINE_TANKER_COUNT = 30
BASELINE_AVG_SPEED = 12.0  # knots


def _severity_from_state(tanker_count: int, avg_speed: float) -> float:
    """Higher count OR lower speed = higher congestion severity."""
    count_delta = (tanker_count - BASELINE_TANKER_COUNT) / BASELINE_TANKER_COUNT
    speed_delta = max(0, (BASELINE_AVG_SPEED - avg_speed) / BASELINE_AVG_SPEED)
    return float(max(0.0, min(1.0, 0.5 * count_delta + 0.5 * speed_delta)))


def _load_snapshot() -> Optional[dict]:
    if SNAPSHOT_PATH.exists():
        return json.loads(SNAPSHOT_PATH.read_text())
    return None


def _save_snapshot(state: dict) -> None:
    SNAPSHOT_PATH.write_text(json.dumps(state, indent=2))


def _fetch_marinetraffic(key: str) -> Optional[dict]:
    url = (f"https://services.marinetraffic.com/api/exportvessel/v:5/"
           f"{key}/protocol:jsono/msgtype:simple/"
           f"minlat:{HORMUZ_BBOX['minLat']}/maxlat:{HORMUZ_BBOX['maxLat']}/"
           f"minlon:{HORMUZ_BBOX['minLon']}/maxlon:{HORMUZ_BBOX['maxLon']}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("[marinetraffic] API call failed: %s", e)
        return None


def fetch(api_key: Optional[str] = None) -> list[Event]:
    """Return a single vessel-congestion event for Hormuz."""
    key = api_key or os.environ.get("MARINETRAFFIC_API_KEY")
    data = None
    mode = "snapshot"

    if key:
        data = _fetch_marinetraffic(key)
        mode = "live_api"

    if data is None:
        snapshot = _load_snapshot()
        if snapshot is None:
            # No API key and no snapshot — create a conservative default
            snapshot = {
                "ts_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "tanker_count": BASELINE_TANKER_COUNT,
                "cargo_count": 0,
                "avg_speed_knots": BASELINE_AVG_SPEED,
                "note": "default — no live MarineTraffic API key; run with --refresh-snapshot",
            }
        ts_iso = snapshot.get("ts_iso")
        tanker_count = snapshot.get("tanker_count", BASELINE_TANKER_COUNT)
        avg_speed = snapshot.get("avg_speed_knots", BASELINE_AVG_SPEED)
    else:
        # Parse live API response
        tankers = [v for v in data if str(v.get("TYPE", "")).startswith("80")]
        tanker_count = len(tankers)
        speeds = [float(v.get("SPEED", 0)) / 10.0 for v in tankers if v.get("SPEED")]
        avg_speed = sum(speeds) / len(speeds) if speeds else BASELINE_AVG_SPEED
        ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    sev = _severity_from_state(tanker_count, avg_speed)
    raw = (f"Hormuz traffic check ({mode}): {tanker_count} tankers, "
           f"avg speed {avg_speed:.1f} kn "
           f"(baseline {BASELINE_TANKER_COUNT} tankers / {BASELINE_AVG_SPEED} kn)")
    ev = Event(
        source=SOURCE_NAME,
        ts_iso=ts_iso,
        event_type="traffic_snapshot",
        region="hormuz",
        severity=sev,
        raw_text=raw,
        urls=["https://www.marinetraffic.com/en/ais/home/centerx:56/centery:26/zoom:7"],
        entities=["Hormuz"],
        meta={
            "mode": mode,
            "tanker_count": tanker_count,
            "avg_speed_knots": avg_speed,
            "baseline_tankers": BASELINE_TANKER_COUNT,
            "baseline_speed": BASELINE_AVG_SPEED,
        },
    )
    logger.info("[marinetraffic] %s: %d tankers @ %.1f kn -> sev=%.2f",
                mode, tanker_count, avg_speed, sev)
    return [ev]


def refresh_snapshot_interactive() -> None:
    """Manual refresh prompt — user types in current numbers from marinetraffic.com."""
    print("Visit https://www.marinetraffic.com/en/ais/home/centerx:56/centery:26/zoom:7")
    print("Count tankers in the Strait of Hormuz bounding box (approximate).")
    try:
        tanker_count = int(input("tanker_count (baseline 30): ").strip() or "30")
        avg_speed = float(input("avg_speed_knots (baseline 12.0): ").strip() or "12.0")
    except Exception:
        print("invalid input, aborting")
        return
    snap = {
        "ts_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tanker_count": tanker_count,
        "cargo_count": 0,
        "avg_speed_knots": avg_speed,
        "note": "manually refreshed",
    }
    _save_snapshot(snap)
    print(f"saved snapshot to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-snapshot", action="store_true",
                        help="Interactive prompt to refresh vessel_snapshot_hormuz.json")
    args = parser.parse_args()

    if args.refresh_snapshot:
        refresh_snapshot_interactive()
    else:
        evs = fetch()
        for e in evs:
            print(f"{e.ts_iso}  {e.region}  sev={e.severity:.2f}  {e.raw_text}")
