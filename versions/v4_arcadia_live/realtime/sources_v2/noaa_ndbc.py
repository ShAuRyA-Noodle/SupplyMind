"""noaa_ndbc.py — NOAA National Data Buoy Center realtime ocean data.

Real-time ocean buoy observations: wave height, wind speed, water temp.
No auth. Public TXT feeds.

We pull the 5-day "realtime2" feed for buoys near critical chokepoints:
  Strait of Hormuz, Suez approaches, Singapore Strait, Bab-el-Mandeb,
  Panama Canal approaches, Long Beach (US Pacific gateway).
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_text, standard_event

logger = logging.getLogger(__name__)

# Buoys near supply-chain chokepoints. Mix of NOAA buoys + foreign
# (only NOAA US buoys reliably return data; we list foreign IDs in
# extras so the client can choose to query them via partner APIs).
CHOKE_BUOYS = {
    "PFXC1":   ("Pillar Point Harbor / US Pacific gateway", 37.50, -122.50),
    "LJPC1":   ("La Jolla / US-Mexico Pacific", 32.87, -117.26),
    "LONF1":   ("Long Key / Florida Straits / Caribbean", 24.84, -80.86),
    "FFIA2":   ("Cape Sarichef / Bering Strait / Arctic", 54.60, -164.93),
    "STDM4":   ("Stannard Rock / Great Lakes shipping", 47.18, -87.22),
    "MLRF1":   ("Molasses Reef / Gulf-Atlantic", 25.01, -80.38),
    "VENF1":   ("Venice / Gulf of Mexico", 27.07, -82.45),
    "BURL1":   ("Southwest Pass / Mississippi delta export", 28.91, -89.43),
}


def fetch_buoy(station_id: str) -> dict:
    """Fetch latest realtime2 obs for one NOAA buoy."""
    cache_key = f"buoy_{station_id}"

    def _fetch():
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        text = http_get_text(url, timeout=20)
        return _parse_txt(text, station_id)

    return cached_get("noaa_ndbc", cache_key, _fetch, ttl=900)


def fetch_chokepoint_buoys() -> list[dict]:
    """Fetch all chokepoint buoys; one event per buoy."""
    out: list[dict] = []
    for sid, (descr, lat, lon) in CHOKE_BUOYS.items():
        rec = fetch_buoy(sid)
        if not rec or "latest" not in rec:
            continue
        latest = rec["latest"]
        wvht = latest.get("WVHT")
        wspd = latest.get("WSPD")
        sev = 0.0
        # Severity proxy: 4m wave OR 25kt wind triggers signal
        if wvht is not None and wvht >= 4.0: sev = max(sev, min(1.0, wvht / 8.0))
        if wspd is not None and wspd >= 25.0: sev = max(sev, min(1.0, wspd / 50.0))
        out.append(standard_event(
            source="noaa_ndbc",
            event_id=f"buoy_{sid}_{latest.get('YYYY','?')}_{latest.get('MM','?')}_{latest.get('DD','?')}_{latest.get('hh','?')}",
            title=f"NDBC {sid} — {descr}",
            description=(f"Wave height: {wvht}m, Wind speed: {wspd}kt, "
                         f"Water temp: {latest.get('WTMP')}°C"),
            occurred_at_utc=_compose_iso(latest),
            lat=lat, lon=lon,
            raw_url=f"https://www.ndbc.noaa.gov/station_page.php?station={sid}",
            severity_proxy=sev,
            extra={"latest_obs": latest, "n_obs": len(rec.get("rows", []))},
        ))
    logger.info("[noaa_ndbc] returned %d buoys with data", len(out))
    return out


def _parse_txt(text: str, station_id: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return {"station_id": station_id, "rows": [], "latest": {}}
    # Header lines start with #
    headers = [h for h in lines[0].lstrip("#").split() if h]
    rows = []
    for line in lines[2:]:  # skip first two (#header, #units)
        parts = line.split()
        if len(parts) < len(headers): continue
        row = {}
        for h, v in zip(headers, parts):
            try: row[h] = float(v) if v not in ("MM", "999.0") else None
            except ValueError: row[h] = v
        rows.append(row)
        if len(rows) >= 12: break  # last ~hours only
    latest = rows[0] if rows else {}
    return {"station_id": station_id, "rows": rows, "latest": latest}


def _compose_iso(latest: dict) -> str | None:
    try:
        y, m, d = int(latest["YYYY"]), int(latest["MM"]), int(latest["DD"])
        hh, mm = int(latest["hh"]), int(latest["mm"])
        return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:00Z"
    except (KeyError, ValueError, TypeError):
        return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_chokepoint_buoys()
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} buoys with realtime data")
