"""noaa_tides.py — NOAA Tides & Currents API.

Real-time water level + tide data at major US ports.
No auth. Public.

Docs: https://api.tidesandcurrents.noaa.gov/api/prod/

We pull current water level + last hour of obs at supply-chain-critical
US ports (Long Beach, Houston, NY/NJ, etc.).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Supply-chain critical US port stations (NOAA station IDs)
PORT_STATIONS = {
    "9410660": ("Los Angeles, CA",     33.72, -118.27),
    "9410230": ("La Jolla, CA",         32.87, -117.26),
    "9414290": ("San Francisco, CA",    37.81, -122.47),
    "9447130": ("Seattle, WA",          47.60, -122.34),
    "8454000": ("Providence, RI",       41.81, -71.40),
    "8443970": ("Boston, MA",           42.35, -71.05),
    "8518750": ("The Battery, NY",      40.70, -74.01),
    "8638610": ("Norfolk, VA",          36.95, -76.33),
    "8723214": ("Virginia Key, FL",     25.73, -80.16),
    "8770570": ("Sabine Pass, TX",      29.73, -93.87),
    "8771341": ("Galveston Bay, TX",    29.48, -94.74),
    "8771013": ("Eagle Point, TX",      29.48, -94.92),
}


def fetch_port_water_level(station_id: str) -> dict:
    """Last 6 hours of water-level observations at a port station."""
    cache_key = f"port_{station_id}"

    def _fetch():
        end = datetime.now(timezone.utc)
        begin = end - timedelta(hours=6)
        params = {
            "begin_date": begin.strftime("%Y%m%d %H:%M"),
            "end_date":   end.strftime("%Y%m%d %H:%M"),
            "station": station_id,
            "product": "water_level",
            "datum": "MLLW",
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "application": "supplymind",
        }
        data = http_get_json(BASE, params=params, timeout=20)
        return data

    return cached_get("noaa_tides", cache_key, _fetch, ttl=900)


def fetch_chokepoint_ports() -> list[dict]:
    """Fetch all chokepoint ports; one event per port."""
    out: list[dict] = []
    for sid, (name, lat, lon) in PORT_STATIONS.items():
        rec = fetch_port_water_level(sid)
        obs = (rec or {}).get("data") or []
        if not obs: continue
        latest = obs[-1]
        try:
            wl = float(latest.get("v") or 0)
            sigma = float(latest.get("s") or 0)
        except (ValueError, TypeError):
            continue

        # Severity proxy: extreme tide deviation (>2.5m) or high noise
        sev = 0.0
        if abs(wl) > 2.5: sev = max(sev, min(1.0, abs(wl) / 5.0))
        if sigma > 0.5: sev = max(sev, 0.4)

        out.append(standard_event(
            source="noaa_tides",
            event_id=f"port_{sid}_{latest.get('t', '?')}".replace(" ", "_"),
            title=f"{name} water level",
            description=(f"Water level: {wl:.2f}m MLLW, "
                         f"noise sigma: {sigma:.2f}m, "
                         f"observation time: {latest.get('t')}"),
            occurred_at_utc=_iso_from_t(latest.get("t")),
            lat=lat, lon=lon,
            raw_url=f"https://tidesandcurrents.noaa.gov/stationhome.html?id={sid}",
            severity_proxy=sev,
            extra={
                "water_level_m": wl,
                "sigma_m": sigma,
                "n_obs_window": len(obs),
                "station_id": sid,
            },
        ))
    logger.info("[noaa_tides] returned %d ports with data", len(out))
    return out


def _iso_from_t(t: str | None) -> str | None:
    if not t: return None
    # NOAA returns "2026-04-25 12:34" — convert to ISO
    try:
        return datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return t


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_chokepoint_ports()
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} ports with realtime data")
