"""nasa_firms.py — NASA FIRMS (Fire Information for Resource Management).

Real active fires worldwide near critical infrastructure. Detects refinery
fires, biomass burning, industrial fires that threaten supply chains.

Reads NASA_FIRMS_MAP_KEY from env.

Docs: https://firms.modaps.eosdis.nasa.gov/api/
"""
from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path

from ._common import cached_get, http_get_text, standard_event

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Critical supply-chain regions to scan for active fires (lon_min, lat_min, lon_max, lat_max)
CRITICAL_REGIONS = {
    "strait_of_hormuz":  (54.0, 24.0, 58.0, 28.0),
    "suez_canal":        (32.0, 28.5, 33.5, 31.5),
    "bab_el_mandeb":     (42.5, 11.0, 44.5, 14.0),
    "panama_canal":      (-82.0, 8.0, -79.0, 10.0),
    "singapore_strait":  (103.0, 1.0, 105.0, 2.0),
    "us_gulf_coast":     (-97.0, 28.0, -89.0, 31.0),  # Houston refineries
    "rotterdam_port":    (4.0, 51.5, 4.5, 52.0),
}


def _key() -> str | None:
    k = os.environ.get("NASA_FIRMS_MAP_KEY")
    if k: return k
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("NASA_FIRMS_MAP_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_active_fires(days_back: int = 1, source: str = "VIIRS_SNPP_NRT") -> list[dict]:
    """Pull active fires across all critical supply-chain regions.

    `source`: VIIRS_SNPP_NRT (375m), MODIS_NRT (1km), VIIRS_NOAA20_NRT.
    `days_back`: 1-10.
    """
    cache_key = f"fires_{source}_d{days_back}_list"

    def _fetch():
        api_key = _key()
        if not api_key:
            logger.warning("[nasa_firms] NASA_FIRMS_MAP_KEY not set; returning []")
            return []
        out: list[dict] = []
        for region_name, (lon_min, lat_min, lon_max, lat_max) in CRITICAL_REGIONS.items():
            url = (f"{BASE}/{api_key}/{source}/"
                   f"{lon_min},{lat_min},{lon_max},{lat_max}/{days_back}")
            try:
                text = http_get_text(url, timeout=25)
                fires = _parse_csv(text, region_name)
                out.extend(fires)
            except Exception as e:  # noqa: BLE001
                logger.warning("[nasa_firms] region %s failed: %s", region_name, str(e)[:80])
        logger.info("[nasa_firms] returned %d fire detections across %d regions",
                    len(out), len(CRITICAL_REGIONS))
        return out

    return cached_get("nasa_firms", cache_key, _fetch, ttl=3600)


def _parse_csv(text: str, region: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    out: list[dict] = []
    for row in reader:
        try:
            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
            frp = float(row.get("frp") or 0)  # Fire Radiative Power
            conf = (row.get("confidence") or "").lower()
        except ValueError:
            continue

        # Severity proxy: high FRP (>100 MW) + high confidence
        sev = min(1.0, frp / 200.0)
        if conf == "h": sev = max(sev, 0.55)

        eid = (f"firms_{region}_{row.get('acq_date','?')}_{row.get('acq_time','?')}_"
               f"{lat:.3f}_{lon:.3f}").replace(":", "")
        occurred = (f"{row.get('acq_date')}T"
                    f"{(row.get('acq_time') or '0000')[:2]}:"
                    f"{(row.get('acq_time') or '0000')[2:]}:00Z")

        out.append(standard_event(
            source="nasa_firms", event_id=eid,
            title=(f"Active fire near {region.replace('_',' ')} "
                   f"(FRP {frp:.0f} MW, conf {conf})"),
            description=(f"NASA FIRMS {row.get('instrument','?')} "
                         f"detection at ({lat:.3f},{lon:.3f}) on "
                         f"{row.get('acq_date')} {row.get('acq_time')}. "
                         f"Fire Radiative Power: {frp} MW. "
                         f"Region: {region}."),
            occurred_at_utc=occurred,
            lat=lat, lon=lon,
            raw_url=f"https://firms.modaps.eosdis.nasa.gov/map/#d:{row.get('acq_date')};l:viirs_noaa20",
            severity_proxy=sev,
            extra={
                "region": region, "frp_mw": frp, "confidence": conf,
                "instrument": row.get("instrument"),
                "satellite": row.get("satellite"),
            },
        ))
    return out


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_active_fires(days_back=1)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} fire detections in critical regions")
