"""eia.py — US Energy Information Admin (EIA) realtime petroleum data.

Real US oil/gas signals: WTI/Brent spot prices, refinery utilization,
crude oil inventories. Free key.

Reads EIA_API_KEY from env.

Docs: https://www.eia.gov/opendata/documentation.php
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = "https://api.eia.gov/v2"


def _key() -> str | None:
    k = os.environ.get("EIA_API_KEY")
    if k: return k
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("EIA_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_petroleum_signals(limit: int = 5) -> list[dict]:
    """Pull most-recent WTI + Brent spot + refinery utilization signals.

    Returns one event per series, with the latest observation.
    """
    cache_key = f"petro_l{limit}_list"

    def _fetch():
        api_key = _key()
        if not api_key:
            logger.warning("[eia] EIA_API_KEY not set; returning []")
            return []

        # 4 EIA series we care about: WTI, Brent, US refinery utilization,
        # US crude oil stocks
        targets = [
            ("petroleum/pri/spt/data/", "WTI Spot Price (Cushing OK)",
             {"facets[product][]": "EPCWTI", "facets[duoarea][]": "Y35NY"}),
            ("petroleum/pri/spt/data/", "Brent Spot Price (Europe FOB)",
             {"facets[product][]": "EPCBRENT", "facets[duoarea][]": "RGB"}),
            ("petroleum/pnp/wiup/data/", "US Refinery Utilization Pct",
             {"facets[product][]": "EPP0", "facets[duoarea][]": "NUS"}),
            ("petroleum/sum/sndw/data/", "US Weekly Crude Oil Stocks",
             {"facets[product][]": "EPC0", "facets[duoarea][]": "NUS"}),
        ]
        out: list[dict] = []
        for ep, name, facets in targets:
            params = {
                "api_key": api_key,
                "frequency": "weekly" if "wiup" in ep or "sndw" in ep else "daily",
                "data[0]": "value",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": limit,
                **facets,
            }
            try:
                d = http_get_json(f"{BASE}/{ep}", params=params, timeout=20)
                rows = (d.get("response") or {}).get("data") or []
                if rows:
                    out.append(_normalize(name, ep, rows))
            except Exception as e:  # noqa: BLE001
                logger.warning("[eia] series %s failed: %s", name[:30], str(e)[:80])
        logger.info("[eia] returned %d petroleum signals", len(out))
        return out

    return cached_get("eia", cache_key, _fetch, ttl=3600)


def _normalize(series_name: str, endpoint: str, rows: list[dict]) -> dict:
    latest = rows[0]
    val = latest.get("value")
    period = latest.get("period")
    units = latest.get("units")

    # Severity proxy: deviation from baseline
    sev = 0.0
    try:
        v = float(val)
        if "Brent" in series_name and v > 100: sev = min(1.0, (v - 80) / 50)
        elif "WTI" in series_name and v > 90: sev = min(1.0, (v - 70) / 50)
        elif "Utilization" in series_name and v < 80: sev = min(1.0, (90 - v) / 30)
    except (ValueError, TypeError):
        pass

    return standard_event(
        source="eia", event_id=f"eia_{endpoint.split('/')[1]}_{period}",
        title=f"{series_name}: {val} {units} ({period})",
        description=(f"EIA series {endpoint}, latest period {period}: "
                     f"{val} {units}. Last {len(rows)} obs returned."),
        occurred_at_utc=f"{period}T00:00:00Z" if period else None,
        raw_url=f"https://www.eia.gov/opendata/browser/{endpoint.rstrip('/data/')}",
        severity_proxy=sev,
        extra={
            "value": val, "units": units, "period": period,
            "n_obs": len(rows),
            "endpoint": endpoint,
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_petroleum_signals()
    print(json.dumps(e, indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} EIA signals")
