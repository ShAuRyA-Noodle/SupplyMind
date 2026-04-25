"""ucdp.py — Uppsala Conflict Data Program GED API.

Real conflict events with fatality counts. Public API, no auth.
Substitute for ACLED. Used by 1000+ peer-reviewed papers.

Docs: https://ucdp.uu.se/apidocs/

Returns events with deaths, lat/lon, conflict_type — direct map to
our supply-chain risk signal (Iran/Israel/Hormuz/Bab-el-Mandeb conflicts).
"""
from __future__ import annotations

import logging
from typing import Any

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://ucdpapi.pcr.uu.se/api/gedevents"


def fetch_recent(
    days: int = 30,
    pagesize: int = 100,
    region: str | None = None,
) -> list[dict]:
    """Pull GED conflict events from the last N days, optionally region-filtered.

    Region codes (UCDP):
      1=Africa, 2=Americas, 3=Asia, 4=Europe, 5=Middle East
    """
    cache_key = f"recent_d{days}_p{pagesize}_r{region or 'all'}_list"

    def _fetch():
        # Latest available year is 2024 in v25.1; we pull recent and filter.
        params: dict[str, Any] = {
            "pagesize": pagesize,
            "StartDate": _start_date(days),
            "EndDate": _today(),
        }
        if region:
            params["Region"] = region
        # Try API v25.1 (2024 data). Fall back to v24.1 if unavailable.
        for ver in ("25.1", "24.1", "23.1"):
            try:
                data = http_get_json(f"{BASE}/{ver}", params=params, timeout=25)
                rows = data.get("Result") or []
                logger.info("[ucdp] api %s returned %d rows", ver, len(rows))
                return [_normalize(r) for r in rows]
            except Exception as e:  # noqa: BLE001
                logger.info("[ucdp] api %s failed (%s); trying older", ver, str(e)[:80])
        return []

    return cached_get("ucdp", cache_key, _fetch, ttl=3600)


def _normalize(row: dict) -> dict:
    eid = str(row.get("id") or row.get("conflict_new_id") or "?")
    deaths = (row.get("best") or row.get("deaths_civilians") or 0) or 0
    side_a = row.get("side_a") or "?"
    side_b = row.get("side_b") or "?"
    where = row.get("where_coordinates") or row.get("country") or ""
    title = f"{side_a} vs {side_b} — {where}"[:160]
    desc = (row.get("source_article") or row.get("source_headline")
            or row.get("source_original") or "")[:1500]
    severity_proxy = min(1.0, deaths / 200.0)  # 200+ deaths -> tier CRITICAL
    return standard_event(
        source="ucdp", event_id=eid, title=title, description=desc,
        occurred_at_utc=row.get("date_start"),
        lat=row.get("latitude"), lon=row.get("longitude"),
        raw_url=f"https://ucdp.uu.se/apidocs/#GEDevent_id={eid}",
        severity_proxy=severity_proxy,
        extra={
            "deaths_best": int(deaths),
            "country": row.get("country"),
            "region_id": row.get("region"),
            "side_a": side_a, "side_b": side_b,
        },
    )


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _start_date(days_back: int) -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=days_back)).isoformat()


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    events = fetch_recent(days=90, pagesize=20, region=5)  # 5 = Middle East
    print(json.dumps(events[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(events)} events")
