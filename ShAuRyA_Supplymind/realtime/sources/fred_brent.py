"""
fred_brent.py — FRED Brent crude daily spot price polling.

Series: DCOILBRENTEU — Crude Oil Prices: Brent - Europe (daily, USD/barrel).
Free with FRED_API_KEY. Docs: https://fred.stlouisfed.org/docs/api/fred/

Signal logic: a price spike > 5% day-over-day or > 10% week-over-week raises
the severity. Normal oscillations are noise.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from ..store import Event

logger = logging.getLogger(__name__)

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "DCOILBRENTEU"  # Brent; alt: DCOILWTICO for WTI
SOURCE_NAME = "fred_brent"


def _severity_from_price_change(dod_pct: float, wow_pct: float) -> float:
    """Higher of (|dod| / 5%) and (|wow| / 10%), capped at 1.0."""
    return float(min(1.0, max(abs(dod_pct) / 5.0, abs(wow_pct) / 10.0)))


def _fetch_series(limit: int = 10, api_key: Optional[str] = None) -> list[dict]:
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        logger.warning("[fred_brent] FRED_API_KEY not set")
        return []
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    resp = requests.get(
        FRED_ENDPOINT,
        params={
            "series_id": SERIES_ID,
            "api_key": key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        logger.warning("[fred_brent] %d %s", resp.status_code, resp.text[:200])
        return []
    return resp.json().get("observations", [])


def fetch(api_key: Optional[str] = None) -> list[Event]:
    """Return a single Brent-crude price event."""
    obs = _fetch_series(limit=10, api_key=api_key)
    if not obs:
        return []

    # Filter to numeric prices (FRED uses "." for missing)
    prices = []
    for o in obs:
        v = o.get("value")
        try:
            prices.append((o["date"], float(v)))
        except (ValueError, TypeError):
            continue

    if not prices:
        return []

    latest_date, latest_price = prices[0]
    # Day-over-day (previous numeric observation)
    dod_pct = 0.0
    if len(prices) >= 2:
        prev_price = prices[1][1]
        dod_pct = (latest_price - prev_price) / prev_price * 100
    # Week-over-week (approx: 5 trading days back)
    wow_pct = 0.0
    if len(prices) >= 6:
        wow_price = prices[5][1]
        wow_pct = (latest_price - wow_price) / wow_price * 100

    sev = _severity_from_price_change(dod_pct, wow_pct)
    raw = (f"Brent crude ({SERIES_ID}) spot {latest_date}: ${latest_price:.2f}/bbl "
           f"(DoD {dod_pct:+.2f}%, WoW {wow_pct:+.2f}%)")
    ev = Event(
        source=SOURCE_NAME,
        ts_iso=f"{latest_date}T00:00:00Z",
        event_type="commodity_signal",
        region="global",
        severity=sev,
        raw_text=raw,
        urls=[f"https://fred.stlouisfed.org/series/{SERIES_ID}"],
        entities=["Brent"],
        meta={
            "series_id": SERIES_ID,
            "latest_price": latest_price,
            "dod_pct": dod_pct,
            "wow_pct": wow_pct,
            "observations_used": len(prices),
        },
    )
    logger.info("[fred_brent] ${:.2f}/bbl DoD={:+.2f}%% WoW={:+.2f}%% sev={:.2f}".format(
        latest_price, dod_pct, wow_pct, sev))
    return [ev]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evs = fetch()
    for e in evs:
        print(f"{e.ts_iso}  {e.region}  sev={e.severity:.2f}  {e.raw_text}")
