"""
gdelt.py — GDELT 2.0 Doc API polling.

GDELT is free, no key. Refreshes every 15 min. We query the DOC 2.0 API for
articles mentioning Hormuz/Iran/Israel/Red Sea with tone filter.

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from ..store import Event

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
SOURCE_NAME = "gdelt"

QUERIES = {
    "hormuz": 'sourcelang:eng ("strait of hormuz" OR "persian gulf tanker" OR "gulf of oman")',
    "iran_israel": 'sourcelang:eng ("iran israel" OR "israeli strike iran" OR "irgc")',
    "red_sea": 'sourcelang:eng ("red sea" OR "bab el mandeb" OR "houthi vessel")',
    "taiwan_strait": 'sourcelang:eng ("taiwan strait" OR "pla exercise taiwan")',
}

REGION_TAG = {k: k for k in QUERIES}


def _parse_seen(seen: str) -> str:
    """GDELT 'seendate' is YYYYMMDDHHMMSS; return ISO-8601."""
    if not seen or len(seen) < 14:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        dt = datetime.strptime(seen[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _severity_from_tone(tone: Optional[float]) -> float:
    """GDELT tone is in [-100, 100]; strong negative tone = high severity."""
    if tone is None:
        return 0.3
    # tone -10 => ~0.55, tone -20 => ~0.8, tone -30+ => 1.0
    return float(max(0.0, min(1.0, (abs(min(tone, 0)) / 30))))


def fetch(lookback_minutes: int = 120) -> list[Event]:
    """Poll GDELT Doc API for tracked queries. No API key required."""
    events: list[Event] = []
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    startdatetime = since.strftime("%Y%m%d%H%M%S")
    enddatetime = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    for q_name, q_str in QUERIES.items():
        try:
            resp = requests.get(
                GDELT_DOC_API,
                params={
                    "query": q_str,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": 30,
                    "startdatetime": startdatetime,
                    "enddatetime": enddatetime,
                },
                timeout=30,
                headers={"User-Agent": "SupplyMind/1.0 (+https://github.com/ShAuRyA-Noodle/Sleep-Token)"},
            )
            if resp.status_code != 200:
                logger.warning("[gdelt] %s -> %d", q_name, resp.status_code)
                continue
            # GDELT sometimes returns HTML on rate limit; guard
            try:
                data = resp.json()
            except Exception:
                logger.warning("[gdelt] %s returned non-JSON, skipping", q_name)
                continue

            for art in data.get("articles", []):
                title = art.get("title") or ""
                if not title:
                    continue
                tone = art.get("tone")
                try:
                    tone_f = float(tone) if tone is not None else None
                except Exception:
                    tone_f = None
                ev = Event(
                    source=SOURCE_NAME,
                    ts_iso=_parse_seen(art.get("seendate") or ""),
                    event_type="news_signal",
                    region=REGION_TAG[q_name],
                    severity=_severity_from_tone(tone_f),
                    raw_text=title,
                    urls=[art.get("url") or ""],
                    entities=[],
                    meta={
                        "query": q_name,
                        "tone": tone_f,
                        "sourcecountry": art.get("sourcecountry"),
                        "domain": art.get("domain"),
                    },
                )
                events.append(ev)
        except Exception as e:  # noqa: BLE001
            logger.error("[gdelt] %s fetch failed: %s", q_name, e)

    logger.info("[gdelt] fetched %d events across %d queries", len(events), len(QUERIES))
    return events


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-min", type=int, default=120)
    args = parser.parse_args()

    evs = fetch(args.lookback_min)
    for e in evs[:10]:
        print(f"{e.ts_iso}  {e.region:15s}  sev={e.severity:.2f}  {e.raw_text[:80]}")
    print(f"\ntotal: {len(evs)}")
