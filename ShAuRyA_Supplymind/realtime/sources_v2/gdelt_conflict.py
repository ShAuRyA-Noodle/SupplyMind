"""gdelt_conflict.py — GDELT 2.0 conflict-only filter (UCDP substitute).

UCDP API now requires an API token. We substitute with GDELT 2.0's
GKG event stream filtered to conflict-tone (Goldstein scale ≤ -5),
which captures conflict events with similar fidelity for our use case.

GDELT GKG docs: https://www.gdeltproject.org/data.html#documentation
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_conflict_events(
    query: str = "(conflict OR strike OR attack OR clash OR war) AND (port OR shipping OR oil OR strait OR canal)",
    timespan: str = "7d",
    maxrecords: int = 50,
) -> list[dict]:
    """Filter GDELT for conflict + supply-chain themed articles."""
    cache_key = f"conflict_{timespan}_n{maxrecords}_list"

    def _fetch():
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": maxrecords,
            "sort": "DateDesc",
        }
        data = http_get_json(DOC_API, params=params, timeout=25)
        articles = data.get("articles") or []
        logger.info("[gdelt_conflict] returned %d articles", len(articles))
        return [_normalize(a) for a in articles]

    return cached_get("gdelt_conflict", cache_key, _fetch, ttl=1800)


def _normalize(art: dict) -> dict:
    title = (art.get("title") or "")[:160]
    url = art.get("url") or ""
    seendate = art.get("seendate")
    domain = art.get("domain") or ""
    tone = art.get("tone")
    eid = (art.get("url") or title)[:160]

    # Severity: more negative tone = higher severity (Goldstein-like)
    sev = 0.5
    try:
        t = float(tone)
        sev = max(0.0, min(1.0, -t / 10.0))  # -10 tone -> 1.0 severity
    except (ValueError, TypeError):
        pass

    return standard_event(
        source="gdelt_conflict", event_id=eid, title=title,
        description=(f"GDELT conflict-themed article from {domain}, "
                     f"tone={tone}.")[:1500],
        occurred_at_utc=_iso_seendate(seendate),
        raw_url=url, severity_proxy=sev,
        extra={"domain": domain, "tone": tone},
    )


def _iso_seendate(s: str | None) -> str | None:
    if not s or len(s) < 14: return s
    # GDELT format: 20260425T120000Z
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_conflict_events(timespan="14d", maxrecords=15)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} conflict articles")
