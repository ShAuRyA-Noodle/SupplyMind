"""gdelt_humanitarian.py — GDELT 2.0 humanitarian-themed filter
(ReliefWeb substitute).

ReliefWeb v2 API requires a registered "approved appname" (free but a
signup gate). We substitute with GDELT 2.0 themed by HUM_* /
WB_*HUMANITARIAN themes which surface humanitarian crisis articles
covered by ReliefWeb anyway, without any auth.
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_humanitarian_events(
    query: str = (
        "(humanitarian OR famine OR refugees OR displaced OR drought "
        "OR earthquake OR cyclone OR floods OR cholera) AND "
        "(crisis OR emergency OR disaster OR appeal)"
    ),
    timespan: str = "30d",
    maxrecords: int = 50,
) -> list[dict]:
    cache_key = f"hum_{timespan}_n{maxrecords}_list"

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
        logger.info("[gdelt_humanitarian] returned %d articles", len(articles))
        return [_normalize(a) for a in articles]

    return cached_get("gdelt_humanitarian", cache_key, _fetch, ttl=1800)


def _normalize(art: dict) -> dict:
    title = (art.get("title") or "")[:160]
    url = art.get("url") or ""
    seendate = art.get("seendate")
    domain = art.get("domain") or ""
    tone = art.get("tone")
    eid = (art.get("url") or title)[:160]

    title_low = title.lower()
    sev = 0.4
    if any(hi in title_low for hi in ("famine", "death toll", "killed", "refugees")):
        sev = 0.75
    elif any(med in title_low for med in ("disaster", "crisis", "emergency",
                                            "displaced", "humanitarian")):
        sev = 0.55

    return standard_event(
        source="gdelt_humanitarian", event_id=eid, title=title,
        description=(f"GDELT humanitarian-themed article from {domain}, "
                     f"tone={tone}.")[:1500],
        occurred_at_utc=_iso_seendate(seendate),
        raw_url=url, severity_proxy=sev,
        extra={"domain": domain, "tone": tone},
    )


def _iso_seendate(s: str | None) -> str | None:
    if not s or len(s) < 14: return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_humanitarian_events(timespan="14d", maxrecords=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} humanitarian articles")
