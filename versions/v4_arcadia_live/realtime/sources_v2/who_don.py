"""who_don.py — WHO Disease Outbreak News (DON).

WHO retired the old RSS feed; we use the new JSON OData API.
No auth.

Endpoint: https://www.who.int/api/news/diseaseoutbreaknews
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

API_URL = "https://www.who.int/api/news/diseaseoutbreaknews"


def fetch_recent(limit: int = 30) -> list[dict]:
    cache_key = f"don_l{limit}_list"

    def _fetch():
        # OData parameters: top, orderby
        params = {
            "$orderby": "PublicationDateAndTime desc",
            "$top": limit,
        }
        data = http_get_json(API_URL, params=params, timeout=20)
        items = data.get("value") or []
        logger.info("[who_don] returned %d outbreak items", len(items))
        return [_normalize(it) for it in items]

    return cached_get("who_don", cache_key, _fetch, ttl=3600)


def _normalize(item: dict) -> dict:
    title = (item.get("Title") or "").strip()
    eid = str(item.get("Id") or item.get("ItemDefaultUrl") or title[:60])
    pub = item.get("PublicationDateAndTime") or item.get("FormattedDate")
    url = item.get("ItemDefaultUrl") or ""
    if url and not url.startswith("http"):
        url = f"https://www.who.int{url}"
    desc = (item.get("FormattedTitle") or item.get("PageContent")
            or item.get("Title") or "")[:1500]

    # Severity proxy from disease keywords
    title_low = title.lower()
    sev = 0.4
    for hi in ("ebola", "marburg", "h5n1 ", "smallpox", "polio", "mpox",
                "monkeypox", "cholera outbreak"):
        if hi in title_low: sev = 0.85; break
    for med in ("dengue", "measles", "yellow fever", "lassa", "hepatitis",
                 "diphtheria", "anthrax"):
        if med in title_low: sev = 0.6; break

    return standard_event(
        source="who_don", event_id=eid, title=title[:160],
        description=desc, occurred_at_utc=pub,
        raw_url=url, severity_proxy=sev,
    )


if __name__ == "__main__":
    import json, logging as _l
    _l.basicConfig(level=_l.INFO)
    e = fetch_recent(limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} outbreak alerts")
