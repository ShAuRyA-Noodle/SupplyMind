"""hackernews.py — HackerNews via Algolia search API.

Tech industry pulse signal. No auth. Free. Real public posts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_supply_chain_signal(query: str = "supply chain", hours_back: int = 48,
                                limit: int = 30) -> list[dict]:
    cache_key = f"hn_{query.replace(' ','_')[:20]}_h{hours_back}_l{limit}_list"

    def _fetch():
        epoch_min = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": f"created_at_i>={epoch_min}",
            "hitsPerPage": limit,
        }
        try:
            data = http_get_json(ALGOLIA, params=params, timeout=20)
        except Exception as e:  # noqa: BLE001
            logger.warning("[hackernews] fetch failed: %s", str(e)[:120])
            return []
        hits = data.get("hits") or []
        out = [_normalize(h) for h in hits]
        logger.info("[hackernews] returned %d HN stories for '%s'", len(out), query)
        return out

    return cached_get("hackernews", cache_key, _fetch, ttl=1800)


def _normalize(h: dict) -> dict:
    title = (h.get("title") or "")[:160]
    url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
    eid = h.get("objectID") or url
    points = h.get("points") or 0
    comments = h.get("num_comments") or 0
    created = h.get("created_at")

    # Severity proxy: high engagement (popular post = real attention)
    sev = min(1.0, (points + comments * 2) / 200.0)

    return standard_event(
        source="hackernews", event_id=str(eid), title=title,
        description=(f"HackerNews story by {h.get('author')}. "
                     f"Points: {points}, comments: {comments}, "
                     f"created: {created}")[:1500],
        occurred_at_utc=created,
        raw_url=url, severity_proxy=sev,
        extra={"points": points, "n_comments": comments,
               "author": h.get("author")},
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_supply_chain_signal(query="supply chain", hours_back=72, limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} HN stories")
