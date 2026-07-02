"""reliefweb.py — UN OCHA ReliefWeb API.

Real humanitarian crisis appeals + situation reports + funding asks.
No auth. Public API.

Docs: https://apidoc.rwlabs.org/
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://api.reliefweb.int/v1"


def fetch_recent_disasters(days: int = 30, limit: int = 30) -> list[dict]:
    """Pull recent disasters with current status from ReliefWeb."""
    cache_key = f"disasters_d{days}_l{limit}_list"

    def _fetch():
        # ReliefWeb POST API with filters
        body = {
            "appname": "supplymind",
            "limit": limit,
            "sort": ["date.created:desc"],
            "filter": {
                "operator": "AND",
                "conditions": [
                    {"field": "status", "value": ["alert", "current", "ongoing"]},
                ],
            },
            "fields": {"include": [
                "name", "status", "date", "country", "type",
                "primary_country", "url_alias", "description",
            ]},
        }
        import httpx
        with httpx.Client(timeout=25) as c:
            r = c.post(f"{BASE}/disasters", json=body)
            r.raise_for_status()
            data = r.json()
        rows = data.get("data") or []
        logger.info("[reliefweb] returned %d disasters", len(rows))
        return [_normalize(r) for r in rows]

    return cached_get("reliefweb", cache_key, _fetch, ttl=1800)


def fetch_recent_reports(query: str = "supply chain disruption", limit: int = 20) -> list[dict]:
    """Pull recent situation reports matching a query string."""
    cache_key = f"reports_q{query[:30]}_l{limit}_list"

    def _fetch():
        body = {
            "appname": "supplymind",
            "limit": limit,
            "query": {"value": query},
            "sort": ["date.created:desc"],
            "fields": {"include": [
                "title", "date", "country", "primary_country", "url",
                "body-html", "format",
            ]},
        }
        import httpx
        with httpx.Client(timeout=25) as c:
            r = c.post(f"{BASE}/reports", json=body)
            r.raise_for_status()
            data = r.json()
        rows = data.get("data") or []
        logger.info("[reliefweb] returned %d reports for '%s'", len(rows), query[:30])
        return [_normalize_report(r) for r in rows]

    return cached_get("reliefweb", cache_key, _fetch, ttl=1800)


def _normalize(row: dict) -> dict:
    fields = row.get("fields") or {}
    eid = str(row.get("id") or "?")
    primary = (fields.get("primary_country") or {}).get("name") or "?"
    types = ", ".join(t.get("name", "") for t in fields.get("type", []))
    title = f"{fields.get('name', '?')} — {primary}"[:160]
    desc = (fields.get("description") or "")[:1500]
    status = (fields.get("status") or "").lower()
    sev = {"alert": 0.95, "current": 0.7, "ongoing": 0.6}.get(status, 0.5)
    return standard_event(
        source="reliefweb", event_id=eid, title=title, description=desc,
        occurred_at_utc=(fields.get("date") or {}).get("created"),
        raw_url=f"https://reliefweb.int/disaster/{fields.get('url_alias', '')}",
        severity_proxy=sev,
        extra={
            "primary_country": primary,
            "type": types,
            "status": status,
        },
    )


def _normalize_report(row: dict) -> dict:
    fields = row.get("fields") or {}
    eid = str(row.get("id") or "?")
    primary = (fields.get("primary_country") or {}).get("name") or "?"
    return standard_event(
        source="reliefweb_report", event_id=eid,
        title=(fields.get("title") or "")[:160], description="",
        occurred_at_utc=(fields.get("date") or {}).get("created"),
        raw_url=fields.get("url") or "",
        severity_proxy=0.4,
        extra={"primary_country": primary},
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_recent_disasters(days=30, limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal disasters: {len(e)}")
