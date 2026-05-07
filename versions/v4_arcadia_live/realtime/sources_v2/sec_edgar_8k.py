"""sec_edgar_8k.py — SEC EDGAR full-text search for 8-K filings.

Real US public-company 8-K filings (force-majeure, supply-chain disruption,
material agreement, cybersecurity incident). No auth required.

Docs: https://efts.sec.gov/LATEST/search-index?
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"


def fetch_supply_chain_filings(
    keyword: str = "supply chain disruption",
    days_back: int = 30,
    limit: int = 25,
) -> list[dict]:
    cache_key = f"edgar_{keyword[:30].replace(' ','_')}_d{days_back}_n{limit}_list"

    def _fetch():
        end = datetime.utcnow().date()
        start = end - timedelta(days=days_back)
        params = {
            "q": f'"{keyword}"',
            "dateRange": "custom",
            "startdt": start.isoformat(),
            "enddt":   end.isoformat(),
            "forms": "8-K",
        }
        try:
            data = http_get_json(
                EDGAR_FULL_TEXT, params=params,
                headers={"User-Agent": "supplymind-research/1.0 (mailto:research@supplymind.dev)"},
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[sec_edgar] fetch failed: %s", str(e)[:120])
            return []
        hits = (data.get("hits") or {}).get("hits") or []
        out = [_normalize(h) for h in hits[:limit]]
        logger.info("[sec_edgar] returned %d 8-K hits for '%s'", len(out), keyword)
        return out

    return cached_get("sec_edgar_8k", cache_key, _fetch, ttl=3600)


def _normalize(h: dict) -> dict:
    src = h.get("_source") or {}
    eid = h.get("_id") or "?"
    company = ", ".join(src.get("display_names") or [])
    file_date = src.get("file_date")
    adsh = src.get("adsh") or ""
    cik = (src.get("ciks") or [""])[0]
    items = ", ".join(src.get("items") or [])
    accession_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                     f"action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10")

    # Severity proxy by filing item codes (Item 8.01 = Other; 1.01 = Material Agreement;
    # 1.05 = Material Cybersecurity; 2.06 = Material Impairment; 5.02 = Officer changes)
    items_low = items.lower()
    sev = 0.4
    if "1.05" in items_low or "cyber" in items_low: sev = 0.75
    elif "2.06" in items_low or "impairment" in items_low: sev = 0.7
    elif "1.01" in items_low or "material agreement" in items_low: sev = 0.55

    return standard_event(
        source="sec_edgar_8k", event_id=eid,
        title=f"{company} — 8-K (items {items})"[:160],
        description=(f"SEC 8-K filing by {company}, items {items}, "
                     f"filed {file_date}, accession {adsh}. "
                     f"Search query matched.")[:1500],
        occurred_at_utc=f"{file_date}T00:00:00Z" if file_date else None,
        raw_url=accession_url,
        severity_proxy=sev,
        extra={
            "company": company, "cik": cik, "items": items,
            "file_date": file_date, "accession": adsh,
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_supply_chain_filings(keyword="supply chain disruption", days_back=60, limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} SEC 8-K filings")
