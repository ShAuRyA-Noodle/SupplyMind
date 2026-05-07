"""cisa_kev.py — CISA Known Exploited Vulnerabilities catalog.

Real cyber attacks: vulnerabilities CISA confirms are being actively
exploited in the wild. Each entry is a real CVE.

No auth. Public JSON.
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_recent(days_back: int = 30, limit: int = 30) -> list[dict]:
    from datetime import date, datetime, timedelta
    cache_key = f"kev_d{days_back}_n{limit}_list"

    def _fetch():
        try:
            data = http_get_json(KEV_URL, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("[cisa_kev] fetch failed: %s", str(e)[:120])
            return []
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        items = data.get("vulnerabilities") or []
        recent = [v for v in items if (v.get("dateAdded") or "") >= cutoff]
        recent.sort(key=lambda v: v.get("dateAdded", ""), reverse=True)
        out = [_normalize(v) for v in recent[:limit]]
        logger.info("[cisa_kev] returned %d recent KEV entries (catalog total %d)",
                    len(out), len(items))
        return out

    return cached_get("cisa_kev", cache_key, _fetch, ttl=21600)


def _normalize(v: dict) -> dict:
    cve = v.get("cveID") or "?"
    vendor = v.get("vendorProject") or "?"
    product = v.get("product") or "?"
    name = v.get("vulnerabilityName") or ""
    desc = v.get("shortDescription") or ""
    ransomware = (v.get("knownRansomwareCampaignUse") or "Unknown").lower()
    notes = v.get("notes") or ""

    sev = 0.5
    if "yes" in ransomware: sev = 0.85
    elif "known" in ransomware: sev = 0.7

    return standard_event(
        source="cisa_kev", event_id=cve,
        title=f"CISA KEV {cve} — {vendor} {product}",
        description=(f"{name}. {desc} Ransomware-use: {ransomware}.")[:1500],
        occurred_at_utc=f"{v.get('dateAdded')}T00:00:00Z" if v.get("dateAdded") else None,
        raw_url=f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search={cve}",
        severity_proxy=sev,
        extra={
            "cve_id": cve, "vendor": vendor, "product": product,
            "ransomware_use": ransomware,
            "due_date": v.get("dueDate"),
            "required_action": (v.get("requiredAction") or "")[:240],
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_recent(days_back=60, limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} recent KEV entries")
