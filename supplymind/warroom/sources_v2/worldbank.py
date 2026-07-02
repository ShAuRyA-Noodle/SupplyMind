"""worldbank.py — World Bank Open Data API.

Real macroeconomic indicators per country. No auth. Free.
Pulls latest GDP / inflation / current-account-balance for key
supply-chain countries.
"""
from __future__ import annotations

import logging

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://api.worldbank.org/v2"

# Country ISO3 → friendly name
COUNTRIES = {
    "USA": "United States", "CHN": "China", "JPN": "Japan",
    "DEU": "Germany", "KOR": "South Korea", "IND": "India",
    "TWN": "Taiwan", "NLD": "Netherlands", "SGP": "Singapore",
    "IRN": "Iran", "ISR": "Israel", "ARE": "UAE",
    "EGY": "Egypt", "PAN": "Panama",
}

# Indicators that move with crisis exposure
INDICATORS = {
    "NY.GDP.MKTP.CD":           "GDP (current US$)",
    "FP.CPI.TOTL.ZG":           "Inflation, consumer prices (% annual)",
    "BN.CAB.XOKA.CD":           "Current account balance (US$)",
}


def fetch_macro_signals() -> list[dict]:
    """Pull latest year of each indicator for each country."""
    cache_key = "macro_list"

    def _fetch():
        out: list[dict] = []
        for ind_code, ind_name in INDICATORS.items():
            for iso3, name in COUNTRIES.items():
                url = f"{BASE}/country/{iso3}/indicator/{ind_code}"
                params = {"format": "json", "per_page": 5,
                          "date": "2018:2024"}
                try:
                    # Tight timeout — World Bank API often hangs.
                    # Max 4s/call * 42 calls = 168s worst-case but
                    # most return quickly; if any hang, we move on.
                    data = http_get_json(url, params=params, timeout=4)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[worldbank] %s/%s skipped (%s)",
                                    iso3, ind_code, str(e)[:40])
                    continue
                # WB returns [meta, list]
                if not isinstance(data, list) or len(data) < 2: continue
                rows = [r for r in (data[1] or []) if r.get("value") is not None]
                if not rows: continue
                latest = rows[0]
                out.append(_normalize(iso3, name, ind_code, ind_name, latest))
        logger.info("[worldbank] returned %d country-indicator latest values", len(out))
        return out

    return cached_get("worldbank", cache_key, _fetch, ttl=86400)


def _normalize(iso3: str, country: str,
                ind_code: str, ind_name: str, latest: dict) -> dict:
    val = latest.get("value")
    year = latest.get("date") or "?"

    sev = 0.0
    try:
        if "Inflation" in ind_name and float(val) > 10:
            sev = min(1.0, float(val) / 50.0)
        elif "Current account" in ind_name and float(val) < -50_000_000_000:
            sev = 0.6
    except (ValueError, TypeError):
        pass

    return standard_event(
        source="worldbank", event_id=f"wb_{iso3}_{ind_code}_{year}",
        title=f"{country} — {ind_name} ({year}): {val}",
        description=(f"World Bank indicator {ind_code} for {country} "
                     f"in {year}: {val}.")[:1500],
        occurred_at_utc=f"{year}-12-31T00:00:00Z" if str(year).isdigit() else None,
        raw_url=(f"https://data.worldbank.org/indicator/{ind_code}?"
                 f"locations={iso3}"),
        severity_proxy=sev,
        extra={
            "country_iso3": iso3, "country": country,
            "indicator": ind_code, "value": val, "year": year,
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_macro_signals()
    print(json.dumps(e[:5], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} World Bank macro signals")
