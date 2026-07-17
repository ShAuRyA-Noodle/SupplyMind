"""provenance.py — source metadata + payload stamping for the live-signal layer.

Every live payload the war room renders must be traceable to a real public
source: which provider, under what licence, fetched when, and a content hash so
the exact bytes behind a number can be verified later. ``stamp()`` attaches that
block; ``source_meta()`` is the single registry of the eight live sources.

No source here is synthetic. A source that is down is still stamped — with the
provider it *would* have come from — and the caller marks it ``degraded`` so the
UI can show "source unreachable" rather than a fabricated value (CLAUDE.md §0).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------- #
# Source registry — the eight live signals behind the golden path's step 1.
# `env_var` names the key required (None = keyless). We never read the key's
# VALUE here; provenance only records which credential a source needs.
# --------------------------------------------------------------------------- #
SOURCE_META: dict[str, dict[str, Any]] = {
    "fred": {
        "provider": "Federal Reserve Bank of St. Louis (FRED)",
        "series": "DCOILBRENTEU — Crude Oil Prices: Brent, Europe (daily, USD/bbl)",
        "docs_url": "https://fred.stlouisfed.org/docs/api/fred/",
        "raw_url": "https://fred.stlouisfed.org/series/DCOILBRENTEU",
        "licence": "Public domain (U.S. Government work); series terms per FRED",
        "env_var": "FRED_API_KEY",
        "kind": "commodity",
    },
    "eia": {
        "provider": "U.S. Energy Information Administration (EIA) Open Data v2",
        "series": "petroleum/pri/spt — Brent spot price (Europe FOB), daily",
        "docs_url": "https://www.eia.gov/opendata/documentation.php",
        "raw_url": "https://www.eia.gov/opendata/browser/petroleum/pri/spt",
        "licence": "Public domain (U.S. Government work)",
        "env_var": "EIA_API_KEY",
        "kind": "commodity",
    },
    "nasa_firms": {
        "provider": "NASA FIRMS (Fire Information for Resource Management System)",
        "series": "VIIRS_SNPP_NRT active-fire detections, Strait of Hormuz bbox",
        "docs_url": "https://firms.modaps.eosdis.nasa.gov/api/",
        "raw_url": "https://firms.modaps.eosdis.nasa.gov/map/",
        "licence": "NASA open data (free use with attribution)",
        "env_var": "NASA_FIRMS_MAP_KEY",
        "kind": "hazard",
    },
    "gfw": {
        "provider": "Global Fishing Watch (GFW) API v3",
        "series": "public-global-port-visits-events — AIS-derived vessel port visits",
        "docs_url": "https://globalfishingwatch.org/our-apis/documentation",
        "raw_url": "https://globalfishingwatch.org/map/",
        "licence": "GFW data terms (free, attribution; CC BY-NC for some layers)",
        "env_var": "GFW_API_TOKEN",
        "kind": "shipping",
    },
    "newsapi": {
        "provider": "NewsAPI.org /v2/everything",
        "series": "Supply-chain / chokepoint headlines (Hormuz, Red Sea, ports)",
        "docs_url": "https://newsapi.org/docs",
        "raw_url": "https://newsapi.org",
        "licence": "NewsAPI Developer terms (free tier, 100 req/day)",
        "env_var": "NEWS_API_KEY",
        "kind": "news",
    },
    "noaa": {
        "provider": "NOAA National Weather Service (api.weather.gov)",
        "series": "Active weather alerts for US port states (TX/LA/GA/NJ/CA)",
        "docs_url": "https://www.weather.gov/documentation/services-web-api",
        "raw_url": "https://www.weather.gov/",
        "licence": "Public domain (U.S. Government work)",
        # NWS realtime alerts are keyless. The NOAA_TOKEN in .env is for the NCEI
        # Climate Data Online (CDO/GHCND) API, which was evaluated for this slot
        # but is unusable as a *live* signal: ~128 s latency and GHCND daily data
        # lags several days (empty for recent windows). NWS gives real, sub-2 s,
        # realtime NOAA weather instead.
        "env_var": None,
        "kind": "weather",
    },
    "usgs": {
        "provider": "USGS Earthquake Hazards Program (GeoJSON feed)",
        "series": "M4.5+ earthquakes, past 24h (keyless backup signal)",
        "docs_url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php",
        "raw_url": "https://earthquake.usgs.gov/earthquakes/map/",
        "licence": "Public domain (U.S. Government work)",
        "env_var": None,
        "kind": "hazard",
    },
    "gdelt": {
        "provider": "GDELT 2.0 DOC API",
        "series": "Chokepoint news volume (Hormuz / Red Sea / Iran-Israel)",
        "docs_url": "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
        "raw_url": "https://www.gdeltproject.org/",
        "licence": "GDELT open data (free use)",
        "env_var": None,
        "kind": "news",
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_meta(name: str) -> dict[str, Any]:
    """Return provenance metadata for a source, or a minimal record if unknown."""
    meta = SOURCE_META.get(name)
    if meta is None:
        return {"provider": name, "docs_url": None, "raw_url": None,
                "licence": "unknown", "env_var": None, "kind": "unknown"}
    return dict(meta)


def sha256_of(payload: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable payload (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stamp(name: str, payload: Any, fetched_at: str | None = None) -> dict[str, Any]:
    """Return a provenance block for ``payload`` fetched from source ``name``.

    The block records provider, licence, docs URL, the fetch timestamp, and a
    content hash of the payload so any downstream number can be traced back to
    the exact bytes it came from.
    """
    meta = source_meta(name)
    return {
        "source": name,
        "provider": meta.get("provider"),
        "series": meta.get("series"),
        "licence": meta.get("licence"),
        "docs_url": meta.get("docs_url"),
        "raw_url": meta.get("raw_url"),
        "requires_key": meta.get("env_var"),
        "fetched_at": fetched_at or _now_iso(),
        "content_sha256": sha256_of(payload),
    }
