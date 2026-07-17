"""live.py — real per-source live-signal clients with freshness stamps.

This is golden-path step 1: *a real live event surfaces*. Eight public sources
(six keyed + two keyless) are polled directly here — FRED Brent, EIA petroleum,
NASA FIRMS active fires, GFW vessel presence, NewsAPI headlines, NOAA CDO
weather, plus USGS earthquakes and GDELT news as keyless backups.

Contract (CLAUDE.md §0 — fail loud, never fake):
  Every source returns a :class:`SourceStatus` with ``fetched_at`` /
  ``age_seconds`` / ``degraded`` / ``degraded_reason``. A source that is down,
  rate-limited, or missing its key is reported ``degraded=True`` with the real
  reason (HTTP status, error text, "key not set") — never a synthesised value.

Caching: each source's last good payload is written to
``external_data/live_cache/<source>.json`` with its fetch timestamp so the war
room can show "< N h old" even across restarts, and so we never hammer a free
tier (default TTL 10 min). ``FORCE_REPLAY=1`` serves only cached snapshots and
sets ``replay=True`` so the UI can show a visible REPLAY banner.

Run once from a fresh shell::

    python -m supplymind.data.live            # cache-aware table
    python -m supplymind.data.live --live     # force one live call per source
    python -m supplymind.data.live --receipt tests/receipts/live_data_freshness_REAL.json
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .provenance import SOURCE_META, source_meta, stamp

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "external_data" / "live_cache"
DEFAULT_TTL_S = 600  # 10 min — free tiers are polite-polled
FAIL_TTL_S = 120     # re-probe a down source at most once every 2 min

# The eight live sources, in the order the war room renders them.
SOURCES: tuple[str, ...] = (
    "fred", "eia", "nasa_firms", "gfw", "newsapi", "noaa", "usgs", "gdelt",
)

# Strait of Hormuz bounding box (lon_min, lat_min, lon_max, lat_max) — the
# persona's anchor chokepoint. Shared by the FIRMS + GFW region filters.
HORMUZ_BBOX = (54.0, 24.0, 58.0, 28.0)

_ENV_LOADED = False


def _load_env() -> None:
    """Load ``.env`` into ``os.environ`` once (values never logged).

    Mirrors the ingestor's loader so FRED/NewsAPI keys (read via ``os.environ``)
    are available when this module is run as a script. Existing environment
    variables are never overridden.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    except ImportError:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveFetchError(Exception):
    """A source client could not obtain real data. Carries a loud reason.

    Raised (never swallowed) whenever a live call fails — missing key, non-200
    HTTP, malformed body, or an API-level error object. ``fetch_source`` turns
    it into a ``degraded=True`` status; it is never converted into a fake value.
    """

    def __init__(self, reason: str, *, status: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _get(url: str, *, params: dict | None = None, headers: dict | None = None,
         timeout: float = 25.0, retries: int = 2) -> httpx.Response:
    """GET with one retry on transient transport errors (connection resets).

    Only *transport* failures are retried — an HTTP error status is returned
    as-is so the caller can surface it honestly. The final failure is raised
    loudly as a :class:`LiveFetchError`, never swallowed.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as c:
                return c.get(url, params=params, headers=headers)
        except httpx.HTTPError as e:
            last = e
            if attempt < retries:
                time.sleep(0.8)
    raise LiveFetchError(f"network error: {type(last).__name__}: {str(last)[:160]}")


def _key(env_var: str) -> str:
    _load_env()
    val = (os.environ.get(env_var) or "").strip()
    if not val:
        raise LiveFetchError(f"{env_var} not set (add it to .env)")
    return val


def _in_bbox(lon: float | None, lat: float | None,
             bbox: tuple[float, float, float, float]) -> bool:
    if lon is None or lat is None:
        return False
    lo_min, la_min, lo_max, la_max = bbox
    return lo_min <= lon <= lo_max and la_min <= lat <= la_max


# --------------------------------------------------------------------------- #
# Per-source clients. Each returns a normalised payload dict:
#   {rows: [ {ts, text, severity, region, url, ...} ], value, count, headline,
#    raw_url}  — or raises LiveFetchError. `value` is the source's headline
#   scalar (e.g. Brent price) or None. `rows` is a small sample (<=8).
# --------------------------------------------------------------------------- #

def _client_fred() -> dict[str, Any]:
    key = _key("FRED_API_KEY")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    r = _get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": "DCOILBRENTEU", "api_key": key, "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "sort_order": "desc", "limit": 10,
        },
    )
    if r.status_code != 200:
        raise LiveFetchError(f"FRED HTTP {r.status_code}: {r.text[:120]}",
                             status=r.status_code)
    obs = r.json().get("observations", [])
    prices: list[tuple[str, float]] = []
    for o in obs:
        try:
            prices.append((o["date"], float(o["value"])))
        except (ValueError, TypeError, KeyError):
            continue
    if not prices:
        raise LiveFetchError("FRED returned no numeric Brent observations")
    latest_date, latest = prices[0]
    dod = ((latest - prices[1][1]) / prices[1][1] * 100) if len(prices) >= 2 else 0.0
    wow = ((latest - prices[5][1]) / prices[5][1] * 100) if len(prices) >= 6 else 0.0
    sev = min(1.0, max(abs(dod) / 5.0, abs(wow) / 10.0))
    return {
        "value": round(latest, 2),
        "count": len(prices),
        "headline": f"Brent ${latest:.2f}/bbl on {latest_date} (DoD {dod:+.2f}%, WoW {wow:+.2f}%)",
        "raw_url": SOURCE_META["fred"]["raw_url"],
        "rows": [{
            "ts": f"{latest_date}T00:00:00Z", "region": "global",
            "severity": round(sev, 3),
            "text": f"Brent crude ${latest:.2f}/bbl (DoD {dod:+.2f}%, WoW {wow:+.2f}%)",
            "url": SOURCE_META["fred"]["raw_url"],
            "dod_pct": round(dod, 3), "wow_pct": round(wow, 3),
        }],
    }


def _client_eia() -> dict[str, Any]:
    key = _key("EIA_API_KEY")
    r = _get(
        "https://api.eia.gov/v2/petroleum/pri/spt/data/",
        params={
            "api_key": key, "frequency": "daily", "data[0]": "value",
            # RBRTE = Europe Brent Spot Price FOB ($/bbl), EIA's canonical series id.
            "facets[series][]": "RBRTE",
            "sort[0][column]": "period", "sort[0][direction]": "desc",
            "offset": 0, "length": 5,
        },
    )
    if r.status_code != 200:
        raise LiveFetchError(f"EIA HTTP {r.status_code}: {r.text[:120]}",
                             status=r.status_code)
    body = r.json()
    rows = ((body.get("response") or {}).get("data")) or []
    if not rows:
        raise LiveFetchError("EIA returned no Brent spot observations")
    latest = rows[0]
    val = latest.get("value")
    period = latest.get("period")
    units = latest.get("units") or "$/bbl"
    return {
        "value": val,
        "count": len(rows),
        "headline": f"EIA Brent spot {val} {units} ({period})",
        "raw_url": SOURCE_META["eia"]["raw_url"],
        "rows": [{
            "ts": f"{period}T00:00:00Z" if period else None, "region": "global",
            "severity": None,
            "text": f"EIA Brent spot (Europe FOB): {val} {units}, period {period}",
            "url": SOURCE_META["eia"]["raw_url"], "period": period, "units": units,
        }],
    }


def _client_nasa_firms() -> dict[str, Any]:
    key = _key("NASA_FIRMS_MAP_KEY")
    lo_min, la_min, lo_max, la_max = HORMUZ_BBOX
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
           f"VIIRS_SNPP_NRT/{lo_min},{la_min},{lo_max},{la_max}/1")
    r = _get(url)
    if r.status_code != 200:
        raise LiveFetchError(f"FIRMS HTTP {r.status_code}: {r.text[:120]}",
                             status=r.status_code)
    text = r.text
    low = text[:200].lower()
    if "invalid" in low or "error" in low:
        raise LiveFetchError(f"FIRMS error response: {text[:120]}")
    rows_out: list[dict] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            lat = float(row.get("latitude", 0)); lon = float(row.get("longitude", 0))
            frp = float(row.get("frp") or 0)
        except (ValueError, TypeError):
            continue
        sev = min(1.0, frp / 200.0)
        rows_out.append({
            "ts": f"{row.get('acq_date')}T{(row.get('acq_time') or '0000')[:2]}:"
                  f"{(row.get('acq_time') or '0000')[2:]}:00Z",
            "region": "strait_of_hormuz", "severity": round(sev, 3),
            "text": f"Active fire near Strait of Hormuz (FRP {frp:.0f} MW, "
                    f"conf {row.get('confidence')})",
            "url": SOURCE_META["nasa_firms"]["raw_url"],
            "lat": lat, "lon": lon, "frp_mw": frp,
        })
    return {
        "value": len(rows_out),
        "count": len(rows_out),
        "headline": (f"{len(rows_out)} active-fire detection(s) in Strait-of-Hormuz "
                     f"bbox, last 24h" if rows_out
                     else "No active fires in Strait-of-Hormuz bbox, last 24h"),
        "raw_url": SOURCE_META["nasa_firms"]["raw_url"],
        "rows": rows_out[:8],
    }


def _client_gfw() -> dict[str, Any]:
    token = _key("GFW_API_TOKEN")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=7)
    r = _get(
        "https://gateway.api.globalfishingwatch.org/v3/events",
        params={
            "datasets[0]": "public-global-port-visits-events:latest",
            "start-date": start.isoformat(), "end-date": end.isoformat(),
            "limit": 30, "offset": 0,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if r.status_code != 200:
        raise LiveFetchError(f"GFW HTTP {r.status_code}: {r.text[:140]}",
                             status=r.status_code)
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise LiveFetchError(f"GFW non-JSON body: {r.text[:120]}")
    entries = data.get("entries") or []
    total = data.get("total")
    rows_out: list[dict] = []
    hormuz_n = 0
    for e in entries:
        pos = e.get("position") or {}
        lat, lon = pos.get("lat"), pos.get("lon")
        in_hz = _in_bbox(lon, lat, HORMUZ_BBOX)
        if in_hz:
            hormuz_n += 1
        vessel = e.get("vessel") or {}
        rows_out.append({
            "ts": e.get("start"),
            "region": "strait_of_hormuz" if in_hz else "global",
            "severity": 0.2,
            "text": f"Port visit — {vessel.get('name') or '?'} "
                    f"(flag {vessel.get('flag') or '?'})",
            "url": SOURCE_META["gfw"]["raw_url"],
            "lat": lat, "lon": lon, "in_hormuz": in_hz,
        })
    return {
        "value": hormuz_n,
        "count": len(entries),
        "headline": (f"{len(entries)} recent port-visit events "
                     f"({hormuz_n} in Strait-of-Hormuz bbox); GFW reports "
                     f"{total} total available"),
        "raw_url": SOURCE_META["gfw"]["raw_url"],
        "rows": rows_out[:8],
    }


def _client_newsapi() -> dict[str, Any]:
    key = _key("NEWS_API_KEY")
    since = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
    r = _get(
        "https://newsapi.org/v2/everything",
        params={
            "q": "Strait of Hormuz OR Red Sea shipping OR Suez Canal OR "
                 "port strike OR supply chain disruption",
            "from": since, "language": "en", "sortBy": "publishedAt",
            "pageSize": 20, "apiKey": key,
        },
    )
    if r.status_code != 200:
        # NewsAPI returns a JSON error body with a machine code.
        try:
            code = r.json().get("code")
        except Exception:  # noqa: BLE001
            code = None
        raise LiveFetchError(f"NewsAPI HTTP {r.status_code}"
                             f"{f' ({code})' if code else ''}: {r.text[:120]}",
                             status=r.status_code)
    body = r.json()
    if body.get("status") != "ok":
        raise LiveFetchError(f"NewsAPI error: {body.get('code')} {body.get('message', '')[:120]}")
    arts = body.get("articles", []) or []
    rows_out = []
    for a in arts[:8]:
        title = a.get("title") or ""
        rows_out.append({
            "ts": (a.get("publishedAt") or "").replace("+00:00", "Z"),
            "region": "global", "severity": None, "text": title,
            "url": a.get("url") or SOURCE_META["newsapi"]["raw_url"],
            "outlet": (a.get("source") or {}).get("name"),
        })
    return {
        "value": body.get("totalResults", len(arts)),
        "count": len(arts),
        "headline": (f"{body.get('totalResults', len(arts))} supply-chain headlines "
                     f"in last 48h" + (f" — top: {arts[0].get('title')}" if arts else "")),
        "raw_url": SOURCE_META["newsapi"]["raw_url"],
        "rows": rows_out,
    }


# NOAA weather uses the realtime NWS API (keyless). The NCEI CDO/GHCND API that
# the NOAA_TOKEN unlocks was evaluated for this slot but is unusable as a live
# signal (~128 s latency; GHCND daily data lags several days → empty for recent
# windows). NWS gives real, sub-2 s, realtime NOAA weather for US port states.
_NWS_UA = ("SupplyMind/1.0 (supply-chain war room; "
           "+https://github.com/ShAuRyA-Noodle/Sleep-Token)")
_ALERT_SEVERITY = {"Extreme": 1.0, "Severe": 0.75, "Moderate": 0.45,
                   "Minor": 0.2, "Unknown": 0.1}


def _client_noaa() -> dict[str, Any]:
    r = _get(
        "https://api.weather.gov/alerts/active",
        params={"area": "TX,LA,GA,NJ,CA", "status": "actual"},
        headers={"User-Agent": _NWS_UA, "Accept": "application/geo+json"},
        timeout=20.0,
    )
    if r.status_code != 200:
        raise LiveFetchError(f"NWS HTTP {r.status_code}: {r.text[:140]}",
                             status=r.status_code)
    try:
        feats = r.json().get("features", []) or []
    except json.JSONDecodeError:
        raise LiveFetchError(f"NWS non-JSON body: {r.text[:120]}")
    rows_out = []
    for f in feats:
        p = f.get("properties", {}) or {}
        sev = _ALERT_SEVERITY.get(p.get("severity"), 0.1)
        rows_out.append({
            "ts": p.get("effective") or p.get("sent"),
            "region": "us_ports", "severity": sev,
            "text": f"{p.get('event')} ({p.get('severity')}) — "
                    f"{(p.get('areaDesc') or '')[:80]}",
            "url": p.get("id") or SOURCE_META["noaa"]["raw_url"],
            "event": p.get("event"), "alert_severity": p.get("severity"),
        })
    rows_out.sort(key=lambda x: x.get("severity") or 0, reverse=True)
    top = rows_out[0] if rows_out else None
    return {
        "value": len(feats),
        "count": len(feats),
        "headline": (f"{len(feats)} active NWS weather alert(s) over US port states"
                     + (f"; top: {top['event']} ({top['alert_severity']})" if top else "")),
        "raw_url": SOURCE_META["noaa"]["raw_url"],
        "rows": rows_out[:8],
    }


def _client_usgs() -> dict[str, Any]:
    # fdsnws query API (the static summary/*.geojson CDN feed is unreachable from
    # some networks — connection reset — while this endpoint is stable).
    start = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    r = _get(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "minmagnitude": 4.5, "starttime": start,
                "orderby": "time", "limit": 200},
    )
    if r.status_code != 200:
        raise LiveFetchError(f"USGS HTTP {r.status_code}", status=r.status_code)
    feats = r.json().get("features", []) or []
    rows_out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        coords = (f.get("geometry", {}) or {}).get("coordinates") or [None, None]
        mag = props.get("mag")
        sev = max(0.0, min(1.0, ((mag or 0) - 4.0) / 4.0))
        t = props.get("time")
        ts = (datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
              if t else None)
        rows_out.append({
            "ts": ts, "region": "global", "severity": round(sev, 3),
            "text": f"M{mag} earthquake — {props.get('place')}",
            "url": props.get("url") or SOURCE_META["usgs"]["raw_url"],
            "lat": coords[1], "lon": coords[0], "mag": mag,
        })
    top = max((r for r in rows_out), key=lambda x: x.get("mag") or 0, default=None)
    return {
        "value": len(feats),
        "count": len(feats),
        "headline": (f"{len(feats)} M4.5+ earthquake(s) in last 24h"
                     + (f"; largest M{top['mag']} — {top['text'].split('— ',1)[-1]}"
                        if top else "")),
        "raw_url": SOURCE_META["usgs"]["raw_url"],
        "rows": rows_out[:8],
    }


def _client_gdelt() -> dict[str, Any]:
    r = _get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": 'sourcelang:eng ("strait of hormuz" OR "red sea shipping" '
                     'OR "suez canal" OR "supply chain")',
            "mode": "ArtList", "format": "json", "maxrecords": 30, "timespan": "1d",
        },
        headers={"User-Agent": "SupplyMind/1.0 (+https://github.com/ShAuRyA-Noodle/Sleep-Token)"},
    )
    if r.status_code != 200:
        raise LiveFetchError(f"GDELT HTTP {r.status_code}", status=r.status_code)
    try:
        data = r.json()
    except json.JSONDecodeError:
        # GDELT serves an HTML notice when rate-limited.
        raise LiveFetchError(f"GDELT non-JSON (rate-limited?): {r.text[:100]}")
    arts = data.get("articles", []) or []
    rows_out = []
    for a in arts[:8]:
        rows_out.append({
            "ts": None, "region": "global", "severity": None,
            "text": a.get("title") or "",
            "url": a.get("url") or SOURCE_META["gdelt"]["raw_url"],
            "domain": a.get("domain"), "country": a.get("sourcecountry"),
        })
    return {
        "value": len(arts),
        "count": len(arts),
        "headline": (f"{len(arts)} chokepoint news article(s) in last 24h"
                     + (f" — top: {arts[0].get('title')}" if arts else "")),
        "raw_url": SOURCE_META["gdelt"]["raw_url"],
        "rows": rows_out,
    }


CLIENTS: dict[str, Callable[[], dict[str, Any]]] = {
    "fred": _client_fred,
    "eia": _client_eia,
    "nasa_firms": _client_nasa_firms,
    "gfw": _client_gfw,
    "newsapi": _client_newsapi,
    "noaa": _client_noaa,
    "usgs": _client_usgs,
    "gdelt": _client_gdelt,
}


# --------------------------------------------------------------------------- #
# Status contract
# --------------------------------------------------------------------------- #

@dataclass
class SourceStatus:
    """Uniform per-source result. Serialise with :meth:`to_dict`."""
    source: str
    ok: bool
    degraded: bool
    degraded_reason: str | None
    provider: str
    raw_url: str | None
    fetched_at: str | None
    age_seconds: float | None
    count: int
    value: Any
    headline: str
    cache_hit: bool
    replay: bool
    latency_ms: float | None
    rows: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _read_cache(name: str) -> dict | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(name: str, record: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(name).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:  # noqa: BLE001 — logged, not swallowed silently
        logger.warning("[live] cache write failed for %s: %s", name, e)


def _age_of(fetched_unix: float | None) -> float | None:
    if fetched_unix is None:
        return None
    return round(max(0.0, time.time() - fetched_unix), 1)


def _status_from_cache(name: str, cache: dict, *, replay: bool,
                       degraded: bool, degraded_reason: str | None) -> SourceStatus:
    payload = cache.get("payload") or {}
    return SourceStatus(
        source=name, ok=not degraded, degraded=degraded,
        degraded_reason=degraded_reason,
        provider=source_meta(name).get("provider", name),
        raw_url=payload.get("raw_url") or source_meta(name).get("raw_url"),
        fetched_at=cache.get("fetched_at"),
        age_seconds=_age_of(cache.get("fetched_at_unix")),
        count=int(payload.get("count", 0) or 0),
        value=payload.get("value"),
        headline=payload.get("headline", ""),
        cache_hit=True, replay=replay,
        latency_ms=None,
        rows=payload.get("rows", []) or [],
        provenance=cache.get("provenance") or stamp(name, payload, cache.get("fetched_at")),
    )


def fetch_source(name: str, *, ttl: int = DEFAULT_TTL_S,
                 force_live: bool = False,
                 force_replay: bool | None = None) -> SourceStatus:
    """Fetch one source with cache + freshness + loud-fail semantics.

    * ``force_replay`` (or ``FORCE_REPLAY=1``): serve only the cached snapshot,
      ``replay=True``. No network call. Degraded only if no snapshot exists.
    * ``force_live``: ignore cache freshness and make a real call now.
    * otherwise: serve a cached snapshot younger than ``ttl``; else fetch live.

    On a live failure with a stale snapshot on disk, the stale snapshot is
    served with ``degraded=True`` and the real error reason — never a fake value.
    """
    if name not in CLIENTS:
        raise ValueError(f"unknown source '{name}' (known: {', '.join(SOURCES)})")
    if force_replay is None:
        force_replay = os.environ.get("FORCE_REPLAY") == "1"

    meta = source_meta(name)

    # ---- replay mode: cache only, clearly labelled -------------------------
    if force_replay:
        cache = _read_cache(name)
        if cache is None:
            return SourceStatus(
                source=name, ok=False, degraded=True,
                degraded_reason="FORCE_REPLAY=1 but no cached snapshot on disk",
                provider=meta.get("provider", name), raw_url=meta.get("raw_url"),
                fetched_at=None, age_seconds=None, count=0, value=None,
                headline="replay mode — no cached snapshot", cache_hit=False,
                replay=True, latency_ms=None, rows=[],
                provenance=stamp(name, {}, None))
        cached_degraded = bool(cache.get("degraded"))
        reason = ("replay mode (FORCE_REPLAY=1) — cached snapshot"
                  if not cached_degraded
                  else f"replay of a degraded snapshot: {cache.get('degraded_reason')}")
        return _status_from_cache(name, cache, replay=True,
                                  degraded=cached_degraded, degraded_reason=reason)

    # ---- cache-fresh short-circuit -----------------------------------------
    # A fresh success snapshot (< ttl) is served directly; a fresh degraded
    # snapshot (< FAIL_TTL) is also served so a slow/down source is not re-hit
    # on every request — still labelled degraded with the real reason.
    if not force_live:
        cache = _read_cache(name)
        if cache is not None:
            age = _age_of(cache.get("fetched_at_unix"))
            if age is not None:
                if not cache.get("degraded") and age < ttl:
                    return _status_from_cache(name, cache, replay=False,
                                              degraded=False, degraded_reason=None)
                if cache.get("degraded") and age < FAIL_TTL_S:
                    return _status_from_cache(
                        name, cache, replay=False, degraded=True,
                        degraded_reason=cache.get("degraded_reason"))

    # ---- live call ----------------------------------------------------------
    t0 = time.perf_counter()
    try:
        payload = CLIENTS[name]()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except LiveFetchError as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        stale = _read_cache(name)
        if stale is not None and not stale.get("degraded"):
            # A prior *good* snapshot exists — serve it, clearly stale+degraded.
            reason = (f"live fetch failed ({e.reason}); serving cached snapshot "
                      f"{_age_of(stale.get('fetched_at_unix'))}s old")
            st = _status_from_cache(name, stale, replay=False,
                                    degraded=True, degraded_reason=reason)
            st.latency_ms = latency_ms
            st.cache_hit = True
            return st
        # No good snapshot: record the failure so we don't re-hit a down/slow
        # source on every request (honoured by the FAIL_TTL short-circuit).
        record = {
            "fetched_at": _now_iso(), "fetched_at_unix": time.time(),
            "degraded": True, "degraded_reason": e.reason,
            "payload": {"raw_url": meta.get("raw_url"), "count": 0,
                        "value": None,
                        "headline": f"{name} unavailable: {e.reason}", "rows": []},
            "provenance": stamp(name, {}, None),
        }
        _write_cache(name, record)
        return SourceStatus(
            source=name, ok=False, degraded=True, degraded_reason=e.reason,
            provider=meta.get("provider", name), raw_url=meta.get("raw_url"),
            fetched_at=record["fetched_at"], age_seconds=0.0, count=0, value=None,
            headline=f"{name} unavailable: {e.reason}", cache_hit=False,
            replay=False, latency_ms=latency_ms, rows=[],
            provenance=record["provenance"])

    fetched_at = _now_iso()
    prov = stamp(name, payload, fetched_at)
    record = {
        "fetched_at": fetched_at, "fetched_at_unix": time.time(),
        "degraded": False, "degraded_reason": None,
        "payload": payload, "provenance": prov,
    }
    _write_cache(name, record)
    return SourceStatus(
        source=name, ok=True, degraded=False, degraded_reason=None,
        provider=meta.get("provider", name),
        raw_url=payload.get("raw_url") or meta.get("raw_url"),
        fetched_at=fetched_at, age_seconds=0.0,
        count=int(payload.get("count", 0) or 0), value=payload.get("value"),
        headline=payload.get("headline", ""), cache_hit=False, replay=False,
        latency_ms=latency_ms, rows=payload.get("rows", []) or [], provenance=prov)


def fetch_all_sources(*, ttl: int = DEFAULT_TTL_S, force_live: bool = False,
                      force_replay: bool | None = None,
                      max_workers: int = 8) -> dict:
    """Poll all eight sources concurrently. Returns statuses + a summary.

    One call per source (never hammered). Failures are isolated — one source
    down never blocks the others.
    """
    _load_env()
    if force_replay is None:
        force_replay = os.environ.get("FORCE_REPLAY") == "1"

    statuses: dict[str, SourceStatus] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(fetch_source, name, ttl=ttl, force_live=force_live,
                        force_replay=force_replay): name
            for name in SOURCES
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                statuses[name] = fut.result()
            except Exception as e:  # noqa: BLE001 — never let one source 500 the fan-out
                logger.error("[live] %s raised: %s", name, e)
                statuses[name] = SourceStatus(
                    source=name, ok=False, degraded=True,
                    degraded_reason=f"unexpected error: {type(e).__name__}: {str(e)[:140]}",
                    provider=source_meta(name).get("provider", name),
                    raw_url=source_meta(name).get("raw_url"),
                    fetched_at=None, age_seconds=None, count=0, value=None,
                    headline=f"{name} errored", cache_hit=False, replay=False,
                    latency_ms=None, rows=[], provenance=stamp(name, {}, None))

    ordered = [statuses[n].to_dict() for n in SOURCES if n in statuses]
    n_ok = sum(1 for s in ordered if s["ok"])
    n_degraded = sum(1 for s in ordered if s["degraded"])
    ages = [s["age_seconds"] for s in ordered if s["age_seconds"] is not None]
    return {
        "generated_at": _now_iso(),
        "replay_mode": bool(force_replay),
        "summary": {
            "n_sources": len(ordered),
            "n_ok": n_ok,
            "n_degraded": n_degraded,
            "n_live_now": sum(1 for s in ordered
                              if not s["cache_hit"] and not s["degraded"]),
            "max_age_seconds": max(ages) if ages else None,
            "freshest_age_seconds": min(ages) if ages else None,
        },
        "sources": ordered,
    }


def signal_counts(*, ttl: int = DEFAULT_TTL_S, force_live: bool = False,
                  force_replay: bool | None = None) -> dict:
    """Compact per-source freshness table for the war-room strip.

    One row per source: count, value, age, degraded flag + reason. This is the
    endpoint the UI polls to render "< N h old" stamps and the REPLAY banner.
    """
    full = fetch_all_sources(ttl=ttl, force_live=force_live, force_replay=force_replay)
    counts = {
        s["source"]: {
            "count": s["count"], "value": s["value"], "headline": s["headline"],
            "fetched_at": s["fetched_at"], "age_seconds": s["age_seconds"],
            "degraded": s["degraded"], "degraded_reason": s["degraded_reason"],
            "cache_hit": s["cache_hit"], "replay": s["replay"],
            "provider": s["provider"], "raw_url": s["raw_url"],
        }
        for s in full["sources"]
    }
    return {
        "generated_at": full["generated_at"],
        "replay_mode": full["replay_mode"],
        "summary": full["summary"],
        "counts": counts,
    }


def recent_events(*, region: str | None = None, limit: int = 20,
                  ttl: int = DEFAULT_TTL_S, force_live: bool = False,
                  force_replay: bool | None = None) -> dict:
    """Flatten each source's sample rows into a single recent-events feed.

    Optional ``region`` filters to rows whose region matches (substring, e.g.
    ``hormuz``). Degraded sources contribute nothing but are listed so the caller
    knows what is missing.
    """
    full = fetch_all_sources(ttl=ttl, force_live=force_live, force_replay=force_replay)
    events: list[dict] = []
    degraded_sources: list[dict] = []
    for s in full["sources"]:
        if s["degraded"]:
            degraded_sources.append(
                {"source": s["source"], "reason": s["degraded_reason"]})
        for row in s["rows"]:
            reg = str(row.get("region") or "")
            if region and region.lower() not in reg.lower():
                continue
            events.append({
                "source": s["source"], "provider": s["provider"],
                "fetched_at": s["fetched_at"], "age_seconds": s["age_seconds"],
                **row,
            })
    events.sort(key=lambda e: e.get("ts") or "", reverse=True)
    return {
        "generated_at": full["generated_at"],
        "replay_mode": full["replay_mode"],
        "region_filter": region,
        "count": len(events[:limit]),
        "degraded_sources": degraded_sources,
        "events": events[:limit],
    }


# Disaster-type sources contribute the "recent-disaster" pick.
_DISASTER_SOURCES = ("usgs", "nasa_firms", "gdelt", "newsapi")


def recent_disaster(*, region: str | None = None, ttl: int = DEFAULT_TTL_S,
                    force_live: bool = False,
                    force_replay: bool | None = None) -> dict:
    """Pick the single highest-severity live disaster/hazard event for a region.

    Draws from the hazard/news sources (USGS quakes, NASA fires, GDELT/NewsAPI
    headlines). If every hazard source is degraded/empty, returns
    ``degraded=True`` with the reasons — never a fabricated event.
    """
    full = fetch_all_sources(ttl=ttl, force_live=force_live, force_replay=force_replay)
    by_source = {s["source"]: s for s in full["sources"]}
    candidates: list[dict] = []
    degraded_sources: list[dict] = []
    for name in _DISASTER_SOURCES:
        s = by_source.get(name)
        if s is None:
            continue
        if s["degraded"]:
            degraded_sources.append({"source": name, "reason": s["degraded_reason"]})
            continue
        for row in s["rows"]:
            reg = str(row.get("region") or "")
            if region and region.lower() not in reg.lower():
                continue
            candidates.append({"source": name, "provider": s["provider"],
                               "fetched_at": s["fetched_at"],
                               "age_seconds": s["age_seconds"], **row})

    if not candidates:
        return {
            "generated_at": full["generated_at"],
            "replay_mode": full["replay_mode"],
            "region_filter": region,
            "degraded": True,
            "degraded_reason": ("no live disaster/hazard events available"
                                + (f" for region '{region}'" if region else "")),
            "degraded_sources": degraded_sources,
            "top_event": None,
            "n_candidates": 0,
        }

    top = max(candidates, key=lambda c: (c.get("severity") or 0.0))
    return {
        "generated_at": full["generated_at"],
        "replay_mode": full["replay_mode"],
        "region_filter": region,
        "degraded": False,
        "degraded_reason": None,
        "degraded_sources": degraded_sources,
        "n_candidates": len(candidates),
        "top_event": top,
    }


# --------------------------------------------------------------------------- #
# CLI / receipt generation
# --------------------------------------------------------------------------- #

def _print_table(full: dict) -> None:
    print(f"\nlive-signal status @ {full['generated_at']}"
          f"  (replay_mode={full['replay_mode']})")
    print("-" * 96)
    print(f"{'source':<12}{'ok':<4}{'age(s)':>9}  {'count':>6}  {'lat(ms)':>8}  headline")
    print("-" * 96)
    for s in full["sources"]:
        mark = "UP " if s["ok"] else "DOWN"
        age = "-" if s["age_seconds"] is None else f"{s['age_seconds']:.0f}"
        lat = "-" if s["latency_ms"] is None else f"{s['latency_ms']:.0f}"
        head = (s["headline"] or "")[:52]
        print(f"{s['source']:<12}{mark:<4}{age:>9}  {s['count']:>6}  {lat:>8}  {head}")
        if s["degraded"]:
            print(f"{'':<12}     -> degraded: {s['degraded_reason']}")
    sm = full["summary"]
    print("-" * 96)
    print(f"summary: {sm['n_ok']}/{sm['n_sources']} up, "
          f"{sm['n_degraded']} degraded, {sm['n_live_now']} fetched live now\n")


def _write_receipt(full: dict, path: Path) -> None:
    import subprocess
    from .provenance import sha256_of
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    per_source = [{
        "source": s["source"], "ok": s["ok"], "degraded": s["degraded"],
        "degraded_reason": s["degraded_reason"], "provider": s["provider"],
        "fetched_at": s["fetched_at"], "age_seconds": s["age_seconds"],
        "count": s["count"], "value": s["value"], "latency_ms": s["latency_ms"],
        "cache_hit": s["cache_hit"], "replay": s["replay"],
        "raw_url": s["raw_url"], "headline": s["headline"],
    } for s in full["sources"]]
    receipt = {
        "receipt": "live_data_freshness_REAL",
        "what": "One real live call to each of the eight SupplyMind live-signal "
                "sources; per-source up/down, timestamp, age, and count recorded. "
                "Degraded sources are reported honestly (no synthetic values).",
        "command": "python -m supplymind.data.live --live "
                   "--receipt tests/receipts/live_data_freshness_REAL.json",
        "git_sha": sha,
        "generated_at": full["generated_at"],
        "summary": full["summary"],
        "per_source": per_source,
        "match": True,
        "match_note": "match:true means the receipt reflects a genuine executed "
                      "fan-out; per-source ok/degraded is the real API outcome at "
                      "run time and may vary on re-run (free-tier availability).",
        "payload_sha256": sha256_of(per_source),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"receipt written: {path}")


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="SupplyMind live-signal status.")
    ap.add_argument("--live", action="store_true",
                    help="Force one live call per source (ignore cache freshness).")
    ap.add_argument("--replay", action="store_true",
                    help="Replay cached snapshots only (sets FORCE_REPLAY behaviour).")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")
    ap.add_argument("--receipt", type=str, default=None,
                    help="Also write a real freshness receipt to this path.")
    args = ap.parse_args()

    full = fetch_all_sources(force_live=args.live,
                             force_replay=True if args.replay else None)
    if args.json:
        print(json.dumps(full, indent=2, ensure_ascii=False))
    else:
        _print_table(full)
    if args.receipt:
        _write_receipt(full, Path(args.receipt))


if __name__ == "__main__":
    main()
