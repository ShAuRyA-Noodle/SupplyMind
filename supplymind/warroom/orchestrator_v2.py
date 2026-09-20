"""orchestrator_v2.py — concurrent, cache-first multi-source live fan-out.

Backs ``POST /live/intel-fan-out`` (and its SSE sibling). Fans out across the
v1 baseline (5 sources) + v2 expansion fleet (15 sources) with three properties
that make the golden-path demo feel instant AND stay honest:

1. **Cache-first.** Every source's last-good payload is written to
   ``.source_cache/_fanout/<label>.json`` with its fetch timestamp. A request
   serves any snapshot younger than the success TTL (10 min) or fail TTL (2 min)
   directly from disk — so a warm request returns in well under a second and
   never re-hammers a free tier. TTLs are reused verbatim from
   :mod:`supplymind.data.live` (the WP6.4 live-signal layer).

2. **Non-blocking, bounded fan-out.** Stale/missing sources are refreshed on a
   *persistent* thread pool with a hard wall-clock deadline. A source that blows
   the budget is **abandoned, not awaited** — its worker keeps running in the
   background to warm the cache for the next request, while this response returns
   the last-good snapshot (stale, clearly degraded with age) or an honest
   "refreshing in background" marker. This is the fix for the 99 s demo-killer:
   the old code wrapped the pool in ``with ThreadPoolExecutor(...)`` whose
   ``shutdown(wait=True)`` on exit blocked on the slowest source no matter what
   timeout was requested.

3. **Truthful per-source freshness.** Every source reports ``age_seconds``,
   ``cache_hit``, ``degraded`` + a real reason, and ``latency_ms``. No source is
   ever faked or silently backfilled — an empty or down source is surfaced as
   such (CLAUDE.md §0, §3 fail-loud).

Public API (backward compatible):
    fan_out_all(*, timeout_s=..., parallel=...) -> {"summary": {...}, "events": [...]}
    fan_out_stream(...) -> async generator of real per-source stage events (SSE)
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

from .sources_v2 import (
    cisa_kev, eia, gdelt_conflict, gdelt_humanitarian, gfw,
    hackernews, nasa_eonet, nasa_firms, noaa_ndbc, noaa_tides,
    ofac_sdn, sec_edgar_8k, who_don, wiki_pageviews, worldbank,
)

# Existing v1 sources (NewsAPI, GDELT, USGS, FRED Brent) wired in via
# server.app /live/recent-events already; we re-aggregate them here for a
# unified view. Adapter converts the v1 Event dataclass -> the v2 event dict.
from .sources import fred_brent, gdelt as gdelt_v1, newsapi, usgs

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FANOUT_CACHE_DIR = REPO_ROOT / ".source_cache" / "_fanout"

# Reuse the live-signal layer's TTL discipline verbatim (CLAUDE.md WP6.4):
# a fresh success snapshot is served for 10 min; a down source is re-probed at
# most once every 2 min. Falls back to the same literals if that module moves.
try:  # pragma: no cover - import shim
    from supplymind.data.live import DEFAULT_TTL_S as _TTL, FAIL_TTL_S as _FAIL_TTL
except Exception:  # noqa: BLE001
    _TTL, _FAIL_TTL = 600, 120

DEFAULT_TTL_S = _TTL              # 10-min success TTL (shared with live.py)
FAIL_TTL_S = _FAIL_TTL            # 2-min fail TTL (shared with live.py)
DEFAULT_DEADLINE_S = 12.0         # cold-path wall-clock budget (< 15 s target)
_MAX_WORKERS = 24                 # >= fleet size so no source waits in a queue


# --------------------------------------------------------------------------- #
# v1 adapter
# --------------------------------------------------------------------------- #
def _v1_event_to_dict(ev) -> dict:
    """Adapter: v1 Event dataclass -> v2 standard event dict."""
    return {
        "source":          getattr(ev, "source", "?"),
        "event_id":        f"{getattr(ev, 'source', '?')}_{getattr(ev, 'text_hash', '?')[:24]}",
        "title":           (getattr(ev, "raw_text", "") or "")[:160],
        "description":     (getattr(ev, "raw_text", "") or "")[:1500],
        "occurred_at_utc": getattr(ev, "ts_iso", None),
        "lat":             (getattr(ev, "meta", {}) or {}).get("lat"),
        "lon":             (getattr(ev, "meta", {}) or {}).get("lon"),
        "severity_proxy":  float(getattr(ev, "severity", 0.0)),
        "raw_url":         (getattr(ev, "urls", []) or ["?"])[0],
        "fetched_at_utc":  None,
        "inference_type":  f"live_{getattr(ev, 'source', '?')}",
        "extra":           {"event_type": getattr(ev, "event_type", "?"),
                             "region": getattr(ev, "region", "?")},
    }


def _wrap_v1(fn: Callable, **kwargs) -> Callable[[], list[dict]]:
    """Wrap a v1 fetch() function so it returns the v2 dict schema."""
    def _inner():
        events = fn(**kwargs) or []
        return [_v1_event_to_dict(e) for e in events]
    return _inner


# Source spec: (label, callable, default_args, role)
SOURCE_FLEET: list[tuple[str, Callable, dict, str]] = [
    # --- v1 baseline (4) ---
    ("newsapi",            _wrap_v1(newsapi.fetch, lookback_minutes=2880),     {}, "news"),
    ("gdelt_v1",           _wrap_v1(gdelt_v1.fetch, lookback_minutes=2880),    {}, "geopol"),
    ("usgs_quakes",        _wrap_v1(usgs.fetch),                                {}, "natural"),
    ("fred_brent",         _wrap_v1(fred_brent.fetch),                          {}, "commodity"),
    # --- v2 expansion (15) ---
    ("who_don",            who_don.fetch_recent,                           {"limit": 20}, "health"),
    ("gdelt_conflict",     gdelt_conflict.fetch_conflict_events,           {"timespan": "7d"}, "conflict"),
    ("gdelt_humanitarian", gdelt_humanitarian.fetch_humanitarian_events,   {"timespan": "14d"}, "humanitarian"),
    ("noaa_ndbc",          noaa_ndbc.fetch_chokepoint_buoys,               {}, "ocean"),
    ("noaa_tides",         noaa_tides.fetch_chokepoint_ports,              {}, "port"),
    ("nasa_eonet",         nasa_eonet.fetch_open_events,                   {"days": 30, "limit": 30}, "natural"),
    ("eia_petroleum",      eia.fetch_petroleum_signals,                    {"limit": 5}, "commodity"),
    ("nasa_firms",         nasa_firms.fetch_active_fires,                  {"days_back": 2}, "fire"),
    ("gfw_port_visits",    gfw.fetch_recent_port_visits,                   {"days_back": 7, "limit_per_region": 3}, "vessel"),
    ("sec_edgar_8k",       sec_edgar_8k.fetch_supply_chain_filings,        {"days_back": 60, "limit": 10}, "corporate"),
    ("cisa_kev",           cisa_kev.fetch_recent,                          {"days_back": 60, "limit": 15}, "cyber"),
    ("hackernews",         hackernews.fetch_supply_chain_signal,           {"hours_back": 72, "limit": 15}, "social"),
    ("wiki_pageviews",     wiki_pageviews.fetch_pageview_pulses,           {"days_back": 7}, "attention"),
    ("worldbank",          worldbank.fetch_macro_signals,                  {}, "macro"),
    ("ofac_sdn",           ofac_sdn.fetch_recent_designations,             {"limit": 30}, "sanctions"),
]

ROLE_MAP: dict[str, str] = {label: role for label, _, _, role in SOURCE_FLEET}
_FN_MAP: dict[str, tuple[Callable, dict]] = {
    label: (fn, kwargs) for label, fn, kwargs, _ in SOURCE_FLEET
}


# --------------------------------------------------------------------------- #
# Persistent (never-joined) thread pool + in-flight de-dupe
# --------------------------------------------------------------------------- #
_EXECUTOR: ThreadPoolExecutor | None = None
_EXEC_LOCK = threading.Lock()
_INFLIGHT: dict[str, Future] = {}
_INFLIGHT_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    """Lazily create the shared pool. Never wrapped in a ``with`` block (which
    would join on exit and re-introduce the blocking bug); shut down best-effort
    at interpreter exit with ``cancel_futures`` so a slow source cannot hang the
    process."""
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXEC_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS, thread_name_prefix="fanout")
                atexit.register(_EXECUTOR.shutdown, wait=False, cancel_futures=True)
    return _EXECUTOR


def _submit_refresh(label: str) -> Future:
    """Submit (or reuse an in-flight) background refresh for one source."""
    with _INFLIGHT_LOCK:
        fut = _INFLIGHT.get(label)
        if fut is not None and not fut.done():
            return fut
        fut = _executor().submit(_refresh_source, label)
        _INFLIGHT[label] = fut
        return fut


# --------------------------------------------------------------------------- #
# Cache layer (same record shape as supplymind.data.live)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_path(label: str) -> Path:
    return FANOUT_CACHE_DIR / f"{label}.json"


def _read_cache(label: str) -> dict | None:
    p = _cache_path(label)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(label: str, record: dict) -> None:
    """Atomic write (temp + os.replace) so a concurrent reader never sees a
    half-written file while a background worker is refreshing."""
    try:
        FANOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _cache_path(label).with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _cache_path(label))
    except OSError as e:  # noqa: BLE001 — logged, not silently swallowed
        logger.warning("[fanout] cache write failed for %s: %s", label, e)


def _age_of(fetched_unix: float | None) -> float | None:
    if fetched_unix is None:
        return None
    return round(max(0.0, time.time() - fetched_unix), 1)


def _refresh_source(label: str) -> dict:
    """Fetch one source live, write its cache, and return the record.

    Runs on a background worker. Never raises: a fetch failure is captured as a
    ``degraded`` record with the real reason. A prior GOOD snapshot is never
    clobbered by a failure — it is retained so the caller can serve it (stale)
    while the source recovers.
    """
    fn, kwargs = _FN_MAP[label]
    t0 = time.time()
    try:
        out = fn(**kwargs)
        events = out if isinstance(out, list) else []
        rec = {
            "fetched_at": _now_iso(), "fetched_at_unix": time.time(),
            "degraded": False, "degraded_reason": None,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "events": events,
        }
        _write_cache(label, rec)
        logger.info("[fanout:%s] %d events in %.1fs (cache warmed)",
                    label, len(events), time.time() - t0)
        return rec
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {str(e)[:160]}"
        latency = round((time.time() - t0) * 1000, 1)
        prior = _read_cache(label)
        if prior is not None and not prior.get("degraded") and prior.get("events"):
            logger.warning("[fanout:%s] refresh failed in %.1fs (%s); prior good "
                           "cache retained", label, time.time() - t0, reason)
            return {**prior, "latency_ms": latency, "refresh_error": reason}
        rec = {
            "fetched_at": _now_iso(), "fetched_at_unix": time.time(),
            "degraded": True, "degraded_reason": reason,
            "latency_ms": latency, "events": [],
        }
        _write_cache(label, rec)
        logger.warning("[fanout:%s] refresh failed in %.1fs: %s",
                       label, time.time() - t0, reason)
        return rec


# --------------------------------------------------------------------------- #
# Per-source status assembly
# --------------------------------------------------------------------------- #
def _tag_events(events: list[dict], role: str) -> list[dict]:
    out = []
    for ev in events or []:
        ev = dict(ev)
        ev.setdefault("role_tag", role)
        out.append(ev)
    return out


def _status(label: str, rec: dict, *, cache_hit: bool,
            from_stale: bool = False, stale_reason: str | None = None,
            background_refreshing: bool = False) -> dict:
    role = ROLE_MAP.get(label, "unknown")
    degraded = bool(rec.get("degraded")) or from_stale
    reason = stale_reason if from_stale else rec.get("degraded_reason")
    events = _tag_events(rec.get("events", []), role)
    return {
        "label": label,
        "role": role,
        "events": events,
        "n_events": len(events),
        "age_seconds": _age_of(rec.get("fetched_at_unix")),
        "fetched_at": rec.get("fetched_at"),
        "latency_ms": rec.get("latency_ms"),
        "degraded": degraded,
        "degraded_reason": reason,
        "cache_hit": cache_hit,
        "background_refreshing": background_refreshing,
    }


def _serve_from_cache(label: str, cache: dict, *, ttl: int, fail_ttl: int) -> dict | None:
    """Return a status if a fresh cache snapshot can be served directly, else
    None (source needs a refresh)."""
    age = _age_of(cache.get("fetched_at_unix"))
    if age is None:
        return None
    if not cache.get("degraded") and age < ttl:
        return _status(label, cache, cache_hit=True)
    if cache.get("degraded") and age < fail_ttl:
        return _status(label, cache, cache_hit=True)
    return None


def _abandon_status(label: str, prior: dict | None, deadline: float) -> dict:
    """Build the status for a source still refreshing past the deadline.

    Serve the last GOOD snapshot (stale, clearly degraded with age) when one
    exists; otherwise emit an honest "refreshing in background" marker and drop a
    transient degraded record so repeated cold requests short-circuit instead of
    each paying the full deadline. The background worker overwrites this record
    the moment it finishes (success or real failure)."""
    if prior is not None and prior.get("events") and not prior.get("degraded"):
        age = _age_of(prior.get("fetched_at_unix"))
        reason = (f"refresh exceeded {deadline:.0f}s budget; serving "
                  f"{age}s-old cached snapshot, refreshing in background")
        return _status(label, prior, cache_hit=True, from_stale=True,
                       stale_reason=reason, background_refreshing=True)
    reason = (f"source exceeded {deadline:.0f}s budget; no cached snapshot yet, "
              f"refreshing in background")
    marker = {
        "fetched_at": _now_iso(), "fetched_at_unix": time.time(),
        "degraded": True, "degraded_reason": reason, "latency_ms": None,
        "events": [],
    }
    _write_cache(label, marker)
    return _status(label, marker, cache_hit=False, background_refreshing=True)


# --------------------------------------------------------------------------- #
# Summary + response assembly
# --------------------------------------------------------------------------- #
def _assemble(statuses: dict[str, dict], *, started: float, deadline: float) -> dict:
    all_events: list[dict] = []
    for label, _, _, _ in SOURCE_FLEET:
        st = statuses.get(label)
        if st:
            all_events.extend(st["events"])

    n_per_source = {label: statuses[label]["n_events"]
                    for label in statuses}
    errors = {label: st["degraded_reason"]
              for label, st in statuses.items()
              if st["degraded"] and st["degraded_reason"]}
    freshness = {
        label: {
            "n_events": st["n_events"],
            "age_seconds": st["age_seconds"],
            "fetched_at": st["fetched_at"],
            "latency_ms": st["latency_ms"],
            "degraded": st["degraded"],
            "degraded_reason": st["degraded_reason"],
            "cache_hit": st["cache_hit"],
            "background_refreshing": st["background_refreshing"],
            "role": st["role"],
        }
        for label, st in statuses.items()
    }
    summary = {
        "n_sources_total": len(SOURCE_FLEET),
        "n_sources_with_data": sum(1 for st in statuses.values() if st["n_events"] > 0),
        "n_sources_errored": len(errors),
        "n_events_total": len(all_events),
        "n_events_per_source": n_per_source,
        "errors_per_source": errors,
        "elapsed_s": round(time.time() - started, 2),
        "deadline_s": deadline,
        "fan_out_concurrency": _MAX_WORKERS,
        "n_cache_hits": sum(1 for st in statuses.values() if st["cache_hit"]),
        "n_refreshed_live": sum(1 for st in statuses.values()
                                if not st["cache_hit"] and not st["background_refreshing"]),
        "n_background_refreshing": sum(1 for st in statuses.values()
                                       if st["background_refreshing"]),
        "per_source_freshness": freshness,
        "inference_type": "live_multi_source_fan_out",
    }
    return {"summary": summary, "events": all_events}


# --------------------------------------------------------------------------- #
# Public: synchronous fan-out (backward compatible)
# --------------------------------------------------------------------------- #
def fan_out_all(
    *,
    timeout_s: float = DEFAULT_DEADLINE_S,
    ttl: int = DEFAULT_TTL_S,
    fail_ttl: int = FAIL_TTL_S,
    force_live: bool = False,
    parallel: int | None = None,  # accepted for backward compat; pool is fixed-size
) -> dict:
    """Fan out across all sources; return ``{"summary": {...}, "events": [...]}``.

    Warm (all snapshots fresh): returns in well under a second (disk only).
    Cold: bounded by ``timeout_s`` — slow sources are abandoned to the background
    and reported as degraded/refreshing, never blocking the response.
    """
    started = time.time()
    deadline = max(0.5, float(timeout_s))
    statuses: dict[str, dict] = {}
    pending: dict[str, tuple[Future, dict | None]] = {}

    for label, _fn, _kw, _role in SOURCE_FLEET:
        cache = None if force_live else _read_cache(label)
        if cache is not None:
            served = _serve_from_cache(label, cache, ttl=ttl, fail_ttl=fail_ttl)
            if served is not None:
                statuses[label] = served
                continue
        pending[label] = (_submit_refresh(label), _read_cache(label))

    if pending:
        fut_to_label = {fut: label for label, (fut, _c) in pending.items()}
        try:
            for fut in as_completed(list(fut_to_label), timeout=deadline):
                label = fut_to_label[fut]
                statuses[label] = _status(label, fut.result(), cache_hit=False)
        except FuturesTimeoutError:
            pass
        # Abandon stragglers — never join (that was the 99 s bug).
        for label, (fut, prior) in pending.items():
            if label in statuses:
                continue
            if fut.done():
                statuses[label] = _status(label, fut.result(), cache_hit=False)
            else:
                statuses[label] = _abandon_status(label, prior, deadline)

    return _assemble(statuses, started=started, deadline=deadline)


# --------------------------------------------------------------------------- #
# Public: async SSE stream of REAL per-source stage events
# --------------------------------------------------------------------------- #
async def fan_out_stream(
    *,
    timeout_s: float = DEFAULT_DEADLINE_S,
    ttl: int = DEFAULT_TTL_S,
    fail_ttl: int = FAIL_TTL_S,
    force_live: bool = False,
) -> AsyncIterator[dict]:
    """Yield one event per source as it genuinely lands, then a final summary.

    These are REAL stage events (no sleep-choreography, CLAUDE.md §0): a cache
    hit is yielded immediately; a live refresh is yielded the moment its worker
    completes; sources still running at the deadline are yielded as
    ``background_refreshing`` so the war room can fill progressively and honestly.
    """
    started = time.time()
    deadline = max(0.5, float(timeout_s))
    statuses: dict[str, dict] = {}
    n_total = len(SOURCE_FLEET)
    yield {"event": "start", "n_sources": n_total, "ts": _now_iso(),
           "deadline_s": deadline}

    aw_to_label: dict[asyncio.Future, tuple[str, dict | None]] = {}
    for label, _fn, _kw, _role in SOURCE_FLEET:
        cache = None if force_live else _read_cache(label)
        if cache is not None:
            served = _serve_from_cache(label, cache, ttl=ttl, fail_ttl=fail_ttl)
            if served is not None:
                statuses[label] = served
                yield {"event": "source", **_stream_row(served, len(statuses), n_total)}
                continue
        fut = _submit_refresh(label)
        aw_to_label[asyncio.wrap_future(fut)] = (label, _read_cache(label))

    remaining = set(aw_to_label)
    while remaining:
        budget = deadline - (time.time() - started)
        if budget <= 0:
            break
        done, remaining = await asyncio.wait(
            remaining, timeout=budget, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            break
        for aw in done:
            label, _prior = aw_to_label[aw]
            statuses[label] = _status(label, aw.result(), cache_hit=False)
            yield {"event": "source",
                   **_stream_row(statuses[label], len(statuses), n_total)}

    # Deadline hit: emit honest background-refreshing rows for the stragglers.
    for aw in remaining:
        label, prior = aw_to_label[aw]
        statuses[label] = _abandon_status(label, prior, deadline)
        yield {"event": "source",
               **_stream_row(statuses[label], len(statuses), n_total)}

    result = _assemble(statuses, started=started, deadline=deadline)
    yield {"event": "summary", **result["summary"]}


def _stream_row(st: dict, landed: int, total: int) -> dict:
    """Compact per-source SSE row (no full event bodies — those come in the
    /live/intel-fan-out aggregate; the stream is for progressive UI fill)."""
    return {
        "label": st["label"],
        "role": st["role"],
        "n_events": st["n_events"],
        "age_seconds": st["age_seconds"],
        "degraded": st["degraded"],
        "degraded_reason": st["degraded_reason"],
        "cache_hit": st["cache_hit"],
        "background_refreshing": st["background_refreshing"],
        "latency_ms": st["latency_ms"],
        "landed": landed,
        "n_sources": total,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = fan_out_all(timeout_s=DEFAULT_DEADLINE_S)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFirst 3 events from {len(result['events'])} total:")
    for ev in result["events"][:3]:
        print(json.dumps(ev, indent=2, ensure_ascii=False)[:400])
        print("...")
