"""orchestrator_v2.py — fan out across all sources_v2 modules concurrently.

Calls every source in parallel via ThreadPoolExecutor. Each source is
isolated so one failure cannot block another. Results are aggregated
into one list with uniform schema; summary stats included.

Used by `POST /live/intel-fan-out` in server/app.py extensions.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor, as_completed,
    TimeoutError as FuturesTimeoutError,
)
from typing import Callable

from .sources_v2 import (
    cisa_kev, eia, gdelt_conflict, gdelt_humanitarian, gfw,
    hackernews, nasa_eonet, nasa_firms, noaa_ndbc, noaa_tides,
    ofac_sdn, sec_edgar_8k, who_don, wiki_pageviews, worldbank,
)

# Existing v1 sources (NewsAPI, GDELT, USGS, MarineTraffic, FRED Brent)
# wired in via server.app /live/recent-events endpoint already; we
# re-aggregate them here for a unified view. Adapter converts the v1
# Event dataclass -> the v2 standard event dict.
from .sources import fred_brent, gdelt as gdelt_v1, newsapi, usgs

logger = logging.getLogger(__name__)


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
    # --- v1 baseline (5) ---
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


def fan_out_all(
    *,
    timeout_s: float = 35.0,
    parallel: int = 8,
) -> dict:
    """Fan out across all 20 sources concurrently. Returns:

      {"summary": {...counts...}, "events": [...20-source merged...]}
    """
    started = time.time()
    results: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {
            ex.submit(_safe_call, label, fn, kwargs): label
            for (label, fn, kwargs, _role) in SOURCE_FLEET
        }
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                label = futures[fut]
                try:
                    ok_events = fut.result(timeout=2)
                    results[label] = ok_events
                except Exception as e:  # noqa: BLE001
                    errors[label] = f"{type(e).__name__}: {str(e)[:160]}"
                    results[label] = []
        except FuturesTimeoutError:
            # Some sources still running — record them as timeouts and move on
            for fut, label in futures.items():
                if label not in results:
                    if fut.done():
                        try:
                            results[label] = fut.result(timeout=1)
                        except Exception as e:  # noqa: BLE001
                            errors[label] = f"{type(e).__name__}: {str(e)[:160]}"
                            results[label] = []
                    else:
                        errors[label] = f"timeout after {timeout_s}s (still running)"
                        results[label] = []
                        fut.cancel()

    all_events: list[dict] = []
    role_map = {label: role for label, _, _, role in SOURCE_FLEET}
    for label, events in results.items():
        for ev in events:
            ev = dict(ev)  # shallow copy
            ev.setdefault("role_tag", role_map.get(label, "unknown"))
            all_events.append(ev)

    elapsed = time.time() - started
    n_per_source = {label: len(events) for label, events in results.items()}
    summary = {
        "n_sources_total": len(SOURCE_FLEET),
        "n_sources_with_data": sum(1 for v in results.values() if v),
        "n_sources_errored": len(errors),
        "n_events_total": len(all_events),
        "n_events_per_source": n_per_source,
        "errors_per_source": errors,
        "elapsed_s": round(elapsed, 2),
        "fan_out_concurrency": parallel,
        "inference_type": "live_multi_source_fan_out",
    }
    return {"summary": summary, "events": all_events}


def _safe_call(label: str, fn: Callable, kwargs: dict) -> list[dict]:
    """Call one source function, log + reraise any exception with the label."""
    t0 = time.time()
    try:
        out = fn(**kwargs) or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[orchestrator_v2:%s] failed in %.1fs: %s",
                        label, time.time() - t0, str(e)[:120])
        raise
    if not isinstance(out, list):
        out = []
    logger.info("[orchestrator_v2:%s] %d events in %.1fs",
                label, len(out), time.time() - t0)
    return out


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = fan_out_all(timeout_s=60)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFirst 3 events from {len(result['events'])} total:")
    for ev in result["events"][:3]:
        print(json.dumps(ev, indent=2, ensure_ascii=False)[:400])
        print("...")
