"""live_signals.py — aggregate live signals for the War Room endpoint.

Wraps the existing v4 sources (NewsAPI, GDELT, FRED Brent) with a uniform
interface that returns:
    {
        "headlines": [{title, url, source, published_at, summary}, ...],
        "brent_usd": float | None,
        "brent_evidence": Evidence dict,
        "concatenated_text_for_keyword_match": str,
        "live_query_failures": [...],
    }

When live APIs are unavailable (no key, rate limit, network down) we fall
back to the offline replay cache produced by
versions/v5_phoenix/realtime_v5/freeze_cache.py. The fallback path is visibly
marked in the response — judges can distinguish live from replayed.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import provenance


def _load_dotenv() -> None:
    """Best-effort .env loader (we don't pull in python-dotenv as a dep)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def fetch_news_headlines(query: str, max_results: int = 5) -> list[dict]:
    """Try v4 NewsAPI source; return [] on any failure (caller falls back)."""
    _load_dotenv()
    if not os.environ.get("NEWS_API_KEY"):
        logger.info("[war_room] NEWS_API_KEY not set; skipping NewsAPI")
        return []
    try:
        from versions.v4_arcadia_live.realtime.sources.newsapi import fetch_recent
        rows = fetch_recent(query=query, page_size=max_results)
        # Normalise — v4's newsapi.fetch_recent shape varies; coerce safely.
        out = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append({
                    "title": r.get("title") or r.get("headline") or "",
                    "url": r.get("url") or r.get("link") or "",
                    "source": r.get("source", {}).get("name") if isinstance(r.get("source"), dict) else r.get("source", ""),
                    "published_at": r.get("publishedAt") or r.get("published_at") or "",
                    "summary": (r.get("description") or r.get("content") or "")[:280],
                })
        return out[:max_results]
    except Exception as e:  # noqa: BLE001
        logger.warning("[war_room] NewsAPI failed: %s", e)
        return []


def fetch_brent_usd() -> tuple[float | None, dict]:
    """Try FRED Brent; return (price, evidence_dict)."""
    _load_dotenv()
    if not os.environ.get("FRED_API_KEY"):
        return None, provenance.Evidence(
            source_type="model_estimate",
            derivation="FRED_API_KEY not set; Brent price omitted from this run."
        ).to_dict()
    try:
        from versions.v4_arcadia_live.realtime.sources.fred_brent import fetch_latest_brent
        price, meta = fetch_latest_brent()
        if price is None:
            return None, provenance.Evidence(
                source_type="model_estimate",
                derivation="FRED responded but no recent Brent observation parseable."
            ).to_dict()
        return float(price), provenance.live_api(
            publisher="FRED (Federal Reserve Economic Data)",
            url=meta.get("url", "https://fred.stlouisfed.org/series/DCOILBRENTEU"),
        ).to_dict()
    except Exception as e:  # noqa: BLE001
        logger.warning("[war_room] FRED Brent fetch failed: %s", e)
        return None, provenance.Evidence(
            source_type="model_estimate",
            derivation=f"FRED fetch error: {e!r}"
        ).to_dict()


def load_replay_fallback() -> list[dict]:
    """Read the offline replay cache from realtime_v5/."""
    cache_path = ROOT / "versions/v5_phoenix" / "realtime_v5" / "replay_cache_latest.json"
    if not cache_path.exists():
        return []
    import json
    try:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for ev_id, ev in blob.get("events", {}).items():
        out.append({
            "title": ev.get("top_analog", {}).get("name", ev_id),
            "url": "",
            "source": "versions/v5_phoenix replay_cache",
            "published_at": ev.get("top_analog", {}).get("date", ""),
            "summary": ev.get("scenario_input", {}).get("scenario_text", "")[:280],
        })
    return out


def aggregate(scenario_text: str, enable_live: bool = True) -> dict:
    """Top-level: return everything the War Room ranker needs."""
    started = time.time()
    failures: list[str] = []
    served_from_replay = False

    headlines: list[dict] = []
    if enable_live:
        try:
            headlines = fetch_news_headlines(scenario_text, max_results=5)
        except Exception as e:  # noqa: BLE001
            failures.append(f"newsapi: {e!r}")
    if not headlines:
        headlines = load_replay_fallback()
        served_from_replay = True

    brent_price, brent_evidence = (None, {})
    if enable_live:
        brent_price, brent_evidence = fetch_brent_usd()

    concatenated = " ".join((h.get("title", "") + " " + h.get("summary", "")) for h in headlines)

    return {
        "headlines": headlines,
        "brent_usd": brent_price,
        "brent_evidence": brent_evidence,
        "concatenated_text_for_keyword_match": concatenated,
        "served_from_replay": served_from_replay,
        "live_query_failures": failures,
        "elapsed_s": round(time.time() - started, 2),
    }
