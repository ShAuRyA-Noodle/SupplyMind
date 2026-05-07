"""_common.py — shared helpers for sources_v2 modules."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / ".source_cache"
CACHE_TTL_SECONDS = 600   # 10 min default


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_path(source: str, key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:120]
    return CACHE_DIR / source / f"{safe}.json"


def cached_get(
    source: str, cache_key: str, fetch_fn,
    ttl: int = CACHE_TTL_SECONDS,
) -> Any:
    """Run fetch_fn() unless we have a fresh cached result. Always writes
    on success. Never raises — returns [] / {} on failure with a warning."""
    p = _cache_path(source, cache_key)
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age < ttl:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    try:
        out = fetch_fn()
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] fetch failed: %s", source, str(e)[:200])
        return [] if cache_key.endswith("_list") else {}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except OSError:
        pass
    return out


def standard_event(
    source: str,
    event_id: str,
    title: str,
    description: str,
    occurred_at_utc: str | None,
    raw_url: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    severity_proxy: float | None = None,
    extra: dict | None = None,
) -> dict:
    out: dict[str, Any] = {
        "source": source,
        "event_id": event_id,
        "title": title or "",
        "description": (description or "")[:1500],
        "occurred_at_utc": occurred_at_utc,
        "lat": lat,
        "lon": lon,
        "severity_proxy": severity_proxy,
        "raw_url": raw_url,
        "fetched_at_utc": _now_iso(),
        "inference_type": f"live_{source}",
    }
    if extra:
        out.update(extra)
    return out


def http_get_json(url: str, *, params: dict | None = None,
                  headers: dict | None = None, timeout: float = 20.0) -> Any:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()


def http_get_text(url: str, *, params: dict | None = None,
                  headers: dict | None = None, timeout: float = 20.0) -> str:
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.text
