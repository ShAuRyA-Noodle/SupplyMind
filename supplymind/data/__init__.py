"""supplymind.data — live-signal data layer for the SupplyMind war room.

The golden path's step 1 ("a real live event surfaces") is served here. Every
public function fetches from a real, keyed-or-keyless public API, timestamps the
result, caches it on disk, and — critically — fails LOUD: a source that is down,
rate-limited, or missing its key returns ``degraded=True`` with a human-readable
``degraded_reason``. Nothing is ever synthesised or hardcoded-as-measured
(CLAUDE.md §0).

Modules:
  * ``live``       — per-source clients + freshness stamps + on-disk cache.
  * ``provenance`` — source metadata (provider, licence, docs) + payload stamping.
"""
from __future__ import annotations

from .live import (
    SourceStatus,
    fetch_all_sources,
    fetch_source,
    recent_disaster,
    recent_events,
    signal_counts,
    SOURCES,
)
from .provenance import source_meta, stamp

__all__ = [
    "SourceStatus",
    "fetch_all_sources",
    "fetch_source",
    "recent_disaster",
    "recent_events",
    "signal_counts",
    "SOURCES",
    "source_meta",
    "stamp",
]
