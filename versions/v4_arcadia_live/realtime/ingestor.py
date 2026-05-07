"""
ingestor.py — Main realtime ingestion loop.

Polls all 5 sources in parallel, writes to SQLite store, emits summary.

Usage:
    # One-shot pull from all sources
    python -m versions.v4_arcadia_live.realtime.ingestor --once

    # Continuous loop, 5-minute cycle
    python -m versions.v4_arcadia_live.realtime.ingestor --interval 300

    # Lookback window (only affects newsapi + gdelt)
    python -m versions.v4_arcadia_live.realtime.ingestor --once --lookback-min 240

    # Skip specific sources
    python -m versions.v4_arcadia_live.realtime.ingestor --once --skip marinetraffic

    # Load .env keys automatically (relies on python-dotenv if available,
    # otherwise pass env inline: FRED_API_KEY=... python -m ...)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .sources import SOURCES
from .store import Event, init_db, insert_events, count_by_source

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv_if_available() -> None:
    """Load .env if python-dotenv installed. Safe no-op otherwise."""
    try:
        from dotenv import load_dotenv  # type: ignore
        loaded = load_dotenv(dotenv_path=ROOT / ".env")
        logger.info("[env] .env loaded: %s", loaded)
    except ImportError:
        # Fallback: manual parse of .env (simple KEY=VALUE lines)
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
            logger.info("[env] manually loaded .env from %s", env_path)


def _run_source(name: str, lookback_minutes: int) -> list[Event]:
    """Run one source's fetch(), with signature variance handled."""
    mod = SOURCES[name]
    try:
        if name in ("newsapi", "gdelt"):
            return mod.fetch(lookback_minutes=lookback_minutes)
        else:
            return mod.fetch()
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] fetch failed: %s", name, e)
        return []


def ingest_once(lookback_minutes: int = 120, skip: tuple[str, ...] = ()) -> dict:
    """One full cycle: parallel fetch + dedup insert."""
    init_db()
    active = [n for n in SOURCES if n not in skip]
    logger.info("[ingestor] cycle start — sources: %s", active)

    all_events: list[Event] = []
    with ThreadPoolExecutor(max_workers=len(active) or 1) as pool:
        futures = {pool.submit(_run_source, name, lookback_minutes): name for name in active}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                evs = fut.result() or []
                logger.info("[ingestor] %s -> %d events", name, len(evs))
                all_events.extend(evs)
            except Exception as e:  # noqa: BLE001
                logger.error("[ingestor] %s raised: %s", name, e)

    new_count = insert_events(all_events)
    logger.info("[ingestor] cycle done — %d fetched, %d new", len(all_events), new_count)
    return {
        "fetched": len(all_events),
        "inserted_new": new_count,
        "counts_by_source_total": count_by_source(),
    }


def ingest_loop(interval_s: int = 300, lookback_minutes: int = 120,
                skip: tuple[str, ...] = (), max_cycles: int = 0) -> None:
    """Continuous loop. max_cycles=0 means infinite."""
    cycle = 0
    while True:
        cycle += 1
        start = time.time()
        try:
            ingest_once(lookback_minutes=lookback_minutes, skip=skip)
        except KeyboardInterrupt:
            logger.info("[ingestor] stopped by user")
            return
        except Exception as e:  # noqa: BLE001
            logger.error("[ingestor] cycle %d crashed: %s", cycle, e)

        if max_cycles and cycle >= max_cycles:
            return

        elapsed = time.time() - start
        sleep_s = max(5, interval_s - elapsed)
        logger.info("[ingestor] cycle %d done in %.1fs, sleeping %.0fs", cycle, elapsed, sleep_s)
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            return


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Single cycle then exit.")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds.")
    parser.add_argument("--lookback-min", type=int, default=120, help="Minutes for newsapi/gdelt.")
    parser.add_argument("--skip", nargs="*", default=[], help="Source names to skip.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=infinite).")
    args = parser.parse_args()

    _load_dotenv_if_available()

    skip = tuple(args.skip)
    if args.once:
        import json as _json
        result = ingest_once(lookback_minutes=args.lookback_min, skip=skip)
        print(_json.dumps(result, indent=2))
    else:
        ingest_loop(
            interval_s=args.interval,
            lookback_minutes=args.lookback_min,
            skip=skip,
            max_cycles=args.max_cycles,
        )


if __name__ == "__main__":
    main()
