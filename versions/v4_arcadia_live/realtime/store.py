"""
store.py — SQLite event store for live geopolitical signals.

Zero-config, single-file DB at `versions/v4_arcadia_live/realtime/events.db`.
Schema is append-only; dedup by (source, text_hash) within a 24h window.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "events.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    ts_iso       TEXT NOT NULL,
    ts_unix      REAL NOT NULL,
    event_type   TEXT NOT NULL,
    severity     REAL,
    region       TEXT,
    raw_text     TEXT,
    text_hash    TEXT NOT NULL,
    urls         TEXT,
    entities     TEXT,
    meta_json    TEXT,
    ingested_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_source_hash ON events(source, text_hash);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_unix);
CREATE INDEX IF NOT EXISTS idx_events_region ON events(region);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


@dataclass
class Event:
    source: str                          # "newsapi" | "gdelt" | "usgs" | "marinetraffic" | "fred_brent"
    ts_iso: str                          # "2026-04-21T14:30:00Z"
    event_type: str                      # "conflict" | "earthquake" | "shipping_delay" | "commodity_spike"
    region: str = ""                     # "hormuz" | "red_sea" | "taiwan_strait" | "iran" | "israel" | ...
    severity: float = 0.0                # 0.0 - 1.0
    raw_text: str = ""
    urls: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ts_unix(self) -> float:
        import datetime as _dt
        return _dt.datetime.fromisoformat(self.ts_iso.replace("Z", "+00:00")).timestamp()

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(
            (self.source + "|" + self.raw_text[:500]).encode("utf-8")
        ).hexdigest()[:16]


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)


def insert_events(events: Iterable[Event], dedup_window_s: int = 86400) -> int:
    """Insert events with dedup.

    Returns number of NEW events inserted.
    """
    init_db()
    inserted = 0
    now = time.time()
    with _conn() as c:
        for e in events:
            h = e.text_hash
            row = c.execute(
                "SELECT id FROM events WHERE source=? AND text_hash=? AND ts_unix > ?",
                (e.source, h, now - dedup_window_s),
            ).fetchone()
            if row is not None:
                continue
            c.execute(
                """
                INSERT INTO events (source, ts_iso, ts_unix, event_type, severity,
                                    region, raw_text, text_hash, urls, entities,
                                    meta_json, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    e.source, e.ts_iso, e.ts_unix, e.event_type, e.severity,
                    e.region, e.raw_text, h,
                    json.dumps(e.urls), json.dumps(e.entities),
                    json.dumps(e.meta), now,
                ),
            )
            inserted += 1
    return inserted


def query_recent(
    since_unix: Optional[float] = None,
    region: Optional[str] = None,
    source: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Query events newer than since_unix, optional filters, sorted desc by ts."""
    init_db()
    clauses, params = [], []
    if since_unix is not None:
        clauses.append("ts_unix >= ?")
        params.append(since_unix)
    if region:
        clauses.append("region = ?")
        params.append(region)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM events {where} ORDER BY ts_unix DESC LIMIT ?"
    params.append(limit)

    with _conn() as c:
        rows = c.execute(sql, params).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        for k in ("urls", "entities", "meta_json"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    d[k] = []
        out.append(d)
    return out


def count_by_source(since_unix: Optional[float] = None) -> dict[str, int]:
    init_db()
    sql = "SELECT source, COUNT(*) as n FROM events"
    params = []
    if since_unix is not None:
        sql += " WHERE ts_unix >= ?"
        params.append(since_unix)
    sql += " GROUP BY source"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    return {r["source"]: r["n"] for r in rows}


def purge_older_than(days: int = 14) -> int:
    """Purge events older than N days. Returns number deleted."""
    init_db()
    cutoff = time.time() - days * 86400
    with _conn() as c:
        cur = c.execute("DELETE FROM events WHERE ts_unix < ?", (cutoff,))
        return cur.rowcount


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--recent", type=int, default=0, help="Show N most recent events")
    parser.add_argument("--purge-days", type=int, default=0)
    args = parser.parse_args()

    if args.init:
        init_db()
        print(f"initialized {DB_PATH}")

    if args.count:
        print(json.dumps(count_by_source(), indent=2))

    if args.recent:
        for e in query_recent(limit=args.recent):
            print(f"{e['ts_iso']}  {e['source']:15s}  {e['region']:15s}  {e['event_type']:15s}  {e['raw_text'][:100]}")

    if args.purge_days:
        n = purge_older_than(args.purge_days)
        print(f"purged {n} events older than {args.purge_days} days")
