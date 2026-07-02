"""ofac_sdn.py — OFAC Specially Designated Nationals list.

Real US sanctions list. Snapshot of the SDN consolidated XML; we extract
recent additions (last 60 days). No auth. Public.

Docs: https://ofac.treasury.gov/specially-designated-nationals-list-data-formats-data-schemas
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from ._common import cached_get, http_get_text, standard_event

logger = logging.getLogger(__name__)

CONS_TXT = "https://www.treasury.gov/ofac/downloads/sdn.csv"


def fetch_recent_designations(days_back: int = 90, limit: int = 30) -> list[dict]:
    """Pull most recent SDN entries.

    The SDN CSV doesn't include addition-dates per row (those live in
    a separate publication record). We pull the full list and return the
    last N entries (file is roughly chronological).
    """
    cache_key = f"sdn_d{days_back}_n{limit}_list"

    def _fetch():
        try:
            text = http_get_text(CONS_TXT, timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("[ofac_sdn] fetch failed: %s", str(e)[:120])
            return []
        # CSV columns (no header):
        #  ent_num,SDN_Name,SDN_Type,Program,Title,Call_Sign,Vess_type,
        #  Tonnage,GRT,Vess_flag,Vess_owner,Remarks
        import csv, io
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        # Take last N (most recent additions)
        recent = rows[-limit:]
        out = [_normalize(r) for r in recent if len(r) >= 4]
        logger.info("[ofac_sdn] returned %d recent designations (catalog total %d)",
                    len(out), len(rows))
        return out

    return cached_get("ofac_sdn", cache_key, _fetch, ttl=86400)


def _normalize(row: list[str]) -> dict:
    ent_num = row[0] if len(row) > 0 else "?"
    name = row[1] if len(row) > 1 else "?"
    sdn_type = row[2] if len(row) > 2 else "?"
    program = row[3] if len(row) > 3 else "?"
    remarks = row[-1] if len(row) >= 12 else ""

    # Severity proxy by program: IRAN, NPWMD, SDGT, RUSSIA = high
    program_low = program.lower()
    sev = 0.4
    for hi in ("iran", "npwmd", "sdgt", "russia", "rusoel", "syria"):
        if hi in program_low: sev = 0.7; break

    return standard_event(
        source="ofac_sdn", event_id=f"sdn_{ent_num}",
        title=f"OFAC SDN: {name[:80]} ({sdn_type}, {program})",
        description=(f"OFAC SDN entry {ent_num}. Name: {name}. "
                     f"Type: {sdn_type}. Program: {program}. "
                     f"Remarks: {remarks[:200]}")[:1500],
        occurred_at_utc=None,  # CSV row doesn't have date; would need consolidated XML
        raw_url="https://sanctionssearch.ofac.treas.gov/",
        severity_proxy=sev,
        extra={
            "ent_num": ent_num, "name": name,
            "sdn_type": sdn_type, "program": program,
        },
    )


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_recent_designations(limit=10)
    print(json.dumps(e[:3], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} recent OFAC SDN designations")
