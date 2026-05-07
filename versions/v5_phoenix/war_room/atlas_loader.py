"""atlas_loader.py — load + validate the three curated atlases.

Returned objects are plain dicts (not Pydantic) because the atlases ship
as JSON-on-disk and the consumer is the war_room ranker, which uses them
read-only. Validation is structural (required keys present, citations have
URLs) — we explicitly do NOT validate the URLs are reachable here; that
would tie the test suite to network availability.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ROOT / "versions/v5_phoenix" / "scenarios"

CHOKEPOINT_PATH = SCENARIOS / "hormuz_chokepoint_atlas.json"
INDIA_PATH = SCENARIOS / "india_supply_chain_exposure.json"
GULF_PATH = SCENARIOS / "gulf_supply_chain_exposure.json"


@dataclass(frozen=True)
class Atlases:
    chokepoint: dict
    india: dict
    gulf: dict


class AtlasValidationError(Exception):
    pass


def _require(d: dict, keys: list[str], where: str) -> None:
    for k in keys:
        if k not in d:
            raise AtlasValidationError(f"{where}: missing required key {k!r}")


def _validate_chokepoint(blob: dict) -> None:
    _require(blob, ["schema_version", "facts", "geographic_anchors"], "chokepoint atlas")
    for i, fact in enumerate(blob["facts"]):
        _require(fact, ["id", "claim", "source_type", "publisher", "url"], f"chokepoint.facts[{i}]")
        if fact["source_type"] not in ("primary", "secondary", "model_estimate"):
            raise AtlasValidationError(f"chokepoint.facts[{i}].source_type invalid: {fact['source_type']}")
    if not blob["geographic_anchors"]:
        raise AtlasValidationError("chokepoint atlas has no geographic anchors")


def _validate_country_atlas(blob: dict, name: str) -> None:
    _require(blob, ["schema_version", "sectors"], f"{name} atlas")
    if not blob["sectors"]:
        raise AtlasValidationError(f"{name} atlas has no sectors")
    for i, sec in enumerate(blob["sectors"]):
        _require(sec, ["rank", "sector_id", "sector_name", "exposure_facts",
                       "first_symptom_when_hormuz_hits"],
                 f"{name}.sectors[{i}]")
        if not sec["exposure_facts"]:
            raise AtlasValidationError(f"{name}.sectors[{i}] has no exposure facts")
        for j, fact in enumerate(sec["exposure_facts"]):
            _require(fact, ["claim", "source_type", "url"],
                     f"{name}.sectors[{i}].exposure_facts[{j}]")


def load() -> Atlases:
    """Read all three atlases from disk and structurally validate them."""
    chokepoint = json.loads(CHOKEPOINT_PATH.read_text(encoding="utf-8"))
    india = json.loads(INDIA_PATH.read_text(encoding="utf-8"))
    gulf = json.loads(GULF_PATH.read_text(encoding="utf-8"))

    _validate_chokepoint(chokepoint)
    _validate_country_atlas(india, "india")
    _validate_country_atlas(gulf, "gulf")

    logger.info("[atlas] loaded: chokepoint=%d facts, india=%d sectors, gulf=%d sectors",
                len(chokepoint["facts"]), len(india["sectors"]), len(gulf["sectors"]))
    return Atlases(chokepoint=chokepoint, india=india, gulf=gulf)


def chokepoint_facts_summary(blob: dict) -> dict:
    """Return a compact dict of {fact_id: claim} for fast lookups."""
    return {f["id"]: f["claim"] for f in blob["facts"]}
