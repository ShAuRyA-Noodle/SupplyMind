"""cook_v2.py — auto-cook crisis library v2 from real EMDAT 16,812 events.

Replaces the hand-curated 8-event v1 library. Every entry is from real
data with severity derived from REAL death/damage/affected counts —
never an LLM judgment, never a hand-set tier.

Pipeline:
  1. Load external_data/emdat/emdat_public_2000_2026.xlsx (16,812 rows)
  2. Filter to events with at least one severity signal (deaths/damage/affected)
  3. Compose embedding text: "Title — Country, Year. Type. N deaths, $X damage."
  4. Severity tier from deterministic rules on real numbers
  5. Embed via mxbai-embed-large (the P@1=0.962 winner)
  6. Save scenarios/crisis_library_v2.json (events) + .faiss (HNSW index)

Usage:
  python -m scripts.crisis_library.cook_v2 --max 1500
  python -m scripts.crisis_library.cook_v2 --max 5000  # slow but full
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EMDAT_XLSX = REPO_ROOT / "external_data" / "emdat" / "emdat_public_2000_2026.xlsx"
OUT_JSON = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2.json"
OUT_FAISS = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2.faiss"
OUT_NPZ = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2_emb.npz"


def severity_tier(deaths: float, damage_usd: float, affected: float) -> str:
    """Deterministic tier from REAL EMDAT numbers (no LLM, no judgment).

    Picks the WORST applicable tier across deaths/damage/affected.
    """
    tiers = []
    # Deaths
    if deaths >= 1000:    tiers.append("CRITICAL")
    elif deaths >= 100:   tiers.append("HIGH")
    elif deaths >= 10:    tiers.append("MEDIUM")
    elif deaths > 0:      tiers.append("LOW")
    # Damage USD
    if damage_usd >= 10_000_000_000: tiers.append("CRITICAL")
    elif damage_usd >= 1_000_000_000: tiers.append("HIGH")
    elif damage_usd >= 100_000_000:   tiers.append("MEDIUM")
    elif damage_usd > 0:              tiers.append("LOW")
    # Total affected
    if affected >= 10_000_000:        tiers.append("CRITICAL")
    elif affected >= 1_000_000:       tiers.append("HIGH")
    elif affected >= 100_000:         tiers.append("MEDIUM")
    elif affected > 0:                tiers.append("LOW")
    if not tiers:
        return "LOW"
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    return max(tiers, key=lambda t: rank[t])


def _to_float(x) -> float:
    try:
        if x is None or x == "" or x == "None": return 0.0
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _compose_text(row: dict) -> tuple[str, str]:
    """Returns (short_title, long_text_for_embedding)."""
    name = row.get("Event Name") or ""
    country = row.get("Country") or "?"
    year = row.get("Start Year") or "?"
    dtype = row.get("Disaster Type") or "?"
    subtype = row.get("Disaster Subtype") or ""
    region = row.get("Region") or ""
    location = row.get("Location") or ""
    magnitude = row.get("Magnitude") or ""
    deaths = _to_float(row.get("Total Deaths"))
    damage = _to_float(row.get("Total Damage, Adjusted ('000 US$)")) * 1000.0
    if damage <= 0:
        damage = _to_float(row.get("Total Damage ('000 US$)")) * 1000.0
    affected = _to_float(row.get("Total Affected"))

    title = (f"{name or dtype} — {country} ({year})")[:160]
    text = (
        f"Disaster: {dtype}{(' / ' + subtype) if subtype else ''}. "
        f"Country: {country}. Region: {region}. "
        f"Location: {location[:200]}. "
        f"Year: {year}. "
        f"Event name: {name[:160]}. "
        f"Magnitude: {magnitude}. "
        f"Total deaths: {int(deaths)}. "
        f"Total damage USD: {damage:,.0f}. "
        f"Total affected: {int(affected)}."
    )
    return title, text


def _to_dict(headers: list, row: tuple) -> dict:
    return {h: v for h, v in zip(headers, row)}


def load_emdat(max_rows: int | None = None) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(str(EMDAT_XLSX), read_only=False, data_only=True)
    ws = wb["EM-DAT Data"]
    rows_iter = ws.iter_rows(values_only=True)
    headers = list(next(rows_iter))
    out: list[dict] = []
    for i, raw in enumerate(rows_iter):
        d = _to_dict(headers, raw)
        deaths = _to_float(d.get("Total Deaths"))
        damage = _to_float(d.get("Total Damage, Adjusted ('000 US$)"))
        affected = _to_float(d.get("Total Affected"))
        if deaths == 0 and damage == 0 and affected == 0:
            continue  # skip events with no severity signal
        out.append(d)
        if max_rows and len(out) >= max_rows:
            break
    logger.info("[cook_v2] loaded %d EMDAT events with severity signal", len(out))
    return out


def embed_batch(texts: list[str], model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
                  batch_size: int = 32) -> np.ndarray:
    """Compute mxbai-embed-large embeddings (P@1=0.962 winner)."""
    from sentence_transformers import SentenceTransformer
    logger.info("[cook_v2] loading embedder %s ...", model_name)
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.astype("float32")


def build_faiss_index(embs: np.ndarray, out_path: Path) -> None:
    import faiss
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product on normalized vectors == cosine
    index.add(embs)
    faiss.write_index(index, str(out_path))
    logger.info("[cook_v2] FAISS index written to %s (%d vectors, dim %d)",
                out_path, embs.shape[0], d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1500,
                        help="Max events to embed (1500 = ~1 min, full = ~10 min)")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()

    # 1. Load
    raw = load_emdat(max_rows=args.max)
    if not raw:
        logger.error("[cook_v2] no EMDAT events loaded — check %s", EMDAT_XLSX)
        return

    # 2. Compose embedding text + per-event metadata
    events: list[dict] = []
    texts: list[str] = []
    for row in raw:
        title, text = _compose_text(row)
        deaths = _to_float(row.get("Total Deaths"))
        damage = _to_float(row.get("Total Damage, Adjusted ('000 US$)")) * 1000.0
        if damage <= 0:
            damage = _to_float(row.get("Total Damage ('000 US$)")) * 1000.0
        affected = _to_float(row.get("Total Affected"))
        tier = severity_tier(deaths, damage, affected)
        events.append({
            "event_id": row.get("DisNo."),
            "title": title,
            "embed_text": text,
            "country": row.get("Country"),
            "iso3": row.get("ISO"),
            "region": row.get("Region"),
            "year": row.get("Start Year"),
            "disaster_type": row.get("Disaster Type"),
            "disaster_subtype": row.get("Disaster Subtype"),
            "severity_tier_emdat": tier,
            "deaths": int(deaths),
            "damage_usd": damage,
            "total_affected": int(affected),
            "magnitude": row.get("Magnitude"),
            "location": row.get("Location"),
            "raw_url": "https://public.emdat.be (EM-DAT, CRED / UCLouvain)",
            "ground_truth_source": "EMDAT_2000-2026_deterministic_severity_rule",
            "embedding_model": "mxbai-embed-large-v1",
        })
        texts.append(text)

    # 3. Embed
    embs = embed_batch(texts)
    assert embs.shape[0] == len(events)

    # 4. Save catalog + raw embeddings + FAISS
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, embeddings=embs)
    build_faiss_index(embs, OUT_FAISS)

    # 5. Tier distribution sanity
    tier_counts: dict[str, int] = {}
    for ev in events:
        t = ev["severity_tier_emdat"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    catalog = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_events": len(events),
        "tier_distribution": tier_counts,
        "embedding_model": "mxbai-embed-large-v1",
        "embedding_dim": int(embs.shape[1]),
        "ground_truth_source": "EMDAT_2000-2026_deterministic_severity_rule",
        "severity_rule": (
            "deaths>=1000 OR damage>=$10B OR affected>=10M -> CRITICAL; "
            "deaths>=100 OR damage>=$1B OR affected>=1M -> HIGH; "
            "deaths>=10 OR damage>=$100M OR affected>=100K -> MEDIUM; "
            "else LOW"
        ),
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    logger.info("[cook_v2] DONE in %.1fs", time.time() - t0)
    logger.info("[cook_v2] wrote %d events to %s", len(events), args.out)
    logger.info("[cook_v2] tier counts: %s", tier_counts)


if __name__ == "__main__":
    main()
