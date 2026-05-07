"""
crisis_library.py — Load + search real crisis JSON for nearest historical analogs.

Two matching modes:
    - cosine_tfidf : lightweight TF-IDF cosine (no embedding model needed)
    - embed_mxbai  : mxbai-embed-large-v1 via sentence-transformers (if available)

Returns top-k analogs with similarity scores and full event metadata.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LIBRARY_PATH = (Path(__file__).resolve().parents[1]
                / "scenarios" / "iran_israel_hormuz_2024_2026.json")
EMBED_CACHE_PATH = Path(__file__).resolve().parent / "library_embeddings.pkl"


@dataclass
class Analog:
    event_id: str
    name: str
    date: str
    severity: float
    summary: str
    similarity: float
    full_record: dict


# --- TF-IDF fallback -------------------------------------------------------

_STOPWORDS = set("""a an and or of the to in on at for by is are was were be been being
this that these those it its their there with from as into under over between
during against through without including not no more less than both""".split())


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", (text or "").lower())
            if w not in _STOPWORDS and len(w) > 2]


def _tfidf_vectors(docs: list[str]) -> tuple[list[dict], dict]:
    """Return (per-doc term-freq dicts, idf dict)."""
    tf_list = [Counter(_tokenize(d)) for d in docs]
    n_docs = len(tf_list)
    df = Counter()
    for tf in tf_list:
        for term in tf:
            df[term] += 1
    idf = {t: math.log((n_docs + 1) / (c + 1)) + 1 for t, c in df.items()}
    # Normalize to tfidf
    tfidf = []
    for tf in tf_list:
        v = {t: f * idf.get(t, 1.0) for t, f in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        tfidf.append({t: x / norm for t, x in v.items()})
    return tfidf, idf


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    return sum(a[t] * b[t] for t in common)


# --- Embedding mode (optional) --------------------------------------------

_embed_model = None


def _load_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embed_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        logger.info("[crisis_library] mxbai-embed-large loaded")
    except Exception as e:  # noqa: BLE001
        logger.info("[crisis_library] mxbai unavailable (%s); using TF-IDF fallback", e)
        _embed_model = False
    return _embed_model


# --- Main API --------------------------------------------------------------

def _corpus_hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def _cached_doc_embeddings(texts: list[str], model) -> np.ndarray:
    """Cache library embeddings to disk; regenerate only if corpus hash changes."""
    h = _corpus_hash(texts)
    if EMBED_CACHE_PATH.exists():
        try:
            blob = pickle.loads(EMBED_CACHE_PATH.read_bytes())
            if blob.get("corpus_hash") == h:
                return blob["embeddings"]
        except Exception as e:  # noqa: BLE001
            logger.warning("[crisis_library] failed to load embed cache: %s", e)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    try:
        EMBED_CACHE_PATH.write_bytes(pickle.dumps({"corpus_hash": h, "embeddings": vecs}))
        logger.info("[crisis_library] cached library embeddings (%d docs)", len(texts))
    except Exception as e:  # noqa: BLE001
        logger.warning("[crisis_library] failed to save embed cache: %s", e)
    return vecs


def load_library(path: Path = LIBRARY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_text(event: dict) -> str:
    """Concatenate descriptive fields for similarity matching."""
    parts = [
        event.get("name", ""),
        event.get("summary", ""),
        event.get("region", ""),
        event.get("event_type", ""),
        " ".join(event.get("supply_chain_nodes_affected", [])),
        " ".join(event.get("affected_routes", [])),
    ]
    return " ".join(p for p in parts if p)


def find_analogs(query_text: str, k: int = 3, mode: str = "auto") -> list[Analog]:
    """Return top-k historical analogs for the given free-text query.

    mode: 'tfidf' | 'embed_mxbai' | 'auto' (tries embed, falls back to tfidf)
    """
    lib = load_library()
    events = lib["events"]
    texts = [_event_text(e) for e in events]

    if mode == "auto":
        m = _load_embed_model()
        mode = "embed_mxbai" if m else "tfidf"

    if mode == "embed_mxbai":
        model = _load_embed_model()
        if model:
            doc_vecs = _cached_doc_embeddings(texts, model)
            q_vec = model.encode([query_text], normalize_embeddings=True)[0]
            sims = [float((q_vec * dv).sum()) for dv in doc_vecs]
        else:
            mode = "tfidf"

    if mode == "tfidf":
        tfidf_docs, idf = _tfidf_vectors(texts)
        q_tf = Counter(_tokenize(query_text))
        q_vec = {t: f * idf.get(t, 1.0) for t, f in q_tf.items()}
        norm = math.sqrt(sum(x * x for x in q_vec.values())) or 1.0
        q_vec = {t: x / norm for t, x in q_vec.items()}
        sims = [_cosine(q_vec, dv) for dv in tfidf_docs]

    # Rank
    idx_ranked = sorted(range(len(events)), key=lambda i: sims[i], reverse=True)[:k]
    out = []
    for i in idx_ranked:
        e = events[i]
        out.append(Analog(
            event_id=e["id"],
            name=e["name"],
            date=e["date"],
            severity=e["severity"],
            summary=e["summary"],
            similarity=round(sims[i], 4),
            full_record=e,
        ))
    return out


SIM_LOW_BAND = 0.35   # below this, no real analog match
SIM_HIGH_BAND = 0.70  # above this, strong analog match
BASELINE_BENIGN_SEVERITY = 0.10


def interpolate_projection(analogs: list[Analog]) -> dict:
    """Weighted average of analog impacts by similarity.

    Includes a similarity-confidence damper: when the top analog has a weak
    match (sim < SIM_LOW_BAND), severity collapses to a LOW baseline instead
    of propagating high-severity analog numbers into a benign scenario.
    """
    if not analogs:
        return {"brent_projection_usd_bbl_p50": None, "duration_days_p50": None,
                "vessel_rerouting_days_p50": None, "severity_p50": None,
                "top_analog_similarity": 0.0, "confidence": 0.0,
                "top_analog_id": None, "top_analog_name": None}

    top_sim = analogs[0].similarity
    total_w = sum(a.similarity for a in analogs) or 1.0
    weights = [a.similarity / total_w for a in analogs]

    def wavg(field_fn):
        vals = []
        for a in analogs:
            v = field_fn(a.full_record)
            vals.append(v if v is not None else 0.0)
        return sum(w * v for w, v in zip(weights, vals))

    brent_raw = wavg(lambda r: (r.get("oil_impact_usd_bbl") or {}).get("peak", None))
    duration_raw = wavg(lambda r: r.get("duration_days", None))
    rerouting_raw = wavg(lambda r: r.get("vessel_rerouting_days", None))
    severity_raw = wavg(lambda r: r.get("severity", None))

    # Confidence scale 0..1 based on top match strength
    conf = max(0.0, min(1.0, (top_sim - SIM_LOW_BAND) / (SIM_HIGH_BAND - SIM_LOW_BAND)))

    # Dampened severity toward benign baseline when confidence is low
    severity_p50 = conf * severity_raw + (1 - conf) * BASELINE_BENIGN_SEVERITY
    # Same treatment for rerouting/duration: collapse toward 0 for weak matches
    rerouting_p50 = conf * rerouting_raw
    duration_p50 = conf * duration_raw
    # Brent projection: collapse toward current typical baseline $80
    brent_p50 = conf * brent_raw + (1 - conf) * 80.0 if brent_raw else 80.0

    return {
        "brent_projection_usd_bbl_p50": round(brent_p50, 2),
        "duration_days_p50": round(duration_p50, 1),
        "vessel_rerouting_days_p50": round(rerouting_p50, 1),
        "severity_p50": round(severity_p50, 3),
        "top_analog_similarity": round(top_sim, 3),
        "confidence": round(conf, 3),
        "top_analog_id": analogs[0].event_id,
        "top_analog_name": analogs[0].name,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mode", default="auto")
    args = parser.parse_args()

    analogs = find_analogs(args.query, k=args.k, mode=args.mode)
    for a in analogs:
        print(f"  {a.similarity:.3f}  [{a.date}]  {a.name[:80]}  sev={a.severity}")
    proj = interpolate_projection(analogs)
    print("\nprojection:", json.dumps(proj, indent=2))
