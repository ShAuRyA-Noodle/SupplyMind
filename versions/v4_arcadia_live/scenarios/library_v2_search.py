"""library_v2_search.py — load + search the cooked crisis library v2.

Singleton-loaded FAISS index + mxbai embedder. Matches a query string
to top-K events from the 1500-event EMDAT-derived library.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2.json"
FAISS_IDX = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2.faiss"

_catalog: dict | None = None
_index = None
_embedder = None


def _load() -> tuple[dict, Any, Any]:
    global _catalog, _index, _embedder
    if _catalog is None:
        if not CATALOG.exists():
            raise FileNotFoundError(f"library v2 not yet cooked: {CATALOG}")
        _catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if _index is None:
        import faiss
        _index = faiss.read_index(str(FAISS_IDX))
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    return _catalog, _index, _embedder


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return top-K events from the library most similar to the query."""
    cat, idx, emb = _load()
    qvec = emb.encode([query], normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")
    distances, indices = idx.search(qvec, top_k)
    out = []
    events = cat["events"]
    for rank, (i, d) in enumerate(zip(indices[0], distances[0])):
        if i < 0 or i >= len(events):
            continue
        ev = dict(events[i])
        ev["_match_score"] = float(d)  # cosine since vectors normalized
        ev["_rank"] = rank + 1
        # Strip large fields for response
        ev.pop("embed_text", None)
        out.append(ev)
    return out


if __name__ == "__main__":
    import json as _json
    logging.basicConfig(level=logging.INFO)
    queries = [
        "Iran threatens to close the Strait of Hormuz",
        "Major earthquake hits Japan with tsunami",
        "Suez Canal blocked by container ship",
        "COVID outbreak disrupts semiconductor supply",
    ]
    for q in queries:
        results = search(q, top_k=3)
        print(f"\n=== {q!r} ===")
        for r in results:
            print(f"  [{r['_rank']}] score={r['_match_score']:.3f}  "
                  f"{r['title'][:80]}  tier={r['severity_tier_emdat']}")
