"""embedder_single.py — single-embedder (mxbai) crisis-library retrieval.

History (rename honesty): this module used to be `embedder_ensemble.py` and
advertised a 2-embedder ensemble (mxbai + Snowflake-Arctic-L) with a cosine
"agreement" score. The Snowflake weights (models/snowflake-arctic-embed-l) were
never present in this checkout, so the "ensemble" silently collapsed to
mxbai-only on every call — a fake, per CLAUDE.md §0. The ensemble path and the
Snowflake loader have been DELETED.

This module now does exactly one honest thing: mxbai bi-encoder retrieval
against the crisis library, clearly labelled as a single embedder. If a second
embedder is ever shipped (weights present AND verified), reintroduce an
ensemble module explicitly — never re-add a silent fallback.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

EMBEDDER = "mxbai-embed-large"


def single_embedder_search(query: str, top_k: int = 5,
                           faiss_k: int = 20) -> dict:
    """mxbai bi-encoder retrieval against the crisis library (single embedder).

    Delegates to the real FAISS-backed mxbai search in library_v2_search and
    returns the top_k matches, explicitly labelled as single-embedder output.
    """
    t0 = time.time()
    try:
        from supplymind.warroom.scenarios.library_v2_search import search
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"library_v2_search_unavailable: {e}"}

    candidates = search(query, top_k=faiss_k) or []
    top = candidates[:top_k]
    return {
        "ok": True,
        "embedder": EMBEDDER,
        "mode": "single_embedder",
        "n_candidates": len(candidates),
        "top_k": top,
        "top_k_names": [
            (c.get("event_id") or c.get("title") or "") for c in top
        ],
        "elapsed_s": round(time.time() - t0, 3),
    }


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = single_embedder_search(
        "Iran-Israel-US escalation restricts Strait of Hormuz",
        top_k=5, faiss_k=20,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "top_k"}, indent=2))
    for c in (res.get("top_k") or []):
        print(f"  {(c.get('title') or c.get('event_id') or '?')[:80]}")
