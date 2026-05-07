"""embedder_ensemble.py — multi-embedder ensemble (mxbai + Snowflake-Arctic).

Returns (a) per-embedder top-K matches against the crisis library, (b) cosine
agreement between embedders, (c) ensemble score = mean(mxbai_score,
snowflake_score) for stability.

mxbai-only P@1 is already 0.962 on R5 — the value of this ensemble is
catching the ~4% borderline cases where one embedder retrieves the wrong
analog while the other retrieves the right one.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SNOWFLAKE_DIR = REPO_ROOT / "models" / "snowflake-arctic-embed-l"

_snowflake = None
_DEVICE = None


def _load_snowflake():
    global _snowflake, _DEVICE
    if _snowflake is not None:
        return _snowflake
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _snowflake = SentenceTransformer(str(SNOWFLAKE_DIR), device=_DEVICE)
        logger.info("[embedder-ensemble] Snowflake-Arctic-L loaded on %s",
                    _DEVICE)
        return _snowflake
    except Exception as e:  # noqa: BLE001
        logger.warning("[embedder-ensemble] Snowflake load failed: %s", e)
        _snowflake = "FAILED"
        return None


def _snowflake_query(query: str, candidates: list[dict],
                      doc_field: str = "title") -> list[dict] | None:
    """Embed query + candidate docs with Snowflake, return candidates ranked
    by Snowflake cosine. Returns None if Snowflake unavailable."""
    sf = _load_snowflake()
    if sf is None or sf == "FAILED":
        return None
    try:
        # Build doc texts
        def _doc(c):
            return (c.get(doc_field) or c.get("title") or
                    c.get("name") or "")[:1024]
        docs = [_doc(c) for c in candidates]
        q_emb = sf.encode([query], normalize_embeddings=True)[0]
        d_embs = sf.encode(docs, normalize_embeddings=True,
                            convert_to_numpy=True)
        scores = (d_embs @ q_emb).astype(np.float32)
        out = []
        for c, s in zip(candidates, scores):
            c2 = dict(c); c2["snowflake_score"] = round(float(s), 4)
            out.append(c2)
        out.sort(key=lambda x: x["snowflake_score"], reverse=True)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("[embedder-ensemble] snowflake encode failed: %s", e)
        return None


def ensemble_search(query: str, top_k: int = 5,
                     faiss_k: int = 20) -> dict:
    """Run mxbai (via library_v2_search) + Snowflake in parallel,
    blend scores, return top_k."""
    t0 = time.time()
    try:
        from versions.v4_arcadia_live.scenarios.library_v2_search import search
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"library_v2_search_unavailable: {e}"}

    # mxbai retrieval
    mxbai_candidates = search(query, top_k=faiss_k) or []
    mxbai_topk_names = set(
        (c.get("event_id") or c.get("title") or "")
        for c in mxbai_candidates[:top_k]
    )

    # snowflake re-scoring on the same FAISS candidate pool
    sf_ranked = _snowflake_query(query, mxbai_candidates)

    if sf_ranked is None:
        return {
            "ok": False, "error": "snowflake_unavailable",
            "mxbai_top_k": mxbai_candidates[:top_k],
            "fallback": "mxbai_only",
        }

    sf_topk_names = set(
        (c.get("event_id") or c.get("title") or "")
        for c in sf_ranked[:top_k]
    )

    # Ensemble score = mean(normalized mxbai_score, snowflake_score)
    score_table: dict[str, dict] = {}
    for c in mxbai_candidates:
        key = c.get("event_id") or c.get("title") or str(id(c))
        score_table.setdefault(key, dict(c))
        score_table[key]["mxbai_score"] = float(c.get("_match_score", 0.0))
    for c in sf_ranked:
        key = c.get("event_id") or c.get("title") or str(id(c))
        if key in score_table:
            score_table[key]["snowflake_score"] = float(c.get("snowflake_score", 0.0))
    for k, v in score_table.items():
        m = v.get("mxbai_score", 0.0)
        s = v.get("snowflake_score", 0.0)
        v["ensemble_score"] = round((m + s) / 2.0, 4)

    ensemble_ranked = sorted(score_table.values(),
                              key=lambda x: x["ensemble_score"], reverse=True)

    # Agreement: how many of mxbai-top-k overlap with snowflake-top-k
    overlap = len(mxbai_topk_names & sf_topk_names)
    agreement = overlap / max(1, top_k)

    return {
        "ok": True,
        "embedders": ["mxbai-embed-large", "snowflake-arctic-embed-l"],
        "n_candidates": len(mxbai_candidates),
        "ensemble_top_k": ensemble_ranked[:top_k],
        "mxbai_top_k_names": sorted(mxbai_topk_names),
        "snowflake_top_k_names": sorted(sf_topk_names),
        "topk_overlap": overlap,
        "topk_agreement": round(agreement, 3),
        "elapsed_s": round(time.time() - t0, 3),
        "device": _DEVICE,
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, str(REPO_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = ensemble_search(
        "Iran-Israel-US escalation restricts Strait of Hormuz",
        top_k=5, faiss_k=20,
    )
    if res.get("ok"):
        print(f"agreement: {res['topk_agreement']*100:.0f}% overlap "
               f"({res['topk_overlap']}/5)  · elapsed {res['elapsed_s']}s")
        for c in res["ensemble_top_k"]:
            print(f"  ens={c['ensemble_score']:.3f}  mx={c.get('mxbai_score',0):.3f}  "
                   f"sf={c.get('snowflake_score',0):.3f}  "
                   f"{(c.get('title') or c.get('event_id') or '?')[:80]}")
    else:
        print(json.dumps(res, indent=2))
