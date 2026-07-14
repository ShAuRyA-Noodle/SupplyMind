"""library_v2_rerank.py — BGE-reranker-v2-m3 cross-encoder rerank stage.

RESURRECTED (R9). This module was DELETED in P0.2 because the BGE-reranker
weights were never on disk and the code silently passed through the FAISS
top-k — a fake per CLAUDE.md §0. The weights are now downloaded to
models/bge-reranker-v2-m3/ and this module reranks for real.

Pipeline:
  1. `library_v2_search.search(query, top_k=faiss_k)` returns FAISS top-k
     crisis-library events by mxbai bi-encoder cosine.
  2. Pass (query, candidate_text) pairs through BGE-reranker-v2-m3
     cross-encoder.
  3. Return top-K by rerank score.

Honesty: whether the reranker actually improves retrieval is MEASURED in
`ghost_models_eval.py` on the 53 gold-labeled BEIR queries and recorded in
tests/receipts/ghost_models_eval_REAL.json. If the reranker is unavailable
(weights absent / load fails) this returns an explicit unavailable status with
a passthrough — never a silent fake that pretends reranking happened.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RERANKER_DIR = REPO_ROOT / "models" / "bge-reranker-v2-m3"

_reranker = None
_DEVICE = None


def _load_reranker():
    global _reranker, _DEVICE
    if _reranker is not None:
        return _reranker
    if not (RERANKER_DIR / "model.safetensors").exists():
        logger.warning("[bge-rerank] weights absent at %s", RERANKER_DIR)
        _reranker = "FAILED"
        return None
    try:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(str(RERANKER_DIR),
                                 use_fp16=(_DEVICE == "cuda"))
        logger.info("[bge-rerank] FlagReranker loaded on %s", _DEVICE)
        return _reranker
    except Exception as e:  # noqa: BLE001
        logger.warning("[bge-rerank] FlagReranker load failed: %s — "
                       "trying sentence-transformers CrossEncoder", e)
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(str(RERANKER_DIR), device=_DEVICE or "cpu",
                                     max_length=512)
            logger.info("[bge-rerank] CrossEncoder fallback OK")
            return _reranker
        except Exception as e2:  # noqa: BLE001
            logger.warning("[bge-rerank] both load paths failed: %s", e2)
            _reranker = "FAILED"
            return None


def _doc_text(c: dict, doc_field: str = "summary") -> str:
    """Prefer an explicit text field, else build one from EMDAT event fields."""
    if c.get(doc_field):
        return str(c[doc_field])[:1024]
    if c.get("embed_text"):
        return str(c["embed_text"])[:1024]
    parts = [c.get("title", ""), c.get("disaster_type", ""),
             c.get("disaster_subtype", ""), c.get("country", ""),
             str(c.get("year", "")), c.get("location", ""),
             c.get("severity_tier_emdat", "")]
    return " · ".join([p for p in parts if p])[:1024]


def rerank_candidates(query: str, candidates: list[dict],
                      top_k: int = 3, doc_field: str = "summary") -> dict:
    """Rerank FAISS candidates by the BGE cross-encoder. Returns top_k + scores,
    or an explicit unavailable status with a passthrough (never a silent fake)."""
    t0 = time.time()
    reranker = _load_reranker()
    if reranker is None or reranker == "FAILED":
        return {
            "ok": False, "error": "reranker_unavailable",
            "reranked_top_k": candidates[:top_k],
            "fallback": "passthrough_top_k_from_faiss",
        }

    pairs = [[query, _doc_text(c, doc_field)] for c in candidates]
    try:
        if hasattr(reranker, "compute_score"):
            scores = reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [float(scores)]
        else:
            scores = reranker.predict(pairs).tolist()
    except Exception as e:  # noqa: BLE001
        logger.warning("[bge-rerank] scoring failed: %s", e)
        return {
            "ok": False, "error": str(e)[:200],
            "reranked_top_k": candidates[:top_k],
            "fallback": "passthrough_top_k_from_faiss",
        }

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = []
    for c, s in ranked[:top_k]:
        c2 = dict(c)
        c2["rerank_score"] = round(float(s), 4)
        top.append(c2)

    return {
        "ok": True, "model": "bge-reranker-v2-m3",
        "n_candidates_reranked": len(candidates),
        "top_k_returned": len(top), "reranked_top_k": top,
        "score_range": [round(float(min(scores)), 4),
                        round(float(max(scores)), 4)],
        "elapsed_s": round(time.time() - t0, 3),
        "device": _DEVICE,
    }


def search_and_rerank(query: str, faiss_k: int = 20, rerank_k: int = 3) -> dict:
    """Full pipeline: crisis-library FAISS top-k -> BGE rerank -> top-k."""
    try:
        from supplymind.warroom.scenarios.library_v2_search import search
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"library_v2_search_unavailable: {e}"}
    candidates = search(query, top_k=faiss_k) or []
    if not candidates:
        return {"ok": False, "error": "no_faiss_candidates"}
    # crisis events carry 'title'/'embed_text', not 'summary'
    return rerank_candidates(query, candidates, top_k=rerank_k,
                             doc_field="summary")


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = search_and_rerank(
        "Iran-Israel-US escalation restricts the Strait of Hormuz, "
        "tanker disruption, Brent spike",
        faiss_k=20, rerank_k=3,
    )
    print(json.dumps({k: v for k, v in res.items()
                      if k != "reranked_top_k"}, indent=2))
    if res.get("reranked_top_k"):
        print("\nTop 3 reranked:")
        for r in res["reranked_top_k"]:
            label = r.get("title") or r.get("event_id") or "?"
            print(f"  rerank={r.get('rerank_score', 0):.3f}  {label[:80]}")
