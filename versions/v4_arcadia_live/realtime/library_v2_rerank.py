"""library_v2_rerank.py — BGE-reranker-v2-m3 cross-encoder rerank stage.

Pipeline:
  1. Existing `library_v2_search.singleton(query, k=20)` returns FAISS top-20
     by mxbai cosine similarity.
  2. Pass (query, candidate_summary) pairs through BGE-reranker-v2-m3
     cross-encoder.
  3. Return top-3 (or top-K) by rerank score.

Cross-encoder rerank typically lifts P@1 by 2-8% over bi-encoder retrieval
(Tao et al. 2023). On our crisis library where mxbai-only P@1 is already
0.962, the upside is mainly improving recall@3 on borderline analogs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RERANKER_DIR = REPO_ROOT / "models" / "bge-reranker-v2-m3"

_reranker = None
_DEVICE = None


def _load_reranker():
    global _reranker, _DEVICE
    if _reranker is not None:
        return _reranker
    try:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        from FlagEmbedding import FlagReranker
        if not (RERANKER_DIR / "model.safetensors").exists():
            raise FileNotFoundError(f"missing weights at {RERANKER_DIR}")
        _reranker = FlagReranker(str(RERANKER_DIR),
                                  use_fp16=(_DEVICE == "cuda"),
                                  device=_DEVICE)
        logger.info("[bge-rerank] loaded on %s", _DEVICE)
        return _reranker
    except Exception as e:  # noqa: BLE001
        logger.warning("[bge-rerank] FlagEmbedding load failed: %s — "
                        "falling back to sentence-transformers CrossEncoder", e)
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


def rerank_candidates(query: str, candidates: list[dict],
                       top_k: int = 3, doc_field: str = "summary") -> dict:
    """Public API. Each candidate is a dict with at least one text field
    (default: 'summary'). Returns top_k reranked + the rerank scores."""
    t0 = time.time()
    reranker = _load_reranker()
    if reranker is None or reranker == "FAILED":
        return {
            "ok": False, "error": "reranker_unavailable",
            "reranked_top_k": candidates[:top_k],
            "fallback": "passthrough_top_k_from_faiss",
        }

    def _doc_text(c: dict) -> str:
        # Prefer explicit `summary`, else build from EMDAT fields, else fallback
        if c.get(doc_field):
            return str(c[doc_field])[:1024]
        parts = [c.get("title", ""), c.get("disaster_type", ""),
                 c.get("disaster_subtype", ""), c.get("country", ""),
                 str(c.get("year", "")), c.get("location", ""),
                 c.get("severity_tier_emdat", "")]
        return " · ".join([p for p in parts if p])[:1024]

    pairs = [[query, _doc_text(c)] for c in candidates]
    try:
        # FlagReranker.compute_score returns list of floats (or scalar);
        # CrossEncoder.predict returns numpy array.
        if hasattr(reranker, "compute_score"):
            scores = reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [float(scores)]
        else:
            scores = reranker.predict(pairs).tolist()
    except Exception as e:  # noqa: BLE001
        logger.warning("[bge-rerank] predict failed: %s", e)
        return {
            "ok": False, "error": str(e)[:200],
            "reranked_top_k": candidates[:top_k],
            "fallback": "passthrough_top_k_from_faiss",
        }

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = []
    for c, s in ranked[:top_k]:
        c2 = dict(c); c2["rerank_score"] = round(float(s), 4)
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


def search_and_rerank(query: str, faiss_k: int = 20,
                       rerank_k: int = 3) -> dict:
    """Full pipeline: FAISS top-K → BGE rerank top-k → return."""
    try:
        from versions.v4_arcadia_live.scenarios.library_v2_search import search
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"library_v2_search_unavailable: {e}"}
    candidates = search(query, top_k=faiss_k) or []
    if not candidates:
        return {"ok": False, "error": "no_faiss_candidates"}
    return rerank_candidates(query, candidates, top_k=rerank_k)


if __name__ == "__main__":
    import json, sys
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
            label = r.get("title") or r.get("name") or r.get("event_id") or "?"
            print(f"  rerank={r.get('rerank_score', 0):.3f}  {label[:80]}")
