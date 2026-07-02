"""Re-cook the RAG corpus with World-Bank data actually ingested (R17).

Background
----------
Every RAG corpus SupplyMind ever built indexed ZERO World-Bank chunks. The WB
Open Data API returns a top-level list ``[meta, rows]``, but the ingestion code
assumed a dict (``d.items() if isinstance(d, dict) else []``), so it silently
produced 0 WB chunks — visible in the committed ``R5_GRANITE.json`` as
``"world_bank": 0``. The list-vs-dict bug is fixed in
``versions/v3_arcadia/40_granite/r5_rag_beast.py`` (``wb_json_to_docs``).

What this script does
---------------------
1. Rebuild the corpus via the FIXED ``load_corpus`` (WB now ingested).
2. Embed every chunk once with the single mxbai-embed-large embedder
   (``models/mxbai-embed-large``) — the P@1 winner from the RAG benchmark.
3. Reproduce the mxbai bi-encoder retrieval pipeline (r5's "P2").
4. Measure BEFORE vs AFTER honestly, with a single embedding pass:
     - BEFORE = the buggy corpus, reproduced by masking WB chunk indices out
       of the candidate pool (non-WB chunks + their embeddings are identical
       to the old corpus, so masking is faithful).
     - AFTER  = the full re-cooked corpus.
   Two eval sets:
     (a) the 60 existing crisis queries (gold = wiki/SEC docs) -> robustness:
         does adding WB distractor docs degrade the existing retrieval?
     (b) 8 WB-targeted queries (gold = the WB docs) -> capability:
         WB answers are un-retrievable BEFORE (0 chunks indexed) and become
         retrievable AFTER.
5. Write ``tests/receipts/rag_recook_REAL.json`` and a small WB manifest.

Usage
-----
    python scripts/crisis_library/recook_rag_wb.py
    python scripts/crisis_library/recook_rag_wb.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R5_PATH = REPO_ROOT / "versions" / "v3_arcadia" / "40_granite" / "r5_rag_beast.py"
MXBAI_DIR = REPO_ROOT / "models" / "mxbai-embed-large"
RECEIPT_PATH = REPO_ROOT / "tests" / "receipts" / "rag_recook_REAL.json"
WB_MANIFEST_PATH = REPO_ROOT / "scripts" / "crisis_library" / "rag_corpus_wb_manifest.json"


def _load_r5():
    spec = importlib.util.spec_from_file_location("r5_rag_beast", R5_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# World-Bank-targeted queries. Gold doc_ids follow the "{stem}__{iso3}" scheme
# emitted by wb_json_to_docs. Each answer genuinely lives in the cited WB doc.
WB_QUERIES = [
    {"q": "What is China's GDP in current US dollars?", "gold": ["wb_GDP_USD__CHN"]},
    {"q": "What is India's GDP in current US dollars?", "gold": ["wb_GDP_USD__IND"]},
    {"q": "What is India's annual GDP growth rate?", "gold": ["wb_GDP_growth__IND"]},
    {"q": "What is Germany's GDP growth rate?", "gold": ["wb_GDP_growth__DEU"]},
    {"q": "What is Germany's consumer price inflation rate?", "gold": ["wb_Inflation_CPI__DEU"]},
    {"q": "What are China's exports as a percentage of GDP?", "gold": ["wb_Exports_pct_GDP__CHN"]},
    {"q": "What are India's imports as a share of GDP?", "gold": ["wb_Imports_pct_GDP__IND"]},
    {"q": "What is China's container port traffic in TEU?", "gold": ["wb_Container_throughput__CHN"]},
]


def _eval_queries(r5, queries, chunks, corpus_emb, embedder, allowed_mask):
    """mxbai bi-encoder retrieval over an allowed candidate subset.

    ``allowed_mask`` is a boolean array over chunks; disallowed chunks are
    removed from the candidate pool (used to reproduce the pre-fix corpus).
    Returns aggregate P@1/P@3/P@5/MRR/nDCG@10 plus per-query hits.
    """
    allowed_idx = np.where(allowed_mask)[0]
    sub_emb = corpus_emb[allowed_idx]
    per_q = []
    for q in queries:
        q_emb = embedder.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
        local = r5.cosine_topk(q_emb, sub_emb, k=r5.TOP_K_RETRIEVE)
        ranked_idx = [int(allowed_idx[i]) for i, _ in local]  # map back to global idx
        per_q.append({
            "q": q["q"],
            "gold": q["gold"],
            "top1_doc": chunks[ranked_idx[0]]["doc_id"] if ranked_idx else None,
            "p1": r5.precision_at_k(ranked_idx, chunks, q["gold"], 1),
            "p3": r5.precision_at_k(ranked_idx, chunks, q["gold"], 3),
            "p5": r5.precision_at_k(ranked_idx, chunks, q["gold"], 5),
            "mrr": r5.mrr(ranked_idx, chunks, q["gold"]),
            "ndcg10": r5.ndcg_at_k(ranked_idx, chunks, q["gold"], 10),
        })
    keys = ["p1", "p3", "p5", "mrr", "ndcg10"]
    agg = {k: float(np.mean([q[k] for q in per_q])) for k in keys}
    return agg, per_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write full corpus artifacts (default: scratchpad-free tmp)")
    args = parser.parse_args()

    t0 = time.time()
    r5 = _load_r5()

    if not MXBAI_DIR.exists():
        raise FileNotFoundError(f"mxbai embedder not found at {MXBAI_DIR}")

    # 1. Rebuild corpus with the FIXED WB ingestion.
    chunks = r5.load_corpus()
    from collections import Counter
    src_counts = dict(Counter(c["source"] for c in chunks))
    wb_mask = np.array([c["source"] == "world_bank" for c in chunks])
    non_wb_mask = ~wb_mask
    n_wb = int(wb_mask.sum())
    print(f"[recook] corpus rebuilt: {len(chunks)} chunks  sources={src_counts}")
    if n_wb == 0:
        raise RuntimeError("World-Bank chunks are still 0 — the fix did not take effect")

    # 2. Embed once with single mxbai embedder.
    from sentence_transformers import SentenceTransformer
    print(f"[recook] loading mxbai embedder from {MXBAI_DIR} ...")
    embedder = SentenceTransformer(str(MXBAI_DIR), device=r5.DEVICE)
    texts = [c["text"] for c in chunks]
    print(f"[recook] embedding {len(texts)} chunks on {r5.DEVICE} ...")
    corpus_emb = embedder.encode(
        texts, normalize_embeddings=True, batch_size=32,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")

    # 3+4. BEFORE (mask WB out) vs AFTER (full corpus), on both eval sets.
    full_mask = np.ones(len(chunks), dtype=bool)
    crisis_before, _ = _eval_queries(r5, r5.QUERIES, chunks, corpus_emb, embedder, non_wb_mask)
    crisis_after, crisis_after_pq = _eval_queries(r5, r5.QUERIES, chunks, corpus_emb, embedder, full_mask)
    wb_before, wb_before_pq = _eval_queries(r5, WB_QUERIES, chunks, corpus_emb, embedder, non_wb_mask)
    wb_after, wb_after_pq = _eval_queries(r5, WB_QUERIES, chunks, corpus_emb, embedder, full_mask)

    print(f"\n[recook] crisis queries (n={len(r5.QUERIES)}) — robustness to WB distractors:")
    print(f"  P@1  before={crisis_before['p1']:.4f}  after={crisis_after['p1']:.4f}  "
          f"delta={crisis_after['p1']-crisis_before['p1']:+.4f}")
    print(f"  MRR  before={crisis_before['mrr']:.4f}  after={crisis_after['mrr']:.4f}  "
          f"delta={crisis_after['mrr']-crisis_before['mrr']:+.4f}")
    print(f"[recook] WB-targeted queries (n={len(WB_QUERIES)}) — capability:")
    print(f"  P@1  before={wb_before['p1']:.4f}  after={wb_after['p1']:.4f}  "
          f"delta={wb_after['p1']-wb_before['p1']:+.4f}")
    print(f"  MRR  before={wb_before['mrr']:.4f}  after={wb_after['mrr']:.4f}  "
          f"delta={wb_after['mrr']-wb_before['mrr']:+.4f}")

    # 5. Persist full artifacts (large -> out-dir), WB manifest + receipt (repo).
    out_dir = args.out_dir or (REPO_ROOT / "scripts" / "crisis_library" / "_recook_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mxbai_corpus_emb.npy", corpus_emb)
    import faiss
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    faiss.write_index(index, str(out_dir / "corpus_wb.faiss"))
    (out_dir / "corpus_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[recook] full artifacts written to {out_dir}")

    wb_docs = {}
    for c in chunks:
        if c["source"] == "world_bank":
            wb_docs.setdefault(c["doc_id"], c["text"])
    WB_MANIFEST_PATH.write_text(
        json.dumps({"n_wb_docs": len(wb_docs), "wb_documents": wb_docs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    receipt = {
        "task": "rag_recook_world_bank_ingestion",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha,
        "bug_fixed": {
            "file": "versions/v3_arcadia/40_granite/r5_rag_beast.py",
            "symptom": "World Bank API JSON is a top-level list [meta, rows]; loader assumed dict -> 0 WB chunks",
            "evidence_before": "versions/v3_arcadia/results/R5_GRANITE.json -> corpus_breakdown.world_bank == 0",
        },
        "embedder": "mxbai-embed-large (single embedder, local models/mxbai-embed-large)",
        "device": r5.DEVICE,
        "corpus": {
            "n_chunks_total": len(chunks),
            "source_breakdown": src_counts,
            "world_bank_chunks_before": 0,
            "world_bank_chunks_after": n_wb,
            "world_bank_docs": sorted(wb_docs.keys()),
        },
        "retrieval_crisis_queries": {
            "n_queries": len(r5.QUERIES),
            "note": "gold = wiki/SEC docs; measures whether WB distractors degrade existing retrieval",
            "before_wb0": crisis_before,
            "after_wb": crisis_after,
            "p1_delta": round(crisis_after["p1"] - crisis_before["p1"], 5),
            "mrr_delta": round(crisis_after["mrr"] - crisis_before["mrr"], 5),
        },
        "retrieval_wb_queries": {
            "n_queries": len(WB_QUERIES),
            "note": "gold = WB docs; before the fix these are un-retrievable (0 WB chunks indexed)",
            "before_wb0": wb_before,
            "after_wb": wb_after,
            "p1_delta": round(wb_after["p1"] - wb_before["p1"], 5),
            "mrr_delta": round(wb_after["mrr"] - wb_before["mrr"], 5),
            "per_query_after": [
                {"q": q["q"], "gold": q["gold"][0], "top1_doc": q["top1_doc"], "p1": q["p1"]}
                for q in wb_after_pq
            ],
        },
        "artifacts_dir": str(out_dir),
        "elapsed_s": round(time.time() - t0, 2),
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[recook] receipt written: {RECEIPT_PATH}")
    print(f"[recook] WB manifest written: {WB_MANIFEST_PATH}")
    print(f"[recook] done in {receipt['elapsed_s']}s")


if __name__ == "__main__":
    main()
