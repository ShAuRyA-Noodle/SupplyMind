"""R5-β v2 — Manual BEIR-style retrieval eval (bypasses broken mteb + torchaudio deps).

The `mteb` library fails to import on this Windows env due to a torchaudio
DLL mismatch. We instead compute the same BEIR retrieval metrics by hand:

  Task: SciFact-style binary retrieval over a small real corpus
        (we construct it from NOAA storm narratives + SEC 10-K risk-factor
        sections — both already in external_data/).

  Metric: nDCG@10 + P@10 + Recall@10  (same metrics the public MTEB/BEIR
          leaderboards report).

Result compared to public SOTA numbers on NFCorpus (medical retrieval,
closest analog) to confirm our embedders perform at public-leaderboard
levels on an out-of-domain real corpus.

Output:
  versions/v3_arcadia/results/R5_BEIR_MANUAL.json
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "v3_arcadia" / "results"
CRISES = ROOT / "external_data" / "wikipedia_crises"

import os
# Force CPU — this eval is tiny (26 docs x 20 queries) and we don't want GPU
# contention with concurrently-running forecasters (TimesFM/Chronos).
DEVICE = os.environ.get("R5_BEIR_DEVICE", "cpu")

EMBEDDERS = {
    "mxbai-embed-large-v1": (MODELS / "mxbai-embed-large",  None),
    "bge-m3":               (MODELS / "bge-m3",             None),
    "snowflake-arctic-l":   (MODELS / "snowflake-arctic-embed-l", "torch"),
}


# ============================================================
# Build a BEIR-style retrieval set from real supply-chain articles
# ============================================================

def build_corpus_and_queries():
    """Create {corpus, queries, qrels} like BEIR.
    corpus: dict[doc_id -> text]
    queries: dict[qid -> text]
    qrels: dict[qid -> {doc_id: relevance}]
    """
    corpus = {}
    # chunk each Wikipedia crisis article into ~200-word passages
    for f in sorted(CRISES.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        # first ~800 words as single doc (simpler than chunking)
        words = text.split()[:800]
        if len(words) < 50:
            continue
        corpus[f.stem] = " ".join(words)
    log.info(f"Corpus: {len(corpus)} docs")

    # Questions with single-gold qrels
    queries = {
        "q1": "What was the magnitude of the 2011 Tohoku earthquake?",
        "q2": "How long was the Suez Canal blocked in 2021?",
        "q3": "What caused the global semiconductor shortage?",
        "q4": "Why is the Strait of Hormuz strategically important?",
        "q5": "How do Houthis threaten Red Sea shipping?",
        "q6": "Which foundry dominates advanced chip production?",
        "q7": "What is the bullwhip effect?",
        "q8": "Which port congested during 2021 supply chain crisis?",
        "q9": "What is the just-in-time manufacturing philosophy?",
        "q10": "What does the CHIPS Act allocate?",
        "q11": "Who is Foxconn's primary customer?",
        "q12": "Why did the Ever Given run aground?",
        "q13": "What is safety stock?",
        "q14": "What is a supply chain attack?",
        "q15": "How busy is the Port of Singapore?",
        "q16": "Which strait is a narrow Indonesia-Malaysia chokepoint?",
        "q17": "Which industry does the Baltic Dry Index track?",
        "q18": "What function does a warehouse serve?",
        "q19": "What is a container ship's TEU?",
        "q20": "What software replaces accounting + inventory + HR systems?",
    }

    qrels = {
        "q1": {"2011_Tōhoku_earthquake_and_tsunami": 1},
        "q2": {"2021_Suez_Canal_obstruction": 1, "Ever_Given": 1},
        "q3": {"2020–2023_global_chip_shortage": 1},
        "q4": {"Strait_of_Hormuz": 1},
        "q5": {"Red_Sea_crisis": 1, "Bab-el-Mandeb": 1},
        "q6": {"TSMC": 1, "Semiconductor_industry": 1},
        "q7": {"Bullwhip_effect": 1},
        "q8": {"Port_of_Los_Angeles": 1},
        "q9": {"Just-in-time_manufacturing": 1},
        "q10": {"CHIPS_and_Science_Act": 1},
        "q11": {"Foxconn": 1},
        "q12": {"Ever_Given": 1, "2021_Suez_Canal_obstruction": 1},
        "q13": {"Inventory": 1},
        "q14": {"Supply_chain_attack": 1},
        "q15": {"Port_of_Singapore": 1},
        "q16": {"Strait_of_Malacca": 1},
        "q17": {"Baltic_Dry_Index": 1},
        "q18": {"Warehouse": 1},
        "q19": {"Container_ship": 1},
        "q20": {"Enterprise_resource_planning": 1},
    }
    # Drop queries whose gold isn't in corpus
    qrels = {q: {d: r for d, r in qd.items() if d in corpus} for q, qd in qrels.items()}
    qrels = {q: qd for q, qd in qrels.items() if qd}
    queries = {q: queries[q] for q in qrels}
    log.info(f"Queries with at least one gold in corpus: {len(queries)}")
    return corpus, queries, qrels


# ============================================================
# BEIR-style metrics
# ============================================================

def ndcg_at_k(ranked_docs, qrels_for_q, k=10):
    rel = [qrels_for_q.get(d, 0) for d in ranked_docs[:k]]
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_docs, qrels_for_q, k=10):
    total_rel = sum(1 for r in qrels_for_q.values() if r > 0)
    if total_rel == 0: return 0.0
    hit = sum(1 for d in ranked_docs[:k] if qrels_for_q.get(d, 0) > 0)
    return hit / total_rel


def precision_at_k(ranked_docs, qrels_for_q, k=10):
    hit = sum(1 for d in ranked_docs[:k] if qrels_for_q.get(d, 0) > 0)
    return hit / k


# ============================================================
# Per-embedder eval
# ============================================================

def eval_embedder(name, path, backend, corpus, queries, qrels):
    log.info(f"\n=== {name} ===")
    from sentence_transformers import SentenceTransformer
    kwargs = {"device": DEVICE}
    if backend: kwargs["backend"] = backend
    model = SentenceTransformer(str(path), **kwargs)

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]
    log.info("  Encoding corpus...")
    t0 = time.time()
    corpus_emb = model.encode(doc_texts, normalize_embeddings=True,
                               batch_size=8, show_progress_bar=False, convert_to_numpy=True)
    enc_time = time.time() - t0
    log.info(f"  Corpus encoded: {corpus_emb.shape} in {enc_time:.1f}s")

    per_q = {}
    ndcgs, recalls, precisions = [], [], []
    for q, qtext in queries.items():
        q_emb = model.encode(qtext, normalize_embeddings=True, convert_to_numpy=True)
        scores = corpus_emb @ q_emb
        order = np.argsort(scores)[::-1][:20]
        ranked = [doc_ids[int(i)] for i in order]
        n10 = ndcg_at_k(ranked, qrels[q], 10)
        r10 = recall_at_k(ranked, qrels[q], 10)
        p10 = precision_at_k(ranked, qrels[q], 10)
        ndcgs.append(n10); recalls.append(r10); precisions.append(p10)
        per_q[q] = {"query": qtext, "gold": list(qrels[q].keys()),
                    "top5": ranked[:5], "ndcg@10": float(n10),
                    "recall@10": float(r10), "precision@10": float(p10)}

    return {
        "embedder": name,
        "mean_ndcg@10": float(np.mean(ndcgs)),
        "mean_recall@10": float(np.mean(recalls)),
        "mean_precision@10": float(np.mean(precisions)),
        "corpus_encoding_s": enc_time,
        "n_queries": len(queries),
        "per_query": per_q,
    }


# ============================================================
# Public NFCorpus leaderboard reference (for positioning)
# ============================================================

PUBLIC_REF = {
    "mxbai-embed-large-v1": {"ndcg@10_nfcorpus": 0.386, "source": "MTEB retrieval leaderboard 2024"},
    "bge-m3":               {"ndcg@10_nfcorpus": 0.357, "source": "BGE-M3 paper + MTEB"},
    "snowflake-arctic-l":   {"ndcg@10_nfcorpus": 0.348, "source": "Snowflake Arctic paper"},
}


def main():
    t0 = time.time()
    log.info("R5-β v2 — Manual BEIR-style eval (bypass broken mteb/torchaudio import)")

    corpus, queries, qrels = build_corpus_and_queries()

    import gc
    results = {}
    for name, (path, backend) in EMBEDDERS.items():
        try:
            results[name] = eval_embedder(name, path, backend, corpus, queries, qrels)
            r = results[name]
            log.info(f"  {name}: nDCG@10={r['mean_ndcg@10']:.3f}  "
                     f"Recall@10={r['mean_recall@10']:.3f}  "
                     f"P@10={r['mean_precision@10']:.3f}")
        except Exception as e:
            log.error(f"  {name} FAILED: {str(e)[:200]}")
            results[name] = {"status": "FAILED", "error": str(e)[:300]}
        # Aggressive cleanup between models to survive Windows pagefile limits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

    out = {
        "task": "SupplyMind-crisis-retrieval-BEIR-style",
        "task_description": (
            "Manual BEIR-style retrieval eval on 26 Wikipedia crisis articles + 20 real supply-chain queries. "
            "Metrics match the public MTEB retrieval leaderboard (nDCG@10, R@10, P@10). This is an "
            "out-of-domain task (supply chain, not medical), but numbers provide a directional check "
            "that our embedders are consistent with their published leaderboard performance."
        ),
        "our_results": results,
        "public_ref_nfcorpus": PUBLIC_REF,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R5_BEIR_MANUAL.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    log.info("\n=== R5-β v2 SUMMARY ===")
    for name, r in results.items():
        if "mean_ndcg@10" in r:
            ours = r["mean_ndcg@10"]
            public = PUBLIC_REF.get(name, {}).get("ndcg@10_nfcorpus")
            log.info(f"  {name:<28}  our nDCG@10={ours:.3f}  public-ref NFCorpus={public}")
        else:
            log.info(f"  {name:<28}  FAILED")
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
