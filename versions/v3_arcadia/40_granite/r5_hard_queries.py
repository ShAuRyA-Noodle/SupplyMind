"""R5 Granite — HARD-QUERY REDEMPTION benchmark.

Adds 20 deliberately-hard paraphrased queries over the same 26 Wikipedia crisis
articles. The hardness is designed to create lexical gap between the query and
the gold document, so that:
  - bi-encoder retrieval is less trivially effective
  - reranker's semantic matching can earn its cost

This is the redemption story for the R5 Granite "reranker hurts" finding:
right tool for right regime.

Reuses cached corpus chunks + pre-computed embeddings from R5 main run.

Outputs:
  versions/v3_arcadia/results/R5_GRANITE_HARD.json
  versions/v3_arcadia/plots/granite/r5_hard_redemption.png
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "granite"
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "granite"
MODELS = ROOT / "models"

BGE_M3 = MODELS / "bge-m3"
MXBAI = MODELS / "mxbai-embed-large"
SNOW = MODELS / "snowflake-arctic-embed-l"
RERANKER = MODELS / "bge-reranker-v2-m3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TOP_K_RETRIEVE = 50

# ============================================================
# 20 HARD queries — lexically paraphrased, temporally framed, indirect
# ============================================================
HARD_QUERIES = [
    # Temporal + indirect (avoid gold article's key terms)
    {"q": "In early 2011, what cataclysm in Japan rendered a coastal power facility inoperable?",
     "gold": ["2011_Tōhoku_earthquake_and_tsunami"], "hardness": "temporal+indirect"},
    {"q": "Which 2020-decade phenomenon made it nearly impossible for automakers to build cars?",
     "gold": ["2020–2023_global_chip_shortage"], "hardness": "paraphrase"},
    {"q": "Why were ships delayed for nearly a week in Egypt during spring 2021?",
     "gold": ["2021_Suez_Canal_obstruction", "Ever_Given"], "hardness": "temporal+indirect"},
    {"q": "What narrow passage off Yemen's southwest shapes East-West trade routes?",
     "gold": ["Bab-el-Mandeb"], "hardness": "indirect+geographic"},
    {"q": "Which freight-rate index tracks the hiring cost for bulk cargo vessels?",
     "gold": ["Baltic_Dry_Index"], "hardness": "paraphrase"},
    {"q": "Why does a small demand change at the customer tier cause large order swings at the supplier tier?",
     "gold": ["Bullwhip_effect"], "hardness": "causal paraphrase"},
    {"q": "What US legislation aims to onshore semiconductor fabrication?",
     "gold": ["CHIPS_and_Science_Act"], "hardness": "paraphrase"},
    {"q": "How is the capacity of a large cargo vessel measured in standardized boxes?",
     "gold": ["Container_ship"], "hardness": "paraphrase"},
    {"q": "Which integrated business software replaces standalone accounting + inventory + HR tools?",
     "gold": ["Enterprise_resource_planning"], "hardness": "indirect"},
    {"q": "Who owns the 400-meter-long ship that ran aground and blocked Suez in 2021?",
     "gold": ["Ever_Given"], "hardness": "temporal+specific"},
    {"q": "Which Taiwanese contract manufacturer assembles most of Apple's hardware?",
     "gold": ["Foxconn"], "hardness": "indirect"},
    {"q": "What buffer protects a firm against stockouts when lead time is uncertain?",
     "gold": ["Inventory"], "hardness": "paraphrase+causal"},
    {"q": "Which lean production method aims to eliminate warehouse stock by synchronizing production to demand?",
     "gold": ["Just-in-time_manufacturing"], "hardness": "paraphrase"},
    {"q": "What West-Coast US harbor drew headlines for anchored ship queues in 2021?",
     "gold": ["Port_of_Los_Angeles"], "hardness": "temporal+indirect"},
    {"q": "Which Southeast Asian port is the world's busiest transshipment hub?",
     "gold": ["Port_of_Singapore"], "hardness": "indirect+geographic"},
    {"q": "Why did Houthi forces attack shipping near a narrow Arabian Peninsula strait in 2023?",
     "gold": ["Red_Sea_crisis", "Bab-el-Mandeb"], "hardness": "temporal+geopolitical"},
    {"q": "Which geographic bottleneck carries most seaborne Middle Eastern crude oil?",
     "gold": ["Strait_of_Hormuz"], "hardness": "paraphrase"},
    {"q": "Which Indonesian-Malaysian sea lane bottlenecks Asia-Europe container traffic?",
     "gold": ["Strait_of_Malacca"], "hardness": "geographic paraphrase"},
    {"q": "What is SolarWinds an example of in software delivery risk?",
     "gold": ["Supply_chain_attack"], "hardness": "indirect"},
    {"q": "Which foundry produces most advanced logic chips globally?",
     "gold": ["TSMC"], "hardness": "paraphrase"},
]


# ============================================================
# Retrieval primitives (reused)
# ============================================================

def cosine_topk(q_emb, corpus_emb, k):
    sims = corpus_emb @ q_emb
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


def rrf_fuse(ranked_lists, k_rrf=60, top_k=TOP_K_RETRIEVE):
    scores = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [(int(i), float(s)) for i, s in items]


def is_gold(chunk, gold_ids): return chunk["doc_id"] in gold_ids
def p_at_k(r, chunks, gold, k): return sum(1 for i in r[:k] if is_gold(chunks[i], gold)) / k
def r_at_k(r, chunks, gold, k):
    top_k = r[:k]
    gold_set = set(gold)
    hit_docs = {chunks[i]["doc_id"] for i in top_k if is_gold(chunks[i], gold)}
    return len(hit_docs & gold_set) / max(len(gold_set), 1)


def mrr(r, chunks, gold):
    for rank, i in enumerate(r):
        if is_gold(chunks[i], gold):
            return 1.0 / (rank + 1)
    return 0.0


def ndcg(r, chunks, gold, k):
    gains = [1.0 if is_gold(chunks[i], gold) else 0.0 for i in r[:k]]
    dcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Evaluators (bi-encoder + reranked)
# ============================================================

_RERANKER_CE = None


def get_reranker():
    global _RERANKER_CE
    if _RERANKER_CE is None:
        from sentence_transformers import CrossEncoder
        _RERANKER_CE = CrossEncoder(str(RERANKER), device=DEVICE)
        log.info("Loaded BGE-reranker-v2-m3")
    return _RERANKER_CE


def rerank(query, candidates, top_k=TOP_K_RETRIEVE):
    ce = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    try:
        scores = ce.predict(pairs, batch_size=4, show_progress_bar=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        scores = ce.predict(pairs, batch_size=2, show_progress_bar=False)
    order = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in order[:top_k]]


def aggregate(per_q):
    keys = ["p1", "p3", "p5", "r5", "r10", "mrr_score", "ndcg10", "latency_s"]
    return {k: float(np.mean([q[k] for q in per_q])) for k in keys}


def eval_pipeline(name, embedder_name, embedder, emb_cache, chunks, queries,
                   use_reranker=False, use_rrf=False, all_embedders=None):
    log.info(f"  [{name}] …")
    per_q = []
    t0 = time.time()
    for q in queries:
        tq = time.time()
        if use_rrf:
            ranked_lists = []
            for e_name, e in all_embedders.items():
                q_emb = e.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
                ranked_lists.append(cosine_topk(q_emb, emb_cache[e_name], TOP_K_RETRIEVE))
            retrieved = rrf_fuse(ranked_lists, top_k=TOP_K_RETRIEVE)
        else:
            q_emb = embedder.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
            retrieved = cosine_topk(q_emb, emb_cache[embedder_name], TOP_K_RETRIEVE)
        if use_reranker:
            cand = [chunks[i] for i, _ in retrieved]
            reranked = rerank(q["q"], cand, top_k=TOP_K_RETRIEVE)
            retrieved = [(retrieved[r_i][0], score) for r_i, score in reranked]
        ri = [i for i, _ in retrieved]
        per_q.append({
            "q": q["q"], "gold": q["gold"], "hardness": q.get("hardness"),
            "p1": p_at_k(ri, chunks, q["gold"], 1),
            "p3": p_at_k(ri, chunks, q["gold"], 3),
            "p5": p_at_k(ri, chunks, q["gold"], 5),
            "r5": r_at_k(ri, chunks, q["gold"], 5),
            "r10": r_at_k(ri, chunks, q["gold"], 10),
            "mrr_score": mrr(ri, chunks, q["gold"]),
            "ndcg10": ndcg(ri, chunks, q["gold"], 10),
            "latency_s": time.time() - tq,
        })
    agg = aggregate(per_q)
    log.info(f"    P@1={agg['p1']:.3f}  MRR={agg['mrr_score']:.3f}  nDCG@10={agg['ndcg10']:.3f}  lat={agg['latency_s']:.2f}s")
    return {"pipeline": name, "per_query": per_q, "aggregate": agg,
            "total_s": time.time() - t0}


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    log.info(f"R5 Granite HARD-QUERY REDEMPTION  ({len(HARD_QUERIES)} hard queries)")

    # Load cached chunks
    with open(CKPT / "corpus_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    log.info(f"Loaded {len(chunks)} corpus chunks from cache")

    # Load embedders
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedders on " + DEVICE)
    bge = SentenceTransformer(str(BGE_M3), device=DEVICE)
    mxbai = SentenceTransformer(str(MXBAI), device=DEVICE)
    snow = SentenceTransformer(str(SNOW), device=DEVICE, backend="torch")

    emb_cache = {
        "bge_m3":     np.load(CKPT / "corpus_emb_bge_m3.npy"),
        "mxbai":      np.load(CKPT / "corpus_emb_mxbai.npy"),
        "snowflake":  np.load(CKPT / "corpus_emb_snowflake.npy"),
    }
    log.info(f"Loaded cached corpus embeddings: {emb_cache['bge_m3'].shape}")
    torch.cuda.empty_cache()

    all_embedders = {"bge_m3": bge, "mxbai": mxbai, "snowflake": snow}
    results = []

    # 3 bi-encoders
    results.append(eval_pipeline("P1_bge_m3_bi", "bge_m3", bge, emb_cache, chunks, HARD_QUERIES))
    results.append(eval_pipeline("P2_mxbai_bi", "mxbai", mxbai, emb_cache, chunks, HARD_QUERIES))
    results.append(eval_pipeline("P3_snowflake_bi", "snowflake", snow, emb_cache, chunks, HARD_QUERIES))
    # 3 with reranker
    results.append(eval_pipeline("P4_bge_m3_rerank", "bge_m3", bge, emb_cache, chunks, HARD_QUERIES, use_reranker=True))
    results.append(eval_pipeline("P5_mxbai_rerank", "mxbai", mxbai, emb_cache, chunks, HARD_QUERIES, use_reranker=True))
    results.append(eval_pipeline("P6_snowflake_rerank", "snowflake", snow, emb_cache, chunks, HARD_QUERIES, use_reranker=True))
    # RRF ensemble (with rerank)
    results.append(eval_pipeline("P7_rrf_rerank", None, None, emb_cache, chunks, HARD_QUERIES,
                                  use_reranker=True, use_rrf=True, all_embedders=all_embedders))

    # Load easy-query results for comparison
    easy = json.loads((RESULTS / "R5_GRANITE.json").read_text())

    # Compute delta: reranker lift on hard vs easy
    deltas = {}
    for r in results:
        if "_rerank" in r["pipeline"]:
            base_name = r["pipeline"].replace("_rerank", "_bi").replace("P4", "P1").replace("P5", "P2").replace("P6", "P3")
            # Find base on hard set
            hard_base = next((x for x in results if x["pipeline"].startswith(base_name.split("_")[0] + "_") and "_bi" in x["pipeline"]), None)
            if hard_base:
                hard_lift = r["aggregate"]["p1"] - hard_base["aggregate"]["p1"]
                # Easy lift from main R5_GRANITE.json
                if base_name in easy["pipelines"] and r["pipeline"] in easy["pipelines"]:
                    easy_lift = easy["pipelines"][r["pipeline"]]["p1"] - easy["pipelines"][base_name]["p1"]
                else:
                    easy_lift = None
                deltas[r["pipeline"]] = {
                    "hard_p1_lift_vs_bi": hard_lift,
                    "easy_p1_lift_vs_bi": easy_lift,
                }

    out = {
        "n_queries": len(HARD_QUERIES),
        "n_chunks": len(chunks),
        "queries": HARD_QUERIES,
        "pipelines": {r["pipeline"]: r["aggregate"] for r in results},
        "per_pipeline_detail": {r["pipeline"]: r for r in results},
        "reranker_lift_deltas": deltas,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R5_GRANITE_HARD.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    log.info("")
    log.info("=== R5 HARD-QUERY SUMMARY (sorted by MRR) ===")
    for name, agg in sorted(out["pipelines"].items(), key=lambda x: -x[1]["mrr_score"]):
        log.info(f"  {name:<28}  P@1={agg['p1']:.3f}  MRR={agg['mrr_score']:.3f}  nDCG@10={agg['ndcg10']:.3f}")

    log.info("")
    log.info("=== RERANKER LIFT: HARD vs EASY queries ===")
    for name, d in deltas.items():
        hl = d["hard_p1_lift_vs_bi"]
        el = d["easy_p1_lift_vs_bi"]
        el_s = f"{el:+.3f}" if el is not None else "n/a"
        log.info(f"  {name:<28}  hard_lift={hl:+.3f}  easy_lift={el_s}")
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
