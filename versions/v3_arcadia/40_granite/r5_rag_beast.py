"""R5 Granite — BEAST mode RAG SOTA benchmark.

Pipelines (8 configurations):
  P1: BGE-M3 bi-encoder only
  P2: mxbai-embed-large bi-encoder only
  P3: Snowflake-Arctic-Embed-L bi-encoder only
  P4: BGE-M3 + BGE-reranker-v2-m3
  P5: mxbai + BGE-reranker-v2-m3
  P6: Snowflake + BGE-reranker-v2-m3
  P7: RRF ensemble (all 3 encoders) + reranker
  P8: HyDE (Qwen-14B hypothetical answer) + RRF + reranker

Metrics per pipeline: P@1/3/5, Recall@5/10, MRR, nDCG@10, mean latency.

Corpus: 26 Wikipedia crisis articles + 20 SEC 10K + policy papers + World Bank macro
Queries: 2-3 per crisis article, gold-labeled to source chunks (60-78 queries)

Outputs: R5_GRANITE.json + 5 plots + markdown report
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
EXT = ROOT / "external_data"
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "granite"
CKPT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "granite"
PLOTS.mkdir(parents=True, exist_ok=True)

BGE_M3 = MODELS / "bge-m3"
MXBAI = MODELS / "mxbai-embed-large"
SNOW = MODELS / "snowflake-arctic-embed-l"
RERANKER = MODELS / "bge-reranker-v2-m3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

CHUNK_WORDS = 256
OVERLAP_WORDS = 32
MIN_CHUNK_WORDS = 30
TOP_K_RETRIEVE = 50
TOP_K_RERANK = 10

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
HYDE_MODEL = "qwen25-14b-local"


# ============================================================
# Corpus construction
# ============================================================
def chunk_text(text: str, source: str, doc_id: str) -> list[dict]:
    words = text.split()
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for i in range(0, len(words), step):
        seg = words[i:i + CHUNK_WORDS]
        if len(seg) < MIN_CHUNK_WORDS: continue
        chunks.append({"source": source, "doc_id": doc_id, "chunk_idx": len(chunks),
                        "text": " ".join(seg)})
    return chunks


def html_to_text(html: str) -> str:
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception as e:
        log.warning(f"  pdf read fail {path.name}: {str(e)[:80]}")
        return ""


def load_corpus() -> list[dict]:
    chunks = []
    # Wikipedia crisis articles
    for f in sorted((EXT / "wikipedia_crises").glob("*.txt")):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(txt, "wiki_crisis", f.stem))
    wiki_n = len(chunks)
    # SEC 10K (HTML -> text)
    sec_dir = EXT / "sec_10k"
    if sec_dir.exists():
        for f in sorted(sec_dir.glob("*.html"))[:25]:
            html = f.read_text(encoding="utf-8", errors="ignore")
            txt = html_to_text(html)
            chunks.extend(chunk_text(txt, "sec_10k", f.stem))
    sec_n = len(chunks) - wiki_n
    # Policy papers (PDF -> text)
    pol_dir = EXT / "policy_papers"
    if pol_dir.exists():
        for f in sorted(pol_dir.glob("*.pdf")):
            txt = pdf_to_text(f)
            if txt:
                chunks.extend(chunk_text(txt, "policy", f.stem))
    pol_n = len(chunks) - wiki_n - sec_n
    # World Bank macro (JSON -> concatenated key-value text)
    wb_dir = EXT / "world_bank_macro"
    if wb_dir.exists():
        for f in sorted(wb_dir.glob("*.json"))[:6]:
            try:
                d = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                lines = [f"{k}: {v}" for k, v in (d.items() if isinstance(d, dict) else [])]
                txt = f.stem + "\n" + "\n".join(lines[:200])
                chunks.extend(chunk_text(txt, "world_bank", f.stem))
            except Exception:
                pass
    wb_n = len(chunks) - wiki_n - sec_n - pol_n
    log.info(f"Corpus: {len(chunks)} chunks (wiki={wiki_n}, sec={sec_n}, policy={pol_n}, wb={wb_n}) "
             f"from {len(set(c['doc_id'] for c in chunks))} docs")
    return chunks


# ============================================================
# Embedder loading (cached singletons)
# ============================================================
_EMBEDDERS = {}


def get_embedder(path: Path, name: str, backend: str = None):
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer
        kwargs = {"device": DEVICE}
        if backend: kwargs["backend"] = backend
        _EMBEDDERS[name] = SentenceTransformer(str(path), **kwargs)
        log.info(f"Loaded {name} on {DEVICE}")
    return _EMBEDDERS[name]


def embed_corpus(chunks: list[dict], embedder_name: str, embedder) -> np.ndarray:
    cache = CKPT / f"corpus_emb_{embedder_name}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(chunks):
            log.info(f"Loaded cached {embedder_name} embeddings: {emb.shape}")
            return emb
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, normalize_embeddings=True, batch_size=16,
                           show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, emb)
    log.info(f"Embedded {embedder_name}: {emb.shape}")
    return emb


# ============================================================
# Reranker
# ============================================================
_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(str(RERANKER), device=DEVICE)
        log.info(f"Loaded BGE-reranker-v2-m3 on {DEVICE}")
    return _RERANKER


def rerank(query: str, chunk_candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[tuple[int, float]]:
    ce = get_reranker()
    pairs = [(query, c["text"]) for c in chunk_candidates]
    try:
        scores = ce.predict(pairs, batch_size=4, show_progress_bar=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        scores = ce.predict(pairs, batch_size=2, show_progress_bar=False)
    order = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in order[:top_k]]


# ============================================================
# Retrieval primitives
# ============================================================
def cosine_topk(q_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = corpus_emb @ q_emb
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], k_rrf: int = 60, top_k: int = TOP_K_RETRIEVE) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.
    Each list is [(chunk_idx, score), ...] already sorted.
    RRF score = sum_i 1/(k_rrf + rank_i) across lists.
    """
    scores = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [(int(i), float(s)) for i, s in items]


# ============================================================
# HyDE: Qwen-14B generates hypothetical answer for retrieval
# ============================================================
def hyde_generate(query: str, timeout: int = 120) -> str:
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": HYDE_MODEL,
            "messages": [{"role": "system", "content":
                          "Write a single 2-3 sentence factual answer as if you read the source document. "
                          "No hedging, no 'I think'. Pure factual prose."},
                         {"role": "user", "content": f"Question: {query}\n\nFactual answer:"}],
            "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0.3, "num_predict": 200, "num_ctx": 4096}
        }, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        log.warning(f"  hyde fail: {str(e)[:80]}")
        return query  # fallback to original query


def precompute_hyde_cache(queries: list[dict]) -> dict:
    """Generate HyDE answers for ALL queries upfront so Qwen-14B gets full VRAM."""
    cache_path = CKPT / "hyde_cache.json"
    if cache_path.exists():
        log.info(f"HyDE cache found: {cache_path.name}")
        return json.loads(cache_path.read_text())
    log.info(f"Precomputing HyDE answers for {len(queries)} queries (Qwen-14B via Ollama)")
    out = {}
    t0 = time.time()
    for i, q in enumerate(queries, 1):
        ans = hyde_generate(q["q"])
        out[q["q"]] = ans
        log.info(f"  HyDE [{i}/{len(queries)}] ({time.time()-t0:.1f}s) {q['q'][:60]}...")
    cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"HyDE cache saved: {cache_path.name}  ({time.time()-t0:.1f}s total)")
    return out


# Unload Ollama model to free VRAM before loading embedders
def unload_ollama(model: str):
    try:
        requests.post(OLLAMA_URL, json={
            "model": model, "messages": [{"role": "user", "content": "."}],
            "stream": False, "keep_alive": 0, "options": {"num_predict": 1}
        }, timeout=60)
    except Exception:
        pass
    time.sleep(3)


# ============================================================
# Query set (derived from crisis articles, gold-labeled)
# Format: {"query": str, "gold_doc_ids": [doc_id, ...]}
# ============================================================
QUERIES = [
    # Tohoku
    {"q": "What was the magnitude of the 2011 Tohoku earthquake?", "gold": ["2011_Tōhoku_earthquake_and_tsunami"]},
    {"q": "How many people died in the 2011 Tohoku tsunami?", "gold": ["2011_Tōhoku_earthquake_and_tsunami"]},
    {"q": "What nuclear facility was damaged by the 2011 Tohoku tsunami?", "gold": ["2011_Tōhoku_earthquake_and_tsunami"]},
    # Chip shortage
    {"q": "What caused the 2020-2023 global chip shortage?", "gold": ["2020–2023_global_chip_shortage"]},
    {"q": "Which industries were hit hardest by the chip shortage?", "gold": ["2020–2023_global_chip_shortage"]},
    # Suez 2021
    {"q": "What ship blocked the Suez Canal in March 2021?", "gold": ["2021_Suez_Canal_obstruction", "Ever_Given"]},
    {"q": "How long was the Suez Canal blocked by Ever Given?", "gold": ["2021_Suez_Canal_obstruction", "Ever_Given"]},
    {"q": "What was the economic impact of the 2021 Suez Canal obstruction?", "gold": ["2021_Suez_Canal_obstruction"]},
    # Bab-el-Mandeb
    {"q": "What is the strategic importance of the Bab-el-Mandeb strait?", "gold": ["Bab-el-Mandeb"]},
    {"q": "How much maritime trade passes through Bab-el-Mandeb?", "gold": ["Bab-el-Mandeb"]},
    # Baltic Dry Index
    {"q": "What does the Baltic Dry Index measure?", "gold": ["Baltic_Dry_Index"]},
    {"q": "Who publishes the Baltic Dry Index?", "gold": ["Baltic_Dry_Index"]},
    # Bullwhip
    {"q": "What is the bullwhip effect in supply chains?", "gold": ["Bullwhip_effect"]},
    {"q": "What causes demand amplification in multi-tier supply chains?", "gold": ["Bullwhip_effect"]},
    # CHIPS Act
    {"q": "What is the CHIPS and Science Act?", "gold": ["CHIPS_and_Science_Act"]},
    {"q": "How much does the CHIPS Act allocate for semiconductor manufacturing?", "gold": ["CHIPS_and_Science_Act"]},
    # Container ship
    {"q": "What is TEU in container shipping?", "gold": ["Container_ship"]},
    {"q": "What is the largest container ship?", "gold": ["Container_ship"]},
    # ERP
    {"q": "What does an ERP system do?", "gold": ["Enterprise_resource_planning"]},
    {"q": "Which vendors dominate the ERP software market?", "gold": ["Enterprise_resource_planning"]},
    # Ever Given
    {"q": "Who owns the Ever Given ship?", "gold": ["Ever_Given"]},
    {"q": "What is the length of the Ever Given container ship?", "gold": ["Ever_Given"]},
    # Foxconn
    {"q": "Who founded Foxconn?", "gold": ["Foxconn"]},
    {"q": "What products does Foxconn manufacture?", "gold": ["Foxconn"]},
    # Inventory
    {"q": "What is safety stock in inventory management?", "gold": ["Inventory"]},
    {"q": "What is the difference between perpetual and periodic inventory?", "gold": ["Inventory"]},
    # JIT
    {"q": "What is just-in-time manufacturing?", "gold": ["Just-in-time_manufacturing"]},
    {"q": "Who developed just-in-time manufacturing?", "gold": ["Just-in-time_manufacturing"]},
    # Logistics
    {"q": "What are the main functions of logistics?", "gold": ["Logistics"]},
    {"q": "What is the difference between logistics and supply chain management?", "gold": ["Logistics", "Supply_chain_management"]},
    # Port of LA
    {"q": "What is the ranking of the Port of Los Angeles by container volume?", "gold": ["Port_of_Los_Angeles"]},
    {"q": "What caused congestion at the Port of Los Angeles in 2021?", "gold": ["Port_of_Los_Angeles"]},
    # Port of Singapore
    {"q": "What makes the Port of Singapore a transshipment hub?", "gold": ["Port_of_Singapore"]},
    {"q": "How many containers does the Port of Singapore handle per year?", "gold": ["Port_of_Singapore"]},
    # Red Sea crisis
    {"q": "What is the 2023-2024 Red Sea crisis?", "gold": ["Red_Sea_crisis"]},
    {"q": "Which group has attacked ships in the Red Sea?", "gold": ["Red_Sea_crisis"]},
    # Samsung
    {"q": "What is Samsung Electronics' role in semiconductors?", "gold": ["Samsung_Electronics"]},
    {"q": "Where are Samsung's main semiconductor fabs located?", "gold": ["Samsung_Electronics"]},
    # Semi industry
    {"q": "How does semiconductor manufacturing work at the foundry level?", "gold": ["Semiconductor_industry", "TSMC"]},
    {"q": "What are the leading semiconductor companies by revenue?", "gold": ["Semiconductor_industry"]},
    # Hormuz
    {"q": "What percentage of oil shipments pass through the Strait of Hormuz?", "gold": ["Strait_of_Hormuz"]},
    {"q": "Why is the Strait of Hormuz a geopolitical chokepoint?", "gold": ["Strait_of_Hormuz"]},
    # Malacca
    {"q": "What is the strategic significance of the Strait of Malacca?", "gold": ["Strait_of_Malacca"]},
    {"q": "What volume of trade passes through the Malacca Strait?", "gold": ["Strait_of_Malacca"]},
    # Suez
    {"q": "When was the Suez Canal built?", "gold": ["Suez_Canal"]},
    {"q": "How many ships transit the Suez Canal annually?", "gold": ["Suez_Canal"]},
    # Supply chain attack
    {"q": "What is the SolarWinds supply chain attack?", "gold": ["Supply_chain_attack"]},
    {"q": "What are common mitigations for software supply chain attacks?", "gold": ["Supply_chain_attack"]},
    # Supply chain mgmt
    {"q": "What are the key processes in supply chain management?", "gold": ["Supply_chain_management"]},
    # TSMC
    {"q": "What percentage of the world's advanced chips does TSMC produce?", "gold": ["TSMC"]},
    {"q": "Where are TSMC's main fabrication plants?", "gold": ["TSMC"]},
    # Warehouse
    {"q": "What is the difference between a warehouse and a distribution center?", "gold": ["Warehouse"]},
    {"q": "What does ASRS stand for in warehousing?", "gold": ["Warehouse"]},
]


# ============================================================
# Metrics
# ============================================================
def is_gold(chunk: dict, gold_doc_ids: list[str]) -> bool:
    return chunk["doc_id"] in gold_doc_ids


def precision_at_k(retrieved: list[int], chunks: list[dict], gold_doc_ids: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for i in top_k if is_gold(chunks[i], gold_doc_ids))
    return hits / k


def recall_at_k(retrieved: list[int], chunks: list[dict], gold_doc_ids: list[str], k: int) -> float:
    # Set-level: did we hit any gold doc in top-k?
    top_k = retrieved[:k]
    gold_set = set(gold_doc_ids)
    hit_docs = {chunks[i]["doc_id"] for i in top_k if is_gold(chunks[i], gold_doc_ids)}
    return len(hit_docs & gold_set) / len(gold_set) if gold_set else 0.0


def mrr(retrieved: list[int], chunks: list[dict], gold_doc_ids: list[str]) -> float:
    for rank, i in enumerate(retrieved):
        if is_gold(chunks[i], gold_doc_ids):
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(retrieved: list[int], chunks: list[dict], gold_doc_ids: list[str], k: int) -> float:
    gains = [1.0 if is_gold(chunks[i], gold_doc_ids) else 0.0 for i in retrieved[:k]]
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    # Ideal: sort gold hits first
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Pipeline evaluators
# ============================================================
def eval_bi_encoder(pipeline_name: str, emb_cache: dict, chunks: list[dict],
                     queries: list[dict], embedder_name: str, embedder,
                     use_reranker: bool = False) -> dict:
    """Single-encoder bi-encoder retrieval, optionally with cross-encoder reranker."""
    log.info(f"\n=== {pipeline_name} ===")
    per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        tq = time.time()
        q_emb = embedder.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
        retrieved = cosine_topk(q_emb, emb_cache[embedder_name], k=TOP_K_RETRIEVE)
        if use_reranker:
            candidates = [chunks[i] for i, _ in retrieved]
            reranked = rerank(q["q"], candidates, top_k=TOP_K_RETRIEVE)
            retrieved = [(retrieved[r_i][0], score) for r_i, score in reranked]
        ranked_idx = [i for i, _ in retrieved]
        per_q.append({
            "q": q["q"],
            "gold": q["gold"],
            "p1": precision_at_k(ranked_idx, chunks, q["gold"], 1),
            "p3": precision_at_k(ranked_idx, chunks, q["gold"], 3),
            "p5": precision_at_k(ranked_idx, chunks, q["gold"], 5),
            "r5": recall_at_k(ranked_idx, chunks, q["gold"], 5),
            "r10": recall_at_k(ranked_idx, chunks, q["gold"], 10),
            "mrr": mrr(ranked_idx, chunks, q["gold"]),
            "ndcg10": ndcg_at_k(ranked_idx, chunks, q["gold"], 10),
            "latency_s": time.time() - tq,
        })
    agg = aggregate(per_q)
    agg["total_s"] = time.time() - t0
    log.info(f"  P@1={agg['p1']:.3f}  P@3={agg['p3']:.3f}  MRR={agg['mrr']:.3f}  "
             f"nDCG@10={agg['ndcg10']:.3f}  total={agg['total_s']:.1f}s")
    return {"pipeline": pipeline_name, "per_query": per_q, "aggregate": agg}


def eval_rrf_ensemble(pipeline_name: str, emb_cache: dict, chunks: list[dict],
                      queries: list[dict], embedders: dict,
                      use_hyde: bool = False, use_reranker: bool = True,
                      hyde_cache: dict | None = None) -> dict:
    log.info(f"\n=== {pipeline_name} ===")
    per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        tq = time.time()
        query_str = q["q"]
        if use_hyde:
            hyde_text = (hyde_cache or {}).get(q["q"], "") or ""
            if hyde_text and hyde_text != q["q"]:
                query_str = hyde_text + "\n\n" + q["q"]  # augment
        # Retrieve top-K per encoder
        ranked_lists = []
        for name, emb in embedders.items():
            q_emb = emb.encode(query_str, normalize_embeddings=True, convert_to_numpy=True)
            ranked_lists.append(cosine_topk(q_emb, emb_cache[name], k=TOP_K_RETRIEVE))
        fused = rrf_fuse(ranked_lists, top_k=TOP_K_RETRIEVE)
        if use_reranker:
            candidates = [chunks[i] for i, _ in fused]
            reranked = rerank(q["q"], candidates, top_k=TOP_K_RETRIEVE)
            fused = [(fused[r_i][0], score) for r_i, score in reranked]
        ranked_idx = [i for i, _ in fused]
        per_q.append({
            "q": q["q"], "gold": q["gold"],
            "p1": precision_at_k(ranked_idx, chunks, q["gold"], 1),
            "p3": precision_at_k(ranked_idx, chunks, q["gold"], 3),
            "p5": precision_at_k(ranked_idx, chunks, q["gold"], 5),
            "r5": recall_at_k(ranked_idx, chunks, q["gold"], 5),
            "r10": recall_at_k(ranked_idx, chunks, q["gold"], 10),
            "mrr": mrr(ranked_idx, chunks, q["gold"]),
            "ndcg10": ndcg_at_k(ranked_idx, chunks, q["gold"], 10),
            "latency_s": time.time() - tq,
        })
    agg = aggregate(per_q)
    agg["total_s"] = time.time() - t0
    log.info(f"  P@1={agg['p1']:.3f}  P@3={agg['p3']:.3f}  MRR={agg['mrr']:.3f}  "
             f"nDCG@10={agg['ndcg10']:.3f}  total={agg['total_s']:.1f}s")
    return {"pipeline": pipeline_name, "per_query": per_q, "aggregate": agg}


def aggregate(per_q: list[dict]) -> dict:
    keys = ["p1", "p3", "p5", "r5", "r10", "mrr", "ndcg10", "latency_s"]
    return {k: float(np.mean([q[k] for q in per_q])) for k in keys}


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    log.info("R5 Granite — BEAST mode RAG SOTA benchmark")

    chunks = load_corpus()
    chunks_cache = CKPT / "corpus_chunks.pkl"
    with open(chunks_cache, "wb") as f: pickle.dump(chunks, f)

    # Phase 0: precompute HyDE answers BEFORE loading embedders (Qwen-14B needs full VRAM)
    hyde_cache = precompute_hyde_cache(QUERIES)
    unload_ollama(HYDE_MODEL)

    # Load all 3 embedders
    bge = get_embedder(BGE_M3, "bge_m3")
    mxbai = get_embedder(MXBAI, "mxbai")
    snow = get_embedder(SNOW, "snowflake", backend="torch")

    # Embed corpus (cached)
    emb_cache = {
        "bge_m3": embed_corpus(chunks, "bge_m3", bge),
        "mxbai": embed_corpus(chunks, "mxbai", mxbai),
        "snowflake": embed_corpus(chunks, "snowflake", snow),
    }
    # Free embedder VRAM so reranker fits (queries are embedded one at a time later -> cheap)
    # Keep embedders accessible for single-query encode
    torch.cuda.empty_cache()
    log.info(f"After corpus embed: VRAM used = {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Evaluate 8 pipelines
    results = []

    results.append(eval_bi_encoder("P1_bge_m3_bi", emb_cache, chunks, QUERIES, "bge_m3", bge))
    results.append(eval_bi_encoder("P2_mxbai_bi", emb_cache, chunks, QUERIES, "mxbai", mxbai))
    results.append(eval_bi_encoder("P3_snowflake_bi", emb_cache, chunks, QUERIES, "snowflake", snow))
    results.append(eval_bi_encoder("P4_bge_m3_rerank", emb_cache, chunks, QUERIES, "bge_m3", bge, use_reranker=True))
    results.append(eval_bi_encoder("P5_mxbai_rerank", emb_cache, chunks, QUERIES, "mxbai", mxbai, use_reranker=True))
    results.append(eval_bi_encoder("P6_snowflake_rerank", emb_cache, chunks, QUERIES, "snowflake", snow, use_reranker=True))
    results.append(eval_rrf_ensemble("P7_rrf_ensemble_rerank", emb_cache, chunks, QUERIES,
                                     {"bge_m3": bge, "mxbai": mxbai, "snowflake": snow}))
    results.append(eval_rrf_ensemble("P8_hyde_rrf_rerank", emb_cache, chunks, QUERIES,
                                     {"bge_m3": bge, "mxbai": mxbai, "snowflake": snow},
                                     use_hyde=True, hyde_cache=hyde_cache))

    # Save
    out = {
        "n_chunks": len(chunks),
        "n_queries": len(QUERIES),
        "corpus_breakdown": {s: sum(1 for c in chunks if c["source"] == s)
                              for s in ["wiki_crisis", "sec_10k", "policy", "world_bank"]},
        "pipelines": {r["pipeline"]: r["aggregate"] for r in results},
        "per_pipeline_detail": {r["pipeline"]: r for r in results},
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R5_GRANITE.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    log.info("\n=== SUMMARY (sorted by MRR) ===")
    sorted_p = sorted(out["pipelines"].items(), key=lambda x: -x[1]["mrr"])
    for pname, m in sorted_p:
        log.info(f"  {pname:<30}  P@1={m['p1']:.3f}  P@3={m['p3']:.3f}  "
                 f"MRR={m['mrr']:.3f}  nDCG@10={m['ndcg10']:.3f}  lat={m['latency_s']:.2f}s")
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
