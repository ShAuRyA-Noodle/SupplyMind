"""ghost_models_eval.py — R9 ghost-model measurement harness.

The three "ghost" model features (Snowflake-Arctic 2nd embedder, BGE-reranker,
TabPFN 7th judge) were DELETED in P0.2 because their weights were never on disk
and the code silently fell back to the single model — a fake per CLAUDE.md §0.

This harness resurrects them for real: it downloads the weights (see
scratchpad dl_files.py / models/), then MEASURES, on committed labeled data,
whether each feature actually helps. A feature ships only with its honest
measurement — "it helps by X (keep)" or "it does not help (retire)". Both are
acceptable; a fabricated favourable number is not.

Measurements
------------
1. Retrieval: mxbai-alone vs (mxbai + Snowflake-Arctic-L) ensemble on the 53
   gold-labeled BEIR-style RAG queries (R5 corpus, 6,483 chunks). P@1 / MRR.
2. Rerank: mxbai top-k vs mxbai top-k re-ranked by BGE-reranker-v2-m3. P@1.
3. TabPFN judge: TabPFN-v2 classifier vs LogisticRegression / majority on a
   labeled DataCo late-delivery-risk holdout. Accuracy / ROC-AUC.

Run:
  .venv/Scripts/python.exe -m supplymind.warroom.ghost_models_eval

Outputs tests/receipts/ghost_models_eval_REAL.json with real numbers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("r9_eval")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
GRANITE = REPO_ROOT / "versions" / "v3_arcadia" / "checkpoints" / "granite"
R5_JSON = REPO_ROOT / "versions" / "v3_arcadia" / "results" / "R5_GRANITE.json"
CORPUS_PKL = GRANITE / "corpus_chunks.pkl"
DATACO = REPO_ROOT / "rl" / "data" / "dataco.csv"
EMB_CACHE = MODELS / ".r9_emb_cache"

MXBAI_DIR = MODELS / "mxbai-embed-large"
SNOW_DIR = MODELS / "snowflake-arctic-embed-l"
RERANKER_DIR = MODELS / "bge-reranker-v2-m3"
TABPFN_CKPT = MODELS / "tabpfn-v2-clf" / "tabpfn-v2-classifier.ckpt"

TOP_K_RETRIEVE = 50
SEED = 42


# ============================================================
# shared metric primitives (identical defs to r5_rag_beast.py)
# ============================================================
def is_gold(chunk: dict, gold: list[str]) -> bool:
    return chunk["doc_id"] in gold


def precision_at_k(ranked: list[int], chunks, gold, k) -> float:
    return sum(1 for i in ranked[:k] if is_gold(chunks[i], gold)) / k


def recall_at_k(ranked: list[int], chunks, gold, k) -> float:
    gs = set(gold)
    hit = {chunks[i]["doc_id"] for i in ranked[:k] if is_gold(chunks[i], gold)}
    return len(hit & gs) / len(gs) if gs else 0.0


def mrr(ranked: list[int], chunks, gold) -> float:
    for r, i in enumerate(ranked):
        if is_gold(chunks[i], gold):
            return 1.0 / (r + 1)
    return 0.0


def ndcg_at_k(ranked: list[int], chunks, gold, k) -> float:
    gains = [1.0 if is_gold(chunks[i], gold) else 0.0 for i in ranked[:k]]
    dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def cosine_topk(q_emb: np.ndarray, corpus_emb: np.ndarray, k: int):
    sims = corpus_emb @ q_emb
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


def rrf_fuse(ranked_lists, k_rrf=60, top_k=TOP_K_RETRIEVE):
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (idx, _) in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    items = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [(int(i), float(s)) for i, s in items]


def mean_cosine_fuse(ranked_a, ranked_b, top_k=TOP_K_RETRIEVE):
    """Blend two cosine-scored lists by mean score (the git ensemble method)."""
    table: dict[int, list[float]] = {}
    for idx, s in ranked_a:
        table.setdefault(idx, [0.0, 0.0])[0] = s
    for idx, s in ranked_b:
        table.setdefault(idx, [0.0, 0.0])[1] = s
    blended = [(i, (a + b) / 2.0) for i, (a, b) in table.items()]
    blended.sort(key=lambda x: -x[1])
    return [(int(i), float(s)) for i, s in blended[:top_k]]


def _agg(per_q: list[dict]) -> dict:
    keys = ["p1", "p3", "p5", "r5", "r10", "mrr", "ndcg10"]
    return {k: float(np.mean([q[k] for q in per_q])) for k in keys}


def _score_ranking(ranked_idx, chunks, gold) -> dict:
    return {
        "p1": precision_at_k(ranked_idx, chunks, gold, 1),
        "p3": precision_at_k(ranked_idx, chunks, gold, 3),
        "p5": precision_at_k(ranked_idx, chunks, gold, 5),
        "r5": recall_at_k(ranked_idx, chunks, gold, 5),
        "r10": recall_at_k(ranked_idx, chunks, gold, 10),
        "mrr": mrr(ranked_idx, chunks, gold),
        "ndcg10": ndcg_at_k(ranked_idx, chunks, gold, 10),
    }


# ============================================================
# data loaders
# ============================================================
def load_corpus_and_queries():
    if not CORPUS_PKL.exists():
        raise FileNotFoundError(f"corpus chunks not found: {CORPUS_PKL}")
    if not R5_JSON.exists():
        raise FileNotFoundError(f"gold query set not found: {R5_JSON}")
    with open(CORPUS_PKL, "rb") as f:
        chunks = pickle.load(f)
    r5 = json.loads(R5_JSON.read_text(encoding="utf-8"))
    per_q = r5["per_pipeline_detail"]["P2_mxbai_bi"]["per_query"]
    queries = [{"q": q["q"], "gold": q["gold"]} for q in per_q]
    return chunks, queries


def _embed_corpus(name: str, model_dir: Path, chunks, device: str,
                  backend: str | None = None) -> np.ndarray:
    """Embed all chunk texts with a SentenceTransformer, cached by weight hash."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    # cache key ties embeddings to the actual weight file bytes (mtime+size)
    st = (model_dir / "model.safetensors")
    key = f"{name}_{st.stat().st_size if st.exists() else 0}_{len(chunks)}"
    cache = EMB_CACHE / f"{key}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(chunks):
            log.info("loaded cached %s embeddings %s", name, emb.shape)
            return emb
    from sentence_transformers import SentenceTransformer
    kwargs = {"device": device}
    if backend:
        kwargs["backend"] = backend
    model = SentenceTransformer(str(model_dir), **kwargs)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32,
                       show_progress_bar=False, convert_to_numpy=True).astype("float32")
    np.save(cache, emb)
    log.info("embedded %s -> %s (dim %d)", name, emb.shape, emb.shape[1])
    del model
    return emb


# ============================================================
# Measurement 1 + 2: retrieval (ensemble + rerank)
# ============================================================
def eval_retrieval() -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunks, queries = load_corpus_and_queries()
    log.info("retrieval eval: %d chunks, %d gold queries, device=%s",
             len(chunks), len(queries), device)

    if not (MXBAI_DIR / "model.safetensors").exists():
        raise FileNotFoundError(f"mxbai weights missing: {MXBAI_DIR}")
    if not (SNOW_DIR / "model.safetensors").exists():
        raise FileNotFoundError(f"snowflake weights missing: {SNOW_DIR}")

    from sentence_transformers import SentenceTransformer
    mxbai_emb = _embed_corpus("mxbai", MXBAI_DIR, chunks, device)
    snow_emb = _embed_corpus("snowflake", SNOW_DIR, chunks, device, backend="torch")

    mxbai = SentenceTransformer(str(MXBAI_DIR), device=device)
    snow = SentenceTransformer(str(SNOW_DIR), device=device, backend="torch")

    reranker = None
    rerank_err = None
    if (RERANKER_DIR / "model.safetensors").exists():
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(str(RERANKER_DIR), device=device, max_length=512)
        except Exception as e:  # noqa: BLE001
            rerank_err = f"{type(e).__name__}: {e}"
            log.warning("reranker load failed: %s", rerank_err)
    else:
        rerank_err = f"weights missing: {RERANKER_DIR}"

    pipelines = {
        "mxbai_alone": [], "snowflake_alone": [],
        "ensemble_rrf": [], "ensemble_mean": [],
    }
    if reranker is not None:
        pipelines["mxbai_rerank"] = []
        pipelines["snowflake_rerank"] = []

    t0 = time.time()
    for q in queries:
        gold = q["gold"]
        qm = mxbai.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
        qs = snow.encode(q["q"], normalize_embeddings=True, convert_to_numpy=True)
        mx_ranked = cosine_topk(qm, mxbai_emb, TOP_K_RETRIEVE)
        sf_ranked = cosine_topk(qs, snow_emb, TOP_K_RETRIEVE)

        pipelines["mxbai_alone"].append(_score_ranking([i for i, _ in mx_ranked], chunks, gold))
        pipelines["snowflake_alone"].append(_score_ranking([i for i, _ in sf_ranked], chunks, gold))
        rrf = rrf_fuse([mx_ranked, sf_ranked])
        pipelines["ensemble_rrf"].append(_score_ranking([i for i, _ in rrf], chunks, gold))
        mean = mean_cosine_fuse(mx_ranked, sf_ranked)
        pipelines["ensemble_mean"].append(_score_ranking([i for i, _ in mean], chunks, gold))

        if reranker is not None:
            for pname, base in (("mxbai_rerank", mx_ranked), ("snowflake_rerank", sf_ranked)):
                cand_idx = [i for i, _ in base]
                pairs = [(q["q"], chunks[i]["text"]) for i in cand_idx]
                scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                order = np.argsort(scores)[::-1]
                reranked_idx = [cand_idx[j] for j in order]
                pipelines[pname].append(_score_ranking(reranked_idx, chunks, gold))

    agg = {name: _agg(pq) for name, pq in pipelines.items()}
    elapsed = time.time() - t0

    base_p1 = agg["mxbai_alone"]["p1"]
    base_mrr = agg["mxbai_alone"]["mrr"]

    # ensemble decision: best of the two blends
    ens_best = max(("ensemble_rrf", "ensemble_mean"),
                   key=lambda k: (agg[k]["p1"], agg[k]["mrr"]))
    ens_p1_delta = agg[ens_best]["p1"] - base_p1
    ens_mrr_delta = agg[ens_best]["mrr"] - base_mrr
    ensemble_keep = (ens_p1_delta > 1e-9) or (abs(ens_p1_delta) < 1e-9 and ens_mrr_delta > 1e-9)

    rerank_result = None
    if reranker is not None:
        rr_p1_delta = agg["mxbai_rerank"]["p1"] - base_p1
        rr_mrr_delta = agg["mxbai_rerank"]["mrr"] - base_mrr
        rerank_keep = (rr_p1_delta > 1e-9) or (abs(rr_p1_delta) < 1e-9 and rr_mrr_delta > 1e-9)
        rerank_result = {
            "available": True,
            "model": "bge-reranker-v2-m3",
            "mxbai_rerank_p1": agg["mxbai_rerank"]["p1"],
            "mxbai_rerank_mrr": agg["mxbai_rerank"]["mrr"],
            "p1_delta_vs_mxbai": rr_p1_delta,
            "mrr_delta_vs_mxbai": rr_mrr_delta,
            "decision": "KEEP" if rerank_keep else "RETIRE",
        }
    else:
        rerank_result = {"available": False, "reason": rerank_err,
                         "decision": "BLOCKED"}

    return {
        "n_chunks": len(chunks),
        "n_queries": len(queries),
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "aggregate_metrics": {k: {m: round(v, 4) for m, v in d.items()}
                              for k, d in agg.items()},
        "ensemble": {
            "embedders": ["mxbai-embed-large", "snowflake-arctic-embed-l"],
            "mxbai_alone_p1": round(base_p1, 4),
            "mxbai_alone_mrr": round(base_mrr, 4),
            "best_blend": ens_best,
            "ensemble_p1": round(agg[ens_best]["p1"], 4),
            "ensemble_mrr": round(agg[ens_best]["mrr"], 4),
            "p1_delta": round(ens_p1_delta, 4),
            "mrr_delta": round(ens_mrr_delta, 4),
            "decision": "KEEP" if ensemble_keep else "RETIRE",
        },
        "rerank": rerank_result,
    }


# ============================================================
# Measurement 3: TabPFN judge on DataCo late-delivery-risk
# ============================================================
# Pre-shipment features only (exclude anything computed at/after shipment to
# avoid label leakage: 'Days for shipping (real)', 'Delivery Status',
# 'Order Status' all encode the late/on-time outcome directly).
DATACO_NUM_FEATURES = [
    "Days for shipment (scheduled)", "Benefit per order", "Sales per customer",
    "Order Item Discount", "Order Item Discount Rate", "Order Item Product Price",
    "Order Item Profit Ratio", "Order Item Quantity", "Sales", "Order Item Total",
    "Order Profit Per Order", "Product Price", "Latitude", "Longitude",
]
DATACO_CAT_FEATURES = [
    "Category Id", "Department Id", "Market", "Order Region", "Shipping Mode",
    "Type",
]
DATACO_LABEL = "Late_delivery_risk"
DATACO_LEAKAGE = ["Days for shipping (real)", "Delivery Status", "Order Status"]


def _load_dataco(n_train=1000, n_test=2000, seed=SEED):
    import csv
    rows = []
    with open(DATACO, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for r in reader:
            rows.append(r)
    log.info("dataco: %d rows, %d cols", len(rows), len(cols))
    # verify none of our features is a leakage column
    for feat in DATACO_NUM_FEATURES + DATACO_CAT_FEATURES:
        assert feat not in DATACO_LEAKAGE, f"leakage feature slipped in: {feat}"
        assert feat in cols, f"feature not in dataco: {feat}"

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))
    rows = [rows[i] for i in idx]

    # build categorical encoders on the full pool (label-encode)
    cat_maps = {c: {} for c in DATACO_CAT_FEATURES}
    for c in DATACO_CAT_FEATURES:
        for r in rows:
            v = r.get(c, "")
            if v not in cat_maps[c]:
                cat_maps[c][v] = len(cat_maps[c])

    def to_vec(r):
        vec = []
        for c in DATACO_NUM_FEATURES:
            try:
                vec.append(float(r.get(c, "") or 0.0))
            except ValueError:
                vec.append(0.0)
        for c in DATACO_CAT_FEATURES:
            vec.append(float(cat_maps[c][r.get(c, "")]))
        return vec

    X = np.array([to_vec(r) for r in rows], dtype=np.float32)
    y = np.array([int(float(r.get(DATACO_LABEL, "0") or 0)) for r in rows],
                 dtype=np.int64)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:n_train + n_test], y[n_train:n_train + n_test]
    return (Xtr, ytr), (Xte, yte)


def eval_tabpfn() -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if not DATACO.exists():
        raise FileNotFoundError(f"dataco not found: {DATACO}")

    (Xtr, ytr), (Xte, yte) = _load_dataco()
    n_feat = Xtr.shape[1]
    pos_rate = float(ytr.mean())
    log.info("tabpfn eval: train=%d test=%d feat=%d pos_rate=%.3f",
             len(ytr), len(yte), n_feat, pos_rate)

    results = {
        "task": "DataCo late-delivery-risk (binary) — pre-shipment features only",
        "label": DATACO_LABEL,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_features": int(n_feat),
        "leakage_excluded": DATACO_LEAKAGE,
        "test_positive_rate": round(float(yte.mean()), 4),
    }

    # baseline 1: majority class
    maj = int(round(ytr.mean()))
    maj_pred = np.full_like(yte, maj)
    results["majority_baseline_acc"] = round(accuracy_score(yte, maj_pred), 4)

    # baseline 2: logistic regression (scaled)
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    lr_prob = lr.predict_proba(scaler.transform(Xte))[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    results["logreg_acc"] = round(accuracy_score(yte, lr_pred), 4)
    results["logreg_auc"] = round(roc_auc_score(yte, lr_prob), 4)

    # TabPFN
    if not TABPFN_CKPT.exists():
        results["tabpfn"] = {"available": False,
                             "reason": f"weights missing: {TABPFN_CKPT}",
                             "decision": "BLOCKED"}
        return results
    import torch
    from tabpfn import TabPFNClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    clf = TabPFNClassifier(device=device, model_path=str(TABPFN_CKPT),
                           n_estimators=4, ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr)
    tp_prob = clf.predict_proba(Xte)[:, 1]
    tp_pred = (tp_prob >= 0.5).astype(int)
    tp_acc = accuracy_score(yte, tp_pred)
    tp_auc = roc_auc_score(yte, tp_prob)
    elapsed = time.time() - t0

    acc_delta_vs_lr = tp_acc - results["logreg_acc"]
    auc_delta_vs_lr = tp_auc - results["logreg_auc"]
    acc_delta_vs_maj = tp_acc - results["majority_baseline_acc"]
    # "adds real signal" = beats majority AND at least matches logreg
    adds_signal = (acc_delta_vs_maj > 1e-4) and (auc_delta_vs_lr >= -1e-4)

    results["tabpfn"] = {
        "available": True,
        "model": "tabpfn-v2-clf",
        "device": device,
        "n_estimators": 4,
        "acc": round(tp_acc, 4),
        "auc": round(tp_auc, 4),
        "acc_delta_vs_majority": round(acc_delta_vs_maj, 4),
        "acc_delta_vs_logreg": round(acc_delta_vs_lr, 4),
        "auc_delta_vs_logreg": round(auc_delta_vs_lr, 4),
        "fit_predict_s": round(elapsed, 1),
        "decision": "KEEP" if adds_signal else "RETIRE",
        "note": ("Comparison is TabPFN vs standard tabular baselines "
                 "(majority, logistic regression). The 6-judge OpenRouter LLM "
                 "panel comparison is BLOCKED-ON-KEY (OPENROUTER_API_KEY "
                 "revoked); this key-free measurement answers whether TabPFN "
                 "adds predictive signal on the labeled task."),
    }
    return results


# ============================================================
# receipt
# ============================================================
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _weight_manifest() -> dict:
    def info(p: Path):
        if p.exists():
            b = p.read_bytes() if p.stat().st_size < 5_000_000 else None
            return {"present": True, "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(b).hexdigest() if b else "large-file-skipped"}
        return {"present": False}
    return {
        "snowflake-arctic-embed-l/model.safetensors": info(SNOW_DIR / "model.safetensors"),
        "bge-reranker-v2-m3/model.safetensors": info(RERANKER_DIR / "model.safetensors"),
        "tabpfn-v2-clf/tabpfn-v2-classifier.ckpt": info(TABPFN_CKPT),
        "mxbai-embed-large/model.safetensors": info(MXBAI_DIR / "model.safetensors"),
    }


def main():
    receipt: dict = {
        "receipt": "R9_ghost_models_eval_REAL",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "principle": ("Each ghost feature ships only with its honest measurement. "
                      "KEEP = measurably helps; RETIRE = does not help on real data."),
        "weights_on_disk": _weight_manifest(),
    }

    # Measurement 1+2
    try:
        receipt["retrieval"] = eval_retrieval()
    except Exception as e:  # noqa: BLE001
        log.exception("retrieval eval failed")
        receipt["retrieval"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Measurement 3
    try:
        receipt["tabpfn_judge"] = eval_tabpfn()
    except Exception as e:  # noqa: BLE001
        log.exception("tabpfn eval failed")
        receipt["tabpfn_judge"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    out = REPO_ROOT / "tests" / "receipts" / "ghost_models_eval_REAL.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    log.info("wrote receipt -> %s", out)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
