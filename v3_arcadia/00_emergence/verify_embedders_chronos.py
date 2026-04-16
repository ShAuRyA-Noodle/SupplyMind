"""Verify: BGE-M3, mxbai-embed-large, BGE-reranker-v2-m3, Snowflake Arctic, Chronos-Bolt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
OUT_PATH = ROOT / "v3_arcadia" / "results" / "embedders_chronos_verify.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
result: dict = {"device": DEVICE}

sample_docs = [
    "Tohoku earthquake 2011 disrupted Toyota supply chain.",
    "Suez Canal blockage in 2021 delayed 400+ vessels.",
    "Red Sea Houthi attacks forced Cape of Good Hope reroute.",
]
sample_query = "Japan tsunami auto parts crisis"

# ----- BGE-M3 -----
# BGE-M3 uses pytorch_model.bin; torch>=2.6 requires weights_only=True but model is safe (trusted source).
# Monkey-patch torch.load for this import only.
try:
    import torch as _torch
    _orig_load = _torch.load
    def _patched_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    _torch.load = _patched_load
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(MODELS / "bge-m3"), device=DEVICE)
    embs = m.encode(sample_docs, normalize_embeddings=True)
    q = m.encode([sample_query], normalize_embeddings=True)[0]
    scores = (embs @ q).tolist()
    result["bge_m3"] = {"status": "OK", "emb_dim": embs.shape[1], "scores": scores,
                        "note": "torch.load monkey-patched (trusted local weights)"}
    print(f"BGE-M3 OK: dim={embs.shape[1]}, scores={[round(s,3) for s in scores]}")
    del m; torch.cuda.empty_cache()
    _torch.load = _orig_load
except Exception as e:
    result["bge_m3"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"BGE-M3 FAIL: {e}")

# ----- mxbai-embed-large -----
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(MODELS / "mxbai-embed-large"), device=DEVICE)
    embs = m.encode(sample_docs, normalize_embeddings=True)
    q = m.encode([sample_query], normalize_embeddings=True)[0]
    scores = (embs @ q).tolist()
    result["mxbai"] = {"status": "OK", "emb_dim": embs.shape[1], "scores": scores}
    print(f"mxbai OK: dim={embs.shape[1]}, scores={[round(s,3) for s in scores]}")
    del m; torch.cuda.empty_cache()
except Exception as e:
    result["mxbai"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"mxbai FAIL: {e}")

# ----- Snowflake Arctic Embed L v2 (force pytorch backend) -----
try:
    from sentence_transformers import SentenceTransformer
    # Prevent ONNX variants that hung before by pointing to sentence-transformers subdir only
    m = SentenceTransformer(str(MODELS / "snowflake-arctic-embed-l"),
                             device=DEVICE, backend="torch", trust_remote_code=True)
    embs = m.encode(sample_docs, normalize_embeddings=True)
    q = m.encode([sample_query], normalize_embeddings=True)[0]
    scores = (embs @ q).tolist()
    result["snowflake_arctic"] = {"status": "OK", "emb_dim": embs.shape[1], "scores": scores}
    print(f"Snowflake OK: dim={embs.shape[1]}, scores={[round(s,3) for s in scores]}")
    del m; torch.cuda.empty_cache()
except Exception as e:
    result["snowflake_arctic"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"Snowflake FAIL: {str(e)[:200]}")

# ----- BGE Reranker v2 -----
try:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(str(MODELS / "bge-reranker-v2-m3"), device=DEVICE)
    pairs = [(sample_query, d) for d in sample_docs]
    rr = ce.predict(pairs, batch_size=8).tolist()
    result["bge_reranker_v2"] = {"status": "OK", "rerank_scores": rr}
    print(f"BGE-reranker-v2 OK: scores={[round(s,3) for s in rr]}")
    del ce; torch.cuda.empty_cache()
except Exception as e:
    result["bge_reranker_v2"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"BGE-reranker FAIL: {e}")

# ----- Chronos-Bolt-Base -----
try:
    from chronos import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained(str(MODELS / "chronos-bolt-base"), device_map=DEVICE)
    ts = np.sin(np.linspace(0, 20, 200)).astype(np.float32)
    ctx = torch.tensor(ts).unsqueeze(0)
    q, _ = pipe.predict_quantiles(inputs=ctx, prediction_length=14, quantile_levels=[0.1, 0.5, 0.9])
    pred = q[0].cpu().numpy()
    result["chronos_bolt"] = {"status": "OK", "pred_shape": list(pred.shape),
                               "sample_p50": pred[:5, 1].tolist()}
    print(f"Chronos-Bolt OK: shape={pred.shape}, p50={pred[:3,1].round(3).tolist()}")
    del pipe; torch.cuda.empty_cache()
except Exception as e:
    result["chronos_bolt"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"Chronos FAIL: {e}")

OUT_PATH.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT_PATH}")
