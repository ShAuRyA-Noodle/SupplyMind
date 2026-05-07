"""R6 Block 10 — Damocles: FastAPI production deployment.

Endpoints:
  POST /assess      — 3-judge risk assessment (R4 Dangerous V2 panel)
  POST /forecast    — multi-model time-series forecast (R3 Past Self)
  POST /rag         — retrieval-augmented answer (R5 Granite mxbai bi-encoder)
  POST /rl/act      — trained PPO policy action (R6 Gethsemane)
  GET  /health      — liveness + dependency check

Run:
  uvicorn v3_arcadia.90_damocles.app:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="SupplyMind v3 Arcadia", version="3.0.0",
              description="Supply-chain risk management API — R4 judges + R3 forecasters + R5 RAG + R6 RL")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# ============================================================
# Global state (lazy-loaded)
# ============================================================
_STATE: dict[str, Any] = {"ready": False, "embedder": None, "corpus_chunks": None,
                          "corpus_emb": None, "rl_model": None, "chronos": None}


def _load_rag():
    if _STATE["embedder"] is not None:
        return
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _STATE["embedder"] = SentenceTransformer(str(ROOT / "models" / "mxbai-embed-large"), device=dev)
    with open(ROOT / "v3_arcadia" / "checkpoints" / "granite" / "corpus_chunks.pkl", "rb") as f:
        _STATE["corpus_chunks"] = pickle.load(f)
    _STATE["corpus_emb"] = np.load(ROOT / "v3_arcadia" / "checkpoints" / "granite" / "corpus_emb_mxbai.npy")


def _load_rl():
    if _STATE["rl_model"] is not None:
        return
    from sb3_contrib import MaskablePPO
    # Use the easy-task model by default (most stable)
    ckpt = ROOT / "v3_arcadia" / "checkpoints" / "gethsemane" / "ppo_easy_typhoon_response.zip"
    if ckpt.exists():
        _STATE["rl_model"] = MaskablePPO.load(str(ckpt))


def _load_chronos():
    if _STATE["chronos"] is not None:
        return
    import torch
    from chronos import ChronosBoltPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _STATE["chronos"] = ChronosBoltPipeline.from_pretrained(
        str(ROOT / "models" / "chronos-bolt-base"),
        device_map=dev, torch_dtype=torch.float32)


# ============================================================
# Health
# ============================================================
@app.get("/health")
def health():
    import torch
    return {
        "ok": True,
        "version": "3.0.0-arcadia",
        "cuda_available": torch.cuda.is_available(),
        "ollama_reachable": _check_ollama(),
        "components": {
            "rag_loaded": _STATE["embedder"] is not None,
            "rl_loaded": _STATE["rl_model"] is not None,
            "chronos_loaded": _STATE["chronos"] is not None,
        },
    }


def _check_ollama() -> bool:
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ============================================================
# Risk assessment (R4 Dangerous panel)
# ============================================================
class AssessRequest(BaseModel):
    context: str = Field(..., description="Factual supply-chain scenario text", max_length=5000)
    judges: list[str] = Field(default=["qwen25-14b-local", "mistral-nemo-local"],
                              description="Ollama model names")


class JudgeResult(BaseModel):
    judge: str
    risk_level: str | None = None
    confidence: float | None = None
    vulnerabilities: list[str] = []
    mitigations: list[str] = []
    reasoning: str | None = None
    latency_s: float
    ok: bool


class AssessResponse(BaseModel):
    context_preview: str
    judges: list[JudgeResult]
    consensus_risk: str
    escalation: str


SYSTEM = """You are a supply-chain risk analyst. Return ONLY valid JSON with keys:
risk_level (LOW/MEDIUM/HIGH/CRITICAL), confidence (0-1), primary_vulnerabilities (3 items),
mitigations (3 actions), reasoning_one_line."""


def _call_judge(model: str, context: str) -> dict:
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"CONTEXT:\n{context}\n\nReturn JSON."}],
            "format": "json", "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 500},
        }, timeout=180)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            parsed = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(m.group()) if m else None
        return {"ok": bool(parsed), "parsed": parsed or {}, "latency_s": time.time() - t0}
    except Exception as e:
        return {"ok": False, "parsed": {}, "latency_s": time.time() - t0, "error": str(e)[:200]}


RISK_ORDINAL = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
RISK_REV = {v: k for k, v in RISK_ORDINAL.items()}


def _escalation(risk_level: str) -> str:
    lv = RISK_ORDINAL.get(risk_level, 1)
    return {
        4: "C_SUITE_IMMEDIATE", 3: "OPS_DIRECTOR_4H",
        2: "REGIONAL_MANAGER", 1: "FYI_DASHBOARD",
    }.get(lv, "FYI_DASHBOARD")


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest):
    if not _check_ollama():
        raise HTTPException(503, "Ollama not reachable")
    judge_results = []
    ratings = []
    for j in req.judges:
        r = _call_judge(j, req.context)
        p = r["parsed"] or {}
        rl = str(p.get("risk_level", "")).upper() if r["ok"] else None
        if rl in RISK_ORDINAL:
            ratings.append(RISK_ORDINAL[rl])
        judge_results.append(JudgeResult(
            judge=j, ok=r["ok"],
            risk_level=rl,
            confidence=p.get("confidence") if isinstance(p.get("confidence"), (int, float)) else None,
            vulnerabilities=[str(x) for x in (p.get("primary_vulnerabilities") or [])],
            mitigations=[str(x) for x in (p.get("mitigations") or [])],
            reasoning=p.get("reasoning_one_line"),
            latency_s=r["latency_s"],
        ))
    consensus = RISK_REV.get(int(np.round(np.median(ratings)))) if ratings else "UNKNOWN"
    return AssessResponse(
        context_preview=req.context[:120],
        judges=judge_results,
        consensus_risk=consensus,
        escalation=_escalation(consensus),
    )


# ============================================================
# RAG (R5 Granite mxbai bi-encoder — best pipeline)
# ============================================================
class RagRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    top_k: int = Field(5, ge=1, le=20)


class RagHit(BaseModel):
    rank: int
    doc_id: str
    source: str
    chunk_idx: int
    score: float
    text_preview: str


class RagResponse(BaseModel):
    query: str
    hits: list[RagHit]
    latency_s: float


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    _load_rag()
    t0 = time.time()
    q_emb = _STATE["embedder"].encode(req.query, normalize_embeddings=True, convert_to_numpy=True)
    sims = _STATE["corpus_emb"] @ q_emb
    idx = np.argsort(sims)[::-1][:req.top_k]
    hits = []
    for rank, i in enumerate(idx):
        c = _STATE["corpus_chunks"][int(i)]
        hits.append(RagHit(
            rank=rank + 1, doc_id=c["doc_id"], source=c["source"],
            chunk_idx=c["chunk_idx"], score=float(sims[int(i)]),
            text_preview=c["text"][:200],
        ))
    return RagResponse(query=req.query, hits=hits, latency_s=time.time() - t0)


# ============================================================
# Forecast (R3 Past Self — Chronos zero-shot)
# ============================================================
class ForecastRequest(BaseModel):
    series: list[float] = Field(..., min_length=30, max_length=2000)
    horizon: int = Field(14, ge=1, le=64)


class ForecastResponse(BaseModel):
    point: list[float]
    lo_80: list[float]
    hi_80: list[float]
    lo_95: list[float]
    hi_95: list[float]
    latency_s: float


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    _load_chronos()
    import torch
    t0 = time.time()
    ctx = torch.tensor(req.series[-1024:], dtype=torch.float32).unsqueeze(0)
    q, _ = _STATE["chronos"].predict_quantiles(
        inputs=ctx, prediction_length=req.horizon,
        quantile_levels=[0.025, 0.1, 0.5, 0.9, 0.975])
    arr = q[0].cpu().numpy()
    return ForecastResponse(
        point=arr[:, 2].tolist(),
        lo_80=arr[:, 1].tolist(), hi_80=arr[:, 3].tolist(),
        lo_95=arr[:, 0].tolist(), hi_95=arr[:, 4].tolist(),
        latency_s=time.time() - t0,
    )


# ============================================================
# RL act (R6 Gethsemane PPO)
# ============================================================
class RlActRequest(BaseModel):
    observation: list[float] = Field(..., min_length=408, max_length=408)
    action_mask: list[bool] = Field(..., min_length=280, max_length=280)
    deterministic: bool = True


class RlActResponse(BaseModel):
    action_type_idx: int
    action_type: str
    target_node: int
    latency_s: float


ACTION_TYPES = [
    "do_nothing", "activate_backup_supplier", "reroute_shipment",
    "increase_safety_stock", "expedite_order", "hedge_commodity", "issue_supplier_alert",
]


@app.post("/rl/act", response_model=RlActResponse)
def rl_act(req: RlActRequest):
    _load_rl()
    if _STATE["rl_model"] is None:
        raise HTTPException(503, "RL model not loaded — run R6 Gethsemane training first")
    t0 = time.time()
    obs = np.asarray(req.observation, dtype=np.float32)[None]
    mask = np.asarray(req.action_mask, dtype=bool)[None]
    flat, _ = _STATE["rl_model"].predict(obs, action_masks=mask, deterministic=req.deterministic)
    flat = int(flat[0] if hasattr(flat, "__len__") else flat)
    a_type, a_target = divmod(flat, 40)
    return RlActResponse(
        action_type_idx=a_type, action_type=ACTION_TYPES[a_type],
        target_node=a_target, latency_s=time.time() - t0,
    )


# ============================================================
# Root
# ============================================================
@app.get("/")
def root():
    return {
        "name": "SupplyMind v3 Arcadia",
        "version": "3.0.0-arcadia",
        "endpoints": ["/health", "/assess", "/forecast", "/rag", "/rl/act", "/docs"],
        "docs": "/docs",
    }
