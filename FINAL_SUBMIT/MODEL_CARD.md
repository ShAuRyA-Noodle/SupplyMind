# Model Card — SupplyMind

**Regenerated 2026-07-15 from committed receipts** (WP8.2). Every quantitative claim here is
machine-checked by [`scripts/verify_claims.py`](../scripts/verify_claims.py) against the cited
receipt. Per-claim status lives in [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).

## Overview
- **Project**: SupplyMind — a live supply-chain-risk decision copilot on an OpenEnv-compliant
  deterministic simulation.
- **License**: MIT.
- **Stack**: Python 3.11+ · PyTorch 2.x · FastAPI · Pydantic v2 · NetworkX · scikit-learn.
- **Local model assets** (sanctioned edge exception — no OpenRouter equivalent): Chronos-Bolt,
  TimesFM-2, TabPFN-v2, mxbai-embed-large.

---

## 1. Components that ship, with their real measurement

Each row is a real model/component and the receipt that measures it. "KEEP / RETIRE" decisions come
from measurement, not preference.

| Component | Type | What it does | Measured result | Receipt |
|---|---|---|---|---|
| **Brent ensemble** | Chronos-Bolt + TimesFM-2 + TabPFN-v2 weighted blend | 30-day crude forecast around crisis events | mean MAPE **7.1%** over 8 real events (walk-forward on real FRED `DCOILBRENTEU`) | [`ensemble_brent_REAL.json`](../tests/receipts/ensemble_brent_REAL.json) |
| **TFT forecaster** | Temporal Fusion Transformer (90,602 params) | quantile WTI forecast (p10/p50/p90) | MAE p50 **$7.83**, 14-day horizon | [`tft_real_metrics.json`](receipts/tft_real_metrics.json) |
| **mxbai RAG** | mxbai-embed-large bi-encoder (1024-dim, FAISS HNSW) | crisis-analog retrieval over 6,483 real chunks | **P@1 0.962, MRR 0.978, nDCG@10 0.961** | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| **TabPFN judge** | TabPFN-v2 classifier | late-delivery-risk on DataCo (pre-shipment features) | **KEEP** — AUC **0.7377** vs logreg 0.7028 (adds signal) | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| **Embedder ensemble** | mxbai + Snowflake-Arctic-Embed-L (RRF) | dual-embedder retrieval | **RETIRE** — P@1 −0.076 vs mxbai alone (ensemble is *worse*) | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| **Reranker** | BGE-reranker-v2-m3 | rerank mxbai top-k | **RETIRE** — P@1 −0.038 vs bi-encoder | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| **Conformal action filter** | split-conformal over a BC reference policy | calibrated action-set with coverage guarantee | 90% nominal → **90.03%** held-out coverage | [`conformal_REAL.json`](../tests/receipts/conformal_REAL.json) |
| **Causal counterfactual** | paired-bootstrap MC + Abadie synthetic control + AR(1) FRED + do-calculus | $ impact of a disruption | 4 real methods; macro synthetic control **$230B** brackets the documented $210–235B Tōhoku headline | [`counterfactual_4method_REAL.json`](receipts/counterfactual_4method_REAL.json) |
| **BC policy + SHAP** | behaviour-cloned policy (408-dim obs) | supply-chain action + explanation | real SHAP: NOAA feature group ≈ **60%** attribution | [`shap_real.json`](receipts/shap_real.json) |
| **Wordle REINFORCE** | small policy net (Williams 1992) | RLVR self-improvement demo | beats random (p=2.7e-18, d=4.27) but **ties info-aware greedy** (p=0.93) | [`pass27_B_real_episodic_bootstrap.json`](receipts/pass27_B_real_episodic_bootstrap.json) |
| **FedAvg** | logistic regression, 5-region FedAvg (McMahan 2017) | privacy-preserving late-delivery model | **honest negative** — AUC 0.7218 = centralized 0.7218 (Δ −0.0) | [`fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json) |

### v3 result receipts (on disk, inspectable; not in the automated gate)
- GNN arrival-time MAE −48/−49/−64% vs MLP — [`R6_PROVIDER_V2.json`](receipts/R6_PROVIDER_V2.json)
- Per-horizon conformal deviation 0.024 on WTI — [`R6_AQUA_REGIA_V2.json`](receipts/R6_AQUA_REGIA_V2.json)
- Cohen κ 0.747 (Qwen × Mistral) judge agreement — [`R4_DANGEROUS_V2_ABLATION.json`](receipts/R4_DANGEROUS_V2_ABLATION.json)
- ONNX roundtrip 4/4 (opset 17) — [`onnx_roundtrip.json`](receipts/onnx_roundtrip.json)

---

## 2. RL agents (trained checkpoints on disk)

Behaviour Cloning, CQL, IQL, TD3+BC, Decision Transformer, MaskablePPO, QR-DQN — checkpoints in
[`rl/checkpoints/`](../rl/checkpoints). Training genuinely ran (varied grades in
`rl/autoresearch_results.json`). Observation is 408-dim; the action space is MultiDiscrete with
action masking.

**The head-to-head RL leaderboard is RERUN-PENDING (WP6.3).** The buffer↔env action-taxonomy
mismatch must be regenerated first, then the real checkpoints evaluated on easy/medium/hard × ≥20
seeds with correct paired-by-seed statistics. Until then, no RL-vs-baseline ranking is cited as
fact, and the RL-vs-scripted product narrative is deliberately left to that evidence
([ledger](CLAIMS_LEDGER.md) D7). The specialist-router mapping (which checkpoint serves which tier)
is recorded in [`specialist_router_real.json`](receipts/specialist_router_real.json).

---

## 3. LLM analyst layer

A domain-calibrated risk-analyst prompt (LOW/MEDIUM/HIGH/CRITICAL gating, confidence caps,
strict-JSON output, real 2011–2026 crisis knowledge). The IP is the prompt + calibration, portable
across providers. The OpenRouter port ([`supplymind/llm/`](../supplymind/llm)) and the Brier-scored
A/B (analyst prompt vs bare model) are **built but RERUN-PENDING** — `OPENROUTER_API_KEY` is revoked
([ledger](CLAIMS_LEDGER.md) C1). No live-panel number is cited as current.

---

## 4. Struck claims (do not cite — see ledger §C/§D)

- Fabricated Wilcoxon / Cohen-d leaderboard significance (sorted-"paired" test) — **STRUCK** (A1).
- Inconsistent Wordle p-values — **STRUCK**, replaced by the real per-episode paired test (A2 → V9).
- "0.9001 conformal on Gaussian noise" — **STRUCK**, replaced by the real 90.03% (A5 → V2).
- "DPO-fine-tuned judge" — **STRUCK**, never trained; real QLoRA re-train is WP7.4 (D2).
- Twin savings figure — **STRUCK** (receipt exit −9); re-run is WP9 (D3).
- "25-judge / 12-frontier panel" α — **RERUN-PENDING on the key** (C1).

---

## 5. Intended use, out-of-scope, ethics

- **Intended**: research + demonstration of an OpenEnv-compliant risk copilot; educational reference
  for RLVR / conformal / causal-counterfactual patterns.
- **Out of scope**: real-world production supply-chain decisions without further validation;
  extrapolating any single benchmark number to other domains.
- **Ethics**: live API calls (NewsAPI, EIA, NASA FIRMS, GFW, FRED, NOAA) access only public-license
  data; no PII. Reward functions are tested against real reward-hacking attacks (see result 1 in the
  root README and the adversarial receipt).

## 6. Reproducibility

```bash
pip install -e .
pytest tests/ -q                                   # 184/184
python scripts/verify_claims.py                    # every headline number → receipt
python -m supplymind.phoenix.counterfactual_v2.causal_methods --analog tohoku_2011 --receipt
```

Receipts are sha256-stamped and embed the git SHA + command. Local model assets (Chronos / TimesFM /
TabPFN / mxbai) are gitignored by size; the receipts record their measured behaviour. Citations in
[`CITATIONS.bib`](CITATIONS.bib).
