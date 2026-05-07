# SupplyMind v3.0-arcadia
### The 13-model stack that sees supply chain disruption 72 hours ahead.

Meta PyTorch OpenEnv Hackathon · ShAuRyA-Noodle · 2026

---

## Slide 1 — The Problem

**Supply chain disruptions cost the global economy $184B in 2023 alone.**

- 2021 Suez blockage: **$9.6B/day** for 6 days (Ever Given)
- 2020–2023 chip shortage: **$210B lost revenue**, 7.7M vehicles unbuilt (AlixPartners)
- Taiwan Strait: **92% of advanced semiconductor manufacturing at risk**
- US East/Gulf port strikes 2024: **$5B/day** economic impact
- Average reaction time for Fortune 1000: **7–14 days AFTER disruption**

**Existing tools** (SAP IBP, Oracle SCM, Resilinc, Everstream, Interos) are **reactive dashboards**. They tell you what already broke.

> *72 hours of advance warning is worth 100× a post-disruption dashboard.*

---

## Slide 2 — What SupplyMind Ships

**An OpenEnv-compliant environment + 13 SOTA models + 154 passing tests — all local, all real data.**

| Layer | Tech | Headline |
|---|---|---|
| **LLM risk panel** | DeepSeek-R1-Q4 + Qwen-14B + Mistral-Nemo + Qwen-Coder critic | 100% parse on 26 crises; α = 0.75; 69% GT acc |
| **RAG** | BGE-M3 + mxbai + Snowflake + BGE-reranker + HyDE | mxbai P@1 = **0.962**; reranker earns +5pp on hard queries |
| **Forecasting** | Chronos-Bolt + TimesFM-2 + ARIMA + Prophet + Bates-Granger | 20-fold backtest; PICP@80 = 0.77–0.89 |
| **RL** | MaskablePPO on MultiDiscrete[7,40] via Discrete(280) wrapper | 8,100-ep bench, CI95 non-overlapping, zero violations |
| **GNN** | Pure-PyTorch 3-layer GCN (no torch_geometric) | Arrival-time regression on real supply graphs |
| **Conformal** | Per-horizon split-conformal | ±2pp of nominal on DCOILWTICO@95% |
| **Production** | FastAPI + MCP JSON-RPC + WebSocket + Docker + ONNX | 12 OpenEnv endpoints + 5 v3 endpoints |

**261,175 real data points** from 8 cited sources (DataCo, NOAA, USGS, FRED, WGI, World Bank, SEC 10-K, Wikipedia).

---

## Slide 3 — Proof: 3 honest charts

### Chart 1: R6 Euclidian — PPO flips the sign
On medium + hard tasks, **greedy heuristic performs WORSE than random**.
PPO_v3 flips the sign: greedy = -1.81 → PPO = **+2.78** (8,100-ep bootstrap CI95 non-overlapping).

### Chart 2: R5 Granite — reranker redemption
Bi-encoder alone P@1 = 0.962 on precise queries (reranker flat), drops to 0.70 on hard paraphrased queries. Reranker adds **+5pp** on hard. **Right tool for right regime.**

### Chart 3: R4 Dangerous — Pareto front on panel configuration
3-judge = best accuracy (69.2%). 2-judge (Qwen+Mistral only) = best consensus (α = 0.75). Rubric agent = matches 2-judge. Honest: the LLM panel adds calibration + structure + novelty coverage, not raw accuracy.

---

## Slide 4 — Research insights (honest wins from negative findings)

1. **R2** TabPFN 10K cap → pre-cache full-data predictions restores stacking advantage
2. **R3** inverse-MAE → **Bates-Granger constrained stacking** (scipy SLSQP, weights ≥ 0, sum = 1) wins 9/21 cells vs 0/21 for inverse-MAE
3. **R4** α = 0.21 on 3-judge → α = 0.75 on 2-judge (Qwen+Mistral); DeepSeek reassigned to **devil's-advocate role** (present, weighted, not voting)
4. **R5** reranker hurts on easy → **hard-query benchmark** shows reranker regime; +5pp P@1 lift
5. **R6 Aqua Regia** pooled conformal under-covers oil → **per-horizon q̂** hits nominal within 2pp
6. **R6 Provider** easy F1 = 1.000 (trivial) → **arrival-time regression** with noisy edge weights as non-trivial GNN benchmark

We document every negative finding with the follow-up fix committed. See `FAILURE_TABLE.md`, `MODEL_CARD.md` §3.

---

## Slide 5 — Why SupplyMind should win top-3

**No comparable hackathon submission combines:**
- ✅ OpenEnv compliance (12 HTTP endpoints + MCP JSON-RPC + WebSocket, formal test suite)
- ✅ 154 + 19 = **173 passing tests**, 5×-run zero-variance grader
- ✅ 13 foundation models locally (zero API cost at inference)
- ✅ 261,175 real data points from 8 cited sources (zero synthetic)
- ✅ 8,100-episode RL benchmark with bootstrap CI95
- ✅ Per-horizon split-conformal prediction intervals
- ✅ 100% LLM parse rate via novel two-pass DeepSeek extraction
- ✅ Custom 3-layer GCN in pure PyTorch (no torch_geometric)
- ✅ Production ONNX artifacts (0.97 MB per policy)
- ✅ Full reproducibility: `pip install -r requirements.txt && pytest tests/ -q` in 1m 47s
- ✅ Honest negative findings with world-class follow-up fixes
- ✅ Sleep Token album theme: every phase commit named after a track, providing a unified narrative

**GitHub**: `github.com/ShAuRyA-Noodle/Sleep-Token` · **Tag**: `v3.0-arcadia` · **HF Space**: `huggingface.co/spaces/Shaurya-Noodle/Supplymind`

*"Even in Arcadia, supply chains break. SupplyMind sees it coming."*

---

### Appendix — generate PDF

```bash
# Requires pandoc + LaTeX
pandoc demo/PITCH_DECK.md -o demo/SupplyMind_pitch.pdf \
  --toc --pdf-engine=xelatex \
  -V geometry:margin=1in -V fontsize=11pt
```
