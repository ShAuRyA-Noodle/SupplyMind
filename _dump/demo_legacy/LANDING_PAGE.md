# SupplyMind v3.0-arcadia — Landing Page (for Notion / GitBook / HF Space README)

> Copy-paste this into a Notion page or use as Landing section of the HF Space README. Every link already works against the public GitHub repo.

---

## Hero

# 🌊 SupplyMind v3.0 "Even In Arcadia"

**Supply chain risk in 12 seconds. 13 SOTA models. 100% local inference. All real data.**

[![GitHub](https://img.shields.io/badge/GitHub-ShAuRyA--Noodle%2FSleep--Token-blue?logo=github)](https://github.com/ShAuRyA-Noodle/Sleep-Token)
[![HF Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/Shaurya-Noodle/Supplymind)
[![Release](https://img.shields.io/badge/Release-v3.0--arcadia-purple)](https://github.com/ShAuRyA-Noodle/Sleep-Token/releases/tag/v3.0-arcadia)
[![Tests](https://img.shields.io/badge/Tests-173%20passing-brightgreen)](https://github.com/ShAuRyA-Noodle/Sleep-Token/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/LICENSE)

> *"Even in Arcadia, supply chains break. SupplyMind sees it coming."*

![SupplyMind v3.0-arcadia hero card](../v3_arcadia/plots/hero_result_card.png)

## Ten headline numbers (30-second read)

| # | Metric | Value |
|---|---|---|
| 1 | RAG nDCG@10 (Snowflake, out-of-domain) | **0.971** |
| 2 | RAG P@1 (mxbai bi-encoder, 6,483 chunks) | **0.962** |
| 3 | RAG MRR (precise queries) | **0.978** |
| 4 | LLM 2-judge Krippendorff α (ordinal) | **0.750** |
| 5 | Cohen κ (Qwen × Mistral weighted) | **0.747** |
| 6 | Per-horizon conformal dev on WTI @ 95% | **0.024** |
| 7 | MaskablePPO masking lift (easy / hard) | **+26.8% / +15.1%** |
| 8 | GNN arrival-time MAE reduction vs MLP | **−48 / −49 / −64%** |
| 9 | TimesFM-CP dev @ 95% (WTI / EUR-USD) | **0.050 / 0.032** |
| 10 | Tests passing | **173** (2m 14s) |

Every number reproducible with `python scripts/run_all.py` against the committed JSONs.

---

## The problem (60-second read)

Global supply-chain disruptions cost **$184 billion in 2023** alone. Existing tools — SAP IBP, Oracle SCM, Resilinc, Everstream — are **reactive dashboards**. They tell you what already broke.

- **2021 Suez blockage**: $9.6 B/day × 6 days ([source](https://en.wikipedia.org/wiki/2021_Suez_Canal_obstruction))
- **2020–2023 chip shortage**: $210 B lost revenue, 7.7 M vehicles unbuilt ([AlixPartners](https://www.alixpartners.com))
- **Taiwan Strait**: **92%** of advanced semiconductor manufacturing concentration ([SemiAnalysis](https://www.semianalysis.com))
- **2024 US port strikes**: $5 B/day economic impact

**Average Fortune-1000 reaction time: 7–14 days AFTER disruption.**

> **72 hours of advance warning is worth 100× a post-mortem dashboard.**

---

## What we shipped

A **full-stack supply-chain risk intelligence platform**, implemented as an **OpenEnv-compliant AI environment** plus a production API:

| Layer | Technology | Headline metric |
|---|---|---|
| **LLM risk panel** | DeepSeek-R1-Q4 + Qwen-2.5-14B + Mistral-Nemo + Qwen-Coder critic | 100% parse rate, Krippendorff α = 0.75 on 2-judge consensus |
| **RAG** | BGE-M3 + mxbai + Snowflake + BGE-reranker + HyDE | **mxbai P@1 = 0.962**, reranker +5pp on hard queries |
| **Forecasting** | Chronos-Bolt + TimesFM-2 + ARIMA + Prophet + Bates-Granger stacking | 20-fold rolling-origin backtest, PICP@80 near-nominal |
| **RL** | MaskablePPO on MultiDiscrete[7,40] → Discrete(280) wrapper | 8,100-ep bootstrap CI95 non-overlapping, **zero constraint violations** |
| **GNN** | Custom 3-layer GCN in pure PyTorch (no torch_geometric) | Arrival-time regression, **+48–64% vs MLP baseline** |
| **Conformal** | Per-horizon split-conformal | Nominal coverage hit within ±2pp on oil@95% |
| **Production** | FastAPI + MCP JSON-RPC + WebSocket + Docker + ONNX | **12 OpenEnv endpoints** + 5 v3 API endpoints |

**Fully local**. Zero API cost at inference. All 13 foundation models sit on a laptop.

---

## 3 demos to try in your browser

### 🧠 Live risk assessment

```bash
curl -X POST https://huggingface.co/spaces/Shaurya-Noodle/Supplymind/assess \
  -H "Content-Type: application/json" \
  -d '{"context":"On March 11, 2011, a M9.0 earthquake struck Tohoku, Japan...", "judges":["qwen25-14b-local","mistral-nemo-local"]}'
```

Returns consensus risk, per-judge verdicts, escalation tier, confidence. **~12 seconds.**

### 📈 72-hour forecast with calibrated intervals

```bash
curl -X POST https://huggingface.co/spaces/Shaurya-Noodle/Supplymind/forecast \
  -H "Content-Type: application/json" \
  -d '{"series":[78.4,79.1,...last 60 days of WTI oil], "horizon":14}'
```

Returns point forecast + 80% + 95% split-conformal-calibrated intervals. **~1 second.**

### 🔎 RAG over 6,483 real crisis chunks

```bash
curl -X POST https://huggingface.co/spaces/Shaurya-Noodle/Supplymind/rag \
  -H "Content-Type: application/json" \
  -d '{"query":"Which foundry produces most advanced logic chips?", "top_k":5}'
```

Returns top-5 chunks from real SEC 10-K filings + Wikipedia. **~40 ms.**

---

## What makes it SOTA

- **13 foundation models actually used** (not just mentioned): DeepSeek-R1-Q4, Qwen-2.5-14B, Qwen-2.5-Coder-14B, Mistral-Nemo, Chronos-Bolt, TimesFM-2, TabPFN-v2-clf, TabPFN-v2-reg, BGE-M3, mxbai-embed-large-v1, BGE-reranker-v2-m3, Snowflake-Arctic-Embed-L-v2, Qwen-2.5-VL-7B.
- **261,175 real data points** from 8 cited sources: DataCo Kaggle, NOAA IBTRACS, USGS, FRED, World Bank WGI, SEC 10-K, Wikipedia crisis articles, policy papers.
- **173 passing tests** including 19 formal OpenEnv compliance tests.
- **Every negative finding has a world-class improvement** (see [`MODEL_CARD.md §3`](https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/MODEL_CARD.md#3-honest-findings--improved-not-hidden)).

---

## Reproducibility in 3 commands

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
pip install -r requirements.txt && pytest tests/ -q   # 173 passing in 2m14s
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Sleep Token album theme 🎵

Every phase commit is named after a track from the Sleep Token albums *Even In Arcadia* (2025) and *Take Me Back to Eden* (2023):

| Track | Phase | Commit |
|---|---|---|
| Emergence | R1 — 13 SOTA models verified | `acc19d8` |
| Caramel | R2 — Tabular stack | `b35f15e` |
| Past Self | R3 — Forecasting | `c2d0798` |
| Dangerous | R4 — Risk panel | `8f14607` |
| Granite | R5 — RAG | `ca7a57d` |
| Gethsemane, Provider, Aqua Regia, Damocles, Infinite Baths | R6 — Full stack | `ea282c4` |
| Euclidian | R6 — 8,100-ep benchmark | `badf3cc` |
| Arcadia (closer) | R7 — `v3.0-arcadia` tag | |

---

## Links

- **GitHub**: https://github.com/ShAuRyA-Noodle/Sleep-Token
- **HF Space**: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
- **Release**: https://github.com/ShAuRyA-Noodle/Sleep-Token/releases/tag/v3.0-arcadia
- **Unified Model Card**: https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/MODEL_CARD.md
- **PyTorch Engineering Story**: https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/PYTORCH_STORY.md
- **Benchmarks vs Public** (M5 / MTEB / MuJoCo): https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/BENCHMARKS_VS_PUBLIC.md
- **Demo Video Script**: https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/demo/DEMO_VIDEO_SCRIPT.md
- **Pitch Deck**: https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/demo/PITCH_DECK.md
- **Colab Quickstart**: https://github.com/ShAuRyA-Noodle/Sleep-Token/blob/main/notebooks/04_v3_quickstart_colab.ipynb

Meta PyTorch OpenEnv Hackathon submission · MIT License · 2026
