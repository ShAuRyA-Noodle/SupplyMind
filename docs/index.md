---
title: SupplyMind v3.0-arcadia
layout: default
description: OpenEnv-compliant supply-chain risk management · 261,175 real data points · 173 tests
---

# SupplyMind v3.0-arcadia

**Mirror of the canonical submission.** Authoritative repo: [ShAuRyA-Noodle/Sleep-Token](https://github.com/ShAuRyA-Noodle/Sleep-Token) · HF Space: [Supplymind](https://huggingface.co/spaces/Shaurya-Noodle/Supplymind)

![Hero card](../v3_arcadia/plots/hero_result_card.png)

## 30-second summary

SupplyMind is a Meta PyTorch OpenEnv-compliant AI stack that **sees supply chain disruptions forming, quantifies the risk with rigorous statistics, and recommends pre-emptive actions** — all from 261,175 real data points across 8 public authoritative sources, with every headline number reproducible by a single shell command.

## Ten headline numbers

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

## Verify any number

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git && cd Sleep-Token
pip install -r requirements.txt
python scripts/run_all.py    # 12 claims → all PASS
pytest tests/ -q              # 173 tests → all PASS
```

## Deep dives

- [MODEL_CARD](../MODEL_CARD) — every benchmark, every design win
- [RESULTS](../RESULTS) — one-page hero with verification commands
- [comparison](../comparison) — category-by-category hackathon positioning
- [BENCHMARKS_VS_PUBLIC](../BENCHMARKS_VS_PUBLIC) — vs MTEB, M5, MuJoCo, LLM-as-judge
- [PYTORCH_STORY](../PYTORCH_STORY) — 11 non-trivial engineering items
- [EXTERNAL_CREDIBILITY](../EXTERNAL_CREDIBILITY) — 10+ cited published sources
- [DEPLOY_HF_SPACE](../DEPLOY_HF_SPACE) — phoenix rebuild guide
- [challenges/R4_RUBRIC_CHALLENGE](../challenges/R4_RUBRIC_CHALLENGE) — reproducibility invitation

## Architecture

```
Real data ──▶ R1 Emergence  (13 foundation models, local-only inference)
           ──▶ R2 Caramel    (TabPFN + XGB + LGB + CAT + Ridge meta-learner)
           ──▶ R3 Past Self  (Chronos + TimesFM + ARIMA + Prophet + Bates-Granger stack)
           ──▶ R4 Dangerous  (DeepSeek + Qwen-14B + Mistral-Nemo 3-judge, α=0.75)
           ──▶ R5 Granite    (mxbai/BGE-M3/Snowflake/reranker/HyDE, 8 pipelines, P@1=0.962)
           ──▶ R6-α Gethsem. (MaskablePPO + ONNX, +26.8%)
           ──▶ R6-β Euclidian(8,100-ep bootstrap CI95 non-overlapping)
           ──▶ R6-γ Provider (custom 3-layer GCN, -48/-49/-64% MAE vs MLP)
           ──▶ R6-δ Aqua Reg.(per-horizon split-conformal, dev=0.024 on WTI)
           ──▶ R7 Arcadia    (OpenEnv server + MCP + Docker + HF Space + CI)
```

## License

MIT (code) · CC BY-SA 4.0 (Wikipedia-sourced text) · Dataset-specific licenses for each of the 8 public sources (documented in [DATA_SOURCES](../DATA_SOURCES)).

---

*Built for the Meta PyTorch OpenEnv Hackathon · Each phase commit named after a Sleep Token track from "Even In Arcadia" (2025) and "Take Me Back to Eden" (2023).*
