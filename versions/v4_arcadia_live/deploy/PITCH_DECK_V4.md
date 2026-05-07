# SupplyMind v4.0-arcadia-live — Pitch Deck

> 7 slides. Render to HTML with `python scripts/build_pitch_html.py` (existing
> v3 tool). Print-to-PDF for offline sharing.

---

## Slide 1 — Title

> # SupplyMind v4.0-arcadia-live
>
> **Real-time supply-chain risk intelligence on one laptop.**
>
> Built solo for Meta PyTorch OpenEnv Hackathon 2026. No cloud, no compromise.
>
> *"Even in Arcadia, supply chains break — and now we watch it happen live."*

---

## Slide 2 — The Problem

- **$184 billion** in supply-chain disruptions in 2023 alone (Business Continuity Institute).
- **2024 Iran-Israel direct attacks**, **2023-ongoing Houthi Red Sea campaign**, **2026 Gulf-of-Oman tanker seizure** — disruption is no longer an occasional event.
- Incumbents (SAP IBP, Oracle SCM, Resilinc) are **dashboards after the fact**. They tell you what already broke.
- The research community has never shipped an OpenEnv-compliant supply-chain environment with 261K real data points + live ingestion.

---

## Slide 3 — The Architecture

```
                                     Judges (Meta LLM scorer + programmatic)
                                                       ↓
         OpenEnv spec → 19 formal tests → /reset /step /grade /baseline /mcp /ws
                                                       ↓
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  server/  (Python + FastAPI + Pydantic v2)                               │
 │                                                                           │
 │  R2 Caramel — TabPFN + XGB + LGB + CAT + Ridge stacking (SHAP + fairness) │
 │  R3 Past Self — Chronos + TimesFM + ARIMA + Prophet, per-horizon conformal │
 │  R4 Dangerous — 3-judge LLM panel (Krippendorff α = 0.750)                │
 │  R5 Granite — 6,483-chunk RAG, mxbai P@1 = 0.962                          │
 │  R6 Gethsemane — MaskablePPO, +26.8% masking lift, structural zero-invalid │
 │  R6 Provider — custom PyTorch GCN, −48/−49/−64% MAE vs MLP                 │
 │  R6 Aqua Regia — split-conformal, WTI dev 0.024 from 95% nominal            │
 └─────────────────────────────────────────────────────────────────────────┘
                                                       ↓
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  versions/v4_arcadia_live/ (v4 arcadia-live)                                    │
 │                                                                           │
 │  L1 autoresearch/  ← Karpathy-pattern autonomous research loop            │
 │  L2 realtime/       ← NewsAPI + GDELT + USGS + FRED + MarineTraffic       │
 │  scenarios/         ← 8 real Iran/Israel/Hormuz events, 26 citations      │
 │  features/          ← 17 unique modules (SPOF, stacking, Pareto, etc.)    │
 │  receipts/          ← 13 one-command verification scripts                 │
 └─────────────────────────────────────────────────────────────────────────┘
```

**13 foundation models locally (159 GB).** **249+ passing tests.** **0 cloud APIs at inference.**

---

## Slide 4 — Live Demo (the 90-second win)

**ONE CURL COMMAND** — hits real 2026 news:

```bash
curl -X POST http://localhost:8000/live/hormuz-closure -d '{
  "scenario_text": "Iran threatens full Hormuz closure after US Navy seizes Iranian cargo ship in Gulf of Oman. Brent crude $123/bbl. Carriers pause Persian Gulf bookings.",
  "region": "hormuz",
  "enable_llm_judges": true
}'
```

Returns in <15 seconds (Ollama warm):

- **Top analog**: `hormuz_trump_cargo_ship_2026_04` at 0.99 similarity to the real 2026-04-18 event
- **3-judge panel**: Qwen-14B + Mistral-Nemo + DeepSeek-R1 — CRITICAL consensus
- **Brent projection**: P50 $142/bbl, P95 $168/bbl
- **5 ranked actions** with cost + loss-avoided in actual dollars
- **Counterfactual**: $324M no-action → $65M with plan = **80% savings**

---

## Slide 5 — The Receipts

| # | Claim | Value | Verify |
|---|-------|-------|--------|
| 1 | RAG P@1 on 6,483-chunk real corpus | **0.9622** | `bash receipts/R5_GRANITE_mxbai_P1.reproduce.sh` |
| 2 | 2-judge Krippendorff α on 26 crisis scenarios | **0.7499** | `bash receipts/R4_2JUDGE_Krippendorff_alpha.reproduce.sh` |
| 3 | MaskablePPO easy-task lift over plain PPO | **26.77 %** | `bash receipts/R6_MaskingAblation_easy_lift.reproduce.sh` |
| 4 | GCN easy-graph MAE reduction vs MLP | **48.0 %** | `bash receipts/R6_GCN_easy_MAE_vs_MLP.reproduce.sh` |
| 5 | Per-horizon conformal deviation at 95% nominal (WTI) | **0.0238** | `bash receipts/R6_AquaRegia_WTI_dev95.reproduce.sh` |
| 6 | v4 SPOF articulation F1 (vs v1 F1=0.949) | **1.000** | `bash receipts/V4_SPOF_V2_F1.reproduce.sh` |
| 7 | Live Brent price ingested 2026-04-21 | **$123.28** | `bash receipts/V4_Live_Brent_202604.reproduce.sh` |

Every receipt captures command + git SHA + data file hashes. Judges can verify any number in **30 seconds**.

---

## Slide 6 — Honest Findings (why you should trust us)

1. **R2 stacking vs best single: null result.** On DataCo `late_delivery_risk` (AUC ~0.97), stacking beats WV by +0.001 but does NOT beat best-single LightGBM within CI95. We publish the null. `versions/v4_arcadia_live/features/R15_STACKING_V2.json`.
2. **supplymind-analyst v3 lost 12% A/B vs base Qwen.** Fix shipped as Modelfile v5 with calibrated few-shots + hard negatives. `versions/v4_arcadia_live/features/analyst_ab_bench.py`.
3. **CUDA kernel never loaded in production.** PyTorch fallback is 0.034ms at batch=1024, 42,778× faster than naive Python. Custom kernel was pedagogical; fallback is production-ready. `versions/v4_arcadia_live/features/F14_CUDA_KERNEL.json`.
4. **DT risk-appetite slider uses a surrogate** where the v2 DT checkpoint is heavy to load. Same conditioning pattern, faster to demo. Clearly labeled.
5. **Qwen-VL-7B** is loaded but only the heuristic path is benchmarked here — the full VL satellite assessment requires pulling the 15 GB model first. Plug-and-play if you do.

---

## Slide 7 — Call to Action

**Code**: https://github.com/ShAuRyA-Noodle/Sleep-Token
**Live demo**: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
**Preprint**: `versions/v4_arcadia_live/docs/PREPRINT.md`
**Reproduce any headline in 30s**: `versions/v4_arcadia_live/receipts/INDEX.md`

**One person. Two months. One laptop. No cloud. Real data everywhere.**

Submit your own agent to `challenges/R4_RUBRIC_CHALLENGE.md` and beat the 2-judge α = 0.750 baseline. We publish everything.

---

*Sleep Token (Even In Arcadia, 2025) — all phase commits named after album tracks. "Arcadia is the closer. This is where we end.""*
