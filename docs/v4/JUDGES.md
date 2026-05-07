# Judges' Quick Reference

> If you only have 4 minutes, start here. This file is hand-maintained for the
> Meta PyTorch OpenEnv Hackathon 2026 finals.

## 30-second pitch

**SupplyMind v5.0-phoenix** is an OpenEnv-compliant supply-chain risk management environment. **13 local SOTA foundation models + 18-model OpenRouter frontier panel**, 261K real data points, **272 passing tests (274 collected)**, and a **live geopolitical pipeline** that polls real-time news from NewsAPI / GDELT / USGS / FRED and feeds a 9-judge ordinal-Krippendorff-scored panel (3 local + 6 frontier via Hermes-3-405B / gpt-oss-120b / Gemma-4-31B / Ling-2.6-1T / Nemotron-3-Super / Qwen3-Next-80B) against the real 2026 Iran / Israel / Hormuz crisis.

### Real execution numbers (2026-04-22 run):
- **supplymind-analyst:v5** wins **8/10 (80%)** exact-risk on the A/B benchmark vs base Qwen-2.5-14B **0/10 (0%)**. Evidence coverage **91.7% vs 0%**.
- **Karpathy autoresearch** loop accepted 2/5 seed hypotheses; best CI95-lower **0.4548** (s2_higher_entropy).
- **Qwen-VL-7B** real assessment of all 7 critical ports (Kaohsiung, Shanghai, Long Beach, Rotterdam, Jebel Ali, Haifa, Hodeidah) with mean confidence 0.786.
- **Live Brent** ingested from FRED on 2026-04-21 at **$123.28/bbl** via the v4 realtime ingestor.
- **SPOF F1**: legacy 0.949 → v2 **1.000** on 3 real supply-chain graphs.
- **Stacking v2** on 60K DataCo rows: +0.0045 AUC vs legacy weighted voting.

## 15 headline receipts (every one produces the real value in 30 seconds)

```bash
# After `pip install -r requirements.txt`, run any of these:
bash versions/v4_arcadia_live/receipts/R5_GRANITE_mxbai_P1.reproduce.sh        # -> 0.9622
bash versions/v4_arcadia_live/receipts/R5_GRANITE_mxbai_MRR.reproduce.sh       # -> 0.9780
bash versions/v4_arcadia_live/receipts/R5_BEIR_snowflake_nDCG10.reproduce.sh   # -> 0.9710
bash versions/v4_arcadia_live/receipts/R4_2JUDGE_Krippendorff_alpha.reproduce.sh  # -> 0.7499
bash versions/v4_arcadia_live/receipts/R4_Cohen_kappa_QwenMistral.reproduce.sh # -> 0.7474
bash versions/v4_arcadia_live/receipts/R6_MaskingAblation_easy_lift.reproduce.sh # -> 26.77 (%)
bash versions/v4_arcadia_live/receipts/R6_GCN_easy_MAE_vs_MLP.reproduce.sh     # -> 48.02 (%)
bash versions/v4_arcadia_live/receipts/R6_AquaRegia_WTI_dev95.reproduce.sh     # -> 0.0238
bash versions/v4_arcadia_live/receipts/R3_TimesFM_CP_WTI_dev95.reproduce.sh    # -> 0.0500
bash versions/v4_arcadia_live/receipts/V4_SPOF_V2_F1.reproduce.sh              # -> 1.0
bash versions/v4_arcadia_live/receipts/V4_STACKING_V2_lift_vs_WV.reproduce.sh  # -> 0.0045
bash versions/v4_arcadia_live/receipts/V4_Analyst_V5_Exact_Acc.reproduce.sh    # -> 0.8 (v5 exact lift vs base Qwen)
bash versions/v4_arcadia_live/receipts/V4_Autoresearch_Best_CI95.reproduce.sh  # -> 0.4548 (autoresearch winner)
bash versions/v4_arcadia_live/receipts/V4_Live_Brent_202604.reproduce.sh       # -> 123.28 ($/bbl on 2026-04-21)
bash versions/v4_arcadia_live/receipts/V4_Tests_Total.reproduce.sh             # -> pytest collection listing
```

All 15 receipts are in `versions/v4_arcadia_live/receipts/INDEX.md`.

## The live Hormuz demo (90 seconds, on my laptop)

```bash
# 1. Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 2. Ingest real-time events (NewsAPI + GDELT + USGS + FRED Brent)
python -m versions.v4_arcadia_live.realtime.ingestor --once --skip marinetraffic
# -> ~150 real 2026 events fetched in <30s

# 3. Live assessment — this hits REAL 2026 news
curl -X POST http://localhost:8000/live/hormuz-closure \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_text": "Iran threatens full Hormuz closure after US seizes Iranian cargo ship. Brent $123/bbl.",
    "region": "hormuz",
    "enable_llm_judges": true,
    "include_recent_signals": true,
    "k_analogs": 3
  }' | jq
```

Expected response:
- Top analog match = `hormuz_trump_cargo_ship_2026_04` at **0.99 similarity**
- Risk level = HIGH or CRITICAL
- 5 recommended actions (hedge, reroute, backup, safety-stock, alert)
- **Counterfactual: $324M no-action loss → $65M with plan = 80% savings**
- 3-judge LLM panel output (if Ollama warm) or rubric fallback

## Sub-4-minute judge path

1. **[30s] Read this file's top section** (you're here).
2. **[60s] Watch the live Hormuz demo** in `versions/v4_arcadia_live/docs/LIVE_DEMO_HORMUZ.md`.
3. **[60s] Pick any 3 receipts** from the top-10 list above and run them.
4. **[30s] Read the preprint abstract** at `versions/v4_arcadia_live/docs/PREPRINT.md` §Abstract.
5. **[30s] Run the test suite**: `pytest tests/ versions/v4_arcadia_live/tests/ versions/v5_phoenix/tests/ -q` (272 passing, 2 skipped, 274 collected as of 2026-04-24).

## What's unique to v4 (vs v3.0-arcadia)

| # | Feature | Where |
|---|---------|-------|
| 1 | **Karpathy-style autonomous research loop** — `program.md` driven, fixed-budget, bootstrap-CI95 accept/reject | `versions/v4_arcadia_live/autoresearch/` |
| 2 | **Live geopolitical ingestion** — 5 real data sources, SQLite store | `versions/v4_arcadia_live/realtime/` |
| 3 | **Real crisis library** — 8 Iran/Israel/Hormuz 2024-2026 events with 26 citations | `versions/v4_arcadia_live/scenarios/` |
| 4 | **Fixed SPOF detector** — F1 0.949 → 1.000 on 3 real graphs | `versions/v4_arcadia_live/features/spof_v2.py` |
| 5 | **Proper stacking framework** — OOF + meta-learner on DataCo | `versions/v4_arcadia_live/features/stacking_v2.py` |
| 6 | **Reproducibility receipts** — every headline number gets `.receipt` + `.reproduce.sh` | `versions/v4_arcadia_live/receipts/` |
| 7 | **GCN attention viz** — edge betweenness + flow importance | `versions/v4_arcadia_live/features/gcn_attention_viz.py` |
| 8 | **Counterfactual explainer** — what-if loss projection with analog lookup | `versions/v4_arcadia_live/features/counterfactual_explainer.py` |
| 9 | **Pareto carbon slider** — cost/resilience/CO₂ multi-objective | `versions/v4_arcadia_live/features/pareto_carbon.py` |
| 10 | **RAG provenance graph** — citation trust tiers | `versions/v4_arcadia_live/features/rag_provenance.py` |
| 11 | **Conformal-calibrated RL** — split-conformal intervals on Q-values | `versions/v4_arcadia_live/features/conformal_rl.py` |
| 12 | **Gradio leaderboard** — external submission harness | `versions/v4_arcadia_live/features/leaderboard.py` |
| 13 | **Qwen-VL port imagery** — 7-port satellite assessment | `versions/v4_arcadia_live/features/qwen_vl_port_imagery.py` |
| 14 | **Multi-agent competition** — Apple/Samsung/Toyota chip-shortage dynamic | `versions/v4_arcadia_live/features/multi_agent_demo.py` |
| 15 | **DT risk-appetite slider** — conservative/balanced/aggressive surrogate | `versions/v4_arcadia_live/features/dt_risk_slider.py` |
| 16 | **CUDA kernel verify** — PyTorch fallback benchmark (0.034ms at B=1024) | `versions/v4_arcadia_live/features/cuda_kernel_verify.py` |
| 17 | **LoRA training harness** — supplymind-analyst v5 with real rubric data | `versions/v4_arcadia_live/features/lora_train.py` |
| 18 | **Modelfile v5** — 8 calibrated few-shots + A/B benchmark | `versions/v4_arcadia_live/features/Modelfile.analyst_v5` |
| 19 | **Arxiv-style preprint** — consolidated 1-page technical summary | `versions/v4_arcadia_live/docs/PREPRINT.md` |
| 20 | **External-outreach playbook** | `versions/v4_arcadia_live/docs/EXTERNAL_OUTREACH.md` |

## The 3 things to ask me in person

1. **"Show me the live Hormuz assessment with the judge panel."** — 90 seconds, on my laptop, hitting real 2026 NewsAPI + FRED Brent.
2. **"Which of your results is the most surprising?"** — Honest answer: R6 Aqua Regia per-horizon conformal (0.024 dev from 95% nominal on oil, 4.7x tighter than pooled). It's textbook methodology realized on real FRED data.
3. **"Where does SupplyMind fail?"** — See `docs/v3/BENCHMARKS_VS_PUBLIC.md` §8 (honest limitations) and the R2 stacking null result (`versions/v4_arcadia_live/features/R15_STACKING_V2.json`). When base learners hit a 0.97+ AUC ceiling, stacking doesn't beat best single. We publish the null.

## If anything fails

- **`pytest` fails**: file a GitHub issue at github.com/ShAuRyA-Noodle/Sleep-Token/issues with the `test output` + `python --version` + `platform.system()`.
- **HF Space offline**: the GitHub repo has everything; clone and run locally.
- **NewsAPI rate-limited**: the crisis library (`versions/v4_arcadia_live/scenarios/iran_israel_hormuz_2024_2026.json`) has 8 hand-curated events with 26 citations, so the demo works offline.
- **Ollama not running**: the `/live/hormuz-closure` endpoint falls back to a deterministic rubric judge; everything else (actions, counterfactual, RAG) is unaffected.

## Reproducibility guarantees

- 272 tests pass + 2 skipped = 274 collected (last verified 2026-04-24).
- Every headline number has a committed receipt.
- All data is public + cited: DataCo (Kaggle), NOAA IBTRACS, FRED, SEC EDGAR, Wikipedia, World Bank, BIS, CNBC, Reuters, IDF, CFR, UNCTAD.
- The 2026-04-18 Hormuz scenario in the crisis library is anchored to a REAL NewsAPI article ingested on 2026-04-21.

---

*Contact: see README.md. Built solo. No compromises. Real data everywhere.*
