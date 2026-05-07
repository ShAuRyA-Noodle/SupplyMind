# v4.0-arcadia-live — GitHub Release notes

*Tag: `v4.0-arcadia-live` · Date: 2026-04-22 · Track: "Rain" (Sleep Token, Even In Arcadia 2025)*

## What's real in this release

Every number here was produced on 2026-04-22 on a single RTX 4080 Laptop + Ollama stack. No synthetic substitution, no mocked outputs.

### 🔥 A/B benchmark: supplymind-analyst:v5 vs base Qwen-2.5-14B

On 10 hand-labeled scenarios with a deterministic rubric judge:

| Model | Exact-risk acc | Partial-risk acc | Evidence coverage |
|---|---|---|---|
| **supplymind-analyst:v5** | **80 % (8/10)** | **90 %** | **91.7 %** |
| base qwen2.5:14b | 0 % (0/10) | 5 % | 0 % |

`+0.80` exact-lift, `+0.85` partial-lift. v3's 12 % win rate is fully inverted.
→ `versions/v4_arcadia_live/features/R9_ANALYST_AB_V5.json`

### 🔥 Karpathy-style autoresearch — 5 seed experiments executed

| Seed | Status | CI95 lower | Hypothesis |
|---|---|---|---|
| s1_bigger_network | ✅ ACCEPTED | 0.4035 | MlpPolicy [256, 256] + ReLU |
| **s2_higher_entropy** | ✅ **ACCEPTED** (best) | **0.4548** | ent_coef=0.1 exploration |
| s3_curriculum_learning | rerun pending FlatDiscrete fix | — | easy→medium→hard warm-start |
| s4_recurrent_ppo | rerun pending FlatDiscrete fix | — | RecurrentPPO LSTM 128 |
| s5_action_diversity_bonus | rerun pending FlatDiscrete fix | — | +0.02 bonus for unseen actions |

`versions/v4_arcadia_live/autoresearch/AUTORESEARCH_LAB_NOTEBOOK.md` (auto-generated).

### 🔥 Qwen-VL-7B port imagery — 7 critical ports

Real Qwen2.5-VL-7B assessment via Ollama /api/generate:

| Port | Risk | Confidence | Unusual activity detected |
|---|---|---|---|
| Kaohsiung (Taiwan) | 0.30 | 0.80 | none |
| Shanghai (China) | 0.10 | 0.80 | none |
| Long Beach (US) | 0.10 | 0.80 | no ships visible |
| Rotterdam (NL) | 0.20 | 0.80 | none |
| Jebel Ali (UAE) | 0.30 | 0.70 | irregular ship pattern |
| Haifa (Israel) | 0.10 | 0.80 | calm |
| Hodeidah (Yemen) | 0.20 | 0.80 | none |

Mean confidence **0.786**. Latency 15-30 s per port.
→ `versions/v4_arcadia_live/features/port_imagery/assessments.json`

### 🔥 Stacking v2 on 60K DataCo rows (full DataCo pipeline)

| Model | AUC | F1 |
|---|---|---|
| xgboost | 0.9779 | 0.972 |
| **lightgbm (best single)** | **0.9818** | 0.973 |
| catboost | 0.9742 | 0.972 |
| random_forest | 0.9750 | 0.972 |
| logistic_regression | 0.9633 | 0.969 |
| mlp | 0.9762 | 0.972 |
| ensemble_wv_v1 | 0.9771 | 0.972 |
| **stacking_v2** | **0.9816** | 0.973 |

- Stacking beats weighted-voting by **+0.0045 AUC** (honest win, confirms v2 finding).
- Stacking ties best single within noise (−0.0002 AUC; 0.97+ ceiling task).

### 🔥 SPOF v2 (articulation-point detector)

F1 on 3 real supply-chain graphs (easy 12 nodes, medium 25, hard 40):

| Graph | v1 legacy F1 | v2 articulation F1 |
|---|---|---|
| easy | 0.889 | **1.000** |
| medium | 1.000 | **1.000** |
| hard | 0.957 | **1.000** |
| **mean** | **0.949** | **1.000** |

### 🔥 Live geopolitical pipeline

On 2026-04-22:
- **FRED Brent ingested live**: $123.28/bbl (DoD +3.54 %, WoW −3.39 %, severity 0.71).
- **NewsAPI**: 80 events across 5 queries (7-day lookback).
- **GDELT 2.0**: 60 events across 4 queries.
- **USGS**: 19 real earthquake events (M4.5+ in last 24 h, region-filtered).
- **Crisis library match**: 0.99 similarity to the 2026-04-18 Gulf-of-Oman event.
- **Counterfactual**: $324 M no-action → $65 M with plan → **80 % savings**.

### 🔥 Test suite

**250 passing, 0 skipped, 0 failed** in ~5 min (173 v3 core + 77 new v4).

### 🔥 HF Space v4 deploy — LIVE

https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
- `/tasks` → 200 (v3 endpoints intact)
- `/live/recent-events` → 200
- `/live/signal-counts` → 200
- `/live/analog-match?query=hormuz` → 200 (returns 2026-04-18 event at 0.99 similarity)
- `/docs` → Swagger UI live

## 20 new v4 modules (all tested, 77 new tests, 250 total)

- **autoresearch/** — Karpathy-pattern loop (9 files): program.md, candidate_train.py, hypothesis_engine, runner, evaluator, lab_notebook, orchestrator, seed_experiments, rerun_seeds
- **realtime/** — Live Hormuz pipeline (9 files): store (SQLite), 5 sources (NewsAPI/GDELT/USGS/MarineTraffic/FRED), ingestor, crisis_library, hormuz_endpoint (mounted on server/app.py)
- **scenarios/** — 8 real 2024-2026 events, 26 citations
- **features/** — 16 unique modules:
  - `spof_v2.py` (G8) — articulation-point SPOF
  - `stacking_v2.py` (G15) — 6-learner OOF stacking framework
  - `analyst_ab_bench.py` + `Modelfile.analyst_v5` (G9) — 10-scenario rubric-judged bench
  - `receipts.py` (F10) — 15 auto-generated reproducibility receipts
  - `gcn_attention_viz.py` (F7) — betweenness + flow edge importance
  - `counterfactual_explainer.py` (F3) — template + LLM counterfactual
  - `pareto_carbon.py` (F9) — multi-objective Pareto w/ EPA/IMO/ICAO emission factors
  - `rag_provenance.py` (F8) — 5-tier trust classifier + graph viz
  - `conformal_rl.py` (F6) — split-conformal Q-value intervals
  - `leaderboard.py` (F5) — Gradio + HTTP submissions
  - `qwen_vl_port_imagery.py` (G3+F1) — 7-port Qwen-VL assessment
  - `multi_agent_demo.py` (G4+F2) — Apple/Samsung/Toyota chip-shortage sim
  - `dt_risk_slider.py` (G6+F4) — 3-slider behavior comparison
  - `cuda_kernel_verify.py` (G14) — PyTorch fallback benchmark + JIT attempt
  - `lora_train.py` (G7) — QLoRA 4-bit NF4 harness for Qwen-14B
- **docs/** — EXTERNAL_OUTREACH, PREPRINT, SECRETS_ROTATION, LIVE_DEMO_HORMUZ, PHOENIX_PLAN_V5
- **deploy/** — HF_DEPLOY_V4, PITCH_DECK_V4
- **receipts/** — 15 committed receipts (13 v3-era + 2 v4-era numbers)

## How to verify (judges, 60 seconds)

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
pip install -r requirements.txt
pytest tests/ versions/v4_arcadia_live/tests/ -q  # 250 passing

# Reproduce the two v4 headline numbers:
bash versions/v4_arcadia_live/receipts/V4_Analyst_V5_Exact_Acc.reproduce.sh  # -> 0.8
bash versions/v4_arcadia_live/receipts/V4_Autoresearch_Best_CI95.reproduce.sh # -> 0.4548

# Start the server and hit the live Hormuz endpoint:
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
curl -X POST http://localhost:8000/live/hormuz-closure \
  -H 'Content-Type: application/json' \
  -d '{"scenario_text":"Iran threatens full Hormuz closure","region":"hormuz"}'
```

Or visit the live HF Space: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind

## Credits

Built solo by ShAuRyA-Noodle for the Meta PyTorch OpenEnv Hackathon 2026. No compromise. Real data everywhere.

*"Arcadia is the closer. This is where we end."*
