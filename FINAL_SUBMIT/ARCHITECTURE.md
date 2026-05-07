# SupplyMind Architecture

## High-level

```
                    ┌───────────────────────────────────┐
                    │      OpenEnv-compliant API        │
                    │   /reset /step /state /tasks      │
                    │   /grader /baseline (port 8000)   │
                    └───────────────┬───────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
   ┌──────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
   │ Game Engine       │ │ 9-agent Arena   │ │ Live Intel Fan-out  │
   │ 3 tasks · 7 acts  │ │ RAP-XC + 8 base │ │ 20 sources          │
   │ × 40 nodes = 280  │ │  + leaderboard  │ │ (NewsAPI/GDELT/USGS │
   │ 7-comp reward     │ │  CI95 paired    │ │  /NOAA/NASA/EIA/    │
   │ BFS cascade decay │ │  bootstrap      │ │  GFW/MarineTraffic/ │
   └────────┬──────────┘ └────────┬────────┘ │  WHO DON/SEC/CISA/  │
            │                     │          │  OFAC/WB/Wiki/HN)   │
            │                     │          └──────────┬──────────┘
            ▼                     ▼                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Hormuz War Room (POST)                     │
   │                     6-stage orchestrator                       │
   ├─────────────────────────────────────────────────────────────────┤
   │  1. Live signals (graceful)                                    │
   │  2. Crisis Library v2 RAG (1500 events, mxbai+FAISS HNSW)      │
   │  3. Chokepoint graph (14 nodes, IEA bypass ceilings)           │
   │     ↓                                                          │
   │     HetTemporalGAT cascade rollout (4 edge types, GRU)         │
   │  4. India + Gulf 7-sector exposure tables (deterministic)      │
   │     + 6-frontier OpenRouter judge cross-check (Krippendorff α) │
   │  5. Ensemble Brent forecaster                                  │
   │     ↓                                                          │
   │     Chronos-Bolt + TimesFM-2 + TabPFN-v2 weighted blend        │
   │  6. Action recommendation                                      │
   │     ↓                                                          │
   │     Hierarchical 4-intent → 280-action mask                    │
   │     ↓                                                          │
   │     RAP-XC policy (3.14M params) + judge prior bias            │
   │     ↓                                                          │
   │     Split-conformal NLL filter (0.9001 coverage)               │
   │  → sha256-anchored receipt                                     │
   └─────────────────────────────────────────────────────────────────┘
```

## Subsystems

### 1. OpenEnv game engine

- 3 tasks: `easy_typhoon_response` (12 nodes, 30 days, ₹40 cr), `medium_multi_front` (25 nodes, 45 days, ₹65 cr), `hard_cascading_crisis` (40 nodes, 60 days, ₹83 cr)
- Action space: 7 discrete action types × 40 node targets = 280 (MultiDiscrete flattened)
- 7-component reward: revenue 35% · stockout 25% · proactive 15% · cost 10% · health 5% · SLA 5% · unnecessary 5%
- Time-discounted: `max(0.3, 1.0 - step_fraction × 0.7)`
- BFS disruption propagation, `SEVERITY_DECAY_PER_HOP = 0.20`
- 9 industry-cited cost constants in `server/engine/financial.py` (ISM/IATA/CSCMP)
- 5 historical crisis-library disruptions for analog matching (Tohoku/Suez/Chip/Ukraine-neon/Red-Sea)
- FastAPI on port 8000, lifespan pre-warm, CORS=*, 20-session LRU pool, asyncio.Lock

### 2. 13 foundation models (all local under `models/`)

| Tier | Model | Role |
|---|---|---|
| **LLM judges** | Qwen-2.5-14B (Q4_K_M) | primary judge / HyDE / JSON extractor |
|  | Mistral-Nemo-2407 (Q4_K_M, 128K) | long-context judge |
|  | DeepSeek-R1-Q4 (4.5GB) | devil's advocate judge |
|  | Qwen-2.5-Coder-14B | action JSON validator |
|  | Qwen-2.5-VL-7B | port imagery vision |
| **Forecasters** | Chronos-Bolt-base | zero-shot quantile Brent |
|  | TimesFM-2 (50L/1280h/16H/2048ctx) | zero-shot 30-day Brent |
|  | TabPFN-v2 regressor | severity-conditioned Δ-Brent |
|  | TabPFN-v2 classifier | tabular risk-tier ensemble |
| **Embedders** | mxbai-embed-large (winner, P@1=0.962) | crisis library v2 primary |
|  | BGE-M3 | RAG fallback (safetensors-converted, CVE-2025-32434 workaround) |
|  | Snowflake-Arctic-Embed-L | embedding diversity |
|  | BGE-reranker-v2-m3 | cross-encoder rerank (planned in war-room v2) |

### 3. 5 custom Ollama analyst models

| Version | What changed | Result |
|---|---|---|
| v1 (deprecated) | base Qwen + 5 in-prompt examples | superseded |
| v2 | improved system prompt + better domain knowledge | iterating |
| v3 | + action costs ($25K/day SLA, Red Sea +10d/+25% fuel) | iterating |
| v4 | 10-shot prompting, used in Phase-4 R3 | iterating |
| **v5** | 8 hard-negative few-shots + calibrated prompt + JSON-format guarantee | **80% exact-risk vs base 0%** |

Plus 4 base wrappers: `qwen25-14b-local`, `qwen25-coder-local`, `mistral-nemo-local`, `deepseek-r1-local-q4`.

5 Modelfiles committed at `rl/lora/Modelfile`, `Modelfile.v2-v4`, `versions/v4_arcadia_live/features/Modelfile.analyst_v5`.

### 4. LoRA fine-tuning track

Qwen-2.5-1.5B → PEFT/LoRA → 4-bit NF4 (bitsandbytes) → TRL → 225 instruction/output pairs at `rl/data/lora_training_data.json` → 20MB adapter checkpoints at `rl/checkpoints/lora/` → `create_ollama_model.py` converts to Modelfile.

### 5. DPO fine-tuning track (Phoenix v5 + ROLL)

Qwen-2.5-3B-Instruct base. 21 preference pairs from R4 ground truth at `dpo_judge/data/preference_pairs.jsonl`. DPO sigmoid loss, β=0.1, LoRA r=8 / α=16, hf strategy (single-GPU 12GB), per_device_train_batch_size=1, gradient_accumulation_steps=4, lr=5e-5, save_adapter_only.

5 trainers in `versions/v5_phoenix/roll_integration/dpo_judge/`:
- `train_dpo_trl.py` — TRL standalone (ROLL-free fallback)
- `train_dpo_roll.py` — ROLL-integrated
- `train_grpo_env.py` — GRPO multi-turn
- `train_grpo_live_env.py` — live-env GRPO
- `evaluate_delta.py` — base vs fine-tuned delta

### 6. ROLL integration (Alibaba framework)

- `SupplyMindRollEnv` wraps SupplyMind as ROLL agentic environment
- `SupplyMind3JudgeRewardWorker` exposes 3-judge panel as ROLL reward worker
- `agentic_supplymind_gigpo.yaml` GiGPO multi-turn config
- 3 tools: `forecast`, `rag`, `rl_act`
- Step-wise reward (not just episode-end)
- Single-GPU 12GB ergonomics
- Class importable without ROLL (graceful degrade)

### 7. Quantization & memory engineering

- Q4_K_M 4-bit: DeepSeek-R1 15GB → 4.5GB (3.3× compression, <2% quality loss documented)
- Industry-standard format (matches llama.cpp output)
- `OLLAMA_MAX_LOADED_MODELS=1` env var prevents VRAM contention
- BGE-M3 safetensors conversion bypasses torch 2.6 CVE-2025-32434 (`weights_only=True`)
- 2GB safetensors output at `models/bge-m3/model.safetensors`
- CUDA host-pinned memory discipline

### 8. RAP-XC retrieval-augmented policy

```
state_feats (64-d) ──→ StateEncoder
crisis_embeds (k=8 × 1024) ──→ CrisisProjector
dag_feats (80-d) ──→ DAGEncoder
                        ↓
              4-layer MHA cross-attention
                        ↓
              fusion + ActionHead (280) + ValueHead
                        ↓
              + frozen judge-prior bias on logits
```

3.14M params · trained on 40,000 real harvested transitions (top-50% return filter → 20,000) · BC + KL + value-MSE + CQL conservative loss · 12 epochs · 17.77s on RTX 4080 (bf16).

### 9. Hierarchical + Conformal action selection

- Level 1 (deterministic 4-intent picker): PROTECT_BUDGET / DIVERSIFY_RISK / EXPEDITE / ABSORB_AND_MONITOR — narrows 280 actions to 80-160 strategy-coherent
- Level 2 (split-conformal NLL filter): finite-sample correction, 8000-row calibration set, **empirical coverage 0.9001**
- Argmax fallback: policy never starves

### 10. HetTemporalGAT cascade predictor

- Edge-type-conditional attention over 4 edge types (SHIPS_TO / SUPPLIES / ROUTES_VIA / ALTERNATE_TO)
- 4-head Velickovic-style GAT with group-softmax
- GRUCell temporal gating fuses node embedding(t) with hidden(t-1)
- 19,489 params · beats v1 GCN: easy +7.77% / medium +12.15% / hard +10.03% MAE

### 11. 4-method Platinum counterfactual

- Paired-bootstrap MC + Synthetic Control + ARIMA-BSTS + SCM do-calculus
- Calibrated to 6 paper anchors: Suez 2021 $9.6B/day · Tohoku $235B · Chip shortage $210B · Ukraine neon 45-65% global · Red Sea 2023 · Iran sanctions oil
- **Tohoku replicated $276 B vs $235 B published (+18%)**
- Returns consensus loss + CI95

### 12. Crisis library v2 (1500 events)

- Auto-cooked from real EMDAT 16,812 disasters
- Deterministic severity tier from real death/damage/affected counts
- mxbai-embed-large 1024-d FAISS HNSW index
- `library_v2_search.singleton(query, k=8)` returns ranked analogs

### 13. 25-judge ensemble + 12-frontier panel + 6-OpenRouter cross-check

- 13 local Ollama models + 12 OpenRouter frontier
- Krippendorff α (ordinal) on R4 corpus = 0.5669
- Cross-corpus α on v2 EMDAT (30 stratified events) = 0.5436
- 6-judge subset for war-room: gpt-oss-120b, gemma-4-31b, glm-4.5-air, minimax-m2.5, nemotron-3-super, gemma-4-26b

### 14. Live data layer (20 sources)

`versions/v4_arcadia_live/realtime/orchestrator_v2.py` fans out to 20 sources via ThreadPoolExecutor with per-source timeouts and graceful failure:

NewsAPI · GDELT · GDELT-Conflict · GDELT-Humanitarian · USGS earthquakes · NOAA NDBC buoys · NOAA Tides · NASA EONET · NASA FIRMS fires · EIA Brent · EIA WTI · EIA natgas · MarineTraffic AIS · Global Fishing Watch · World Bank commodities · WHO DON · SEC EDGAR · CISA KEV · OFAC sanctions · Wikipedia pageviews · HN tech ticker

## Data flow for one war-room request

```
POST /demo/hormuz-war-room
  ↓
Stage 1 (≤8s)  hormuz_signal_filter.fan_out()  →  evidence + cosines
Stage 2 (≤3s)  library_v2_search(query, k=8)  →  8 EMDAT analogs
Stage 3 (≤4s)  hetgat.rollout(chokepoint_graph, 5d)  →  cascade map
Stage 4 (≤6s)  india_exposure + gulf_exposure scorers (deterministic, instant)
              + openrouter_panel async × 6  →  ranked tables + α
Stage 5 (≤8s)  forecast_v2.ensemble_brent(history, severity, horizon)
              → Chronos+TimesFM+TabPFN p10/p50/p90
Stage 6 (≤2s)  /agent/decide  →  RAP-XC → hierarchical → conformal → top-K
Stage 7 (≤1s)  receipts_v2.assemble()  →  sha256 receipt
————————————————————————————————————————————————————————————————
Total budget: ~32s  (current observed 22-39s)
```

## Why this is paper-grade

> Retrieval-augmented policy that conditions on a 1500-event historical disaster corpus via FAISS cross-attention, with a 25-model judge ensemble distilled into action-logit priors, against a 4-method causal counterfactual ensemble (paired-bootstrap MC + synthetic control + ARIMA-BSTS + SCM do-calculus) calibrated to 6 published economic-impact anchors, on an OpenEnv-compliant supply-chain RL environment with 20 real-data live sources, evaluated against 7 RL/IL baselines with paired-bootstrap CI95, with hierarchical-intent + split-conformal action selection (0.9001 empirical coverage), heterogeneous-temporal GAT cascade prediction (+12.15% MAE vs GCN baseline), Chronos-Bolt + TimesFM-2 + TabPFN-v2 ensemble forecaster (3.32% median Brent backtest error on 8 documented events), all running locally on a 12GB GPU with zero synthetic substitution.

Every clause maps to a committed file with a live test.
