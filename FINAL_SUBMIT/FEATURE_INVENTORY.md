# SupplyMind Feature Inventory

Verification: every bullet point in the project plan mapped to file:line.

**Stats:** 116 PRESENT (was 113) · 3 PARTIAL (was 6) · 1 MISSING

## Recently moved PARTIAL → PRESENT (pass 10)

| Component | Previous | Now wired in |
|---|---|---|
| Chronos-Bolt-base | PARTIAL (verify only) | `versions/v5_phoenix/forecast_v2/ensemble_brent.py:53-71` |
| TimesFM-2 | PARTIAL (verify only) | `versions/v5_phoenix/forecast_v2/ensemble_brent.py:74-99` |
| TabPFN-v2 regressor | PARTIAL (verify only) | `versions/v5_phoenix/forecast_v2/ensemble_brent.py:101-145` |

Closed Brent backtest gap from 6/8 to **8/8 within ±30%** (median rel err 3.3%).
See `tests/receipts/ensemble_brent_validation.json`.

---

## A.1 Custom Ollama Models

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| supplymind-analyst:v1 | MISSING | — | only v2-v5 retained; v1 superseded |
| supplymind-analyst:v2-v5 | PRESENT | `rl/lora/Modelfile.v2:1-20`, `Modelfile.v3:1-20`, `Modelfile.v4:1-20`, `versions/v4_arcadia_live/features/Modelfile.analyst_v5:1-20` | 4 versions |
| qwen25-14b-local Modelfile | PRESENT | `versions/v3_arcadia/00_emergence/qwen25-14b.Modelfile:1-19` | Q4_K_M |
| qwen25-coder-local Modelfile | PRESENT | `versions/v3_arcadia/00_emergence/qwen25-coder-14b.Modelfile:1-19` | JSON-mode |
| mistral-nemo-local Modelfile | PRESENT | `versions/v3_arcadia/00_emergence/mistral-nemo.Modelfile:1-18` | num_ctx 32768 |
| deepseek-r1-local-q4 Modelfile | PRESENT | `docs/OLLAMA_FINE_TUNING_FINAL_UPGRADE.md` | Q4_K_M reference |
| 5 Modelfile files (rl/lora/*) | PRESENT | `rl/lora/Modelfile, .v2, .v3, .v4` + `versions/v4_arcadia_live/features/Modelfile.analyst_v5` | All 5 present |

## A.2 Modelfile Crafting

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| TSMC 54% foundry / 92% advanced | PRESENT | `rl/lora/Modelfile:8`, `Modelfile.v2:7-8`, `Modelfile.v3:7`, `Modelfile.v4:8`, `analyst_v5:9` | hardcoded |
| Tohoku/Suez/Chip/Ukraine/Red Sea facts | PRESENT | `Modelfile.v3:8-15`, `analyst_v5:9-20` | all 5 crises |
| In-prompt training examples | PRESENT | `Modelfile:27-110` (5 ex), `Modelfile.v2:44-96`, `Modelfile.v3:47-178` | MESSAGE blocks |
| temperature/top_p/num_ctx | PRESENT | `Modelfile:113-114`, `analyst_v2:98-103`, `analyst_v5:107-108` | 0.1-0.3 range |
| JSON format mode | PRESENT | `Modelfile.v4:16-28`, `analyst_v5:38-50` | strict JSON |

## A.3 LoRA Fine-Tuning

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| `rl/lora/finetune.py` (Qwen-2.5-1.5B PEFT/LoRA) | PRESENT | `rl/lora/finetune.py:1-19` | NF4 4-bit |
| bitsandbytes 4-bit NF4 | PRESENT | `rl/lora/finetune.py:9` | "bitsandbytes 4-bit NF4" |
| TRL training | PRESENT | `rl/lora/finetune.py:12-13` | Windows-compatible |
| `lora_training_data.json` (225 pairs) | PRESENT | `rl/data/lora_training_data.json:1` | 225 verified |
| `create_ollama_model.py` | PRESENT | `rl/lora/create_ollama_model.py:1-50+` | LoRA → Modelfile |
| `checkpoints/lora/` | PRESENT | `rl/checkpoints/lora/` | runtime artifacts |

## A.4 DPO Fine-Tuning (Phoenix v5)

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| `dpo_judge/*` directory | PRESENT | `versions/v5_phoenix/roll_integration/dpo_judge/` | 6 files |
| `prepare_preference_data.py` | PRESENT | `dpo_judge/prepare_preference_data.py:1-50+` | DPO pair builder |
| `train_dpo_trl.py` | PRESENT | `dpo_judge/train_dpo_trl.py:1-50+` | TRL trainer |
| `train_dpo_roll.py` | PRESENT | `dpo_judge/train_dpo_roll.py:1-30+` | ROLL-integrated |
| `train_grpo_env.py` | PRESENT | `dpo_judge/train_grpo_env.py:1-50+` | GRPO multi-turn |
| `train_grpo_live_env.py` | PRESENT | `dpo_judge/train_grpo_live_env.py:1-50+` | live-env GRPO |
| `evaluate_delta.py` | PRESENT | `dpo_judge/evaluate_delta.py:1-50+` | base vs DPO delta |
| 21 preference pairs | PRESENT | `dpo_judge/data/preference_pairs.jsonl:1` | 21 lines |
| `dpo_qwen25_3b_supplymind.yaml` | PRESENT | `roll_integration/configs/dpo_qwen25_3b_supplymind.yaml:1` | DPO config |
| `agentic_supplymind_gigpo.yaml` | PRESENT | `roll_integration/configs/agentic_supplymind_gigpo.yaml:1` | GiGPO config |

## A.5 ROLL Integration

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| `SupplyMindRollEnv` | PRESENT | `roll_integration/env/supplymind_roll_env.py:32-89` | reset/step/grade |
| `SupplyMind3JudgeRewardWorker` | PRESENT | `roll_integration/reward_bridge/supplymind_judge_worker.py:44-100` | majority-vote |
| forecast tool | PRESENT | `agentic_supplymind_gigpo.yaml:9-11` | endpoint defined |
| rag tool | PRESENT | `agentic_supplymind_gigpo.yaml:12-14` | endpoint defined |
| rl_act tool | PRESENT | `agentic_supplymind_gigpo.yaml:15` | RL-act endpoint |

## A.6 Quantization & Memory

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| Q4_K_M references | PRESENT | `mistral-nemo.Modelfile:1`, `qwen25-14b.Modelfile:1`, `qwen25-coder-14b.Modelfile:1` | all 3 specify q4km |
| `OLLAMA_MAX_LOADED_MODELS=1` | PRESENT | `docs/OLLAMA_FINE_TUNING_FINAL_UPGRADE.md` | VRAM discipline |
| `convert_bge_to_safetensors.py` | PRESENT | `versions/v3_arcadia/00_emergence/convert_bge_to_safetensors.py:1-45` | CVE-2025-32434 workaround |
| 2GB safetensors output | PRESENT | `models/bge-m3/model.safetensors` | 2.2GB verified |

## B. 13 Foundation Models

| Model | Status | Path(s) | War-Room Wired? |
|---|---|---|---|
| DeepSeek-R1-Q4 | PRESENT | `models/deepseek-r1-distill-qwen-7b/` | YES (3-judge panel) |
| Qwen-2.5-14B | PRESENT | `models/qwen25-14b/` | YES (analyst v5 base, 3-judge panel) |
| Qwen-2.5-Coder-14B | PRESENT | `models/qwen25-coder-14b/` | partial (verified, not wired in war-room) |
| Mistral-Nemo-2407 | PRESENT | `models/mistral-nemo/` | YES (3-judge panel, 128K ctx) |
| Chronos-Bolt-base | **NEW** | `models/chronos-bolt-base/` | **YES (forecast_v2 ensemble)** |
| TimesFM-2 | **NEW** | `models/timesfm-2/` | **YES (forecast_v2 ensemble)** |
| TabPFN-v2-clf | PRESENT | `models/tabpfn-v2-clf/` | partial (verified) |
| TabPFN-v2-reg | **NEW** | `models/tabpfn-v2-reg/` | **YES (forecast_v2 ensemble, severity-conditioned delta)** |
| BGE-M3 | PRESENT | `models/bge-m3/` | YES (RAG fallback, safetensors-converted) |
| mxbai-embed-large | PRESENT | `models/mxbai-embed-large/` | YES (crisis library v2 primary, P@1=0.962) |
| BGE-reranker-v2-m3 | PARTIAL | `models/bge-reranker-v2-m3/` | not wired into war-room (next pass) |
| Snowflake-Arctic-Embed-L | PARTIAL | `models/snowflake-arctic-embed-l/` | not wired (next pass) |
| Qwen-2.5-VL-7B | PARTIAL | `models/qwen25-vl-7b/` | verify script + downstream demo only |

### Verification Scripts (one per model)

| Script | Status | Path |
|---|---|---|
| `verify_qwen14b.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_qwen14b.py` |
| `verify_mistral_nemo.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_mistral_nemo.py` |
| `verify_qwen_coder.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_qwen_coder.py` |
| `verify_qwen_vl.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_qwen_vl.py` |
| `verify_tabpfn.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_tabpfn.py` |
| `verify_timesfm.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_timesfm.py` |
| `verify_embedders_chronos.py` | PRESENT | `versions/v3_arcadia/00_emergence/verify_embedders_chronos.py` |
| `r1_qwen_vl_downstream.py` | PRESENT | `versions/v3_arcadia/00_emergence/r1_qwen_vl_downstream.py` |

## C.1 Game-Engine Tasks & Action Space

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| `easy_typhoon_response` | PRESENT | `server/tasks/task_easy.py:19-48` | 1 disruption, 12 nodes |
| `medium_multi_front` | PRESENT | `server/tasks/task_medium.py:19-50` | 3 concurrent disruptions |
| `hard_cascading_crisis` | PRESENT | `server/tasks/task_hard.py:19-50+` | 8-event cascade, 40 nodes |
| 7 actions × 40 nodes = 280 | PRESENT | `models.py:106-112`, `rl/gym_env.py` | MultiDiscrete |
| Reward range [-1,1] dense | PRESENT | `server/engine/rewards.py` | 7-component blend |
| Episodes 30/45/60 days | PRESENT | task files | tier-specific |
| Budgets ₹40/65/83 cr ($5/$8/$10M) | PRESENT | task files | tier-specific |
| 12/25/40-node graphs | PRESENT | `server/data/graphs/` | easy/medium/hard JSON |

## C.2 Disruption Modeling

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| Sigmoid warning + bell active + exp decay | PRESENT | `server/engine/disruption.py` | curve composition |
| BFS propagation through graph | PRESENT | `server/engine/graph.py` | BFS w/ severity decay |
| `SEVERITY_DECAY_PER_HOP = 0.20` | PRESENT | `server/engine/graph.py:30` | exact constant |
| `_ever_offline` / `_customer_delays` / `_active_hedges` / `_alerted_suppliers` / `_rerouted_edges` | PRESENT | `server/supply_environment.py` | episode flags |

## C.3 Reward Components (7 weighted)

| Bullet | Weight | Path(s) |
|---|---|---|
| Revenue preservation | 35% | `server/engine/rewards.py:4-12, 39` |
| Stockout prevention | 25% | `server/engine/rewards.py:96` |
| Proactive bonus | 15% | `server/engine/rewards.py:148` |
| Cost penalty | 10% | `server/engine/rewards.py:150` |
| Health maintenance | 5% | `server/engine/rewards.py` |
| SLA compliance | 5% | `server/engine/rewards.py` |
| Unnecessary action penalty | 5% | `server/engine/rewards.py` |
| Time-discounted `max(0.3, 1.0 - step×0.7)` | PRESENT | `server/engine/rewards.py` | step-rate decay |
| One-per-episode bonus spam guard | PRESENT | `server/engine/rewards.py` | flags |

## C.4 Cost Engine (industry-cited)

| Constant | Value | Source | Path |
|---|---|---|---|
| `BACKUP_QUALIFICATION_COST` | $150K | ISM survey | `server/engine/financial.py:29` |
| `BACKUP_PREMIUM_RATE` | 12% | industry avg / McKinsey | `financial.py:30` |
| `REROUTE_COST_PER_PORT` | $35K | industry avg | `financial.py:31` |
| `CARRYING_COST_RATE` | 25%/yr | CSCMP | `financial.py:32` |
| `HEDGE_PREMIUM_RATE` | 6% | commodity options | `financial.py:33` |
| `SLA_PENALTY_PER_DAY` | $25K | industry SLA | `financial.py:34` |
| `EXPEDITE_MULTIPLIERS["air"]` | 10× | IATA | `financial.py:47` |
| `EXPEDITE_MULTIPLIERS["rail"]` | 2.5× | industry avg | `financial.py:48` |
| `EXPEDITE_MULTIPLIERS["express_sea"]` | 2× | industry avg | `financial.py:49` |
| `alert action` | $0 | free intel | (zero-cost action) |

## C.5 Episode Lifecycle

| Bullet | Status | Path(s) |
|---|---|---|
| `reset(task_id, seed)` deterministic | PRESENT | `server/supply_environment.py` |
| `step(action) -> obs` | PRESENT | `server/supply_environment.py` |
| `grade()` 0-1 + breakdown | PRESENT | `server/engine/grader.py` |
| Graceful no-op after `done=True` | PRESENT | `supply_environment.py` |
| Jitter within determinism | PRESENT | seeded `np.random.default_rng` |

## C.6 Crisis Library (5 historical disruptions)

| Disruption | Status | Path |
|---|---|---|
| Tohoku 2011 ($235B, 180-d) | PRESENT | `benchmark/crisis_library/tohoku_2011.json` |
| Suez 2021 (Ever Given, 6d, $9.6B/d) | PRESENT | `benchmark/crisis_library/suez_2021.json` |
| Chip shortage 2020 ($210B, sev 0.85) | PRESENT | `benchmark/crisis_library/chip_shortage_2020.json` |
| Ukraine neon 2022 (45-65% global) | PRESENT | `benchmark/crisis_library/ukraine_neon_2022.json` |
| Red Sea 2023 (Houthi, +10d, +25% fuel) | PRESENT | `benchmark/crisis_library/red_sea_2023.json` |

## C.7 Server Infrastructure

| Bullet | Status | Path(s) | Note |
|---|---|---|---|
| FastAPI app on port 8000 | PRESENT | `server/app.py:75-86` | `app = FastAPI(...)` |
| Lifespan pre-warming (3 tasks) | PRESENT | `server/app.py:49-68` | `@asynccontextmanager` |
| CORS `allow_origins=["*"]` | PRESENT | `server/app.py:90-96` | `CORSMiddleware` |
| 20-session pool LRU | PRESENT | `server/app.py:195` | `_MAX_SESSIONS = 20` |
| `asyncio.Lock` concurrency | PRESENT | `server/app.py` | session lock |
| Async/sync hybrid endpoints | PRESENT | `server/app.py` | mixed `async def` + `def` |

---

## Pass-9/10 additions (not in original bullet list, but relevant)

| Component | Path | Purpose |
|---|---|---|
| Hormuz War Room orchestrator | `versions/v4_arcadia_live/realtime/hormuz_war_room_router.py` | `/demo/hormuz-war-room` POST + UI route |
| India 7-sector exposure | `versions/v4_arcadia_live/scenarios/india_industry_exposure.py` | 7 cited sectors + deterministic scorer |
| Gulf 7-sector exposure | `versions/v4_arcadia_live/scenarios/gulf_industry_exposure.py` | 7 cited sectors + bypass-credit scorer |
| Hormuz chokepoint graph | `versions/v4_arcadia_live/scenarios/hormuz_chokepoint_graph.py` | 14 nodes + 18 edges + 5 IEA facts |
| OpenRouter 6-judge cross-check | `versions/v4_arcadia_live/realtime/openrouter_war_room_panel.py` | gpt-oss-120b, gemma, glm, minimax, nemotron, gemma-26b |
| War-Room dashboard HTML | `server/static/hormuz_war_room.html` | dark-mode 6-panel UI |
| War-Room validation harness | `scripts/validate_war_room.py` | 8-event historical backtest |
| Ensemble Brent forecaster | `versions/v5_phoenix/forecast_v2/ensemble_brent.py` | Chronos+TimesFM+TabPFN, 8/8 ±30% |
| Ensemble Brent validator | `scripts/validate_ensemble_brent.py` | 8-event closed-form backtest |
| Master demo HTML | `server/static/master.html` | 9-card live integration page |
| RAP-XC weights | `versions/v5_phoenix/experiments/rap_xc_v1/rapxc.pt` | 3.14M params, BC 5.62→0.23 |
| Conformal weights | `versions/v5_phoenix/action_v2/conformal_calibrated.pt` | α=0.1, coverage 0.9001 |
| HetGAT report | `versions/v5_phoenix/experiments/hetgat_v1/report.json` | +7.77/+12.15/+10.03% |

## API Keys (every key reaches a UI element)

| Key | Used in |
|---|---|
| `OPENROUTER_API_KEY` | War-Room 6-judge panel · cross-corpus α · panel-consensus endpoint |
| `EIA_API_KEY` | Live Brent in 20-source fan-out · Ensemble Brent forecaster history |
| `NASA_FIRMS_MAP_KEY` | Wildfire signal in `live/intel-fan-out` v2 sources |
| `GFW_API_TOKEN` | Global Fishing Watch tanker AIS in fan-out |
