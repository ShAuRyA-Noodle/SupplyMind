# SupplyMind Grand Finale — Definitive Implementation Plan

## Context

Top 3,000 from 52,000+ applicants for Meta PyTorch OpenEnv Hackathon Grand Finale (Bangalore, on-campus). Current project: solid supply chain risk management environment (6,285 lines, 154 tests, 7 actions, Monte Carlo, LLM+scripted agents). **Critical gap: zero PyTorch/ML components in a PyTorch hackathon.**

**Hardware:** Mac (development) + Alienware M16 R1 RTX 4080 16GB VRAM (training + demo machine). Local Ollama with qwen2.5:14b + aya:8b installed.

**Goal:** Build genuinely novel ML features that make Meta FAIR engineers say "wait, how did they build that?" — not table-stakes PPO+dashboard that 200 other teams will submit.

---

## Architecture Overview

```
Layer 0: Real-world data (FRED commodity prices, Baltic Dry Index CSV)
    ↓
SupplyMind Core Environment (6,285 lines, untouched)
    ↓
4 Novel ML Components:
  ├─ Decision Transformer (offline RL as sequence prediction)
  ├─ QR-DQN (distributional RL, CVaR-optimal tail-risk policy)
  ├─ Neural Surrogate World Model (100x faster Monte Carlo)
  └─ LLM-RL Hybrid Explainability (Ollama local, zero API limits)
    ↓
Streamlit Dashboard (live crisis simulation, counterfactual panel)
```

**Zero existing files modified for functionality.** All new code in `rl/`, `dashboard/`, `benchmark/`. All 154 tests guaranteed to pass.

---

## PHASE 1 — Foundation (Day 1)

### 1.1 Gymnasium Wrapper
**New file:** `rl/gym_env.py`

State encoding (fixed-size float tensor):
- Per-node (N × 10): is_operational, risk_score, inventory_days/90, has_backup, node_type_onehot(5), revenue_normalized
- Global (8): day/max_steps, budget_remaining/total, health/100, num_disruptions_norm, max_severity, cum_loss_norm, mc_p50_norm, mc_p95_norm
- Pad all tasks to 408 floats (hard task size)

Action space: `MultiDiscrete([7, 40])` with action masking
- Extra params auto-filled: backup→first available, reroute→first operational port, stock→10 days, hedge→most-spiked commodity at 5% budget, expedite→air

**Integration:** imports `SupplyMindEnvironment` directly from `server/supply_environment.py` (line 21), reads `obs.node_statuses`, `obs.financials`, `obs.active_signals`

### 1.2 Offline Dataset Generation
**New file:** `rl/offline/dataset.py`

Run 5,000 episodes with scripted agent + 5,000 with random agent across all 3 tasks. Collect (state, action, reward, next_state, done, returns_to_go) tuples → 300K+ transitions.

Inject real commodity price data from FRED API (copper PCOPPUSDM, oil DCOILWTICO, cached to `rl/data/fred_cache.json`).

Run overnight on Alienware GPU: env runs at ~1000 steps/sec → ~5 hours for 500K transitions.

### 1.3 PPO Baseline (sanity check)
**New file:** `rl/train_ppo.py`

MaskablePPO from sb3-contrib with 32 parallel SubprocVecEnv on GPU. 2M steps in ~8 min on RTX 4080. This is the sanity check — if PPO converges, the environment wrapper works correctly.

**Critical files to read/import:**
- `server/supply_environment.py:21` — SupplyMindEnvironment class
- `models.py:90-174` — SupplyMindAction (7 types, validation)
- `models.py:177-231` — SupplyMindObservation (state encoding source)
- `server/engine/rewards.py:37` — RewardCalculator (dense reward, already [-1,1])

---

## PHASE 2 — Novel ML (Days 2-3)

### 2.1 Decision Transformer (P0 — The Meta-Relevant One)
**New file:** `rl/decision_transformer/model.py`

Uses GPT-2 backbone from HuggingFace `transformers`. Reframes RL as sequence prediction: feed (return-to-go, state, action) tuples, predict next action autoregressively.

```
Architecture:
  embed_return(1→128) + embed_state(408→128) + embed_action(280→128) + embed_timestep(60→128)
  → interleave (r1,s1,a1,r2,s2,a2,...) → GPT2(3 layers, 1 head, 128 hidden)
  → predict_action head from state token positions
```

**Why this wins:** DT lets you query different risk appetites at inference time via return-to-go conditioning. Slider in dashboard: "Desired outcome: 0.5→0.9" → agent behavior visibly changes. No retraining needed. Meta engineers will immediately recognize the LLM↔RL connection.

Training: Cross-entropy loss on action predictions. 10 epochs on 150K transitions on RTX 4080 = ~25 min.

**New file:** `rl/decision_transformer/train.py`

### 2.2 QR-DQN — Distributional RL (P0)
**New file:** `rl/distributional/qr_dqn.py`

Quantile Regression DQN with 51 quantiles. ~150 lines of PyTorch.

```python
class QRDQNNetwork(nn.Module):
    # state(408) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(n_actions × 51)
    # Reshape to (batch, n_actions, 51) quantile values
    
    def cvar_policy(self, x, alpha=0.1):
        # Pick action minimizing CVaR at alpha (worst 10% of outcomes)
        k = int(alpha * 51)
        cvar = quantile_values[:, :, :k].mean(dim=-1)
        return cvar.argmax(dim=-1)
```

**Why this wins:** "Our policy minimizes conditional value-at-risk, not expected cost. Companies care about P5 worst-case, not averages." Dashboard shows full return distribution as violin plot per step.

Training: Quantile regression loss, 200K steps, ~30 min on RTX 4080.

**New file:** `rl/distributional/train.py`

### 2.3 Neural Surrogate World Model (P1)
**New file:** `rl/surrogate/world_model.py`

MLP that learns (state, action) → (next_state, reward, done). Train on 500K transitions from dataset.

```
Architecture: Linear(408+280, 512) → ReLU → Linear(512, 256) → ReLU
  → state_head: Linear(256, 408)
  → reward_head: Linear(256, 1)
  → done_head: Linear(256, 1) + Sigmoid
```

**Two killer uses:**
1. **GPU Monte Carlo:** 100K scenarios in <80ms on RTX 4080 (vs seconds in Python engine)
2. **Counterfactual engine:** After each action, replay with do_nothing from that point. Show: "Without this backup activation, P50 additional loss: $4.2M"

**New file:** `rl/surrogate/counterfactual.py`, `rl/surrogate/gpu_monte_carlo.py`

Training: MSE loss on state/reward, BCE on done. 500K transitions, ~40 min CPU / ~4 min GPU.

### 2.4 MC Dropout Uncertainty (P1 — 30 lines, absurd ROI)
**New file:** `rl/uncertainty.py`

Keep model.train() during inference, run 50 forward passes with dropout. Variance = epistemic uncertainty.

Output: "activate_backup(TSMC): 87% confidence, ±$340K"

---

## PHASE 3 — Explainability + Dashboard (Days 3-4)

### 3.1 LLM-RL Hybrid Explainability
**New file:** `rl/explainer.py`

Uses LOCAL Ollama (qwen2.5:14b) — zero API limits, zero internet needed, ~3-4 sec per explanation on RTX 4080.

After each RL action, decode state to text + call Ollama:
> "The RL agent observed TSMC (risk: 0.87, trending up) entering warning phase with 6 days inventory. It activated backup because P95 loss ($12.3M) exceeds backup cost ($0.8M) by 15×."

Pre-populate 50 common scenarios to `cache/explanations.json` for instant demo.

**Hindi/regional toggle:** Use aya:8b for Hindi explanations. "Supply chain risk management, explained in Indian languages." Scaler is Indian. This lands differently.

### 3.2 Streamlit Dashboard
**New file:** `dashboard/app.py` (~500 lines)

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│ SUPPLYMIND  [Task▼] [Agent▼] [Risk Appetite ——●——] ▶ │
├─────────────┬────────────────────────────────────────┤
│ Supply Chain│ Return Distribution (QR-DQN violin)    │
│ Graph       │ P5/P50/P95 markers, live per step      │
│ (plotly     ├────────────────────────────────────────┤
│  network,   │ Counterfactual Panel                   │
│  NOT pyvis) │ "Without this action: +$4.2M loss"     │
│ Color by    │ Surrogate model output                 │
│ risk score  ├────────────────────────────────────────┤
│ Edge width =│ Agent Reasoning Log                    │
│ GNN attn    │ LLM narrates each RL decision          │
├─────────────┤ Causal chain visible                   │
│ Disruption  │ Side-by-side agent performance         │
│ Timeline    ├────────────────────────────────────────┤
│ (Gantt)     │ Agent Comparison (bar + radar chart)    │
│             │ DT vs QR-DQN vs Scripted vs LLM        │
└─────────────┴────────────────────────────────────────┘
```

**Key: Use `plotly.graph_objects` for network graph, NOT pyvis** (pyvis breaks in Streamlit iframe). Nodes as scatter, edges as lines, color by risk, thickness by GNN attention weights.

**Crisis Library Dropdown:** 5 famous crises (2011 Tohoku, 2021 Suez, 2020-22 Chip Shortage, 2022 Ukraine Neon, 2023 Red Sea). Each is a JSON file in `benchmark/crisis_library/` mapping to disruption parameters.

### 3.3 SHAP on RL Policy
**New file:** `rl/interpretability/shap_analysis.py`

SHAP values on the MLP policy network. Feature importance per action decision. Dashboard shows: "Top 3 factors: TSMC risk (0.87), inventory days (6), budget ratio (0.52)."

~50 lines using `shap.DeepExplainer` on PyTorch model.

---

## PHASE 4 — Production Polish (Days 4-5)

### 4.1 FastAPI Inference Endpoint
**Modified file:** `server/app.py` (additive — one new endpoint)

```python
@app.post("/predict")
async def predict(state: dict) -> dict:
    # Encode state → tensor → RL policy → action + confidence + explanation + counterfactual
```

"Any Fortune 500 ERP system can call this endpoint."

### 4.2 Benchmarking Suite
**New file:** `benchmark/run_benchmark.py`

All agents × all tasks × 5 seeds. Confidence intervals. Publication-quality charts.

| Agent | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| Do-Nothing | 0.27±0.00 | 0.25±0.00 | 0.32±0.00 | 0.28±0.00 |
| Scripted | 0.77±0.02 | 0.70±0.03 | 0.67±0.02 | 0.71±0.02 |
| PPO | 0.80±0.03 | 0.72±0.04 | 0.69±0.03 | 0.74±0.03 |
| QR-DQN (CVaR) | 0.79±0.02 | 0.74±0.02 | 0.72±0.02 | 0.75±0.02 |
| Decision Transformer | 0.82±0.03 | 0.75±0.03 | 0.71±0.03 | 0.76±0.03 |

(Target scores — actual will vary)

### 4.3 ONNX Export + Model Card
**New file:** `rl/export_onnx.py`, `MODEL_CARD.md`

Export policy to ONNX for cross-platform deployment. Model card in HuggingFace style with training data, evaluation metrics, intended use, limitations, ethical considerations.

### 4.4 MLflow Experiment Tracking
Wrap training loops with `mlflow.log_params/metrics/model`. Screenshot the MLflow UI for README.

### 4.5 GitHub Actions CI
**New file:** `.github/workflows/ci.yml`

Every push: run 154 tests + RL smoke test. Green badge in README.

### 4.6 Weights & Biases Integration
Real-time training dashboards. Share URL with judges. "Here's our training run — 2M steps, 50 Optuna trials."

---

## LoRA Fine-Tune LLaMA 3 8B (STRETCH — Day 4-5, GPU overnight)

**New directory:** `rl/lora/`

Fine-tune Meta's own model on supply chain decisions. Use Unsloth for 2-5x speedup.

Dataset: 50K instruction-following pairs generated from environment + scripted agent episodes:
- Input: state decoded to text
- Output: action + reasoning (generated once by Ollama during dataset creation)

Training: 4-bit quantized LoRA, r=16, 3 hours on RTX 4080 (~10GB VRAM).

Push to HuggingFace Hub as `Shaurya-Noodle/supplymind-8b`.

**Demo line:** "We fine-tuned Meta's own LLaMA 3 on 50,000 supply chain decisions."

---

## File Structure (All Additive)

```
rl/
  __init__.py
  gym_env.py                     # Gymnasium wrapper + state/action encoding
  uncertainty.py                 # MC Dropout (30 lines)
  export_onnx.py                 # ONNX export
  
  offline/
    dataset.py                   # Offline buffer generation + FRED data
  
  train_ppo.py                   # PPO baseline (sb3-contrib MaskablePPO)
  
  decision_transformer/
    model.py                     # GPT-2 backbone DT
    train.py                     # Training loop
  
  distributional/
    qr_dqn.py                    # QR-DQN network + CVaR policy
    train.py                     # Training loop
  
  surrogate/
    world_model.py               # Neural surrogate MLP
    counterfactual.py            # "What if we hadn't acted?"
    gpu_monte_carlo.py           # 100K scenarios in 80ms
  
  explainer.py                   # Ollama LLM explanation layer
  
  interpretability/
    shap_analysis.py             # SHAP on RL policy
  
  lora/                          # STRETCH: LLaMA fine-tuning
    finetune.py
    generate_dataset.py
  
  checkpoints/                   # Saved weights (gitignored except best)
  data/                          # Cached FRED/BDI data

dashboard/
  app.py                         # Streamlit dashboard

benchmark/
  run_benchmark.py               # Multi-agent benchmark
  visualize.py                   # Chart generation
  crisis_library/                # 5 historical crisis JSONs
  results/                       # Output (gitignored)

# Modified (minimal, additive):
  pyproject.toml                 # Add [rl], [dashboard] optional deps
  README.md                      # Add ML sections, benchmark results
  MODEL_CARD.md                  # HuggingFace style model card
  .github/workflows/ci.yml       # CI pipeline
```

---

## Dependencies

**`requirements-rl.txt`** (separate from HF Space requirements.txt):
```
torch==2.1.2
gymnasium==0.29.1
stable-baselines3==2.2.1
sb3-contrib==2.2.1
d3rlpy==2.3.0
transformers>=4.36.0
streamlit>=1.32.0
plotly>=5.18.0
shap>=0.43.0
mlflow>=2.10.0
wandb>=0.16.0
ollama>=0.1.0
```

**Alienware GPU install:**
```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric
```

---

## Day-by-Day Schedule

| Day | Mac (development) | Alienware (GPU training) |
|-----|-------------------|-------------------------|
| **1** | Gym wrapper, state/action encoding, dataset generation script | Install CUDA deps, verify GPU, start overnight dataset generation (500K transitions) |
| **2** | QR-DQN network + training loop, Decision Transformer model | Train PPO (8 min), Train QR-DQN (30 min), Train DT (25 min) |
| **3** | Neural surrogate + counterfactual engine, Dashboard skeleton | Train surrogate (4 min GPU), GPU Monte Carlo, start LoRA dataset gen |
| **4** | Dashboard polish (all panels), Benchmark suite, SHAP | LoRA fine-tuning overnight (3 hrs), Optuna HPO sweep |
| **5** | README update, Model Card, CI, ONNX export, demo rehearsal | Final benchmark runs, record backup demo video |

---

## Constraints & Risk Mitigations

| Constraint | Mitigation |
|------------|------------|
| PyG install hell | Try GPU install first. If >30 min, drop GNN, go pure MLP. GNN is impressive but not load-bearing. |
| LoRA on Windows | Unsloth doesn't support Windows. Use PEFT+trl instead (slower, 5-6 hrs). Or dual-boot Ubuntu. |
| Venue internet unreliable | OFFLINE_MODE=true in dashboard loads all cached data. Pre-compute everything. Zero live API calls. |
| VRAM ceiling (16GB) | Never run LoRA (10GB) + DreamerV3 (6GB) simultaneously. Train sequentially. |
| Thermal throttling | `nvidia-smi -pl 150` if GPU hits 90°C. ~20% slowdown but prevents crashes. |
| Demo time (3-5 min) | Pre-run dashboard. Record 3-min YouTube backup. Have URL ready. |
| 154 tests must pass | Zero existing files modified. Run `pytest tests/ -q` after every addition. |
| HF Space (no GPU) | Space hosts only the API. Dashboard + RL runs locally on Alienware at venue. |
| Free API rate limits | FRED: 500/day (cache on first fetch). NewsAPI: 100/day (pre-cache 10 scenarios). Groq: use local Ollama instead. |

---

## The Demo Narrative (3 minutes)

**Open (30s):** "Supply chain disruptions cost $4 trillion annually. Companies simulate risks, but their agents optimize for averages. We built SupplyMind — an environment calibrated from real TSMC, McKinsey, and CSCMP data, with four novel ML approaches that solve actual production problems."

**Show Decision Transformer (45s):** "This is a Decision Transformer — it treats RL as sequence prediction, just like a language model. Watch: I drag this slider from 'conservative' to 'aggressive' and the agent's behavior changes in real-time. No retraining. The same model handles different risk appetites." [Drag slider, show decisions change]

**Show QR-DQN (45s):** "Standard RL maximizes expected reward. But supply chain managers care about worst-case. Our QR-DQN policy minimizes conditional value-at-risk. See the full return distribution — not a bar chart, the actual distribution of outcomes. The CVaR agent activates backup 2 days earlier because it protects the tail." [Show violin plot updating live]

**Show Counterfactual (30s):** "Our neural surrogate model learned the simulation dynamics in PyTorch. 100,000 Monte Carlo scenarios in 80 milliseconds. And it enables this: the counterfactual panel shows 'without this action, additional loss: $4.2M.' Every decision is justified." [Point to counterfactual panel]

**Close (30s):** "Every decision is explained in plain language by a locally-running LLM — no cloud, no API limits. The entire system runs on-device. We fine-tuned Meta's LLaMA 3 specifically for supply chain reasoning. This is production-ready: offline training from historical data, budget-constrained, fully explainable, deployable today."

---

## Verification Plan

1. `pytest tests/ -q` — all 154 pass (no existing files touched)
2. `python -m rl.train_ppo --task easy --steps 50000` — PPO converges to positive reward
3. `python -m rl.distributional.train --task easy` — QR-DQN loss decreases
4. `python -m rl.decision_transformer.train` — DT action prediction accuracy >60%
5. `python -m rl.surrogate.world_model --train` — MSE loss <0.01 on held-out set
6. `streamlit run dashboard/app.py` — all panels render without errors
7. `python -m benchmark.run_benchmark` — all agents produce valid scores
8. Demo rehearsal: full 3-minute run-through, timed
