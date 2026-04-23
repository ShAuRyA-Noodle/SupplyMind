# CLONE_AND_STUDY — full tour of SupplyMind for the reader on a fresh machine

You cloned `Sleep-Token` on your Mac and want to understand every piece of what got built across 5 passes. This file is the reading order + runnable tour. Budget: **30 min to orient, 2 hours to fully absorb.**

---

## 0. Clone + install (5 min)

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token

# Python 3.11 recommended
python3.11 -m venv .venv
source .venv/bin/activate

# Minimal install — enough to boot the server + run tests
pip install -r requirements.txt

# Optional: if you want the RL trainer path locally
# pip install -r requirements-rl.txt
# pip install -r requirements-damocles.txt
```

If you want to hit the paid OpenRouter models too:
```bash
cp .env.example .env
# Edit .env: put your own OPENROUTER_API_KEY there (mine is local-only in chat).
```

**You do NOT need any API key to read the project.** All committed evidence runs entirely offline from committed JSON.

---

## 1. The 3-file orientation (10 min)

Read these three files in order — they give you the thesis in 10 minutes.

1. **[README.md](../README.md)** — headline claims + 10 numbers + hackathon evidence table. Lines 33-80 are the finals submission; skim the rest.
2. **[JUDGES.md](../JUDGES.md)** — the 4-minute version. Every receipt linked.
3. **[docs/FINAL_AUDIT_REPORT.md](FINAL_AUDIT_REPORT.md)** — the unified 60-row limitation ledger across 3 audit sources. For every claim, where the evidence lives.

---

## 2. The architecture you actually shipped (15 min)

```
                            ┌──────────────────────────────────┐
                            │     server/app.py (FastAPI)       │
                            │   OpenEnv /reset /step /state     │
                            └──────┬──────────────────┬─────────┘
                                   │                  │
      ┌────────────────────────────┼──────────────────┼──────────────────────┐
      │                            │                  │                      │
      ▼                            ▼                  ▼                      ▼
 /analyst/*              /agent/decide         /v3/e2e                /live/hormuz-closure
 (training oracle)   (IntegratedAgent)     (5-stage chain)       (realtime geopolitical)
  ├─ grade                  ├─ RAG              ├─ RAG                    ├─ NewsAPI
  ├─ scenarios              ├─ Panel vote       ├─ Rubric                 ├─ GDELT
  ├─ next-scenario          ├─ GCN cascade      ├─ Forecast               ├─ USGS
  ├─ holdout-eval           ├─ RL policy        └─ RL                     ├─ FRED
  └─ panel-consensus        └─ Forecast                                    └─ MarineTraffic
                               (+ /stream)
```

Study each branch:

### 2a. OpenEnv compliance — 20 min read
- **[server/openenv_adapter.py](../server/openenv_adapter.py)** — `OpenEnvSupplyMind(Environment[ActT,ObsT,StateT])`, subclasses `openenv.core.Environment`, implements reset/step/state/close. Uses `TrajectoryRubric` (composable, not monolithic).
- **[openenv.yaml](../openenv.yaml)** — the official manifest. env_id, action dataclass, observation dataclass, endpoints.
- **[client/supplymind_client.py](../client/supplymind_client.py)** — the *separate* client. Zero `from server` imports. Judges call this against local or HF Space.

### 2b. Env-connected training (this is the critical differentiator)
- **[ShAuRyA_Phoenix/roll_integration/dpo_judge/train_grpo_live_env.py](../ShAuRyA_Phoenix/roll_integration/dpo_judge/train_grpo_live_env.py)** — the GRPO trainer that pulls rewards via HTTP `POST /analyst/grade`. Every reward comes over the wire from the running env server. Not a static dataset.
- **[scripts/run_frontier_judge_panel.py](../scripts/run_frontier_judge_panel.py)** — runs a 12-judge frontier panel against 26 real R4 crisis scenarios. Every call cached per (model, scenario) to `.openrouter_cache/` so re-runs don't re-spend.

### 2c. The IntegratedAgent (closes the "5 museums" critique)
- **[server/integrated_agent.py](../server/integrated_agent.py)** — one class, one pipeline, 5 stages: RAG → panel → GNN → RL → forecast. Every stage tagged `inference_type`. Exposed as `POST /agent/decide`.

### 2d. Reward design + anti-hacking
- **[server/app.py `/analyst/grade`](../server/app.py)** — the verifiable reward oracle. 3 independent components (match + format + length), proximity scoring (ordinal 4-tier ordinal distance), `r_length` returns -0.5 for over-length attacks.
- **[tests/test_reward_hacking_adversarial.py](../tests/test_reward_hacking_adversarial.py)** — 6 attack vectors we designed against the reward. All 6 verified rejected.
- **[tests/receipts/adversarial_reward_audit.json](../tests/receipts/adversarial_reward_audit.json)** — the committed receipt (FAQ §57: "don't optimize a reward you haven't tried to break").

### 2e. RLVE adaptive curriculum
- **[server/app.py `/analyst/next-scenario`](../server/app.py)** — picks scenarios at the policy's zone of proximal development using real R4 3-judge-disagreement as difficulty oracle.
- **[server/app.py `/analyst/holdout-eval`](../server/app.py)** — sealed holdout (last 6 scenarios) never served to the sampler.

---

## 3. Run the whole thing locally (20 min)

### 3a. Boot the env server (no API keys, no GPU)
```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000

# In another terminal:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metadata | jq
curl http://127.0.0.1:8000/tasks | jq '.[] | .task_id'
```

### 3b. Run the full IntegratedAgent (touches RAG + panel + GNN + RL + forecast)
```bash
curl -X POST http://127.0.0.1:8000/agent/decide \
  -H 'Content-Type: application/json' \
  -d '{"query":"Iran announces full closure of Strait of Hormuz","task_id":"easy_typhoon_response","seed":42}' \
  | jq
```
Expect: `risk_level: HIGH`, RAG evidence cites Strait_of_Hormuz chunk, GNN top-3 nodes, RL action, forecast $126 ± $12. 267ms.

### 3c. Run the v3 end-to-end
```bash
curl -X POST http://127.0.0.1:8000/v3/e2e \
  -H 'Content-Type: application/json' \
  -d '{"query":"Typhoon Koinu bearing NNW toward Taiwan","task_id":"easy_typhoon_response","seed":42}' \
  | jq '.pipeline_stages'
```
Every stage has `inference_type` — look for `live_retrieval`, `live_rubric`, `live_compute_from_cached_conformal`, `live_onnx_inference`. No `mocked` or `synthetic` anywhere.

### 3d. Panel consensus on a single scenario
```bash
curl 'http://127.0.0.1:8000/analyst/panel-consensus/2011_T%C5%8Dhoku_earthquake_and_tsunami' | jq
```
SSE streaming version:
```bash
curl -N 'http://127.0.0.1:8000/analyst/panel-consensus/2011_T%C5%8Dhoku_earthquake_and_tsunami/stream'
```

### 3e. RLVE adaptive sampler
```bash
# Weak policy (ability=0.1) → easy scenario
curl -X POST http://127.0.0.1:8000/analyst/next-scenario \
  -H 'Content-Type: application/json' \
  -d '{"recent_reward_mean":0.1,"headroom":0.15}' | jq

# Strong policy (ability=0.8) → hardest scenario in training set
curl -X POST http://127.0.0.1:8000/analyst/next-scenario \
  -H 'Content-Type: application/json' \
  -d '{"recent_reward_mean":0.8,"headroom":0.15}' | jq
```

### 3f. The full test suite
```bash
pytest tests/ ShAuRyA_Supplymind/tests/ ShAuRyA_Phoenix/tests/ -q
# Should show: 272 passing, 2 skipped in ~3 minutes
```

---

## 4. The evidence files — every claim, diff-able (15 min)

Sorted by judge-visibility:

| File | What it proves |
|---|---|
| `v3_arcadia/results/R4_DANGEROUS_V2.json` | 3-judge LLM panel on 26 real 2024-2026 crisis scenarios. Per-judge verdicts, Krippendorff α ordinal, cohen κ. **The source of truth for ground-truth labels.** |
| `v3_arcadia/results/R4_FRONTIER_PANEL_V2.json` | 12-model frontier judge panel (pass 5g). Cross-provider cross-lab ordinal agreement. |
| `v3_arcadia/results/R6_EUCLIDIAN.json` | MaskablePPO bootstrap CI95 vs random / greedy on 3 supply-chain tasks. Non-overlapping intervals. |
| `v3_arcadia/results/R6_AQUA_REGIA_V2.json` | Per-horizon conformal prediction coverage stats on 5 FRED targets. |
| `v3_arcadia/plots/gethsemane/learning_curves.png` | The RL training curve. |
| `ShAuRyA_Supplymind/features/R9_ANALYST_AB_V5.json` | supplymind-analyst:v5 vs base Qwen2.5-14B on 10 rubric-labeled scenarios. **80% exact vs 0%.** |
| `tests/receipts/adversarial_reward_audit.json` | 6-attack reward-hacking audit, all rejected. |
| `tests/receipts/frontier_panel_alpha.json` | Real Krippendorff α recomputed on the committed panel cache. |
| `tests/receipts/openrouter_liveness.json` | Per-model cold-probe liveness timestamped 2026-04-24. |
| `ShAuRyA_Supplymind/autoresearch/AUTORESEARCH_LAB_NOTEBOOK.md` | Karpathy-style autoresearch history: 5 seed experiments, 3 accepted, +0.148 CI95 lift over baseline. |
| `ShAuRyA_Supplymind/scenarios/iran_israel_hormuz_2024_2026.json` | 8 real 2024-2026 events, 26 citations. The crisis library. |

---

## 5. The Colab notebook (no install needed on Mac)

Click the badge in README §1 or go directly to:

https://colab.research.google.com/github/ShAuRyA-Noodle/Sleep-Token/blob/main/notebooks/06_trl_training_colab.ipynb

That gives you TRL + Unsloth DPO training on 21 real preference pairs in ~10 min on free T4. Plots loss + chosen/rejected reward margins. Full judge before/after comparison.

---

## 6. Read the pass history — how we got here

Every pass committed, diff-able via `git log`:

| Commit | Pass | What shipped |
|---|---|---|
| `b19a169` | pre-5 baseline | v4 snapshot with some synthetic contamination still live |
| `44ff75b` | 1 | Colab TRL notebook + GRPO trainer (FAQ Gate 2) |
| `0b31f97` | 2 | /analyst/grade + env-connected GRPO + killed fake data in /v3/e2e |
| `a28dd4c` | 3 | Unsloth integration + multi-reward TRL |
| `9474505` | 4 | RLVE adaptive sampler + holdout eval + adversarial audit |
| `369b121` | 5a | OpenRouter client + Tier 1 truth-gap fixes |
| `1567c53` | 5b | /analyst/panel-consensus + real α receipt |
| `7ac79c7` | 5c | IntegratedAgent + /agent/decide |
| `b755e5a` | 5d | FINAL_AUDIT_REPORT.md (60-row ledger) |
| `bca3c34` | 5e | OpenAI dropped (stack is local + OpenRouter only) |
| `9c49e3c` | 5f | Paid-route unlocks for 5 previously-blocked judges |
| `<next>` | 5g | Full 12-frontier-judge panel + real α on 15 judges |

Read any commit diff: `git show <hash>`. The commit messages are substantive.

---

## 7. The 31-model stack (map)

Everything you could possibly exercise:

**Local (Ollama — `ollama list` to see):**
- 3 judges: deepseek-r1-local-q4, qwen2.5:14b, mistral-nemo-local
- 1 critic: qwen25-coder-local
- 1 vision: qwen2.5vl:7b
- 1 analyst: supplymind-analyst:v5 (our fine-tune)
- Others: nomic-embed-text, qwen2.5:7b-instruct, aya:8b, gemma4:e4b-it-bf16

**Local (Python — foundation models downloaded via sentence-transformers / HF):**
- 3 embedders: mxbai-embed-large (P@1=0.962), BGE-M3, Snowflake-arctic-embed
- 1 reranker: BGE-reranker
- 3 forecasters: Chronos-Bolt, TimesFM-2, ARIMA+Prophet
- 1 tabular: TabPFN-v2
- 1 GNN: custom 3-layer PyTorch

**OpenRouter (18 in `scripts/openrouter_client.py`):**
- 12 judges (see panel run)
- 1 red-team: qwen3-coder-flash
- 3 vision: nemotron-12b-vl, gemma-3-12b, gemma-3-4b
- 2 utility: gpt-oss-20b, llama-3.2-3b

**All probed and verified.** Liveness receipt at `tests/receipts/openrouter_liveness.json`.

---

## 8. Things you can change / experiment with

These edit points are well-factored:

| Want to... | Edit this |
|---|---|
| Add a new judge model | `scripts/openrouter_client.py:MODELS` (add ModelSpec) |
| Change reward weights | `server/app.py` `/analyst/grade` — `0.7, 0.2, 0.1` constants |
| Add a new adversarial attack | `tests/test_reward_hacking_adversarial.py:ATTACKS` |
| Wire a new local model into IntegratedAgent | `server/integrated_agent.py` — add a `_stage_*` method |
| Change holdout scenario count | `server/app.py` `_HOLDOUT_TAIL_N` constant |
| Change RLVE difficulty headroom | `/analyst/next-scenario` `headroom` param (default 0.15) |

---

## 9. If something doesn't work

```bash
# Test everything
pytest tests/ ShAuRyA_Supplymind/tests/ ShAuRyA_Phoenix/tests/ -q

# Test just the adversarial reward audit
pytest tests/test_reward_hacking_adversarial.py -v

# Test OpenRouter liveness
python scripts/verify_openrouter_models.py

# Check OpenEnv compliance endpoints
curl http://127.0.0.1:8000/schema | jq
curl http://127.0.0.1:8000/metadata | jq
```

If Ollama models aren't pulled locally on your Mac yet:
```bash
ollama pull qwen2.5:14b
ollama pull deepseek-r1
ollama pull mistral-nemo
```

---

## 10. The one-paragraph summary of the whole project

**SupplyMind is an OpenEnv-compliant RL environment for supply-chain risk management where an LLM agent interacts with live geopolitical data (NewsAPI + GDELT + USGS + FRED + MarineTraffic) to assess crisis severity on real 2024-2026 events. The reward is verifiable (ordinal 4-tier proximity match against a 15-judge panel: 3 local Ollama + 12 frontier OpenRouter, cross-provider Krippendorff α). The training loop connects to the env via HTTP — every GRPO reward is an `POST /analyst/grade` call on the running server, never a static dataset. Adaptive RLVE curriculum pulls scenarios at the policy's zone of proximal development. Six adversarial reward-hacking vectors have been tested and all rejected, receipt committed. The IntegratedAgent single class wires RAG + panel + GNN + RL + conformal into one pipeline, exposed as `/agent/decide` — 267ms end-to-end on a live Hormuz query.**

That's the beast. Enjoy the tour.

---

*Questions? The project's commit history is your co-author — `git log --all --oneline | head -20` is a map.*
