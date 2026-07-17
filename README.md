---
title: SupplyMind
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Supply chain risk management OpenEnv environment
tags:
  - openenv
  - supply-chain
  - risk-management
  - reinforcement-learning
  - ai-agents
---

# SupplyMind

**A live supply-chain-risk decision copilot.** It watches real-world signals (shipping,
commodities, disasters, news), tells a risk officer what is about to break, what to do about it,
and what it costs if they don't — and **every headline number is traceable to a committed receipt
and machine-checked in CI**.

Built on an OpenEnv-compliant, deterministic supply-chain simulation. Product persona and golden
path in [`PRODUCT_THESIS.md`](PRODUCT_THESIS.md); engineering constitution in [`CLAUDE.md`](CLAUDE.md).

[![Tests](https://img.shields.io/badge/tests-184%20passing%20%2F%20184-brightgreen)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-llama/open-env)

---

## Honesty policy (read this first)

This repository went through a brutal internal fabrication audit (2026-07-02). The engine core was
found genuinely strong; the *presentation/receipts layer* contained fabricated numbers. Those were
**removed, not hidden**, and the capabilities behind them were **rebuilt for real and re-measured**.

- **Every headline number below cites a committed receipt** and is machine-checked by
  [`scripts/verify_claims.py`](scripts/verify_claims.py), which **fails CI if any claim lacks
  backing**. Run it: `python scripts/verify_claims.py`.
- Full per-claim status (VERIFIED / CORRECTED / STRUCK / RERUN-PENDING) is in
  [`FINAL_SUBMIT/CLAIMS_LEDGER.md`](FINAL_SUBMIT/CLAIMS_LEDGER.md).
- Some results are **honest negatives** (federated learning does *not* beat centralized; the
  embedder ensemble is *worse* than mxbai alone; the RL policy *ties* a strong greedy baseline). We
  report them as measured. A disappointing real number is worth more than a flattering fake.
- Items blocked on the (currently revoked) OpenRouter key or on the pending full benchmark are
  labeled **RERUN-PENDING**, never faked around.

---

## Receipt-backed headline results

Every row is asserted against the cited receipt by `scripts/verify_claims.py`. Values are copied
from the receipts, not from memory.

| # | Result | Number | Receipt |
|---|---|---|---|
| 1 | **Adversarial robustness** — real attacks executed against the real system | **318 / 318 blocked (100%), 0 breaches**, 8/8 benign controls accepted | [`tests/receipts/adversarial_gauntlet_REAL.json`](tests/receipts/adversarial_gauntlet_REAL.json) |
| 2 | **Split-conformal calibration** on real harvested transitions | 90% nominal → **90.03% coverage** on a disjoint held-out test set (gap +0.0003) | [`tests/receipts/conformal_REAL.json`](tests/receipts/conformal_REAL.json) |
| 3 | **Brent forecast backtest** — real FRED `DCOILBRENTEU` walk-forward over 8 crisis events | ensemble **mean MAPE 7.1%** (median 6.4%) | [`tests/receipts/ensemble_brent_REAL.json`](tests/receipts/ensemble_brent_REAL.json) |
| 4 | **Federated learning (FedAvg)** on 180K real DataCo orders, 5 regions — *honest negative* | FedAvg AUC **0.7218 = centralized 0.7218 (Δ −0.0)** | [`tests/receipts/fedavg_REAL.json`](tests/receipts/fedavg_REAL.json) |
| 5 | **Ghost-model triage** (each model measured on real data) | embedder ensemble **RETIRE**, reranker **RETIRE**, TabPFN judge **KEEP** (AUC 0.738 vs logreg 0.703) | [`tests/receipts/ghost_models_eval_REAL.json`](tests/receipts/ghost_models_eval_REAL.json) |
| 6 | **RAG retrieval** (mxbai bi-encoder, 6,483-chunk real corpus) | **P@1 0.962, MRR 0.978, nDCG@10 0.961** | [`tests/receipts/ghost_models_eval_REAL.json`](tests/receipts/ghost_models_eval_REAL.json) |
| 7 | **RAG World-Bank fix** (list-vs-dict ingestion bug) | **0 → 20 WB chunks, 6,602 total**; WB-query P@1 0 → 1.0 | [`tests/receipts/rag_recook_REAL.json`](tests/receipts/rag_recook_REAL.json) |
| 8 | **4-method causal counterfactual** (Tōhoku 2011) | 4 real methods; macro synthetic control **$230B** brackets the documented $210–235B whole-economy headline; methods honestly disagree by scope | [`FINAL_SUBMIT/receipts/counterfactual_4method_REAL.json`](FINAL_SUBMIT/receipts/counterfactual_4method_REAL.json) |
| 9 | **Wordle REINFORCE** (per-episode paired test) | beats random (p=2.7e-18, d=4.27) but **ties the info-aware greedy baseline** (p=0.93) | [`FINAL_SUBMIT/receipts/pass27_B_real_episodic_bootstrap.json`](FINAL_SUBMIT/receipts/pass27_B_real_episodic_bootstrap.json) |
| 10 | **TFT quantile forecaster** on real WTI | **MAE p50 $7.83**, 90,602 params | [`FINAL_SUBMIT/receipts/tft_real_metrics.json`](FINAL_SUBMIT/receipts/tft_real_metrics.json) |
| 11 | **Real SHAP** on the trained BC policy | NOAA feature group ≈ **60%** of attribution | [`FINAL_SUBMIT/receipts/shap_real.json`](FINAL_SUBMIT/receipts/shap_real.json) |
| 12 | **Test suite** | **184 / 184 pass**, CPU-only, exit 0 (~125s) | [`tests/receipts/test_suite_grand_total.json`](tests/receipts/test_suite_grand_total.json) |

Axis-labeled plots for several of these: [`FINAL_SUBMIT/plots/`](FINAL_SUBMIT/plots) (e.g.
[`brent_backtest.png`](FINAL_SUBMIT/plots/brent_backtest.png),
[`conformal_coverage.png`](FINAL_SUBMIT/plots/conformal_coverage.png)).

**Pending (honest):** the full RL leaderboard (real checkpoints × ≥20 seeds, correct paired stats),
the live OpenRouter judge-panel α, and a re-run twin savings figure are **RERUN-PENDING** — see the
[claims ledger](FINAL_SUBMIT/CLAIMS_LEDGER.md) §C/§D. The RL-vs-scripted narrative is decided by
that benchmark, not pre-asserted.

---

## The golden path (the demo)

1. **Signal** — a real event surfaces (a FRED Brent move, a NewsAPI headline, a shipping anomaly).
2. **Assessment** — the analyst returns a strict-JSON risk verdict with a calibrated confidence.
3. **Decision** — the engine + agent recommend a concrete action (reroute / hedge / pre-buy /
   activate backup) grounded in the real 408-dim observation.
4. **Counterfactual** — real methods estimate the $ impact of acting vs not, with a confidence
   interval, on real Tōhoku / Suez / Hormuz analogs ([counterfactual receipt](FINAL_SUBMIT/receipts/counterfactual_4method_REAL.json)).
5. **War room** renders it live, a receipt behind every number
   ([`supplymind/warroom/`](supplymind/warroom)).

The Hormuz scenario library is real 2024–2026 events:
[`supplymind/warroom/scenarios/iran_israel_hormuz_2024_2026.json`](supplymind/warroom/scenarios/iran_israel_hormuz_2024_2026.json).

---

## Quick start

```bash
# 1. Clone + install (installable package)
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git supplymind && cd supplymind
pip install -e .          # or: pip install -r requirements.txt

# 2. Run the test suite (~125s on CPU, offline; 184/184 pass)
pytest tests/ -q

# 3. Verify every headline claim traces to a receipt (CI gate)
python scripts/verify_claims.py

# 4. Start the OpenEnv server
uvicorn server.app:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/reset?task_id=easy_typhoon_response"
```

Reproduce the causal counterfactual receipt directly:

```bash
python -m supplymind.phoenix.counterfactual_v2.causal_methods --analog tohoku_2011 --receipt
```

---

## What this is

**SupplyMind** is an OpenEnv-compliant supply-chain-risk intelligence system:

- **Simulation environment** ([`server/app.py`](server/app.py), [`server/engine/`](server/engine),
  [`server/graders/`](server/graders), [`server/tasks/`](server/tasks)) — deterministic, seeded
  supply-chain graph simulation (nodes, disruptions, finances, rewards, Monte Carlo) behind a
  FastAPI server with reset/step/state/grade endpoints and an easy/medium/hard task registry.
- **RL decision layer** ([`rl/`](rl)) — Gymnasium wrapper (408-dim obs, MultiDiscrete action space,
  action masking), MaskablePPO / QR-DQN(CVaR) / offline RL (BC/CQL/IQL/TD3+BC/DT), ONNX export.
  Trained checkpoints live in [`rl/checkpoints/`](rl/checkpoints).
- **LLM analyst layer** ([`supplymind/llm/`](supplymind/llm)) — a domain-calibrated risk-analyst
  prompt with strict-JSON output. The OpenRouter port + Brier-scored A/B is built and
  **RERUN-PENDING on the key**.
- **Forecasting** ([`supplymind/phoenix/forecast_v2/`](supplymind/phoenix/forecast_v2)) —
  Chronos-Bolt + TimesFM-2 + TabPFN ensemble on real FRED Brent (result 3 above).
- **Counterfactual** ([`supplymind/phoenix/counterfactual_v2/`](supplymind/phoenix/counterfactual_v2))
  — 4 real causal methods (result 8 above).
- **Conformal action filter** ([`supplymind/phoenix/action_v2/`](supplymind/phoenix/action_v2)) —
  split-conformal calibration (result 2 above).
- **War room** ([`supplymind/warroom/`](supplymind/warroom)) — Hormuz-strait geopolitical dashboard,
  backend-driven.
- **Real data corpus** — SEC EDGAR 10-Ks, EM-DAT (16,811 rows), DataCo (180K orders), NOAA IBTRACS,
  FRED, World Bank, Wikipedia. Provenance with per-file SHA-256 in
  [`external_data/PROVENANCE.md`](external_data/PROVENANCE.md).

Documentation: [`FINAL_SUBMIT/MODEL_CARD.md`](FINAL_SUBMIT/MODEL_CARD.md) ·
[`FINAL_SUBMIT/DATASET_CARD.md`](FINAL_SUBMIT/DATASET_CARD.md) ·
[`FINAL_SUBMIT/HONEST_LIMITATIONS.md`](FINAL_SUBMIT/HONEST_LIMITATIONS.md) ·
[`FINAL_SUBMIT/CLAIMS_LEDGER.md`](FINAL_SUBMIT/CLAIMS_LEDGER.md) ·
data sources [`docs/core/DATA_SOURCES.md`](docs/core/DATA_SOURCES.md).

---

## Environment: motivation

Global supply-chain disruptions cost an estimated **$184 billion in 2023**. Events like the 2021
Suez blockage, COVID-era semiconductor shortages, and Taiwan-Strait tensions exposed the fragility
of interconnected networks. SupplyMind simulates an AI agent operating as a **supply-chain risk
manager**: it receives early-warning disruption signals (typhoons, port strikes, sanctions,
cascading geopolitical crises) and takes actions — activating backup suppliers, rerouting shipments,
hedging commodity exposure, expediting orders — to minimize financial impact within a limited budget.

Environment parameters are calibrated against published industry data (see
[`docs/core/DATA_SOURCES.md`](docs/core/DATA_SOURCES.md) for full citations). **Stack:** Python 3.11
+ FastAPI + Pydantic v2 + NetworkX + NumPy.

---

## Action space

One action per step from 7 action types, mapped to the CSCMP risk-management taxonomy (Avoid /
Mitigate / Transfer / Accept-Monitor):

| Action Type | Parameters | Cost | Description |
|---|---|---|---|
| `do_nothing` | None | Free | Take no action. May be optimal when no disruption is active. |
| `activate_backup_supplier` | `target_node_id`, `backup_supplier_id` | 15–30% premium | Switch to a pre-qualified backup. Validates the backup is not itself disrupted. |
| `reroute_shipment` | `target_node_id`, `reroute_via` | Variable | Alternative route; doubles transit time through disrupted ports. |
| `increase_safety_stock` | `target_node_id`, `additional_stock_days` (1–90) | Variable | Extra inventory buffer. |
| `expedite_order` | `target_node_id`, `expedite_mode` (`air`/`rail`/`express_sea`) | 5–10× for air | Faster transport mode. |
| `hedge_commodity` | `commodity`, `hedge_amount_usd` | Hedge premium | Hedge against a commodity price spike. |
| `issue_supplier_alert` | `target_node_id` | Free | Request a supplier status update (information only). |

```json
{ "action_type": "activate_backup_supplier", "target_node_id": "SUP_TSMC", "backup_supplier_id": "SUP_SAMSUNG" }
```

## Observation space

Each step returns a `SupplyMindObservation` with structured data (for programmatic agents) and
natural-language summaries (`situation_summary` full, `compact_summary` ~100–200 tokens for
token-constrained LLMs). Fields include `current_day`, `days_remaining`, `active_signals`,
`new_signals`, `node_statuses`, `financials`, `last_action_result`, `reward`, `done`, `info`.
`DisruptionSignal` carries severity/confidence/region/time-to-impact/lifecycle-phase;
`FinancialSnapshot` carries budget, revenue-at-risk, health score, and Monte-Carlo p50/p95 loss.

## Tasks

| Task | ID | Network | Steps | Budget | Disruptions |
|---|---|---|---|---|---|
| Typhoon Response (Easy) | `easy_typhoon_response` | 12 nodes | 30 | $5M | Single typhoon (Taiwan) |
| Multi-Front Crisis (Medium) | `medium_multi_front` | 25 nodes | 45 | $8M | Port strike + flood + sanctions (concurrent) |
| Cascading Crisis (Hard) | `hard_cascading_crisis` | 40 nodes, 6 countries | 60 | $10M | Taiwan-Strait escalation → shipping + chip cutoff + price spike + cyber |

All scenarios use pre-scripted, real-world-calibrated disruptions for deterministic grading. An
optional `seed` enables scenario jitter (trigger-day and severity variation) to prevent
memorization while preserving structure.

## Reward design

Dense 7-component reward per step, range [−1.0, 1.0]: revenue preservation (35%), stockout penalty
(25%), proactive-action bonus (15%), cost penalty (10%), unnecessary-action penalty (5%), health
maintenance (5%), SLA compliance (5%). Per-step rewards (learning signal) are distinct from grader
scores in [0.0, 1.0] (post-episode evaluation of the full trajectory).

---

## API endpoints (port 8000)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (`200` when ready). |
| `POST` | `/reset` | Reset; accepts `{"task_id": "...", "seed": 42}`. Returns initial observation. |
| `POST` | `/step` | Execute one `SupplyMindAction`. Returns observation. |
| `GET` | `/state` | Current `SupplyMindState`. |
| `GET` | `/tasks` | Available tasks + action schema. |
| `POST` | `/grader` | Grade a completed episode → score in [0.0, 1.0]. |
| `POST` | `/baseline` | Run baseline inference on all 3 tasks. |

Interactive docs at `/docs` (Swagger) and `/redoc`. Full OpenEnv compliance: subclasses
`openenv.core.Environment[ActT, ObsT, StateT]`, grades via `openenv.core.rubrics.TrajectoryRubric`,
exposes `/ws` and `/mcp` WebSocket endpoints, ships a valid [`openenv.yaml`](openenv.yaml).

---

## Baseline scores (deterministic, reproducible)

The zero-LLM scripted agent and do-nothing baselines are fully reproducible (byte-identical across
runs). These are engine baselines, **not** the RL leaderboard (which is RERUN-PENDING, WP6.3):

| Task | Do-Nothing | Scripted Agent |
|---|---|---|
| Typhoon (Easy) | 0.32 | **0.77** |
| Multi-Front (Medium) | 0.17 | **0.70** |
| Cascading (Hard) | 0.32 | **0.67** |

Reproduce: `python scripted_agent.py`. LLM baselines via [`inference.py`](inference.py) with an
OpenAI-compatible key.

> **Historical note.** Earlier v1 (simulated) and v2 (DataCo) offline-RL benchmark tables were
> removed from this README because their paired statistics could not be re-verified in the
> 2026-07-02 audit. The authoritative RL benchmark — real checkpoints × ≥20 seeds with correct
> paired-by-seed tests — is **RERUN-PENDING** (see [ledger](FINAL_SUBMIT/CLAIMS_LEDGER.md) D7).

---

## Repository layout

```
Sleep-Token/
├── supplymind/          # installable package: llm/, warroom/, phoenix/, contracts.py
├── server/              # FastAPI app + engine/ + graders/ + tasks/  (the real deterministic core)
├── rl/                  # RL training + offline agents + checkpoints/
├── benchmark/           # honest benchmark harness (leaderboard RERUN-PENDING)
├── tests/               # 184 pytest tests + receipts/ (sha256-stamped)
├── notebooks/           # pruned notebook set (outputs RERUN-PENDING, WP7.1)
├── external_data/       # real data + tracked fetchers + PROVENANCE.md (per-file SHA-256)
├── scripts/             # tooling incl. verify_claims.py (the credibility gate)
├── FINAL_SUBMIT/        # judge-facing docs, receipts/, plots/
└── docs/                # engineering + data-source docs
```

Package imports: `supplymind.contracts` (Pydantic models), `supplymind.llm.client` (OpenRouter
gateway), `supplymind.warroom.*`, `supplymind.phoenix.*`.

---

## Security & keys

Keys live in `.env` only (never in git or code). See [`SECURITY.md`](SECURITY.md). The
`OPENROUTER_API_KEY` is currently **revoked** — every LLM-panel/analyst-live claim is labeled
RERUN-PENDING and never faked around.

## License

[MIT](LICENSE).
