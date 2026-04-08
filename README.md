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

**An OpenEnv-compliant environment for AI-driven supply chain risk management.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-llama/open-env)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Quick Start

```bash
# Clone and install
git clone https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
cd Supplymind
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Reset the environment (easy task)
curl -X POST http://localhost:8000/reset?task_id=easy_typhoon_response

# Take an action (activate Samsung as backup for TSMC)
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action_type": "activate_backup_supplier", "target_node_id": "SUP_TSMC", "backup_supplier_id": "SUP_SAMSUNG"}'
```

---

## Environment Description and Motivation

Global supply chain disruptions cost an estimated **$184 billion in 2023** alone. Events like the 2021 Suez Canal blockage, COVID-induced semiconductor shortages, and geopolitical tensions in the Taiwan Strait have exposed the fragility of interconnected supply networks.

SupplyMind simulates an AI agent operating as a **supply chain risk manager** navigating these real-world disruptions. The agent receives early-warning disruption signals (typhoons, port strikes, sanctions, cascading geopolitical crises) and must take actions -- activating backup suppliers, rerouting shipments, hedging commodity exposure, expediting orders -- to minimize financial impact on a global supply chain network, all within a limited budget.

**Every parameter is calibrated against published industry data** -- not synthetic estimates. See [DATA_SOURCES.md](DATA_SOURCES.md) for full citations. Key calibration points:

- **Company financials**: TSMC $87.1B revenue (2024 earnings), Apple ~25% of TSMC ($22B/yr, TrendForce), Samsung SDI $20B, CATL $50B, Bosch $55B (annual reports)
- **Semiconductor costs**: TSMC N5 wafer $16,000-$17,000 (SemiAnalysis), lead times 16-20 weeks (Susquehanna Financial Group)
- **Commodity prices**: LME copper $9,100/MT, Freightos container $4,200 Shanghai-LA, Asian Metal rare earths $280/kg, Fastmarkets lithium $14,000/MT
- **Disruption scenarios**: Typhoon Gaemi 2024 (2-day port closure, $1-2B losses per AON/Swiss Re), 2011 Thailand floods ($45.7B loss per World Bank), 2002 ILWU lockout ($1B/day per Anderson Economic Group), August 2022 Taiwan Strait exercises (50-100bp insurance surge per Lloyd's)
- **Supply chain costs**: CSCMP carrying cost 25%, McKinsey dual-sourcing premium 10-30%, IATA air freight 4-12x sea
- **Auto chip shortage calibration**: $210B lost revenue, 7.7M vehicles not produced in 2021 (AlixPartners)

**Stack:** Python 3.11 + FastAPI + Pydantic v2 + NetworkX + NumPy

---

## Action Space

The agent selects **one action per step** from 7 action types, derived from the [CSCMP Supply Chain Risk Management Framework](https://cscmp.org/) taxonomy of operational risk responses. The framework identifies four response categories: **Avoid** (do nothing / withdraw), **Mitigate** (backup suppliers, safety stock, rerouting), **Transfer** (commodity hedging), and **Accept/Monitor** (supplier alerts). Our 7 actions map directly:

| CSCMP Category | SupplyMind Actions |
|---|---|
| **Avoid** | `do_nothing` |
| **Mitigate** | `activate_backup_supplier`, `reroute_shipment`, `increase_safety_stock`, `expedite_order` |
| **Transfer** | `hedge_commodity` |
| **Accept/Monitor** | `issue_supplier_alert` |

This forces prioritization under resource constraints.

| Action Type | Parameters | Cost | Description |
|---|---|---|---|
| `do_nothing` | None | Free | Take no action. May be optimal when no disruption is active. |
| `activate_backup_supplier` | `target_node_id`, `backup_supplier_id` | 15-30% cost premium | Switch production to a pre-qualified backup supplier. **Validates** that the backup is not itself disrupted before activation. |
| `reroute_shipment` | `target_node_id`, `reroute_via` (list of port IDs) | Variable | Use an alternative shipping route to bypass disruptions. **Degrades** transit times (2x) if reroute ports are disrupted. |
| `increase_safety_stock` | `target_node_id`, `additional_stock_days` (1-90) | Variable | Order extra inventory buffer to ride out disruptions. |
| `expedite_order` | `target_node_id`, `expedite_mode` (`air`, `rail`, `express_sea`) | 5-10x for air | Upgrade transport mode for faster delivery. |
| `hedge_commodity` | `commodity`, `hedge_amount_usd` | Hedge premium | Hedge against commodity price spikes (e.g., semiconductors, rare earths). |
| `issue_supplier_alert` | `target_node_id` | Free | Request a status update from a supplier. Information-only action. |

**Action model** (`SupplyMindAction`):
```json
{
  "action_type": "activate_backup_supplier",
  "target_node_id": "SUP_TSMC",
  "backup_supplier_id": "SUP_SAMSUNG"
}
```

---

## Observation Space

Each step returns a `SupplyMindObservation` with both **structured data** (for programmatic agents) and **natural language summaries** (for LLM-based agents). Two summary formats are provided: a full `situation_summary` and a token-efficient `compact_summary`.

| Field | Type | Description |
|---|---|---|
| `current_day` | `int` | Current simulation day (0-based) |
| `days_remaining` | `int` | Days left in the episode |
| `active_signals` | `list[DisruptionSignal]` | All currently active disruption signals |
| `new_signals` | `list[DisruptionSignal]` | Signals that appeared this step |
| `node_statuses` | `list[SupplierStatus]` | Status of every supply chain node |
| `financials` | `FinancialSnapshot` | Budget, revenue at risk, costs, health score, Monte Carlo projections |
| `last_action_result` | `ActionResult` | Success/failure and cost of the previous action |
| `situation_summary` | `str` | Full human-readable situation summary for LLM reasoning |
| `compact_summary` | `str` | Token-efficient summary (~100-200 tokens) with top risks, budget, disruptions, and urgent action |
| `reward` | `float` | Reward for this step |
| `done` | `bool` | Whether the episode has ended |
| `info` | `dict` | Additional metadata (reward component breakdown, Monte Carlo projections) |

**DisruptionSignal** includes: `signal_id`, `disruption_type`, `severity` (0-1), `confidence` (0-1), `affected_region`, `affected_node_ids`, `time_to_impact_hours`, `estimated_duration_days`, `lifecycle_phase` (warning / active / recovery / resolved), and a human-readable `description`.

**FinancialSnapshot** includes: `budget_remaining`, `cumulative_revenue_lost`, `supply_chain_health_score` (0-100), `monte_carlo_p50_loss`, `monte_carlo_p95_loss`, and `commodity_price_changes`.

---

## Tasks

SupplyMind provides three tasks with clear difficulty progression. All scenarios use pre-scripted disruptions for deterministic, reproducible grading.

### Task 1: Typhoon Response (Easy)

| Property | Value |
|---|---|
| **Task ID** | `easy_typhoon_response` |
| **Network** | 12 nodes, 2 tiers |
| **Episode Length** | 30 steps |
| **Budget** | $5,000,000 |
| **Disruptions** | Single typhoon affecting Taiwan |
| **Challenge** | Agent receives 72-hour warning signals and must activate backup supplier and expedite critical orders before impact. Straightforward cause-and-effect. |

### Task 2: Multi-Front Crisis (Medium)

| Property | Value |
|---|---|
| **Task ID** | `medium_multi_front` |
| **Network** | 25 nodes, 3 tiers |
| **Episode Length** | 45 steps |
| **Budget** | $8,000,000 |
| **Disruptions** | US port strike + Thailand flooding + Chinese supplier sanctions (concurrent) |
| **Challenge** | Budget only covers mitigation for roughly 2 of 3 disruptions. The agent must triage and prioritize under resource constraints. |

### Task 3: Cascading Crisis (Hard)

| Property | Value |
|---|---|
| **Task ID** | `hard_cascading_crisis` |
| **Network** | 40 nodes, 3 tiers, 6 countries |
| **Episode Length** | 60 steps |
| **Budget** | $10,000,000 |
| **Disruptions** | Taiwan Strait escalation triggers shipping disruption, semiconductor cutoff, commodity price spikes, and a cyber attack |
| **Challenge** | Cascading failures create compounding effects. Very tight budget relative to the scale of disruption forces hard trade-offs. Requires long-horizon planning. |

---

## Reward Design

SupplyMind uses a **dense 7-component reward** computed every step (not sparse end-of-episode). Each step's reward is in the range [-1.0, 1.0].

| Component | Weight | What It Measures |
|---|---|---|
| Revenue preservation | 35% | Fraction of at-risk revenue successfully protected |
| Stockout penalty | 25% | Penalizes nodes that run out of inventory |
| Proactive action bonus | 15% | Rewards acting before disruptions hit (early warning response) |
| Cost penalty | 10% | Penalizes overspending relative to budget |
| Unnecessary action penalty | 5% | Penalizes actions taken when no disruption threatens the target |
| Health score maintenance | 5% | Rewards maintaining high supply chain health score |
| SLA compliance | 5% | Rewards meeting delivery SLA targets |

This design rewards partial progress, penalizes wasteful or destructive behavior, and provides useful signal throughout the entire trajectory.

**Note:** Per-step rewards (range [-1.0, 1.0]) are distinct from grader scores (range [0.0, 1.0]). The per-step reward guides agent learning during the episode. The grader score is computed after the episode ends by examining the full action-observation history and engine state. These are intentionally different metrics serving different purposes.

---

## Design Decisions

Several deliberate design choices shape the environment:

- **Budget constraint**: Mitigation budgets ($5M-$10M) are intentionally small relative to supply chain exposure ($28B-$268B annual revenue). This mirrors real crisis management where resources are always insufficient, forcing the agent to **triage** rather than mitigate everything. A supply chain risk manager with unlimited budget is not an interesting problem.

- **Compressed timelines**: Real disruptions (port strikes, floods, geopolitical crises) unfold over weeks to months. Episodes compress these to 30-60 simulation days to keep training practical. Disruption parameters (severity, duration) are scaled proportionally so relative impact is preserved.

- **Single action per step**: Agents select one action per day, forcing prioritization. Real risk managers also face bandwidth constraints -- they can't execute 10 mitigations simultaneously.

- **Pre-scripted disruptions with seed-based variation**: Base scenarios use hand-crafted, real-world-calibrated disruption scripts for reproducible grading. Passing an optional `seed` parameter to `reset()` enables **scenario jitter** -- trigger days shift by 0-2 days, peak severity varies by +/-8%, and affected nodes may swap with same-type graph neighbors. Same seed = same episode (reproducible). No seed = default deterministic behavior (backward compatible). This prevents agent memorization while preserving the calibrated scenario structure.

- **Emergent cascade triggers**: Beyond pre-scripted disruptions, the engine dynamically injects **supply shortage cascades** when a supplier stays offline long enough to exhaust downstream warehouse inventory buffers (inventory < 3 days AND offline duration > buffer). Cascade severity is proportional to the dependency ratio between the disrupted supplier and the warehouse. This creates emergent, agent-responsive failure propagation that compounds the pre-scripted scenarios.

- **Action validation and degradation**: The environment validates actions realistically. `activate_backup_supplier` checks whether the backup is itself disrupted (risk > 50% or offline) and rejects with a clear error if so -- preventing the agent from wasting budget on non-functional backups. `reroute_shipment` checks reroute port status and doubles transit times through disrupted ports, with a warning in the action result.

- **Dual observation format**: Each observation includes both a full `situation_summary` (~1500 tokens, rich context for large-context LLMs) and a `compact_summary` (~100-200 tokens, top 3 risks + budget + urgent action for token-constrained models). This ensures the environment is usable across different agent architectures.

---

## API Endpoints

All endpoints are served on port **8000**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check. Returns `200` when the server is ready. |
| `POST` | `/reset` | Reset the environment. Accepts `{"task_id": "...", "seed": 42}`. Optional `seed` enables scenario jitter for episode variation. Returns initial `SupplyMindObservation`. |
| `POST` | `/step` | Execute one action. Accepts a `SupplyMindAction` JSON body. Returns `SupplyMindObservation`. |
| `GET` | `/state` | Returns current `SupplyMindState` (episode metadata, step count, cumulative reward). |
| `GET` | `/tasks` | Returns the list of available tasks and the action schema. |
| `POST` | `/grader` | Grade a completed episode. Returns a score in [0.0, 1.0]. |
| `POST` | `/baseline` | Run baseline inference on all 3 tasks. Returns scores. |

Interactive API docs are available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Setup and Usage

### Local Installation

```bash
# Requires Python 3.11+
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t supplymind .

# Run
docker run -p 8000:8000 supplymind
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | For baseline | Hugging Face API key (or any OpenAI-compatible key). Competition **MANDATORY** variable. Falls back to `OPENAI_API_KEY`. |
| `API_BASE_URL` | For baseline | API endpoint for the LLM (default: `https://router.huggingface.co/v1`). Competition **MANDATORY** variable. |
| `MODEL_NAME` | For baseline | Model identifier (default: `gpt-4o`). Competition **MANDATORY** variable. |
| `OPENAI_API_KEY` | Fallback | Accepted as a fallback for `HF_TOKEN`. |
| `ENV_URL` | For inference.py | URL of the deployed SupplyMind server (default: `http://localhost:8000`). |

### Running the Baseline

```bash
# Via /baseline endpoint (runs inside the server process):
export HF_TOKEN="your-hf-token"
export MODEL_NAME="gpt-4o"
curl -X POST http://localhost:8000/baseline

# Via standalone inference script (connects to deployed server via HTTP):
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-hf-token"
export ENV_URL="http://localhost:8000"
python inference.py
```

The baseline agent uses the OpenAI-compatible API to make decisions across all three tasks and returns reproducible scores.

---

## Baseline Scores

All scores below are reproducible by running the corresponding script in this repository.

| Task | Do-Nothing | Scripted Agent | Gemini 3 Flash |
|---|---|---|---|
| Typhoon Response (Easy) | 0.3211 | **0.7711** | 0.6527 |
| Multi-Front Crisis (Medium) | 0.1650 | **0.6962** | 0.5613 |
| Cascading Crisis (Hard) | 0.3211 | **0.6715** | ~0.65* |
| **Average** | 0.2691 | **0.7129** | ~0.62 |

*Hard task Gemini score estimated from 21/60 steps completed (free-tier API quota limit).

**How to reproduce:**
- Do-Nothing: `python -c "..."` (any action→do_nothing loop)
- Scripted Agent: `python scripted_agent.py` (zero-LLM, deterministic heuristics)
- Gemini 3 Flash: `MODEL_NAME=gemini-3-flash-preview HF_TOKEN=<key> python inference.py`

Expected score ranges for LLM agents:

| Task | Difficulty | Expected LLM Score Range |
|---|---|---|
| Typhoon Response | Easy | 0.65 -- 0.85 |
| Multi-Front Crisis | Medium | 0.45 -- 0.70 |
| Cascading Crisis | Hard | 0.50 -- 0.75 |

**Score interpretation:**
- **0.00 -- 0.20**: Agent took no meaningful actions or made critical errors
- **0.20 -- 0.40**: Minimal engagement; some natural revenue preserved but no real mitigation
- **0.40 -- 0.60**: Competent triage with partial coverage; typical for medium/hard tasks
- **0.60 -- 0.80**: Strong performance; proactive, well-targeted, budget-efficient
- **0.80 -- 1.00**: Near-optimal; requires surgical precision across all grader components

The do-nothing scores are nonzero because some revenue is naturally preserved even without intervention. The **action_coverage** and **active_mitigation** grader components explicitly penalize agents that take no cost-bearing mitigation actions.

**Reproducibility:** All scores are deterministic. Running the same strategy N times produces byte-identical scores (verified by `TestScoreVariance` -- 5x runs, 0 variance).

---

## OpenEnv Compliance

SupplyMind fully implements the [OpenEnv specification](https://github.com/meta-llama/open-env):

- **OpenEnv SDK integration**: Subclasses `openenv.core.Environment[ActT, ObsT, StateT]` with typed generics
- **OpenEnv Rubric framework**: Grading uses `openenv.core.rubrics.TrajectoryRubric` with `RubricDict` for task-specific sub-rubrics
- **WebSocket support**: `/ws` (persistent sessions) and `/mcp` (MCP JSON-RPC) WebSocket endpoints via `openenv.core.env_server.HTTPEnvServer`
- Typed Pydantic v2 models for actions, observations, and state
- `step(action)` returns observation, reward, done, info
- `reset(task_id, seed?)` returns a clean initial observation; optional seed enables episode variation
- `state()` returns episode metadata
- Valid `openenv.yaml` with environment metadata and task list
- 3 tasks with deterministic, reproducible graders that produce different scores for different strategies
- Dense per-step reward signal (not sparse binary)
- Dual observation summaries: full `situation_summary` + compact `compact_summary` for LLM agents
- Emergent cascading behavior via dynamic disruption injection
- Action validation: disrupted backup rejection, reroute port degradation
- Baseline inference script using the OpenAI API
- Working Dockerfile for containerized deployment

---

## Project Structure

```
supplymind/
├── models.py              # Pydantic v2 models (action, observation, state)
├── openenv.yaml           # OpenEnv metadata and task definitions
├── inference.py           # Competition entrypoint (standalone, uses OpenAI client)
├── baseline.py            # Baseline agent (imported by server /baseline endpoint)
├── client.py              # Example HTTP client
├── server/
│   ├── app.py             # FastAPI endpoints (thin HTTP layer)
│   ├── supply_environment.py  # Environment wrapper (reset, step, grade)
│   ├── engine/            # Pure simulation logic (graph, financial, rewards, disruptions)
│   ├── tasks/             # Task definitions (easy, medium, hard)
│   ├── graders/           # Deterministic grading logic
│   └── data/              # JSON data files (graphs, disruption scenarios, commodities)
├── scripted_agent.py      # Deterministic rule-based agent (no LLM needed)
├── tests/                 # 154 pytest tests
├── Dockerfile             # Multi-stage Docker build
├── pyproject.toml         # Project config with entry points
├── requirements.txt       # Python dependencies
├── uv.lock                # Deterministic dependency lock
├── DATA_SOURCES.md        # Real-world calibration sources (40+ citations)
└── README.md
```

---

## License

MIT
