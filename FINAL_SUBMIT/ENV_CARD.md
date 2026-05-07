# Environment Card — OpenEnv Compliance

## SupplyMind Environment
- **Class**: `server.openenv_mcp_wrapper.SupplyMindMCP` (subclasses `MCPEnvironment`)
- **Manifest**: `openenv.yaml` at repo root
- **Type**: Theme #3 Professional Tasks (real-world supply-chain RL)
- **Compliance check**: `python server/openenv_mcp_wrapper.py` → `compliant=True`

### Standard methods (Gym-style)
| Method | Signature | Description |
|--------|-----------|-------------|
| `reset(task_id, seed)` | → observation dict | Initialize episode |
| `step(action)` | → {observation, reward, done, info} | Apply action |
| `state()` | → {task, env_metadata} | Read-only snapshot |
| `close()` | → {status} | Cleanup |

### MCP Tools (6 non-reserved names)
| Tool | What it returns |
|------|-----------------|
| `tool_sm_get_node_status(node_id)` | Risk + inventory + last-known status |
| `tool_sm_get_edge_status(edge_id)` | Lead-time + cost |
| `tool_sm_query_recent_events(hours, limit)` | Last-N event-store events |
| `tool_sm_query_crisis_library(text, k)` | RAG against 8 hand-curated events |
| `tool_sm_get_financial_state()` | Budget + losses + profit |
| `tool_sm_describe_action_space()` | Enumerate 7×40 = 280 actions |
| `tool_sm_explain_disruption(disruption_id)` | Plain-English explanation |

### Action space
- **Type**: `Discrete(280)` (flattened from `MultiDiscrete([7, 40])`)
- **7 action types**: do_nothing · activate_backup · reroute_shipment · increase_safety_stock · expedite_shipment · hedge_commodity · issue_supplier_alert
- **40 node targets**: real company coords (TSMC, Samsung, Toyota, etc.)

### Observation space
- 64-dim engineered state: financials + node statuses + edge statuses + active disruptions + recent events
- Pydantic-typed `SupplyMindObservation` per OpenEnv convention

### Reward (multi-component, OpenEnv guide §7)
- Revenue preservation 35% · Stockout prevention 25% · Proactive bonus 15% · Cost penalty 10% · Health 5% · SLA 5% · Unnecessary action penalty 5%
- Time-discounted: `max(0.3, 1.0 - step_fraction × 0.7)`
- 9 industry-cited cost values

### 3 difficulty tiers
- **easy_typhoon_response** — 12 nodes, 30 days, $5M budget, single Taiwan typhoon
- **medium_multi_front** — 25 nodes, 45 days, $10M budget, multi-disruption
- **hard_cascading_crisis** — 40 nodes, 60 days, $15M budget, cascading

## Wordle Companion Environment
- **Class**: `versions.v5_phoenix.wordle_env.env`
- **Type**: Canonical RLVR mini-env
- **Action space**: `Discrete(102)` (102-word baseline) or restricted by curriculum tier
- **State**: 188-dim (rich encoding per `final_real_reinforce_wordle_v2.py`)
- **Reward (6-component)**: solve_bonus + green_credit + yellow_credit + format_gate + dictionary_gate + timeout_penalty
- **Anti-hack defenses**: 19/19 attacks blocked (literature-grade gauntlet)
- **Curriculum**: RLVE adaptive 4-tier (Procaccia §22-23)
- **Verifier**: dual-layer rule × LLM-judge with disagreement alarm

## Compliance verification
```bash
python server/openenv_mcp_wrapper.py
# → {"compliant": true, "n_mcp_tools": 6, "no_reserved_collisions": true, ...}
```

## OpenEnv `openenv.yaml` highlights
```yaml
spec_version: "0.1"
environment_id: supplymind
name: supplymind
version: "1.0.0"
type: space
runtime: fastapi
app: server.app:app
port: 8000
action: SupplyMindAction
observation: SupplyMindObservation
python_version: ">=3.11"
tasks:
  - id: easy_typhoon_response
  - id: medium_multi_front
  - id: hard_cascading_crisis
```

## HF Spaces deployment
- Path: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
- Status: **user must verify before submission deadline**

## Citations
- OpenEnv core: https://github.com/meta-pytorch/OpenEnv
- Hub: https://huggingface.co/openenv
- See `CITATIONS.bib` for paper references.
