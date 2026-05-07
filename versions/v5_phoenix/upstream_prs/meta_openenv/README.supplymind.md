# examples/supplymind

An OpenEnv-compliant supply-chain risk management environment.

## What it is

Three difficulty-calibrated RL tasks on a real-data-calibrated supply-chain
simulator:

| Task | Nodes | Steps | Budget |
|---|---|---|---|
| easy_typhoon_response | 12 | 30 | $5 M |
| medium_multi_front | 25 | 45 | $8 M |
| hard_cascading_crisis | 40 | 60 | $10 M |

Observation: 408-dim (network state + financials + risk signals + disruption
history).  Action: MultiDiscrete[7, 40] = 7 action types × 40 target nodes.
Grader: deterministic (revenue preservation, stockout penalty, action cost,
health maintenance).

## Running

### Docker (recommended)

```bash
cd examples/supplymind
docker compose up -d
curl http://localhost:8000/health
# -> {"status": "ok", ...}
```

### Local

```bash
pip install openenv-core stable-baselines3 sb3-contrib gymnasium
cd examples/supplymind
uvicorn supplymind_env:app --host 0.0.0.0 --port 8000
```

### OpenEnv

```python
import openenv
env = openenv.make("supplymind", task_id="easy_typhoon_response")
obs, info = env.reset(seed=42)
for _ in range(30):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
grade = env.grade()
print(f"Episode score: {grade['score']}")
```

## Trained policies

ONNX bundles in `policies/`:

- `maskable_ppo_easy.onnx`   (0.97 MB)
- `maskable_ppo_medium.onnx`
- `maskable_ppo_hard.onnx`

Numerical round-trip error vs stable-baselines3 MaskablePPO: max 1.9e-6.

## Benchmarks (from R6 Euclidian, 10,800 episodes)

| Task | Policy | Reward mean | 95 % CI |
|---|---|---|---|
| easy | MaskablePPO | 1.200 | [1.186, 1.215] |
| easy | Random | 0.748 | [0.738, 0.757] |
| easy | Greedy | 0.980 | [0.980, 0.981] |
| medium | MaskablePPO | 2.776 | [2.758, 2.795] |
| medium | Random | -0.972 | [-1.025, -0.921] |
| medium | Greedy | -1.807 | [-1.813, -1.802] |
| hard | MaskablePPO | 2.652 | [2.596, 2.708] |
| hard | Random | -1.309 | [-1.365, -1.258] |
| hard | Greedy | -1.414 | [-1.448, -1.385] |

All MaskablePPO CIs strictly above all baselines on all three tasks.

## Data sources (261 K real points; no synthetic substitution)

- DataCo Supply Chain (Kaggle): 180,519 orders
- NOAA IBTRACS: 243,495 storm records / 4,289 typhoons
- FRED Economic Data: 17,011 points × 12 series
- World Bank WGI: 214 countries × 6 dims × 24 years
- SEC 10-K filings: 25 Fortune 500
- Wikipedia crisis articles: 26 curated

Simulation parameters (lead times, dual-sourcing premiums, air-freight
multipliers, etc.) all calibrated from industry reports. See `docs/core/DATA_SOURCES.md`
in the parent SupplyMind repo for ≥40 citations.

## Compliance

19 formal OpenEnv compliance tests in `tests/test_openenv_compliance.py`.
`pytest tests/ -q` should report 19 passed.

## License

MIT.

## Credit

Submitted by ShAuRyA-Noodle, 2026-04-25 (Meta PyTorch OpenEnv Hackathon 2026
finals). Parent SupplyMind repo: https://github.com/ShAuRyA-Noodle/Sleep-Token
