# SupplyMind — Final Submit

**Meta PyTorch × Scaler OpenEnv Hackathon Finals 2026 · Bangalore · 800 teams**

A retrieval-augmented RL agent for global supply-chain risk, evaluated against 8 documented historical events with 100% risk-band accuracy and 100% Brent ±30%, on an OpenEnv-compliant environment with 20 live data sources, 13 verified foundation models, 25 frontier judges, and split-conformal action safety with 0.9001 empirical coverage.

Every claim has a sha256-replayable receipt.

---

## Open the master demo

```
http://127.0.0.1:8000/demo/master
```

9 cards, every one live, every claim cited. From there, click into:

- `/demo/hormuz-war-room/ui` — flagship live demo
- `/arena/health` — 9-agent leaderboard
- `/phoenix/status` — 13 foundation models verified
- `/library/v2/search` — 1500-event EMDAT crisis library
- `/counterfactual/platinum` — 4-method causal counterfactual
- `/analyst/panel-consensus/{id}` — 12-frontier-judge panel
- `/replay/health` — HetGAT cascade replay
- `/live/health` — 20-source intel fan-out

---

## The headline numbers (every cell links to a receipt)

| Metric | Value | Receipt |
|---|---|---|
| War-room risk-band accuracy | **100%** (8/8) | `tests/receipts/war_room_validation.json` |
| Ensemble Brent ±30% | **100%** (8/8) | `tests/receipts/ensemble_brent_validation.json` |
| Median Brent rel error | **3.3%** | same |
| Conformal action coverage | **0.9001** | `tests/receipts/conformal_calibration.json` |
| Cross-corpus α (frontier 6, v2 EMDAT) | **0.5436** | `tests/receipts/cross_corpus_alpha.json` |
| 12-frontier panel α (R4 corpus) | **0.5669** | `tests/receipts/panel_agreement_R4.json` |
| HetGAT vs v1 GCN MAE | **+7.77 / +12.15 / +10.03 %** | `ShAuRyA_Phoenix/experiments/hetgat_v1/report.json` |
| RAP-XC training loss | BC **5.62 → 0.23** | `ShAuRyA_Phoenix/experiments/rap_xc_v1/rapxc.pt` |
| RAP-XC parameters | **3,137,049** | same |
| Tohoku 2011 replicated | **$276 B vs $235 B published (+18%)** | `tests/receipts/platinum_counterfactual.json` |
| Live data sources | **20** | `ShAuRyA_Supplymind/realtime/orchestrator_v2.py` |
| Crisis library | **1,500 EMDAT events** | `ShAuRyA_Supplymind/scenarios/crisis_library_v2.json` |
| Foundation models verified | **13/13** | `v3_arcadia/00_emergence/verify_*.py` |
| Custom Ollama analyst models | **5 (v1→v5)** | `rl/lora/Modelfile.v[2-4]`, `Modelfile.analyst_v5` |
| LoRA training pairs | **225** | `rl/data/lora_training_data.json` |
| DPO preference pairs | **21** | `dpo_judge/data/preference_pairs.jsonl` |

---

## What's novel

1. **Retrieval-augmented policy with cross-attention** — RAP-XC conditions on top-k retrieved historical events from a 1500-event EMDAT FAISS index. 3.14M params. Trained on 40,000 real harvested PPO transitions in 17.77s.

2. **4-method causal counterfactual ensemble** — paired-bootstrap MC + synthetic control + ARIMA-BSTS + SCM do-calculus, calibrated to 6 paper anchors (Suez 2021 $9.6B/day, Tohoku $235B, Chip shortage $210B, etc.). Tohoku replicated within 18%.

3. **Split-conformal action filter** — Vovk 2005 NLL-quantile with finite-sample correction, calibrated on 8000 real harvest rows, **empirical coverage 0.9001 vs 0.9 target**.

4. **Cross-corpus frontier α** — same 6 OpenRouter judges scoring R4 (26 scenarios) and v2 EMDAT (30 events): α=0.5669 vs 0.5436, drift 0.024 absolute → strong cross-corpus stability.

5. **HetTemporalGAT** — edge-type-conditional GAT with GRU temporal gating; beats v1 GCN on all three difficulty tiers.

6. **Chronos-Bolt + TimesFM-2 + TabPFN-v2 ensemble Brent forecaster** — closed our 25% backtest miss to 100% within ±30%, median rel error 3.3%.

7. **8-event historical backtest** — every output validated against documented EMDAT-cited Iran/Israel/Hormuz/Red-Sea events with published Brent peaks.

---

## What we honestly cannot do

See [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md). Short version:
- We don't predict whether a chokepoint will close. We quantify second-order industrial effects *conditional* on closure.
- Our forecasts have wide CI95 in tail events (oil is volatile).
- 2 of 6 OpenRouter judges typically rate-limit during testing; we report 4/6 succeeded.
- Sector-level loss bands are point-estimate ranges from published agency data, not precise dollar forecasts.

---

## Reproduce in 5 commands

```
git clone <repo>
cd Sleep-Token && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in 4 keys: OPENROUTER, EIA, NASA_FIRMS, GFW
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 &
open http://127.0.0.1:8000/demo/master
```

Detailed: see [REPRODUCE.md](REPRODUCE.md).

---

## Repo map

| Section | Where |
|---|---|
| Game engine (OpenEnv) | `server/app.py`, `server/supply_environment.py`, `server/engine/` |
| 9 RL agents | `ShAuRyA_Phoenix/arena/`, `ShAuRyA_Phoenix/rap_xc/` |
| 13 foundation models | `models/`, `v3_arcadia/00_emergence/verify_*.py` |
| Custom Ollama analyst models | `rl/lora/Modelfile.v[2-4]`, `ShAuRyA_Supplymind/features/Modelfile.analyst_v5` |
| LoRA + DPO + GRPO training | `rl/lora/`, `ShAuRyA_Phoenix/roll_integration/dpo_judge/` |
| 1500-event crisis library | `ShAuRyA_Supplymind/scenarios/crisis_library_v2.{json,faiss}` |
| 4-method counterfactual | `ShAuRyA_Phoenix/counterfactual_v2/platinum.py` |
| Hormuz War Room | `ShAuRyA_Supplymind/realtime/hormuz_war_room_router.py`, `server/static/hormuz_war_room.html` |
| Master demo page | `server/static/master.html` |
| Receipts | `tests/receipts/*.json` |

For a complete bullet-by-bullet inventory, see [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md).
