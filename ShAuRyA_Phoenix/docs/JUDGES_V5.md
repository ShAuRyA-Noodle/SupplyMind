# Judges' Quick Reference — Phoenix v5

**Meta PyTorch OpenEnv Hackathon 2026 Finals**. You have 4 minutes. Here's the path.

---

## The 30-second pitch

SupplyMind v5.0-phoenix-ascensionism is an OpenEnv-compliant supply-chain risk
environment. 13 local SOTA models, 261K real data points, **275 passing tests**
(277 collected; 2 live tests skipped unless API keys are present), 20 one-bash-command receipts, live geopolitical
pipeline, Karpathy autoresearch loop with two accepted experiments, a
DPO-fine-tuned risk judge, an OpenEnv Arena where you can drop your own
PyTorch policy, and two upstream PRs — to **meta-pytorch/openenv** and
**alibaba/ROLL**.

All built solo in 3 months. Everything reproducible. No synthetic substitution
anywhere.

---

## The live demo (90 seconds, on my laptop)

```bash
# Start the Phoenix server (v4 routes + v5 routes in one process)
uvicorn ShAuRyA_Phoenix.server.phoenix_app:app --host 0.0.0.0 --port 8000 &

# Optional: freeze an offline replay cache for resilience
python -m ShAuRyA_Phoenix.realtime_v5.freeze_cache

# Live Hormuz assessment (hits real 2026 NewsAPI + FRED Brent if keys present)
curl -X POST http://localhost:8000/live/hormuz-closure \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_text": "Iran threatens Hormuz closure; Brent at $123/bbl.",
    "region": "hormuz",
    "enable_llm_judges": true,
    "include_recent_signals": true,
    "k_analogs": 3
  }' | jq
```

Expected:
- Top analog match @ ≥ 0.9 similarity
- risk_level = HIGH or CRITICAL
- 5 recommended actions (hedge, reroute, backup, safety-stock, alert)
- **Counterfactual**: no-action loss vs with-plan loss in USD, savings %

If NewsAPI is rate-limited or offline, add `?replay=1` — same shape, served
from the frozen cache.

---

## Drop-in-your-policy arena (60 seconds)

```bash
# UI
python -m ShAuRyA_Phoenix.arena.gradio_app
# -> http://localhost:7860, upload policy.pt, wait ~90s

# Or CLI
curl -X POST http://localhost:8000/arena/run \
  -F "policy=@/path/to/policy.pt" \
  -F "name=my_awesome_agent" \
  -F "episodes=50"
```

You'll get back:

```json
{
  "policy_name": "my_awesome_agent",
  "per_task": {
    "easy_typhoon_response":  {"reward_mean": 1.15, "ci95": [1.09, 1.21], ...},
    "medium_multi_front":     {"reward_mean": 2.11, "ci95": [2.05, 2.17], ...},
    "hard_cascading_crisis":  {"reward_mean": 2.35, "ci95": [2.20, 2.49], ...}
  },
  "overall_reward_mean": 1.87,
  "overall_ci95": [1.82, 1.92],
  "rank_against_baseline": "near MaskablePPO baseline"
}
```

Current leaderboard baselines (from `R6_EUCLIDIAN.json`, 10,800 episodes):

| Rank | Policy | Overall reward mean |
|---|---|---|
| 1 | MaskablePPO v3 (ours) | +2.209 |
| 2 | RecurrentPPO v3 | +1.081 |
| 3 | PPO v3 (no masking) | +0.947 |
| 4 | A2C v3 | +0.874 |
| 5 | Random | −0.511 |
| 6 | Greedy | −0.749 |

---

## Reproducibility receipts (30 seconds each)

```bash
bash ShAuRyA_Phoenix/receipts_v2/R5_GRANITE_mxbai_P1.reproduce.sh   # -> 0.9622
bash ShAuRyA_Phoenix/receipts_v2/R5_BEIR_snowflake_nDCG10.reproduce.sh   # -> 0.971
bash ShAuRyA_Phoenix/receipts_v2/R4_2JUDGE_Krippendorff_alpha.reproduce.sh   # -> 0.7499
bash ShAuRyA_Phoenix/receipts_v2/R6_MaskingAblation_easy_lift.reproduce.sh   # -> 26.77
bash ShAuRyA_Phoenix/receipts_v2/R6_GCN_easy_MAE_vs_MLP.reproduce.sh   # -> 48.02
bash ShAuRyA_Phoenix/receipts_v2/V5_Autoresearch_best_experiment.reproduce.sh   # -> s2_higher_entropy
bash ShAuRyA_Phoenix/receipts_v2/V5_Arena_baseline_leaderboard.reproduce.sh   # -> 6 baselines
```

Each receipt emits `command`, full `stdout`, `exit_code`, `expected`, `actual`,
`match`, `hardware`, `timestamp`. Grade-A format from the `verification-before-
completion` discipline.

---

## The 5-minute full inspection path

1. `cat ShAuRyA_Phoenix/README.md` — v5 overview
2. `cat ShAuRyA_Phoenix/docs/PREPRINT_V5.md` — technical abstract
3. `cat ShAuRyA_Phoenix/receipts_v2/INDEX.md` — 20 receipts
4. `cat ShAuRyA_Phoenix/autoresearch_fixed/lab_notebook.md` — Karpathy loop
5. `cat ShAuRyA_Phoenix/upstream_prs/meta_openenv/PR.md` and `upstream_prs/alibaba_roll/PR.md`
6. `pytest tests/ ShAuRyA_Supplymind/tests/ ShAuRyA_Phoenix/tests/ -q` — all green

---

## What to ask me in person

1. **"Show me the live Hormuz assessment with the judge panel."** — 90 sec.
2. **"Upload my policy to the Arena."** — 1–3 min.
3. **"Walk me through the autoresearch lab notebook."** — shows s1 baseline, s2 accepted over threshold, s3/s4/s5 with fixes applied and pending rerun.
4. **"Where does SupplyMind fail?"** — honest answer: (a) Arena baselines are pre-seeded, not re-run at submission time; (b) Phoenix autoresearch has 3 pending seeds; (c) ROLL install is Phase-A/B/C gated.

---

## If anything fails

- **NewsAPI rate-limited**: `FORCE_REPLAY=1 uvicorn ...` — offline replay cache with 8 real 2024-2026 Iran/Israel/Hormuz events.
- **Ollama not warm**: live endpoint falls back to deterministic rubric judge. Arena and Counterfactual Twin don't use Ollama.
- **ROLL not installed**: everything v4 and most v5 works unchanged. DPO-judge falls back to `trl.DPOTrainer` (same scientific result).
- **Phoenix server won't start**: run `pytest ShAuRyA_Phoenix/tests/ -q` first — the test that lives closest to each router will tell you what's broken.

---

## Open-source contributions (the hackathon's key signal)

1. **meta-pytorch/openenv** — SupplyMind as a reference environment. Draft at `upstream_prs/meta_openenv/`.
2. **alibaba/ROLL** — SupplyMind as a registered agentic-RL training target. Draft at `upstream_prs/alibaba_roll/`.
3. **obra/superpowers-marketplace** — `supplymind-skills` skill pack (3 skills: benchmark-runner, autoresearch-experiment, live-demo-orchestrator). Source at `supplymind_skills/`.

---

*Contact: see `README.md`. Built solo. No compromises. Real data everywhere.*
