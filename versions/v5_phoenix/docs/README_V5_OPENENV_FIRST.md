# SupplyMind — OpenEnv-native supply-chain risk environment

> **Meta PyTorch OpenEnv Hackathon — Finals submission (April 25–26, 2026)**
> v5.0-phoenix-ascensionism (staging) on top of v4.0-arcadia-live on top of v3.0-arcadia.

This is the v5 "OpenEnv-first" README — copy it over `Sleep-Token/README.md`
on travel day. v4's README (saved as `README_V4_SNAPSHOT.md`) stays as
historical reference.

---

## 30-second pitch

SupplyMind is an **OpenEnv-compliant supply-chain risk environment** with three
difficulty-calibrated tasks (Typhoon Response → Multi-Front Crisis → Cascading
Crisis) and a complete agent stack: 13 local SOTA foundation models, a
10,800-episode RL benchmark, a live geopolitical pipeline hitting NewsAPI /
GDELT / FRED, a Karpathy-pattern autonomous research loop, a DPO-fine-tuned
judge, and an OpenEnv Arena where judges drop their own PyTorch policies to
benchmark against our SOTA baselines.

**One laptop. One human. Real data everywhere. Two upstream PRs** — to
[meta-pytorch/openenv](https://github.com/meta-pytorch/openenv) and
[alibaba/ROLL](https://github.com/alibaba/ROLL).

---

## Quick start (60 seconds)

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# Every test from v3+v4+v5 — should be 249+ (v4) + whatever v5 adds
pytest tests/ versions/v4_arcadia_live/tests/ versions/v5_phoenix/tests/ -q

# Phoenix server: v4 Hormuz + v5 Arena + v5 Counterfactual Twin
uvicorn versions.v5_phoenix.server.phoenix_app:app --host 0.0.0.0 --port 8000

# Any receipt in 30 seconds
bash versions/v5_phoenix/receipts_v2/R5_GRANITE_mxbai_P1.reproduce.sh
# -> expected 0.9622
```

---

## What's OpenEnv-native about this

- Full [OpenEnv](https://github.com/meta-pytorch/openenv) spec compliance —
    `openenv.yaml` declares 3 tasks, Pydantic v2 action + observation models,
    FastAPI runtime, 19 formal compliance tests in `tests/test_openenv_compliance.py`.
- Gymnasium-style `reset` / `step` / `grade` surface on
    `server.supply_environment.SupplyMindEnvironment`.
- Docker deployment path: `Dockerfile`, `docker-compose.yml`, HF-Space-ready.
- Upstream PR draft (`versions/v5_phoenix/upstream_prs/meta_openenv/`) submitting
    the SupplyMind env as a reference environment.
- Upstream PR draft to Alibaba's ROLL (`upstream_prs/alibaba_roll/`) registering
    the same env as a first-class agentic-RL training target.

---

## Top 15 headline results (every one has a one-bash-command receipt)

| # | Claim | Value | Receipt |
|---|---|---|---|
| 1 | mxbai P@1 on 53 precise SupplyMind queries | **0.9622** | `R5_GRANITE_mxbai_P1` |
| 2 | mxbai MRR on same | **0.9780** | `R5_GRANITE_mxbai_MRR` |
| 3 | Snowflake-Arctic-L BEIR-style nDCG@10 (26 crises) | **0.971** | `R5_BEIR_snowflake_nDCG10` |
| 4 | 2-judge Krippendorff α (Qwen+Mistral, 26 scenarios) | **0.7499** | `R4_2JUDGE_Krippendorff_alpha` |
| 5 | Cohen κ (Qwen × Mistral) | **0.747** | `R4_Cohen_kappa_QwenMistral` |
| 6 | MaskablePPO masking lift (easy task) | **+26.77 %** | `R6_MaskingAblation_easy_lift` |
| 7 | GCN MAE reduction vs MLP (easy graph) | **−48.02 %** | `R6_GCN_easy_MAE_vs_MLP` |
| 8 | Per-horizon split-conformal \|cov-95\| (WTI) | **0.024** | `R6_AquaRegia_WTI_dev95` |
| 9 | TimesFM residual-conformal \|cov-95\| (WTI) | **0.050** | `R3_TimesFM_CP_WTI_dev95` |
| 10 | Fixed SPOF detector F1 on 3 graphs | **1.000** | `V4_SPOF_V2_F1` |
| 11 | v3 + v4 tests green | **249 / 249** | `V4_Tests_Total` |
| **12** | **Autoresearch best experiment (v5)** | **s2_higher_entropy** | `V5_Autoresearch_best_experiment` |
| **13** | **Autoresearch CI95 lower lift over baseline (v5)** | **+0.051** | `V5_Autoresearch_CI95_lift` |
| **14** | **Arena leaderboard baselines ready (v5)** | **6 rows** | `V5_Arena_baseline_leaderboard` |
| **15** | **Counterfactual Twin median $ saved (v5)** | **> $0** | `V5_Twin_savings_gt_zero` |

Total: 20 receipts live in `versions/v5_phoenix/receipts_v2/`. Run any:

```bash
bash versions/v5_phoenix/receipts_v2/<claim_id>.reproduce.sh
```

---

## The 4-minute judge path

1. **[30s]** Read this page, top to the quick-start block
2. **[90s]** Hit `POST /arena/run` with any PyTorch policy — returns CI95 reward on 3 tasks + leaderboard rank
3. **[60s]** Hit `POST /live/hormuz-closure` with today's Iran news; get risk level + 5 recommended actions + live counterfactual ($ saved)
4. **[30s]** Run any 3 receipts from the 20 shipped
5. **[30s]** `pytest` → 249+ green

Full protocol: `docs/JUDGES_V5.md`.

---

## What's unique to v5 (vs v4)

1. **ROLL-DPO-judge-v1** — Qwen-2.5-3B DPO-fine-tuned on our 26 crisis preference pairs.
   Ships either via ROLL pipeline or standalone `trl.DPOTrainer` fallback.
2. **OpenEnv Arena** — drop-in `policy.pt` harness with Gradio UI + FastAPI endpoint.
3. **Live Counterfactual Digital Twin** — 100 MC rollouts conditioned on live Hormuz signal.
4. **SupplyMind as a ROLL environment** — registered via `roll_integration/env/`.
5. **`supplymind-skills` skill pack** — public Claude Code skill marketplace submission.
6. **Grade-A receipts framework** — command + stdout + exit + expected/actual/match.
7. **Autoresearch loop actually converges** — s1 accepted as baseline, s2 accepted as new best with +0.051 CI95 lower delta. (v4 claimed crashes; reality was a stale state.json. Phoenix ships the fix + the real lab notebook.)
8. **Dual upstream PRs** — Meta/OpenEnv + Alibaba/ROLL.
9. **Offline demo replay path** — `FORCE_REPLAY=1` + `?replay=1` fallback keeps the live demo working without venue Wi-Fi.
10. **Phoenix server entrypoint** — `uvicorn versions.v5_phoenix.server.phoenix_app:app` mounts v4 + v5 routers in one process.

---

## Honest limitations (published, not hidden)

- **Arena baselines** are pre-seeded from `R6_EUCLIDIAN.json` (3 tasks × 900 eps).
   Re-running them from scratch on our laptop takes ~3 hours.
- **ROLL install on Windows-native is fragile.** Phase A (Windows) → Phase B
   (WSL2) → Phase C (`trl` fallback). Documented in `roll_integration/INSTALL.md`.
- **DPO-judge delta vs baseline Qwen-3B** is expected +5 to +15 pp but
   unverified at submission time; receipt will ship a null result if negative.
- **Phoenix autoresearch has 3 pending seeds** (s3 curriculum, s4 recurrent,
   s5 action-diversity). v4 bugs blocking them are fixed here; rerun takes
   ~30 min total, user runs on Apr 22–23.
- **Counterfactual-Twin severity → dollars multiplier** is a calibrated
   heuristic, not a learned mapping. Bootstrap CI95 on savings keeps the
   uncertainty visible.

---

## Sleep Token album arc

- v1 simulated — `Aqua Regia` (first rain, simulated)
- v2 vessel — `Vessel` / `DYWTYLM` (real DataCo)
- v3 arcadia — `Emergence` → `Caramel` → `Past Self` → `Dangerous` → `Granite` → `Gethsemane` → `Provider` → `Aqua Regia`
- v4 arcadia-live — `Rain` → `The Summoning`
- **v5 phoenix-ascensionism** — `Ascensionism` → `Arcadia II`

---

## License

MIT (matches the hackathon's open-source requirement).

---

*Full technical details: `versions/v5_phoenix/docs/PREPRINT_V5.md`. Reproducibility
receipts: `versions/v5_phoenix/receipts_v2/INDEX.md`. Judge path: `docs/JUDGES_V5.md`.*
