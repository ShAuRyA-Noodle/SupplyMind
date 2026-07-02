# versions/v5_phoenix — v5 Ascensionism Layer

> "Ashes don't forget, they remember forward."

This directory contains the **Phoenix v5 ascensionism layer** being built on top of:
- `versions/v3_arcadia/` (frozen at commit `02251e9` — the v3 ashes)
- `versions/v4_arcadia_live/` (frozen as v4.0-arcadia-live — the first phoenix)

**Phoenix v5 is isolated by directive.** If anything here fails, `versions/v3_arcadia/` and `versions/v4_arcadia_live/` remain a complete, self-sufficient top-10 hackathon submission. Every new capability lives here.

## What's in this folder

| Dir | Purpose |
|---|---|
| `roll_integration/` | Alibaba ROLL framework integration: DPO judge fine-tuning, SupplyMind-as-a-ROLL-env, LLMJudgeRewardWorker bridge, YAML configs, + `trl_fallback/` for standalone DPO if ROLL fails to install |
| `supplymind_skills/` | Publishable Claude Code skill pack (`benchmark-runner`, `autoresearch-experiment`, `live-demo-orchestrator`) for `obra/superpowers-marketplace` submission |
| `arena/` | OpenEnv Arena — drop-in-your-policy harness (Gradio + FastAPI). Judges upload `policy.pt`, get CI95 reward + violations on 3 tasks |
| `counterfactual_twin/` | Live Counterfactual Digital Twin — 100 MC rollouts of MaskablePPO vs no-action vs greedy, conditioned on live Hormuz signal |
| `autoresearch_fixed/` | Fixed copy of v4's autoresearch loop (v4 crashed all 5 seeds in ~5s; root cause patched here) |
| `receipts_v2/` | Grade-A reproducibility receipts: `command` + full `stdout` + `exit_code` + `expected` + `actual` + `match` (upgrade of v4's 13 receipts) |
| `server/` | `phoenix_app.py` — legacy standalone launcher. **The deployed entrypoint is the repo-root `server/app.py`** (`uvicorn server.app:app`), which already mounts the v5 `/arena`, `/twin`, and replay routers. Use `server.app:app`, not this file. |
| `upstream_prs/meta_openenv/` | Draft PR to `github.com/meta-pytorch/openenv` — SupplyMind as a reference env |
| `upstream_prs/alibaba_roll/` | Draft PR to `github.com/alibaba/ROLL` — `examples/supplymind_crisis/` reference agentic environment |
| `experiments/` | ROLL training runs, checkpoints, lab notebook outputs |
| `docs/` | `PREPRINT_V5.md`, `PITCH_DECK_V5.md`, `JUDGES_V5.md`, `DEMO_VIDEO_SCRIPT_V5.md`, `PHOENIX_COMPLETION_AUDIT.md` |
| `tests/` | Unit + integration tests for every new module |
| `scripts/` | Convenience runners (install ROLL, launch arena, build receipts, etc.) |

## Design invariants

- **v3 and v4 are untouched.** `tests/` and `versions/v4_arcadia_live/tests/` (261 total as of 2026-07-02: 184 core + 77 v4) must stay green throughout Phoenix work.
- **Copy-before-edit.** Any existing v4 file being modified is copied into Phoenix first; edits happen on the copy.
- **Isolated Python env.** ROLL has a massive dependency graph (Megatron, DeepSpeed, vLLM, Ray, flash-attn). Its venv lives at `.venv-roll/` inside this folder and never touches the main venv.
- **Fail gracefully.** Every Phoenix endpoint, feature, and demo path has an offline fallback. `--replay` flags, cached outputs, `trl.DPOTrainer` fallback for DPO, `transformers` fallback for vLLM.
- **Reproducibility is non-negotiable.** Every headline claim has a receipt in `receipts_v2/` with the correct current-layout command. As of 2026-07-02, 10 of the 20 receipts run green on this machine; the other 10 are honestly marked `not_yet_run` in each receipt's `requires` field (they need local GGUF judges, an absent embedder, GPU-hours, or a live FRED key — see `receipts_v2/INDEX.md`). No receipt is a `<pending-first-run>` stub any more.

## Track: "Ascensionism"

> *"I do not want to feel this way / But I cannot look away"*

Phoenix v5 is the ascensionism phase of the v3 → v4 → v5 arc. Final tag will be `v5.0-phoenix-ascensionism` when all components are green.

## Phase gates

- **Phase 0** complete when: autoresearch converges with ≥1 accepted experiment, Hormuz replay cache frozen, ROLL install smoke-test decision made (Phase A green OR Phase B green OR `trl` fallback chosen).
- **Phase 1** complete when: ROLL-DPO-judge-v1 produces a measurable delta vs baseline, OpenEnv Arena serves `POST /arena/run` successfully, Counterfactual Twin returns a live-signal-conditioned distribution.
- **Phase 2** complete when: HF Space green, demo video uploaded, skill pack submitted to marketplace, grade-A receipts cover all 13 original + 5 new headline claims.

## For judges (once we're at finals)

See `docs/JUDGES_V5.md` for the 4-minute path. TL;DR:

```bash
# Clone (public repo, no auth)
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# v3 core + v4 tests (frozen baseline, 261 passing)
pytest tests/ versions/v4_arcadia_live/tests/ -q

# Phoenix tests (new)
pytest versions/v5_phoenix/tests/ -q

# Live server (Arena + Counterfactual Twin + v4 Hormuz all mounted here)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Any headline receipt
bash versions/v5_phoenix/receipts_v2/<claim>.reproduce.sh
```

---

*Phoenix v5 plan: `versions/v4_arcadia_live/docs/PHOENIX_PLAN_V5.md` (Sections 10–16 cover ROLL + Superpowers deep integration).*
