# FOUNDATION SOLIDIFICATION PLAN — SupplyMind

> Planner: Fable orchestrator, 2026-07-03. Execution: owner drives Opus agents (≤5–7/wave),
> Fable orchestrates/verifies/commits. NO new fakes, ever (CLAUDE.md §0). Every capability
> real, measured, reproducible. Free/open resources only (solo student author).
>
> **This file is the single execution checklist to take the EXISTING project from "honest +
> de-faked" to "world-class real-world product."** Nothing here is future-scope (Qualcomm parked).
> Each Work Package (WP) = one Opus agent task: scope · steps · acceptance · evidence · resource.

---

## 0. WHERE WE ARE (honest scorecard, 2026-07-03)

| Layer | State | Grade |
|---|---|---|
| Core engine (`server/engine`, `supply_environment`, graders, tasks) | Real, deterministic, seeded. Untouched-good. | A |
| Package (`supplymind/`) | Reorg done; `pip install -e .` works; 95 modules. | A− |
| Foundation P0 (paths, de-fake, CI, security, archive) | DONE. 184/184 tests, CI armed, ~37GB reclaimed, history clean. | A |
| R-wave rebuilds (R6,R7,R9,R10,R11,R12,R17,R20) | DONE + measured (real receipts). | A |
| LLM analyst layer | Ollama-only IP (v5 prompt). Port to OpenRouter NOT done. | C (blocked on key) |
| Benchmark / leaderboard | Fakes removed; real full run (checkpoints × seeds) NOT done. | C |
| War room live data | Backend real; live feeds not wired end-to-end w/ freshness. | C+ |
| Dashboards (master, war room) | De-faked, honest, vendored. Not yet "world-class product" polish. | B− |
| Notebooks | Pruned 13→7, de-faked; NOT executed with committed outputs. | C+ |
| Docs / claims | Lies struck; CLAIMS_LEDGER exists. Not regenerated from receipts. | C+ |
| Observability (W&B) | Fake curves deleted; real runs not wired. | C |
| Deploy (HF Space) | Old Space frozen; new lean deploy not built. | C |

**Test suite: 184/184. Commits on `p0/foundation`: 30. Working tree: clean.**

**Blocked-on-owner (hard):** fresh OpenRouter key (revoked → 401). Everything tagged `KEY` waits.

---

## 1. DEFINITION OF "SOLID FOUNDATION" (done criteria)

The foundation is solid when ALL of these are true and evidenced by a committed receipt:
1. Fresh clone → `pip install -e .` → `pytest` green, on a clean CI runner.
2. `docker build` < 2GB, container serves the full golden path (not just /health).
3. Every one of the ~42 endpoints returns real data or an honest `degraded` flag — none fake.
4. Every headline number in README/docs traces to a receipt with `match:true`, real command, git SHA.
5. The golden path (signal → analyst → decision → counterfactual → war room) runs end-to-end,
   live, in ≤4 minutes, every number clickable to its source.
6. Zero `np.random`/hardcoded-as-measured/`except:pass`-on-prod anywhere (adversarial re-audit clean).
7. Benchmark leaderboard regenerable end-to-end; RL-vs-scripted narrative set by that evidence.
8. War room shows live-data freshness stamps < 1h; panel verdicts trace to usage-log lines.

---

## 2. WORK PACKAGES — grouped into executable waves

Legend: **[READY]** do now · **[KEY]** needs OpenRouter key · **[GPU]** needs 4080 · **[LONG]** hours ·
**[OWNER]** needs a decision/asset. Each WP is sized for ONE Opus agent. Disjoint ownership within a wave.

### WAVE 6 — MAKE-REAL: CORE PRODUCT SPINE  (do first — this is the golden path)

**WP6.1 — LLM provider layer + analyst-v5 port** `[KEY]` (build code now, measure on key arrival)
- Scope: `supplymind/llm/providers.py` (openrouter|ollama switch, structured-JSON enforced, retry,
  `.openrouter_usage.jsonl` logging), `supplymind/llm/analyst.py` (port v5 system prompt +
  calibration rules + 8 few-shots from `ollama-modelfiles-backup/supplymind-analyst_v5.modelfile.txt`).
- Steps: single interface; Instructor or OpenRouter `require_parameters` JSON-schema output; model
  slug env-driven (never hardcoded); startup liveness check; degrade loud on failure.
- Acceptance: mock-transport unit tests pass in CI (no live calls); with a live key, `python -m
  supplymind.llm.analyst --holdout` writes real usage lines.
- Free resource: OpenRouter free-tier slugs for bulk (e.g. Llama-3.x / Qwen free), frontier only for judge.

**WP6.2 — Analyst A/B with Brier calibration** `[KEY]`
- Scope: real A/B — analyst-v5-prompt vs bare model on holdout scenarios; Brier score + reliability
  curve; receipt `tests/receipts/analyst_ab_brier_REAL.json` + calibration plot (dataviz skill).
- Acceptance: plot generated from that run's data only; both models' real per-scenario outputs stored.
- Reference: Guo 2017 "On Calibration of Modern Neural Networks" (ECE/Brier).

**WP6.3 — Real benchmark suite (RL-vs-everything)** `[GPU][LONG]`  ← decides the whole narrative
- Scope: evaluate ACTUAL `rl/checkpoints/` (scripted, random, PPO, QR-DQN, BC/CQL/IQL/TD3+BC, DT,
  ensemble) on easy/medium/hard × ≥20 seeds. Fix action-taxonomy regen first (buffers), then eval.
  Real paired-by-seed stats, correct Wilcoxon r. Regenerate `benchmark_summary.csv` + leaderboard.
- Acceptance: every leaderboard number → run artifact (command + seed list + SHA). Raw episode
  arrays committed. `benchmark/results/` regenerable on this machine.
- **Then update PRODUCT_THESIS RL positioning from the RESULT** (honest, whichever wins).
- Free resource: local compute only; no paid anything.

**WP6.4 — Live-signal layer + war-room end-to-end** `[READY]` (keys already in `.env`)
- Scope: `supplymind/data/live.py` — per-source fetch + freshness stamp + cache + loud fail for
  EIA, NASA FIRMS, GFW, FRED, NewsAPI, NOAA. Wire into war room UI with visible `< Nh old` stamps.
  Kill/complete the cosmetic stage pills (real SSE stage events). `FORCE_REPLAY` shows a REPLAY banner.
- Acceptance: war room loads with live timestamps < 1h; each panel value traces to a fetch log line.
- Free resource: all 6 keys are free-tier and already present. USGS/GDELT keyless as backups.

**WP6.5 — Golden-path integration test** `[READY]`
- Scope: one scripted E2E (`tests/test_golden_path.py`): reset → step → analyst grade → decision →
  counterfactual (R6) → war-room render, asserting real data at each hop; runs in CI against local server.
- Acceptance: green in CI; fails loud if any hop returns a degraded/fake marker.

### WAVE 7 — MAKE-REAL: SUPPORTING CREDIBILITY  (parallel-safe after Wave 6)

**WP7.1 — Notebooks execute + commit outputs** `[LONG]`
- Scope: `jupyter nbconvert --execute` on the 7 active notebooks (01,02,03,06,08,09,13); commit WITH
  outputs. Fixes already landed; this proves they RUN. Heavy cells: cap epochs, mark as smoke.
- Acceptance: every kept notebook has visible committed outputs on GitHub; nbconvert exit 0 each.

**WP7.2 — Real judge panel (12) + agreement** `[KEY]`
- Scope: verify current OpenRouter catalog live (`verify_openrouter_models.py`), pick 12 real slugs,
  run panel on 26 scenarios, real Krippendorff α on ≥3 judges; regenerate `openrouter_liveness.json`,
  clears 5 RERUN_QUEUE receipts. Reference: Zheng 2023 "Judging LLM-as-a-Judge."
- Acceptance: α computed from real transcripts; unparseable replies counted separately (not faked).

**WP7.3 — RL rebuilds: real ablation + Pareto + GAT** `[GPU][LONG]` (R3, R4, R5)
- Scope: R3 component ablation (±CVaR, ±real-buffer, ±uncertainty, ±ensemble — real deltas); R4
  reward-scalarized Pareto sweep (real frontier, carbon from real tonnage); R5 train `SupplyChainGAT`
  on MC-derived cascade labels, serve real attention.
- Acceptance: each ships its receipt with real measured deltas; retire honestly if a component doesn't help.

**WP7.4 — DPO judge, trained for real** `[GPU]` (R8)
- Scope: fix the torch/peft float8 skew (crash root-cause logged), QLoRA a Qwen2.5-1.5B judge on the
  21 real preference pairs; honest before/after judge eval. If it doesn't beat the prompt-only judge, say so.
- Free resource: local 4080; unsloth ungated mirror for base weights.

**WP7.5 — Observability: real W&B** `[OWNER: W&B]`
- Scope: wire real training runs (WP6.3/7.3/7.4) to W&B; scrub the old fabricated "live_dashboard_proof"
  runs from the project; `.openrouter_usage.jsonl` cost view.
- Acceptance: only real runs remain in the W&B project; a cost dashboard renders from real usage log.

### WAVE 8 — WORLD-CLASS POLISH

**WP8.1 — System dashboard rebuild** `[READY]` (R16, R21)
- Scope: master.html → honest system dashboard; every card a real endpoint or receipt link; LED
  probes that truly exercise endpoints; real charts from `benchmark/results/` (dataviz skill); real
  SHAP on the trained QR-DQN. Each rebuilt feature earns its card back (target: MORE cards, all real).
- Skills: dataviz, interface-design, interaction-design, web-design-guidelines review pass.

**WP8.2 — Docs regenerated from receipts** `[READY]`
- Scope: `scripts/verify_claims.py` (parses inventory tables → checks paths/endpoints/receipts exist,
  fails on any unverified claim); regenerate README + one MODEL_CARD + one DATASET_CARD +
  HONEST_LIMITATIONS from verified receipts only; CLAIMS_LEDGER → all rows VERIFIED or STRUCK.
- Acceptance: `verify_claims.py` green in CI; zero claim without a receipt.

**WP8.3 — Test expansion + adversarial re-audit** `[READY]`
- Scope: cover the new `supplymind/llm/` layer (mock transport); Playwright E2E of war room `[OWNER:
  Playwright MCP]`; final fresh-agent fake-hunt (zero-finding gate before any submission).
- Acceptance: coverage up; adversarial re-audit returns zero fakes on production paths.

**WP8.4 — Deploy: new lean HF Space** `[OWNER: HF token + Space]`
- Scope: new Space; Docker < 2GB; smoke = full golden-path journey not just /health; deploy log artifact.
- Acceptance: cloned Space serves the golden path; CI deploy green.

### WAVE 9 — REMAINING REBUILD + DECISIONS

- **R1/R2** real-world benchmark + leaderboard render (folds into WP6.3 output). `[LONG]`
- **R11** extend real conformal to multi-level once transitions data restored. `[DATA]`
- **R13** cross-env transfer honest experiment (may be ~0 — a finding). `[GPU]`
- **R18** SAC-Discrete: implement or RETIRE (owner call; QR-DQN already real). `[OWNER]`
- **R19** Damocles endpoints: fold /assess /forecast /rag /rl-act into main server. `[KEY for assess]`
- **R22** twin $135.5M receipt: re-run to completion, publish the real number. `[LONG]`
- Clear the rest of RERUN_QUEUE.md as blockers lift.

---

## 3. SUBSYSTEM COVERAGE MATRIX (nothing missed)

| Subsystem | Owning WP(s) | Solid-when |
|---|---|---|
| Core engine + graders + tasks | (protected; regression-guarded) WP6.5 | golden-path test green |
| `supplymind/llm` | WP6.1, 6.2, 7.2 | provider switch + analyst A/B + panel all real |
| `supplymind/warroom` (+ scenarios, sources, features) | WP6.4, 8.1 | live feeds + honest dashboard |
| `supplymind/phoenix/arena` | WP6.3, 7.3 | real leaderboard feeds it |
| `supplymind/phoenix/counterfactual_v2` + `_twin` | R6 done; R22 | twin receipt real |
| `supplymind/phoenix/forecast_v2` | R10 done | (maintain) |
| `supplymind/phoenix/wordle_env` + `action_v2` | WP7.1 (nb), R11 | conformal + notebook outputs |
| `supplymind/phoenix/realtime_v5` | WP6.4 | replay banner honest |
| `rl/` training + offline + distributional | WP6.3, 7.3, 7.4 | real benchmark + ablation + DPO |
| `benchmark/` | WP6.3, 8.2 | regenerable, claims verified |
| `server/` (42 endpoints) | WP6.4, 6.5, 8.1 | every endpoint real-or-degraded |
| Data layer (`external_data`, `supplymind/data`) | WP6.4 | live.py + provenance |
| `tests/` | WP6.5, 8.3 | golden-path + llm + E2E + re-audit |
| CI/CD | WP8.4 | green on push; deploy smoke = journey |
| Docs / FINAL_SUBMIT | WP8.2 | receipt-backed only |
| Observability | WP7.5 | real W&B + cost view |

---

## 4. OWNER-PROVIDES (free where possible)

| # | Need | Why | Free? |
|---|---|---|---|
| 1 | **Fresh OpenRouter API key** → `.env` | Unblocks WP6.1/6.2/7.2 + war-room live panel + 6 RERUN receipts | Yes (free-tier slugs exist; add small credit for frontier judge only) |
| 2 | `SEC_CONTACT_EMAIL=you@…` → `.env` | SEC EDGAR fair-access UA (fetchers) | Free |
| 3 | Key rotation (FRED/NewsAPI/W&B/HF/NOAA) | Chat-log exposure (SECURITY.md) — git history is clean | Free |
| 4 | **Playwright MCP** install (`claude mcp add playwright -- npx "@playwright/mcp@latest"`) | War-room E2E (WP8.3) | Free |
| 5 | New HF Space + write-scoped HF token | Lean deploy (WP8.4) | Free |
| 6 | W&B project confirm (scrub old fake runs) | Real observability (WP7.5) | Free tier |
| 7 | Decisions: persona (PRODUCT_THESIS), retire R18?, fold R19? | Steer priority | — |
| 8 | (Optional) more free datasets if wanted: UN Comtrade, IMF IFS, Lloyd's List open, Marine Traffic free tier | Deepen live layer | Free tiers |

**No paid datasets or models required.** Local models (Chronos/TimesFM/TabPFN/mxbai) are sanctioned
free edge assets. OpenRouter free-tier + your existing keyless/free APIs cover the rest.

---

## 5. SEQUENCING (dependency-ordered)

```
Wave 6 (spine) ── WP6.3 benchmark ──► sets PRODUCT_THESIS RL story
   │  WP6.1 llm ─► WP6.2 A/B ─► (KEY)          │
   │  WP6.4 live ─► WP6.5 golden-path test ◄───┘
   ▼
Wave 7 (support): 7.1 nb · 7.2 panel(KEY) · 7.3 ablation/pareto/gat · 7.4 dpo · 7.5 wandb
   ▼
Wave 8 (polish): 8.1 dashboard · 8.2 docs · 8.3 tests+reaudit · 8.4 deploy
   ▼
Wave 9 (remaining rebuild + decisions) → then Phase B (Qualcomm brief)
```

Critical path = WP6.3 (benchmark) because it decides the product narrative. Start it early (it's [LONG]).
KEY-blocked WPs (6.1/6.2/7.2) build code now, run measurement the moment the key lands.

## 6. EXECUTION PROTOCOL (per CLAUDE.md §5)
- ≤5–7 Opus agents/wave, disjoint file ownership, one WP each. Fable orchestrates: plan → launch →
  verify evidence against acceptance → run pytest myself → commit per-WP on `p0/foundation`.
- Every WP ships a receipt (real command, git SHA, `match:true`) or it is not done.
- After each wave: fresh reviewer agent re-audits the diff vs §0 (use `/code-review`).
- Merge `p0/foundation` → `main` only after WP6.5 golden-path test is green in CI.
