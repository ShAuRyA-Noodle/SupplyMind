# SupplyMind (Sleep-Token) — Master Plan & Engineering Constitution

> **Read this first, every agent, every session.** This file is the single source of truth for
> transforming SupplyMind into a world-class, fully-real, end-to-end-verified system for the
> upcoming Qualcomm hackathon. It was produced by a Fable-5 orchestrator from a 12-agent brutal
> audit (9 completed, ~1.67M tokens of evidence, 2026-07-02) plus direct verification.
> Implementation agents execute the workstreams below; they do not re-litigate the rules.

---

## 0. NON-NEGOTIABLE RULES (the Constitution)

1. **ZERO fake anything.** No mock data, no hardcoded numbers pretending to be measured, no
   `np.random`/`Math.random` dressed as model output, no canned LLM responses, no placeholder
   receipts, no `alert()` cards pretending to be endpoints, no dead UI. If a feature cannot be
   real yet, it is **removed or explicitly labeled experimental** — never faked.
   **1b. REBUILD-REAL DOCTRINE (owner directive 2026-07-02): removal is triage, not the endpoint.**
   Every capability that was deleted/stubbed/relabeled because it was fake MUST get an entry in
   `REBUILD_BACKLOG.md` and, where feasible, be REBUILT as the genuine world-class version.
   The goal is not a smaller honest project — it is a complete phenomenal one. Deleted code is
   recoverable from git history; ambition is mandatory. Limitations get engineered into
   strengths, not merely documented.
2. **Evidence-gated "done".** No task is complete until its acceptance criterion is executed and
   the output is captured (test run, curl output, screenshot, committed receipt with real
   exit_code 0). Claims without artifacts are lies.
3. **Fail loud.** No `except: pass` on production paths. Degraded fallbacks must be surfaced in
   the API response (`"degraded": true`, reason) and logged. A silent fallback is a fake.
4. **One AI gateway.** All LLM calls go through `scripts/openrouter_client.py` (to be promoted to
   `supplymind/llm/`). OpenRouter is the primary backbone (key already in `.env`). Ollama is an
   **optional, explicitly-labeled local/edge mode** behind the same interface — never a hidden
   hard dependency. Non-LLM local models (Chronos, TimesFM, TabPFN, mxbai embeddings) have **no
   OpenRouter equivalent**; they are a *sanctioned local exception* (see §7.3) until replaced.
5. **Reproducibility.** Every headline number must have a working one-command reproduction from a
   fresh clone. A receipt with `match: false`, `exit_code: -1`, or a stale module path is worse
   than no receipt.
6. **No secrets in code or git.** Keys live in `.env` only. See §12 Security — 5 keys were
   flagged compromised in `versions/v5_phoenix/docs/PHOENIX_PUSH_REPORT.md` §3.6 and an HF token
   was once committed to `scripts/push_to_hf_space.py` ("# SCRUBBED" marker). Rotate before the
   Qualcomm event.
7. **Statistics must be valid.** No sorted-"paired" Wilcoxon tests, no circular training targets,
   no label leakage, no p-values from synthesized samples. If a statistician would laugh, delete it.
8. **Commit hygiene.** Work on feature branches (`p0/...`, `p1/...`), small commits, run the test
   suite before every merge to main. Never commit binaries >10MB.

---

## 1. WHAT THIS PROJECT IS

**SupplyMind** — an open-source supply-chain-risk intelligence system:

- **OpenEnv-compliant simulation environment** (`server/supply_environment.py` + `server/engine/*`):
  deterministic, seeded supply-chain graph simulation (nodes, disruptions, finances, rewards,
  Monte Carlo), FastAPI server (`server/app.py`) with reset/step/state/grade endpoints,
  task registry (easy/medium/hard scenarios). **This core is real and good.**
- **RL decision layer** (`rl/`): Gymnasium wrapper (408-dim obs, MultiDiscrete[7,40], action
  masking), MaskablePPO / QR-DQN(CVaR) / offline RL (BC/CQL/IQL/TD3+BC/Decision Transformer),
  AutoResearch HPO loop, ONNX export. **Training genuinely ran** (100+ checkpoints, varied grades
  0.24–0.85 in `rl/autoresearch_results.json`).
- **LLM analyst layer**: `supplymind-analyst` v2→v5 (Modelfiles in `ollama-modelfiles-backup/`) —
  base Qwen2.5 + domain-knowledge system prompt (real 2011–2026 events), calibration rules
  (LOW/MEDIUM/HIGH/CRITICAL gating, confidence caps), strict-JSON output, 8 few-shot exemplars.
  **The IP is the prompt+calibration, not weights — fully portable to OpenRouter.**
- **Live war room** (`server/static/hormuz_war_room.html` + `versions/v4_arcadia_live/realtime/`):
  Hormuz-strait geopolitical dashboard, genuinely backend-driven (zero Math.random), fed by a
  frontier judge panel over OpenRouter + live APIs.
- **Real data corpus** (`external_data/`, `rl/data/`): genuine SEC EDGAR 10-Ks (75MB), EM-DAT
  16,811-row export, DataCo 180K orders (96MB), NOAA IBTRACS (111MB), FRED, World Bank, Wikipedia.
  **Data verified real** — the plumbing around it is broken (§6.6).
- **Live API keys already in `.env`**: OPENROUTER_API_KEY, EIA, NASA_FIRMS, GFW (ship tracking),
  FRED, NEWS_API, NOAA, WANDB.

History: built for OpenEnv India 2026 ("Facebook hackathon" — did not win). Versions v3_arcadia →
v4_arcadia_live → v5_phoenix → v6.0-genesis badge. Next target: **Qualcomm hackathon (brief TBD —
owner will supply; expect on-device/edge AI to matter, which makes the ONNX export path and the
Ollama edge mode strategic assets, not legacy)**.

---

## 2. BRUTAL AUDIT VERDICT (2026-07-02)

**Overall: a genuinely strong engine wearing a cardboard costume. Average subsystem score 4.1/10.
132 fake/mock findings, 69 dead-code items, 83 broken items.** The pattern is identical
everywhere: real core + fabricated presentation layer ("receipt factory") + reorg bitrot from the
phase-4 `versions/` rename that nobody caught because CI is disabled.

| Subsystem | Score | One-line verdict | Salvage |
|---|---|---|---|
| server/ + UI | 4/10 | Real deterministic engine; 6 endpoints 503 on stale `v3_arcadia` paths; /predict returns canned fallbacks; master.html has ~25 alert() fake cards under a "no mocks" banner | Refactor |
| rl/ | 5/10 | Training core is real & strong; wrapped in fabricated benchmarks, fake leaderboard, untrained-GNN "attention", action-taxonomy mismatch between buffers and env | Refactor |
| root pipeline | 4.5/10 | scripted_agent verified working E2E (0.77/0.70/0.67); Damocles container can't start; 43GB docker build context; placebo DEMO_MODE vars | Refactor |
| benchmark/ + tests/ | 4/10 | tests/ = best engineering in repo (176 pass, CPU, 140s — "needs Ollama+GPU" excuse is FALSE); benchmark/ = scripted-vs-scripted "ablations" and hardcoded figures | tests: keep; benchmark: rebuild |
| notebooks/ | 3/10 | nb13 flagship contains rng.normal "causal counterfactuals", rigged 269/269 gauntlet, a SyntaxError proving it never ran; ZERO notebooks have outputs | Surgical clean |
| scripts/ | 3/10 | ~14 genuinely real tools drowning in a receipt factory: sorted-"paired" Wilcoxon p=1e-18, noise-as-conformal, fake W&B curves, 6 judge-verification scripts path-broken | Aggressive refactor |
| external_data/ | 5/10 | Data is REAL (verified); fetch scripts untracked/gitignored, platinum.py "real FRED" counterfactual always hits hardcoded fallback (CSV column mismatch) | Keep data, rebuild plumbing |
| models/ + OpenRouter | 4.5/10 | openrouter_client.py = best file in repo (real, rate-limited, 753 usage-log lines); 9.68GB local weights (~6GB pure waste); 3 ghost model dirs referenced but absent; JUDGES_12 slugs never tested | Refactor |
| versions/+vendor/+_dump/ | 3.5/10 | NOT legacy — server imports it at runtime (102 refs / 27 files); 14/20 "grade-A receipts" are unexecuted stubs; PREPRINT claims a DPO judge that never trained; 31GB local bloat | Promote live parts, archive rest |
| CI/deploy (gap-fill) | 4/10 | Only deploy-hf-space.yml active; test CI + benchmark guard disabled (excuse false); deployed Space excludes so much it degrades 6+ endpoints by design | Rebuild CI |
| docs/ + FINAL_SUBMIT (gap-fill) | 3/10 | 54 judge-facing docs incl. "ALL_250_FEATURES_LIVE_PROOF.md", "VICTORY_CALCULUS.md"; audits proved multiple "PRESENT/verified" claims false (e.g. tabpfn-v2-clf); claims layer must be regenerated from verified receipts only | Rebuild from evidence |

**Genuine strengths to protect during all work** (do not break these):
`server/supply_environment.py`, `server/engine/*`, `server/graders/*`, `server/tasks/*`,
`static/hormuz_war_room.html`, `rl/` training core (gym_env, train_ppo, distributional/, offline
v2, autoresearch, export_onnx), `tests/` suite + sha256 receipts, `scripts/openrouter_client.py`,
crisis-library cook pipeline (`scripts/crisis_library/cook_v2.py` → `crisis_library_v2.json`),
the real datasets, the `supplymind-analyst` prompt lineage, `scripts/final_real_reinforce_wordle_v2.py`,
live-API ingest blocks (pass28 K1–K3), nb06 DPO on real preference pairs.

---

## 3. TARGET ARCHITECTURE

### 3.1 Final package layout (end-state after P0.3)

```
Sleep-Token/
├── supplymind/                  # single installable package (pip install -e .)
│   ├── core/                    # from server/: supply_environment, engine/, graders/, tasks/
│   ├── server/                  # FastAPI app (thin; routers only)
│   ├── llm/                     # openrouter_client (promoted), analyst.py (v5 prompt port),
│   │                            # panel.py (judge panel), providers.py (openrouter|ollama switch)
│   ├── warroom/                 # from versions/v4_arcadia_live/realtime + scenarios (live code)
│   ├── phoenix/                 # from versions/v5_phoenix live modules: arena, twin,
│   │                            # counterfactual_v2, wordle_env, action_v2, realtime_v5
│   ├── rl/                      # existing rl/ package (cleaned)
│   ├── data/                    # fetchers (tracked!), loaders, provenance manifests
│   └── contracts.py             # from root models.py (renamed — kills models.py vs models/ collision)
├── apps/dashboard/              # ONE dashboard (war room); Streamlit dashboard/ deleted
├── benchmark/                   # rebuilt honest harness
├── tests/                       # kept, expanded
├── notebooks/                   # pruned to 5-6, all committed WITH outputs
├── external_data/               # data + tracked fetch scripts + PROVENANCE.md
├── archive/                     # read-only: v3 result JSONs, judge caches, FAILURE_TABLE lineage
├── docs/                        # rewritten honest docs
└── (Dockerfile, compose, Makefile, pyproject — all rebuilt)
```

### 3.2 Reorg ground rules
- **Two-step, never big-bang**: P0.1 unbreak in place → P0.3 move into package. Tests green after
  every step.
- Use `git mv`; one commit per coherent move; update `.github/workflows/deploy-hf-space.yml`
  paths + Dockerfiles + `pyproject.toml` in the SAME commit as the move they affect.
- The audit `couplings` data (see `_audit/` scratchpad exports, or re-derive by grep) lists every
  cross-boundary import. Verify each with grep before and after.
- `versions/` may not be deleted until §P0.3 promotion completes — the server imports it at boot.

---

## 4. PHASE A — SOLIDIFY (P0 → P2). Everything here precedes any new feature work.

### P0 — FOUNDATION (unbreak + de-fake + de-bloat). Order matters.

**P0.1 — Fix reorg bitrot in place (no moves yet).** The phase-4 rename (`v3_arcadia` →
`versions/v3_arcadia`) broke ~20 call sites. Fix `ROOT/'v3_arcadia'` → `ROOT/'versions'/'v3_arcadia'`
(or a shared `find_repo_root()` helper) in:
- `server/app.py:789,880` (+6 more sites ~842–1350) — revives 6 dead /analyst endpoints and the
  8 failing adversarial tests
- `server/integrated_agent.py` (4 path constants)
- `scripts/run_all.py:20`, `scripts/check_benchmarks.py:17`, `scripts/bootstrap_leaderboard.py:43`,
  `scripts/compute_panel_agreement.py:21`, `scripts/run_frontier_judge_panel.py:43`,
  `scripts/export_all_onnx.py:36`
- `versions/v3_arcadia/**` ROOT parent-count off-by-one (r4_*.py, r5_*.py, 90_damocles/app.py,
  train_v3_block*.py) — external_data/models resolve to nonexistent `versions/external_data`
- `rl/analysis/trained_models.py:43` + `rl/data/build_unified_buffer_v2.py:47` — WGI xlsx now at
  `_dump/data_legacy/` (restore file to `external_data/wgi/` and point both there)
- `dashboard/app.py:412` imports renamed `_heuristic_explanation` — fix import or function name
- **Acceptance:** `pytest tests/` → 184/184 pass (8 currently failing revive); `python scripts/run_all.py`
  finds all 14 files; server boots with zero 503 endpoints.

**P0.2 — Execute the Fake-Elimination Ledger (§6).** Every item DELETE/REWRITE/RELABEL as listed.
This is the owner's #1 demand and the single biggest disqualification risk.
- **Acceptance:** `grep`-sweep for each ledger pattern returns clean; a fresh adversarial reviewer
  agent re-audits each subsystem and finds zero `fake_ai_output|mock_data|hardcoded-as-measured`
  on production paths.

**P0.3 — Reorg into `supplymind/` package (§3.1).** Promote live modules out of `versions/`
(v4 realtime+scenarios, v5 arena/twin/counterfactual_v2/wordle_env/action_v2/realtime_v5), root
`models.py` → `supplymind/contracts.py`, `scripts/openrouter_client.py` → `supplymind/llm/`.
Update all 102 versions/-references across 27 live files + Dockerfiles + compose + CI + pyproject
(`py-modules` fix so `pip install -e .` works).
- **Acceptance:** fresh venv, `pip install -e .`, `pytest` green, server boots, `docker build`
  context < 500MB (fix `.dockerignore`: exclude models/, versions/, archive/, external_data/,
  _dump/, rl/checkpoints/).

**P0.4 — Archive & de-bloat (~37GB reclaim).**
- Move OUT of repo to `C:\Users\Dell\Desktop\Sleep-Token-ARCHIVE\`: `versions/v3_arcadia/gguf_out`
  (28.7GB), `versions/v3_arcadia/tools/llama.cpp` (263MB), `versions/v5_phoenix/.venv-roll` (869MB),
  `vendor/ROLL` (141MB), `versions/v5_phoenix/experiments/rap_xc_v1/*.npz` (1.19GB, keep a pointer),
  `_dump/` (after git rm of its 210 tracked files)
- Delete inside `models/`: mxbai `{gguf,onnx,openvino}/` (~3.35GB, zero references),
  `timesfm-2/model.safetensors` (2GB duplicate), 5 unused tabpfn ckpt variants (~180MB), `.DS_Store`,
  `.cache/huggingface/` leftovers
- Keep in-repo: `archive/results/` = v3 result JSONs + judge caches + FAILURE_TABLE.md (evidence,
  read-only)
- Delete: committed `__pycache__` dirs (server/, tests/, scripts/), quadruplicate simulated buffers
  (keep one), `rl/analysis/trained/v3/*.pkl` (~230MB, zero loaders)
- **Acceptance:** repo working tree < 2GB (excluding external_data + models kept per §7.3);
  `git status` clean; nothing live broke (`pytest` green).

**P0.5 — Re-enable CI.** The "needs Ollama+GPU" excuse is false — the suite is CPU-only/offline/140s.
- Restore `.github/workflows-disabled/ci.yml.disabled` → active `ci.yml`: pytest on ubuntu-latest
  (fix its stale `v3_arcadia.utils` import), remove `continue-on-error` masks
- `check_benchmarks.py` must **fail closed** (missing file = FAILURE, not [SKIP]+exit 0)
- Keep deploy-hf-space.yml but retarget later to the NEW Space (owner decision, §12)
- **Acceptance:** green CI run on GitHub Actions for a trivial PR; a deliberately broken test turns
  it red.

### P1 — MAKE EVERYTHING REAL

**P1.1 — LLM migration: Ollama → OpenRouter-first (§7).** ~46 files / 332 Ollama refs.
Priority order: `rl/explainer.py` (mandatory-Ollama today), `versions/.../hormuz_endpoint.py`
3-judge panel, `autoresearch/hypothesis_engine.py + orchestrator.py`, `rl/rag/indexer.py`
(embeddings — see §7.3), `wordle_env/dual_verifier.py`, `qwen_vl_port_imagery.py` (vision →
OpenRouter vision models).
- **Acceptance:** full user journey (reset→step→analyst decision→war room render) with Ollama
  daemon STOPPED; then same journey with `PROVIDER=ollama` and daemon running (edge mode).

**P1.2 — Port supplymind-analyst v5 to OpenRouter.** New `supplymind/llm/analyst.py`: v5 system
prompt + calibration rules + 8 few-shots (from `ollama-modelfiles-backup/supplymind-analyst_v5.modelfile.txt`),
structured-output enforced (Instructor or OpenRouter `require_parameters` JSON schema), model
configurable (default a strong Qwen or frontier model). Then run a REAL A/B: analyst-v5-prompt vs
bare model on the holdout scenarios, Brier score + calibration curve, committed receipt.
- **Acceptance:** `python -m supplymind.llm.analyst --holdout` produces receipt with real API
  usage lines in `.openrouter_usage.jsonl`; calibration plot generated from that run's data only.

**P1.3 — Real benchmark suite (replaces the theater).** Delete scripted-placeholder fallbacks;
evaluate the ACTUAL trained checkpoints (they exist in `rl/checkpoints/`): scripted, random, PPO,
QR-DQN, BC/CQL/IQL/TD3+BC, DT, ensemble on easy/medium/hard × ≥20 seeds. Valid stats only
(true paired tests on same-seed episode pairs; correct Wilcoxon effect size). Regenerate
`benchmark_summary.csv` + leaderboard from this run. Fix the buffer↔env **action-taxonomy mismatch**
first (`build_unified_buffer*.py` labels vs `rl/gym_env.py` ACTION_TYPES — indices 1/3/4/5/6
disagree, "cancel" doesn't exist) and regenerate `real_unified*.npz`, else offline agents are
evaluated on scrambled action semantics.
- **Acceptance:** every number in the leaderboard traceable to a run artifact with command,
  seed list, and git SHA; `benchmark/results/` regenerable end-to-end on this machine.

**P1.4 — Receipts: execute or delete.** The 14 stub receipts in `versions/v5_phoenix/receipts_v2/`
(match:false / exit -1 / unimportable commands): rewrite commands against the new package layout,
execute each; any that can't be made real gets deleted AND its claim struck from every doc
(README table, PREPRINT_V5 abstract — the "DPO-fine-tuned judge" claim is false today; twin
$135.5M receipt shows exit -9).
- **Acceptance:** every receipt in the repo has `match: true` + exit 0, or does not exist.

**P1.5 — Notebooks: prune to ~6, execute, commit WITH outputs.** Keep 01–03 (fixed bootstrap),
06 (DPO), 08 or 09 (GRPO), 13 (surgically cleaned per §6.4). Archive the rest. Fix nb13 §7
SyntaxError + SMGym obs-unpack bug; fix nb10 dead repo URL or archive it.
- **Acceptance:** `jupyter nbconvert --execute` succeeds on every kept notebook; outputs visible
  on GitHub.

**P1.6 — Data layer hardening.** Track the fetchers (`external_data/fetch_all.py` is currently
gitignored WITH its directory — un-ignore scripts, keep data ignored), write `PROVENANCE.md`
(source URL + date + checksum for EM-DAT xlsx, policy PDFs, world_bank_macro), fix `platinum.py`
FRED parsing (real column names `observation_date,PCU484121484121` — or better: fetch real daily
Brent `DCOILBRENTEU`), delete the fake `frbny_supply_chain.pdf` (it's an HTML error page), retry
the 6 failed Wikipedia crisis articles, fix `r5_rag_beast.py` World-Bank list-vs-dict bug, wire
`benchmark/crisis_library/*.json` into backtesting or delete them.
- **Acceptance:** fresh clone + `python -m supplymind.data.fetch_all` rebuilds the corpus (minus
  registered-download EM-DAT, documented); zero dead datasets listed in DATASET_CARD.md.

**P1.7 — War room to fully live.** Wire the real feeds already keyed in `.env` (EIA, NASA FIRMS,
GFW, FRED, NewsAPI, NOAA) through a `supplymind/data/live.py` with per-source freshness stamps
shown in the UI. Kill the cosmetic 6-stage progress pills (make them reflect actual pipeline
stages via SSE) or remove them. `FORCE_REPLAY` mode stays but must render a visible "REPLAY"
banner — never silently canned. Verify or delete the JUDGES_12 extension slugs (zero cache/usage
evidence today); regenerate `openrouter_liveness.json` (current one shows 4/14 OK); prune the
expired `ling-2.6-1t` slug.
- **Acceptance:** war room loads with live data timestamps < 1h old; panel verdicts traceable to
  `.openrouter_usage.jsonl` lines from that request.

### P2 — WORLD-CLASS POLISH

- **P2.1 Dashboard rebuild**: master.html replaced by an honest system dashboard — every card a
  real endpoint or receipt link, LED probes that actually exercise endpoints (no 405=green).
  Use the installed Claude Code skills: `dataviz` (all charts), `interface-design`,
  `interaction-design`, `web-design-guidelines` review pass. Vendor Tailwind/fonts locally
  (demo must survive no-WiFi).
- **P2.2 Docs rewrite**: README (43KB → honest, current, working links), one MODEL_CARD, one
  DATASET_CARD (fixed provenance), HONEST_LIMITATIONS kept and expanded. FINAL_SUBMIT/ regenerated
  for Qualcomm from verified receipts only — every "PRESENT/verified" claim machine-checked by a
  script (`scripts/verify_claims.py`: parses inventory tables, checks paths/endpoints exist).
- **P2.3 Test expansion**: fix the 2 theater assertions (`test_openenv_compliance.py:190`
  `assert ... or True`, `:142` 200-or-422), cover the new llm/ layer (mock transport, no live
  calls in CI), E2E Playwright test of war room (needs Playwright MCP — §11).
- **P2.4 Observability**: real W&B runs only (DELETE fabricated-curve scripts pass28_K4*;
  scrub/overwrite the fake "live_dashboard_proof" runs in the W&B project), request logging,
  `.openrouter_usage.jsonl` cost dashboard.
- **P2.5 Deploy**: NEW HF Space (owner creates, §12), Docker image < 2GB, smoke test = full
  reset→step→grade journey not just /health.

## PHASE B — FUTURE SCOPE (blocked on Qualcomm brief)

Do NOT start. Held until owner supplies the hackathon brief. Prepared angles (from current assets):
ONNX on-device policy inference (`supplymind_policy.onnx` + 4 offline-agent ONNX — Qualcomm AI Hub
/ Snapdragon NPU story), Ollama edge mode as offline-first resilience narrative, small-model
distillation of analyst-v5 (real training this time), live-data war room as the demo centerpiece.
Design doc to be written jointly with the brief.

---

## 5. AGENT EXECUTION PROTOCOL (for the owner's implementation agents)

- **Orchestrator**: Fable (this session's role). **Implementation agents: Opus 4.8** (owner's
  directive) — one workstream item per agent, never two agents writing the same subsystem
  concurrently.
- Every agent MUST: (1) read this file first; (2) read the relevant audit JSON in the scratchpad
  `audits/` export if available (else re-grep); (3) work on a branch; (4) run
  `pytest tests/` before declaring done; (5) paste actual command output as evidence; (6) obey §0.
- **Verification gate**: after each P-item, a SEPARATE reviewer agent (fresh context) re-audits
  the diff against §0 rules and the item's acceptance criterion. Use `/code-review` skill.
- Suggested wave plan: Wave 1 = P0.1+P0.2 (parallel-safe: paths vs fake-deletion touch different
  lines — coordinate on server/app.py). Wave 2 = P0.3 (single agent, big context). Wave 3 =
  P0.4+P0.5. Wave 4 = P1.1+P1.2 → P1.3–P1.7 parallel. Wave 5 = P2.
- Session-limit note: subagent fleets hit the Claude session cap on 2026-07-02 ~03:00 IST
  (resets 08:00 IST). Batch big fleets after reset.

---

## 6. FAKE-ELIMINATION LEDGER (P0.2 — exhaustive, from audit evidence)

Verbs: **DELETE** (remove file/feature), **REWRITE** (make real), **RELABEL** (keep but honest).

### 6.1 server/ + dashboards
| Target | Verb | Detail |
|---|---|---|
| `dashboard/app.py:283` np.random "QR-DQN Quantiles" violin | DELETE | with the whole deprecated Streamlit dashboard (keep `scenario_builder.py` logic if wanted) |
| `dashboard/app.py:242,260,303,353,384,414` hardcoded bars/radar/ablation/SHAP/counterfactual/MockObs | DELETE | same |
| `server/app.py:697` /predict bare-except → canned action 0/conf 0.5 | REWRITE | surface `"degraded": true` + reason; log |
| `server/app.py:711` counterfactual="Train surrogate model..." placeholder | DELETE | field or real value |
| `server/app.py:1546` /v3/e2e hardcoded $123.28 + severity-shift "forecast" | RELABEL→REWRITE | call real ensemble or label "heuristic anchor+shift" |
| `server/app.py:1482` keyword-lookup "3-judge panel" + static α=0.750/κ=0.747 in response | REWRITE | real panel via OpenRouter or drop the stage |
| `server/app.py:1371` SSE fake "judges arriving" (asyncio.sleep over static file) | REWRITE | stream real panel completions or remove animation |
| `server/integrated_agent.py:294` RL policy on all-zeros obs | REWRITE | build real obs→408-dim projector |
| `server/integrated_agent.py:324` hardcoded forecast anchor 123.28 | REWRITE | live FRED fetch w/ cache |
| `server/integrated_agent.py:21` "3-layer GCN" docstring for degree-counting | RELABEL | honest name or train real GCN |
| `server/static/master.html:279` ~25 alert() cards | REWRITE | real links or delete cards |
| `server/static/master.html:716,105` "No mocks" footer + static hero stats "9 cards all live" | DELETE/REWRITE | stats fetched live |
| `server/static/master.html:757` 405⇒green LED probes | REWRITE | probe with real requests |
| `server/static/hormuz_war_room.html:1258` cosmetic 6-stage pills | REWRITE | real stage events or remove |
| `server/openenv_mcp_wrapper.py` step() 4-tuple crash + 3 tools calling nonexistent method + hardcoded compliant:True | REWRITE | it's cited as compliance evidence |
| `client/supplymind_client.py` wrapped payload → 422; root `client.py` duplicate | REWRITE / DELETE dup |

### 6.2 rl/
| Target | Verb | Detail |
|---|---|---|
| `rl/real_world_benchmark.py:157,164` hardcoded agent scores + "$50/delay" fantasy; 0.881 contradicts real 0.527 | DELETE or full REWRITE against real checkpoints |
| `rl/leaderboard.py:57,88` fabricated fallback table + "p<0.01" string | REWRITE | serve only real `benchmark_summary.csv` (P1.3), fail loud if absent |
| `rl/pareto/frontier.py:168` "20 policies" = same scripted agent ×20 + invented carbon constants | REWRITE or RELABEL "seed study" |
| `rl/gnn/attention.py:198` untrained random attention to dashboard | DELETE feed or train GAT for real |
| `rl/forecasting/tft.py:217` predict stub returns metadata; ckpt name mismatch | REWRITE |
| `rl/dataco_integration.py:206` "backtest" never calls agent_fn | REWRITE |
| `rl/analysis/trained_models.py:76` circular target + label leakage | REWRITE with valid targets |
| `rl/offline/dataset.py:119` silent hardcoded FRED fallback | RELABEL loud (`"source":"fallback"` surfaced) |
| `rl/constrained_ppo.py:11` "mathematically guaranteed" overclaim; `rl/federated/fedavg.py:7` "23%" pre-written result | DELETE claims |
| buffers action-taxonomy mismatch (`build_unified_buffer*.py` vs `gym_env.ACTION_TYPES`) | REWRITE + regenerate + retrain (P1.3) |
| `rl/gym_env.py:88` except:pass hiding 15-20× MC slowdown | REWRITE hard assert |
| analysis/ vs legacy/fallbacks byte-identical duplication; dead `_heuristic_explanation_REMOVED` | DELETE dups |

### 6.3 benchmark/ + tests/
| Target | Verb |
|---|---|
| `run_benchmark.py:88` + `ablation.py:76` scripted-placeholder for 8 "trained" agents | DELETE fallback (P1.3) |
| `benchmark/results/ablation_results.csv` (7 rows byte-identical) | DELETE, regenerate real |
| `visualize.py:53,97,132` three hardcoded "publication-quality" figures | DELETE generators, rewrite to read real files & fail loud |
| `statistics.py:71` wrong Wilcoxon effect-size formula; crashes on missing CSV | REWRITE |
| `backtesting.py` key mismatch (duration_steps vs days), is_credible=false 3/3 committed | REWRITE, wire crisis_library/*.json |
| `tests/test_openenv_compliance.py:190` `assert ... or True`; `:142` 200-or-422 | REWRITE falsifiable |
| `tests/receipts/api_keys_live_proof.json` GFW 503 counted ok:true; `test_suite_grand_total.json` collected≠passed | REWRITE receipts from real runs |
| `run_full_benchmark.py:12` hardcoded `os.chdir("c:/Users/Dell/...")`; torch.load per-step | REWRITE |

### 6.4 notebooks/ (nb13 unless noted)
| Target | Verb |
|---|---|
| §9 rng.normal "4-method causal counterfactual" → receipt | DELETE or implement real methods |
| §1 rigged 269/269 gauntlet (n_safe both branches, +19 constant, compliant:True fallback) | REWRITE honest counting, attacks actually executed |
| §5 hardcoded trajectory + "2735x" receipt | REWRITE from real `_score_guess` run |
| §8 noise-vector "transfer" probe | DELETE |
| §7 SyntaxError + SMGym mis-unpack | REWRITE (proves notebook never ran — must run after fix) |
| cert cell locals().get() silent defaults + hardcoded "= 100%" | REWRITE fail-loud |
| §12 `PROJECT_TOTAL_DEMONSTRATED = 248` constant | DELETE |
| nb11 K4 fabricated W&B curve; nb12 "ensemble" placeholder receipt | DELETE scripts + scrub W&B runs |
| nb10 dead repo URL + gated Llama + same unpack bug; nb08 hash-seeded reward + unpaired "paired" Wilcoxon; TARGET-in-prompt leak (nb10 N1, nb13 §6) | REWRITE or ARCHIVE |
| All 13 notebooks: zero outputs committed | EXECUTE + commit outputs (P1.5) |

### 6.5 scripts/
| Target | Verb |
|---|---|
| `bootstrap_leaderboard.py` + `wilcoxon_pairwise_leaderboard.py` (hardcoded RAP-XC stats, sorted-"paired" tests) + their receipts + plots quoting p=3.9e-18/d=+2.73 | DELETE chain (pass27-B is the honest replacement) |
| `pass27_killshot.py:678` + `pass28_killshot_v2.py:499` noise-as-conformal receipts | DELETE (keep `calibrate_conformal_from_harvest.py` — real) |
| `pass28_keys_ingest.py:229` + `pass28_K4_wandb_retry.py` fake W&B curves | DELETE + scrub W&B |
| `final_adversarial_20suite.py` self-testing inline gates + hardcoded attack #12 | REWRITE against real env |
| `pass22_full_squeeze.py:522,665,258` hardcoded causal estimates, constant feature counts, false provenance | DELETE receipts |
| `pass26:274` wrong-feedback hand-crafted trajectory; `:435` hardcoded ok:True | REWRITE/DELETE |
| `pass27:657` unconditional `key_authenticated=True` patch | DELETE (never mutate receipts post-hoc) |
| `pass28_killshot_v2.py:601` mock featurizers labeled trained; `:660` hand-typed "license audit" | DELETE / REWRITE from importlib.metadata |
| `export_all_onnx.py:84` untrained random-weight GCN in judge bundle | REWRITE with real weights or drop entries |
| `validate_ensemble_brent.py:33` synthetic sinusoid "Real-style" backtest | REWRITE on real FRED (K1 data exists) |
| 7× `patch_nb13*.py`, superseded v1 scripts, `push_to_hf_space.py`, `final_conformal_multilevel.py`, `scripts/__pycache__` | DELETE |
| 6 divergent .env parsers | REWRITE one shared loader (python-dotenv) |
| `run_frontier_judge_panel.py:90` regex-fabricated verdict conf=0.5 feeding agreement stats | REWRITE (exclude unparseable from stats, count separately) |

### 6.6 data/models/legacy/docs
| Target | Verb |
|---|---|
| `platinum.py:393,352` FRED col mismatch → always-hardcoded fallback; PPI labeled "Brent daily" | REWRITE (P1.6) |
| `frbny_supply_chain.pdf` = HTML error page counted in receipts | DELETE + refetch real GSCPI paper |
| `R1_VERIFIED.json` overcounts (fred=2 actual 1; "noaa_realtime" is static 2024 zip) | REWRITE counts |
| `DATASET_CARD.md:44` "snapshot 2024-01" (actual 2000–2026, updated 2025-12) | REWRITE |
| Never-consumed downloads (openflights 3.9MB, wto zip, pink sheet, noaa zip; empty un_comtrade/, imf_ifs/) | WIRE IN or DELETE + de-list |
| `FEATURE_INVENTORY.md:97` tabpfn-v2-clf "PRESENT (verified)" — dir doesn't exist | REWRITE + machine-check all claims (P2.2) |
| 3 ghost model deps (snowflake-arctic, bge-reranker, tabpfn-v2-clf) silently-dead features | SHIP weights or DELETE features+claims |
| `openrouter_war_room_panel.py:46` JUDGES_12 untested slugs; master.html "25-judge" claim | VERIFY live or DELETE |
| `tests/receipts/openrouter_liveness.json` 4/14 OK committed as proof | REGENERATE |
| 14 stub receipts `receipts_v2/` + README "20 receipts live" + PREPRINT DPO-judge + twin $135.5M (exit -9) + cherry-picked α=0.750 headline (raw 3-judge α=0.210) | P1.4: execute-or-delete + strike claims |
| `_dump/_ci_runs.json` (3-byte BOM) + 2 untracked helpers | DELETE / archive |
| `Makefile:60` v4.0 tag + baked benchmark claims; "9 probes"=8; POSIX activate on Windows | REWRITE |
| `baseline.py:117` TASK_HINTS leak ground-truth into baseline prompt (dup in inference.py:155) | Report with+without hints, or strip (owner call — default: strip for headline, keep hinted as separate labeled row) |
| docker-compose placebo DEMO_MODE/OFFLINE_MODE; Dockerfile.damocles unstartable CMD; baseline.py incoherent HF-router+gpt-4o defaults | REWRITE/DELETE |
| `release_assets.sh` retracted α claim + missing PITCH_DECK upload | REWRITE |

---

## 7. AI-LAYER SPEC

### 7.1 Provider architecture
`supplymind/llm/providers.py`: single interface, two backends — `openrouter` (default; existing
client promoted) and `ollama` (edge mode). Selection via `SUPPLYMIND_LLM_PROVIDER` env. Structured
JSON enforced at the interface (retry-on-invalid). All calls logged to `.openrouter_usage.jsonl`
(cost visibility).

### 7.2 Model policy (OpenRouter)
- Analyst/judges: strong open models (Qwen-2.5-72B-class or better) — configurable, never hardcoded slugs
  without a liveness check at startup. Registry pruned of dead slugs; `verify_openrouter_models.py`
  run becomes a CI-adjacent cron.
- Bulk/cheap ops (RAG summaries, hypothesis gen): free/cheap tier with fallback chain.
- Vision (port imagery): OpenRouter vision model replaces local qwen-vl.

### 7.3 Sanctioned local exceptions (until replaced/decided)
Chronos-Bolt (821MB), TimesFM-2 ckpt (2GB), TabPFN-reg (45MB), one mxbai copy — time-series +
embeddings have no OpenRouter route. Options ranked: (a) keep local (works today, edge-friendly),
(b) HF Inference API for embeddings, (c) drop features. **Default: (a) keep, documented as
first-class edge assets — likely a Qualcomm advantage.** Resolve mxbai split-brain: point
SentenceTransformer at ONE canonical source (local dir), delete redundant formats.

### 7.4 Custom-model story (real this time)
The v2→v5 analyst lineage + real A/B history (v3 lost to base Qwen; v5 calibration fixed it) is
gold — port it, re-run the A/B on OpenRouter with Brier scores (P1.2). If a fine-tune is wanted
for Phase B: do it for real (the DPO data exists — 21 verified preference pairs; the v5 crash logs
show exactly what to fix: torch/peft version skew), or distill analyst→small model with honest evals.

---

## 8. DATA LAYER SPEC
Live sources (keys in `.env`): FRED (Brent DCOILBRENTEU daily — replace truck-PPI misuse), EIA
(petroleum), NASA FIRMS (thermal anomalies), GFW (AIS/fishing — token is 798-char JWT; re-verify,
last liveness saw 503/422), NewsAPI (headlines), NOAA CDO (weather). Keyless: USGS, GDELT.
Static corpora: EM-DAT (registered download, document provenance), SEC EDGAR, DataCo, IBTRACS, WGI.
Each source: fetcher module + freshness stamp + cache + loud failure. `PROVENANCE.md` mandatory.

---

## 9. TESTING & VERIFICATION PROTOCOL
- `pytest tests/` = gate for every merge (CI, P0.5). Target: 184 → grow with llm/ + data/ mocks.
- Receipts: only from real runs, sha256-stamped, command + git SHA embedded, `match:true` required.
- E2E: scripted journey (reset→step→grade→analyst→war room) in CI against local server; Playwright
  browser test for dashboards (P2.3).
- Adversarial re-audit after P0.2 and before any submission: fresh agent hunts fakes; zero-finding
  required.
- Benchmarks: seeds ≥20, paired-by-seed stats, effect sizes correct, raw episode arrays committed.

---

## 10. DEPLOYMENT
- NEW HF Space (owner creates; old `Shaurya-Noodle/Supplymind` frozen as-is). No 1GB anxiety:
  keep the deploy lean anyway (image <2GB) — judges clone repos.
- `deploy-hf-space.yml` retargeted; smoke test = full journey; deploy log artifact kept.
- Docker: fixed `.dockerignore` (P0.3), one compose with api + dashboard only (damocles decision:
  fix module path + model mounts, or fold its endpoints into main server — default: fold in).

---

## 11. EXTERNAL RESOURCES TO ACQUIRE (owner fetches / installs)
Deep-research fleet died on session limit (retry after 08:00 IST reset — command in §13).
High-confidence shortlist meanwhile:
- **Repos to study/borrow** (clone into `ARCHIVE/references/` or read online):
  `karpathy/nanoGPT`, `karpathy/nanochat`, `karpathy/minbpe` (training-loop/eval hygiene, small-model
  story), `vwxyzjn/cleanrl` (single-file RL reference implementations), `DLR-RM/stable-baselines3` +
  `Farama-Foundation/Gymnasium` (already deps — upgrade patterns), `EleutherAI/lm-evaluation-harness`
  (credible LLM evals), `jxnl/instructor` (structured outputs over OpenRouter), `quic/ai-hub-models`
  (Qualcomm AI Hub — Phase B on-device), `microsoft/onnxruntime` (edge inference).
- **Papers**: Guo et al. 2017 "On Calibration of Modern Neural Networks" (Brier/ECE for analyst),
  Zheng et al. 2023 "Judging LLM-as-a-Judge" (panel design), Dabney et al. 2017 QR-DQN (cite
  correctly), Vovk conformal prediction (do it right this time).
- **Claude Code skills/MCP to install**: **Playwright MCP** (browser E2E for dashboards — the one
  genuinely missing tool), optionally HF MCP. Already installed & sufficient: superpowers, dataviz,
  interface-design, interaction-design, web-design-guidelines, code-review, taste-* packs.

## 12. OWNER MUST PROVIDE / DECIDE
1. **Qualcomm hackathon brief** (theme, judging criteria, deadline) — unblocks Phase B design.
2. **Rotate keys** flagged compromised (FRED, NEWS_API, WANDB, HF_TOKEN, NOAA — PHOENIX_PUSH_REPORT
   §3.6) + confirm the once-committed HF token is dead; keep new values in `.env` only.
3. **Create new HF Space** + fresh write-scoped HF_TOKEN (GitHub secret).
4. OpenRouter credit level / monthly budget (key already present; budget shapes model policy §7.2).
5. Decision (default provided): TASK_HINTS in baseline (§6.6) — strip or dual-report. Default: dual-report.
6. Decision: Damocles service — fold into main server (default) or fix as separate container.
7. Playwright MCP install approval (P2.3).

## 13. STATUS LOG
- 2026-07-02: 12-agent audit ran (9 done; ci-deploy/docs/final-submit + chief synthesis killed by
  Claude session limit, resets 08:00 IST). Gap-filled manually. Deep-research workflow also killed
  (39/46 agents) — **retry after reset**: re-run workflow `deep-research`
  (resumeFromRunId `wf_80933858-5b9`, script at
  `~/.claude/projects/C--Users-Dell-Desktop-Sleep-Token/.../workflows/scripts/deep-research-wf_80933858-5b9.js`).
- Audit JSON exports (full evidence, file:line): session scratchpad
  `...\scratchpad\audits\audit_0..8_*.json` — copy into `_audit/` in-repo if persistence wanted.
- This plan written by Fable orchestrator; implementation NOT started (owner directive).
