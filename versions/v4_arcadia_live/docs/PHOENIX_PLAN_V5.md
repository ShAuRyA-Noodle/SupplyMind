# Phoenix Plan v5 — Win the Meta PyTorch OpenEnv Hackathon Finals

*Authored 2026-04-22. Finals 2026-04-25/26 (48 hrs on-campus, Bangalore). Runway: ~3 days.*

This is the strategic + tactical plan to maximize P(top-3) for SupplyMind v4.0-arcadia-live.
Goal: **$10K first prize + Meta / Hugging Face interview gateway.**

---

## 1. Strategic Situation

### Hackathon reality check (from scaler.com + github.com/raun/openenv-course)

- **Theme**: building RL *environments* on Meta's **OpenEnv** framework (Gymnasium-style, Docker, HF Spaces).
- **Round 1 passed**: we're in the ~15 finalist cohort out of ~800 teams.
- **Round 2 (finals)**: April 25–26, Scaler Bangalore, **48-hour on-campus build**. Mentorship from Meta engineers *during* the hackathon. Judged by Meta's global team via programmatic checks + LLM scoring.
- **Prizes**: $10K 1st · $4.55K 2nd · $4K 3rd · $2K × 5 (4th–8th) · $650 × 7 (9th–15th). $30K total.
- **Interview pipeline**: Winners get direct interview opportunities with Meta and Hugging Face AI teams. "Your hackathon performance becomes your application."
- **What wins** (direct quote from Scaler's page): *"practical AI environment design, clean code architecture, meaningful open-source contributions, and ability to implement production-grade RL systems using standardized frameworks."*

### Critical insight

The hackathon grades **the environment + the agent + the open-source contribution back**, not just "a cool ML product." SupplyMind's supply-chain story is our narrative wrapper; the OpenEnv compliance + trained agents + live eval loop are the technical substance judges score. We must re-weight the pitch so OpenEnv sits in the foreground, not as an afterthought.

### Phoenix verdict

**DO NOT start a third rebuild.** The phoenix already rose:

- `versions/v3_arcadia/` is frozen at commit `02251e9` (the "ashes").
- `versions/v4_arcadia_live/` (v4.0-arcadia-live) is the phoenix — 19,521 lines of production code, 76 new tests (249 total), 13 receipts, 20 modules. Committed at `6729e54`.
- Creating `versions/v4_arcadia_live_v2/` now and re-writing 19K LOC in 3 days = certain loss.

**Recommendation**: *extend* v4 with 5 new killer features + close 3 identified weaknesses. All new work still lands under `versions/v4_arcadia_live/` as the user requested.

---

## 2. What's World-Class Already (KEEP, don't touch)

| Pillar | Headline number | Receipt |
|---|---|---|
| OpenEnv compliance | 19 formal tests pass in 2s | `tests/test_openenv_compliance.py` |
| Real data | 261,175 points from 8 cited public sources | `docs/core/DATA_SOURCES.md` |
| 13 SOTA models locally | all verified, Q4_K_M quantized where needed | `versions/v3_arcadia/results/R1_VERIFIED.json` |
| RAG | mxbai P@1 = **0.962**, MRR **0.978**, BEIR nDCG@10 **0.971** | `R5_GRANITE_mxbai_P1.reproduce.sh` |
| LLM 2-judge panel | Krippendorff α = **0.750**, Cohen κ = **0.747** | `R4_2JUDGE_Krippendorff_alpha.reproduce.sh` |
| RL | MaskablePPO **+26.8%** vs PPO, **0 violations**, 10,800-episode CI95 | `R6_MaskingAblation_easy_lift.reproduce.sh` |
| GNN | Custom 3-layer GCN in pure PyTorch, **−48 / −49 / −64%** MAE vs MLP | `R6_GCN_easy_MAE_vs_MLP.reproduce.sh` |
| Conformal PIs | Per-horizon dev @ 95% = **0.024** (4.7× tighter than pooled) | `R6_AquaRegia_WTI_dev95.reproduce.sh` |
| Forecasting | TimesFM-CP dev @ 95% **0.050 / 0.032** on WTI & EUR-USD | `R3_TimesFM_CP_WTI_dev95.reproduce.sh` |
| Reproducibility | 13 one-bash-command receipts, 40 committed JSONs | `versions/v4_arcadia_live/receipts/INDEX.md` |
| Honest limitations | stacking null result on ≥0.97 ceiling, DeepSeek 31% GT acc | `V4_STACKING_V2_lift_vs_WV.reproduce.sh`, `docs/v3/BENCHMARKS_VS_PUBLIC.md` §8 |

---

## 3. What's At Risk (MUST fix before finals)

| # | Risk | Evidence | Impact | Fix cost |
|---|---|---|---|---|
| R1 | **Autoresearch loop crashes on all 5 seeds** | `autoresearch/state.json`: every hypothesis `"status": "rejected", "reason": "status=crash; no valid scores"`, `wall_clock_s ~5`, `best: null` | **CRITICAL** — our flagship unique feature doesn't converge | 4–6 h |
| R2 | Live Hormuz pipeline needs venue Wi-Fi | NewsAPI/GDELT/FRED all require network; rate limits at demo time | Demo could fail mid-pitch | 3–4 h |
| R3 | HF Space deployment unverified | `docs/v3/DEPLOY_HF_SPACE.md` is a plan, no confirmed live URL | Can't hand judges a URL | 2–3 h |
| R4 | Demo video not recorded | `demo/DEMO_VIDEO_SCRIPT.md` = script only | Pitch lacks asynchronous artifact | 2–3 h |
| R5 | No "try-it-yourself" path for the env | Gradio leaderboard exists but no "drop your agent, get CI95" flow | Judges can only read, not play | 4–6 h |
| R6 | Round 1 problem statement alignment unknown | 200+ problems on Scaler page; unclear which one we submitted against | Narrative may drift from judge rubric | user confirms |
| R7 | OpenEnv narrative under-weighted in current README | README opens with "13 SOTA models, 261K real data" — should open with "OpenEnv-compliant environment with trained SOTA agents" | Judges read the first 30 seconds | 1 h |

---

## 4. Phase-by-phase plan

### Phase 0 — TODAY (Apr 22, ~6–8 h) — Unbreak the flagship

**Non-negotiable before anything else.**

| Step | What | Why | Est |
|---|---|---|---|
| P0.A | Read `autoresearch/experiments/s1_bigger_network/train.stdout.log`, identify why every seed crashes in ~5s (likely path bug, openenv adapter, or seed-gen fault) | Root-cause before fixing | 30 min |
| P0.B | Patch `candidate_train.py` smoke path. Run 3 seeds on `easy_typhoon_response` to convergence (50k steps, ~30 min/seed). | At least 1 experiment must produce `accepted: true` and `metric_ci95_lower > baseline` | 2–3 h |
| P0.C | Generate a real `autoresearch/lab_notebook.md` with 3 hypotheses → results → accept/reject reasoning (Karpathy's actual notebook pattern) | Judges will ask to see the notebook | 1 h |
| P0.D | Run `python -m versions.v4_arcadia_live.realtime.ingestor --once` TODAY, freeze outputs to `realtime/replay_cache_2026_04_22.json`. Add `--replay` flag to ingestor. | Offline demo works without venue Wi-Fi | 2 h |
| P0.E | `pytest tests/ versions/v4_arcadia_live/tests/ -q` → confirm 249 green. Fix Windows path flakes before travel. | Regression guard | 1 h |
| P0.F | Rewrite README.md first 30 seconds to lead with **"OpenEnv-compliant RL environment for supply-chain risk"**, then 13 SOTA agents, then reproducibility. | Recenter judge framing on OpenEnv | 45 min |

**Exit criteria for Phase 0**: `state.json.best` is not null, 249 tests green, offline replay cache exists, README opens with OpenEnv.

---

### Phase 1 — Apr 23 (~12 h) — Unique-feature build

**Target: ship at least 2 of the 3 features below. They are the differentiators.**

#### P1.A — OpenEnv Arena (6 h) — 🏆 highest judge-impact feature

- Gradio + FastAPI page at `GET /arena` on the main server.
- A judge uploads `policy.pt` (any PyTorch nn.Module with `forward(obs) → action`).
- Server runs **50 episodes per task × 3 tasks** (easy, medium, hard) and returns: reward mean + bootstrap CI95, violations/ep, latency, comparison against our MaskablePPO baseline on the same seeds.
- Pre-populated leaderboard: MaskablePPO (ours), PPO, A2C, RecurrentPPO, Random, Greedy.
- Prompt for judges in docs/v4/JUDGES.md: *"Bring your own PyTorch policy, drop it in /arena, see where you land."*
- **Why this wins**: turns a static env into a playable product; directly aligns with hackathon's "environment design" axis; Meta engineers and judges can use it in the room.

#### P1.B — Live Counterfactual Digital Twin (4 h)

- When the Hormuz endpoint fires, run **100 Monte-Carlo rollouts** of the trained MaskablePPO vs "no action" vs "greedy baseline" on a supply graph conditioned on the live event severity.
- Return a loss distribution (histogram + median + p95), not just a point estimate.
- Existing scripted claim: "$324M → $65M = 80% savings." Turn this into a LIVE number tied to the day's NewsAPI + FRED Brent reading.

#### P1.C — Self-improving reward curriculum via autoresearch (3 h)

- Extend `hypothesis_engine.py` so one class of mutations proposes new reward-shaping terms (e.g., "penalty for starving tier-2 during crisis") rather than only hyperparams.
- Evaluator rejects any reward-mutation that increases reward but *also* increases constraint violations (a cheating guard).
- Ship one accepted reward-shaping improvement with documented lift.
- **Why this wins**: the "Karpathy move" — an agent improving its own supervision signal is exactly the story judges at a PyTorch hackathon want to hear.

---

### Phase 2 — Apr 24 (~10–12 h) — Polish + deploy

| Step | What | Est |
|---|---|---|
| P2.A | HF Space deploy per `docs/v3/DEPLOY_HF_SPACE.md`. Smoke test 12 endpoints including `/arena` and `/live/*`. Put live URL in `docs/v4/JUDGES.md`. | 3 h |
| P2.B | Record 3-min demo video on user's Mac: Hormuz live → autoresearch notebook → OpenEnv Arena → reproducibility receipts. | 3 h |
| P2.C | Pitch deck v2 (8 slides): (1) OpenEnv env, (2) 13-model stack, (3) Karpathy autoresearch, (4) live Hormuz, (5) OpenEnv Arena, (6) reproducibility, (7) honest limitations, (8) roadmap. | 2 h |
| P2.D | End-to-end dry run: fresh venv → clone → `pip install` → run docs/v4/JUDGES.md's 4-minute path → 5 receipts → pytest. Retime to <4 min. | 2 h |
| P2.E | Travel prep: `.env` rotated, laptop+charger+USB, mobile hotspot test, offline replay verified. | 2 h |

**Exit criteria for Phase 2**: HF Space green, demo video uploaded to Drive + YouTube unlisted, pitch deck printed, dry-run <4min.

---

### Phase 3 — Apr 25 AM (~4–6 h) — Travel + arrival

- Land Bangalore, test venue Wi-Fi, run a smoke test at the venue itself.
- Confirm both online (live Hormuz) and offline (replay cache) paths still work.
- Talk to 1–2 Meta engineers before the clock starts — learn what they weight.

---

### Phase 4 — Apr 25–26 — On-campus finals (48 h)

**This is where "being at finals" matters. Use Meta mentors as input.**

| Block | Hours | Plan |
|---|---|---|
| **A: Recon + first demo** | 0–6 | Pitch live Hormuz to ≥2 Meta engineers for 5-min reactions. Adjust weighting based on feedback. Identify the 1 thing they flagged. |
| **B: Real-time build** | 6–20 | Ship ONE feature suggested by a Meta engineer. Candidates already lined up (red-team agent, mentor demo mode, OpenEnv upstream PR). This turns mentorship into a tangible artifact judges hear about. |
| **C: Unique-thing-#2** | 20–36 | Formal upstream PR to `github.com/meta-pytorch/openenv` (or HF's mirror) submitting SupplyMind as a reference env. Hackathon page literally says "code ships to Meta-backed projects." Highest open-source signal available. |
| **D: Pitch + rehearsal + present** | 36–48 | 3 full rehearsals with sleep in between. Final demo to judges. |

---

## 5. Ten NEW unique features (beyond autoresearch + Hormuz)

Ordered by judge-impact-per-hour. Pick 3–5 to actually ship.

| # | Feature | Phase | Cost | Judge signal |
|---|---|---|---|---|
| 1 | **OpenEnv Arena** — drop-in-your-agent harness | P1.A | 6 h | ⭐⭐⭐⭐⭐ "this IS the hackathon theme" |
| 2 | **Live Counterfactual Digital Twin** | P1.B | 4 h | ⭐⭐⭐⭐⭐ live$-saved during pitch |
| 3 | **Formal OpenEnv upstream PR** | P4.C | 4 h at venue | ⭐⭐⭐⭐⭐ "ships to Meta-backed projects" |
| 4 | **Self-improving reward curriculum** | P1.C | 3 h | ⭐⭐⭐⭐ the Karpathy move |
| 5 | **Red-team adversarial agent** | P4.B | 4 h at venue | ⭐⭐⭐⭐ robustness + autoresearch synergy |
| 6 | **Mentor Demo Mode** — judge types free-text crisis → full pipeline | P4.B | 3 h | ⭐⭐⭐⭐ unscripted + live, high-drama |
| 7 | **Reproducibility Bounty** — $100 if someone beats α=0.750 | P2.C | 30 min | ⭐⭐⭐ memorable gesture |
| 8 | **"Zero-to-Deploy in 2 min" Colab** — clone → train → render policy video | P2.D | 2 h | ⭐⭐⭐ judges who click not read |
| 9 | **Carbon-adjusted Pareto live** — live Brent moves the frontier | P1.B bundled | 1 h extension | ⭐⭐ polish on existing Pareto module |
| 10 | **arXiv submission** — upload PREPRINT.md the morning of Apr 25 | P2.E | 1 h | ⭐⭐⭐ only submission with an arXiv link |

---

## 6. Data / clarifications I need from you (please answer before Phase 0 starts)

1. **Round 1 problem statement**: which of Scaler's ~200 problems did we submit? This anchors the Round 2 narrative.
2. **API keys in `.env` right now**: are `NEWSAPI_KEY`, `FRED_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` (optional) populated and working? I need to verify the live pipeline works *today* before we lose time.
3. **Travel + venue**: flying to Bangalore Apr 25? Hotel Wi-Fi plan? Mobile hotspot backup? Is the Alienware laptop (the one with the RTX 4080) the travel machine, or a different laptop?
4. **Team**: solo or with 1–2 teammates? With partners I'd parallelize Phase 1. Hackathon allows teams up to 3.
5. **HF Space**: is there a live URL already, or does it need a Phoenix rebuild per `docs/v3/DEPLOY_HF_SPACE.md`?
6. **Mac recording**: is the Mac set up (Keynote / OBS / ScreenFlow) for the demo video shoot?
7. **Ollama models**: are `qwen2.5:14b-instruct-q4_K_M`, `mistral-nemo:12b-instruct-q4_K_M`, `deepseek-r1-local-q4` all loaded and warm? (Needed for 3-judge panel in the live demo.)

---

## 7. Kill criteria (things we will NOT do, even if tempted)

- No full-from-scratch rebuild. v4 IS the phoenix. Starting over with 3 days = certain loss.
- No new SOTA model downloads (15GB downloads eat a day; existing 13 are enough).
- No new benchmarks without reproducibility receipts. Every claim = one bash command.
- No untested code in the 90-second demo path. If it's in the demo, it has a test.
- No API dependencies without an offline fallback. Every live feature has a replay cache.
- No skipping OpenEnv compliance tests. Those 19 tests are the first signal judges check.
- No renaming / reorganizing the existing structure. `docs/v4/JUDGES.md` paths are already advertised.
- No untagged commits during finals. Every commit = Sleep Token track name + phase marker (Rain, The Summoning, Vore, Chokehold, DYWTYLM, Ascensionism, Arcadia II).
- No `--no-verify`, no hook skipping, no force-pushes to main.

---

## 8. Probability assessment

With plan executed end-to-end:

| Outcome | Prob |
|---|---|
| Top 3 ($4K–$10K) | **45–60%** |
| Top 10 finalist ($650–$2K guaranteed) | **85–92%** |
| Meta / HF interview opportunity | **90%+** |

Current state if we stop here: top-10 essentially locked, top-3 at ~30–40%. The delta between "top-10" and "top-3" lives in Phase 1 (unique features the other 14 finalists won't have) and Phase 4.C (upstream OpenEnv PR).

---

## 9. One-sentence strategic summary (pre-ROLL / pre-superpowers)

> **Fix autoresearch today, ship the OpenEnv Arena tomorrow, deploy + polish Apr 24, land in Bangalore Apr 25 with every demo path tested both online and offline, then use mentor hours on-campus to ship one feature Meta engineers hand-pick and submit the env upstream — and we win.**

---

## 10. ROLL framework deep integration (Alibaba, Apache 2.0)

Upstream: `github.com/alibaba/ROLL` (v0.2.1, Mar 2026). Vendored copy at `vendor/ROLL/` is current — no upstream drift. 259 Python files, 56.3k LOC core, 17.7k LOC tests. **This is not a toy; it's what Alibaba ships to thousand-GPU clusters.**

### 10.1 What ROLL actually gives us

| Capability | Feasible on 12GB/solo/3 days? | Judge impact |
|---|---|---|
| **DPO pipeline** (preference pairs → fine-tuned judge) | ✅ 2–4h on RTX 4080, Qwen-3B + LoRA r=8 | ⭐⭐⭐⭐⭐ |
| **RLVR** (reinforcement learning with verifiable reasoning) | ⚠️ Possible with LoRA, risky in time | ⭐⭐⭐⭐ |
| **Agentic RL with GiGPO** (step-wise multi-turn) | ⚠️ 1–2 day build; scaffolded + partial results is honest | ⭐⭐⭐⭐⭐ |
| **LLMJudgeRewardWorker** (3 modes: API / local / cluster) | ✅ 2–3h wrapper | ⭐⭐⭐⭐ |
| **MCP tool integration** (already in ROLL's agentic pipeline) | ✅ maps directly to our existing tools | ⭐⭐⭐⭐ |
| **Action parser** (`Qwen3CoderActionParser`) | ✅ <1h | ⭐⭐⭐ |
| **Custom ROLL environment** (register `supplymind_crisis_env`) | ✅ 4–6h, huge for upstream PR | ⭐⭐⭐⭐⭐ |
| **On-policy distill** (Qwen-14B → Qwen-3B) | ❌ too heavy in 3 days | ⭐⭐ |
| **Megatron 5D parallelism** | ❌ needs multi-GPU cluster | N/A |
| **FSDP2 / DeepSpeed ZeRO-3** | ⚠️ works with CPU offload, slow | ⭐⭐ |

### 10.2 The five ROLL integrations I recommend (ranked)

1. **ROLL-DPO-judge-v1** — Fine-tune Qwen-2.5-3B-Q4 with DPO on our 26 crisis scenarios as preference pairs (GT-correct response = chosen; worst-judge output = rejected). LoRA r=8, ~3h training. Publishable receipt: `V4_DPO_JUDGE_accuracy_delta.reproduce.sh`. This proves we actually did LLM post-training, not just prompt engineering.
2. **ROLL agentic RL loop for supplymind-analyst** — Register SupplyMind as a ROLL environment (`env_manager.tags: [supplymind]`). Multi-turn: observe crisis → call tool (forecast / RAG / RL-policy) → observe outcome → act → report. Train with **GiGPO** (step-wise, dense feedback). Even partial convergence is a killer demo.
3. **LLMJudgeRewardWorker integration** — Our existing 3-judge panel becomes the reward signal for ROLL training. Novel composition of our R4 Dangerous panel feeding a ROLL RLVR loop.
4. **MCP tool-use bridge** — ROLL already supports MCP-registered tools in agentic pipelines. Our forecast/RAG/RL endpoints already exist. Wire them as MCP tools, train the analyst to call them. Dual signal: MCP (Anthropic standard) + OpenEnv (Meta standard) in one agent.
5. **Upstream PR to alibaba/ROLL** — Submit `examples/supplymind/` as a reference agentic environment. Same open-source signal as the OpenEnv PR, doubled. Even an unmerged PR shows intent.

### 10.3 Install strategy — isolated `versions/v5_phoenix/` folder + WSL2 day-budget

**User directive (Apr 22)**: if ROLL install fights us, invest a full day on **WSL2 with CUDA passthrough** to push it through rather than falling back. All ROLL work lives in a **new `versions/v5_phoenix/` folder at the repo root** so the existing `versions/v4_arcadia_live/` v4 stays frozen and safe.

**Directory layout**:
```
Sleep-Token/
├── versions/v3_arcadia/              # frozen at 02251e9
├── versions/v4_arcadia_live/      # frozen v4 (249 tests, 13 receipts)  <- DO NOT TOUCH
├── versions/v5_phoenix/         # NEW — all ROLL + superpowers work lives here
│   ├── README.md
│   ├── .venv-roll/          # isolated Python env
│   ├── roll_integration/
│   │   ├── dpo_judge/       # ROLL-DPO-judge-v1
│   │   ├── env/             # SupplyMind registered as ROLL env (upstream PR)
│   │   ├── reward_bridge/   # LLMJudgeRewardWorker -> our 3 judges
│   │   └── configs/         # Hydra/YAML configs
│   ├── supplymind_skills/   # publishable skill pack
│   │   ├── benchmark-runner/
│   │   ├── autoresearch-experiment/
│   │   └── live-demo-orchestrator/
│   ├── experiments/         # training runs + checkpoints
│   ├── receipts/            # grade-A receipts (command+stdout+exit+expected/actual)
│   └── docs/                # PREPRINT_V5.md, PHOENIX_STORY.md
```

**Two-phase install**:

*Phase A (Windows-native, 0.5 day)*: Try the path of least resistance first.
```bash
cd versions/v5_phoenix
python -m venv .venv-roll
.venv-roll\Scripts\activate
pip install -e ../vendor/ROLL/[hf]   # HF strategy only, no megatron/vllm/sglang
pip install peft trl==0.9.6 accelerate bitsandbytes
python -c "from roll.pipeline.dpo import DPOPipeline; print('ok')"
```

If this works → we're done, 3.5h reclaimed for other features.

*Phase B (WSL2 + CUDA, full day)*: only if Phase A fails.
```bash
wsl --install -d Ubuntu-22.04         # if not installed
# inside WSL2:
sudo apt install nvidia-cuda-toolkit
python -m venv .venv-roll-wsl
pip install -e /mnt/c/Users/Dell/Desktop/Sleep-Token/vendor/ROLL/[hf,deepspeed]
pip install vllm==0.6.3 flash-attn --no-build-isolation
```
WSL2 gets us proper Linux wheels for vLLM + flash-attn + DeepSpeed. CUDA passes through to the RTX 4080. `.venv-roll-wsl/` stays separate from Windows venv.

**Worst case**: if even WSL2 fights us, fall back to standalone `trl.DPOTrainer` for ROLL-DPO-judge-v1 (same DPO result, loses env-PR and agentic-RL). Phase A + Phase B budget: **~8h max** before calling it.

---

## 11. Superpowers framework deep integration (obra, MIT, v5.0.7)

Vendored copy at `superpowers-main/superpowers-main/` is current (v5.0.7, Mar 31 2026). 15 skills, SessionStart hook, platform-aware (Claude Code / Cursor / Copilot CLI / Gemini / OpenCode).

### 11.1 What superpowers actually gives us (methodology, not code)

| Skill / pattern | Value | Cost |
|---|---|---|
| `subagent-driven-development` | Per-task fresh subagent → 2-stage review (spec → quality) | Already used during v3/v4 builds |
| `writing-plans` | Bite-sized tasks (2–5 min each), zero-context-assumed | This Phoenix Plan already mirrors the pattern |
| `verification-before-completion` | "Claim = evidence"; fresh command output required | Maps to our 13 receipts |
| `test-driven-development` | Iron law: no production code before failing test | Matches our existing testing culture |
| `using-git-worktrees` | Parallel branches without context switching | Useful on-campus if teammate joins |
| `dispatching-parallel-agents` | Concurrent subagents for independent subsystems | Speed at finals |
| Platform-aware SessionStart hook | One hook, all IDEs | Minor (we're on Claude Code) |
| **The meta-move: publish a skill pack** | Judges install your skill, see methodology | 2–3h authoring |

### 11.2 The three superpowers integrations I recommend

1. **`supplymind-skills` skill pack — public marketplace submission** — Ship 3 skills:
    - `benchmark-runner` (TDD for benchmarks: baseline → change → verify)
    - `autoresearch-experiment` (maps to our autoresearch/ module — plan → run → receipt)
    - `live-demo-orchestrator` (pre-demo checklist, fallback, post-demo receipt)
    
    Publish to `obra/superpowers-marketplace` + Claude Code plugins marketplace. Add to `docs/v4/JUDGES.md`: *"Judges: install `supplymind-skills` in your Claude Code to reproduce our methodology."* **This is a second open-source artifact, on top of the upstream OpenEnv/ROLL PRs.**

2. **Adopt `writing-plans` + `subagent-driven-development` for the 48-hour finals** — Every hour of on-campus work starts with a bite-sized plan in `docs/superpowers/plans/2026-04-25-<phase>.md`, executed by subagents, receipt-verified. Git log becomes a TDD-discipline artifact judges can read. I already structured the Phoenix Plan this way; we formalize it at finals.

3. **`verification-before-completion` receipt upgrade** — Our 13 receipts currently emit a value (e.g., `0.9622`). Upgrade to superpowers-grade receipts: include `command`, `full stdout`, `exit code`, `expected`, `actual`, `match: true/false`. Auto-generate on commit via a tiny pre-commit hook. One morning's work, massive judge-facing credibility bump.

### 11.3 What we do NOT take from superpowers

- The `.cursor-plugin/` / `.codex/` / `gemini-extension.json` plumbing — we're solo, Claude Code only.
- The deprecated `commands/` slash commands — superseded by Skill tool.
- The brainstorm WebSocket server — we don't need live-collab visualization.

---

## 12. Revised top-20 unique features (expanded from 10)

Marked ⚑ = ROLL-enabled, ⚒ = superpowers-enabled, 🌐 = live geopolitics, 🔬 = research-rigor, 📦 = open-source contribution.

| # | Feature | Tags | Phase | Cost | Impact |
|---|---|---|---|---|---|
| 1 | OpenEnv Arena — drop-in PyTorch policy | — | P1.A | 6h | ⭐⭐⭐⭐⭐ |
| 2 | Live Counterfactual Digital Twin | 🌐 | P1.B | 4h | ⭐⭐⭐⭐⭐ |
| 3 | Upstream PR to Meta's OpenEnv repo | 📦 | P4.C | 4h @ venue | ⭐⭐⭐⭐⭐ |
| 4 | Self-improving reward curriculum | 🔬 | P1.C | 3h | ⭐⭐⭐⭐ |
| 5 | Red-team adversarial agent | 🔬 | P4.B | 4h @ venue | ⭐⭐⭐⭐ |
| 6 | Mentor Demo Mode — free-text crisis → full pipe | 🌐 | P4.B | 3h @ venue | ⭐⭐⭐⭐ |
| 7 | Reproducibility Bounty $100 | 📦 | P2.C | 30min | ⭐⭐⭐ |
| 8 | Zero-to-Deploy Colab (2 min) | 📦 | P2.D | 2h | ⭐⭐⭐ |
| 9 | Carbon-adjusted Pareto live (FRED Brent) | 🌐 | P1.B | 1h | ⭐⭐ |
| 10 | arXiv submission of PREPRINT.md | 📦 | P2.E | 1h | ⭐⭐⭐ |
| **11** | **ROLL-DPO-judge-v1**: Qwen-3B DPO on 26 crisis pairs | ⚑🔬 | P1 (new) | 4h | ⭐⭐⭐⭐⭐ |
| **12** | **SupplyMind as a ROLL environment** (upstream PR) | ⚑📦 | P4.C | 4h @ venue | ⭐⭐⭐⭐⭐ |
| **13** | **Agentic RL for supplymind-analyst via GiGPO** | ⚑🔬 | P1/P4 | 8–10h | ⭐⭐⭐⭐ |
| **14** | **LLMJudgeRewardWorker bridge** (our 3 judges → ROLL reward) | ⚑ | P1 (new) | 3h | ⭐⭐⭐⭐ |
| **15** | **MCP tool-use analyst** (forecast/RAG/RL as MCP tools + ROLL train) | ⚑ | P1/P4 | 4h | ⭐⭐⭐⭐ |
| **16** | **`supplymind-skills` skill pack** — publish to marketplace | ⚒📦 | P2 | 3h | ⭐⭐⭐⭐⭐ |
| **17** | **Superpowers-driven 48h finals execution** — `docs/superpowers/plans/` artifact | ⚒ | P4 | 0 (method) | ⭐⭐⭐ |
| **18** | **Grade-A receipt upgrade** (command + stdout + exit + expected/actual) | ⚒🔬 | P2 | 3h | ⭐⭐⭐⭐ |
| **19** | **Dual upstream PRs** — Meta/OpenEnv + Alibaba/ROLL in 48h | ⚑📦 | P4.C | bundled | ⭐⭐⭐⭐⭐ |
| **20** | **Methodology video** — show brainstorm → plan → subagent → receipt chain | ⚒ | P2.B | +30min | ⭐⭐⭐ |

**Target for ship**: 8–12 of these 20. Ranked by impact-per-hour above.

---

## 13. Revised 3-day plan with ROLL + Superpowers woven in

### Phase 0 — TODAY (Apr 22, 8–10h) — Unbreak + install + framework audit

| # | Task | Frame | Est |
|---|---|---|---|
| 0.1 | Root-cause the autoresearch crash (`state.json` shows all 5 seeds = `status=crash, wall_clock_s~5`) | — | 30 min |
| 0.2 | Patch `candidate_train.py`, run 3 seeds to convergence, 1 accepted | — | 2–3h |
| 0.3 | Real `lab_notebook.md` with 3 hypotheses + accept/reject | — | 1h |
| 0.4 | Freeze Hormuz replay cache → `realtime/replay_cache_2026_04_22.json` | — | 2h |
| **0.5** | **Create `versions/v5_phoenix/` folder** with directory skeleton (see §10.3) + placeholder README | ⚑⚒ | 30 min |
| **0.6** | **ROLL install Phase A** (Windows-native, HF-only, `.venv-roll/`) + Qwen-0.5B smoke test | ⚑ | up to 4h; if green, stop here |
| **0.6b** | **ROLL install Phase B** (WSL2 + CUDA + full extras) — only if Phase A fails | ⚑ | up to 4h more |
| **0.7** | **Superpowers skill pack scaffold** — `versions/v5_phoenix/supplymind_skills/{benchmark-runner,autoresearch-experiment,live-demo-orchestrator}/SKILL.md` stubs | ⚒ | 1h |
| 0.8 | Rewrite README.md first 30s to lead with OpenEnv | — | 45 min |
| 0.9 | `pytest tests/ versions/v4_arcadia_live/tests/ -q` → 249 green (unchanged; `versions/v5_phoenix/` not in suite yet) | — | 30 min |

**Gate**: autoresearch converges AND (Phase A green OR Phase B green OR `trl` fallback decision made) AND replay cache exists. `versions/v4_arcadia_live/` tests still 249 green (we never touch it). Budget ceiling: if total Phase 0 > 12h, stop + pivot to `trl` fallback regardless.

---

### Phase 1 — Apr 23 (14–16h) — Unique features + ROLL-DPO-judge

| # | Task | Frame | Est |
|---|---|---|---|
| 1.1 | **ROLL-DPO-judge-v1** — Qwen-2.5-3B + LoRA r=8, DPO on 26 crisis preference pairs | ⚑🔬 | 4h (includes training wait) |
| 1.2 | OpenEnv Arena (Gradio + FastAPI at `/arena`, judges drop in `policy.pt`) | — | 6h |
| 1.3 | Live Counterfactual Digital Twin — 100 MC rollouts conditioned on live Hormuz signal | 🌐 | 4h |
| 1.4 | **LLMJudgeRewardWorker bridge** — our 3 judges → ROLL reward function | ⚑ | 3h |

**Gate**: pick any 3 of the 4. With ROLL installed, 1.1 is cheap; without ROLL, 1.1 uses `trl.DPOTrainer` and still ships.

---

### Phase 2 — Apr 24 (12–14h) — Deploy + skill pack + polish

| # | Task | Frame | Est |
|---|---|---|---|
| 2.1 | HF Space deploy + smoke test all endpoints (incl. `/arena` + `/live/*`) | — | 3h |
| 2.2 | Record 3-min demo video (Hormuz live → autoresearch lab notebook → Arena → ROLL-DPO delta → receipts) | — | 3h |
| 2.3 | Pitch deck v2 (8 slides) | — | 2h |
| **2.4** | **Publish `supplymind-skills` skill pack** to `obra/superpowers-marketplace` fork + Claude Code plugins | ⚒📦 | 3h |
| **2.5** | **Grade-A receipt upgrade** — auto-include command + stdout + exit + expected/actual; pre-commit hook | ⚒🔬 | 3h |
| 2.6 | End-to-end dry-run <4 min judge path | — | 2h |
| 2.7 | Travel prep, API rotation, offline caches verified | — | 2h |

**Gate**: HF Space green, demo video uploaded, skill pack discoverable by judges' `/plugin install`.

---

### Phase 3 — Apr 25 AM — Travel + venue smoke (4h)

No new features; only proving everything still works at venue + talking to Meta engineers.

---

### Phase 4 — Apr 25–26 — On-campus 48h (ROLL + superpowers in full force)

| Block | Hours | Focus |
|---|---|---|
| A | 0–6 | Recon + pitch to ≥2 Meta engineers for reactions; run live demo in the room |
| **B** | **6–20** | **ROLL upstream PR draft**: fork `alibaba/ROLL`, add `examples/supplymind_crisis/` with env+config+README. Dispatch a subagent per sub-task (superpowers pattern). | ⚑📦⚒ |
| **C** | **20–36** | **OpenEnv upstream PR**: meta-pytorch/openenv, submit SupplyMind as reference env. + Mentor-suggested feature (red-team agent likely) | 📦 |
| D | 36–48 | Pitch rehearsals (3×), final demo |

**Dual upstream PRs** = dual open-source signal. Hackathon page says "code ships to Meta-backed projects" — we go one better and ship to Alibaba too.

---

## 14. Framework-specific risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **ROLL install fails on Windows-native (Phase A)** | Medium-High | Loses ~0.5 day of unique-feature build time | Phase B: WSL2 + CUDA passthrough, full 4h budget. If even WSL2 fails → `trl.DPOTrainer` fallback (same DPO science, loses env-PR and agentic). Hard ceiling: 8h total on install before pivot. |
| **ROLL install succeeds but blows up `versions/v4_arcadia_live/` venv** | — | Would break 249 tests | Isolated `.venv-roll/` inside `versions/v5_phoenix/` only. User directive: never touch existing v4. |
| **ROLL-DPO training OOMs on 12GB** | Low (LoRA r=8 on 3B fits) | No DPO demo | Drop to Qwen-1.5B or shrink LoRA r=4; both fit comfortably |
| **Skill pack marketplace PR doesn't merge pre-finals** | High | Can't say "install our skill pack" | Host as a public GitHub repo + pointer in README; judges can `git clone` even without marketplace merge |
| **ROLL env PR doesn't merge pre-finals** | High (Alibaba review is slow) | Less upstream impact | PR draft + link from README counts; even an open PR is the artifact |
| **Subagent-driven dev at finals creates conflicting commits** | Medium | Git hell | Use git worktrees (superpowers skill #7) for isolation |
| **Ollama can't host the DPO-trained LoRA adapter** | Medium | No live judge serving | Serve via `vllm serve Qwen2.5-3B-Instruct --enable-lora --lora-modules supplymind=./adapters/`. Fallback: `transformers` pipeline with `peft.PeftModel.from_pretrained`. |
| **ROLL dependency pins conflict with existing `.venv`** | High | Breaks existing tests | Isolated `.venv-roll/`; never touch main venv |

---

## 15. Updated probability assessment with ROLL + Superpowers integration

With full plan + ROLL-DPO-judge + skill pack + dual upstream PRs landing:

| Outcome | Prob (pre-ROLL plan) | Prob (with ROLL + superpowers) |
|---|---|---|
| Top 3 ($4K–$10K) | 45–60% | **60–75%** |
| Top 10 finalist | 85–92% | **92–97%** |
| Meta / HF interview | 90%+ | **95%+** |
| Alibaba / other downstream offers | — | **meaningful non-zero** |

The ROLL + superpowers additions primarily unlock the **top-3 tier** (from 45–60% to 60–75%) because they give us two things every single other finalist will lack: (a) actual LLM post-training results on a real domain, (b) two separate upstream open-source contributions + a public skill pack.

---

## 16. Revised one-sentence strategic summary (post-ROLL / post-superpowers)

> **Fix autoresearch today + bring up ROLL in a brand-new isolated `versions/v5_phoenix/` folder (WSL2 if Windows-native fails); ship OpenEnv Arena + ROLL-DPO-judge on Apr 23; publish the `supplymind-skills` skill pack + deploy HF Space + record demo Apr 24; land in Bangalore Apr 25 with both online and offline demo paths tested, then spend the 48h finals shipping dual upstream PRs (Meta/OpenEnv and Alibaba/ROLL) plus one mentor-suggested feature — and we don't just make top 10, we're in the top-3 fight.**

**Non-negotiable**: `versions/v4_arcadia_live/` (v4 with 249 tests, 13 receipts, frozen) stays untouched throughout. `versions/v5_phoenix/` is the new home for ROLL + superpowers integration. If Phoenix fails for any reason, v4 is still a complete top-10 submission on its own.

---

*Tracks: "Rain" opens v4. "The Summoning" opens finals. "Ascensionism" marks ROLL+superpowers integration. "Arcadia II" closes the cycle.*
