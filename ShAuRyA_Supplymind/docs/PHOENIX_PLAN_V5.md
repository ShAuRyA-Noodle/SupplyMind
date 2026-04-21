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

- `v3_arcadia/` is frozen at commit `02251e9` (the "ashes").
- `ShAuRyA_Supplymind/` (v4.0-arcadia-live) is the phoenix — 19,521 lines of production code, 76 new tests (249 total), 13 receipts, 20 modules. Committed at `6729e54`.
- Creating `ShAuRyA_Supplymind_v2/` now and re-writing 19K LOC in 3 days = certain loss.

**Recommendation**: *extend* v4 with 5 new killer features + close 3 identified weaknesses. All new work still lands under `ShAuRyA_Supplymind/` as the user requested.

---

## 2. What's World-Class Already (KEEP, don't touch)

| Pillar | Headline number | Receipt |
|---|---|---|
| OpenEnv compliance | 19 formal tests pass in 2s | `tests/test_openenv_compliance.py` |
| Real data | 261,175 points from 8 cited public sources | `DATA_SOURCES.md` |
| 13 SOTA models locally | all verified, Q4_K_M quantized where needed | `v3_arcadia/results/R1_VERIFIED.json` |
| RAG | mxbai P@1 = **0.962**, MRR **0.978**, BEIR nDCG@10 **0.971** | `R5_GRANITE_mxbai_P1.reproduce.sh` |
| LLM 2-judge panel | Krippendorff α = **0.750**, Cohen κ = **0.747** | `R4_2JUDGE_Krippendorff_alpha.reproduce.sh` |
| RL | MaskablePPO **+26.8%** vs PPO, **0 violations**, 10,800-episode CI95 | `R6_MaskingAblation_easy_lift.reproduce.sh` |
| GNN | Custom 3-layer GCN in pure PyTorch, **−48 / −49 / −64%** MAE vs MLP | `R6_GCN_easy_MAE_vs_MLP.reproduce.sh` |
| Conformal PIs | Per-horizon dev @ 95% = **0.024** (4.7× tighter than pooled) | `R6_AquaRegia_WTI_dev95.reproduce.sh` |
| Forecasting | TimesFM-CP dev @ 95% **0.050 / 0.032** on WTI & EUR-USD | `R3_TimesFM_CP_WTI_dev95.reproduce.sh` |
| Reproducibility | 13 one-bash-command receipts, 40 committed JSONs | `ShAuRyA_Supplymind/receipts/INDEX.md` |
| Honest limitations | stacking null result on ≥0.97 ceiling, DeepSeek 31% GT acc | `V4_STACKING_V2_lift_vs_WV.reproduce.sh`, `BENCHMARKS_VS_PUBLIC.md` §8 |

---

## 3. What's At Risk (MUST fix before finals)

| # | Risk | Evidence | Impact | Fix cost |
|---|---|---|---|---|
| R1 | **Autoresearch loop crashes on all 5 seeds** | `autoresearch/state.json`: every hypothesis `"status": "rejected", "reason": "status=crash; no valid scores"`, `wall_clock_s ~5`, `best: null` | **CRITICAL** — our flagship unique feature doesn't converge | 4–6 h |
| R2 | Live Hormuz pipeline needs venue Wi-Fi | NewsAPI/GDELT/FRED all require network; rate limits at demo time | Demo could fail mid-pitch | 3–4 h |
| R3 | HF Space deployment unverified | `DEPLOY_HF_SPACE.md` is a plan, no confirmed live URL | Can't hand judges a URL | 2–3 h |
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
| P0.D | Run `python -m ShAuRyA_Supplymind.realtime.ingestor --once` TODAY, freeze outputs to `realtime/replay_cache_2026_04_22.json`. Add `--replay` flag to ingestor. | Offline demo works without venue Wi-Fi | 2 h |
| P0.E | `pytest tests/ ShAuRyA_Supplymind/tests/ -q` → confirm 249 green. Fix Windows path flakes before travel. | Regression guard | 1 h |
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
- Prompt for judges in JUDGES.md: *"Bring your own PyTorch policy, drop it in /arena, see where you land."*
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
| P2.A | HF Space deploy per `DEPLOY_HF_SPACE.md`. Smoke test 12 endpoints including `/arena` and `/live/*`. Put live URL in `JUDGES.md`. | 3 h |
| P2.B | Record 3-min demo video on user's Mac: Hormuz live → autoresearch notebook → OpenEnv Arena → reproducibility receipts. | 3 h |
| P2.C | Pitch deck v2 (8 slides): (1) OpenEnv env, (2) 13-model stack, (3) Karpathy autoresearch, (4) live Hormuz, (5) OpenEnv Arena, (6) reproducibility, (7) honest limitations, (8) roadmap. | 2 h |
| P2.D | End-to-end dry run: fresh venv → clone → `pip install` → run JUDGES.md's 4-minute path → 5 receipts → pytest. Retime to <4 min. | 2 h |
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
5. **HF Space**: is there a live URL already, or does it need a Phoenix rebuild per `DEPLOY_HF_SPACE.md`?
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
- No renaming / reorganizing the existing structure. `JUDGES.md` paths are already advertised.
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

## 9. One-sentence strategic summary

> **Fix autoresearch today, ship the OpenEnv Arena tomorrow, deploy + polish Apr 24, land in Bangalore Apr 25 with every demo path tested both online and offline, then use mentor hours on-campus to ship one feature Meta engineers hand-pick and submit the env upstream — and we win.**

---

*Track: "Rain" opens v4. "The Summoning" opens finals. "Arcadia II" closes the cycle.*
