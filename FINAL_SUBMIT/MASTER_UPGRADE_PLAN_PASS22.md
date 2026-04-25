# MASTER UPGRADE PLAN — pass 22 (the killshot)

**Goal**: lift weighted score from 82.5/100 → 94/100. Every upgrade below is real, executable, and closes a specific gap surfaced by `HYPERMODE_DEEP_AUDIT_PASS22.md`.

Each upgrade has: ID · objective · file/script · effort · impact-on-criterion · receipt to ship.

Ordered by (impact ÷ effort) descending. Pick top N you can fit in remaining time.

---

## TIER 1 · KILLSHOT UPGRADES (each closes a high-impact honest limitation)

### U1 · Real episodic bootstrap re-run (closes L5, B5)
**Why**: `bootstrap_leaderboard.json` currently uses sufficient-stats reconstruction via truncated-normal draws. Single biggest credibility risk if a stats-savvy judge looks closely.

**How**:
1. `scripts/pass22_real_episodic_bootstrap.py` — re-run RAP-XC + MaskablePPO-v3 + scripted_baseline on `hard_cascading_crisis` for 100 episodes each, persisting raw per-episode reward arrays.
2. Re-bootstrap CI95 from raw arrays.
3. Receipt: `bootstrap_leaderboard_v2_real_episodic.json` with `method=real_per_episode_paired_bootstrap`.

**Effort**: ~30 min compute · ~15 min code.
**Impact**: criterion 4 (10%) + general credibility (across all criteria). Marginal lift ~3 points.

---

### U2 · Fill 16/27 leaderboard no-data cells (closes L6, B5)
**Why**: DQN, QRDQN, TRPO, Decision Transformer have never been run on the 3 difficulty tiers in the persisted v3_arcadia evals.

**How**:
1. `scripts/pass22_fill_baseline_grid.py` using Stable-Baselines3 (DQN, A2C, TRPO via sb3-contrib) and d3rlpy (CQL, BC, IQL, AWAC).
2. 30 episodes per (algo × tier) — enough for CI95 but not optimal training.
3. Receipt: `algo_grid_complete.json` showing 27/27 cells filled with mean ± std + bootstrap CI95.

**Effort**: ~60 min compute · ~45 min code (algorithms are off-the-shelf).
**Impact**: criterion 3 (20%) + criterion 4 (10%). Marginal lift ~3 points.

---

### U3 · Real FRED Brent backfill for 8 events (closes L9)
**Why**: `validate_ensemble_brent.py` currently synthesizes 200-day pre-event Brent history via AR(1) + sinusoid. FRED key is in `.env` and unused.

**How**:
1. `scripts/pass22_fred_brent_real.py` — for each of 8 documented events, fetch real FRED `DCOILBRENTEU` slice 200 trading days pre-event.
2. Re-run ensemble Brent validation against real history.
3. Receipt: `ensemble_brent_real_fred_v2.json` with `prehistory_source=fred_dcoilbrenteu_real`.

**Effort**: ~15 min code · ~5 min API.
**Impact**: criterion 1 (40%) + criterion 4 (10%). Marginal lift ~2 points.

---

### U4 · 90-second YouTube video recorded and live (closes mandatory submission requirement)
**Why**: Hackathon explicitly requires "<2 minute video on YouTube OR mini-blog on HuggingFace OR slide deck". We have the slide deck, but a recorded video lifts storytelling (30% weight) materially.

**How**:
1. Use the existing `DEMO_SCRIPT_90S.md` as exact teleprompter.
2. Record 3 takes against `JUDGE_DASHBOARD.html` + `/demo/master` + Hormuz war-room.
3. Upload unlisted to YT, link from README.

**Effort**: ~30 min recording + edit.
**Impact**: criterion 2 (30%). Marginal lift ~3 points.

---

### U5 · HuggingFace mini-blog post live (redundant insurance with U4)
**Why**: Submission requirements list "mini-blog on HuggingFace OR mini-video on YouTube". Both is better than either.

**How**:
1. Cross-post `HACKATHON_README.md` (compressed to ~500 words) as HF blog under `Shaurya-Noodle/supplymind-openenv-hackathon`.
2. Link from README.

**Effort**: ~20 min.
**Impact**: criterion 2 (30%). Marginal lift ~1 point.

---

## TIER 2 · BUG FIXES (high-leverage micro-edits)

### U6 · Fix WTI parsing in chained demo (closes B1)
**Why**: `chained_live_demo.json` reports `latest_wti_price_usd: "2.612"`. Real WTI ~$60–85/bbl. Looks like reading wrong column.

**How**: Inspect EIA response, fix series_id and field selection in `scripts/pass20_grand_final.py:stage_A_eia`.
**Effort**: ~5 min.
**Impact**: removes embarrassing error in flagship demo.

### U7 · Mirror v2 reinforce headline metrics to root keys (closes B2)
**Why**: `wordle_real_reinforce_v2_curve.json` has `final_eval` and `cohen_d_vs_null` only nested under `summary`. Receipt-readers checking root will miss them.

**How**: Patch in `scripts/final_real_reinforce_wordle_v2.py` to emit root-level mirrors.
**Effort**: ~5 min.
**Impact**: prevents accidental judge mis-read.

### U8 · GFW receipt honesty patch (closes B4)
**Why**: GFW returns 503 transient but receipt marks `ok=true` because key authenticated. Cleaner: separate `key_authenticated` vs `data_ok` fields.

**How**: Patch `api_keys_live_proof.json` and `chained_live_demo.json` writers.
**Effort**: ~5 min.
**Impact**: cleaner honesty.

### U9 · Tier-3 100-word pool real eval (closes B7)
**Why**: Receipt shows 89% solve at both 50-word and 100-word — should degrade.

**How**: Re-run `final_real_reinforce_wordle_v2.py --pool-size 100` with 200 eps. Expect ~78–82%.
**Effort**: ~10 min.
**Impact**: scaling honesty.

### U10 · Mark v1 REINFORCE as superseded (closes B6)
**Why**: `wordle_real_reinforce_curve.json` (v1, 36% solve) still indexed. Judge may pick lower number.

**How**: Add `"superseded_by": "wordle_real_reinforce_v2_curve.json"` to v1 file metadata, also note in `phoenix_v5_receipts_INDEX.json`.
**Effort**: ~3 min.
**Impact**: defensive.

### U11 · Tighten conformal v3 receipt payload (closes B8)
**Why**: `conformal_tight_v3.json` is only 710 bytes — looks truncated. Compare to v1 1120 bytes.

**How**: Re-run with full report payload (per-alpha breakdown + per-Mondrian-cell coverage).
**Effort**: ~5 min.
**Impact**: completeness.

---

## TIER 3 · COVERAGE-FILL UPGRADES (close 28-feature gap to 99.2%)

### U12 · Multi-agent K2-K6 individual receipts
Files exist; need standalone smoke receipts for each subcomponent (negotiate / belief / comm / coalition / mixed coop-comp).
- `scripts/pass22_multi_agent_subreceipts.py` writes 5 small JSONs.
- Effort: ~20 min.

### U13 · Federated J2-J4 individual receipts
Files exist; need DP noise smoke + FedAvg round demo + cross-silo simulation receipts.
- `scripts/pass22_federated_subreceipts.py`.
- Effort: ~20 min.

### U14 · Live data M9-M20 keyless smoke receipts
12 free / keyless data sources need a smoke fetch each: OpenStreetMap / MarineTraffic public / Suez / Hormuz / RedSea / TradeBalance / ContainerIndex / Brent spot / WTI spot / Reuters trades / Bloomberg ticker / Twitter geo.
- `scripts/pass22_keyless_data_smokes.py` — 12-source fetch with 5s timeout each, sha256-stamp response.
- Skip any that hit firewall transparently.
- Effort: ~30 min.

### U15 · Quantile regression standalone receipt
- File `forecasting/quantile_reg.py` exists; needs its own receipt.
- Effort: ~10 min.

### U16 · BGE rerank quality benchmark
- Real BGE rerank quality on Win fallback path needs a comparative receipt.
- Effort: ~15 min.

---

## TIER 4 · NEW CAPABILITIES (innovation surface)

### U17 · Reasoning Gym integration as alt env
**Why**: RL guide §22-23 RLVE point. Reasoning Gym is the canonical RLVE source. Plugging it as an alt env lifts criterion 1 (40%).

**How**:
1. `ShAuRyA_Phoenix/reasoning_gym_env/` — wrap 3-5 Reasoning Gym tasks as OpenEnv environments.
2. Run REINFORCE for 500 episodes each, report reward curves.
3. Receipt: `reasoning_gym_smoke.json`.

**Effort**: ~90 min.
**Impact**: criterion 1 (40%). Marginal lift ~2 points.

### U18 · TRL GRPO real demo (not just SB3 / REINFORCE)
**Why**: We currently use REINFORCE + SB3 baselines. TRL GRPO is the canonical hackathon stack. A real GRPO run (even small) closes the gap.

**How**:
1. Use the existing `wordle_env/train_grpo.py` scaffold.
2. Run 50 GRPO steps on Wordle env with Qwen2.5-0.5B.
3. Receipt: `trl_grpo_wordle_real.json` + reward curve plot.

**Effort**: ~60 min compute + ~20 min code.
**Impact**: criterion 4 (10%). Marginal lift ~1.5 points.

### U19 · ROLL upstream PR draft live (not just local)
**Why**: `upstream_prs/` already has PR drafts for meta-pytorch/openenv and alibaba/ROLL. Pushing them to actual GitHub forks (not yet merged, but visible) lifts community-credibility.

**How**:
1. Create personal forks of meta-pytorch/openenv and alibaba/ROLL.
2. Push the prepared PR branches.
3. Open draft PRs (do not request review until after submission to avoid noise).

**Effort**: ~20 min.
**Impact**: criterion 1 (40%). Marginal lift ~1 point.

### U20 · Auto-extract scenario params from news (closes L11)
**Why**: War-room currently takes operator-asserted severity / brent_price / duration. Auto-extraction is flashy.

**How**:
1. `server/scenario_extractor.py` — Qwen3-4B prompt that parses incoming news headlines and outputs `{severity, brent_price, duration_days}`.
2. Receipt: `scenario_extractor_smoke.json` showing 5 historical news → extracted params vs ground truth.

**Effort**: ~45 min.
**Impact**: criterion 1 (40%). Marginal lift ~1 point.

### U21 · MCP tool stress with adversarial inputs
**Why**: `is_openenv_compliant()` returns true, but adversarial MCP-tool inputs haven't been tested.

**How**:
1. `scripts/pass22_mcp_adversarial.py` — fuzzes each `tool_sm_*` MCP tool with malformed JSON, oversized payloads, injection attempts.
2. Receipt: `mcp_adversarial.json`.

**Effort**: ~30 min.
**Impact**: criterion 4 (10%). Marginal lift ~0.5 points.

---

## TIER 5 · STORYTELLING POLISH

### U22 · Judge objection handbook
- One-pager with 12 anticipated objections + crisp rebuttal each.
- File: `JUDGE_OBJECTION_HANDBOOK.md` (this pass).
- Effort: included.

### U23 · 90-second teleprompter rehearsal track
- Audio-only rehearsal with timing markers.
- Effort: ~15 min.

### U24 · 1-minute "who is this for" non-technical pitch
- For non-technical judges. Sticks to "what changes if you have this".
- Effort: ~15 min.

### U25 · Live HF Space verification badge
- `JUDGE_DASHBOARD.html` already exists. Add a live "🟢 HF Space verified at TIMESTAMP" pulled via JS fetch.
- Effort: ~10 min.

---

## TIER 6 · METRIC-PRESSURE UPGRADES (push every metric to 97-98%)

### U26 · Tighter REINFORCE re-train with longer horizon
- Current: 95.5–97% solve. Push to ≥97% deterministic by training 30K episodes on tier-2 with adaptive entropy.
- Effort: ~3 hr GPU.
- Receipt: `wordle_real_reinforce_v3_curve.json`.

### U27 · MaskablePPO retrain on hard tier with curriculum
- Current ceiling 0.78. Push to ≥0.85 with 3-stage curriculum.
- Effort: ~2 hr GPU.

### U28 · Conformal coverage at α=0.05 with 16K calib (already at 16K — verify exact 0.95)
- Current 0.9544 at α=0.05. Re-cal with calibration-set 32K to land exactly 0.9500 within 1e-4.
- Effort: ~30 min compute.

### U29 · Brent ensemble — re-fit weights via constrained Brent's method on FRED real history
- Replaces synthetic + tightens median rel error from 3.32% → < 2.5%.
- Effort: ~30 min.

### U30 · Wilcoxon p-value tightening for primary leaderboard (RAP-XC vs MaskablePPO)
- Current p=3.9e-18 with reconstructed arrays. Re-run with real episodic data → expect p < 1e-30.
- Effort: included with U1.

---

## TIER 7 · ARTIFACT FINISHING

### U31 · GitHub release v4.1-final-killshot tag
- Tag with all FINAL_SUBMIT contents bundled into release zip.
- Effort: ~10 min.

### U32 · HF Hub model upload
- Push REINFORCE v2 trained checkpoint to HF Hub `Shaurya-Noodle/supplymind-reinforce-v2`.
- Effort: ~15 min.

### U33 · License audit
- Verify MIT compatibility for all third-party deps (Stable-Baselines3 MIT ✅, d3rlpy MIT ✅, Unsloth Apache 2.0 ✅).
- Effort: ~10 min.

### U34 · README final pass
- Embed updated metrics post-pass-22.
- Add screenshot from recorded video as hero image.
- Effort: ~20 min.

### U35 · Judge cold-open opening line
- 1-line hook for the first 8 seconds. Currently "SupplyMind: a retrieval-augmented RL agent..." — too long.
- Try: "If Hormuz closes tomorrow, India loses ₹X-trillion in 30 days. Watch what 1 LLM, RL-trained, does about it."
- Effort: ~5 min.

---

## CRITICAL PATH (if you only have 4 hours)

| Order | Upgrade | Cumulative time | Why first |
|---|---|---|---|
| 1 | U6 (fix WTI parsing) | 5 min | Embarrassing bug in flagship demo |
| 2 | U7 (mirror reinforce keys) | 10 min | Defensive |
| 3 | U10 (mark v1 superseded) | 13 min | Defensive |
| 4 | U3 (real FRED Brent) | 33 min | Eliminates L9 — high-impact, low-effort |
| 5 | U1 (real episodic bootstrap) | 78 min | Eliminates L5 — single biggest credibility risk |
| 6 | U4 (record 90s video) | 108 min | Mandatory submission, criterion 2 weight 30% |
| 7 | U2 (fill 16 no-data cells) | 168 min | Eliminates L6 |
| 8 | U22 (objection handbook) | 188 min | Defensive arsenal — already shipping in this pass |
| 9 | U17 (reasoning gym) | 280 min | Innovation surface |
| 10 | U31 (GH release tag) | 290 min | Artifact bundling |

**4-hour ship state**: weighted score lifts from 82.5 → ~91/100. Top-3 prob 60–72%. Top-1 prob 28–42%.

## CRITICAL PATH (if you have 12+ hours)

Run all 35 upgrades. Weighted score ceiling 94/100. Top-3 prob 65–75%. Top-1 prob 32–45%.

---

## What this plan does NOT promise

- Does not promise 90% top-1 win probability. That number is not credible against unknown competition.
- Does not promise every metric will hit 98%. Some metrics (Brent rel error) have irreducible variance from real-world data.
- Does not promise zero gaps. 250-feature coverage will land at ~99% not 100% — some features are inherently consolidated under multi-feature receipts.
- Does not promise the recorded video will be Hollywood quality. It will be clear, scripted, and on time.

**These honesty admissions ARE the headline.** Every team will pitch their model. We pitch a system that can be audited.

End of plan. Owner: ship in priority order; checkpoint after every 3 upgrades.
