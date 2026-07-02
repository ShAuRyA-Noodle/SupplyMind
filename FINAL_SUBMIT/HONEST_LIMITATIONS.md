# Honest limitations

What SupplyMind does **not** claim. We list these explicitly because the credibility of every claim above depends on the honesty of these exclusions.

## 0. Fabrication cleanup in progress (2026-07-02 internal audit)

A brutal 12-agent internal audit on 2026-07-02 found that the engine core is genuinely strong
but the **presentation/receipts layer contained fabricated and overstated claims**. We are
removing them rather than hiding them. Judges should know, up front, what is being re-run:

- **Fabricated statistics chain struck.** The "RAP-XC vs MaskablePPO Wilcoxon p / Cohen's d" and
  "REINFORCE p / d" headlines came from the bootstrap-CI and Wilcoxon-pairwise leaderboard
  generator scripts, which **sorted two independently synthesized samples and
  called them "paired"** — a statistically meaningless test. Those p-values and the plots quoting
  them are struck and pending an honest re-run on real paired-by-seed episodes.
- **Conformal coverage (0.9001) unverified.** The committed conformal receipts were computed on
  `rng.normal()` Gaussian noise labeled "model-NLL split conformal", not on real model NLLs. The
  real number must come from `calibrate_conformal_from_harvest.py` on real transitions.
- **Ensemble Brent "8/8 within ±30%" unverified.** `validate_ensemble_brent.py` backtested on a
  synthetic sinusoid+AR(1) price history, not real FRED Brent. Re-run on real `DCOILBRENTEU` pending.
- **Adversarial "[STRUCK] = 100% blocked" (and the "257/257" variant) rigged.** The gauntlet counted
  every non-crashing call as "safe" in *both* branches, never passed injection payloads to two of
  the tools, and added 19 attacks as a constant — 100% was guaranteed by construction. Being
  rewritten to attack the real env `step()`/gates.
- **No DPO-fine-tuned judge exists.** Every DPO training run crashed or trained with `None`
  gradients. Any claim of a "DPO-fine-tuned Qwen-2.5-3B judge" is false today and is struck.
- **"$135.5M twin savings" receipt is not real** — the on-disk `V5_Twin_savings_gt_zero` receipt
  records the process killed (`exit_code: -9`, `match: false`). Do not cite the figure until re-run.
- **14 of 20 "grade-A" receipts are unexecuted stubs** (`<pending-first-run>`, `exit_code: -1`)
  with reproduce commands that cannot run (pre-reorg, digit-leading module paths).
- **Three "foundation model" dirs do not exist**: `tabpfn-v2-clf`, `snowflake-arctic-embed-l`,
  `bge-reranker-v2-m3`. The dependent features (7th TabPFN judge, 2-embedder ensemble, reranker)
  silently fall back and never run. FEATURE_INVENTORY.md's old "PRESENT (verified)" rows were false.
- **"[STRUCK] features demonstrated" was a hardcoded constant**, not a measured count. Struck.
- **Krippendorff α headline.** The advertised "α = 0.750" is a cherry-picked **2-judge** sub-panel;
  the raw **3-judge** panel scored **α = 0.210**. Both must be stated together or neither. We now
  publish the full ladder (0.210 3-judge / 0.750 2-judge sub-panel) instead of the flattering one.

Tracking table with per-claim status: [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).

## 1. We do not predict whether a chokepoint will close

The Hormuz War Room is **conditional**: *if* Iran-Israel-US escalation restricts Hormuz, here are the second-order industrial effects. We report base rates from EMDAT v2 + analog events but we never claim a probability that war happens. That's a political-economic question, not a supply-chain one.

## 2. Ensemble Brent forecast accuracy is UNVERIFIED (being re-run)

The previously advertised "closes 6/8 → 8/8 within ±30%, median rel error 3.3%" was produced by
`validate_ensemble_brent.py` backtesting on a **synthetic** sinusoid+AR(1) price history, not real
FRED Brent — so it is not a real out-of-sample accuracy figure. It is struck pending a re-run on
real `DCOILBRENTEU` slices for each event. The forecast ensemble code (Chronos+TimesFM+TabPFN) is
real; only the backtest data was synthetic.

## 3. OpenRouter free-tier judges rate-limit

In live HTTP testing, **2/6 frontier OpenRouter judges typically return 429 rate-limit** under the free tier (Gemma-4 family). We report **4/6 succeeded** in the war-room receipt rather than retrying until 6/6 — that would mask the real production behavior.

## 4. Sector-level loss bands are point-estimate ranges, not precise dollar forecasts

The `impact_inr_cr_30d_band` and `impact_usd_m_30d_band` fields on each sector are published agency-data ranges (PPAC/MoPNG/IATA/CSCMP/ADNOC). The score function interpolates within the band but the interpolation is a deterministic heuristic, not a calibrated prior. Treat them as "order of magnitude" rather than "decision-quality forecast."

## 5. Bootstrap leaderboard "paired" test is invalid — STRUCK

The v3_arcadia eval runs persisted only (n, mean, std, min, max) per (task, agent) — not raw
per-episode arrays. The bootstrap-CI leaderboard generator script reconstructs arrays via truncated-normal draws,
then **sorts the two reconstructed samples and pairs them by quantile rank** before running the
"paired" bootstrap / Wilcoxon. Sorting two independent samples and pairing them guarantees
near-zero variance in the difference and manufactures significance — the resulting p-values
([STRUCK], etc.) are **not valid**. The whole RAP-XC-vs-MaskablePPO headline pair is struck until a
real paired-by-seed comparison is run on the actual trained checkpoints (P1.3).

## 6. 16 of 27 leaderboard cells say `no_data`

DQN, QRDQN, TRPO, Decision Transformer were never run on the 3 difficulty tiers in v3_arcadia. recurrent_ppo and a2c only ran on `easy_typhoon_response`. Rather than fabricate, we mark these `status="no_data"`. They are queued for v2.

## 7. Cross-corpus α drift may be optimistic

The 30-event v2 sample was stratified (5 per tier × 4 tiers + 10 random). Stratification artificially compresses inter-judge disagreement. A purely random sample from the 1500-event corpus would likely show somewhat lower α. The 0.024 absolute drift is the *stratified* drift, which we state in the receipt's `inference_type` field as `cross_corpus_panel_v2_library_stratified`.

## 8. Tohoku "4-method counterfactual" is not 4 real methods — STRUCK as evidence

The "$276 B vs $235 B published, +18%" Tohoku replication came from a "4-method causal
counterfactual" in which **3 of the 4 methods (synthetic control, BSTS, SCM do-calculus) are
hardcoded literals** (250 / 263 / 285) and the 4th is `np.random.normal()` draws (audit_5, audit_4).
The `platinum.py` "real FRED" path also always hits a hardcoded fallback table because it parses
CSV columns that don't exist (audit_6). The number is struck as a scientific claim until real
causal methods are implemented. (An honest framing: it is a hand-set economic *anchor*, not a
replicated estimate.)

## 9. Synthetic Brent pre-history in ensemble validation

`scripts/validate_ensemble_brent.py` constructs a 200-day pre-event Brent history by anchoring at the documented `pre` price + AR(1) noise + sinusoidal seasonal. This is not real FRED Brent data on the actual pre-event day window — it's a deterministic synthesizer. The validation method note is explicit about this. A future v2 should fetch real FRED Brent slices for each event.

## 10. We don't have ACLED, Reddit OAuth, or full SAR access

Three sources we wanted but didn't get:
- **ACLED** (conflict events) requires institutional access we don't have. We use GDELT-Conflict instead.
- **Reddit OAuth** app credentials weren't approved during build window. We use HN tech ticker.
- **Full Synthetic Aperture Radar** access for port congestion would cost real money. We use Qwen-VL on free RGB satellite imagery.

## 11. War-Room is conditional on operator-asserted scenario parameters

The user supplies `severity`, `brent_price_usd_bbl`, `duration_days`. The model does not detect these from the live signal stream — it accepts them as inputs. A future v2 should auto-extract scenario parameters from incoming news + sentiment.

## 12. The "no AI fluff" rule is a discipline, not a guarantee

Every claim in this submission is intended to be sha256-replayable from a committed file. If you find a claim that isn't, file an issue and we will either fix the receipt or retract the claim.

---

**These honesty admissions are the headline.** Every team will pitch their model. We pitch a system that can be audited.
