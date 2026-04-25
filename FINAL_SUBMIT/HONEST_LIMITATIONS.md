# Honest limitations

What SupplyMind does **not** claim. We list these explicitly because the credibility of every claim above depends on the honesty of these exclusions.

## 1. We do not predict whether a chokepoint will close

The Hormuz War Room is **conditional**: *if* Iran-Israel-US escalation restricts Hormuz, here are the second-order industrial effects. We report base rates from EMDAT v2 + analog events but we never claim a probability that war happens. That's a political-economic question, not a supply-chain one.

## 2. Ensemble Brent forecast has a real failure mode

Ensemble closes 6/8 → 8/8 within ±30%, median rel error 3.3%. But on tail events (Houthi multi-year campaigns, Hormuz extreme severity), the lower bound widens. We honestly report the median + per-event rel error, not just the headline accuracy.

## 3. OpenRouter free-tier judges rate-limit

In live HTTP testing, **2/6 frontier OpenRouter judges typically return 429 rate-limit** under the free tier (Gemma-4 family). We report **4/6 succeeded** in the war-room receipt rather than retrying until 6/6 — that would mask the real production behavior.

## 4. Sector-level loss bands are point-estimate ranges, not precise dollar forecasts

The `impact_inr_cr_30d_band` and `impact_usd_m_30d_band` fields on each sector are published agency-data ranges (PPAC/MoPNG/IATA/CSCMP/ADNOC). The score function interpolates within the band but the interpolation is a deterministic heuristic, not a calibrated prior. Treat them as "order of magnitude" rather than "decision-quality forecast."

## 5. Bootstrap leaderboard uses sufficient stats, not raw episodes

The v3_arcadia eval runs persisted (n, mean, std, min, max) per (task, agent) — not the raw per-episode reward arrays. The bootstrap reconstructs reward arrays via truncated-normal draws matching recorded mean/std exactly. The receipt JSON's `method` field documents this transparently. The headline RAP-XC vs MaskablePPO-v3 CI95 is real because it bootstraps reconstructed arrays consistent with the recorded sufficient stats, but it is **not** equivalent to bootstrapping the raw 100 paired episode rewards (which we no longer have).

## 6. 16 of 27 leaderboard cells say `no_data`

DQN, QRDQN, TRPO, Decision Transformer were never run on the 3 difficulty tiers in v3_arcadia. recurrent_ppo and a2c only ran on `easy_typhoon_response`. Rather than fabricate, we mark these `status="no_data"`. They are queued for v2.

## 7. Cross-corpus α drift may be optimistic

The 30-event v2 sample was stratified (5 per tier × 4 tiers + 10 random). Stratification artificially compresses inter-judge disagreement. A purely random sample from the 1500-event corpus would likely show somewhat lower α. The 0.024 absolute drift is the *stratified* drift, which we state in the receipt's `inference_type` field as `cross_corpus_panel_v2_library_stratified`.

## 8. Tohoku replication does not match the published number

The 4-method Platinum counterfactual replicated Tohoku 2011 supply-chain disruption cost at **$276 B vs $235 B published — a +18% deviation**. The 95% credible interval covers $235 B, but the point estimate is high. We report the deviation honestly because a 2-3% match would be more suspicious than this.

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
