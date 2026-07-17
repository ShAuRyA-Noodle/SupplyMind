# Honest limitations

**Regenerated 2026-07-15.** What SupplyMind does **not** claim, and what is still pending. The
credibility of every result depends on the honesty of these exclusions. Per-claim status:
[`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md). Machine gate: [`scripts/verify_claims.py`](../scripts/verify_claims.py).

---

## 0. The audit, and the rebuild

A brutal internal audit (2026-07-02) found a genuinely strong engine wearing fabricated presentation.
We removed the fakes **and rebuilt the capabilities behind them for real**. Eight results that were
then "STRUCK / being re-run" now have committed, machine-checked receipts:

| Was (2026-07-02) | Now (real receipt) |
|---|---|
| Rigged "269/269 = 100%" gauntlet | **318/318 real attacks blocked, 0 breaches** ([`adversarial_gauntlet_REAL.json`](../tests/receipts/adversarial_gauntlet_REAL.json)) |
| "0.9001 conformal" on Gaussian noise | **90.03% held-out coverage** on real transitions ([`conformal_REAL.json`](../tests/receipts/conformal_REAL.json)) |
| "8/8 within ±30%" on a synthetic price series | **mean MAPE 7.1%** on real FRED Brent walk-forward ([`ensemble_brent_REAL.json`](../tests/receipts/ensemble_brent_REAL.json)) |
| "$276B, +18%" from 3 hardcoded literals + 1 `np.random` | **4 real causal methods**, honest scope disagreement ([`counterfactual_4method_REAL.json`](receipts/counterfactual_4method_REAL.json)) |
| 3 "absent" model dirs | obtained + **measured** → 2 RETIRE, 1 KEEP ([`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json)) |
| World Bank "in corpus" (actually 0 chunks) | **0 → 20 WB chunks** ([`rag_recook_REAL.json`](../tests/receipts/rag_recook_REAL.json)) |
| Inconsistent Wordle p-values | one **real per-episode paired test** ([`pass27_B_real_episodic_bootstrap.json`](receipts/pass27_B_real_episodic_bootstrap.json)) |
| No real federated result | **honest negative** — FedAvg = centralized ([`fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json)) |

The point of the audit was not to shrink the project; it was to make every remaining claim true.

---

## 1. Results that are honest negatives or ties (we report them as measured)

- **Federated learning does not beat centralized.** On 180K real DataCo orders across 5 regions,
  FedAvg AUC 0.7218 equals centralized 0.7218 (Δ −0.0). The data is close to IID by region, so
  federation buys privacy, not accuracy. We say so.
- **The embedder ensemble is worse than mxbai alone** (P@1 −0.076), and **the reranker lowers P@1**
  (−0.038). Both are **retired by measurement**, not shipped as wins. TabPFN, which *does* add
  signal (AUC 0.738 vs 0.703), is kept.
- **The Wordle RL policy ties a strong baseline.** Trained REINFORCE beats random decisively
  (p=2.7e-18, d=4.27) but is statistically indistinguishable from an info-aware greedy baseline
  (p=0.93). It is a real RLVR demonstration, not a claim of RL superiority.

## 2. Scope and confounding caveats we surface, not hide

- **The 4 counterfactual methods disagree by ~1,459×** because they measure different scopes: the
  macro synthetic control captures whole-economy output loss (~$230B, which brackets the documented
  $210–235B Tōhoku headline), while the supply-graph MC/do-calculus methods capture only the modelled
  auto-supplier subset, and the AR(1) FRED method captures only the oil channel. **Do not average
  across scopes.** The oil-channel estimate is further **confounded** by the concurrent 2011 Libya /
  Arab-Spring oil rally — stated in the receipt, not buried.
- **Conformal coverage** assumes exchangeability; transitions are shuffled across trajectories before
  splitting to approximate it, but within-trajectory autocorrelation is a known caveat (in the
  receipt's `method` field). Coverage is reported on a **disjoint** held-out test set, not the
  circular calibration set.
- **War-room scenarios are conditional.** The operator supplies severity / pre-event price / duration;
  the system reports second-order industrial effects *given* those. It does not predict whether a
  chokepoint closes. Sector loss bands are published-agency ranges interpolated by a deterministic
  heuristic — "order of magnitude", not decision-quality forecasts.

## 3. Still blocked or pending (never faked around)

- **OpenRouter key revoked.** The LLM analyst A/B (Brier-scored) and the multi-judge panel α are
  **built but RERUN-PENDING**. No live-panel number is cited as current (ledger C1–C3, B1). The
  adversarial gauntlet therefore tests injection only against data-handling surfaces (RAG, MCP tool
  args, string action fields); live-LLM instruction-override resistance is explicitly not claimed.
- **The full RL leaderboard is RERUN-PENDING** (WP6.3). Real checkpoints exist; the head-to-head
  ranking requires fixing the buffer↔env action-taxonomy mismatch, then evaluating on ≥20 seeds with
  correct paired-by-seed statistics. The RL-vs-scripted narrative follows that evidence (ledger D7).
- **No DPO-fine-tuned judge exists** — every DPO run crashed (`float8` / `None` gradients). Real
  QLoRA re-train is WP7.4 (ledger D2). Struck until it trains and is evaluated honestly.
- **The twin savings figure is STRUCK** — the committed receipt records `exit_code:-9` (process
  killed). Re-run to completion is WP9 (ledger D3).
- **Notebook outputs are RERUN-PENDING** (WP7.1) — the pruned notebook set is committed but not yet
  executed with outputs.
- **War-room `war_room_validation.json`** has a malformed JSON escape and is not machine-gated until
  re-serialized (ledger D6) — the run is real; the receipt file needs a re-emit.

## 4. Data we wanted but do not have

- **ACLED** (conflict events) needs institutional access — we use GDELT-Conflict instead.
- **Full SAR** imagery for port congestion costs money — we use free RGB satellite + vision models.
- **Reddit OAuth** wasn't approved in the build window — we use a tech-news ticker.

## 5. The discipline

Every headline number is intended to be sha256-replayable from a committed receipt and is checked by
`scripts/verify_claims.py`, which fails CI if any claim lacks backing. If you find a claim that is
not receipt-backed, it is a bug — file it and we fix the receipt or retract the claim.

**These honesty admissions are the headline.** Every team pitches a model; we pitch a system that
can be audited, negatives and all.
