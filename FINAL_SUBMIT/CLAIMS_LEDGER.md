# Claims Ledger — SupplyMind headline claims → status

**Regenerated 2026-07-15 from the real receipts on disk** (Wave-6 docs pass, WP8.2). This is the
single source of truth for every judge-facing headline number. It supersedes all pre-audit copies.

Every **VERIFIED** row is machine-checked by [`scripts/verify_claims.py`](../scripts/verify_claims.py):
that script loads the cited receipt, asserts the number is actually in it, and **exits non-zero if
any claim lacks backing**. Run it yourself:

```bash
python scripts/verify_claims.py            # human report
python scripts/verify_claims.py --json     # machine summary
```

**Status legend**
- **VERIFIED** — machine-checked against a committed receipt by `verify_claims.py`. Safe to cite.
- **VERIFIED (v3)** — real committed receipt on disk from the v3 result set; not in the automated
  gate but present and inspectable.
- **CORRECTED** — the earlier number was wrong/misleading; the true value is stated here.
- **STRUCK** — fabricated or un-makeable-real; removed from every judge-facing claim.
- **RERUN-PENDING** — the code is real but the number is not yet producible (blocked on the
  OpenRouter key, on the full benchmark run, or on a receipt re-emit). Honestly not cited as fact.

> **What changed since the 2026-07-02 audit:** eight capabilities that were then "STRUCK / being
> re-run" have since been **rebuilt for real and measured** — the adversarial gauntlet (R7), the
> ghost models (R9, *retired by measurement*), the Brent backtest (R10), split-conformal (R11),
> federated learning (R12, *an honest negative*), the RAG World-Bank fix (R17), the 4-method causal
> counterfactual (R6), and the Wordle REINFORCE significance. Their fabricated predecessors are
> struck below; their real receipts are cited. The honesty of the negative results (FedAvg does not
> beat centralized; the embedder ensemble is worse than mxbai alone; REINFORCE ties the greedy
> baseline) is deliberate — a disappointing measured number is worth more than a flattering fake.

---

## A. Machine-verified headline claims (checked by `verify_claims.py`)

| # | Claim | Value (from receipt) | Status | Receipt |
|---|---|---|---|---|
| V1 | Adversarial gauntlet | **318 real attacks executed, 318 blocked (100%), 0 breaches**; 8/8 benign controls accepted (0% false-positive) | **VERIFIED** | [`tests/receipts/adversarial_gauntlet_REAL.json`](../tests/receipts/adversarial_gauntlet_REAL.json) |
| V2 | Split-conformal coverage | α=0.10 → **90.03% empirical coverage on a disjoint held-out test set** (gap +0.00025); α=0.05→95.14%, α=0.20→79.35% | **VERIFIED** | [`tests/receipts/conformal_REAL.json`](../tests/receipts/conformal_REAL.json) |
| V3 | Brent ensemble backtest | **mean MAPE 7.12%** (median 6.42%) over 8 real crisis events, real FRED `DCOILBRENTEU` walk-forward; members chronos 7.30 / timesfm 7.21 / tabpfn 8.40 | **VERIFIED** | [`tests/receipts/ensemble_brent_REAL.json`](../tests/receipts/ensemble_brent_REAL.json) |
| V4 | Federated learning (FedAvg) | **honest negative: FedAvg AUC 0.7218 == centralized 0.7218 (Δ −0.0)** on 180,519 real DataCo orders across 5 regions | **VERIFIED** | [`tests/receipts/fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json) |
| V5 | Ghost-model triage | **embedder ensemble RETIRE** (P@1 −0.076 vs mxbai), **reranker RETIRE** (P@1 −0.038), **TabPFN KEEP** (AUC 0.7377 vs logreg 0.7028) — each measured on real data | **VERIFIED** | [`tests/receipts/ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| V6 | RAG retrieval (mxbai) | **P@1 0.962, MRR 0.978, nDCG@10 0.961** on the 6,483-chunk real corpus (53 crisis queries) | **VERIFIED** | [`tests/receipts/ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| V7 | RAG World-Bank fix | **0 → 20 World-Bank chunks**, 6,602 total; WB queries P@1 **0 → 1.0** (list-vs-dict bug fixed) | **VERIFIED** | [`tests/receipts/rag_recook_REAL.json`](../tests/receipts/rag_recook_REAL.json) |
| V8 | 4-method causal counterfactual | Tōhoku 2011: **4 real methods run** (paired-bootstrap MC, Abadie synthetic control, AR(1) FRED Brent, do-calculus); they **disagree by scope** — the macro synthetic control ($230B) brackets the documented $210–235B whole-economy headline; supply-graph methods sit below it | **VERIFIED** | [`FINAL_SUBMIT/receipts/counterfactual_4method_REAL.json`](receipts/counterfactual_4method_REAL.json) |
| V9 | Wordle REINFORCE significance | trained policy **beats random** (Wilcoxon p=2.7×10⁻¹⁸, Cohen d=4.27, bootstrap CI excludes 0) but **ties the info-aware greedy baseline** (p=0.93) — a real per-episode paired test | **VERIFIED** | [`FINAL_SUBMIT/receipts/pass27_B_real_episodic_bootstrap.json`](receipts/pass27_B_real_episodic_bootstrap.json) |
| V10 | TFT quantile forecaster | **MAE p50 $7.83** on real WTI (`DCOILWTICO`), 90,602 params, 14-day horizon | **VERIFIED** | [`FINAL_SUBMIT/receipts/tft_real_metrics.json`](receipts/tft_real_metrics.json) |
| V11 | Real SHAP explanation | SHAP on the trained BC checkpoint over 200 states: **NOAA feature group ≈60% importance**, node features ≈22% | **VERIFIED** | [`FINAL_SUBMIT/receipts/shap_real.json`](receipts/shap_real.json) |
| V12 | Real FRED Brent anchoring | **8/8 crisis events anchored to real FRED Brent daily; `no_synthetic_substitution: true`** | **VERIFIED** | [`FINAL_SUBMIT/receipts/pass28_K1_fred_brent_real.json`](receipts/pass28_K1_fred_brent_real.json) |
| V13 | Test suite | **184/184 pass, CPU-only, exit 0** (~125s) | **VERIFIED** | [`tests/receipts/test_suite_grand_total.json`](../tests/receipts/test_suite_grand_total.json) |

## B. Real v3 receipts (on disk; inspectable, not in the automated gate)

| # | Claim | Status | Receipt |
|---|---|---|---|
| F1 | GNN arrival-time MAE −48/−49/−64% vs MLP | **VERIFIED (v3)** | [`FINAL_SUBMIT/receipts/R6_PROVIDER_V2.json`](receipts/R6_PROVIDER_V2.json) |
| F2 | Per-horizon split-conformal deviation 0.024 on WTI | **VERIFIED (v3)** | [`FINAL_SUBMIT/receipts/R6_AQUA_REGIA_V2.json`](receipts/R6_AQUA_REGIA_V2.json) |
| F3 | RAG mxbai P@1 0.962 (6,483-chunk corpus) — v3 result, re-confirmed by V6 above | **VERIFIED (v3)** | [`FINAL_SUBMIT/receipts/R5_GRANITE.json`](receipts/R5_GRANITE.json) |
| F4 | Cohen κ 0.747 (Qwen × Mistral), majority-vote 69.2% vs ground truth | **VERIFIED (v3)** | [`FINAL_SUBMIT/receipts/R4_DANGEROUS_V2_ABLATION.json`](receipts/R4_DANGEROUS_V2_ABLATION.json) |
| F5 | ONNX roundtrip 4/4 verified (opset 17) | **VERIFIED (v3)** | [`FINAL_SUBMIT/receipts/onnx_roundtrip.json`](receipts/onnx_roundtrip.json) |
| F6 | EM-DAT (16,811 rows) → 1,500-event crisis library v2 (deterministic, no-LLM severity) | **VERIFIED** | [`scripts/crisis_library/cook_v2.py`](../scripts/crisis_library/cook_v2.py) → `crisis_library_v2.json` |

## C. Fabricated statistics — STRUCK (generators deleted or invalid)

| # | Claim | Status | Basis / replacement |
|---|---|---|---|
| A1 | RAP-XC vs MaskablePPO **Wilcoxon p≈3.9×10⁻¹⁸ / Cohen d≈2.73**, "+15–35% leaderboard win" | **STRUCK** | The bootstrap-CI + Wilcoxon-pairwise leaderboard scripts sorted two independently synthesized samples and called them "paired" — a meaningless test. Scripts deleted. Real paired-by-seed benchmark is WP6.3 (RERUN-PENDING, D-series). |
| A2 | REINFORCE Wordle p=9.39×10⁻³⁵ / 1.87×10⁻³⁴ / 6.6×10⁻³⁵ (inconsistent across docs) | **STRUCK → replaced by V9** | The inconsistent p-values were a fabrication flag. The real per-episode paired test (V9) gives p=2.7×10⁻¹⁸ vs random and an honest **tie vs the greedy baseline**. Legacy-doc occurrences struck in place. |
| A3 | Adversarial **269/269** (and README "257/257") = 100% | **STRUCK → replaced by V1** | The old gauntlet counted every non-crashing call "safe" in both branches and added attacks as a constant. Real single-assertion count = **318/318** (V1). |
| A4 | Process-supervision **2735× variance amplification** | **STRUCK** | Hardcoded 4-row demo trajectory + hardcoded receipt value; the Wordle feedback in it was factually wrong. |
| A5 | **0.9001 conformal coverage** on `rng.normal()` noise | **STRUCK → replaced by V2** | The committed pass27/28 conformal receipts used Gaussian noise labeled "model-NLL". Real split-conformal on real transitions gives **90.03%** on a held-out test set (V2). |
| A6 | **4-method causal counterfactual** with 3 hardcoded literals + 1 `np.random.normal()`; "$276B Tōhoku, +18%" | **STRUCK → replaced by V8** | The old `platinum.py` path always hit a hardcoded fallback. The real 4-method run (V8) computes each method and honestly reports scope disagreement instead of a single flattering number. |
| A7 | Ensemble Brent **"8/8 within ±30%, median 3.3%"** on a synthetic sinusoid+AR(1) series | **STRUCK → replaced by V3** | Real walk-forward on real FRED Brent gives mean MAPE 7.12% (V3). |
| A8 | **248-of-250 features demonstrated (99.2%)** | **STRUCK** | A hardcoded constant (`PROJECT_TOTAL_DEMONSTRATED = 248`), not a measured count. Removed from judge-facing docs. |

## D. Absent-then-measured, and still-pending

| # | Claim | Status | Basis |
|---|---|---|---|
| B1 | TabPFN-v2-clf "7th judge" | **CORRECTED → KEEP (measured)** | The model was obtained (29 MB ckpt on disk) and measured: AUC 0.7377 vs logreg 0.7028 on DataCo late-delivery — it **adds real predictive signal** (V5). The 6-judge OpenRouter LLM-panel comparison remains BLOCKED-ON-KEY. |
| B2 | Snowflake-Arctic-Embed-L "2-embedder ensemble" | **CORRECTED → RETIRE (measured)** | Model obtained (1.3 GB on disk) and measured: the ensemble is **worse** than mxbai alone (P@1 −0.076). Retired by evidence, not hidden (V5). |
| B3 | BGE-reranker-v2-m3 "reranker +5pp on hard" | **CORRECTED → RETIRE (measured)** | Model obtained (2.3 GB on disk) and measured: reranking **lowers** P@1 by 0.038 vs mxbai bi-encoder. Retired by evidence (V5). |
| C1 | 25-judge / 12-frontier OpenRouter panel; Krippendorff α | **RERUN-PENDING (KEY)** | `OPENROUTER_API_KEY` is revoked. The panel code exists; live α cannot be re-measured until a key lands (WP7.2). Do not cite a panel α as current. |
| C2 | `openrouter_liveness.json` (4/14 OK) as a liveness proof | **RERUN-PENDING (KEY)** | Regenerate when the key lands. |
| C3 | Krippendorff α = 0.750 headline | **CORRECTED** | Cherry-picked 2-judge sub-panel; raw 3-judge α = 0.210. If cited at all, both must be stated together. Superseded by the key-blocked re-measure (C1). |
| D1 | "20 receipts live / every headline one-bash-command" | **CORRECTED** | The 13 machine-verified claims (§A) each have a real receipt. Other legacy `receipts_v2/*.yaml` stubs are not cited. |
| D2 | DPO-fine-tuned Qwen-2.5-3B judge | **STRUCK** | Never trained (all runs crashed on `float8_e8m0fnu` / `None` gradients). Real QLoRA re-train is WP7.4. |
| D3 | Twin **$135.5M savings** receipt | **STRUCK / RERUN-PENDING** | The on-disk receipt records `exit_code:-9` (process killed). Re-run to completion is WP9 (R22). Not cited as fact. |
| D4 | "275 passing tests" | **CORRECTED → 184/184** | 184 pass CPU-only (V13). "275" counted collected, not passed. |
| D5 | HetGAT +7.77/+12.15/+10.03% | **RERUN-PENDING** | Receipt was a `<pending-first-run>` stub. Real GAT training is WP7.3 (R5). |
| D6 | War-room 8/8 risk-band backtest | **RERUN-PENDING (receipt)** | The run is real but the committed `tests/receipts/war_room_validation.json` has a malformed JSON escape (`\scenarios`) that makes it un-loadable, so it is **not** machine-gated. Re-emit is owned by the receipts wave. The capability (risk band correct 8/8, counterfactual savings per event) is real but the receipt must be re-serialized before it is cited as verified. |
| D7 | Full RL benchmark (checkpoints × ≥20 seeds, correct paired stats) | **RERUN-PENDING** | WP6.3. Fix the buffer↔env action-taxonomy mismatch first, then evaluate the real checkpoints. The RL-vs-scripted narrative is decided by that run, not asserted. |

## E. Data provenance (CORRECTED — see [DATASET_CARD.md](DATASET_CARD.md) and [external_data/PROVENANCE.md](../external_data/PROVENANCE.md))

| # | Claim | Status | Basis |
|---|---|---|---|
| E1 | EM-DAT "snapshot 2024-01" | **CORRECTED** | `emdat_public_2000_2026.xlsx`, 16,811 rows, Last-Update 2025-12-20. |
| E2 | FRED supply-chain "2 files" | **CORRECTED → 1 (+Brent)** | Real daily Brent `DCOILBRENTEU` (9,922 obs) + truck-transport PPI; the dead `GLBLALSCMINDX` pull (404) removed. |
| E3 | NOAA "realtime" | **CORRECTED** | `CurrentStorms.json` is a live feed used for a freshness demo; the RL storm data is the separate `ibtracs_wp.csv`. |
| E4 | `frbny_supply_chain.pdf` policy paper | **CORRECTED** | Was an HTML error page; deleted. Replaced by the real FRBNY GSCPI Staff Report 1017 + BIS + FRBSF papers. |
| E5 | OpenFlights / WTO zip / pink-sheet / NOAA zip in corpus | **CORRECTED (de-listed)** | Downloaded but consumed by no code; removed. |
| E6 | GFW "key authenticated / 200 OK" | **CORRECTED** | Last live probe returned 503 (key authenticated, service returned no data). Surfaced, not counted as OK. |
| E7 | RAG corpus incl. World Bank | **CORRECTED → V7** | WB ingestion produced 0 chunks (list-vs-dict bug); now 20 chunks, 6,602 total (V7). |

---

## Residual (owned by other waves; tracked, not fixable in a docs pass)

1. **OpenRouter-blocked items (C1–C3, part of B1):** the live LLM-judge panel and its α cannot be
   measured until a fresh `OPENROUTER_API_KEY` lands (WP7.2). Code is built; measurement waits.
2. **Full RL benchmark (D7):** WP6.3 — regenerate the leaderboard from the real checkpoints with
   correct paired-by-seed statistics; the RL-vs-scripted story follows that evidence.
3. **DPO judge (D2), twin $135.5M (D3), HetGAT (D5):** WP7.4 / WP9 — train/re-run for real or stay
   struck.
4. **War-room receipt (D6):** re-serialize `war_room_validation.json` (malformed escape) — receipts wave.
5. **Legacy `FINAL_SUBMIT/receipts/**` filenames** still carry some struck stems on disk; they are
   owned by the receipts wave. The docs no longer cite those numbers as fact.
6. **Notebook outputs (WP7.1)** and the **live war-room freshness stamps (WP6.4)** are separate WPs.

*Legacy pre-audit marketing docs in `FINAL_SUBMIT/` carry a `PRE-AUDIT ARTIFACT` banner and had
their struck numbers marked; they are superseded by this ledger and the receipt-backed cards.*
