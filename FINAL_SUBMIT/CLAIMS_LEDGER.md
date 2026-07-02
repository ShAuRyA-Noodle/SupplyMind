# Claims Ledger — SupplyMind headline claims → status

This ledger is the receipt for the **docs de-fake pass (P2.2, 2026-07-02)**. It maps every
judge-facing headline claim to its current status against the 12-agent audit
(`_audit/audit_*.json`) and on-disk reality.

**Status legend**
- **VERIFIED** — traces to a real committed artifact / real run; safe to cite (receipt path given).
- **CORRECTED** — was wrong/misleading; the doc now states the true value.
- **STRUCK** — fabricated or unmakeable-real; removed from judge-facing claims entirely.
- **RERUN-PENDING** — the underlying code is real, but the *number* came from a synthetic/invalid
  run and must be regenerated (P1.3/P1.4/P1.6) before it may be cited again.

> Scope note: the Wave-1/2 pass edited **markdown/HTML only**. **Wave-3 (2026-07-02)** additionally
> struck the inherited fabricated stats in `notebooks/archive/07_HACKATHON_TRAINING.ipynb`
> (cells 0/11/12/17) — see the Wave-3 delta at the end of this file. Fabricated JSON receipts under
> `FINAL_SUBMIT/receipts/` and `tests/receipts/` are owned by the receipts/scripts waves (P1.4)
> and are NOT yet fixed — e.g. `master_audit_summary_pass28_v9_FINAL.json` still contains the
> struck "248-of-250" string. Docs no longer cite those numbers as true.

---

## A. Fabricated statistics (STRUCK — generators deleted or invalid)

| # | Claim | Status | Basis / where |
|---|---|---|---|
| A1 | RAP-XC vs MaskablePPO **Wilcoxon significance (p≈1e-18, large Cohen's d)** (all-3-tasks) | **STRUCK** | The bootstrap-CI + Wilcoxon-pairwise leaderboard generator scripts sorted two synthesized samples and called them "paired" → fabricated significance. Both scripts **deleted** (confirmed missing on disk). audit_5. **W3:** every doc/notebook citation of the exact p-value / Cohen's d struck in place (see Wave-3 delta below). |
| A2 | REINFORCE Wordle **100% solve · p=2.71 × 10⁻¹⁸ · d=4.28** (also quoted as 9.39e-35 / 1.87e-34 / 6.6e-35 elsewhere) | **RERUN-PENDING** | Training in `final_real_reinforce_wordle_v2.py` is real; the p/d headlines are inconsistent across docs and notebooks have **zero executed outputs** — never actually run. Needs honest paired-by-seed re-run. audit_4 |
| A3 | Adversarial **269-of-269 = 100% blocked** (and README "257/257") | **STRUCK → 174/174** | Gauntlet counted every non-crashing call "safe" in both branches, never passed injection payloads to 2 tools, added 19 as a constant → 100% by construction. Honest re-count = **174/174 real executed attacks** (nb13 §1 rewrite). audit_4 nb13 §1 / audit_5 pass28. **W3:** the rigged 100%-gauntlet citations converted to the honest count across owned docs. |
| A4 | Process-supervision **2735× variance amplification** | **STRUCK** | Hardcoded 4-row demo trajectory + hardcoded receipt value; Wordle feedback in it is factually wrong. audit_4 §5 / audit_5 pass26 |
| A5 | **0.9001 conformal coverage** (α=0.10) | **RERUN-PENDING** | Committed `pass27`/`pass28` conformal receipts computed on `rng.normal()` noise labeled model-NLL. Real path = `calibrate_conformal_from_harvest.py` (loads real transitions) — must regenerate. audit_5 |
| A6 | **4-method causal counterfactual** / Tohoku **$276B, +18%** replication | **STRUCK** | 3 of 4 "methods" are hardcoded literals (250/263/285), 4th is `np.random.normal()`; `platinum.py` "real FRED" path always hits a hardcoded fallback (CSV column mismatch). audit_5 pass22 / audit_4 §9 / audit_6 |

## B. Absent models / silently-dead features (CORRECTED → ABSENT)

| # | Claim | Status | Basis |
|---|---|---|---|
| B1 | **TabPFN-v2-clf** "7th judge" (old inventory falsely marked it available) | **CORRECTED → ABSENT** | `models/tabpfn-v2-clf/` does not exist; `tabpfn_risk_judge.py` always returns `tabpfn_unavailable`. audit_7 |
| B2 | **Snowflake-Arctic-Embed-L** / "2-embedder ensemble" | **CORRECTED → ABSENT** | `models/snowflake-arctic-embed-l/` does not exist; ensemble silently returns `mxbai_only`. audit_7 |
| B3 | **BGE-reranker-v2-m3** / "reranker +5pp on hard" | **CORRECTED → ABSENT** | `models/bge-reranker-v2-m3/` does not exist; reranker path is a dead fallback. audit_7 |
| B4 | "**13 foundation models** all local, zero API at inference" | **CORRECTED** | 3 dirs absent (B1–B3); LLMs are Ollama/GGUF not `models/` dirs; embedder downloads from HF hub at runtime; judge panel is a network dep. audit_7/audit_8 |
| B5 | Snowflake RAG **nDCG@10 = 0.971** headline | **RERUN-PENDING** | Model dir absent → not reproducible today. mxbai/BGE-M3 numbers stand. audit_7 |

## C. Judge-panel / ensemble overcounts (CORRECTED)

| # | Claim | Status | Basis |
|---|---|---|---|
| C1 | **25-judge ensemble / 12 OpenRouter frontier** panel | **CORRECTED** | Only **6** frontier judges appear in cache + usage logs; the 12-/25-extension slugs have zero execution evidence and several are likely invalid IDs. audit_7 |
| C2 | `openrouter_liveness.json` presented as liveness proof | **RERUN-PENDING** | Committed receipt shows **4/14 OK** with 400 "invalid model ID" errors + a dead `ling-2.6-1t` slug. Regenerate. audit_7 |
| C3 | Krippendorff **α = 0.750** (headline) | **CORRECTED** | Cherry-picked **2-judge** sub-panel; raw **3-judge α = 0.210**. Docs now state both. audit_8 / `Sleep-Token-ARCHIVE/_dump/FAILURE_TABLE.md` |
| C4 | Cohen κ **0.747** (Qwen×Mistral), majority-vote **69.2%** | **VERIFIED** | `R4_DANGEROUS_V2_ABLATION.json` (genuine v3 result). |

## D. Receipts / test counts (CORRECTED)

| # | Claim | Status | Basis |
|---|---|---|---|
| D1 | "**20 receipts live** / every headline has a one-bash-command receipt" | **CORRECTED** | 14 of 20 `receipts_v2/*.yaml` are `<pending-first-run>` stubs (`match:false`, `exit_code:-1`) with unimportable reproduce commands. audit_8 |
| D2 | **DPO-fine-tuned Qwen-2.5-3B judge** | **STRUCK** | Never trained — all DPO runs crashed (`float8_e8m0fnu` AttributeError) or trained with `None` gradients. Config/script files exist; no model does. audit_8 |
| D3 | Twin **$135.5M savings** receipt (✅) | **STRUCK / RERUN-PENDING** | On-disk `V5_Twin_savings_gt_zero` receipt records `exit_code:-9`, `match:false`. Do not cite until re-run. audit_8 |
| D4 | "**275 passing tests**" (badge + prose) | **CORRECTED → 184/184** | Now **184 pass** CPU-only (~115s, re-verified 2026-07-02 W3); the 8 adversarial tests that had failed on a stale `v3_arcadia` path were revived in P0.1. "275" had counted collected, not passed. README badge updated 176→184. audit_3 |
| D5 | "**128 sha256-stamped receipts**" as blanket proof | **CORRECTED (caveat)** | A subset are fabricated (§A) or stubs (D1). Docs now flag "under audit". |
| D6 | Ensemble Brent **8/8 within ±30%, median 3.3%** | **RERUN-PENDING** | `validate_ensemble_brent.py` backtests on a synthetic sinusoid+AR(1) series, not real FRED Brent. audit_5 |
| D7 | War-room **8/8 = 100%** risk-band backtest | **CORRECTED (caveat)** | Generator (`validate_war_room.py`) is real, but n=8 hand-curated events graded against the model's own classification — a smoke test, not a benchmark. |
| D8 | Cross-corpus α **0.5669 / drift 0.0233** | **RERUN-PENDING** | `compute_cross_corpus_alpha.py` reads `.openrouter_cache/cross_corpus`, which is absent. audit_7 |
| D9 | "**248-of-250 features demonstrated** (99.2%)" | **STRUCK** | Hardcoded constant (`PROJECT_TOTAL_DEMONSTRATED = 248`), not a measured count. audit_4 §12 / audit_5 pass22 |
| D10 | HetGAT **+7.77/+12.15/+10.03%** | **RERUN-PENDING** | R6_GCN/HetGAT receipt is a `<pending-first-run>` stub. audit_8 |

## E. Data provenance (CORRECTED — DATASET_CARD.md)

| # | Claim | Status | Basis |
|---|---|---|---|
| E1 | EM-DAT "snapshot 2024-01" | **CORRECTED** | `emdat_public_2000_2026.xlsx`, 16,811 rows, `Last Update` 2025-12-20. audit_6 |
| E2 | FRED supply-chain "2 files" | **CORRECTED → 1** | Only `fred_truck_transport.csv`; `fred_supply_chain_pressure.csv` 404'd. audit_6 |
| E3 | NOAA "realtime" | **CORRECTED** | Static June-2024 archive zip; live `CurrentStorms.json` fetch failed. audit_6 |
| E4 | `frbny_supply_chain.pdf` policy paper | **CORRECTED** | It is an HTML error page, not a PDF. audit_6 |
| E5 | OpenFlights / WTO zip / pink-sheet / NOAA zip in corpus | **CORRECTED (de-listed)** | Downloaded but consumed by no code. audit_6 |
| E6 | GFW "key authenticated" / 200 OK | **CORRECTED** | Last live probe returned 503; receipt counted it ok:true. audit_3 |
| E7 | RAG corpus 6,483 chunks incl. World Bank | **CORRECTED** | World Bank ingestion silently produced **0** chunks (list-vs-dict bug). audit_6 |

## F. Verified — safe to cite (unflagged)

| # | Claim | Status | Receipt |
|---|---|---|---|
| F1 | RAG **mxbai P@1 = 0.962, MRR = 0.978** (6,483-chunk corpus) | **VERIFIED** | `versions/v3_arcadia/results/R5_GRANITE.json` |
| F2 | GNN arrival-time MAE **−48/−49/−64%** vs MLP | **VERIFIED** | `R6_PROVIDER_V2.json` |
| F3 | MaskablePPO masking lift **+26.8%** (matches Huang 2020 range) | **VERIFIED** | `R6_GETHSEMANE_MASKING_ABLATION.json` |
| F4 | Per-horizon conformal deviation **0.024** on WTI | **VERIFIED** | `R6_AQUA_REGIA_V2.json` |
| F5 | PPO vs random/greedy CI95 non-overlapping (all 3 tasks) | **VERIFIED** | `R6_EUCLIDIAN.json` |
| F6 | RAP-XC **3.14M params, BC loss 5.62→0.23** (training real) | **VERIFIED** | `versions/v5_phoenix/experiments/rap_xc_v1/rapxc.pt` |
| F7 | EM-DAT → 1500-event crisis library v2 (deterministic severity) | **VERIFIED** | `scripts/crisis_library/cook_v2.py` → `crisis_library_v2.json` |
| F8 | HF Space live rollout 20/20 steps 200 OK; FRED real-Brent ingest | **VERIFIED** | pass26/27/28 live-rollout + pass28 K1 blocks |
| F9 | ONNX roundtrip 4/4 verified (opset 17) | **VERIFIED** | `onnx_roundtrip.json` |

---

## Residual items for later waves (not fixable in docs-only pass)

1. **JSON receipts still contain struck numbers** — e.g. `FINAL_SUBMIT/receipts/master_audit_summary_pass28_v9_FINAL.json` (`"248-of-250 = 99.2%"`), the fabricated leaderboard receipts (bootstrap-CI + Wilcoxon-pairwise) and conformal receipts. Owned by P1.4 (receipts) and P1.3 (benchmark). *(The receipt JSON filenames still carry the struck stems on disk; they are not owned by the docs wave.)*
2. **`scripts/run_all.py` path bug** — the canonical verifier reports MISSING for 14/14 checks until `ROOT/'v3_arcadia'` → `ROOT/'versions'/'v3_arcadia'` lands (scripts wave).
3. **PREPRINT_V5.md DPO-judge claim** — lives in `versions/v5_phoenix/docs/` (not in the docs-wave ownership); must be struck by whoever owns `versions/**`.
4. **README broken/fragile links**: the `FAILURE_TABLE.md` lineage moved during `_dump/` cleanup → now at `Sleep-Token-ARCHIVE/_dump/FAILURE_TABLE.md` (in-repo pointer `_dump/POINTER.md`); `FINAL_SUBMIT/Blog.MD` (case-sensitive on GitHub — reference corrected). The model-fetch script REPRODUCE.md referenced does **not exist** (P1.6 fetcher pending; local embedders pull from HF hub at runtime). **W3:** FAILURE_TABLE pointers + the missing-model-fetch note updated across owned docs.
5. **`verify_claims.py` (P2.2)** must be built to machine-check every remaining "PRESENT" row in the FEATURE_INVENTORY files against disk before the aggregate counts are trusted.

---

## Wave-3 docs-residue delta (2026-07-02)

Wave-3 swept the Wave-1/2 blockers left in judge-facing docs. All edits are working-tree only (no commits).

**Fabricated fingerprints purged from owned files** (`README.md`, `docs/**`, `FINAL_SUBMIT/**/*.md` +
`*.html`, `notebooks/archive/07_HACKATHON_TRAINING.ipynb`): the six audited fingerprints — the rigged
100%-gauntlet attack ratio, the two deleted leaderboard script/receipt stems (bootstrap-CI +
Wilcoxon-pairwise), the fabricated Wilcoxon p-value and Cohen's-d literals, and the non-existent
model-download shell script — now return **0 raw hits** under a literal grep. (This delta deliberately
does not re-print the exact literals, so the grep stays clean.) Struck claims are preserved as honest
disclosures, reworded so they no longer re-quote the exact fabricated fingerprints.

| Change | Files touched |
|---|---|
| Fabricated leaderboard stats (Wilcoxon p / Cohen's d / bootstrap CI95, **A1**) struck in place | HACKATHON_README, JUDGE_FAQ_30, JUDGE_OBJECTION_HANDBOOK, VICTORY_CALCULUS, THREE_THEME_HAT_TRICK, PITCH_DECK, HYPERMODE_DEEP_AUDIT_PASS22, BENCHMARK_REPORT, HONEST_LIMITATIONS, REPRODUCE, PRESENTATION_FINAL_CHECKLIST, BRUTAL_BREAKDOWN_19PART, FINAL_SUBMIT_INDEX, ALL_250_FEATURES_LIVE_PROOF, FEATURE_INVENTORY_DI, FEATURE_INVENTORY_UBB, MASTER_FEATURE_USECASE_MAP_250 |
| Rigged 100%-gauntlet attack overcount → honest **174/174 real executed attacks** (**A3**, nb13 §1) | PASS28_HYPERMODE_FINAL, SUBMISSION_PACKAGE_FINAL |
| Moved-notebook deep-links repointed to `notebooks/archive/…` (04/05/07/10/11/12) + `(archived)` annotations | EXEC_SUMMARY_ONE_PAGE, HACKATHON_README, HACKATHON_BLOG_FINAL, PASS28_HYPERMODE_FINAL, SUBMISSION_PACKAGE_FINAL, JUDGE_FAQ_30, JUDGE_OBJECTION_HANDBOOK, FEATURE_INVENTORY_UBB, docs/v3/FINAL_DEMO.md, docs/v4/AUDIT_PLAN.md |
| `FAILURE_TABLE.md` pointers → `Sleep-Token-ARCHIVE/_dump/FAILURE_TABLE.md` (in-repo `_dump/POINTER.md`) | README.md, HACKATHON_README, FEATURE_INVENTORY_UBB, CLAIMS_LEDGER (this file) |
| Missing model-fetch script note (does not exist; embedders pull from HF hub at runtime; P1.6 fetcher pending) | REPRODUCE, CLAIMS_LEDGER (this file) |
| Inherited fabricated stats struck (p / Cohen's d / 25-judge / 4-method causal) + `[ARCHIVED/UNVERIFIED]` banners | `notebooks/archive/07_HACKATHON_TRAINING.ipynb` cells 0, 11, 12, 17 |

**Not touched — logged for P2.2 / other waves (avoids concurrent-edit collisions):**
- `FINAL_SUBMIT/receipts/**` + `FINAL_SUBMIT/plots/**` still hold struck receipt JSON filenames + numbers → **evidence-refresh wave (P1.4/P1.3)**.
- Judge-panel "**25-judge / 12-frontier**" overcounts (**C1**) and "**4-method causal / Tōhoku ±18%**" (**A6**) still read as live claims in several docs (PITCH_DECK, HACKATHON_README, FEATURE_INVENTORY*, etc.) → **integration / models-openrouter sweep**.
- `notebooks/README.md` and `notebooks/archive/10_PRO_COLAB_KILLSHOT.ipynb` still carry struck tokens → **notebooks agent** (only `notebooks/archive/07` is in the docs-wave ownership).
- `Makefile`, `scripts/**`, `versions/**`, `CLAUDE.md`, `_audit/**` token hits → code / legacy / audit-source files, out of docs-wave ownership.
