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

> Scope note: this pass edited **markdown/HTML only**. Fabricated JSON receipts under
> `FINAL_SUBMIT/receipts/` and `tests/receipts/` are owned by the receipts/scripts waves (P1.4)
> and are NOT yet fixed — e.g. `master_audit_summary_pass28_v9_FINAL.json` still contains the
> struck "248-of-250" string. Docs no longer cite those numbers as true.

---

## A. Fabricated statistics (STRUCK — generators deleted or invalid)

| # | Claim | Status | Basis / where |
|---|---|---|---|
| A1 | RAP-XC vs MaskablePPO **Wilcoxon p=3.9 × 10⁻¹⁸, Cohen d=+2.73** (all-3-tasks) | **STRUCK** | `bootstrap_leaderboard.py` + `wilcoxon_pairwise_leaderboard.py` sorted two synthesized samples and called them "paired" → fabricated significance. Both scripts **deleted** (confirmed missing on disk). audit_5 |
| A2 | REINFORCE Wordle **100% solve · p=2.71 × 10⁻¹⁸ · d=4.28** (also quoted as 9.39e-35 / 1.87e-34 / 6.6e-35 elsewhere) | **RERUN-PENDING** | Training in `final_real_reinforce_wordle_v2.py` is real; the p/d headlines are inconsistent across docs and notebooks have **zero executed outputs** — never actually run. Needs honest paired-by-seed re-run. audit_4 |
| A3 | Adversarial **269-of-269 = 100% blocked** (and README "257/257") | **STRUCK** | Gauntlet counted every non-crashing call "safe" in both branches, never passed injection payloads to 2 tools, added 19 as a constant → 100% by construction. audit_4 nb13 §1 / audit_5 pass28 |
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
| C3 | Krippendorff **α = 0.750** (headline) | **CORRECTED** | Cherry-picked **2-judge** sub-panel; raw **3-judge α = 0.210**. Docs now state both. audit_8 / `_dump/FAILURE_TABLE.md` |
| C4 | Cohen κ **0.747** (Qwen×Mistral), majority-vote **69.2%** | **VERIFIED** | `R4_DANGEROUS_V2_ABLATION.json` (genuine v3 result). |

## D. Receipts / test counts (CORRECTED)

| # | Claim | Status | Basis |
|---|---|---|---|
| D1 | "**20 receipts live** / every headline has a one-bash-command receipt" | **CORRECTED** | 14 of 20 `receipts_v2/*.yaml` are `<pending-first-run>` stubs (`match:false`, `exit_code:-1`) with unimportable reproduce commands. audit_8 |
| D2 | **DPO-fine-tuned Qwen-2.5-3B judge** | **STRUCK** | Never trained — all DPO runs crashed (`float8_e8m0fnu` AttributeError) or trained with `None` gradients. Config/script files exist; no model does. audit_8 |
| D3 | Twin **$135.5M savings** receipt (✅) | **STRUCK / RERUN-PENDING** | On-disk `V5_Twin_savings_gt_zero` receipt records `exit_code:-9`, `match:false`. Do not cite until re-run. audit_8 |
| D4 | "**275 passing tests**" (badge + prose) | **CORRECTED → 176/184** | 176 pass CPU-only in ~140s; 8 adversarial tests fail on a stale `v3_arcadia` path (fix in flight); "275" counted collected, not passed. audit_3 |
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

1. **JSON receipts still contain struck numbers** — e.g. `FINAL_SUBMIT/receipts/master_audit_summary_pass28_v9_FINAL.json` (`"248-of-250 = 99.2%"`), the fabricated `bootstrap_leaderboard.json` / `wilcoxon_pairwise_leaderboard.json` / conformal receipts. Owned by P1.4 (receipts) and P1.3 (benchmark).
2. **`scripts/run_all.py` path bug** — the canonical verifier reports MISSING for 14/14 checks until `ROOT/'v3_arcadia'` → `ROOT/'versions'/'v3_arcadia'` lands (scripts wave).
3. **PREPRINT_V5.md DPO-judge claim** — lives in `versions/v5_phoenix/docs/` (not in the docs-wave ownership); must be struck by whoever owns `versions/**`.
4. **README broken/fragile links to fix or restore later**: `_dump/FAILURE_TABLE.md` (moved during `_dump/` cleanup — reference softened); `FINAL_SUBMIT/Blog.MD` (case-sensitive on GitHub — reference corrected). `scripts/download_models.sh` referenced by REPRODUCE.md does not exist.
5. **`verify_claims.py` (P2.2)** must be built to machine-check every remaining "PRESENT" row in the FEATURE_INVENTORY files against disk before the aggregate counts are trusted.
