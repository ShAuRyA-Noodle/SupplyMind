# SupplyMind — Final Submit

**Regenerated 2026-07-15 from committed receipts** (WP8.2). This folder is the judge-facing
package. Everything here is receipt-backed or honestly marked pending; the machine gate
[`scripts/verify_claims.py`](../scripts/verify_claims.py) fails CI if any headline claim lacks
backing.

> SupplyMind is a live supply-chain-risk decision copilot on an OpenEnv-compliant deterministic
> simulation. See the product thesis in [`../PRODUCT_THESIS.md`](../PRODUCT_THESIS.md) and the full
> project README in [`../README.md`](../README.md).

---

## Start here (authoritative, receipt-backed)

| Doc | What |
|---|---|
| [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md) | **Every headline claim → VERIFIED / CORRECTED / STRUCK / RERUN-PENDING**, with its receipt path. Machine-checked. |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Every model/component and its real measured result. |
| [`DATASET_CARD.md`](DATASET_CARD.md) | Data sources, splits, provenance (per-file SHA-256). |
| [`HONEST_LIMITATIONS.md`](HONEST_LIMITATIONS.md) | Honest negatives, scope caveats, and what's still pending. |
| [`../README.md`](../README.md) | Full project README (engine, endpoints, quick start). |

Verify everything yourself:

```bash
pip install -e . && pytest tests/ -q      # 184/184
python scripts/verify_claims.py           # every headline number → receipt; exits nonzero if not
```

---

## The receipt-backed headline numbers

Each is asserted against its receipt by `verify_claims.py` (values copied from the receipts):

| Result | Number | Receipt |
|---|---|---|
| Adversarial gauntlet | **318/318 blocked (100%), 0 breaches** | [`adversarial_gauntlet_REAL.json`](../tests/receipts/adversarial_gauntlet_REAL.json) |
| Split-conformal coverage | 90% nominal → **90.03%** held-out | [`conformal_REAL.json`](../tests/receipts/conformal_REAL.json) |
| Brent ensemble backtest (real FRED) | mean MAPE **7.1%** over 8 events | [`ensemble_brent_REAL.json`](../tests/receipts/ensemble_brent_REAL.json) |
| Federated learning (honest negative) | FedAvg AUC **0.7218 = centralized** | [`fedavg_REAL.json`](../tests/receipts/fedavg_REAL.json) |
| Ghost-model triage | ensemble **RETIRE**, reranker **RETIRE**, TabPFN **KEEP** | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| RAG retrieval (mxbai) | **P@1 0.962, MRR 0.978, nDCG@10 0.961** | [`ghost_models_eval_REAL.json`](../tests/receipts/ghost_models_eval_REAL.json) |
| RAG World-Bank fix | **0 → 20 WB chunks, 6,602 total** | [`rag_recook_REAL.json`](../tests/receipts/rag_recook_REAL.json) |
| 4-method causal counterfactual | 4 real methods; **$230B** brackets the $210–235B Tōhoku headline | [`counterfactual_4method_REAL.json`](receipts/counterfactual_4method_REAL.json) |
| Wordle REINFORCE | beats random (p=2.7e-18) but **ties greedy** (p=0.93) | [`pass27_B_real_episodic_bootstrap.json`](receipts/pass27_B_real_episodic_bootstrap.json) |
| TFT forecaster (real WTI) | **MAE p50 $7.83** | [`tft_real_metrics.json`](receipts/tft_real_metrics.json) |
| Real SHAP | NOAA group ≈ **60%** attribution | [`shap_real.json`](receipts/shap_real.json) |
| Test suite | **184/184 pass**, exit 0 | [`test_suite_grand_total.json`](../tests/receipts/test_suite_grand_total.json) |

Plots: [`plots/`](plots) — [`brent_backtest.png`](plots/brent_backtest.png),
[`conformal_coverage.png`](plots/conformal_coverage.png). All receipts: [`receipts/`](receipts).

---

## What's honest about this submission

- **Negatives are reported as measured**: federated learning ties centralized; the embedder
  ensemble is worse than mxbai alone; the RL policy ties a strong greedy baseline. See
  [`HONEST_LIMITATIONS.md`](HONEST_LIMITATIONS.md).
- **Blocked items are labeled, not faked**: the OpenRouter judge panel and analyst A/B are
  RERUN-PENDING on a revoked key; the full RL leaderboard is RERUN-PENDING on the benchmark run.
- **Fabricated predecessors are struck**, and their real rebuilds are cited — the diff is documented
  row-by-row in [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).

## Legacy documents

Other markdown files in this folder are **pre-audit artifacts** (marketing decks, feature maps,
FAQ). Each carries a `PRE-AUDIT ARTIFACT` banner and had its struck numbers marked. They are
superseded by the four authoritative docs above; treat any number there as struck unless it also
appears, receipt-backed, in [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md).
