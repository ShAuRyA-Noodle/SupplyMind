# SupplyMind Notebooks

Colab-facing demo and training notebooks. The **active set** below is the maintained,
code-clean set (every code cell compiles; fakes removed). Older / superseded notebooks live
in [`archive/`](archive/) and are kept for reference only.

Each notebook that imports the repo (`import rl`, `from server...`) begins with a bootstrap
cell that git-clones the repo on Colab and adds the repo root to `sys.path` locally, so the
advertised one-click path works in both environments.

## Active notebooks

| Notebook | Purpose |
|---|---|
| `01_environment_quickstart.ipynb` | Reset/step/inspect the SupplyMind env and run the scripted agent across the easy/medium/hard tasks. |
| `02_training_your_own_agent.ipynb` | Train MaskablePPO on the SupplyMind Gymnasium env and evaluate it. |
| `03_reproducing_benchmarks.ipynb` | Reproduce the backtesting/benchmark numbers against the historical crisis library. |
| `06_trl_training_colab.ipynb` | TRL DPO fine-tune on the 21 real, committed preference pairs (Qwen2.5-0.5B via Unsloth). |
| `08_HACKATHON_FOOLPROOF.ipynb` | Self-contained REINFORCE Wordle training (masking + curriculum + variance reduction) with a genuinely paired eval, plus an optional GRPO micro-finetune using a fixed hidden per-prompt target. |
| `09_LLAMA_GRPO_FOOLPROOF.ipynb` | GRPO recipe on the ungated `unsloth/Llama-3.2-1B` mirror over the Wordle env. |
| `13_MASTER_HACKATHON_FINAL.ipynb` | Master submission notebook: OpenEnv compliance + adversarial defense gauntlet, REINFORCE Wordle, conformal action filter, process-supervision credit, QLoRA hyperparameter sweep, SB3 baseline grid, live data ingest, and an HTML submission certificate. |

## Archived notebooks (`archive/`)

| Notebook | Why archived |
|---|---|
| `04_v3_quickstart_colab.ipynb` | v3 quickstart demo; superseded by `01`/`03`. Dead `FAILURE_TABLE.md` link removed. |
| `05_v4_hormuz_live.ipynb` | v4 Hormuz war-room receipt-check demo tied to `versions/v3_arcadia` result JSONs; superseded. (Nested-quote f-string SyntaxError fixed.) |
| `07_HACKATHON_TRAINING.ipynb` | Earlier hackathon training recipe; superseded by `08`/`09`/`13` (nb13 is the single canonical run). Placeholder repo handle filled. See note below. |
| `10_PRO_COLAB_KILLSHOT.ipynb` | "pass 28 GPU upgrades"; overlaps heavily with nb13 §6/§7. Fakes fixed (dead repo URL, TARGET-in-prompt leak, single-`Observation` unpack bug, gated Meta-Llama → ungated unsloth mirror) but archived as redundant. |
| `11_REAL_DATA_INGEST.ipynb` | Real keyed API ingest (FRED/NewsAPI/NOAA/ACLED/Exa/HF). The fabricated K4 W&B curve was removed; the real ingest also lives in nb13 §10. |
| `12_FRED_BRENT_REFIT.ipynb` | Brent refit — relabeled honestly as a **naive statistical baseline** (the earlier "Chronos+TimesFM+TabPFN ensemble" claim was false); superseded by nb13 §11. |

**Note on `07_HACKATHON_TRAINING.ipynb`:** its submission-summary cells inherited unverified
headline statistics (a fabricated Wilcoxon p-value / Cohen's d / relative-error figure) from the
deleted `bootstrap_leaderboard` / `wilcoxon_pairwise_leaderboard` receipts (CLAIMS_LEDGER A1).
Those were **struck in place** during the Wave-3 docs pass — cells 0/11/12/17 now carry
`[ARCHIVED/UNVERIFIED]` banners and no longer print the fabricated numbers. The notebook is
archived and superseded by `13_MASTER_HACKATHON_FINAL.ipynb` regardless.
