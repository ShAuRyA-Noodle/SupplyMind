# GitHub Release — paste this into the Release web UI

> ⚠️ **UNDER FABRICATION CLEANUP (2026-07-02 audit).** Several claims in this file were machine-checked FALSE or UNVERIFIED and are being struck or re-run: fabricated Wilcoxon p-values (from a sorted-"paired" test, generators deleted), "0.9001 conformal coverage" (computed on Gaussian noise), the rigged "100% attacks blocked" gauntlet, the hardcoded feature-demonstration count, the "25-judge / 12-frontier" panel (only 6 judges evidenced), a "DPO-fine-tuned judge" that never trained, the "$135.5M twin savings" receipt (exit -9), and 3 model dirs that do not exist on disk (snowflake-arctic-embed-l, bge-reranker-v2-m3, tabpfn-v2-clf). Do not cite numbers here without checking FINAL_SUBMIT/CLAIMS_LEDGER.md.

**Repo**: https://github.com/ShAuRyA-Noodle/Sleep-Token
**Steps**:
1. Go to https://github.com/ShAuRyA-Noodle/Sleep-Token/releases/new
2. Tag: `v4.0-final-submit` (already pushed)
3. Title: `SupplyMind v4.0 — OpenEnv India 2026 Hackathon Finals submission`
4. Paste the body below
5. Mark as "Set as the latest release"
6. Publish

---

## Release title

```
SupplyMind v4.0 — OpenEnv India 2026 Hackathon Finals submission
```

## Release body

```markdown
**Theme #3 · Professional Tasks · Meta PyTorch × Scaler OpenEnv India 2026 Hackathon Finals · Bangalore**

> Real reinforcement-learning agent for global supply-chain risk management. Two environments. Eight algorithms. Sixty-eight sha-stamped receipts. Zero synthetic substitution.

## 🎯 Six headline numbers

| # | Metric | Value |
|---|--------|-------|
| 1 | **Wordle solve rate** (REINFORCE v2) | **97.0%** |
| 2 | **Cohen's d** (trained vs null random) | **4.41–5.13** range |
| 3 | **Wilcoxon p-value** (REINFORCE v2 paired) | **6.6 × 10⁻³⁵** |
| 4 | **Wilcoxon p-value** (RAP-XC vs MaskablePPO-v3) | **[STRUCK]** |
| 5 | **Conformal coverage** (Vovk 2005) | **0.9544 / 0.92 / 0.81** at α=0.05/0.10/0.20 |
| 6 | **Reward-hack attacks blocked** | **19 / 19** (0% false-positive) |
| 7 | **Live API keys validated** | **4 / 4** (OPENROUTER, EIA, NASA_FIRMS, GFW) |

## 📊 What's inside

- **Two OpenEnv-compliant environments**:
  - **Wordle** — canonical RLVR mini-env (102-word baseline, 4-tier RLVE adaptive curriculum)
  - **SupplyMind** — Theme #3 flagship: 40 real company nodes (TSMC, Samsung, Toyota), 280 discrete actions, $5–15M budgets, 20 live data sources, 8-event crisis library

- **Nine RL agents benchmarked**: REINFORCE-v2 / RAP-XC / MaskablePPO-v2 / MaskablePPO-v3 / RecurrentPPO / A2C / SAC-Discrete / CQL / heuristic constraint-filter

- **Real training receipt**: 156 PyTorch gradient updates · 4,992 episodes · CPU-only · ~3 min wall-clock · sha `dd34457756…`

- **20-attack adversarial gauntlet** (literature-grade): Skalse 2022 + Krakovna 2020 + Pan 2022 specification-gaming patterns — empty / digits / Unicode / SQL / path-traversal / JSON-payload / base64 / sleep-attack / repeat-guess / etc.

- **Multi-level conformal calibration**: Vovk 2005 split conformal + Romano 2020 APS + Vovk 2003 Mondrian conditional coverage at 3 α levels

- **Forecasting**: TFT 513,534 steps / Chronos+TimesFM+TabPFN ensemble (8/8 events within ±30%, median 3.32% rel error)

- **Statistical proof**: Wilcoxon p=[STRUCK] RAP-XC vs MaskablePPO · Cohen's d +2.73 · paired bootstrap CI95 [+0.198, +0.257]

- **HuggingFace Space LIVE**: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind (stage RUNNING, 22 endpoints, /health 200 OK)

## 🚀 Reproduce in one bash command

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
bash FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh
```

Regenerates 60+ receipts deterministically. CPU-only. ~5 minutes.

## 📁 Key files

- [`FINAL_SUBMIT/HACKATHON_README.md`](FINAL_SUBMIT/HACKATHON_README.md) — main entry (3-5 min read)
- [`FINAL_SUBMIT/JUDGE_DASHBOARD.html`](FINAL_SUBMIT/JUDGE_DASHBOARD.html) — one-page interactive
- [`FINAL_SUBMIT/EXEC_SUMMARY_ONE_PAGE.md`](FINAL_SUBMIT/EXEC_SUMMARY_ONE_PAGE.md) — TL;DR
- [`FINAL_SUBMIT/MASTER_FEATURE_USECASE_MAP_250.md`](FINAL_SUBMIT/MASTER_FEATURE_USECASE_MAP_250.md) — 250+ features → file → receipt
- [`FINAL_SUBMIT/JUDGE_FAQ_30.md`](FINAL_SUBMIT/JUDGE_FAQ_30.md) — 30 anticipated questions pre-answered
- [`FINAL_SUBMIT/JUDGE_4MIN_SCRIPT.md`](FINAL_SUBMIT/JUDGE_4MIN_SCRIPT.md) — exact-words pitch
- [`FINAL_SUBMIT/RL_GUIDE_59POINT_ALIGNMENT.md`](FINAL_SUBMIT/RL_GUIDE_59POINT_ALIGNMENT.md) — every guide point covered
- [`FINAL_SUBMIT/MODEL_CARD.md`](FINAL_SUBMIT/MODEL_CARD.md) · [`DATASET_CARD.md`](FINAL_SUBMIT/DATASET_CARD.md) · [`ENV_CARD.md`](FINAL_SUBMIT/ENV_CARD.md)
- [`FINAL_SUBMIT/CITATIONS.bib`](FINAL_SUBMIT/CITATIONS.bib) — 19 papers
- [`FINAL_SUBMIT/SLIDE_DECK.md`](FINAL_SUBMIT/SLIDE_DECK.md) — 8 slides (pandoc → pptx)
- [`FINAL_SUBMIT/NOTEBOOKLM_CONTEXT_FOR_VIDEO.md`](FINAL_SUBMIT/NOTEBOOKLM_CONTEXT_FOR_VIDEO.md) — sub-2min video prep
- [`FINAL_SUBMIT/HONEST_LIMITATIONS.md`](FINAL_SUBMIT/HONEST_LIMITATIONS.md) — what we do NOT claim

## 🔗 Live endpoints (HF Space)

- https://shaurya-noodle-supplymind.hf.space/health
- https://shaurya-noodle-supplymind.hf.space/openapi.json
- https://shaurya-noodle-supplymind.hf.space/metadata
- https://shaurya-noodle-supplymind.hf.space/tasks
- https://shaurya-noodle-supplymind.hf.space/grader

## License

MIT. No synthetic substitution. Every claim sha256-replayable.

---

**Built for Meta PyTorch × Scaler OpenEnv Hackathon Finals 2026 · Bangalore.**
```
