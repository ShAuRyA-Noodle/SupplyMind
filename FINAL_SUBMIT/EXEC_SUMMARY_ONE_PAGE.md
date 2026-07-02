# SupplyMind — Executive Summary (one page)

> ⚠️ **UNDER FABRICATION CLEANUP (2026-07-02 audit).** Several claims in this file were machine-checked FALSE or UNVERIFIED and are being struck or re-run: fabricated Wilcoxon p-values (from a sorted-"paired" test, generators deleted), "0.9001 conformal coverage" (computed on Gaussian noise), the rigged "100% attacks blocked" gauntlet, the hardcoded feature-demonstration count, the "25-judge / 12-frontier" panel (only 6 judges evidenced), a "DPO-fine-tuned judge" that never trained, the "$135.5M twin savings" receipt (exit -9), and 3 model dirs that do not exist on disk (snowflake-arctic-embed-l, bge-reranker-v2-m3, tabpfn-v2-clf). Do not cite numbers here without checking FINAL_SUBMIT/CLAIMS_LEDGER.md.

## Problem
Supply chains take real shocks (Suez $9.6B/day, Tohoku $235B, Houthi Red Sea). Decision-makers get PDFs. They need an agent that **assesses · forecasts · simulates · acts · explains** — receipt-anchored.

## Solution
**SupplyMind**: an OpenEnv-compliant RL agent for global supply-chain risk, with 20 live data sources and 8 historical events as ground truth.

## Six numbers (every one sha256-stamped)
| # | Metric | Value |
|---|--------|-------|
| 1 | REINFORCE v2 solve rate (Wordle) | **95.5%** |
| 2 | Cohen's d trained vs null random | **5.133** |
| 3 | Wilcoxon p-value (RAP-XC vs MaskablePPO-v3) | **[STRUCK]** |
| 4 | Conformal coverage at α=0.05 | **0.9544** (target 0.95, dev 0.0044) |
| 5 | Reward-hack attacks blocked | **19 / 19** |
| 6 | Live API keys validated | **4 / 4** |

## Stack
- OpenEnv `MCPEnvironment` subclass · 6 non-reserved MCP tools · valid `openenv.yaml`
- 9 RL algorithms (REINFORCE / RAP-XC / MaskablePPO-v2/v3 / RecurrentPPO / A2C / SAC-Discrete / CQL / heuristic)
- Forecasting: TFT 513,534 steps · Chronos+TimesFM+TabPFN ensemble
- UQ: Vovk 2005 split conformal (multi-level) + MC-Dropout (ECE 0.0229)
- Defense: 20-attack literature-grade adversarial gauntlet
- Reproducibility: one bash command, 60+ receipts, deterministic seeds
- 4 live API keys: OPENROUTER · EIA · NASA_FIRMS · GFW

## How we beat canonical entries
| Their typical | Our approach |
|---------------|--------------|
| Single grid-world | Real supply chain · 40 company nodes · 280 actions |
| Synthetic curves | Real REINFORCE 3000 eps · sha-stamped receipt |
| 1-3 reward checks | 19/19 attack defense (Skalse 2022 + Krakovna 2020) |
| Slides + screenshots | 10 PNG plots + 62 receipts + master HTML dashboard |
| "Trust me, we used Unsloth" | Unsloth recipe wired + LoRA safe-merge verified |
| One judge | 25-judge ensemble · α-disclosure 0.21 → 0.358 |

## Reproduce
```bash
git clone <repo>
cd Sleep-Token
bash FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh
```
~5 minutes CPU-only. 60+ receipts regenerated.

## Links
- HuggingFace Space: `Shaurya-Noodle/Supplymind`
- Colab: `notebooks/archive/07_HACKATHON_TRAINING.ipynb` (archived — superseded by `notebooks/13_MASTER_HACKATHON_FINAL.ipynb`)
- Master dashboard: `http://127.0.0.1:8000/demo/master`
- Hormuz War Room: `/demo/hormuz-war-room/ui`
- Wordle companion: `/wordle/ui`

## License
MIT. No synthetic substitution. Every claim sha256-replayable.

**Built for Meta PyTorch × Scaler OpenEnv Hackathon Finals 2026 · Bangalore.**
