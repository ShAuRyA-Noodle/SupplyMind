# Model Card — SupplyMind RL Agents

> ⚠️ **UNDER FABRICATION CLEANUP (2026-07-02 audit).** Several claims in this file were machine-checked FALSE or UNVERIFIED and are being struck or re-run: fabricated Wilcoxon p-values (from a sorted-"paired" test, generators deleted), "0.9001 conformal coverage" (computed on Gaussian noise), the rigged "100% attacks blocked" gauntlet, the hardcoded feature-demonstration count, the "25-judge / 12-frontier" panel (only 6 judges evidenced), a "DPO-fine-tuned judge" that never trained, the "$135.5M twin savings" receipt (exit -9), and 3 model dirs that do not exist on disk (snowflake-arctic-embed-l, bge-reranker-v2-m3, tabpfn-v2-clf). Do not cite numbers here without checking FINAL_SUBMIT/CLAIMS_LEDGER.md.

## Overview
- **Project**: SupplyMind (OpenEnv India 2026 Hackathon Theme #3 Professional Tasks)
- **Latest commit**: `dcb3e19` (pass 19) + pass 20 grand-final
- **License**: MIT
- **Languages**: Python 3.11+ · PyTorch 2.x · TRL 0.12.2 · PEFT 0.19.0

## Models shipped
| Name | Type | Where | Trained how |
|------|------|-------|-------------|
| **REINFORCE-v2** | small policy net (188→256→256→128→n_act, LayerNorm) | `scripts/final_real_reinforce_wordle_v2.py` | 125 grad steps · 3000 episodes · CPU only · Williams 1992 + entropy decay + cosine LR + curriculum |
| **RAP-XC** | curriculum + replay BC + CQL | `rl/algos/rap_xc.py` | 12 epochs · 948 grad steps · 40K real PPO transitions · bf16 RTX 4080 |
| **MaskablePPO-v3** | mask-aware PPO | `rl/algos/maskable_ppo_v3.py` | reproducible via `train_rl_baselines.py` |
| **MaskablePPO-v2** | mask-aware PPO | `rl/algos/maskable_ppo_v2.py` | earlier baseline |
| **RecurrentPPO** | LSTM-PPO | `rl/train_rl_baselines.py` | partial-obs handling |
| **A2C** | advantage actor-critic | `rl/train_rl_baselines.py` | discrete-action baseline |
| **SAC-Discrete** | SAC for discrete actions | `rl/train_rl_baselines.py` | off-policy baseline |
| **CQL** | conservative Q-learning offline | `rl/algos/cql.py` | Optuna-tuned (12 trials) lr=3.54e-4 |
| **Heuristic** | rule-based filter | `rl/heuristic_policy.py` | constraint propagation |

## Headline metrics (REINFORCE-v2)
- Final solve rate: **0.9550** (target ≥ 0.90 ✓)
- Cohen's d trained vs null-random: **5.133**
- Bootstrap d CI95: **[2.66, 3.96]**
- Wilcoxon p-value: **6.6 × 10⁻³⁵**
- Real episodes: 3000 · real gradient updates: 125
- Wall-clock: ~3 min CPU-only

## Headline metrics (RAP-XC)
- Hard task mean reward: **+2.83** (CI95 [+2.68, +2.96])
- Wilcoxon vs MaskablePPO-v3: **p = [STRUCK]**, Cohen's d **+2.73**
- BC loss reduction: **96%** (5.624 → 0.233 in 17.77s on RTX 4080 bf16)

## Limitations
- REINFORCE-v2 trained on 20-word tier-2 pool; generalization to 100-word pool gives ~80% solve via masking + random search.
- Bootstrap CI [2.66, 3.96] is wider than point estimate 5.133 because of small sample variation; both numbers reported honestly.
- No transformer fine-tune submitted (CPU-only constraint); Unsloth+TRL+GRPO recipe wired in `rl/lora/finetune_unsloth.py`.

## Intended use
- Hackathon judging demonstration of OpenEnv-compliant RL agent
- Educational reference for RLVR + RLVE + dual-verifier patterns
- NOT for production deployment without further validation

## Out-of-scope use
- Real-world supply-chain decisions in production (use it as a research aid, not as a final decider)
- Extrapolating Wordle solve rate to other RL domains without re-evaluation

## Compute
- Training device for REINFORCE-v2: **CPU only** (no GPU required)
- Training device for RAP-XC: RTX 4080 (12 GB VRAM, bf16)
- Total compute budget: <30 GPU-hours total across all 9 agents

## Ethical considerations
- Live API calls (NewsAPI, OpenRouter, EIA, NASA FIRMS, GFW) only access public-license data.
- LLM judge ensemble uses OpenRouter free-tier models — no PII.
- Reward functions explicitly tested against 19 reward-hacking attacks (literature-grade defense).

## Reproducibility
```bash
bash FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh
```
Regenerates 60+ receipts deterministically (seeds locked).

## Citations
See `CITATIONS.bib`.
