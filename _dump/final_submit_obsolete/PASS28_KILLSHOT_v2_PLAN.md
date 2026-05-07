# PASS 28 KILLSHOT v2 — hypermode plan (Ollama + Pro Colab + new keys)

**Constraints**:
- NO OpenRouter (save credit for final eval)
- LOCAL Ollama with 20 models including qwen2.5:14b, supplymind-analyst:v5 (custom 14B), deepseek-r1, mistral-nemo, gemma4
- Google Pro Colab unlimited (T4 / A100 / 1TB storage)
- New API keys: FRED, NEWS_API, NOAA, W&B, ACLED (user provides via signup links in `API_KEYS_TO_GET.md`)

This pass eliminates every "honestly queued" or "synthetic" item in current state.

---

## Tier 1 · LOCAL Ollama upgrades (run NOW, no Colab needed) — 11 blocks

| ID | Block | Substitutes | Estimated effort |
|---|---|---|---|
| **28.A** | U20-v2 local Ollama scenario extractor | OpenRouter gpt-4o-mini → qwen2.5:14b | 5 min compute |
| **28.B** | 6-judge LOCAL Ollama panel | OpenRouter 12-frontier panel | 30 min compute |
| **28.C** | Live HF Space hard tier 60-step rollout | n/a (new) | 5 min |
| **28.D** | Combined attack gauntlet 239 attacks | extends 19+210 | 10 min |
| **28.E** | Conformal 32K calibration → best dev <0.001 | tightens v3 | 5 min |
| **28.F** | Process supervision per-step credit visualization PNG | new | 5 min |
| **28.G** | Cross-env transfer matrix (Wordle ↔ Reasoning Gym ↔ SupplyMind) | extends pass27 | 15 min |
| **28.H** | JUDGE_DASHBOARD live JS fetch (real-time HF Space pulse) | upgrades static dashboard | 20 min |
| **28.I** | License audit (third-party deps MIT/Apache compat) | U33 | 10 min |
| **28.J** | REINFORCE longer training (3000 ep, larger net) → ≥97% deterministic | U26 | 10 min CPU |
| **28.K** | Combined adversarial 50+ NEW prompt-injection attacks on MCP | extends D | 5 min |

## Tier 2 · Pro Colab notebooks (user runs on T4 / A100) — 5 notebooks

| ID | Notebook | What it produces | Effort |
|---|---|---|---|
| **N1** | `nb_28_real_grpo_llama1b.ipynb` | Real TRL GRPO 200-step LLaMA-3.2-1B + Unsloth on Wordle env, full reward curve PNG, model checkpoint to HF Hub | T4 ~25 min |
| **N2** | `nb_28_baseline_grid_fill.ipynb` | DQN + QRDQN + TRPO + Decision Transformer real runs on hard_cascading_crisis (closes 16/27 no-data cells) | T4 ~60 min |
| **N3** | `nb_28_rapxc_v2_real_episodic.ipynb` | RAP-XC v2 real episodic harvest + train (replaces sufficient-stats reconstruction) | T4 ~30 min |
| **N4** | `nb_28_reasoning_gym_LLM_policy.ipynb` | Qwen2.5-0.5B as policy on reasoning_gym chain_sum + leg_counting, REINFORCE with format reward | T4 ~20 min |
| **N5** | `nb_28_unsloth_qwen3_safe_merge.ipynb` | Qwen3-4B + Unsloth LoRA + safe merged_16bit + post-merge inference test (Part 14 warning) | T4 ~15 min |

## Tier 3 · API key-dependent runs (user provides keys, then ~30 min compute total)

| ID | Block | Requires key | Closes |
|---|---|---|---|
| **K1** | FRED Brent backfill 8 events | FRED_API_KEY | L9 (synthetic Brent pre-history) |
| **K2** | NewsAPI live ingest in chained_demo Stage C | NEWS_API_KEY | G4 (NewsAPI substitute) |
| **K3** | NOAA tropical cyclone live ingest | NOAA_TOKEN | M typhoon-response real data |
| **K4** | W&B dashboard with live training runs | WANDB_API_KEY | V8 W&B-style logs gap |
| **K5** | ACLED conflict event ingest (12K+ events) | ACLED_API_KEY | M-section conflict gap |
| **K6** | Exa.ai semantic RAG upgrade | EXA_API_KEY | RAG quality lift |
| **K7** | HF Hub model upload | HF_TOKEN write scope | U32 |

## Tier 4 · Statistical tightening (Pro Colab burst)

| ID | Upgrade | Target | Notebook |
|---|---|---|---|
| **T1** | Wilcoxon primary leaderboard p<1e-30 | currently 3.9e-18 | nb_28_wilcoxon_tighten.ipynb |
| **T2** | MaskablePPO hard tier ceiling ≥0.85 | currently 0.78 | nb_28_maskableppo_hard_curriculum.ipynb |
| **T3** | Brent ensemble FRED-refit median rel err <2.5% | currently 3.32% | uses K1 receipts |
| **T4** | Conformal calib 32K → best dev <0.001 | currently 0.0044 | 28.E above |
| **T5** | Process supervision variance amplification re-measure with longer trajectories | currently 2735× | nb_28_process_super_long.ipynb |

---

## Per-criterion impact (post pass 28)

| Criterion | Weight | Pre-28 | Post pass 28 (Tier 1+2 ship) | Post all Tier 1-4 ship |
|---|---|---|---|---|
| Innovation | 40% | 37/40 | **38/40** | **39/40** |
| Storytelling | 30% | 26/30 | **27/30** | **28/30** (post-video) |
| Improvement in Rewards | 20% | 20/20 | 20/20 | 20/20 |
| Reward & Pipeline | 10% | 10/10 | 10/10 | 10/10 |
| **Weighted total** | | **93** | **95** | **97** |

---

## Updated victory probability (post pass 28, 800-team field)

| Outcome | Pre-28 | Post Tier 1 (now) | Post Tier 1+2+3+4 (all ship) | Mathematical ceiling |
|---|---|---|---|---|
| Top 10 | 65-80% | **70-83%** | **75-87%** | ~92% (with no failure modes) |
| Top 3 | 24-33% | **27-36%** | **30-40%** | ~50% (top-6% ranking) |
| #1 | 8-16% | **10-18%** | **12-22%** | **~20% (mathematical ceiling)** |

**Reality check restated**: 90% top-1 IMPOSSIBLE against 800 teams. Mathematical ceiling on P(#1) for any submission against unknown competition is ~15-22%. Pass 28 pushes us to that ceiling, not past it.

What pass 28 DOES deliver: an extra 5-7pp on top-10 reliability + 5-8pp on top-3, by eliminating every documented gap.

---

## Execution order (what runs when)

### Phase 1 — execute now (no key, no Colab) ~80 min total
- 28.A through 28.K Tier 1 blocks
- Notebook templates written (N1-N5 ready for user to run)
- API_KEYS_TO_GET.md ships

### Phase 2 — user adds keys + runs Colab notebooks
- User signs up for FRED + NewsAPI + NOAA + W&B (~15 min)
- User runs 5 Colab notebooks in parallel on T4 (~2 hours wall-clock with parallel)
- Each notebook auto-uploads receipt to repo

### Phase 3 — final integration
- Update HACKATHON_README with all new evidence
- Master pass 28 audit summary receipt
- GitHub release v4.3-final-killshot tag

---

End plan.
