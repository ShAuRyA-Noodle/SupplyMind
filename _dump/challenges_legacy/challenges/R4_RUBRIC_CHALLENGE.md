# R4 Dangerous — Rubric Reproducibility Challenge

This is a reproducible challenge anyone can run. Your goal: **match or beat** the SupplyMind 2-judge LLM panel on 26 real Wikipedia crisis articles, using whatever method you want.

**Current standings** (from `v3_arcadia/results/R4_DANGEROUS_V2_*.json`):

| Method | Accuracy vs ground truth (26 scenarios) | Agreement (α) |
|---|---|---|
| DeepSeek-R1-Q4 alone | 30.8% | — |
| Rubric agent (keyword-based, deterministic) | 61.5% | deterministic |
| 2-judge panel (Qwen-2.5-14B + Mistral-Nemo) | 61.5% | α = 0.750 |
| 3-judge panel (+ DeepSeek devil's-advocate) | 69.2% | α = 0.210 |

**Your mission**: build a system that hits **>75%** accuracy with **α > 0.70** on the same 26 scenarios.

---

## The scenarios

All 26 Wikipedia crisis articles are in `external_data/wikipedia_crises/`:

```
2011_Tōhoku_earthquake_and_tsunami.txt       → CRITICAL
2020–2023_global_chip_shortage.txt           → CRITICAL
2021_Suez_Canal_obstruction.txt              → HIGH
Bab-el-Mandeb.txt                            → HIGH
Baltic_Dry_Index.txt                         → LOW
Bullwhip_effect.txt                          → MEDIUM
CHIPS_and_Science_Act.txt                    → MEDIUM
Container_ship.txt                           → LOW
Enterprise_resource_planning.txt             → LOW
Ever_Given.txt                               → HIGH
Foxconn.txt                                  → MEDIUM
Inventory.txt                                → LOW
Just-in-time_manufacturing.txt               → MEDIUM
Logistics.txt                                → LOW
Port_of_Los_Angeles.txt                      → MEDIUM
Port_of_Singapore.txt                        → MEDIUM
Red_Sea_crisis.txt                           → CRITICAL
Samsung_Electronics.txt                      → MEDIUM
Semiconductor_industry.txt                   → HIGH
Strait_of_Hormuz.txt                         → HIGH
Strait_of_Malacca.txt                        → HIGH
Suez_Canal.txt                               → HIGH
Supply_chain_attack.txt                      → HIGH
Supply_chain_management.txt                  → LOW
TSMC.txt                                     → HIGH
Warehouse.txt                                → LOW
```

---

## Ground truth labeling rubric (deterministic)

Apply this rubric to each article, reading only the first 3000 characters:

1. **CRITICAL** — worldwide disruption, multi-sector, recovery > 6 months.
   Examples: full national tsunami disaster, multi-year industry-wide chip shortage, multi-year armed blockade of major shipping lane.

2. **HIGH** — regional or single-sector disruption, recovery 1–6 months.
   Examples: single-incident ship grounding, regional geopolitical chokepoint, concentrated single-supplier dependency (e.g. TSMC).

3. **MEDIUM** — localized disruption with spillover, OR systemic concept article with ongoing-risk relevance.
   Examples: port congestion, manufacturer concentration, systemic dynamics (bullwhip, JIT).

4. **LOW** — concept or infrastructure article with no active disruption, or recoverable in < 30 days.
   Examples: Wikipedia article on "Container ship" as a ship class, "Inventory" as an accounting concept.

This rubric is what we used to hand-label ground truth in `v3_arcadia/30_dangerous/r4_ablation_and_baseline.py` (see `GROUND_TRUTH` dict).

---

## Evaluation metrics

Your submission must report all four:

1. **Accuracy vs ground truth**: fraction of 26 scenarios where your predicted risk level matches the rubric-derived ground truth.
2. **Krippendorff α (ordinal)**: if you use multiple judges, inter-rater agreement on ordinal risk levels.
3. **Confidence calibration (ECE)**: if your system produces confidence scores, expected calibration error on the accuracy indicator.
4. **Latency**: seconds per scenario on your hardware (report hardware).

---

## Submission format

Submit a pull request to the SupplyMind repo with:

1. `challenges/submissions/<your-handle>/solution.py` — your code.
2. `challenges/submissions/<your-handle>/results.json` — per-scenario predictions + the 4 metrics.
3. `challenges/submissions/<your-handle>/README.md` — method summary + hardware used.

---

## Reference implementation

- 2-judge panel: `v3_arcadia/30_dangerous/r4_v2_beast.py`
- Rubric agent baseline: `v3_arcadia/30_dangerous/r4_ablation_and_baseline.py` function `rubric_score()`
- Metrics: see `krippendorff_alpha_ordinal`, `ece_binary`, `cohen_weighted_kappa_pairwise` in the same file.

---

## Why this matters

Most LLM-as-judge benchmarks (MT-Bench, RewardBench) use open-ended prompts without ground truth. This challenge has:
- **Deterministic ground truth** via the above rubric
- **Independent data** (Wikipedia, not our own writing)
- **Multi-metric evaluation** (not just accuracy)
- **Open reproducibility** (no hidden test set)

If you beat 2-judge panel at α>0.70 and >75% accuracy, send a PR and we'll add your method to `MODEL_CARD.md`.
