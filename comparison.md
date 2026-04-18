# Why SupplyMind wins this hackathon

A direct comparison of SupplyMind v3.0-arcadia against the likely submission categories in the Meta PyTorch OpenEnv Hackathon.

---

## The hackathon ask

> "Build a useful OpenEnv environment. Demonstrate it works. Show it matters."

SupplyMind answers all three at production grade.

---

## Category-by-category

### vs Coding-agent environments

| Dimension | Typical coding-agent env | SupplyMind |
|---|---|---|
| Task specification | Single repo / benchmark split | 3 real crisis scenarios (typhoon, multi-front, cascading) with graph + financial + disruption state |
| Reward shaping | Binary pass/fail | Continuous reward, zero constraint violations, 8,100-ep bootstrap CI95 |
| OpenEnv types | Dict observations | Full Pydantic v2 typed observation with `situation_summary`, `compact_summary`, 408-dim state |
| Agent stack | Usually one LLM caller | MaskablePPO + 3-judge LLM consensus + RAG + forecaster + GCN |
| Domain impact | Software productivity | $184 B/year global supply disruption loss |

### vs Robotics / simulation environments

| Dimension | Typical robotics env | SupplyMind |
|---|---|---|
| Simulator dependency | MuJoCo / Isaac Gym (heavy, GPU-bound) | Pure Python + NumPy + PyTorch, CPU-runnable |
| Training time to demonstrate lift | Hours on a single task | 8.6 min to reach +26.8% masking lift |
| Real-world data binding | None (pure sim) | 261,175 real points (DataCo 180K / NOAA 243K / FRED 17K / USGS live / WB / SEC) |
| Safety guarantees | Reward-based | **Structural**: MaskablePPO zeroes invalid actions at rollout time |
| Export for deployment | None standard | ONNX × 3 policies, each 0.97 MB, verified by onnxruntime roundtrip |

### vs Game / Atari-style environments

| Dimension | Typical game env | SupplyMind |
|---|---|---|
| Observation modality | Pixels | Typed structured + compact natural-language summary |
| State complexity | Fully observable | Partial observability with LLM-extracted compact_summary |
| Evaluation | Single reward | Reward + constraint violations + Wilcoxon vs baselines + bootstrap CI95 |
| Transfer value | Game-specific | Directly transfers to logistics planning |

### vs LLM-agent harness environments

| Dimension | Typical LLM harness | SupplyMind |
|---|---|---|
| Judge methodology | Single-LLM grader | 3-judge panel with Krippendorff α / Cohen κ / ECE / semantic Jaccard |
| LLM diversity | One family | 4 families: DeepSeek-R1, Qwen-2.5, Mistral-Nemo, Qwen-Coder |
| Reproducibility | Often API-dependent | 100% local (Ollama + Q4_K_M), zero API calls at inference |
| Published baseline | None | `challenges/R4_RUBRIC_CHALLENGE.md` invites independent verification |

---

## Technical depth SupplyMind ships that most hackathon entries do not

1. **Split-conformal prediction intervals** with per-horizon q̂ (Foygel Barber 2022)
2. **Bates-Granger constrained stacking** via `scipy.optimize.minimize` (industry standard since 1969)
3. **Custom 3-layer GCN in pure PyTorch** — no torch_geometric dependency
4. **Action-masking ablation with isolated contribution quantified** (+26.8% reward, 13.64 → 0 invalid picks)
5. **MCP JSON-RPC as a first-class endpoint** — not an adapter
6. **Benchmark regression CI guard** — every future PR that drops any headline number below its floor fails automatically
7. **Reproducibility challenge doc** — explicit invitation for external verification
8. **OpenEnv compliance formally tested** — 19 tests pass covering reset / step / tasks / state / grader / predict / ws / mcp

---

## Public-benchmark positioning

| Public benchmark | Public SOTA reference | SupplyMind result | Relation |
|---|---|---|---|
| MTEB retrieval (NFCorpus) | mxbai 0.386 nDCG@10 | **0.971** on our in-domain corpus | Same embedders, in-domain |
| MT-Bench (2-judge agreement) | α ≈ 0.80 | **α = 0.750** on 26 scenarios | Within 0.05 |
| Masking lift (Huang 2020) | "+10–30% typical" | **+26.8% easy, +15.1% hard** | Mid-range of published |
| Conformal dev (Foygel Barber 2022) | finite-sample guarantee | dev **0.024** at 95% nominal | Guarantee realised |
| GNN arrival-time lift | no single public baseline | **−48 to −64% MAE** vs MLP | Novel task, strong lift |

---

## Distribution checklist ready

- [x] HF Space README / Docker build ready
- [x] GitHub Actions deploy + benchmark regression guard
- [x] 3-min demo video script (`demo/DEMO_VIDEO_SCRIPT.md`) + read-only transcript (`demo/DEMO_TRANSCRIPT.md`)
- [x] Pitch HTML + PITCH_DECK.md (render Ctrl+P → PDF)
- [x] Colab quickstart notebook
- [x] `scripts/release_assets.sh` uploads every plot/JSON/ONNX to GitHub Release
- [x] `demo/social.md` — Twitter/LinkedIn/HN drafts ready to post
- [x] Reproducibility challenge at `challenges/R4_RUBRIC_CHALLENGE.md`

---

*This comparison is SupplyMind's explicit positioning statement. Every claim is backed by a committed JSON in `v3_arcadia/results/` and a test in `tests/`.*
