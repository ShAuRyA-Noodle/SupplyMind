# SupplyMind v4.0-arcadia-live: An OpenEnv-Compliant Supply-Chain Risk Management Environment with Live Geopolitical Ingestion, Autonomous Research Loop, and Honest Multi-Era Benchmarking

**Author**: ShAuRyA-Noodle
**Affiliation**: Meta PyTorch OpenEnv Hackathon 2026 (finals: Bangalore, April 25–26, 2026)
**Release**: v4.0-arcadia-live (supersedes v3.0-arcadia)
**Code**: https://github.com/ShAuRyA-Noodle/Sleep-Token
**Live demo**: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind

---

## Abstract

We present SupplyMind, an OpenEnv-compliant reinforcement-learning environment for supply-chain risk management trained entirely on 261,175 real-world data points from 8 authoritative sources (Kaggle DataCo, NOAA IBTRACS, USGS, FRED, World Bank WGI, SEC 10-K filings, Wikipedia crisis articles, and policy papers from BIS / FRBSF / FRBNY). Over a 2-month iteration we shipped three sequentially-tagged releases (`v1.0-real-data-complete`, `v2.0-vessel`, `v3.0-arcadia`) and now present `v4.0-arcadia-live` which adds: (a) a Karpathy-style autonomous research loop driven by `program.md`-specified constraints and a single bootstrap-CI95 metric; (b) a live geopolitical signal ingestor polling NewsAPI, GDELT, USGS, and FRED Brent crude in real time to feed the `/live/hormuz-closure` endpoint; (c) a real-crisis reference library of 8 Iran/Israel/Hormuz/Red Sea events (2024–2026) with 26 independent citations; (d) a proper articulation-point SPOF detector (closes the v2 F1 = 0.000 finding); (e) a multi-family stacking framework for G15; (f) a calibrated SupplyMind-Analyst Modelfile v5 with 8 hard-negative few-shots; and (g) formal reproducibility receipts. Headline v3 numbers (P@1 = 0.962 RAG, α = 0.750 judge agreement, +26.8 % masking lift, −48–64 % GNN MAE, 0.024 per-horizon conformal deviation) remain the scientific core; v4 is the operational envelope that makes them live-reproducible against 2026 news.

## 1. Contribution summary

| # | Contribution | Evidence |
|---|--------------|----------|
| 1 | OpenEnv-compliant environment with full Pydantic-v2 type contract, MCP JSON-RPC + WebSocket endpoints, and 19 formal compliance tests | `tests/test_openenv_compliance.py` + `openenv.yaml` |
| 2 | 13 locally-hosted SOTA foundation models (4 LLMs Q4\_K\_M, 3 retrieval embedders + 1 cross-encoder, 2 foundation-model forecasters, 2 TabPFN variants, 1 VL model) | `versions/v3_arcadia/results/R1_VERIFIED.json` |
| 3 | 3-judge LLM consensus with Krippendorff α = 0.750, Cohen κ = 0.747, 100 % parse rate on 26 real Wiki crisis scenarios via DeepSeek-R1 two-pass extraction | `R4_DANGEROUS_V2.json` / `R4_DANGEROUS_V2_ABLATION.json` |
| 4 | 8-pipeline RAG bench on 6,483-chunk real corpus (SEC + Wikipedia + policy PDFs) with published per-regime (precise vs paraphrased) Pareto front | `R5_GRANITE.json` / `R5_GRANITE_HARD.json` |
| 5 | MaskablePPO with isolated masking ablation showing +26.8 % reward, **structural 0 invalid actions** (13.6 → 0 / ep), CI95 non-overlapping vs random + greedy on 8,100-ep bootstrap | `R6_GETHSEMANE_MASKING_ABLATION.json`, `R6_EUCLIDIAN.json` |
| 6 | Custom 3-layer GCN in **pure PyTorch** (no `torch_geometric`), arrival-time regression with −48 % / −49 % / −64 % MAE vs MLP on 12 / 25 / 40-node real supply graphs | `versions/v3_arcadia/70_provider/r6_gnn.py`, `R6_PROVIDER_V2.json` |
| 7 | Per-horizon split-conformal prediction intervals on Chronos-Bolt + ARIMA: WTI deviation 0.024 from 95 % nominal (vs pooled 0.112 = 4.7 × tighter) | `R6_AQUA_REGIA_V2.json` |
| 8 | TimesFM-CP residual-quantile wrapper beating Chronos-native: 0.050 vs 0.239 deviation on WTI at 95 % | `R3_TIMESFM_QUANTILE.json` |
| 9 | **v4 Karpathy-autoresearch loop** — `program.md` + mutable `candidate_train.py` + fixed 50 k-step budget + bootstrap CI95 lower accept/reject + auto lab notebook | `versions/v4_arcadia_live/autoresearch/` |
| 10 | **v4 live Hormuz ingestion pipeline** — NewsAPI + GDELT + USGS + FRED Brent cached in SQLite, polled into `/live/hormuz-closure` with 3-judge panel + mxbai analog match (similarity 0.99 on 2026-04-18 Gulf-of-Oman event) + counterfactual loss projection | `versions/v4_arcadia_live/realtime/` |
| 11 | **v4 SPOF v2** (G8) — articulation-point detector with F1 = 1.000 on all 3 real graphs vs 0.949 legacy | `versions/v4_arcadia_live/features/spof_v2.py`, `R6_SPOF_V2.json` |
| 12 | **v4 stacking v2** (G15) — 4-base-learner meta-stacked classifier beating weighted voting on DataCo; honest null result vs best single on 0.97+-AUC ceiling | `versions/v4_arcadia_live/features/stacking_v2.py`, `R15_STACKING_V2.json` |

## 2. Environment design

We model a global supply chain as a directed graph with five node types (supplier, warehouse, port, factory, customer) and edges capturing `supplies`, `ships_via`, `stores_at`, `delivers_to`. The agent receives a 408-dimensional observation (40 nodes × 10 features + 8 global) plus a 280-dim action mask. Action space `MultiDiscrete([7, 40])` = 7 CSCMP-framework-mapped actions × 40 target nodes = 280 flat actions. Reward is dense per-step in [−1, 1] decomposed into 7 components (revenue preservation 35 %, stockout penalty 25 %, proactive-action bonus 15 %, cost penalty 10 %, unnecessary-action penalty 5 %, health maintenance 5 %, SLA compliance 5 %).

Three tasks with increasing complexity: `easy_typhoon_response` (12 nodes, 30 steps, $5 M budget), `medium_multi_front` (25, 45, $8 M), `hard_cascading_crisis` (40, 60, $10 M). Budget–to–exposure ratios are deliberately small (as in real crisis management); seed jitter perturbs trigger days ± 2 and severity ± 8 % to prevent memorization.

## 3. Live ingestion + real crisis library (v4 / G10)

The v4 runtime adds a SQLite-backed event store polling five public sources:

1. **NewsAPI** — 5 regional queries (`Hormuz`, `Iran Israel strike`, `Houthi Red Sea`, `Taiwan Strait`, `port strike`), ~ 80 events / cycle over 7-day lookback.
2. **GDELT 2.0 Doc API** — no API key, 15-minute refresh, tone-derived severity.
3. **USGS earthquake feed** — M4.5+ / 24 h, region-boxed to 6 supply-critical geographies.
4. **FRED `DCOILBRENTEU`** — daily Brent spot, day-over-day + week-over-week severity triggers.
5. **MarineTraffic snapshot** — graceful fallback to a committed JSON if API key absent.

On 2026-04-21 the live ingestion cycle returned 159 fetched events (80 NewsAPI + 60 GDELT + 19 USGS + 1 FRED Brent at $123.28/bbl DoD +3.54 %). This $123 value is ground-truth FRED data: it **drove the counterfactual** we ran through the `/live/hormuz-closure` endpoint that same day, which matched our committed 2026-04-18 Gulf-of-Oman crisis-library entry at similarity 0.99 via `mxbai-embed-large-v1`.

The crisis library itself contains 8 fully-cited events: Iran True Promise I/II (2024-04 and 2024-10), Houthi Red Sea campaign (2023-11 → ongoing), US–UK Operation Poseidon Archer (2024-01), Haifa port attacks (2024-10), Houthi Yaffa drone + IAF Hodeidah retaliation (2024-07), the 2026-04-18 Gulf-of-Oman / Hormuz incident, and the 2022 Ukraine neon / palladium shock for contextual breadth. Every entry has ≥ 3 independent publisher citations (Reuters, NYT, BBC, Al Jazeera, CFR, UNCTAD, Lloyd's, DOD, IDF, IMF, FT, Bloomberg, CNBC).

## 4. Autonomous research loop (v4 / L1)

Adapted from Karpathy's `karpathy/autoresearch`: the agent reads `program.md` (task spec + safe-to-modify markers + frozen metric), proposes a unified diff of `candidate_train.py`, the runner executes it in an isolated 10-minute subprocess with VRAM / NaN / test-gate guards, the evaluator computes `bootstrap_ci95_lower(grader_scores_9)` and accepts only if the new CI95 lower bound exceeds the current best by 0.005, the lab-notebook auto-generates markdown entries, and the loop iterates until the time budget is exhausted or 50 consecutive rejections. We bootstrap the loop with 5 hand-crafted seeds (bigger MLP, higher entropy, curriculum, RecurrentPPO, action-diversity bonus) to seed diverse starting points before the LLM agent (Qwen-14B local or Claude) takes over.

## 5. Reproducibility protocol

1. `git clone https://github.com/ShAuRyA-Noodle/Sleep-Token && cd Sleep-Token`
2. `pip install -r requirements.txt`
3. `pytest tests/ versions/v4_arcadia_live/tests/ -q` — verify 190 + tests pass (187 core + 3 SPOF + 4 stacking + 4 analyst-bench + live-router smoke).
4. `uvicorn server.app:app` — start the OpenEnv server on :8000.
5. `curl http://localhost:8000/live/health` — see v4 router mounted.
6. Optional live ingestion: `python -m versions.v4_arcadia_live.realtime.ingestor --once` (needs `.env` keys).
7. Optional autoresearch: `python -m versions.v4_arcadia_live.autoresearch.orchestrator --budget 6h`.

Every headline number is reproducible from the committed JSONs with a single `jq` command; see `docs/v3/RESULTS.md` §"Verify any number in under 60 seconds."

## 6. Honest limitations (explicit)

- **Stacking v2 G15 null result**: on DataCo `late_delivery_risk` (AUC ~ 0.97+), stacking beats weighted voting (+0.001 AUC) but does not beat best-single LightGBM within CI95. We publish the null result; stacking wins require decorrelated base learners.
- **Qwen-VL-7B** is verified in R1 but not benchmarked (15 GB model reserved for v4.1 port-imagery extension).
- **LoRA fine-tune** of `supplymind-analyst:v5` is implemented as Modelfile prompt-engineering + A/B bench; actual LoRA weight training is deferred to v4.1 (Ollama HF-offline blocker).
- **Forecasting** trained on 2015–2026 FRED; regime-change generalization (e.g. sustained $150+ Brent) is extrapolation.
- **Supply-chain graph is static**; live topology learning is v4.1 roadmap.

## 7. Positioning vs public benchmarks

See `docs/v3/BENCHMARKS_VS_PUBLIC.md` for side-by-side positioning against M5 (forecasting), BEIR / MTEB (retrieval), MuJoCo / Meta-World (RL), Kaggle DataCo (tabular), RewardBench / MT-Bench (LLM-as-judge), and the conformal prediction literature. We do **not** claim leaderboard dominance; we claim that no comparable published submission integrates OpenEnv compliance + the 13-model stack + 173 tests + 261 K real data points + live geopolitical ingestion into a single artifact.

## 8. Citation

```bibtex
@software{supplymind_v4_arcadia_live_2026,
  author  = {ShAuRyA-Noodle},
  title   = {SupplyMind v4.0-arcadia-live: OpenEnv-compliant Supply-Chain Risk
             Management with Live Geopolitical Ingestion and Autonomous Research Loop},
  year    = {2026},
  version = {v4.0-arcadia-live},
  url     = {https://github.com/ShAuRyA-Noodle/Sleep-Token},
  note    = {Meta PyTorch OpenEnv Hackathon 2026 submission}
}
```

*Preprint generated on 2026-04-21. Regenerate PDF via `pandoc versions/v4_arcadia_live/docs/PREPRINT.md -o preprint.pdf --pdf-engine=xelatex` (or print-to-PDF from browser if no LaTeX available).*
