# REBUILD BACKLOG — every de-faked capability, resurrected for real

> Owner directive: deletion was triage. Each capability below was removed, stubbed, or relabeled
> during P0.2 because its old implementation was fake. Each MUST come back as the genuine,
> world-class version — or be consciously retired with the owner's sign-off (marked RETIRE?).
> Old code is recoverable via `git log -p -- <path>` on branch `p0/foundation`.
> Status legend: READY (can build now) · KEY (blocked on OpenRouter key) · GPU (needs local
> RTX-4080 time) · DATA (needs regenerated artifacts first) · LONG (multi-hour run).

| # | Capability | What was fake | The REAL rebuild | Needs | Wave |
|---|---|---|---|---|---|
| R1 | Real-world benchmark (`rl/real_world_benchmark.py`, deleted) | Hardcoded agent scores, invented $-savings arithmetic | Evaluate the ACTUAL checkpoints in `rl/checkpoints/` on DataCo real-order replay: proper per-order predictions, real late-rate deltas, documented $-model with cited cost assumptions | DATA (buffers regen) + LONG | P1.3 |
| R2 | Agent leaderboard (fallback table deleted) | Fabricated blueprint scores ±fake std, "p<0.01" string | Full benchmark run: 10 agents × 3 tasks × ≥20 seeds → real `benchmark_summary.csv` → leaderboard renders real numbers with true paired stats | LONG | P1.3 |
| R3 | Ablation study (CSV deleted) | 7 configs byte-identical to scripted | Real component ablation: train/eval QR-DQN variants (±CVaR, ±real-data buffer, ±uncertainty head, ±ensemble) — real deltas per component | GPU + LONG | P1.3 |
| R4 | Pareto frontier "20 policies" (relabeled seed-study) | Same scripted agent ×20, weights recorded but unused | Reward-scalarized training: sweep objective weights (cost vs resilience vs carbon), short PPO runs per weight vector → genuine Pareto frontier; carbon constants from real per-episode shipment tonnage | GPU + LONG | P1.3+ |
| R5 | GNN attention criticality (untrained feed removed) | Random-init GAT weights served to dashboard | Actually train `SupplyChainGAT` on the supply graph (edge criticality labels derivable from Monte-Carlo cascade frequencies — engine exists); checkpoint + serve real attention | GPU (small) | P1.7+ |
| R6 | 4-method causal counterfactual (nb13 §9 deleted; pass22 deleted) | rng.normal draws labeled ARIMA-BSTS/do-calculus | Implement the four methods for real on Tōhoku 2011: paired-bootstrap MC via the ENGINE's Monte-Carlo (exists), synthetic-control from World-Bank peer series (data on disk), ARIMA/BSTS-lite on real FRED, do-calculus = engine intervention on TSMC node (engine supports inject_disruption) | READY (engine+data exist) | **R-wave** |
| R7 | Adversarial gauntlet scale (269→174) | +19 phantom attacks, both-branches counting | Grow the REAL attack corpus past the old fake number: more injection payloads, fuzz classes, reward-hack probes — executed against real env, honest counting (target: 300+ real attacks) | READY | **R-wave** |
| R8 | DPO-fine-tuned judge (claim struck) | Training crashed 4×, never existed; claimed in PREPRINT abstract | Actually train it: fix the torch/peft version skew (crash root-cause is logged in train_gpu.log), 21 real preference pairs exist, Qwen2.5-1.5B QLoRA on the 4080; honest before/after judge eval | GPU | P1+ |
| R9 | Ghost features: Snowflake-Arctic 2nd embedder, BGE reranker, TabPFN-clf 7th judge (removed) | Weights never on disk; silent fallbacks | Download the real weights (arctic-embed-l ~1.3GB, bge-reranker-v2-m3 ~2.2GB, tabpfn-v2-clf ~100MB — disk freed: 37GB available), restore the modules from git, wire + eval: does 2-embedder ensemble beat single mxbai on the committed BEIR-style queries? Keep only if measurably better — that measurement is the feature | READY (download) | **R-wave** |
| R10 | Ensemble Brent forecast backtest (synthetic sinusoid removed) | "Real-style" fake price history | Chronos+TimesFM+TabPFN weights EXIST locally; real FRED daily Brent now on disk — run the true 8-event backtest with real history windows | READY | **R-wave** |
| R11 | Conformal calibration receipts (noise receipts deleted) | Gaussian noise labeled model-NLL | Extend `calibrate_conformal_from_harvest.py` (real transitions.npz, 1.16GB in ARCHIVE — restore or repoint): real NLL split-conformal with honest coverage | DATA | P1.4 |
| R12 | Federated learning (claim deleted) | Inline toy regression with false provenance | Real FedAvg on `rl/federated/fedavg.py`: split DataCo by market region into K clients, train local risk models, aggregate — report the real delta vs single-client (whatever it is) | READY | R-wave |
| R13 | Cross-env transfer (noise probes deleted) | rng featurizers, untrained policy | Real experiment: train on Wordle, evaluate representation on SupplyMind with a REAL featurizer from observation fields; report honest transfer (may be ~zero — that's a finding) | GPU (small) | P1+ |
| R14 | 12-judge frontier panel (JUDGES_12 deleted) | 6 untested model slugs | Verify current OpenRouter catalog live, pick 12 real slugs, run the full panel on the 26 scenarios, real Krippendorff α on 3+ judges | KEY | P1.1 |
| R15 | W&B live dashboards (fake-curve scripts deleted) | Formula-generated reward curves | Real training runs already log real metrics (nb13 §3 REINFORCE) — make W&B the real observability layer for P1.3 benchmark + R4/R5/R8 training; scrub the old fake runs from the W&B project | READY (+owner scrub) | P2.4 |
| R16 | Analytics dashboard (Streamlit `dashboard/` deleted) | np.random violins, hardcoded SHAP/radar | P2.1 rebuild inside the war-room stack: real charts from real `benchmark/results/` artifacts via dataviz skill; real SHAP on a real trained model (shap lib against QR-DQN obs features) | DATA (after P1.3) | P2.1 |
| R17 | World-Bank RAG ingestion (bug fixed, corpus stale) | list-vs-dict bug meant 0 WB chunks ever indexed | Re-cook the RAG corpus with WB data actually ingested; re-run the retrieval benchmark; publish honest before/after P@1 | READY | R-wave |
| R18 | SAC-Discrete trainer (honest skeleton) | "skeleton_only" return | Implement it (CleanRL reference) or RETIRE? — owner call; low value vs QR-DQN already real | GPU | P2+ |
| R19 | Damocles service endpoints (container deleted) | Unstartable image, 4/5 endpoints DOA | Fold /assess /forecast /rag /rl-act into main server per P0.3 with the real local models + OpenRouter judges | KEY (assess) | P0.3/P1 |
| R20 | Wikipedia crisis corpus gaps (6 articles fetched ✅) | silently missing | DONE in Wave 2 — fetched real. Fold into RAG re-cook (R17) | — | done |
| R21 | Master dashboard capability cards (25 deleted) | alert() popups | As real features return (R1–R19), each earns its card back — linked to its live endpoint or committed receipt served via /static. Target: MORE cards than before, all real | rolling | P2.1 |
| R22 | Twin counterfactual $135.5M receipt (exit -9) | doc claimed success | Re-run `counterfactual_twin` to completion; publish the real number whatever it is | READY/LONG | Wave 3 (in flight) |

## R-wave (next after Wave 3): READY items needing no key/GPU-marathon
R6 (real causal methods) · R7 (gauntlet 300+) · R9 (ghost weights download + eval) · R10 (real Brent backtest) · R12 (real FedAvg) · R17 (RAG re-cook + honest P@1).

## Principle
A rebuilt feature ships ONLY with its measurement. "We built X and it improves Y by Z% (receipt)"
— or "we built X, it did not help, here is the honest result" — both are world-class. Silent
fakery was the only disqualifying state, and it is gone.
