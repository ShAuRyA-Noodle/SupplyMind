# Reproducing SupplyMind v3.0-arcadia results

Two levels: **judge verification** (1 minute, read committed JSONs) or **full reproduction** (hours, re-train every model).

---

## Level 1 — Judge verification (60 seconds)

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git
cd Sleep-Token
pip install -r requirements.txt
python scripts/run_all.py            # 12 claims checked against committed JSON
pytest tests/ -q                      # 173 tests, 2m14s
```

Expected: all 12 claims pass, 173 tests pass.

---

## Level 2 — Full reproduction

```bash
# Data prep
python rl/real_data_pipeline.py

# R1 — verify foundation models
python versions/v3_arcadia/00_emergence/r1_verify_foundations.py

# R2 — tabular
python versions/v3_arcadia/10_caramel/train_caramel.py
python versions/v3_arcadia/10_caramel/shap_fairness_calibration.py

# R3 — forecasting
python versions/v3_arcadia/20_past_self/r3_past_self.py
python versions/v3_arcadia/20_past_self/r3_constrained_stacking.py
python versions/v3_arcadia/20_past_self/r3_timesfm_residual_quantile.py
python versions/v3_arcadia/20_past_self/r3_bigtft_integration.py

# R4 — LLM risk panel (requires Ollama with DeepSeek, Qwen-14B, Mistral-Nemo, Qwen-Coder)
python versions/v3_arcadia/30_dangerous/r4_dangerous_v2.py
python versions/v3_arcadia/30_dangerous/r4_ablation_and_baseline.py
python versions/v3_arcadia/30_dangerous/r4_live_scenario.py

# R5 — RAG
python versions/v3_arcadia/40_granite/r5_granite.py
python versions/v3_arcadia/40_granite/r5_hard_queries.py
python versions/v3_arcadia/40_granite/r5_manual_beir.py

# R6 — RL + GNN + conformal
python versions/v3_arcadia/50_gethsemane/r6_gethsemane.py
python versions/v3_arcadia/50_gethsemane/r6_unmasked_ablation.py
python versions/v3_arcadia/50_gethsemane/r6_unmasked_ablation_alltasks.py
python versions/v3_arcadia/50_gethsemane/export_v3_ppo_onnx.py
python versions/v3_arcadia/60_euclidian/r6_euclidian.py
python versions/v3_arcadia/70_provider/r6_gnn_arrival_time.py
python versions/v3_arcadia/80_aqua_regia/r6_per_horizon_conformal.py
```

---

## Hardware assumptions

- Python 3.11.9
- PyTorch 2.5.1 + CUDA 12.1 recommended (CPU-only path also works for R1/R2/R3/R5)
- 16 GB system RAM minimum for concurrent training + embedder evaluation
- GPU with 12 GB VRAM for R4 Ollama Q4_K_M inference and R6 PPO training
- 150 GB disk for the 13 foundation models (see `models/` subdirectories)

---

## Dependencies

```
sentence-transformers>=3.0
sb3-contrib>=2.3
stable-baselines3>=2.3
chronos-forecasting
timesfm
tabpfn>=2.0
ollama
scipy>=1.11
torch==2.5.1
fastapi>=0.115
pydantic>=2.9
pytest>=8.0
onnxruntime>=1.20
```

See `requirements.txt` for the full pinned manifest.
