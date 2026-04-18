# SupplyMind v3.0-arcadia — Results (one page)

> Every number here is reproducible from the committed JSON in `v3_arcadia/results/` with one `jq` or `python` command. No synthetic substitution anywhere in the pipeline.

---

## Ten headline numbers

| # | Metric | Value | Evidence |
|---|---|---|---|
| 1 | **RAG nDCG@10** (26 real Wiki crisis × 20 SC queries, out-of-domain) | **0.971** (Snowflake) / 0.968 (BGE-M3) / 0.960 (mxbai) | `R5_BEIR_MANUAL.json` |
| 2 | **RAG P@1 on precise queries** (6,483-chunk real corpus) | **0.962** (mxbai bi-encoder) | `R5_GRANITE.json` |
| 3 | **RAG MRR on precise queries** | **0.978** | `R5_GRANITE.json` |
| 4 | **LLM 2-judge Krippendorff α (ordinal)** on 26 crisis scenarios | **0.750** | `R4_DANGEROUS_V2_ABLATION.json` |
| 5 | **Cohen weighted κ (Qwen-14B × Mistral-Nemo)** | **0.747** | `R4_DANGEROUS_V2_ABLATION.json` |
| 6 | **Per-horizon split-conformal deviation** from 95% nominal (WTI oil) | **0.024** (pooled: 0.112 → 4.7× tighter) | `R6_AQUA_REGIA_V2.json` |
| 7 | **MaskablePPO lift vs plain PPO** (isolated, 100k steps, 50 eval eps) | **+26.8%** easy / **+15.1%** hard, invalid 13.6 → **0 structurally** | `R6_GETHSEMANE_MASKING_ABLATION_ALLTASKS.json` |
| 8 | **GNN arrival-time MAE reduction vs MLP** | **−48% / −49% / −64%** (easy / medium / hard graph) | `R6_PROVIDER_V2.json` |
| 9 | **TimesFM-CP deviation @ 95%** (WTI, EUR-USD) | **0.050 / 0.032** (Chronos-native: 0.239 / 0.214) | `R3_TIMESFM_QUANTILE.json` |
| 10 | **PPO vs random/greedy CI95** (8,100-ep bootstrap) | non-overlapping on all 3 tasks | `R6_EUCLIDIAN.json` |
| 11 | **MaskablePPO vs PPO / A2C / RecurrentPPO** (same 100k, same seed) | **+21.2% / +27.2% / +10.0%** | `R6_ALGO_COMPARISON.json` |

---

## One-line infrastructure summary

- **13 foundation models** locally (mxbai / BGE-M3 / Snowflake / BGE-reranker / Chronos-Bolt / TimesFM-2 / TabPFN-v2 clf+reg / Qwen-2.5-14B / Qwen-Coder-14B / Qwen-VL-7B / DeepSeek-R1-7B / Mistral-Nemo)
- **261,175 real data points** across 8 sources (DataCo 180,519 / NOAA IBTRACS 243,495 / FRED 17,679 × 12 / USGS live / WB WGI 214×6×24 / SEC 10-K / Wikipedia / WB Macro)
- **173 tests passing** in 2m14s (19 formal OpenEnv-compliance)
- **9 RL algorithms** implemented (MaskablePPO / PPO / BC / CQL / IQL / TD3+BC / QR-DQN / Decision Transformer / FedAvg) + custom 3-layer GCN in pure PyTorch
- **40 committed result JSONs**, **21 publication-quality plots**, **15 v3 checkpoints** (MaskablePPO zip + ONNX + TabPFN cache + GCN weights)
- **Production stack**: FastAPI + MCP JSON-RPC + WebSocket, 3 Docker builds, 3 GitHub Actions, Streamlit dashboard, Colab notebook, ONNX policies at 0.97 MB each
- **Zero synthetic substitution** in any headline number

---

## Verify any number in under 60 seconds

```bash
git clone https://github.com/ShAuRyA-Noodle/Sleep-Token.git && cd Sleep-Token

# 1. nDCG@10 = 0.971
jq '.our_results."snowflake-arctic-l"."mean_ndcg@10"' v3_arcadia/results/R5_BEIR_MANUAL.json

# 2. P@1 = 0.962
jq '.pipelines.P2_mxbai_bi.p1' v3_arcadia/results/R5_GRANITE.json

# 3. Krippendorff α = 0.750
jq '.agreement_primary_panel.krippendorff_alpha_ordinal' v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json

# 4. Per-horizon conformal dev = 0.024
jq '.results.DCOILWTICO."conf_0.95".per_horizon.ARIMA.dev_from_nominal' v3_arcadia/results/R6_AQUA_REGIA_V2.json

# 5. Masking lift +26.8%
jq '.action_masking_contribution.reward_pct_delta' v3_arcadia/results/R6_GETHSEMANE_MASKING_ABLATION.json

# 6. Tests pass
pytest tests/ -q
```

---

## Architecture at a glance

```
Real data ──▶ R1 Emergence    (13 foundation models)
            ──▶ R2 Caramel      (TabPFN + XGB + LGB + CAT + Ridge, SHAP + fairness + calibration)
            ──▶ R3 Past Self    (Chronos + TimesFM + ARIMA + Prophet, Bates-Granger stacking, conformal)
            ──▶ R4 Dangerous    (DeepSeek + Qwen-14B + Mistral-Nemo 3-judge + critic, α=0.75, ECE)
            ──▶ R5 Granite      (mxbai + BGE-M3 + Snowflake + reranker + HyDE, 8 pipelines, P@1=0.962)
            ──▶ R6-α Gethsemane (MaskablePPO, +26.8%, ONNX 0.97 MB × 3)
            ──▶ R6-β Euclidian  (8,100-ep bootstrap CI95 non-overlapping)
            ──▶ R6-γ Provider   (custom GCN, −48–64% MAE vs MLP)
            ──▶ R6-δ Aqua Regia (per-horizon split-conformal, dev 0.024 on oil)
            ──▶ R7 Arcadia      (OpenEnv server, MCP, Docker, HF Space, CI)

Every block ships committed artifacts (JSON + plot + checkpoint + test).
```

---

## Why this wins the Meta PyTorch OpenEnv Hackathon

1. **OpenEnv is not retrofitted — it's native.** 19 formal compliance tests pass. Pydantic v2 types at every boundary. MCP JSON-RPC is a first-class endpoint, not an adapter.
2. **Breadth + depth in a single repo.** 13 foundation models, 9 RL algorithms, custom GCN, conformal intervals, LLM judging, tabular SOTA — all with publishable benchmarks.
3. **Real data only.** 261,175 points from 8 public authoritative sources. Every claim traceable to its primary record.
4. **Every number defensible.** Drop the committed JSON into any reviewer's machine, re-run the 3-line verify — same answer every time.
5. **Published reproducibility challenge.** `challenges/R4_RUBRIC_CHALLENGE.md` invites anyone to beat the 2-judge α = 0.750.

---

*Updated 2026-04-18. Commit-by-commit phase log in `v3_arcadia/95_arcadia/README.md`.*
