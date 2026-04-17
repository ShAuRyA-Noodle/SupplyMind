# SupplyMind × PyTorch — the engineering story

**Context**: The Meta PyTorch OpenEnv Hackathon awards points not just for shipping an environment, but for demonstrating **PyTorch mastery** in its construction. This document catalogs the non-trivial PyTorch work in SupplyMind v3.0-arcadia.

Everything below is **live code**, not slides. Paths point to real files.

---

## 1. Custom 3-layer Graph Convolutional Network — pure PyTorch, zero torch_geometric

**Location**: `v3_arcadia/70_provider/r6_gnn.py`, `v3_arcadia/70_provider/r6_gnn_arrival_time.py`

The most common shortcut in supply-chain GNN papers is `import torch_geometric`. We did not. The 3-layer GCN for disruption propagation and arrival-time prediction is implemented in ~50 lines of pure PyTorch message passing using `index_add_`.

```python
class GCNLayer(nn.Module):
    """Concat(self, mean_neighbors) -> Linear."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index
        agg = torch.zeros_like(x)
        count = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, src, x[dst])                    # scatter-aggregate
        count.index_add_(0, src, torch.ones(src.size(0), 1, device=x.device))
        agg = agg / count.clamp(min=1.0)                   # mean normalization
        return self.lin(torch.cat([x, agg], dim=1))        # self+neighbor concat
```

**Why it matters**:
- Demonstrates understanding of message passing from first principles
- No heavyweight dependencies: deploys anywhere PyTorch runs
- Modifiable: swap mean-aggregate for sum, max, or attention without touching any library

**Result**: On the Arcadia disruption-propagation task, +30pp F1 over direct-neighbors baseline on the hard 40-node graph. On the v2 arrival-time regression task, significant MAE reduction over MLP-only baselines.

---

## 2. MaskablePPO over Discrete(280) — a clean PyTorch action-space wrapper

**Location**: `v3_arcadia/50_gethsemane/train_rl_beast.py`

The SupplyMind env has action space `MultiDiscrete([7, 40])` (7 action types × 40 target nodes). Action masking via sb3-contrib's `MaskablePPO` expects a *flat* mask, while MultiDiscrete masks are *marginal per-dim*.

Standard solution would be to rewrite the environment. We wrote a 10-line wrapper that flattens to `Discrete(280)` at the gym layer and unflattens inside the env step:

```python
class FlatDiscreteEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        n_type, n_target = env.action_space.nvec
        self._n_target = int(n_target)
        self.action_space = spaces.Discrete(int(n_type) * int(n_target))

    def step(self, action):
        a_type, a_target = divmod(int(action), self._n_target)
        return self.env.step(np.array([a_type, a_target]))
```

Combined with `ActionMasker` and the env's `_compute_action_mask()`, this gave us:
- **100k-step training in 6-17 min per task** on RTX 4080
- **Zero constraint violations** across 8,100 benchmark episodes
- **Bootstrap-CI non-overlapping** reward lift vs random and greedy baselines on all 3 tasks
- **Sign-flip result** on medium/hard tasks — greedy is worse than random, PPO flips the sign

---

## 3. Temporal Fusion Transformer (pure PyTorch, 513,534 params) on real FRED data

**Location**: `rl/forecasting/tft.py`

A full TFT implementation (LSTM encoder + multi-head attention + quantile head) written from scratch, trained on WTI crude oil daily prices:
- Test MAE on WTI: **$7.83**
- Multi-target heads: DCOILWTICO, PCOPPUSDM, PPICMM
- Rolling-origin 10-fold backtest

This was done in v2 (commit `aa31639`), and in v3 Aqua Regia the conformal-prediction wrapper sits on top of Chronos-Bolt + TFT-like architectures.

---

## 4. CUDA-host pinned-memory engineering on Windows

**Location**: `FAILURE_TABLE.md`, `rl/cuda/action_mask_kernel.cu`, project memory notes

Running 13 foundation models (~156 GB total, with 15 GB DeepSeek-R1-F16 as the worst) on a 15.7 GB-RAM laptop required non-trivial CUDA host-memory management:

- **Q4_K_M quantization** of all four 14B-parameter Ollama LLMs via `llama-quantize` built from source on Windows with VS Build Tools cmake. Reduced DeepSeek from **15 GB → 4.5 GB**, eliminating the "resource already mapped" CUDA_Host error class.
- **`OLLAMA_MAX_LOADED_MODELS=1` + judge-first iteration** to keep only one model resident in GPU at any time. Documented recovery protocol: system reboots clear the CUDA context when fragmentation accumulates.
- **VRAM-safe orchestration for RAG ensembles**: precompute LLM outputs (HyDE) *before* loading embedders, so Qwen-14B has the full 12 GB VRAM. Then unload and load 3 embedders + reranker.
- **Custom CUDA kernel** for action masking (`rl/cuda/action_mask_kernel.cu`) — valid source, compilation deferred on Windows without MSVC `cl.exe`.

This is the kind of engineering you don't see in a leaderboard number but makes the difference between "runs on a workstation" and "runs on a laptop."

---

## 5. ONNX export pipeline — production-ready policy artifacts

**Location**: `rl/export_onnx.py`, `v3_arcadia/50_gethsemane/export_v3_ppo_onnx.py`, `rl/checkpoints/supplymind_policy.onnx`

Every MaskablePPO policy (3 tasks × 1 checkpoint) is exported to ONNX:
- **0.97 MB** per task, runs on CPU or GPU via `onnxruntime`
- **Max torch-vs-onnx numerical diff: 1.9e-6** (essentially identical)
- Verified with `onnxruntime.InferenceSession`
- Exposed via `v3_arcadia/90_damocles/app.py` `/rl/act` endpoint

Production path: `obs [408] → features extractor → MLP policy net → action_net → logits [280]`. Action masking applied as simple post-processing outside the ONNX graph.

---

## 6. Numba-JIT Monte Carlo engine — custom accelerated fallback

**Location**: `rl/fast_engine/fast_monte_carlo.py`

Financial-impact Monte Carlo simulation needed to run inside episode rewards (~5 ms budget). Pure-Python MC was ~100 ms. Numba-JIT compilation brought it to:
- **<0.01 ms empty-sim** (warm)
- **<100 ms 10k-rollout**
- Drop-in NumPy API

Example:
```python
from rl.fast_engine.fast_monte_carlo import FastMonteCarloEngine
engine = FastMonteCarloEngine(seed=42)
p50, p95 = engine.simulate(orders, 1000)  # 10× faster than Python baseline
```

---

## 7. MC-Dropout epistemic uncertainty on the BC policy

**Location**: `rl/forecasting/mc_dropout_eval.py`, `rl/analysis/confidence.py`

Classical point-prediction BC has no uncertainty. We added Monte-Carlo Dropout at inference time (Gal & Ghahramani 2016):
- **Low-uncertainty quartile**: 99.76% accuracy
- **High-uncertainty quartile**: 55.92% accuracy
- **ECE (expected calibration error)** after isotonic calibration: 0.0017

This demonstrates **learned epistemic uncertainty** — when the agent says "I don't know," it's correctly right less often. The calibration enables a human-in-the-loop escalation rubric: if MC-Dropout variance is above threshold, flag for review rather than act.

---

## 8. Split-conformal prediction intervals (R6 Aqua Regia v2)

**Location**: `v3_arcadia/80_aqua_regia/r6_per_horizon_conformal.py`

Chronos-Bolt + ARIMA forecast intervals are wrapped in **per-horizon split-conformal** (Foygel Barber et al.; Lei et al.): a finite-sample-guarantee wrapper that re-calibrates to hit nominal coverage.

- 30-fold calibration + 30-fold held-out test
- Separate q̂₁...q̂₁₄ per horizon step (adapts to growing residual magnitude)
- Empirical coverage within **±2pp of nominal (0.95)** on DCOILWTICO (oil), which pooled-conformal missed by **11pp**

Why this matters for PyTorch: Chronos-Bolt is a PyTorch transformer. Stacking a conformal wrapper on top of its predictions is a non-trivial engineering pattern that generalizes to **any PyTorch forecaster** — the q̂ computation doesn't care what produced the residuals.

---

## 9. Semantic Jaccard via mxbai-embed-large for inter-judge agreement

**Location**: `v3_arcadia/30_dangerous/r4_v2_beast.py` `semantic_jaccard()`

Pairwise string Jaccard on judge outputs was broken (always near 0 because LLMs phrase lists differently). Replaced with:
- Embed each bullet with **mxbai-embed-large-v1** (1024-d)
- Cosine >= 0.65 → same concept
- Jaccard on concept-matched set

This is a clean PyTorch-sentence-transformers composition that any researcher can reuse for inter-rater agreement on free-text fields.

---

## 10. DeepSeek-R1 two-pass extraction (CoT → structured JSON)

**Location**: `v3_arcadia/30_dangerous/r4_v2_beast.py` `deepseek_free_single()` + `qwen_extract_single()`

DeepSeek-R1's chain-of-thought interferes with `format=json` mode (mixes reasoning into the JSON). Solution: two-pass protocol.

1. **Pass A**: DeepSeek reasons freely, ending with `FINAL_RISK=<LOW|MEDIUM|HIGH|CRITICAL>`
2. **Pass B**: Qwen-14B ingests DeepSeek's free text and extracts strict JSON
3. **Fallback**: regex scrape of `FINAL_RISK=` marker if Qwen fails

Took us from **50% parse rate** (single-pass) to **100% parse rate** on 26 scenarios.

---

## 11. Per-stage JSON caching for resume-safe multi-hour benchmarks

**Location**: Multiple (`r4_v2_beast.py`, `r5_rag_beast.py`, `r6_euclidian`)

On a consumer laptop running 8+ hour benchmarks, crashes are inevitable. Every phase writes intermediate caches:
- `R4_DANGEROUS_V2_phaseA_cache.json` (DeepSeek raw CoT per scenario)
- `R4_DANGEROUS_V2_phaseB_cache.json` (Qwen-extracted JSON)
- `R4_DANGEROUS_V2_judge_*.json` (per-judge results)
- `R4_DANGEROUS_V2_critic_cache.json` (critic outputs)
- `hyde_cache.json` (HyDE-precomputed queries)
- `corpus_emb_*.npy` (embedder-per-corpus matrices)

Re-runs skip completed stages. On our 8-hour 8,100-episode R6 Euclidian benchmark, this saved ~6 hours after a mid-run crash during the hard task.

---

## Summary for judges

**If the Meta PyTorch OpenEnv Hackathon is about demonstrating PyTorch mastery**, the non-trivial things we built *with* PyTorch (not just on top of it):

1. A from-scratch GCN without torch_geometric
2. A custom action-space wrapper that made MaskablePPO work on Discrete(280)
3. A TFT + conformal wrapper on real oil prices
4. A CUDA-host memory discipline that runs 13 SOTA models on a 12 GB laptop
5. An ONNX export pipeline producing 0.97 MB production artifacts
6. A Numba-JIT acceleration for the MC engine
7. MC-Dropout uncertainty for BC policies
8. Per-horizon split-conformal intervals
9. Semantic agreement via sentence-transformers
10. Two-pass DeepSeek extraction for 100% parse rate
11. Resume-safe per-stage caching

All **live code**. All **real data**. All **local inference** (zero API cost at runtime). All **committed** in `github.com/ShAuRyA-Noodle/Sleep-Token`.
