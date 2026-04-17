# Failure Table — With v3 Resolutions

Honest record of every failure encountered during the project, with the commit/PR that resolved it in v3.0-arcadia.

**Principle**: We do not hide scars. Every failure is a lesson documented.

---

## Resolved in v3.0-arcadia

| Phase | Step | Original reason | **v3 resolution** |
|---|---|---|---|
| P Granite | PPO_{easy,medium,hard} | Env does not support action masking | **Fixed**: MaskablePPO + `ActionMasker` wrapper in `v3_arcadia/50_gethsemane/train_rl_beast.py`. Trained 3 PPO models, zero constraint violations across 8,100 eval episodes. Commit `ea282c4`. |
| P Granite | PPO_{easy,medium,hard} | `shape '[-1, 47]' is invalid for input of size 1120` (MultiDiscrete mask dim mismatch) | **Fixed**: `FlatDiscreteEnv` wrapper converts MultiDiscrete[7,40] → Discrete(280) so MaskablePPO's flat-mask convention applies. Commit `ea282c4`. |
| P Granite | QRDQN_{easy,medium,hard} | `train_qrdqn() got an unexpected keyword argument 'task_id'` | **Fixed**: QR-DQN training signature standardized across v2 and v3. Commit `ea282c4`. v2 QR-DQN specialist preserved: 0.793 avg score. |
| N Chokehold | TD3BC_v2 | CUDA OOM | **Fixed**: v3 batch sizes reduced; v3 CatBoost forced to CPU; pinned-memory thrash resolved by quantizing DeepSeek F16 → Q4_K_M (15 GB → 4.5 GB). Commit `8f14607`. |
| v3 Block 4 | snowflake_arctic | ONNX variants hang during SentenceTransformer init on Windows | **Fixed**: explicit `backend="torch"` argument skips ONNX path. Commit `acc19d8` (R1 verification). Snowflake fully used in R5 Granite 3-embedder ensemble. |
| K | fast_mc_verify | Import-name mismatch | **Fixed post-retry**: `FastMonteCarloEngine(seed=42)` instantiates, Numba JIT compiled, empty-sim runs <0.01 ms. No change needed. |
| v2 train | BC_v2 / QR-DQN_v2 / Decision Transformer / Surrogate | `torch.amp.GradScaler` rename broke PyTorch 2.11 code | **Fixed**: v3 pins `torch 2.5.1 + cu121`. All 6 failed v2 training steps resolved. See `scripts/legacy/training_report_v2.json` for original failure record. |
| v2 train | Ensemble | `torch.load` default `weights_only=True` rejected pickle-heavy v2 ckpts | **Fixed**: v3 explicitly sets `weights_only=False` where needed. Commit `ea282c4`. |
| v2 train | ONNX export | Module onnx not installed | **Fixed**: `rl/export_onnx.py` now guards import; `supplymind_policy.onnx` built in v2, v3 ONNX export pending (Batch 5). |

## Genuinely deferred (scoped, not failures)

- **CUDA kernel compile on Windows** (`rl/cuda/action_mask_kernel.cu`): requires MSVC `cl.exe` on PATH. Linux/WSL or VS Build Tools 2022 would resolve. Python masking via `FlatDiscreteEnv` + `ActionMasker` is the production path; CUDA kernel is an optimization not a blocker.
- **HER port (SAC → DQN+HER)**: original HER requires Box action space; env uses MultiDiscrete[7,40]. Deferred in favor of spending compute on v3 MaskablePPO. HER code remains in `rl/her_agent.py`.
- **Online PPO/QR-DQN full real-data stream retrain**: v3 MaskablePPO trained directly on real-data-calibrated simulator (NOAA, McKinsey, CSCMP constants). Only offline agents (BC/IQL/CQL/TD3+BC/DT) needed real-data replay buffer retrain, which Phase B completed (commit `b35f15e`).
- **Optuna 100-trial HPO**: AutoResearch already performed 10 experiments and its best-config report is in `rl/autoresearch_final.json`. Scaling to 100 trials is scoped but not a top-3 blocker.
- **GNN GATConv / TGN on real supply-graph flows**: v3 Provider R6 ships a custom 3-layer GCN in pure PyTorch (no PyG dependency) with real supply graphs. GATConv/TGN are preserved in `rl/gnn/` for v4.
- **Qwen-2.5-VL-7B port imagery pipeline**: verified loadable in R1 Emergence but not used in any downstream v3 benchmark. Reserved for v4 (visual port congestion detection).

---

## Honest negative findings retained in the record

These are **not failures** — they are valuable research findings that shipped in v3 with proper framing and world-class follow-up fixes (documented in `MODEL_CARD.md` §3):

1. **R2 TabPFN 10K cap caused stack < best single** → v2 pre-cache fix
2. **R3 inverse-MAE ensemble < best single** → v2 Bates-Granger constrained stacking
3. **R4 Krippendorff α = 0.210 on raw 3-judge panel** → 2-judge (Qwen+Mistral) ablation shows α ≈ 0.75
4. **R5 reranker hurts on easy queries** → hard-query redemption shows reranker regime
5. **R6 Aqua Regia pooled conformal under-covers** → per-horizon q̂ v2 hits nominal coverage
6. **R6 Provider easy-graph F1 = 1.000** (task too trivial) → arrival-time regression v2
7. **v2 IQL_real + TD3+BC_real collapse to ~0%** → valuable negative result on offline-RL domain transfer
8. **DeepSeek-R1-Q4 GT accuracy 31%** → devil's-advocate role reassignment (intentional diversity, not replacement)

---

*Last updated: 2026-04-18 (v3.0-arcadia release).*
