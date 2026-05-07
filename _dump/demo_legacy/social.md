# Social thread drafts (pre-judging social proof)

Three drafts tailored to each platform. **Post 24–36 hours before hackathon judging closes** for max discovery.

---

## Twitter / X (8 tweets — thread)

### Tweet 1 (hook)
> Supply chain disruptions cost $184B in 2023. Suez blockage alone was $9.6B/day.
> 
> Existing tools (SAP IBP, Oracle, Resilinc) are reactive dashboards.
> 
> I built SupplyMind v3.0-arcadia: 13 SOTA models predicting disruption risk in 12 seconds — all running **locally on a laptop**.
> 
> 🧵👇

### Tweet 2 (the stack)
> The stack:
> • 3-judge LLM panel: DeepSeek-R1 + Qwen-2.5-14B + Mistral-Nemo (Q4_K_M quantized)
> • RAG: BGE-M3 + mxbai + Snowflake + BGE-reranker + HyDE
> • Forecasting: Chronos-Bolt + TimesFM-2 + ARIMA + Prophet
> • RL: MaskablePPO on MultiDiscrete[7,40]
> • GNN: custom 3-layer GCN in pure PyTorch
> 
> All local. Zero API cost.

### Tweet 3 (real data)
> 261,175 REAL data points from 8 cited sources:
> • DataCo Kaggle: 180K supply-chain orders
> • NOAA IBTRACS: 243K storm records (1884–2024)
> • FRED Economic: 17K commodity/FX points
> • World Bank WGI: 214 countries × 24 years
> • SEC 10-K filings: 25 Fortune 500
> 
> Zero synthetic. Zero simulated. All cited.

### Tweet 4 (the RL result)
> The most striking benchmark:
> 
> On medium & hard supply-chain tasks, a greedy heuristic performs **WORSE than random**.
> 
> PPO_v3 flips the sign: -1.81 → +2.78 (medium), -1.41 → +2.65 (hard).
> 
> 8,100-episode bootstrap. CI95 non-overlapping. Zero constraint violations.

### Tweet 5 (RAG result)
> 6,483 real chunks (SEC + Wikipedia). 73 queries.
> 
> mxbai-embed-large: P@1 = 0.962, MRR = 0.978 on precise queries.
> 
> On paraphrased HARD queries, bi-encoder drops to 0.70.
> BGE-reranker picks up **+5pp**.
> 
> Right tool, right regime — not "reranker always helps."

### Tweet 6 (honest findings)
> What I love most: every negative finding IMPROVED instead of hidden.
> 
> • R4 α=0.21 → 0.75 on 2-judge panel
> • R3 inverse-MAE → Bates-Granger constrained stacking (9/21 wins)
> • R6 pooled conformal → per-horizon q-hat (oil @ 95%: dev 0.024 vs 0.112)
> • R6 F1=1.000 triviality → arrival-time regression (+48–64% GNN lift)

### Tweet 7 (engineering)
> PyTorch engineering highlights:
> • Custom GCN from scratch (no torch_geometric)
> • MaskablePPO Discrete(280) flatten wrapper
> • Q4_K_M quantization: DeepSeek 15GB → 4.5GB (fits 12GB VRAM)
> • ONNX export: 0.97 MB per policy
> • 173 passing tests (incl. 19 formal OpenEnv compliance)

### Tweet 8 (CTA)
> SupplyMind v3.0 "Even In Arcadia" — Meta PyTorch OpenEnv Hackathon.
> 
> Every phase commit named after a Sleep Token track 🎵
> 
> 🔗 GitHub: github.com/ShAuRyA-Noodle/Sleep-Token
> 🤗 HF Space: huggingface.co/spaces/Shaurya-Noodle/Supplymind
> 📺 Demo: [video link]
> 
> #PyTorch #OpenEnv #SupplyChain #RL #LLM

---

## LinkedIn (long-form, single post)

**Title**: Building SupplyMind v3.0-arcadia — 13 SOTA Models, 100% Local, for the Meta PyTorch OpenEnv Hackathon

Supply chain disruptions cost the global economy $184 billion in 2023. The 2021 Suez blockage cost $9.6B per day. The 2020-2023 chip shortage erased $210B in automotive revenue. Existing enterprise tools (SAP IBP, Oracle SCM, Resilinc) tell you *after* it breaks.

For the Meta PyTorch OpenEnv Hackathon, I built SupplyMind v3.0-arcadia — an OpenEnv-compliant supply-chain risk management environment with a 13-model AI stack, all running locally on a 12GB-VRAM laptop.

**What's in the box:**
• 3-judge LLM risk panel (DeepSeek-R1, Qwen-2.5-14B, Mistral-Nemo — all Q4_K_M quantized, all local via Ollama)
• RAG with 3 embedders (BGE-M3, mxbai-embed-large, Snowflake-Arctic) + BGE-reranker + HyDE
• Forecasting ensemble (Chronos-Bolt + TimesFM-2 + ARIMA + Prophet + Bates-Granger stacking)
• Reinforcement Learning via MaskablePPO on a 408-dim, MultiDiscrete[7,40] action space (280 valid joint actions)
• Custom 3-layer Graph Convolutional Network in pure PyTorch — no torch_geometric dependency
• Split-conformal prediction intervals with per-horizon calibration
• FastAPI + MCP JSON-RPC + WebSocket OpenEnv-compliant server with 12 endpoints

**Real data only:** 261,175 records from 8 cited sources — DataCo (Kaggle), NOAA IBTRACS, USGS, FRED, World Bank WGI, SEC 10-K filings, Wikipedia crisis articles, FRBSF/BIS policy papers.

**Statistical rigor:** Wilcoxon signed-rank p<0.001 for every RL-vs-baseline comparison. Bootstrap 95% CIs. Krippendorff α, Fleiss κ, Cohen weighted κ, ECE for calibration, PICP for interval coverage.

**Honest findings improved, not hidden:** When the initial 3-judge Krippendorff α was 0.21, I didn't hide it — I ran a 2-judge ablation (α climbs to 0.75) and added a deterministic rubric-agent as human-baseline. When the first-pass reranker underperformed bi-encoder on easy queries, I added 20 paraphrased hard queries where the reranker earns a +5pp P@1 lift.

**173 passing tests** (including 19 formal OpenEnv compliance tests) in 2m14s.

Every commit is named after a Sleep Token track from the albums "Even In Arcadia" and "Take Me Back to Eden" — Emergence, Caramel, Past Self, Dangerous, Granite, Gethsemane, Provider, Aqua Regia, Damocles, Infinite Baths, Euclidian, Arcadia.

Full project, MIT-licensed, reproducible in 3 commands:
`github.com/ShAuRyA-Noodle/Sleep-Token`

Live demo on HuggingFace Spaces:
`huggingface.co/spaces/Shaurya-Noodle/Supplymind`

Would love feedback from anyone working on:
• Supply chain analytics
• OpenEnv environment design
• Local LLM quantization & multi-model orchestration
• Real-world-calibrated RL benchmarks

#MetaPyTorch #OpenEnv #SupplyChain #ReinforcementLearning #LLM #PyTorch #MLOps #RealData

---

## Hacker News (Show HN post)

**Title**: Show HN: SupplyMind v3.0 — 13 SOTA models for supply-chain risk, 100% local

**Body**:

I built SupplyMind v3.0-arcadia for the Meta PyTorch OpenEnv Hackathon. The goal: show that you can build a production-grade supply-chain risk-management environment with 13 state-of-the-art AI models running entirely on a 12GB-VRAM laptop.

Repo: https://github.com/ShAuRyA-Noodle/Sleep-Token
HF Space: https://huggingface.co/spaces/Shaurya-Noodle/Supplymind
Tag: v3.0-arcadia (180+ commits, 22 Sleep-Token-named phases)

Design principles:
1. **Zero cloud API calls at inference.** All LLMs local via Ollama + Q4_K_M. Chronos/TimesFM/embedders direct PyTorch.
2. **Real data only.** 261K records from 8 cited sources. Zero synthetic substitution for any headline number.
3. **OpenEnv-compliant.** Full spec: Pydantic v2 action/observation, 12 HTTP endpoints including MCP JSON-RPC and WebSocket, deterministic graders (5×-run zero variance), openenv.yaml manifest, 19 formal compliance tests.
4. **Honest negatives.** Every negative finding has a documented world-class improvement (not reframed — improved).

Things I'm particularly happy with:

- Custom 3-layer GCN from scratch (no torch_geometric). Pure PyTorch `index_add_` message passing.
- MaskablePPO wrapper that flattens MultiDiscrete[7,40] → Discrete(280) so sb3-contrib's flat-mask works. Zero constraint violations across 8,100 eval episodes.
- Two-pass DeepSeek-R1 extraction (free CoT → Qwen-14B JSON extractor) achieving 100% parse rate.
- Per-horizon split-conformal prediction intervals that hit nominal coverage within ±2pp on heavy-tailed oil prices.
- 173 passing tests including 19 OpenEnv compliance in 2m14s on CPU.

Would love feedback on architecture, OpenEnv compliance approach, or local LLM engineering on constrained hardware.
