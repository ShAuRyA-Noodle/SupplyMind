# examples/supplymind_crisis

An agentic RL example that uses ROLL to train a supply-chain risk analyst
LLM on a real-data env with a 3-judge LLM panel as the reward signal.

## Env

`supplymind_roll_env.py` registers SupplyMind as a ROLL agentic env
(`env_id: supplymind_crisis`). Three difficulty-calibrated tasks:

- `easy_typhoon_response` — 12 nodes, 30 steps, $5 M budget
- `medium_multi_front` — 25 nodes, 45 steps, $8 M budget
- `hard_cascading_crisis` — 40 nodes, 60 steps, $10 M budget

Multi-turn interactive: each step is "observe state → pick 1 of 7 action types
× 40 target nodes → observe outcome". Supports both step-reward (for GiGPO)
and trajectory-reward (for StarPO / PPO).

## Reward

`supplymind_judge_worker.py` is a drop-in `LLMJudgeRewardWorker` subclass that
routes reward computation through a 3-judge panel:

- DeepSeek-R1-Distill-Qwen-7B (Q4) — devil's-advocate
- Qwen-2.5-14B-Instruct (Q4) — strong baseline
- Mistral-Nemo-Instruct-2407 (Q4) — third voice

Reward ∈ [−0.2, 1.0]:
- 1.0 when ≥2 judges agree with ground-truth risk tier AND candidate matches
- 0.6 when exactly 1 judge agrees
- 0.0 when none agree
- −0.2 format penalty if candidate JSON is unparseable

## Configs

### DPO judge fine-tuning (single GPU, LoRA, ~3 h)

```bash
python -m roll.pipeline.dpo --config configs/dpo_qwen25_3b_supplymind.yaml
```

Fine-tunes Qwen-2.5-3B-Instruct on 26 preference pairs derived from real
crisis scenarios. Uses LoRA r=8 and `strategy_name: hf` (no Megatron needed).

### GiGPO agentic training (single GPU, ~6 h)

```bash
python -m roll.pipeline.agentic --config configs/agentic_supplymind_gigpo.yaml
```

Trains the fine-tuned analyst on multi-turn decision-making with step-wise
advantages. Rollout calls our FastAPI endpoints as MCP-style tools
(forecast, RAG, RL-policy).

## Data

`data/preference_pairs.jsonl` — 26 DPO pairs. Each: {`prompt`, `chosen`,
`rejected`}. Generated from the v4 crisis library using the `quality gap`
heuristic described in the parent SupplyMind repo.

## Hardware profiles

| Setup | Min VRAM | Strategy | Wall-clock |
|---|---|---|---|
| DPO Qwen-3B LoRA r=8 | 10 GB | hf + bf16 + gckpt | 3 h / 2 epochs |
| GiGPO Qwen-3B LoRA | 10 GB | hf + bf16 | 6 h / 100 steps |
| DPO Qwen-7B full | 40 GB+ | megatron TP=2 | (multi-GPU; not included here) |

## License

Apache-2.0 — matches ROLL's license. Original env is MIT under the parent
SupplyMind repo; the reward-bridge + configs are Apache-2.0 to be compatible.
