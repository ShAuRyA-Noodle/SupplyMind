# SupplyMind Ollama + Fine-Tuning Final Upgrade

This document is the evidence map for the final local-model layer. It covers
the custom Ollama models, Modelfile evolution, LoRA/QLoRA explanation tuning,
Phoenix DPO/GRPO, ROLL integration, and quantization/memory engineering. The
offline gate is:

```bash
python scripts/verify_ollama_finetuning_stack.py
```

That script checks committed files and datasets only. Live model execution is
separate because it depends on the local Ollama daemon and GPU.

## A.1 Custom Ollama Models

| Model | Evidence | Role |
|---|---|---|
| `supplymind-analyst:v1` | `rl/lora/Modelfile` | First custom analyst prompt on Qwen 2.5 14B with TSMC, Red Sea, action-cost, SLA, reward-weight, and 5 environment examples. |
| `supplymind-analyst:v2` | `rl/lora/Modelfile.v2` | Adds executive Markdown structure, richer Tohoku/Suez/chip-shortage/DataCo knowledge, and stricter evidence/counterfactual/precedent sections. |
| `supplymind-analyst:v3` | `rl/lora/Modelfile.v3` | Expands action costs, SLA, Red Sea, Panama, Ukraine/neon, WGI, and 10-shot explanation coverage. |
| `supplymind-analyst:v4` | `rl/lora/Modelfile.v4` | Switches to strict JSON for Phase 4/R3-style LLM block use. |
| `supplymind-analyst:v5` | `ShAuRyA_Supplymind/features/Modelfile.analyst_v5` | Final calibrated JSON judge with 8 hard-negative few-shots and explicit LOW/MEDIUM/HIGH/CRITICAL rules. |
| `qwen25-14b-local` | `v3_arcadia/00_emergence/qwen25-14b.Modelfile` | Offline Qwen judge wrapper. |
| `qwen25-coder-local` | `v3_arcadia/00_emergence/qwen25-coder-14b.Modelfile` | JSON/code-specialist wrapper. |
| `mistral-nemo-local` | `v3_arcadia/00_emergence/mistral-nemo.Modelfile` | Long-context 128K judge wrapper. |
| `deepseek-r1-local-q4` | `v3_arcadia/00_emergence/deepseek-r1.Modelfile` | Quantized DeepSeek-R1 devil's-advocate judge. |

Creation tooling:

```bash
python -m rl.lora.create_ollama_model --version v5 --test
python -m rl.lora.create_ollama_model --all
python -m rl.lora.create_ollama_model --wrapper qwen25-coder-local
```

The creator sets `OLLAMA_MAX_LOADED_MODELS=1` when running `ollama create`.

## A.2 Modelfile Crafting

The five committed analyst Modelfiles cover the full progression:

- Domain facts: TSMC 54 percent foundry share, 92 percent advanced nodes, Tohoku Toyota loss, Suez $9.6B/day and 400+ vessels, chip shortage $210B, Red Sea +10 to 14 days and +25 percent fuel, Hormuz/Brent v5 crisis facts.
- Prompt examples: v1 uses real environment explanation examples from `rl/data/lora_training_data.json`; v5 uses 8 hard-negative few-shots across LOW through CRITICAL.
- Determinism: v4 uses `temperature 0.1`; v5 uses low `temperature 0.15` with calibrated confidence rules.
- Controlled diversity: all analyst Modelfiles keep `top_p 0.9`.
- Context: v5 ships `num_ctx 16384`; older versions ship 8192 and are documented as historical iterations.
- Parseability: v4 and v5 require JSON-only output; v5 smoke test parses required keys.

## A.3 LoRA / QLoRA Fine-Tuning

`rl/lora/finetune.py` now has a real QLoRA path:

- Base: `Qwen/Qwen2.5-1.5B` by default.
- Data: 225 instruction/output records in `rl/data/lora_training_data.json`.
- Stack: PEFT LoRA + TRL `SFTTrainer`.
- Quantization: bitsandbytes 4-bit NF4 via `BitsAndBytesConfig`.
- Adapter: saves only `rl/checkpoints/lora/supplymind_lora/`, plus `supplymind_lora_manifest.json`.
- Isolation: intended command stays under `.venv311` as documented in the script header.

Command:

```bash
.venv311\Scripts\python.exe -m rl.lora.finetune --model Qwen/Qwen2.5-1.5B --quantization nf4
```

## A.4 Phoenix DPO + GRPO

Phoenix v5 keeps DPO/GRPO in `ShAuRyA_Phoenix/roll_integration/dpo_judge/`:

- `prepare_preference_data.py`: builds 21 chosen/rejected pairs from R4 ground truth and local judge outputs.
- `data/preference_pairs.jsonl`: committed 21-pair dataset.
- `train_dpo_trl.py`: standalone TRL fallback with Qwen 2.5 3B, DPO beta 0.1, LoRA `r=8`, alpha 16, batch 1, grad accum 4, LR `5e-5`.
- `train_dpo_roll.py`: ROLL DPO path using `configs/dpo_qwen25_3b_supplymind.yaml`.
- `train_grpo_env.py`: standalone GRPO/RLVR reward prototype.
- `train_grpo_live_env.py`: live env-connected GRPO; every reward goes through HTTP `POST /analyst/grade`.
- `evaluate_delta.py`: base-vs-adapter accuracy delta evaluator for the current dict-shaped R4 cache.

The DPO strategy is explicitly `hf`, not Megatron, so it stays realistic on a
single 12 GB GPU.

## A.5 ROLL Integration

ROLL integration is represented by three pieces:

- `env/supplymind_roll_env.py`: `SupplyMindRollEnv`, step reward capable, trajectory reward capable, importable even when ROLL is absent.
- `reward_bridge/supplymind_judge_worker.py`: `SupplyMind3JudgeRewardWorker`, using DeepSeek/Qwen/Mistral local judges with guarded ROLL registration.
- `configs/agentic_supplymind_gigpo.yaml`: GiGPO multi-turn config with `forecast`, `rag`, and `rl_act` tools and `step_reward: true`.

The ROLL path is a real integration surface, not a hard dependency for normal
repo tests. If ROLL is absent, the TRL fallback remains the executable training
path.

## A.6 Quantization + Memory Engineering

Evidence:

- `v3_arcadia/results/R1_VERIFIED.json`: records Q4_K_M formats, 3.3x compression rationale, <2 percent quality-loss claim source note, and BGE safetensors rationale.
- `v3_arcadia/00_emergence/convert_bge_to_safetensors.py`: converts BGE-M3 `pytorch_model.bin` to `model.safetensors` to avoid unsafe `torch.load` behavior under CVE-2025-32434 constraints.
- `v3_arcadia/40_granite/r5_rag_beast.py`: unloads Ollama and clears CUDA memory around RAG phases to prevent VRAM thrash.
- `rl/lora/create_ollama_model.py`: enforces one loaded Ollama model during creation.

Live checks to run on the GPU machine:

```bash
ollama list
python scripts/verify_ollama_finetuning_stack.py
python -m rl.lora.create_ollama_model --version v5 --test
python -m ShAuRyA_Phoenix.roll_integration.dpo_judge.train_grpo_live_env --env-url http://localhost:8000 --dry-run
```

No synthetic substitution is introduced by this upgrade. The verifier checks
committed evidence; live model quality still must be demonstrated with the
runtime commands above when Ollama and the GPU are available.
