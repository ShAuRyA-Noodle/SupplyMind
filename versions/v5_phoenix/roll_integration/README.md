# ROLL integration — Phoenix v5

## What this is

Alibaba ROLL (`github.com/alibaba/ROLL`, Apache 2.0, v0.2.1) is an enterprise-grade
RL framework for LLM post-training at scale. Phoenix v5 integrates it in three
targeted ways, each shipping its own receipt, each with a graceful fallback
that doesn't require ROLL to be installed.

## The three integrations

### 1. DPO judge fine-tuning (`dpo_judge/`)

Take our 26 R4 crisis scenarios + GT labels, turn them into preference pairs
(`chosen` = correct judge response, `rejected` = incorrect), DPO-fine-tune
Qwen-2.5-3B-Instruct with LoRA r=8. Receipt: `V5_DPO_JUDGE_accuracy_delta.reproduce.sh`.

Expected delta: +5 to +15 pp accuracy on R4 scenarios. If negative, we publish
the null (per the no-compromise policy).

**Two paths**:
- `train_dpo_trl.py` — standalone HuggingFace trl (works without ROLL)
- `train_dpo_roll.py` — uses ROLL's production DPO pipeline (requires ROLL install)

Both produce the same adapter format, so downstream `evaluate_delta.py` is path-agnostic.

### 2. SupplyMind as a ROLL env (`env/`)

`supplymind_roll_env.py` wraps `server.supply_environment.SupplyMindEnvironment`
in ROLL's expected agentic-env interface (`reset/step/grade` + `env_id`, `tags`,
`supports_step_reward`). Auto-registers with ROLL at import time if ROLL is
present; works standalone for testing if ROLL is absent.

This is the basis of the **Alibaba/ROLL upstream PR** (see `upstream_prs/alibaba_roll/`).

### 3. Reward bridge (`reward_bridge/`)

`supplymind_judge_worker.py` is a drop-in subclass of ROLL's
`LLMJudgeRewardWorker` that wraps our existing 3-judge panel (DeepSeek-R1-Q4,
Qwen-2.5-14B-Q4, Mistral-Nemo-Q4, Ollama-served). Reward formula matches R4
V2's majority-vote accuracy scoring. Plugs into any ROLL RLVR or agentic
config via `reward.backend: supplymind_3judge`.

## Configs

| File | Purpose | Target algorithm |
|---|---|---|
| `configs/dpo_qwen25_3b_supplymind.yaml` | DPO judge training | DPO (sigmoid loss, beta=0.1) |
| `configs/agentic_supplymind_gigpo.yaml` | Multi-turn agent training | GiGPO (step-wise) |

Both configs use `strategy_name: hf` (HuggingFace strategy) because the
RTX 4080 Laptop 12GB can't host Megatron TP/PP.

## Install

See `INSTALL.md` for the Phase A / Phase B / Phase C decision flowchart.

TL;DR:
1. Phase A (Windows-native pip, 30 min)
2. Phase B (WSL2 + CUDA, full day)
3. Phase C (standalone trl, always works)

## Why this matters for the hackathon

Three judge-facing signals:

1. **Real LLM post-training** — not prompt-engineering. DPO adapter is a 20MB
   file we can ship on HF Hub and judges can download + verify.
2. **Dual open-source impact** — upstream PR to Meta's OpenEnv *and* Alibaba's
   ROLL. Hackathon page says "code ships to Meta-backed projects"; we go one
   better.
3. **Reproducibility** — every ROLL artifact has a companion trl fallback, so
   reviewers reproducing on non-Linux machines aren't blocked.

## What's NOT in scope for v5

- Megatron 5D parallelism (single-GPU, out of scope)
- Multi-node distributed training
- VLM distillation (defer; Qwen-VL is already in v4 via port imagery)
- Full ROLL Flash async — we use sync `HFStrategy` for simplicity
