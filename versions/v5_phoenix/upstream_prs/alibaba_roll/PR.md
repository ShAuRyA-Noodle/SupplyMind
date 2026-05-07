# PR draft — alibaba/ROLL: SupplyMind as an agentic-RL reference example

**Target**: https://github.com/alibaba/ROLL
**Branch name**: `add-supplymind-crisis-agentic-example`
**Status**: draft (will open during 48 h finals)

---

## Title

`Add examples/supplymind_crisis: agentic RL for supply-chain risk (GiGPO, 3-judge reward, Qwen-2.5)`

## Body

### Summary

Adds `examples/supplymind_crisis/` — a working agentic-RL example using
ROLL's pipeline infrastructure applied to a real supply-chain risk
environment. Demonstrates:

- **Custom env integration** via `supplymind_roll_env.py` (registers with
  ROLL's agentic env factory).
- **GiGPO step-wise training** on a multi-turn decision task.
- **Custom `LLMJudgeRewardWorker` bridging a 3-judge panel** (DeepSeek-R1,
  Qwen-2.5-14B, Mistral-Nemo) as the reward signal.
- **Dual training path**: ROLL DPO pipeline (`dpo_qwen25_3b_supplymind.yaml`)
  + `trl` fallback for machines where the full ROLL stack isn't installable.
- **Single-GPU 12 GB VRAM configuration** (HF strategy + LoRA), supplementing
  the existing multi-node configs.

### What's in this PR

```
examples/supplymind_crisis/
├── README.md                              # env overview + recipes
├── supplymind_roll_env.py                 # env factory registered as 'supplymind_crisis'
├── supplymind_judge_worker.py             # 3-judge reward backend
├── configs/
│   ├── dpo_qwen25_3b_supplymind.yaml      # DPO fine-tune the judge
│   └── agentic_supplymind_gigpo.yaml      # multi-turn agentic RL
├── data/
│   └── preference_pairs.jsonl             # 26 DPO pairs from crisis scenarios
├── tests/
│   ├── test_env_loop.py
│   └── test_reward_bridge.py
└── benchmarks/
    └── V5_DPO_JUDGE_accuracy_delta.json   # baseline vs DPO adapter
```

### Why this belongs in alibaba/ROLL

- **Fills a gap in the examples directory**: ROLL currently has math/code/QA
  agentic envs (GEM, Sokoban, FrozenLake). A supply-chain risk env is a
  structured, real-world, industrial agentic scenario with none of the
  usual game-dynamics artifacts.
- **Demonstrates single-GPU ergonomics**: most existing configs assume
  8×A100. We show that `strategy_name: hf` + LoRA works cleanly on a 12 GB
  laptop, lowering the barrier for new adopters.
- **Shows the `LLMJudgeRewardWorker` extension path**: other teams can
  mirror the 3-judge panel pattern for domain-specific reward signals.

### How to run (after merge)

```bash
# DPO fine-tune the judge (single-GPU, ~3h on RTX 4080)
python -m roll.pipeline.dpo \
    --config examples/supplymind_crisis/configs/dpo_qwen25_3b_supplymind.yaml

# Multi-turn agentic RL with GiGPO
python -m roll.pipeline.agentic \
    --config examples/supplymind_crisis/configs/agentic_supplymind_gigpo.yaml

# Smoke tests
python -m pytest examples/supplymind_crisis/tests -q
```

### Attribution

Author: ShAuRyA-Noodle (solo, Meta PyTorch OpenEnv Hackathon 2026 finalist).
The underlying SupplyMind environment is itself submitted to
meta-pytorch/openenv in a parallel PR.

### Open questions for maintainers

1. GiGPO config gamma/lambda values — we used defaults (γ=0.99, λ=0.95).
   Does alibaba/ROLL have preferred step-wise defaults for sparse-reward
   multi-turn envs?
2. Reward bridge queries Ollama by default; want us to add a vLLM path too?
3. Preference-pair generation code is in our Phoenix repo; should we also
   add a generator script here that pulls from arbitrary crisis JSON
   libraries?

### License

Apache-2.0 (matches ROLL's license).

---

## Pre-merge checklist for me

- [ ] Fork alibaba/ROLL to ShAuRyA-Noodle/ROLL
- [ ] Create branch `add-supplymind-crisis-agentic-example`
- [ ] Copy `examples/supplymind_crisis/` materials from this folder into the branch
- [ ] Verify `pytest examples/supplymind_crisis/tests -q` passes in the fork
- [ ] Smoke test at least one config runs to `max_steps: 10` without crash
- [ ] Open PR; link SupplyMind main repo, mention OpenEnv parallel PR
- [ ] Tag any relevant CNA / Alibaba maintainers politely if known

## File copy map

```
versions/v5_phoenix/roll_integration/env/supplymind_roll_env.py         -> examples/supplymind_crisis/supplymind_roll_env.py
versions/v5_phoenix/roll_integration/reward_bridge/supplymind_judge_worker.py -> examples/supplymind_crisis/supplymind_judge_worker.py
versions/v5_phoenix/roll_integration/configs/dpo_qwen25_3b_supplymind.yaml -> examples/supplymind_crisis/configs/dpo_qwen25_3b_supplymind.yaml
versions/v5_phoenix/roll_integration/configs/agentic_supplymind_gigpo.yaml -> examples/supplymind_crisis/configs/agentic_supplymind_gigpo.yaml
versions/v5_phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl -> examples/supplymind_crisis/data/preference_pairs.jsonl
versions/v5_phoenix/upstream_prs/alibaba_roll/README.crisis.md          -> examples/supplymind_crisis/README.md
```

Build script: `versions/v5_phoenix/upstream_prs/alibaba_roll/build_pr_branch.sh`.
