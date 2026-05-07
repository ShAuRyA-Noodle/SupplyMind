# `trl_fallback/` — when to reach for this

This directory is a symbolic "use these modules if ROLL isn't installed"
signpost, not a separate implementation. The trl-based fallback code
already lives in `../dpo_judge/train_dpo_trl.py` — it's the ROLL-free
path from day one.

## Reach for the trl fallback when

- `pip install -e ../vendor/ROLL/[hf]` fails on Windows AND
- `wsl --install` + the Phase-B WSL2 stack fails OR the user decides the
  WSL2 route isn't worth the time left on the clock.

## What's identical

- The preference-pair format (`prompt`, `chosen`, `rejected` JSONL)
- The adapter format (LoRA safetensors + adapter_config.json)
- The evaluation script (`evaluate_delta.py`) — agnostic to training path
- The receipt (`V5_DPO_JUDGE_accuracy_delta.reproduce.sh`) — just swaps
  which `train_dpo_*.py` to invoke

## What's lost

| Capability | Loss if trl-only |
|---|---|
| Alibaba/ROLL upstream PR | PR draft still ships; env code still valid; but no runnable demo in ROLL pipeline |
| GiGPO agentic multi-turn training | Deferred — trl doesn't have GiGPO |
| Async reward computation | Sync only (acceptable for 26 scenarios × 3 judges) |
| ROLL's config-driven experiment inheritance | Hydra configs still ship in `../configs/`, just not consumed |

Scientific result (fine-tuned Qwen-2.5-3B judge with measurable delta vs
baseline) is unchanged.
