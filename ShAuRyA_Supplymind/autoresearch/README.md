# ShAuRyA_Supplymind/autoresearch — Karpathy-style autonomous research loop

> "Letting an AI agent systematically explore a narrow modification space overnight is more productive than manual hyperparameter tuning." — Karpathy

This directory implements [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for supply-chain RL.

## The pattern (Karpathy's core insight)

```
program.md (skill spec)
    │
    ▼
LLM agent (Qwen-14B local or Claude)
    │ reads program.md + current candidate_train.py + last N experiment results
    ▼
Proposes a unified diff of candidate_train.py
    │
    ▼
Fixed-budget runner (50k steps, 10 min max)
    │
    ▼
Evaluator (single metric: bootstrap CI95 lower across 3 tasks × 3 seeds)
    │
    ▼
Accept (new_ci95_lower > best + 0.005)?
    │
    ├─ YES → keep diff, update best, log to AUTORESEARCH_LAB_NOTEBOOK.md
    └─ NO  → revert candidate_train.py, log to AUTORESEARCH_REJECTED.md
    │
    ▼
Loop until time budget exhausted
```

## Files

| File | Purpose | Modifiable by agent? |
|---|---|---|
| `program.md` | Skill specification — the contract. | ❌ Human only |
| `candidate_train.py` | The RL training script. Agent mutates inside SAFE-TO-MODIFY markers. | ✅ Agent |
| `hypothesis_engine.py` | Generates hypothesis + diff using Qwen-14B (Ollama) or Claude (API). | ❌ Fixed |
| `runner.py` | Executes candidate_train.py with fixed budget + safety guards. | ❌ Fixed |
| `evaluator.py` | Runs the 9-episode eval, computes bootstrap CI95 lower. | ❌ Fixed |
| `lab_notebook.py` | Auto-generates lab notebook entries. | ❌ Fixed |
| `orchestrator.py` | Main loop: propose → run → eval → accept/reject → log. | ❌ Fixed |
| `seed_experiments.py` | 5 hand-crafted starter hypotheses to bootstrap the loop. | ❌ Fixed |
| `state.json` | Persistent state: current best, history, diff chain. | auto |
| `experiments/` | Per-experiment outputs (diff, metric, log, checkpoint, plots). | auto |
| `AUTORESEARCH_LAB_NOTEBOOK.md` | Accepted experiments, sorted by improvement. | auto |
| `AUTORESEARCH_REJECTED.md` | Rejected experiments with reasons. | auto |

## Quick start

```bash
# One-shot: run autoresearch for 6 hours
python -m ShAuRyA_Supplymind.autoresearch.orchestrator --budget 6h

# Quick sanity check: run 3 seed experiments (no LLM, no mutation)
python -m ShAuRyA_Supplymind.autoresearch.orchestrator --seeds-only

# Use Claude API instead of local Qwen (faster hypothesis generation)
python -m ShAuRyA_Supplymind.autoresearch.orchestrator --agent claude --budget 6h

# Resume from existing state
python -m ShAuRyA_Supplymind.autoresearch.orchestrator --resume

# Graceful halt
touch ShAuRyA_Supplymind/autoresearch/stop_autoresearch.flag
```

## Safety guards (not in Karpathy's original)

RL is messier than LLM training. We add:

1. **Wall-clock kill**: if a single experiment runs > 10 min, SIGTERM it.
2. **OOM guard**: torch.cuda.empty_cache() between experiments; abort if VRAM < 2 GB.
3. **NaN guard**: if loss hits NaN, reject immediately.
4. **Test gate**: `pytest tests/ -q` must still pass after any accepted change. If it fails, the diff is reverted and logged.
5. **Seed hash check**: eval seeds (42, 99, 7) must never match any training seed. Orchestrator asserts this on every experiment.
6. **Diff size limit**: agent-proposed diffs ≤ 150 LOC changed. Larger diffs are rejected pre-run (too risky, too much at once).
7. **Signature lock**: `run_experiment(seed, total_steps) -> dict` signature is frozen. Any diff that changes it is rejected.

## The metric

`bootstrap_ci95_lower(grader_scores)` where `grader_scores` is a length-9 array (3 tasks × 3 seeds).

Why CI95 lower and not mean?
- Mean gets fooled by lucky seeds.
- CI95 lower is the conservative "worst-case plausible performance" — exactly what a risk-aware supply-chain manager cares about.
- It aligns with our R6 Euclidian bootstrap methodology.

## Reference

Karpathy's repo: https://github.com/karpathy/autoresearch

Paper / thread by Karpathy: https://x.com/karpathy/status/... (autoresearch announcement)

The core idea is *not* to outperform a human researcher on any single experiment — it's to run **100 experiments overnight** while the human sleeps, so the search space is explored 10× denser.
