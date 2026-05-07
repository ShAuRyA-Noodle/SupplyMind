---
name: autoresearch-experiment
description: Use when running an automated research loop that proposes, executes, and judges ML experiments (hyperparameter / architecture / algorithm variants). Enforces Karpathy-pattern single mutable file + fixed budget + single metric + bootstrap CI95 accept/reject, with lab notebook auto-maintenance.
---

# Autoresearch Experiment

## The iron law

ONE MUTABLE FILE. ONE METRIC. BOOTSTRAP CI95. NOTEBOOK BEFORE DECISION.

If you're running multiple experiments by hand, copy-pasting configs, and
eyeballing means, you'll cherry-pick. This skill structures the loop so you
can't.

## When to invoke

- Any time you'd run 3+ training variants and want to pick one
- Ablation studies (hyperparams, architecture, reward shaping, curriculum)
- "Does X beat Y?" questions with stochastic training
- Before committing to a config for a downstream receipt

## When NOT to invoke

- One-off debugging runs
- Deterministic experiments (just run once)
- Exploratory runs where you don't have a comparison yet

## Setup — the six files

```
autoresearch/
├── program.md              # formal contract (frozen; what the agent MUST preserve)
├── candidate_train.py      # the ONE mutable file; has # SAFE TO MODIFY markers
├── seed_experiments.py     # 3-5 hand-crafted mutator functions for cold-start
├── runner.py               # subprocess executor with wall-clock timeout + NaN scrape
├── evaluator.py            # bootstrap CI95 + decide(new_scores) -> Decision
├── lab_notebook.md         # append-only narrative (hypothesis, result, reasoning)
└── state.json              # {best: ..., history: [...]}
```

## The loop

```
for seed in seed_experiments + agent_generated:
    1. orchestrator reads state.json, composes a hypothesis with justification
    2. orchestrator applies mutator(old_code) -> new_code; compile-syntax-checks
    3. orchestrator writes candidate_train.py.bak, then new_code
    4. runner.run_candidate(seed, budget=50k steps, timeout=10min)
         - VRAM pre-check (min 2GB free)
         - subprocess with stdout/stderr captured
         - reads result.json from exp_dir/
    5. evaluator.decide(new_scores, new_name)
         - status != ok -> REJECT
         - scores empty -> REJECT
         - first successful -> ACCEPT (seed baseline)
         - delta_ci95_lower > 0.005 -> ACCEPT
         - else -> REJECT
    6. lab_notebook append: hypothesis, wall_clock, scores, decision, reasoning
    7. if ACCEPT: state.best = new; else: revert candidate_train.py from .bak
```

## The metric — bootstrap CI95 lower

```python
def bootstrap_ci95_lower(scores, n=1000, seed=12345):
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(scores, size=len(scores), replace=True).mean()
    return np.percentile(means, 2.5)   # <-- this is the metric
```

**Why the lower bound, not the mean**: protects against cherry-picked means on
small samples (n=9 for us: 3 tasks × 3 held-out seeds). A mean can win by
+0.05 while the CI95 lower is flat — that's noise, not signal.

## Accept epsilon

`ACCEPT_EPSILON = 0.005` — 0.5 percentage-point delta on CI95 lower.

Lower than typical ablation thresholds because the budget is small and the
cost of a false-accept (taking a worse hypothesis) is: the next hypothesis
starts from a worse baseline. Conservative by design.

## Held-out eval seeds — non-negotiable

```python
EVAL_SEEDS = (42, 99, 7)          # never used for training
EVAL_TASKS = (easy, medium, hard) # frozen at program.md write time
```

If the agent ever uses 42/99/7 for training, the contract is broken, the run
is void, and the lab notebook marks it `HOLDOUT_LEAKAGE=true`. This is how
you detect reward hacking.

## Lab notebook format

Every experiment gets an entry:

```markdown
### S<N> — `<experiment_name>` — **ACCEPTED|REJECTED|PENDING**

**Hypothesis**: <one sentence>
**Justification**: <why, with references>
**Expected delta**: <range>

**Outcome** (<budget> steps, wall <time>s):
- grader_scores = [...]
- mean = X, std = Y
- bootstrap 95% CI on mean: [low, high]
- delta vs best on ci95_lower: <signed>
- decision: **<verb>** (<reason>)

**Reading between the lines**: <what the data tells you beyond pass/fail>
```

## Anti-patterns

- Running with different seeds and picking the best
- Using a mean delta instead of CI95 lower
- Re-running a failed experiment without changing the code
- Editing candidate_train.py outside the SAFE markers
- Changing EVAL_SEEDS between experiments
- Skipping the lab notebook entry "because the result was obvious"

## When the agent proposes a mutation that would break the contract

Reject immediately. Record in state.json under `invalid_mutations`. Never
execute. The contract is sacred — the whole experiment history becomes
un-interpretable if it drifts.
