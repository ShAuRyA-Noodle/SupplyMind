# SupplyMind Autoresearch Program Specification

> Adapted from Karpathy's `karpathy/autoresearch` pattern. This markdown file IS the skill interface — the LLM agent reads this, proposes modifications to `candidate_train.py`, runs a fixed-budget training job, and the orchestrator accepts or rejects based on a single metric.

## Task

Train a reinforcement learning policy that maximizes the **grader score** on the SupplyMind OpenEnv environment. The environment models a 40-node global supply chain navigating cascading crises (typhoons, port strikes, geopolitical escalation). Action space: `MultiDiscrete([7, 40])` → 280 discrete actions. Observation space: 408 floats + action mask.

## Single metric (strict)

```
metric = bootstrap_ci95_lower(grader_scores_across(3_tasks × 3_seeds))
```

Where `grader_scores_across` returns 9 scalar scores in [0, 1]. We use the **bootstrap CI95 lower bound** (not the mean) as the accept criterion. This is Karpathy's "single metric" principle with an anti-noise wrapper: a change is accepted only if its *worst-case plausible* performance beats the current best's worst-case plausible performance.

**Accept if**: `new_ci95_lower > best_ci95_lower + 0.005`
**Reject otherwise** — revert `candidate_train.py` to prior state.

## Fixed budget (hard)

- **50,000 environment steps** per experiment
- **10 minutes wall-clock max** (kill if exceeded)
- **3 eval seeds** (42, 99, 7) × **3 tasks** (easy, medium, hard) = **9 episodes per evaluation**

These numbers are platform-independent; any laptop with a CUDA GPU completes one experiment in ~6-8 min.

## The file you modify (exactly one)

`ShAuRyA_Supplymind/autoresearch/candidate_train.py`

You may change anything between `# --- SAFE TO MODIFY BELOW ---` and `# --- SAFE TO MODIFY ABOVE ---`. You may NOT change:
- The function signature `def run_experiment(seed: int, total_steps: int) -> dict`.
- The import of `SupplyMindGymnasiumEnv` or `MaskablePPO`.
- The output JSON schema returned by `run_experiment` (keys: `grader_scores`, `wall_clock_s`, `total_steps`, `architecture_summary`).

## What's fair game

- RL algorithm (PPO / MaskablePPO / A2C / RecurrentPPO / DQN / QR-DQN).
- Policy network architecture (depth, width, activation, residual connections, layer norm, attention).
- Optimizer (Adam, AdamW, Muon, custom LR schedule).
- Hyperparameters (learning rate, batch size, clip range, entropy coeff, gamma, GAE lambda, n_steps).
- Observation preprocessing (normalization, feature selection, PCA, custom embeddings).
- Reward shaping (add auxiliary rewards provided they derive from env state — no hand-labeling).
- Action masking strategy (standard, joint, softmax over valid).

## What's NOT fair game

- No changes to the environment itself (`server/engine/`, `server/graders/`, `server/tasks/`).
- No changes to the evaluator (that's cheating — you'd be optimizing for the evaluator, not the task).
- No hard-coding task-specific rules. If your policy only works on `easy_typhoon_response`, it will fail the hard-task evaluation and be rejected.
- No calls to external APIs during training (offline constraint).
- No increases to the step or time budget.

## Hypothesis format (what you output each round)

```json
{
  "experiment_name": "e.g., recurrent_ppo_gru_128",
  "hypothesis": "RecurrentPPO with GRU memory should beat MLP PPO on hard_cascading_crisis because the task has long-horizon dependencies across disruption phases.",
  "expected_metric_delta": "+0.03 to +0.08 on CI95 lower, driven mostly by hard-task gain.",
  "justification": "Huang et al. 2020 shows RecurrentPPO matches MaskablePPO on memory-heavy MuJoCo tasks. Our R6 Euclidian result shows RecurrentPPO is 10% below MaskablePPO on this env — but that was with no GRU tuning. A 128-unit GRU with orthogonal init is the published default.",
  "modified_code": "<full unified diff of candidate_train.py>",
  "references": ["https://arxiv.org/abs/2006.14171", "R6_EUCLIDIAN.json line 47"]
}
```

## Karpathy's 3 rules (applied here)

1. **Repo is one-shot runnable**: `python -m ShAuRyA_Supplymind.autoresearch.orchestrator --budget 6h` kicks off the full overnight loop.
2. **Eval on holdout, never train set**: eval uses `seed != training_seed`. The orchestrator auto-checks and fails if reused.
3. **Plot literally everything**: each experiment writes `learning_curve.png`, `eval_boxplot.png`, `ci95_over_time.png` to `experiments/<timestamp>/`.

## Known starting point (baseline to beat)

From `v3_arcadia/results/R6_EUCLIDIAN.json`:
- MaskablePPO, 100k steps (we only have 50k, so expect slightly lower)
- Grader scores: easy 0.86, medium 0.72, hard 0.65 (approx)
- CI95 lower (bootstrap 1000): ~0.68 (aggregated)

**Your goal**: push CI95 lower above 0.75 within 50k steps per experiment.

## Lab notebook convention

Every accepted experiment appends an entry to `AUTORESEARCH_LAB_NOTEBOOK.md`:
- timestamp
- diff summary (files changed, LOC)
- hypothesis (copy from JSON)
- metric delta (before/after with CI95)
- plot links (relative paths)
- surprise flag (if result wildly different from expected, write "SURPRISE: X happened because Y")

Rejected experiments go in `AUTORESEARCH_REJECTED.md` with the same format + reason for rejection.

## Stopping condition

Orchestrator stops when:
1. `--budget` time elapsed
2. OR 50 consecutive rejections (exploration exhausted)
3. OR `stop_autoresearch.flag` file appears in autoresearch/ dir (graceful halt)

---

*This program.md is the contract. The agent reads this, the runner enforces it, the lab notebook records it. No ambiguity, no leakage, no moving goalposts.*
