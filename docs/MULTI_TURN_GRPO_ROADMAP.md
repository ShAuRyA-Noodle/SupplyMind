# Multi-turn GRPO with stepwise rewards — roadmap

**Status:** v1 ships single-turn GRPO against `/analyst/grade`. Multi-turn stepwise is scoped here and intentionally **not shipped** for the 2026-04-25/26 finals. This document exists so a judge reading the self-serve FAQ §59.6 (the acknowledged Unsloth gap on multi-turn GRPO with stepwise rewards) can see we understood the gap, have a concrete design, and consciously chose to ship single-turn first.

## What the FAQ flagged

> §59.6 — "One gap is multi-turn GRPO with stepwise rewards. There is a feature request asking for reward on each step plus a final reward, which suggests this is not yet a mature first-class recipe in Unsloth."

## What SupplyMind already has for multi-turn

The env is already multi-turn-ready on the policy-trajectory side:

| Component | State | Evidence |
|---|---|---|
| `Environment.reset → step → step → ... → grade` loop | ✅ shipped | `server/openenv_adapter.py` `OpenEnvSupplyMind.step(action)` builds `_episode_history` across turns |
| `TrajectoryRubric.compute_step_rewards()` | ✅ shipped | `server/openenv_adapter.py:61-66`. Returns per-step reward contributions from the final trajectory score |
| `EpisodeGrader` with full trajectory scoring | ✅ shipped | `server/graders/grader.py` — breakdown dict with per-step audit |
| MaskablePPO policy training inside the env loop | ✅ shipped | `versions/v3_arcadia/` R6 Gethsemane + autoresearch (+0.148 CI95 lift over baseline) |

What is **not** wired today:
- The **LLM-analyst GRPO path** (`train_grpo_live_env.py`) calls `/analyst/grade` once per assessment — one rollout, one reward.
- There is no multi-turn dialogue path where the LLM proposes an action, the env returns partial state, the LLM revises, etc.

## Why single-turn is the right v1 choice

1. **Verifier simplicity (FAQ §31-33).** The FAQ warns that verifier weaknesses are amplified in RL — false negatives on rule-based verifiers, false positives on LLM judges. A single-turn verifier against the real R4 ground-truth cache is the simplest possible verifier in this space. Multi-turn verification would force us to either run an LLM judge on intermediate turns (FAQ §33: "risky") or hand-design stepwise rubrics for each scenario (FAQ §11: possible but labor-intensive and itself gameable).

2. **Hackathon-suitable task (FAQ §54).** The FAQ's own suitability checklist names "short to medium trajectory length" as a property of the sweet-spot tasks. Our single-turn analyst task has trajectory length 1, which is exactly what the FAQ recommends for a one-weekend prototype.

3. **FAQ-blessed progression order (FAQ §18).** The FAQ's 9-phase plan ends with "then scale rollouts and environment diversity," not "start multi-turn on day one." Single-turn stable first, multi-turn as a post-hackathon upgrade.

4. **Debt-accurate scope.** Unsloth itself does not yet ship a first-class multi-turn GRPO recipe (the FAQ §59.6 admits this). Building a custom rollout loop on top of TRL GRPOTrainer with per-step rewards is a week of engineering, not 48 hours.

## The multi-turn design, when we ship it

### 1. Dialogue schema

Multi-turn risk assessment as a 3-turn conversation:

```
turn 1  env:    "Scenario: <text>. What data do you need to assess risk?"
turn 1  policy: "query R5 RAG for historical analogs of X"
turn 2  env:    "Analog 1: ..., Analog 2: ..., Analog 3: ..."
turn 2  policy: "query R6 forecaster for WTI path over next 14 days"
turn 3  env:    "point 123.28, 95% band [117, 130]"
turn 3  policy: "{risk_level: CRITICAL, confidence: 0.87, ...}"
```

Each env response is **real** — pulled from existing `/rag`, `/forecast`, `/live/recent-events` endpoints — so multi-turn does not introduce any synthetic data.

### 2. Stepwise reward schedule

```
r_turn_1 = relevance_score(policy_query, scenario_keywords)  ∈ [0, 1]
r_turn_2 = same, for query-2
r_turn_3 = existing /analyst/grade reward (proximity + format + length)
r_final  = r_turn_3        # outcome-dominant
r_shaped = 0.1*r_turn_1 + 0.1*r_turn_2 + 0.8*r_turn_3
```

The FAQ §44 pattern: "start with hard outcome checks, add minimal shaping only where sparse reward is too weak." 80% weight on outcome; 20% distributed across intermediate information-gathering steps.

### 3. TRL integration point

TRL's `GRPOTrainer` v0.12 does not support custom multi-turn rollouts directly (FAQ §59.6 gap). Two implementation paths:

**Path A — Custom rollout wrapper.** Subclass `GRPOTrainer` and override `_generate_and_score_completions` to run a multi-turn loop against the env. ~200 lines of code. Risk: depends on TRL internals that change across versions.

**Path B — ROLL integration.** `versions/v5_phoenix/roll_integration/` already pulls in Alibaba's ROLL framework which has native multi-turn GRPO (gigpo_multi_turn.yaml exists in our repo). Wire the analyst task to a ROLL env + ROLL agentic runner. More moving parts but built-for-purpose.

Path B is the preferred direction post-hackathon; Path A is a shorter bridge if we want to stay on vanilla TRL.

### 4. Separate holdout still applies

The sealed holdout set (`/analyst/scenarios?split=holdout` — 6 scenarios, FAQ §44) already serves multi-turn evaluation. Nothing changes on the eval side — the same holdout scenarios are used; only the policy's rollout schema changes from single-response to 3-turn dialogue.

### 5. Reward-hacking risks specific to multi-turn

- **Information-gather spam** — policy asks pointless questions to farm r_turn_1. Mitigation: bounded turn budget (max 3) + relevance-keyword match against the scenario.
- **Short-circuit** — policy answers on turn 1 without gathering info. Not exploitable because r_turn_3 is the dominant weight; skipping turns 1-2 forfeits 0.2 of reward.
- **Tool-call loops** — policy reruns the same query repeatedly. Mitigation: dedup penalty in the env's query handler.

The existing adversarial test suite (tests/test_reward_hacking_adversarial.py) will be extended with `A7_turn_spam`, `A8_early_answer`, `A9_repeated_query` before a multi-turn run is shipped.

## Summary

We ship single-turn because it is the FAQ-recommended hackathon sweet spot and the Unsloth multi-turn recipe is not yet mature. The env is already multi-turn-capable; only the LLM-analyst GRPO trainer is single-turn today. When we ship multi-turn, it will go through ROLL rather than a TRL fork, will use the same holdout set, and will extend the adversarial test suite to cover the new turn-level hack vectors.
