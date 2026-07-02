"""
SupplyMind Baseline Inference Script

Uses an LLM via the OpenAI client to run a baseline agent on all 3 SupplyMind
tasks. The agent receives the observation's situation_summary and structured
data, then chooses one of 7 action types per step.

Shared prompt / parsing / formatting logic lives in llm_agent_common.py (also
used by inference.py) — this file only owns the in-process (no-HTTP) run loop.

Environment variables:
    OPENROUTER_API_KEY  Preferred LLM key (OpenRouter). Falls back to
                        API_KEY / OPENAI_API_KEY / HF_TOKEN.
    API_BASE_URL        LLM endpoint (default: https://openrouter.ai/api/v1)
    MODEL_NAME          Model slug (default: qwen/qwen-2.5-72b-instruct)
    OPENROUTER_SITE_URL / OPENROUTER_APP_NAME
                        Optional OpenRouter attribution headers.

Usage:
    # Direct invocation (calls environment directly, no HTTP):
    from baseline import run_all_baselines
    from server.supply_environment import SupplyMindEnvironment
    env = SupplyMindEnvironment()
    results = run_all_baselines(env)

    # Standalone mode (honest, no task hints):
    OPENROUTER_API_KEY=sk-or-... python baseline.py

    # Dual-report mode (append ground-truth task hints; records labelled hinted:true):
    OPENROUTER_API_KEY=sk-or-... python baseline.py --hints
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from llm_agent_common import build_system_prompt, format_observation, parse_action
from models import SupplyMindAction, SupplyMindObservation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — read from environment
# ---------------------------------------------------------------------------

TASK_IDS = [
    "easy_typhoon_response",
    "medium_multi_front",
    "hard_cascading_crisis",
]

# OpenRouter is the primary AI backbone (see CLAUDE.md §0.4). A coherent
# base-url + model pair is REQUIRED — the previous HF-router + gpt-4o default
# could never succeed and silently degraded every episode to do_nothing.
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
TEMPERATURE = 0.1

# Seeds actually run per task (each is a full episode) to showcase episode
# variation. run_all_baselines runs ALL of these, not just the first.
BASELINE_SEEDS = [42, 99, 7]


def _resolve_api_key() -> str | None:
    """Resolve the LLM API key, OpenRouter first."""
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
    )


def _build_client(api_key: str) -> OpenAI:
    """Build an OpenAI-compatible client, wiring optional OpenRouter headers."""
    default_headers: dict[str, str] = {}
    site = os.getenv("OPENROUTER_SITE_URL")
    app = os.getenv("OPENROUTER_APP_NAME")
    if site:
        default_headers["HTTP-Referer"] = site
    if app:
        default_headers["X-Title"] = app
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
        default_headers=default_headers or None,
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


def get_action(
    client: OpenAI,
    obs: SupplyMindObservation,
    conversation_history: list[dict[str, str]],
    task_id: str = "easy_typhoon_response",
    use_hints: bool = False,
) -> SupplyMindAction:
    """
    Ask the LLM to choose an action given the current observation.

    Maintains a rolling conversation history for context, bounded to avoid
    token overflow. Retries transient API errors (429/5xx/timeouts) with
    exponential backoff. A malformed action falls back to do_nothing WITHOUT
    consuming a retry (only transient transport errors retry).
    """
    user_message = format_observation(obs.model_dump(mode="json"))
    conversation_history.append({"role": "user", "content": user_message})

    # Keep conversation bounded (system + last 10 turns) to reduce token usage.
    messages = [{"role": "system", "content": build_system_prompt(task_id, use_hints)}]
    messages.extend(conversation_history[-10:])

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=4096,  # Thinking models need room for reasoning tokens
            )
            msg = response.choices[0].message
            assistant_text = msg.content or ""
            # Some models (Qwen3, etc.) put output in reasoning_content
            if not assistant_text:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    assistant_text = rc
            conversation_history.append({"role": "assistant", "content": assistant_text})

            action_dict = parse_action(assistant_text)
            try:
                return SupplyMindAction(**action_dict)
            except Exception as e:  # noqa: BLE001 - bad action, not transient
                logger.warning("Action dict failed validation (%s): %s -> do_nothing", e, action_dict)
                return SupplyMindAction(action_type="do_nothing")

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry only on transient errors: rate limits, server errors, timeouts.
            is_transient = any(
                kw in error_str
                for kw in ("429", "rate", "limit", "500", "502", "503", "timeout", "connection")
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
                continue
            break

    logger.error("OpenAI API call failed after %d attempts: %s. Falling back to do_nothing.", MAX_RETRIES, last_error)
    return SupplyMindAction(action_type="do_nothing")


# ---------------------------------------------------------------------------
# Run one task (single seed)
# ---------------------------------------------------------------------------


def run_task(
    env: Any,
    task_id: str,
    client: OpenAI,
    seed: int | None = None,
    use_hints: bool = False,
) -> dict[str, Any]:
    """
    Run a single task episode to completion using the LLM agent.

    Returns the grader dict augmented with ``seed`` and ``elapsed_seconds``.
    """
    logger.info("Starting task: %s (seed=%s)", task_id, seed)
    start = time.time()

    obs = env.reset(task_id=task_id, seed=seed)
    conversation_history: list[dict[str, str]] = []
    step_count = 0

    while not obs.done:
        action = get_action(client, obs, conversation_history, task_id=task_id, use_hints=use_hints)
        obs = env.step(action)
        step_count += 1

        if step_count % 10 == 0:
            logger.info(
                "  [%s seed=%s] Step %d -- reward=%.3f, health=%.1f, budget=$%.0f",
                task_id,
                seed,
                step_count,
                obs.reward,
                obs.financials.supply_chain_health_score,
                obs.financials.budget_remaining,
            )

    # Grade the episode
    result = env.grade()
    elapsed = time.time() - start

    logger.info(
        "Completed %s (seed=%s): score=%.4f, steps=%d, time=%.1fs",
        task_id,
        seed,
        result["score"],
        step_count,
        elapsed,
    )

    result["seed"] = seed
    result["elapsed_seconds"] = round(elapsed, 1)
    return result


# ---------------------------------------------------------------------------
# Run all baselines (called by app.py)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_all_baselines(env: Any, use_hints: bool = False) -> dict[str, Any]:
    """
    Run the baseline LLM agent on all 3 tasks across all BASELINE_SEEDS.

    This is the entry point called by app.py's /baseline endpoint. Each task's
    ``score`` is the mean across seeds; per-seed grader dicts are kept under
    ``runs``.

    Args:
        env: SupplyMindEnvironment instance.
        use_hints: When True, append ground-truth task hints and label the
            output ``"hinted": true``. Default False (honest baseline).

    Raises:
        RuntimeError: If no LLM API key is set (fails loud, never runs silent
            do-nothing episodes).
    """
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "No LLM API key found. Set OPENROUTER_API_KEY (preferred) — or "
            "API_KEY / OPENAI_API_KEY / HF_TOKEN — before running the baseline. "
            "Refusing to run silent do-nothing episodes without a key."
        )

    client = _build_client(api_key)

    results: dict[str, Any] = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "api_base_url": API_BASE_URL,
        "hinted": use_hints,
        "seeds": BASELINE_SEEDS,
        "tasks": {},
    }

    total_score = 0.0
    for task_id in TASK_IDS:
        per_seed: list[dict[str, Any]] = []
        for seed in BASELINE_SEEDS:
            try:
                per_seed.append(run_task(env, task_id, client, seed=seed, use_hints=use_hints))
            except Exception as e:
                logger.error("Task %s (seed=%s) failed: %s", task_id, seed, e)
                per_seed.append({
                    "task_id": task_id,
                    "seed": seed,
                    "score": 0.0,
                    "steps_taken": 0,
                    "total_steps": 0,
                    "cumulative_reward": 0.0,
                    "is_done": False,
                    "breakdown": {"error": {"score": 0.0, "weight": 1.0}},
                    "elapsed_seconds": 0.0,
                    "error": str(e),
                })

        scores = [r.get("score", 0.0) for r in per_seed]
        mean_score = round(_mean(scores), 4)
        results["tasks"][task_id] = {
            "task_id": task_id,
            "score": mean_score,  # mean across seeds
            "score_per_seed": {str(r["seed"]): round(r.get("score", 0.0), 4) for r in per_seed},
            "seeds": BASELINE_SEEDS,
            "runs": per_seed,
        }
        total_score += mean_score

    results["average_score"] = round(total_score / len(TASK_IDS), 4)

    logger.info(
        "Baseline complete (hinted=%s). Average score: %.4f", use_hints, results["average_score"]
    )
    return results


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SupplyMind baseline LLM agent (in-process).")
    parser.add_argument(
        "--hints", dest="use_hints", action="store_true",
        help="append ground-truth task hints (dual-report; records labelled hinted:true)",
    )
    parser.add_argument(
        "--no-hints", dest="use_hints", action="store_false",
        help="honest baseline, no ground-truth hints (default)",
    )
    parser.set_defaults(use_hints=False)
    args = parser.parse_args()

    if not _resolve_api_key():
        print("ERROR: Set OPENROUTER_API_KEY (or API_KEY / OPENAI_API_KEY / HF_TOKEN) first.")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    # Direct mode: import the environment and run locally (no HTTP server needed)
    from server.supply_environment import SupplyMindEnvironment

    print("=" * 60)
    print("SupplyMind Baseline Inference")
    print(f"Model:    {MODEL}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Temp:     {TEMPERATURE}")
    print(f"Seeds:    {BASELINE_SEEDS}")
    print(f"Hinted:   {args.use_hints}")
    print("=" * 60)

    env = SupplyMindEnvironment()
    results = run_all_baselines(env, use_hints=args.use_hints)

    print("\n" + "=" * 60)
    print(f"RESULTS (hinted={results['hinted']})")
    print("=" * 60)

    for task_id, task_result in results["tasks"].items():
        print(f"\n  {task_id}:")
        print(f"    Mean score: {task_result['score']:.4f}  (seeds {task_result['seeds']})")
        for seed, score in task_result["score_per_seed"].items():
            print(f"      seed {seed}: {score:.4f}")

    print(f"\n  Average Score: {results['average_score']:.4f}")
    print("=" * 60)
