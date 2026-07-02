"""
SupplyMind Inference Script (OpenEnv-compliant entrypoint)

HTTP client agent: drives a running SupplyMind server (/reset, /step, /grader)
and emits the competition-spec [START]/[STEP]/[END] stdout lines.

Shared prompt / parsing / formatting logic lives in llm_agent_common.py (also
used by baseline.py) — this file only owns the HTTP run loop and stdout format.

Environment variables:
    OPENROUTER_API_KEY  Preferred LLM key (OpenRouter). Falls back to
                        API_KEY / OPENAI_API_KEY / HF_TOKEN.
    API_BASE_URL        LLM endpoint (default: https://openrouter.ai/api/v1)
    MODEL_NAME          Model slug (default: qwen/qwen-2.5-72b-instruct)
    ENV_URL             SupplyMind server URL (default: http://localhost:8000)
    OPENROUTER_SITE_URL / OPENROUTER_APP_NAME
                        Optional OpenRouter attribution headers.

Usage:
    # Honest baseline (no task hints):
    OPENROUTER_API_KEY=sk-or-... MODEL_NAME=qwen/qwen-2.5-72b-instruct \
    python inference.py

    # Dual-report mode (append ground-truth task hints; records labelled hinted:true):
    OPENROUTER_API_KEY=sk-or-... python inference.py --hints
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

from llm_agent_common import build_system_prompt, format_observation, parse_action

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

# OpenRouter is the primary AI backbone (see CLAUDE.md §0.4). A coherent
# base-url + model pair is REQUIRED.
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
TEMPERATURE = 0.1
MAX_TOKENS = 4096  # Thinking models (Gemini 3, Qwen3) use tokens for reasoning

# SupplyMind server URL (the deployed HF Space or local server)
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "supplymind"

TASK_IDS = [
    "easy_typhoon_response",
    "medium_multi_front",
    "hard_cascading_crisis",
]

logger = logging.getLogger(__name__)


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
# Mandatory STDOUT format: [START], [STEP], [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line per competition spec."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Emit [STEP] line per competition spec."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit [END] line per competition spec."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP client for SupplyMind environment
# ---------------------------------------------------------------------------


class SupplyMindHTTPClient:
    """Simple HTTP client for the SupplyMind environment server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def reset(self, task_id: str) -> dict:
        resp = self.client.post("/reset", params={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = self.client.post("/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> dict:
        resp = self.client.post("/grader")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 3.0  # Longer backoff for free-tier rate limits


def get_action(
    client: OpenAI,
    obs: dict,
    conversation_history: list[dict[str, str]],
    task_id: str,
    use_hints: bool = False,
) -> dict:
    """Ask the LLM to choose an action given the current observation dict."""
    user_message = format_observation(obs)
    conversation_history.append({"role": "user", "content": user_message})

    # Keep conversation bounded (system + last 10 turns)
    messages = [{"role": "system", "content": build_system_prompt(task_id, use_hints)}]
    messages.extend(conversation_history[-10:])

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            msg = response.choices[0].message
            assistant_text = msg.content or ""
            # Some models (Qwen3, etc.) put output in reasoning_content
            if not assistant_text:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    assistant_text = rc
            conversation_history.append({"role": "assistant", "content": assistant_text})
            return parse_action(assistant_text)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_transient = any(
                kw in error_str
                for kw in ("429", "rate", "limit", "500", "502", "503", "timeout", "connection")
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                # Extract server-suggested retry delay if present
                import re
                retry_match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str)
                if retry_match:
                    wait = min(float(retry_match.group(1)) + 1, 90)
                else:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning("API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                               attempt + 1, MAX_RETRIES, e, wait)
                time.sleep(wait)
                continue
            break

    logger.error("LLM API call failed after %d attempts: %s", MAX_RETRIES, last_error)
    return {"action_type": "do_nothing"}


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


def run_task(
    env_client: SupplyMindHTTPClient,
    llm_client: OpenAI,
    task_id: str,
    use_hints: bool = False,
) -> dict[str, Any]:
    """Run a single task to completion using the LLM agent."""
    logger.info("Starting task: %s", task_id)
    start = time.time()

    rewards: list[float] = []
    step_count = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_id)
        conversation_history: list[dict[str, str]] = []

        while not obs.get("done", False):
            action = get_action(llm_client, obs, conversation_history, task_id, use_hints=use_hints)
            obs = env_client.step(action)
            step_count += 1

            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            error = None
            last_result = obs.get("last_action_result")
            if last_result and not last_result.get("success", True):
                error = last_result.get("message")

            rewards.append(reward)

            # Format action for log (compact representation)
            action_str = action.get("action_type", "do_nothing")
            target = action.get("target_node_id")
            if target:
                action_str += f"({target})"

            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)

            if step_count % 10 == 0:
                fin = obs.get("financials", {})
                logger.info(
                    "  [%s] Step %d -- reward=%.3f, health=%.1f, budget=$%.0f",
                    task_id, step_count,
                    obs.get("reward", 0),
                    fin.get("supply_chain_health_score", 0),
                    fin.get("budget_remaining", 0),
                )

        # Grade the episode
        result = env_client.grade()
        elapsed = time.time() - start
        score = result.get("score", 0.0)
        success = score > 0.0

        logger.info("Completed %s: score=%.4f, steps=%d, time=%.1fs",
                    task_id, score, step_count, elapsed)

        result["elapsed_seconds"] = round(elapsed, 1)

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e)
        result = {
            "task_id": task_id,
            "score": 0.0,
            "steps_taken": step_count,
            "cumulative_reward": sum(rewards),
            "elapsed_seconds": round(time.time() - start, 1),
            "error": str(e),
        }

    finally:
        log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return result


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------


def run_all_baselines(
    env_client: SupplyMindHTTPClient,
    llm_client: OpenAI,
    use_hints: bool = False,
) -> dict[str, Any]:
    """Run the baseline LLM agent on all 3 tasks."""
    results: dict[str, Any] = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "api_base_url": API_BASE_URL,
        "hinted": use_hints,
        "tasks": {},
    }

    total_score = 0.0
    for task_id in TASK_IDS:
        task_result = run_task(env_client, llm_client, task_id, use_hints=use_hints)
        results["tasks"][task_id] = task_result
        total_score += task_result.get("score", 0.0)

    results["average_score"] = round(total_score / len(TASK_IDS), 4)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SupplyMind inference agent (HTTP client).")
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

    # Validate mandatory environment: fail loud on a missing key rather than
    # running silent do-nothing episodes.
    if not API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY (or API_KEY / OPENAI_API_KEY / HF_TOKEN).")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: Set MODEL_NAME environment variable.")
        print("  export MODEL_NAME=qwen/qwen-2.5-72b-instruct")
        sys.exit(1)

    print("=" * 60)
    print("SupplyMind Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Env URL:  {ENV_URL}")
    print(f"Temp:     {TEMPERATURE}")
    print(f"Hinted:   {args.use_hints}")
    print("=" * 60)

    # Create clients
    llm_client = _build_client(API_KEY)
    env_client = SupplyMindHTTPClient(ENV_URL)

    try:
        results = run_all_baselines(env_client, llm_client, use_hints=args.use_hints)

        print("\n" + "=" * 60)
        print(f"RESULTS (hinted={results['hinted']})")
        print("=" * 60)

        for task_id, task_result in results["tasks"].items():
            print(f"\n  {task_id}:")
            print(f"    Score:      {task_result.get('score', 0):.4f}")
            print(f"    Steps:      {task_result.get('steps_taken', 0)}")
            print(f"    Reward:     {task_result.get('cumulative_reward', 0):.4f}")
            print(f"    Time:       {task_result.get('elapsed_seconds', 0)}s")
            breakdown = task_result.get("breakdown")
            if breakdown:
                print(f"    Breakdown:  {json.dumps(breakdown, indent=6)}")

        print(f"\n  Average Score: {results['average_score']:.4f}")
        print("=" * 60)

    finally:
        env_client.close()


if __name__ == "__main__":
    main()
