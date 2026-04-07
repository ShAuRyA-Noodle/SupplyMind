"""
SupplyMind Inference Script (OpenEnv-compliant entrypoint)

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (e.g., https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \
    MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct \
    HF_TOKEN=hf_... \
    python inference.py

    # Or with OpenAI-compatible endpoint:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o \
    HF_TOKEN=sk-... \
    python inference.py
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

# ---------------------------------------------------------------------------
# Environment variables (MANDATORY per competition rules)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: Docker image name for from_docker_image()
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
# System prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """\
You are a senior supply chain risk manager for a global manufacturing company.
You are playing a simulation where disruptions (typhoons, strikes, sanctions,
cascading crises) hit your supply chain and you must take actions each day to
minimize financial impact.

You have a LIMITED BUDGET -- do not waste money on unnecessary actions.
You receive one observation per day and must choose exactly ONE action.

## Available Actions (pick exactly one per step)

1. **do_nothing** -- Take no action. Use when the situation is stable or
   when no cost-effective mitigation exists.

2. **activate_backup_supplier** -- Switch production to a backup supplier.
   Requires: target_node_id (the disrupted supplier), backup_supplier_id
   (the backup to activate). Costs 15-30% premium. Use when a key supplier
   is down or at high risk.

3. **reroute_shipment** -- Use an alternative shipping route/port.
   Requires: target_node_id (the affected port/route), reroute_via (list of
   alternative port IDs). Use when a port or shipping lane is blocked.

4. **increase_safety_stock** -- Order extra inventory buffer.
   Requires: target_node_id (the warehouse/factory), additional_stock_days
   (1-90 days). Use proactively when disruptions are approaching.

5. **expedite_order** -- Upgrade transport mode (sea to air, etc).
   Requires: target_node_id, expedite_mode ("air", "rail", or "express_sea").
   Very expensive (5-10x normal cost). Use only for critical shortages.

6. **hedge_commodity** -- Hedge against commodity price spikes.
   Requires: commodity (e.g., "semiconductors", "rare_earths"),
   hedge_amount_usd (dollar amount). Use when commodity prices are rising.

7. **issue_supplier_alert** -- Request status update from a supplier.
   Requires: target_node_id. FREE action, provides information only.
   Use to gather intel before committing budget.

## Decision Guidelines
- Act PROACTIVELY: respond to warning signals before disruptions hit
- PRIORITIZE high-revenue nodes and critical supply paths
- Use issue_supplier_alert (free) to gather info before spending budget
- Activate backups for nodes with high risk and available backups
- Increase safety stock when disruptions are approaching but not yet active
- Reroute shipments when ports/routes are blocked
- Expedite orders only as a last resort (very expensive)
- Hedge commodities when you see price spike signals
- do_nothing when the situation is stable and no action is needed

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation):
{
    "action_type": "<one of the 7 types>",
    "target_node_id": "<node ID or null>",
    "backup_supplier_id": "<backup ID or null>",
    "reroute_via": ["<port_id>"] or null,
    "additional_stock_days": <int or null>,
    "expedite_mode": "<air|rail|express_sea or null>",
    "commodity": "<commodity name or null>",
    "hedge_amount_usd": <float or null>
}
"""

TASK_HINTS = {
    "easy_typhoon_response": """
## Task-Specific Guidance (Easy: Typhoon Response)
- Single disruption: typhoon approaching Taiwan (affects TSMC semiconductor supply)
- You have 72 hours of warning before impact -- ACT DURING WARNING PHASE
- Priority: activate backup supplier for TSMC, then increase safety stock at warehouses
- Budget is ample ($5M) -- spend 15-25% on targeted mitigation
- Timing matters most: early action scores much higher than reactive scrambling
""",
    "medium_multi_front": """
## Task-Specific Guidance (Medium: Multi-Front Crisis)
- THREE simultaneous disruptions: US port strike, Thailand flooding, China sanctions
- Budget ($8M) only covers ~2 of 3 -- you MUST TRIAGE
- Priority order: (1) port strike (highest immediate revenue impact), (2) Thailand floods, (3) sanctions
- Use alerts early to assess which nodes need action most urgently
- Hedge rare_earths/semiconductors for the sanctions disruption
""",
    "hard_cascading_crisis": """
## Task-Specific Guidance (Hard: Cascading Crisis)
- Geopolitical cascade: Taiwan Strait → shipping disruption → semiconductor cutoff → commodity spikes → cyber attack
- Budget ($10M) is VERY tight relative to $2B+ potential losses
- Use alerts strategically in early steps to map the cascade path
- Prioritize semiconductor supply chain (highest revenue) over commodities
- Hedge early before commodity prices spike
- Accept some losses -- focus on preventing catastrophic cascading failures
""",
}


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
# Observation formatting
# ---------------------------------------------------------------------------


def format_observation(obs: dict) -> str:
    """Format a raw observation dict into a concise user message for the LLM."""
    parts = []

    current_day = obs.get("current_day", 0)
    days_remaining = obs.get("days_remaining", 0)
    total_days = current_day + days_remaining
    parts.append(f"=== Day {current_day}/{total_days} | {days_remaining} days remaining ===")
    parts.append("")

    # Compact summary (token-efficient overview for LLM decision-making)
    compact = obs.get("compact_summary", "")
    if compact:
        parts.append("--- Quick Brief ---")
        parts.append(compact)
        parts.append("")

    # Situation summary
    summary = obs.get("situation_summary", "")
    if summary:
        parts.append(summary)
        parts.append("")

    # Last action feedback
    last_result = obs.get("last_action_result")
    if last_result:
        status = "SUCCESS" if last_result.get("success") else "FAILED"
        parts.append(f"Last action: {status} -- {last_result.get('message', '')}")
        cost = last_result.get("cost", 0)
        if cost > 0:
            parts.append(f"  Cost: ${cost:,.0f}")
        effect = last_result.get("effect_description", "")
        if effect:
            parts.append(f"  Effect: {effect}")
        parts.append("")

    # Financials
    fin = obs.get("financials", {})
    parts.append("--- Financials ---")
    parts.append(f"Budget: ${fin.get('budget_remaining', 0):,.0f} / ${fin.get('budget_total', 0):,.0f}")
    parts.append(f"Revenue at risk: ${fin.get('total_revenue_at_risk', 0):,.0f}")
    parts.append(f"Revenue lost so far: ${fin.get('cumulative_revenue_lost', 0):,.0f}")
    parts.append(f"Costs incurred: ${fin.get('cumulative_cost_incurred', 0):,.0f}")
    parts.append(f"Health score: {fin.get('supply_chain_health_score', 100):.1f}/100")
    commodity_changes = fin.get("commodity_price_changes", {})
    if commodity_changes:
        changes = ", ".join(f"{k}: {v:.2f}x" for k, v in commodity_changes.items())
        parts.append(f"Commodity prices: {changes}")
    parts.append("")

    # Active disruption signals
    active_signals = obs.get("active_signals", [])
    new_signals = obs.get("new_signals", [])
    new_ids = {s.get("signal_id") for s in new_signals}

    if active_signals:
        parts.append("--- Active Disruptions ---")
        for sig in active_signals:
            is_new = sig.get("signal_id") in new_ids
            new_tag = " [NEW]" if is_new else ""
            parts.append(
                f"  {sig.get('signal_id', '?')}{new_tag}: {sig.get('disruption_type', '?')} "
                f"(severity={sig.get('severity', 0):.1f}, phase={sig.get('lifecycle_phase', '?')}) "
                f"in {sig.get('affected_region', '?')}"
            )
            parts.append(
                f"    Impact in {sig.get('time_to_impact_hours', 0):.0f}h, "
                f"duration ~{sig.get('estimated_duration_days', 0):.0f}d"
            )
            affected = sig.get("affected_node_ids", [])
            if affected:
                parts.append(f"    Affected nodes: {', '.join(affected)}")
            parts.append(f"    {sig.get('description', '')}")
        parts.append("")

    # At-risk nodes
    node_statuses = obs.get("node_statuses", [])
    at_risk = [
        n for n in node_statuses
        if n.get("current_risk_score", 0) > 0.2
        or not n.get("is_operational", True)
        or n.get("active_disruption_ids")
    ]
    if at_risk:
        parts.append("--- At-Risk Nodes ---")
        for n in at_risk:
            status = "OFFLINE" if not n.get("is_operational", True) else f"risk={n.get('current_risk_score', 0):.2f}"
            backup_info = ""
            if n.get("has_backup"):
                backup_info = f" [backups: {', '.join(n.get('backup_supplier_ids', []))}]"
            parts.append(
                f"  {n.get('node_id', '?')} ({n.get('name', '?')}, {n.get('node_type', '?')}, "
                f"{n.get('country', '?')}): {status}, inventory={n.get('inventory_days_cover', 0):.0f}d, "
                f"revenue=${n.get('revenue_contribution', 0):,.0f}{backup_info}"
            )
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM action parsing
# ---------------------------------------------------------------------------


def _clean_json_quirks(text: str) -> str:
    """Remove common LLM JSON quirks: JS comments, trailing commas."""
    import re
    # Remove single-line comments (// ...)
    text = re.sub(r'//[^\n]*', '', text)
    # Remove multi-line comments (/* ... */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output, handling code fences and prose."""
    text = text.strip()
    if not text:
        return "{}"

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        inside = False
        json_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines).strip()

    # Find JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    # Clean LLM quirks (comments, trailing commas)
    text = _clean_json_quirks(text)

    return text


def parse_action(response_text: str) -> dict:
    """Parse LLM response into an action dict. Falls back to do_nothing."""
    try:
        text = _extract_json(response_text)
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"action_type": "do_nothing"}

        # Remove null values
        cleaned = {k: v for k, v in data.items() if v is not None}

        # Validate action_type
        valid_actions = {
            "do_nothing", "activate_backup_supplier", "reroute_shipment",
            "increase_safety_stock", "expedite_order", "hedge_commodity",
            "issue_supplier_alert",
        }
        action_type = cleaned.get("action_type", "do_nothing")
        if action_type not in valid_actions:
            # Try fuzzy match
            lower_map = {a.lower().replace("_", ""): a for a in valid_actions}
            normalized = action_type.lower().replace("_", "").replace("-", "").replace(" ", "")
            if normalized in lower_map:
                cleaned["action_type"] = lower_map[normalized]
            else:
                return {"action_type": "do_nothing"}

        # Auto-fix reroute_via as string
        if "reroute_via" in cleaned and isinstance(cleaned["reroute_via"], str):
            cleaned["reroute_via"] = [cleaned["reroute_via"]]

        # Auto-fix additional_stock_days as float
        if "additional_stock_days" in cleaned:
            try:
                cleaned["additional_stock_days"] = int(cleaned["additional_stock_days"])
            except (ValueError, TypeError):
                cleaned.pop("additional_stock_days")

        return cleaned

    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse action: %s. Falling back to do_nothing.", e)
        return {"action_type": "do_nothing"}


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
) -> dict:
    """Ask the LLM to choose an action given the current observation."""
    user_message = format_observation(obs)
    conversation_history.append({"role": "user", "content": user_message})

    # Build system prompt with task hints
    hint = TASK_HINTS.get(task_id, "")
    system_prompt = BASE_SYSTEM_PROMPT + hint

    # Keep conversation bounded (system + last 10 turns)
    messages = [{"role": "system", "content": system_prompt}]
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
            action = get_action(llm_client, obs, conversation_history, task_id)
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
) -> dict[str, Any]:
    """Run the baseline LLM agent on all 3 tasks."""
    results: dict[str, Any] = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "api_base_url": API_BASE_URL,
        "tasks": {},
    }

    total_score = 0.0
    for task_id in TASK_IDS:
        task_result = run_task(env_client, llm_client, task_id)
        results["tasks"][task_id] = task_result
        total_score += task_result.get("score", 0.0)

    results["average_score"] = round(total_score / len(TASK_IDS), 4)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate mandatory environment variables
    if not API_KEY:
        print("ERROR: Set HF_TOKEN (or API_KEY) environment variable.")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: Set MODEL_NAME environment variable.")
        print("  export MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct")
        sys.exit(1)

    print("=" * 60)
    print("SupplyMind Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Env URL:  {ENV_URL}")
    print(f"Temp:     {TEMPERATURE}")
    print("=" * 60)

    # Create clients
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SupplyMindHTTPClient(ENV_URL)

    try:
        results = run_all_baselines(env_client, llm_client)

        print("\n" + "=" * 60)
        print("RESULTS")
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
