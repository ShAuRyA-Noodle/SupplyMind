"""
SupplyMind Baseline Inference Script

Uses an LLM via the OpenAI client to run a baseline agent on all 3 SupplyMind
tasks. The agent receives the observation's situation_summary and structured
data, then chooses one of 7 action types per step.

Required environment variables (per competition rules):
    API_BASE_URL   The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (default: gpt-4o)
    HF_TOKEN       Your Hugging Face / API key (falls back to OPENAI_API_KEY)

Usage:
    # Direct invocation (calls environment directly, no HTTP):
    from baseline import run_all_baselines
    from server.supply_environment import SupplyMindEnvironment
    env = SupplyMindEnvironment()
    results = run_all_baselines(env)

    # Standalone mode:
    HF_TOKEN=hf_... MODEL_NAME=gpt-4o python baseline.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from models import SupplyMindAction, SupplyMindObservation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — read from environment per competition rules
# ---------------------------------------------------------------------------

TASK_IDS = [
    "easy_typhoon_response",
    "medium_multi_front",
    "hard_cascading_crisis",
]

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL = os.getenv("MODEL_NAME", "gpt-4o")
TEMPERATURE = 0.1

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

# Task-specific strategy hints appended to the system prompt
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
- Priority order: (1) port strike (highest immediate revenue impact), (2) Thailand floods (Tier 2 but cascading), (3) sanctions (slower onset, hedge-able)
- Use alerts early to assess which nodes need action most urgently
- Hedge rare_earths/semiconductors for the sanctions disruption (cheaper than direct mitigation)
""",
    "hard_cascading_crisis": """
## Task-Specific Guidance (Hard: Cascading Crisis)
- Geopolitical cascade: Taiwan Strait → shipping disruption → semiconductor cutoff → commodity spikes → cyber attack
- Budget ($10M) is VERY tight relative to $2B+ potential losses
- Use alerts strategically in early steps to map the cascade path
- Prioritize semiconductor supply chain (highest revenue) over commodities
- Hedge early before commodity prices spike (hedging gets more expensive during crisis)
- Accept some losses -- focus on preventing catastrophic cascading failures
- Balance information gathering (alerts) with decisive action (roughly 20-30% alerts)
""",
}


def _get_system_prompt(task_id: str) -> str:
    """Build task-specific system prompt with strategy hints."""
    hint = TASK_HINTS.get(task_id, "")
    return BASE_SYSTEM_PROMPT + hint


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------


def format_observation(obs: SupplyMindObservation) -> str:
    """Format an observation into a concise user message for the LLM."""
    parts = []

    total_days = obs.current_day + obs.days_remaining
    parts.append(f"=== Day {obs.current_day}/{total_days} | {obs.days_remaining} days remaining ===")
    parts.append("")

    # Compact summary (token-efficient overview for LLM decision-making)
    if obs.compact_summary:
        parts.append("--- Quick Brief ---")
        parts.append(obs.compact_summary)
        parts.append("")

    # Situation summary (natural language)
    if obs.situation_summary:
        parts.append(obs.situation_summary)
        parts.append("")

    # Last action feedback
    if obs.last_action_result:
        r = obs.last_action_result
        status = "SUCCESS" if r.success else "FAILED"
        parts.append(f"Last action: {status} -- {r.message}")
        if r.cost > 0:
            parts.append(f"  Cost: ${r.cost:,.0f}")
        if r.effect_description:
            parts.append(f"  Effect: {r.effect_description}")
        parts.append("")

    # Financials
    f = obs.financials
    parts.append("--- Financials ---")
    parts.append(f"Budget: ${f.budget_remaining:,.0f} / ${f.budget_total:,.0f}")
    parts.append(f"Revenue at risk: ${f.total_revenue_at_risk:,.0f}")
    parts.append(f"Revenue lost so far: ${f.cumulative_revenue_lost:,.0f}")
    parts.append(f"Costs incurred: ${f.cumulative_cost_incurred:,.0f}")
    parts.append(f"Health score: {f.supply_chain_health_score:.1f}/100")
    if f.commodity_price_changes:
        changes = ", ".join(
            f"{k}: {v:.2f}x" for k, v in f.commodity_price_changes.items()
        )
        parts.append(f"Commodity prices: {changes}")
    parts.append("")

    # Active disruption signals
    if obs.active_signals:
        parts.append("--- Active Disruptions ---")
        for sig in obs.active_signals:
            is_new = sig in obs.new_signals
            new_tag = " [NEW]" if is_new else ""
            parts.append(
                f"  {sig.signal_id}{new_tag}: {sig.disruption_type} "
                f"(severity={sig.severity:.1f}, phase={sig.lifecycle_phase}) "
                f"in {sig.affected_region}"
            )
            parts.append(f"    Impact in {sig.time_to_impact_hours:.0f}h, "
                         f"duration ~{sig.estimated_duration_days:.0f}d")
            if sig.affected_node_ids:
                parts.append(f"    Affected nodes: {', '.join(sig.affected_node_ids)}")
            parts.append(f"    {sig.description}")
        parts.append("")

    # Node statuses -- only show at-risk or disrupted nodes
    at_risk_nodes = [
        n for n in obs.node_statuses
        if n.current_risk_score > 0.2 or not n.is_operational or n.active_disruption_ids
    ]
    if at_risk_nodes:
        parts.append("--- At-Risk Nodes ---")
        for n in at_risk_nodes:
            status = "OFFLINE" if not n.is_operational else f"risk={n.current_risk_score:.2f}"
            backup_info = ""
            if n.has_backup:
                backup_info = f" [backups: {', '.join(n.backup_supplier_ids)}]"
            parts.append(
                f"  {n.node_id} ({n.name}, {n.node_type}, {n.country}): "
                f"{status}, inventory={n.inventory_days_cover:.0f}d, "
                f"revenue=${n.revenue_contribution:,.0f}{backup_info}"
            )
            if n.active_disruption_ids:
                parts.append(f"    Active disruptions: {', '.join(n.active_disruption_ids)}")
        parts.append("")

    # Inventory warnings for warehouses running low
    low_inv = [
        n for n in obs.node_statuses
        if n.node_type == "warehouse" and 0 < n.inventory_days_cover <= 7
    ]
    if low_inv:
        parts.append("--- LOW INVENTORY WARNING ---")
        for n in low_inv:
            parts.append(f"  {n.node_id} ({n.name}): {n.inventory_days_cover:.0f} days remaining")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM action selection
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
    """
    Extract JSON from LLM output, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose around JSON
    - Arrays instead of objects (take first element)
    - JS-style comments and trailing commas
    - Empty strings
    """
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

    # Try to find JSON object in the text (LLM may add prose around it)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    bracket_start = text.find("[")

    # If we found an array before an object, extract first element
    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        try:
            cleaned = _clean_json_quirks(text[bracket_start:text.rfind("]") + 1])
            arr = json.loads(cleaned)
            if isinstance(arr, list) and arr:
                return json.dumps(arr[0]) if isinstance(arr[0], dict) else "{}"
        except json.JSONDecodeError:
            pass

    if brace_start != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    # Clean LLM quirks (comments, trailing commas)
    text = _clean_json_quirks(text)

    return text


def parse_action(response_text: str) -> SupplyMindAction:
    """
    Parse the LLM response into a SupplyMindAction.

    Handles all common LLM failure modes:
    - Markdown code fences
    - Arrays instead of objects
    - Prose around JSON
    - Empty / whitespace responses
    - Invalid JSON
    - Missing required fields
    - Typos in action_type (fuzzy match)
    Falls back to do_nothing on any unrecoverable error.
    """
    try:
        text = _extract_json(response_text)
        data = json.loads(text)

        if not isinstance(data, dict):
            logger.warning("LLM returned non-dict JSON: %s", type(data).__name__)
            return SupplyMindAction(action_type="do_nothing")

        # Remove null values so Pydantic defaults work
        cleaned = {k: v for k, v in data.items() if v is not None}

        # Fuzzy-match action_type for common typos
        action_type = cleaned.get("action_type", "do_nothing")
        valid_actions = {
            "do_nothing", "activate_backup_supplier", "reroute_shipment",
            "increase_safety_stock", "expedite_order", "hedge_commodity",
            "issue_supplier_alert",
        }
        if action_type not in valid_actions:
            # Try case-insensitive match
            lower_map = {a.lower().replace("_", ""): a for a in valid_actions}
            normalized = action_type.lower().replace("_", "").replace("-", "").replace(" ", "")
            if normalized in lower_map:
                cleaned["action_type"] = lower_map[normalized]
                logger.debug("Fuzzy-matched action_type '%s' -> '%s'", action_type, cleaned["action_type"])
            else:
                logger.warning("Unknown action_type '%s', defaulting to do_nothing.", action_type)
                return SupplyMindAction(action_type="do_nothing")

        # Auto-fix: actions needing target_node_id but missing one
        action_type = cleaned.get("action_type", "do_nothing")
        needs_target = action_type in (
            "activate_backup_supplier", "reroute_shipment",
            "increase_safety_stock", "expedite_order", "issue_supplier_alert",
        )
        if needs_target and "target_node_id" not in cleaned:
            logger.debug("LLM sent %s without target_node_id, defaulting to do_nothing.", action_type)
            return SupplyMindAction(action_type="do_nothing")

        # Auto-fix: reroute_via as string instead of list
        if "reroute_via" in cleaned and isinstance(cleaned["reroute_via"], str):
            cleaned["reroute_via"] = [cleaned["reroute_via"]]

        # Auto-fix: additional_stock_days as float
        if "additional_stock_days" in cleaned:
            try:
                cleaned["additional_stock_days"] = int(cleaned["additional_stock_days"])
            except (ValueError, TypeError):
                cleaned.pop("additional_stock_days")

        return SupplyMindAction(**cleaned)

    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s. Input: %s", e, response_text[:200])
        return SupplyMindAction(action_type="do_nothing")
    except Exception as e:
        logger.warning("Failed to parse LLM action: %s. Falling back to do_nothing.", e)
        return SupplyMindAction(action_type="do_nothing")


MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


def get_action(
    client: OpenAI,
    obs: SupplyMindObservation,
    conversation_history: list[dict[str, str]],
    task_id: str = "easy_typhoon_response",
) -> SupplyMindAction:
    """
    Ask GPT-4o to choose an action given the current observation.

    Maintains a rolling conversation history for context, but keeps it
    bounded to avoid token overflow. Retries on transient API errors
    (429 rate limit, 5xx server errors, timeouts) with exponential backoff.
    """
    user_message = format_observation(obs)
    conversation_history.append({"role": "user", "content": user_message})

    # Keep conversation bounded (system + last 10 turns) to reduce token usage
    # and API latency — recent context is most relevant for decision-making
    messages = [{"role": "system", "content": _get_system_prompt(task_id)}]
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
            return parse_action(assistant_text)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on transient errors: rate limits, server errors, timeouts
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
# Run one task
# ---------------------------------------------------------------------------


BASELINE_SEEDS = [42, 99, 7]  # Run 3 seeds per task to showcase episode variation


def run_task(
    env: Any,
    task_id: str,
    client: OpenAI,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Run a single task to completion using the LLM agent.

    Args:
        env: SupplyMindEnvironment instance.
        task_id: Task identifier.
        client: OpenAI client.
        seed: Optional episode variation seed.

    Returns:
        Dict with task_id, score, steps, cumulative_reward, and breakdown.
    """
    logger.info("Starting task: %s", task_id)
    start = time.time()

    obs = env.reset(task_id=task_id, seed=seed)
    conversation_history: list[dict[str, str]] = []
    step_count = 0

    while not obs.done:
        action = get_action(client, obs, conversation_history, task_id=task_id)
        obs = env.step(action)
        step_count += 1

        if step_count % 10 == 0:
            logger.info(
                "  [%s] Step %d -- reward=%.3f, health=%.1f, budget=$%.0f",
                task_id,
                step_count,
                obs.reward,
                obs.financials.supply_chain_health_score,
                obs.financials.budget_remaining,
            )

    # Grade the episode
    result = env.grade()
    elapsed = time.time() - start

    logger.info(
        "Completed %s: score=%.4f, steps=%d, time=%.1fs",
        task_id,
        result["score"],
        step_count,
        elapsed,
    )

    result["elapsed_seconds"] = round(elapsed, 1)
    return result


# ---------------------------------------------------------------------------
# Run all baselines (called by app.py)
# ---------------------------------------------------------------------------


def run_all_baselines(env: Any) -> dict[str, Any]:
    """
    Run the baseline LLM agent on all 3 tasks.

    This is the entry point called by app.py's /baseline endpoint.

    Args:
        env: SupplyMindEnvironment instance.

    Returns:
        Dict with per-task results and an overall summary.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN (or OPENAI_API_KEY) environment variable is not set. "
            "Set it to run the baseline: export HF_TOKEN=hf_..."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    results: dict[str, Any] = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "tasks": {},
    }

    total_score = 0.0
    for task_id in TASK_IDS:
        try:
            # Run with a seed to exercise episode variation (jitter/cascades)
            task_result = run_task(env, task_id, client, seed=BASELINE_SEEDS[0])
        except Exception as e:
            logger.error("Task %s failed with unrecoverable error: %s", task_id, e)
            task_result = {
                "task_id": task_id,
                "score": 0.0,
                "steps_taken": 0,
                "total_steps": 0,
                "cumulative_reward": 0.0,
                "is_done": False,
                "breakdown": {"error": {"score": 0.0, "weight": 1.0}},
                "elapsed_seconds": 0.0,
                "error": str(e),
            }
        results["tasks"][task_id] = task_result
        total_score += task_result["score"]

    results["average_score"] = round(total_score / len(TASK_IDS), 4)

    logger.info("Baseline complete. Average score: %.4f", results["average_score"])
    return results


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set HF_TOKEN (or OPENAI_API_KEY) environment variable first.")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    # Direct mode: import the environment and run locally (no HTTP server needed)
    from server.supply_environment import SupplyMindEnvironment

    print("=" * 60)
    print("SupplyMind Baseline Inference")
    print(f"Model:    {MODEL}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Temp:     {TEMPERATURE}")
    print("=" * 60)

    env = SupplyMindEnvironment()
    results = run_all_baselines(env)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for task_id, task_result in results["tasks"].items():
        print(f"\n  {task_id}:")
        print(f"    Score:      {task_result['score']:.4f}")
        print(f"    Steps:      {task_result['steps_taken']}")
        print(f"    Reward:     {task_result['cumulative_reward']:.4f}")
        print(f"    Time:       {task_result['elapsed_seconds']}s")
        if "breakdown" in task_result:
            print(f"    Breakdown:  {json.dumps(task_result['breakdown'], indent=6)}")

    print(f"\n  Average Score: {results['average_score']:.4f}")
    print("=" * 60)
