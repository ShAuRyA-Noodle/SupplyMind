"""
Shared building blocks for the SupplyMind LLM agents.

Both entry points — baseline.py (in-process, pydantic observations) and
inference.py (HTTP client, dict observations) — used to carry byte-drifted
copies of the same system prompt, task hints, JSON extraction, action parsing,
and observation formatting (~350 duplicated lines). This module is the single
source of truth for all of that. The two callers differ only in transport and
return type:

    * baseline.py   feeds ``SupplyMindObservation.model_dump(mode="json")`` in
      and wraps the returned action dict in ``SupplyMindAction(**action)``.
    * inference.py  feeds the raw ``/step`` response dict in and uses the
      returned action dict directly as the ``/step`` payload.

TASK_HINTS leak the ground-truth solution of each benchmark task into the
agent's system prompt. They are therefore OFF by default (``use_hints=False``):
an honest baseline measures the model's own supply-chain reasoning. Hinted runs
are still supported for dual-reporting, but callers must label those output
records ``"hinted": true`` (see baseline.py / inference.py).
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

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

# Task-specific strategy hints. These reveal the intended solution of each
# benchmark task, so they are appended to the system prompt ONLY when a run
# explicitly opts in (use_hints=True) and the resulting records are labelled
# "hinted": true. Default runs never see these.
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


def build_system_prompt(task_id: str, use_hints: bool = False) -> str:
    """Build the agent system prompt.

    Args:
        task_id: Benchmark task identifier.
        use_hints: When True, append the task's ground-truth strategy hint.
            Default False for an honest baseline. Hinted runs must be labelled
            ``"hinted": true`` by the caller.
    """
    if use_hints:
        return BASE_SYSTEM_PROMPT + TASK_HINTS.get(task_id, "")
    return BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# JSON extraction / cleaning
# ---------------------------------------------------------------------------


def _clean_json_quirks(text: str) -> str:
    """Remove common LLM JSON quirks: JS comments, trailing commas."""
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


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "do_nothing", "activate_backup_supplier", "reroute_shipment",
    "increase_safety_stock", "expedite_order", "hedge_commodity",
    "issue_supplier_alert",
}

# Actions that are meaningless without a target node.
_ACTIONS_NEEDING_TARGET = {
    "activate_backup_supplier", "reroute_shipment",
    "increase_safety_stock", "expedite_order", "issue_supplier_alert",
}


def parse_action(response_text: str) -> dict:
    """
    Parse an LLM response into a validated action dict.

    Handles all common LLM failure modes: markdown fences, arrays instead of
    objects, prose around JSON, empty responses, invalid JSON, missing required
    fields, and action_type typos (fuzzy match). Returns a plain dict with a
    valid ``action_type``; callers may pass it straight to the ``/step`` payload
    or splat it into ``SupplyMindAction(**action)``. Falls back to
    ``{"action_type": "do_nothing"}`` on any unrecoverable error.
    """
    try:
        text = _extract_json(response_text)
        data = json.loads(text)

        if not isinstance(data, dict):
            logger.warning("LLM returned non-dict JSON: %s", type(data).__name__)
            return {"action_type": "do_nothing"}

        # Remove null values so downstream defaults apply.
        cleaned = {k: v for k, v in data.items() if v is not None}

        # Fuzzy-match action_type for common typos / casing.
        action_type = cleaned.get("action_type", "do_nothing")
        if action_type not in VALID_ACTIONS:
            lower_map = {a.lower().replace("_", ""): a for a in VALID_ACTIONS}
            normalized = action_type.lower().replace("_", "").replace("-", "").replace(" ", "")
            if normalized in lower_map:
                cleaned["action_type"] = lower_map[normalized]
                logger.debug("Fuzzy-matched action_type '%s' -> '%s'", action_type, cleaned["action_type"])
            else:
                logger.warning("Unknown action_type '%s', defaulting to do_nothing.", action_type)
                return {"action_type": "do_nothing"}

        # Actions needing a target but missing one -> safe no-op.
        action_type = cleaned.get("action_type", "do_nothing")
        if action_type in _ACTIONS_NEEDING_TARGET and "target_node_id" not in cleaned:
            logger.debug("LLM sent %s without target_node_id, defaulting to do_nothing.", action_type)
            return {"action_type": "do_nothing"}

        # Auto-fix: reroute_via as string instead of list.
        if "reroute_via" in cleaned and isinstance(cleaned["reroute_via"], str):
            cleaned["reroute_via"] = [cleaned["reroute_via"]]

        # Auto-fix: additional_stock_days as float / string.
        if "additional_stock_days" in cleaned:
            try:
                cleaned["additional_stock_days"] = int(cleaned["additional_stock_days"])
            except (ValueError, TypeError):
                cleaned.pop("additional_stock_days")

        return cleaned

    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s. Input: %s", e, response_text[:200])
        return {"action_type": "do_nothing"}
    except Exception as e:  # noqa: BLE001 - last-resort guard, logged loudly
        logger.warning("Failed to parse LLM action: %s. Falling back to do_nothing.", e)
        return {"action_type": "do_nothing"}


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------


def format_observation(obs: dict) -> str:
    """Format an observation dict into a concise user message for the LLM.

    Operates on a plain dict so both callers share it: baseline.py passes
    ``SupplyMindObservation.model_dump(mode="json")`` and inference.py passes
    the raw ``/step`` response dict (identical field names / shape).
    """
    parts: list[str] = []

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

    # Situation summary (natural language)
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

    # Node statuses -- only show at-risk or disrupted nodes
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
            active_disruptions = n.get("active_disruption_ids")
            if active_disruptions:
                parts.append(f"    Active disruptions: {', '.join(active_disruptions)}")
        parts.append("")

    # Inventory warnings for warehouses running low
    low_inv = [
        n for n in node_statuses
        if n.get("node_type") == "warehouse" and 0 < n.get("inventory_days_cover", 0) <= 7
    ]
    if low_inv:
        parts.append("--- LOW INVENTORY WARNING ---")
        for n in low_inv:
            parts.append(f"  {n.get('node_id', '?')} ({n.get('name', '?')}): {n.get('inventory_days_cover', 0):.0f} days remaining")
        parts.append("")

    return "\n".join(parts)
