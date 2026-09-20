"""
LLM-RL Hybrid Explainability for SupplyMind.

Routes through the single gateway ``supplymind.llm.providers`` (CLAUDE.md §0.4,
§7.1). After each RL action, decodes state to text and asks the configured
provider for a natural-language explanation of why the agent chose that action.

Provider is env-driven (``SUPPLYMIND_LLM_PROVIDER``): ``openrouter`` (default,
cloud) or ``ollama`` (local edge mode). Ollama is selectable EDGE MODE only — it
is never a hidden hard dependency. When no provider is available (revoked key /
stopped daemon), the explainer degrades LOUDLY by raising :class:`ExplainerError`
with the provider's surfaced reason; it never fabricates an explanation.

See ``rl.lora.edge_capability_matrix`` for the per-feature cloud/edge/local map.

Pre-populates common scenarios to cache/explanations.json for demo speed.

Usage:
    from rl.explainer import explain_action
    explanation = explain_action(obs, action, reward_components)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = CACHE_DIR / "explanations_cache.json"

ACTION_NAMES = {
    "do_nothing": "Wait and observe",
    "activate_backup_supplier": "Activate backup supplier",
    "reroute_shipment": "Reroute shipment via alternate port",
    "increase_safety_stock": "Increase safety stock buffer",
    "expedite_order": "Expedite order via air freight",
    "hedge_commodity": "Hedge commodity price exposure",
    "issue_supplier_alert": "Issue early warning alert to supplier",
}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _cache_key(state_summary: str, action: str) -> str:
    return hashlib.md5(f"{state_summary}|{action}".encode()).hexdigest()


def decode_state_to_text(obs) -> str:
    """Decode structured observation to natural language summary.

    Args:
        obs: SupplyMindObservation (Pydantic model from core env).

    Returns:
        Human-readable state summary (~200 words).
    """
    lines = []
    lines.append(f"Day {obs.current_day} of {obs.current_day + obs.days_remaining}.")

    # Financials
    fin = obs.financials
    lines.append(f"Budget: ${fin.budget_remaining:,.0f} of ${fin.budget_total:,.0f} remaining.")
    lines.append(f"Revenue lost so far: ${fin.cumulative_revenue_lost:,.0f}.")
    lines.append(f"Supply chain health: {fin.supply_chain_health_score:.0f}/100.")
    lines.append(f"Monte Carlo P50 loss: ${fin.monte_carlo_p50_loss:,.0f}, "
                 f"P95: ${fin.monte_carlo_p95_loss:,.0f}.")

    # Active disruptions
    if obs.active_signals:
        lines.append(f"\nActive disruptions ({len(obs.active_signals)}):")
        for sig in obs.active_signals[:5]:
            lines.append(f"  - {sig.disruption_type} ({sig.lifecycle_phase}): "
                         f"severity {sig.severity:.2f}, "
                         f"affecting {', '.join(sig.affected_node_ids[:3])}")

    # Critical nodes
    critical = [n for n in obs.node_statuses if not n.is_operational or n.current_risk_score > 0.5]
    if critical:
        lines.append(f"\nCritical nodes ({len(critical)}):")
        for n in critical[:5]:
            status = "OFFLINE" if not n.is_operational else f"risk={n.current_risk_score:.2f}"
            lines.append(f"  - {n.name} ({n.node_type}): {status}, "
                         f"inventory={n.inventory_days_cover:.0f}d")

    return "\n".join(lines)


def _build_prompt(state_text: str, action_type: str, target_node: str | None,
                  reward_components: dict | None,
                  shap_top: list | None = None,
                  counterfactual_p50: float | None = None,
                  rag_precedent: str | None = None) -> str:
    """Build structured 4-section Ollama prompt.

    Optional context injections:
      - shap_top:            List of (feature_name, importance) tuples
      - counterfactual_p50:  Projected P50 loss without this action ($)
      - rag_precedent:       Most relevant historical precedent text
    """
    action_desc = ACTION_NAMES.get(action_type, action_type)
    target_text = f" targeting {target_node}" if target_node else ""

    reward_text = ""
    if reward_components:
        reward_text = "\nReward breakdown: " + ", ".join(
            f"{k}={v:+.3f}" for k, v in reward_components.items()
        )

    shap_text = ""
    if shap_top:
        shap_text = "\nSHAP top features: " + ", ".join(
            f"{n}={v:+.3f}" for n, v in shap_top[:5]
        )

    cf_text = ""
    if counterfactual_p50 is not None:
        cf_text = f"\nCounterfactual (no action): projected P50 loss = ${counterfactual_p50:,.0f}"

    rag_text = ""
    if rag_precedent:
        rag_text = f"\nHistorical precedent: {rag_precedent}"

    return f"""You are a supply chain risk analyst AI explaining an RL agent's decision. Produce a 4-section structured explanation.

CURRENT STATE:
{state_text}

ACTION TAKEN: {action_desc}{target_text}
{reward_text}{shap_text}{cf_text}{rag_text}

Respond with EXACTLY these four sections, each labelled:

## Decision
One sentence stating the action taken and the immediate intent.

## Evidence
Cite 2-3 specific state facts (node names, financial figures, SHAP features) that justify the decision.

## Counterfactual
State what would likely happen if this action were NOT taken, using the P50 projection when available.

## Precedent
Reference the historical precedent (if provided) or a comparable pattern from the supply chain domain.

Be specific, concise, and grounded in the state facts. No generic platitudes."""


_REQUIRED_SECTIONS = ("## Decision", "## Evidence", "## Counterfactual", "## Precedent")


def _passes_quality_gate(text: str) -> bool:
    """Production quality gate: all four sections must be present."""
    return all(section in text for section in _REQUIRED_SECTIONS)


class ExplainerError(RuntimeError):
    """Raised when the provider is unavailable or output fails the quality gate.
    No fabricated fallback — the failure is surfaced loud (CLAUDE.md §0/§3)."""


def explain_action(
    obs,
    action_type: str,
    target_node: str | None = None,
    reward_components: dict | None = None,
    model_name: str | None = None,
    shap_top: list | None = None,
    counterfactual_p50: float | None = None,
    rag_precedent: str | None = None,
    max_regen: int = 2,
) -> str:
    """Generate a structured 4-section explanation via the LLM gateway.

    PRODUCTION PATH: routes through ``supplymind.llm.providers`` — provider is
    env-driven (``openrouter`` default | ``ollama`` edge). No heuristic fallback.
    Raises ExplainerError if no provider is available (degraded, reason surfaced)
    or the output fails the quality gate after max_regen attempts.

    ``model_name`` is an OPTIONAL explicit slug (e.g. an Ollama edge tag). When
    None the gateway resolves the model from the provider's env var; there is no
    hardcoded default slug (a silent wrong-model default is a fake per §0).

    For legacy heuristic output (tests/comparison only), import from
    rl.legacy.fallbacks.explainer_heuristic.
    """
    from supplymind.llm.providers import get_provider

    from rl.lora.edge_capability_matrix import run_coro_sync

    state_text = decode_state_to_text(obs)
    cache = _load_cache()
    key = _cache_key(state_text[:200], action_type)

    if key in cache and _passes_quality_gate(cache[key]):
        return cache[key]

    prompt = _build_prompt(
        state_text, action_type, target_node, reward_components,
        shap_top=shap_top, counterfactual_p50=counterfactual_p50, rag_precedent=rag_precedent,
    )

    provider = get_provider()
    last_output = ""
    for attempt in range(max_regen + 1):
        resp = run_coro_sync(provider.complete(
            [{"role": "user", "content": prompt}],
            model=model_name, schema=None, max_tokens=768, temperature=0.2,
            retries=0,
        ))
        if not resp.content:
            # No usable content: surface the provider's honest reason loud.
            reason = resp.degrade_reason or resp.error or "provider returned empty content"
            raise ExplainerError(
                f"explainer provider '{resp.provider}' unavailable: {reason}"
            )
        text = resp.content.strip()
        last_output = text
        if _passes_quality_gate(text):
            cache[key] = text
            _save_cache(cache)
            return text
        logger.warning(
            "Quality gate fail (attempt %d): missing section. Regenerating...", attempt + 1,
        )
        prompt += "\n\nYour previous answer omitted a required section. Produce ALL FOUR section headers exactly: ## Decision, ## Evidence, ## Counterfactual, ## Precedent."

    raise ExplainerError(
        f"Explainer failed quality gate after {max_regen + 1} attempts. "
        f"Last output: {last_output[:300]}"
    )


def pre_populate_cache(n_scenarios: int = 50) -> int:
    """Pre-populate the explanation cache with common scenarios.

    Runs the scripted agent on all 3 tasks and calls explain_action() for each
    action taken, caching the result. explain_action() routes through the LLM
    gateway, so a live provider (OpenRouter key, or an Ollama edge daemon with a
    pulled model) is REQUIRED — this does not use heuristic explanations and will
    raise ExplainerError if no provider is available.

    Returns number of explanations cached.
    """
    import sys
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action

    env = SupplyMindEnvironment()
    cache = _load_cache()
    count = 0

    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        step = 0
        while not obs.done and count < n_scenarios:
            action = choose_action(obs, step)
            explanation = explain_action(obs, action.action_type, action.target_node_id)
            state_text = decode_state_to_text(obs)
            key = _cache_key(state_text[:200], action.action_type)
            cache[key] = explanation
            count += 1
            obs = env.step(action)
            step += 1

    _save_cache(cache)
    logger.info("Pre-populated %d explanations to cache", count)
    return count
