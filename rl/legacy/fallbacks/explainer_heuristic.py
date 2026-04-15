"""
LEGACY: Rule-based heuristic explainer. Kept for reference/tests only.
Production path (rl/explainer.py) uses Ollama exclusively with no fallback.

This code is imported only by tests and historical comparison notebooks.
"""

from __future__ import annotations


def heuristic_explanation(obs, action_type: str, target_node: str | None) -> str:
    """Rule-based explanation. Not used in production."""
    fin = obs.financials
    health = fin.supply_chain_health_score
    budget_pct = (fin.budget_remaining / max(fin.budget_total, 1)) * 100

    explanations = {
        "do_nothing": (
            f"With health at {health:.0f}/100 and {budget_pct:.0f}% budget remaining, "
            f"the agent conserves resources. No immediate threats require action."
        ),
        "activate_backup_supplier": (
            f"Activating backup for {target_node} to mitigate disruption risk. "
            f"With P95 projected loss of ${fin.monte_carlo_p95_loss:,.0f}, "
            f"the $150K qualification cost is justified to protect revenue."
        ),
        "reroute_shipment": (
            f"Rerouting shipments away from {target_node} via alternate port. "
            f"The additional transit cost is offset by avoiding delays that "
            f"would increase cumulative losses beyond ${fin.cumulative_revenue_lost:,.0f}."
        ),
        "increase_safety_stock": (
            f"Building safety stock buffer at {target_node} to absorb "
            f"potential supply disruption. Current inventory cover may be insufficient "
            f"given active disruption severity."
        ),
        "expedite_order": (
            f"Expediting via air freight to {target_node} due to critical inventory levels. "
            f"The premium cost is warranted: stockout risk would trigger "
            f"SLA penalties exceeding ${fin.cumulative_penalty_fees:,.0f}."
        ),
        "hedge_commodity": (
            f"Hedging commodity exposure to lock in current prices. "
            f"Price volatility signals suggest further increases, "
            f"which would compound supply chain costs."
        ),
        "issue_supplier_alert": (
            f"Issuing early warning to {target_node} (free action). "
            f"Proactive alerting improves supplier response time and "
            f"earns the proactive bonus in grading."
        ),
    }
    return explanations.get(action_type, f"Action: {action_type} on {target_node}")
