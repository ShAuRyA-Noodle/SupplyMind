"""hierarchical.py — strategic-intent layer over the flat 280-action space.

Maps any (state, risk_tier) to one of 4 high-level intents, then
restricts the underlying policy's logits to the action subset that's
compatible with the chosen intent. This narrows the search space and
forces the policy to commit to a coherent strategy instead of
oscillating across budget/diversify/expedite each step.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

import torch


class HierarchicalIntent(str, Enum):
    PROTECT_BUDGET     = "PROTECT_BUDGET"      # cheap actions only
    DIVERSIFY_RISK     = "DIVERSIFY_RISK"      # backup suppliers, hedging
    EXPEDITE           = "EXPEDITE"             # spend now to recover speed
    ABSORB_AND_MONITOR = "ABSORB_AND_MONITOR"   # do_nothing-heavy, info actions


# 7 action types in the env (flat 280 = 7 types × 40 targets):
ACTION_TYPES = [
    "do_nothing",                # 0
    "activate_backup_supplier",  # 1
    "reroute_shipment",          # 2
    "increase_safety_stock",     # 3
    "expedite_order",            # 4
    "hedge_commodity",           # 5
    "issue_supplier_alert",      # 6
]


# Intent → set of permitted action_type indices
INTENT_TO_TYPES: dict[HierarchicalIntent, set[int]] = {
    HierarchicalIntent.PROTECT_BUDGET:     {0, 6},               # do_nothing, alert
    HierarchicalIntent.DIVERSIFY_RISK:     {0, 1, 5, 6},          # backup, hedge
    HierarchicalIntent.EXPEDITE:           {0, 2, 4, 6},          # reroute, expedite
    HierarchicalIntent.ABSORB_AND_MONITOR: {0, 3, 6},             # safety stock + alert
}


def intent_for_state(
    *,
    risk_tier: str,
    budget_remaining_usd: float,
    days_remaining: int,
    cumulative_cost_usd: float,
) -> HierarchicalIntent:
    """Deterministic intent picker — no model, no magic.

    Rules (in order of precedence):

      1. budget < 5% of total spent so far AND days_remaining > 5
           -> PROTECT_BUDGET (out of money, ride it out)
      2. risk_tier == CRITICAL -> EXPEDITE (spend now to limit cascade)
      3. risk_tier == HIGH AND days_remaining > 7 -> DIVERSIFY_RISK
      4. risk_tier == HIGH -> EXPEDITE (short horizon, time-limited)
      5. risk_tier == MEDIUM AND days_remaining > 14 -> DIVERSIFY_RISK
      6. else -> ABSORB_AND_MONITOR
    """
    cum = max(1.0, cumulative_cost_usd)
    budget_ratio = budget_remaining_usd / (cum + budget_remaining_usd + 1.0)

    if budget_ratio < 0.05 and days_remaining > 5:
        return HierarchicalIntent.PROTECT_BUDGET
    if risk_tier == "CRITICAL":
        return HierarchicalIntent.EXPEDITE
    if risk_tier == "HIGH":
        return (HierarchicalIntent.DIVERSIFY_RISK if days_remaining > 7
                else HierarchicalIntent.EXPEDITE)
    if risk_tier == "MEDIUM" and days_remaining > 14:
        return HierarchicalIntent.DIVERSIFY_RISK
    return HierarchicalIntent.ABSORB_AND_MONITOR


def compatible_actions_for_intent(
    intent: HierarchicalIntent,
    n_actions: int = 280,
    n_targets: int = 40,
) -> torch.Tensor:
    """Boolean mask of length n_actions: True iff action is compatible with intent."""
    types_allowed = INTENT_TO_TYPES.get(intent, set(range(7)))
    mask = torch.zeros(n_actions, dtype=torch.bool)
    for type_idx in types_allowed:
        start = type_idx * n_targets
        end = min(n_actions, start + n_targets)
        mask[start:end] = True
    return mask


def restrict_logits(
    logits: torch.Tensor,
    intent: HierarchicalIntent,
    n_actions: int | None = None,
) -> torch.Tensor:
    """Apply intent mask to logits, returning -inf on incompatible actions."""
    if n_actions is None:
        n_actions = logits.size(-1)
    mask = compatible_actions_for_intent(intent, n_actions=n_actions).to(logits.device)
    return logits.masked_fill(~mask, float("-inf"))


def smoke_test() -> dict:
    """Verify every (risk_tier × budget × horizon) maps to a coherent intent
    and that the resulting mask narrows the action space sensibly."""
    import torch as _t
    cases: list[dict] = []
    for risk in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        for budget in (10_000.0, 1_000_000.0):
            for days in (3, 14, 28):
                cum_cost = 5_000_000.0
                intent = intent_for_state(
                    risk_tier=risk, budget_remaining_usd=budget,
                    days_remaining=days, cumulative_cost_usd=cum_cost,
                )
                mask = compatible_actions_for_intent(intent, n_actions=280)
                cases.append({
                    "risk_tier": risk, "budget": budget, "days": days,
                    "intent": intent.value,
                    "n_actions_allowed": int(mask.sum().item()),
                    "n_actions_total": 280,
                })
    # Aggregate: every intent should appear at least once, mask sizes vary
    intents_seen = set(c["intent"] for c in cases)
    return {
        "n_cases": len(cases),
        "intents_seen": sorted(intents_seen),
        "all_4_intents_reachable": len(intents_seen) == 4,
        "n_actions_distribution": sorted(set(c["n_actions_allowed"] for c in cases)),
        "first_3_cases": cases[:3],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(smoke_test(), indent=2))
