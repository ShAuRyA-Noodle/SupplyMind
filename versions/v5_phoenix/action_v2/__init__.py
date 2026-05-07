"""action_v2 — Hierarchical + Conformal action selection.

Two-level decision wrapper around any flat policy:

  Level 1 (strategic): pick high-level intent
      ∈ {PROTECT_BUDGET, DIVERSIFY_RISK, EXPEDITE, ABSORB_AND_MONITOR}
      based on budget / risk_tier / horizon — deterministic rule.

  Level 2 (tactical): given the chosen intent, the underlying flat
      policy is restricted to the action subset compatible with that
      intent, and conformal uncertainty filters reject actions
      whose predictive interval exceeds a calibrated threshold.

Pass-7 C14.
"""
from .hierarchical import (HierarchicalIntent, intent_for_state,
                            compatible_actions_for_intent)
from .conformal import ConformalActionFilter, calibrate_conformal

__all__ = [
    "HierarchicalIntent", "intent_for_state",
    "compatible_actions_for_intent",
    "ConformalActionFilter", "calibrate_conformal",
]
