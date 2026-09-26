"""Expose SupplyMind's simulation through ROLL's GEM environment contract.

Importing this module does not require ROLL or GEM. In a ROLL worker, call
``register_env()`` before ``gem.make("supplymind_crisis")``; registration is
deliberately explicit so importing the rest of SupplyMind has no side effects.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from models import SupplyMindAction, SupplyMindObservation
from server.supply_environment import SupplyMindEnvironment

try:
    from gem import Env as GemEnv
except ImportError:  # The main server and tests do not depend on ROLL/GEM.
    GemEnv = object


ENV_ID = "supplymind_crisis"
ENTRY_POINT = (
    "versions.v5_phoenix.roll_integration.env.supplymind_roll_env:SupplyMindRollEnv"
)


class SupplyMindRollEnv(GemEnv):
    """Text-action GEM wrapper with dense rewards from the real simulation."""

    supports_step_reward = True

    def __init__(
        self,
        task_id: str = "easy_typhoon_response",
        max_steps: int | None = None,
        format_penalty: float = -0.2,
        **_unused: Any,
    ) -> None:
        self.task_id = task_id
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.environment = SupplyMindEnvironment()
        self._observation: SupplyMindObservation | None = None
        self._turns = 0

    @staticmethod
    def _render(observation: SupplyMindObservation) -> str:
        """Keep the LLM prompt bounded while preserving usable target IDs."""
        ranked_nodes = sorted(
            observation.node_statuses,
            key=lambda node: node.current_risk_score,
            reverse=True,
        )[:12]
        return json.dumps(
            {
                "day": observation.current_day,
                "days_remaining": observation.days_remaining,
                "summary": observation.compact_summary or observation.situation_summary,
                "budget_remaining": observation.financials.budget_remaining,
                "active_signals": [
                    {
                        "type": signal.disruption_type,
                        "severity": signal.severity,
                        "affected_node_ids": signal.affected_node_ids,
                    }
                    for signal in observation.active_signals
                ],
                "highest_risk_nodes": [
                    {
                        "id": node.node_id,
                        "risk": node.current_risk_score,
                        "operational": node.is_operational,
                        "backups": node.backup_supplier_ids,
                    }
                    for node in ranked_nodes
                ],
                "last_action": (
                    observation.last_action_result.model_dump(mode="json")
                    if observation.last_action_result
                    else None
                ),
            },
            ensure_ascii=False,
        )

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        if GemEnv is not object:
            GemEnv.reset(self, seed=seed)
        self._observation = self.environment.reset(task_id=self.task_id, seed=seed)
        self._turns = 0
        instruction = (
            "Manage the supply-chain crisis. Respond with one JSON object per turn "
            "matching SupplyMindAction. Required key: action_type. Valid actions: "
            "do_nothing; activate_backup_supplier (target_node_id, backup_supplier_id); "
            "reroute_shipment (target_node_id, reroute_via list); "
            "increase_safety_stock (target_node_id, additional_stock_days); "
            "expedite_order (target_node_id, expedite_mode); "
            "hedge_commodity (commodity, hedge_amount_usd); "
            "issue_supplier_alert (target_node_id). Return JSON only."
        )
        return self._render(self._observation), {"env_instruction": instruction}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("Call reset() before step().")
        if self._observation.done or (self.max_steps is not None and self._turns >= self.max_steps):
            raise RuntimeError("Episode is over. Call reset() before step().")

        valid = True
        error = ""
        try:
            payload = json.loads(action)
            if not isinstance(payload, dict):
                raise ValueError("Action must be a JSON object")
            parsed_action = SupplyMindAction.model_validate(payload)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            valid = False
            error = str(exc)
            parsed_action = SupplyMindAction(action_type="do_nothing")

        observation = self.environment.step(parsed_action)
        self._observation = observation
        self._turns += 1
        terminated = bool(observation.done)
        truncated = not terminated and self.max_steps is not None and self._turns >= self.max_steps
        # Invalid syntax advances time, but cannot earn a positive simulation reward.
        reward = float(observation.reward) if valid else min(0.0, float(observation.reward)) + self.format_penalty
        metrics = {"action_is_valid": valid, "success": terminated and observation.financials.supply_chain_health_score > 0}
        info: dict[str, Any] = {
            "metrics": metrics,
            "metrics_agg_mode": {"action_is_valid": "mean", "success": "last"},
            "action_error": error,
            "raw_step_reward": float(observation.reward),
        }
        if terminated or truncated:
            info["episode_score"] = self.environment.grade()["score"]
        return self._render(observation), reward, terminated, truncated, info

    def close(self) -> None:
        self._observation = None


def register_env() -> None:
    """Register the adapter with GEM in the current process, idempotently."""
    try:
        import gem
        from gem.envs.registration import ENV_REGISTRY
    except ImportError as exc:
        raise RuntimeError("Install gem-llm==0.0.4 to register the ROLL adapter") from exc
    existing = ENV_REGISTRY.get(ENV_ID)
    if existing is not None:
        if existing.entry_point != ENTRY_POINT:
            raise ValueError(f"{ENV_ID} is already registered to {existing.entry_point}")
        return
    gem.register(ENV_ID, entry_point=ENTRY_POINT)
