"""
SupplyMind Dense Reward Function

7-component dense reward in [-1.0, 1.0] per step:
1. Revenue preservation (35%)
2. Proactive bonus (15%)
3. Cost penalty (10%)
4. Stockout penalty (25%)
5. Unnecessary action penalty (5%)
6. Health maintenance (5%)
7. SLA compliance (5%)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from models import SupplyMindAction, ActionResult, DisruptionSignal, SupplierStatus

if TYPE_CHECKING:
    from server.engine.graph import SupplyChainGraph
    from server.engine.financial import FinancialEngine


@dataclass
class StepState:
    """Snapshot of state needed for reward calculation."""
    revenue_at_risk: float = 0.0
    health_score: float = 100.0
    sla_compliance: float = 1.0
    budget_total: float = 5_000_000.0
    active_signals: list[DisruptionSignal] = field(default_factory=list)
    node_statuses: list[SupplierStatus] = field(default_factory=list)
    total_customers: int = 3


class RewardCalculator:
    """
    Computes dense per-step reward with 7 weighted components.

    Each component contributes a bounded signal. The final reward is
    clamped to [-1.0, 1.0].
    """

    def __init__(self, initial_total_revenue: float, episode_length: int = 30) -> None:
        self.initial_total_revenue: float = initial_total_revenue
        self._component_history: list[dict[str, float]] = []
        # Anti-spam: general preparatory bonus (hedge/stock) awarded once per episode
        self._proactive_general_collected: bool = False
        # Time discounting: early proactive actions are worth more
        self._episode_length: int = max(1, episode_length)
        self._steps_computed: int = 0

    def capture_state(
        self,
        graph: SupplyChainGraph,
        financial: FinancialEngine,
        active_signals: list[DisruptionSignal],
    ) -> StepState:
        """Capture the current state for reward computation."""
        node_statuses = graph.get_node_statuses()
        customer_count = len(graph.get_customer_ids())

        return StepState(
            revenue_at_risk=graph.get_total_revenue_at_risk(),
            health_score=graph.get_health_score(),
            sla_compliance=graph.get_sla_compliance(),
            budget_total=financial.budget_total,
            active_signals=active_signals,
            node_statuses=node_statuses,
            total_customers=max(1, customer_count),
        )

    def compute_step_reward(
        self,
        prev_state: StepState,
        current_state: StepState,
        action: SupplyMindAction,
        action_result: ActionResult,
    ) -> float:
        """
        Compute reward for current step.

        Returns float in [-1.0, 1.0].
        """
        components: dict[str, float] = {}
        reward = 0.0

        # ─────────────────────────────────────────
        # 1. REVENUE PRESERVATION (35%) - continuous
        # ─────────────────────────────────────────
        # If revenue-at-risk decreased (agent's action helped), positive reward
        delta_risk = prev_state.revenue_at_risk - current_state.revenue_at_risk
        max_risk = self.initial_total_revenue if self.initial_total_revenue > 0 else 1.0
        revenue_signal = delta_risk / max_risk
        revenue_component = 0.35 * max(-1.0, min(1.0, revenue_signal * 10))
        components["revenue_preservation"] = revenue_component
        reward += revenue_component

        # ─────────────────────────────────────────
        # 2. PROACTIVE BONUS (15%) - sparse
        # ─────────────────────────────────────────
        # Acting during WARNING phase (before disruption hits) gets bonus
        proactive_component = 0.0
        # Exclude do_nothing and free issue_supplier_alert (no real mitigation)
        cost_bearing = action.action_type not in ("do_nothing", "issue_supplier_alert")
        if cost_bearing and action_result.success:
            warning_signals = [
                s for s in current_state.active_signals
                if s.lifecycle_phase == "warning"
            ]
            if warning_signals:
                # Check if action targets an affected node
                target_in_warning = False
                if action.target_node_id:
                    for sig in warning_signals:
                        if action.target_node_id in sig.affected_node_ids:
                            target_in_warning = True
                            break

                if target_in_warning:
                    proactive_component = 0.15
                elif action.action_type in ("hedge_commodity", "increase_safety_stock"):
                    # General preparatory bonus awarded once per episode to prevent spam
                    if not self._proactive_general_collected:
                        proactive_component = 0.08
                        self._proactive_general_collected = True

        # Time discounting: early proactive actions are worth more than late ones
        step_fraction = self._steps_computed / self._episode_length
        time_discount = max(0.3, 1.0 - step_fraction * 0.7)
        proactive_component *= time_discount

        components["proactive_bonus"] = proactive_component
        reward += proactive_component

        # ─────────────────────────────────────────
        # 3. COST PENALTY (10%) - continuous
        # ─────────────────────────────────────────
        cost_component = 0.0
        if action_result.cost > 0:
            cost_ratio = action_result.cost / current_state.budget_total
            cost_component = -0.10 * min(1.0, cost_ratio * 5)
        components["cost_penalty"] = cost_component
        reward += cost_component

        # ─────────────────────────────────────────
        # 4. STOCKOUT PENALTY (25%) - event-driven
        # ─────────────────────────────────────────
        stockout_component = 0.0
        stockout_nodes = []
        for node in current_state.node_statuses:
            if node.node_type == "warehouse" and node.inventory_days_cover <= 0:
                stockout_nodes.append(node)

        if stockout_nodes:
            # Count how many customers are downstream of stockout warehouses
            stockout_fraction = len(stockout_nodes) / max(
                1, len([n for n in current_state.node_statuses
                        if n.node_type == "warehouse"])
            )
            stockout_component = -0.25 * stockout_fraction

        components["stockout_penalty"] = stockout_component
        reward += stockout_component

        # ─────────────────────────────────────────
        # 5. UNNECESSARY ACTION PENALTY (5%) - sparse
        # ─────────────────────────────────────────
        unnecessary_component = 0.0
        if action.action_type not in ("do_nothing", "issue_supplier_alert"):
            if action.target_node_id is not None:
                target_affected = any(
                    action.target_node_id in s.affected_node_ids
                    for s in current_state.active_signals
                )
                if not target_affected:
                    unnecessary_component = -0.05
        components["unnecessary_action_penalty"] = unnecessary_component
        reward += unnecessary_component

        # ─────────────────────────────────────────
        # 6. HEALTH MAINTENANCE (5%) - continuous
        # ─────────────────────────────────────────
        health_delta = current_state.health_score - prev_state.health_score
        health_component = 0.05 * max(-1.0, min(1.0, health_delta / 20.0))
        components["health_maintenance"] = health_component
        reward += health_component

        # ─────────────────────────────────────────
        # 7. SLA COMPLIANCE (5%) - continuous
        # ─────────────────────────────────────────
        sla_component = 0.05 * current_state.sla_compliance
        components["sla_compliance"] = sla_component
        reward += sla_component

        # Clamp final reward
        reward = max(-1.0, min(1.0, reward))
        components["total"] = reward

        self._component_history.append(components)
        self._steps_computed += 1
        return reward

    @property
    def component_history(self) -> list[dict[str, float]]:
        """Get the full history of reward components for debugging."""
        return self._component_history

    def get_last_components(self) -> dict[str, float]:
        """Get the most recent reward component breakdown."""
        if self._component_history:
            return self._component_history[-1]
        return {}
