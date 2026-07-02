"""
SupplyMind OpenEnv Models

Defines the typed contract between agent and environment:
- SupplyMindAction: What the agent can do (7 action types)
- SupplyMindObservation: What the agent sees (signals, node statuses, financials)
- SupplyMindState: Episode metadata
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────
# Sub-models used in Observation
# ──────────────────────────────────────────────

class DisruptionSignal(BaseModel):
    """A single disruption signal detected in the environment."""
    signal_id: str
    disruption_type: str = Field(
        description="Type of disruption, e.g.: cyclone, flood, labor_strike, "
        "sanctions, cyber_attack, geopolitical, shipping_disruption, blockade, "
        "production_halt, supply_shortage, commodity_shock, recovery_signal"
    )
    severity: float = Field(ge=0.0, le=1.0, description="Disruption severity 0.0-1.0")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence 0.0-1.0")
    affected_region: str = Field(description="Geographic region name")
    affected_node_ids: list[str] = Field(default_factory=list, description="Supply chain nodes in blast radius")
    time_to_impact_hours: float = Field(description="Estimated hours until impact hits")
    estimated_duration_days: float = Field(description="Expected disruption duration in days")
    description: str = Field(description="Human-readable summary of the signal")
    lifecycle_phase: str = Field(
        default="warning",
        description="One of: warning, active, recovery, resolved"
    )


class SupplierStatus(BaseModel):
    """Current status of a supply chain node."""
    node_id: str
    name: str
    node_type: str = Field(description="One of: supplier, warehouse, port, factory, customer")
    tier: int = Field(default=0, description="Supply chain tier (1=direct, 2=indirect, 3=deep)")
    country: str = Field(default="", description="Country code (e.g., TW, US, KR)")
    is_operational: bool = Field(default=True)
    current_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    inventory_days_cover: float = Field(default=0.0, description="Days of buffer remaining")
    has_backup: bool = Field(default=False)
    backup_supplier_ids: list[str] = Field(default_factory=list)
    active_disruption_ids: list[str] = Field(default_factory=list, description="Signal IDs affecting this node")
    revenue_contribution: float = Field(default=0.0, description="Annual revenue contribution in USD")


class FinancialSnapshot(BaseModel):
    """Current financial state of the supply chain."""
    total_revenue_at_risk: float = Field(default=0.0, description="Current revenue at risk in USD")
    budget_remaining: float = Field(description="Budget available for mitigation actions in USD")
    budget_total: float = Field(description="Total starting budget in USD")
    cumulative_cost_incurred: float = Field(default=0.0, description="Total mitigation costs spent")
    cumulative_revenue_lost: float = Field(default=0.0, description="Total revenue lost to disruptions")
    cumulative_penalty_fees: float = Field(default=0.0, description="Total SLA penalty fees incurred")
    supply_chain_health_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Composite health 0-100")

    # Monte Carlo projections
    monte_carlo_p50_loss: float = Field(default=0.0, description="P50 projected total loss")
    monte_carlo_p95_loss: float = Field(default=0.0, description="P95 projected total loss")

    # Commodity prices
    commodity_price_changes: dict[str, float] = Field(
        default_factory=dict,
        description="Commodity price changes as multipliers (1.0 = no change, 1.5 = 50% increase)"
    )


class ActionResult(BaseModel):
    """Feedback on the last action taken."""
    success: bool = Field(default=True)
    message: str = Field(default="")
    cost: float = Field(default=0.0, description="Cost of the action in USD")
    effect_description: str = Field(default="", description="What the action achieved")


# ──────────────────────────────────────────────
# Core OpenEnv Models
# ──────────────────────────────────────────────

class SupplyMindAction(BaseModel):
    """
    An action taken by the supply chain risk manager.

    The agent selects one action type per step with relevant parameters.
    Different action types require different parameters:

    - do_nothing: No parameters needed
    - activate_backup_supplier: target_node_id + backup_supplier_id
    - reroute_shipment: target_node_id + reroute_via (list of port IDs)
    - increase_safety_stock: target_node_id + additional_stock_days
    - expedite_order: target_node_id + expedite_mode
    - hedge_commodity: commodity + hedge_amount_usd
    - issue_supplier_alert: target_node_id
    """
    action_type: Literal[
        "do_nothing",
        "activate_backup_supplier",
        "reroute_shipment",
        "increase_safety_stock",
        "expedite_order",
        "hedge_commodity",
        "issue_supplier_alert",
    ] = Field(description="The type of action to take")

    # Target node in supply chain graph
    target_node_id: Optional[str] = Field(
        default=None,
        description="Target supply chain node ID (supplier/warehouse/port)"
    )

    # For activate_backup_supplier
    backup_supplier_id: Optional[str] = Field(
        default=None,
        description="ID of the backup supplier to activate"
    )

    # For reroute_shipment
    reroute_via: Optional[list[str]] = Field(
        default=None,
        description="List of port IDs for the alternative route"
    )

    # For increase_safety_stock
    additional_stock_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=90,
        description="Number of extra days of inventory to order (1-90)"
    )

    # For expedite_order
    expedite_mode: Optional[Literal["air", "rail", "express_sea"]] = Field(
        default=None,
        description="Transport mode upgrade"
    )

    # For hedge_commodity
    commodity: Optional[str] = Field(
        default=None,
        description="Commodity name to hedge (e.g., 'semiconductors', 'rare_earths')"
    )
    hedge_amount_usd: Optional[float] = Field(
        default=None,
        gt=0,
        description="Hedge notional amount in USD"
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "SupplyMindAction":
        """Enforce required parameters per action type."""
        t = self.action_type
        if t == "activate_backup_supplier" and not self.backup_supplier_id:
            raise ValueError("activate_backup_supplier requires backup_supplier_id")
        if t == "reroute_shipment" and not self.reroute_via:
            raise ValueError("reroute_shipment requires reroute_via")
        if t == "hedge_commodity" and not self.commodity:
            raise ValueError("hedge_commodity requires commodity")
        if t == "expedite_order" and not self.expedite_mode:
            raise ValueError("expedite_order requires expedite_mode")
        if t in ("activate_backup_supplier", "reroute_shipment",
                 "increase_safety_stock", "expedite_order",
                 "issue_supplier_alert") and not self.target_node_id:
            raise ValueError(f"{t} requires target_node_id")
        return self


class SupplyMindObservation(BaseModel):
    """
    Full observation of the supply chain state.

    Contains both structured data (for programmatic agents) and a natural
    language situation_summary (for LLM-based agents).
    """
    # Time management
    current_day: int = Field(description="Current simulation day (0-based)")
    days_remaining: int = Field(description="Days left in episode")

    # Disruption signals
    active_signals: list[DisruptionSignal] = Field(
        default_factory=list,
        description="All currently active disruption signals"
    )
    new_signals: list[DisruptionSignal] = Field(
        default_factory=list,
        description="Signals that appeared THIS step (subset of active_signals)"
    )

    # Supply chain node statuses
    node_statuses: list[SupplierStatus] = Field(
        default_factory=list,
        description="Current status of all supply chain nodes"
    )

    # Financial state
    financials: FinancialSnapshot = Field(
        default_factory=lambda: FinancialSnapshot(budget_remaining=0, budget_total=0)
    )

    # Feedback on last action
    last_action_result: Optional[ActionResult] = Field(
        default=None,
        description="Result of the previous action taken"
    )

    # Natural language summary for LLM agents
    situation_summary: str = Field(
        default="",
        description="Human-readable summary of current situation for LLM reasoning"
    )

    # Compact summary for token-constrained LLM agents (≤500 tokens)
    compact_summary: str = Field(
        default="",
        description="Compact summary (≤500 tokens) with top risks, budget, disruptions, and suggested action"
    )

    # Episode control
    reward: float = Field(default=0.0, description="Reward for this step")
    done: bool = Field(default=False, description="Whether the episode is over")
    info: dict = Field(default_factory=dict, description="Additional metadata")


class SupplyMindState(BaseModel):
    """Episode metadata and tracking."""
    episode_id: str = Field(default="", description="Unique episode identifier")
    step_count: int = Field(default=0, description="Current step number")
    task_id: str = Field(default="", description="Current task identifier")
    task_name: str = Field(default="", description="Human-readable task name")
    task_difficulty: str = Field(default="", description="easy, medium, or hard")
    total_steps: int = Field(default=0, description="Maximum steps in this episode")
    is_done: bool = Field(default=False, description="Whether episode has ended")
    cumulative_reward: float = Field(default=0.0, description="Sum of all rewards so far")
