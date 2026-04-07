"""
SupplyMind Financial Engine

Tracks all financial state: budget, costs, revenue loss, SLA penalties,
and commodity prices. Calculates action costs and ongoing financial impact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from models import SupplyMindAction, FinancialSnapshot

if TYPE_CHECKING:
    from server.engine.graph import SupplyChainGraph


# ──────────────────────────────────────────────
# Constants — calibrated from real-world data
# ──────────────────────────────────────────────
# Sources:
#   - Inventory carrying cost: CSCMP State of Logistics Report (23-27% standard)
#   - Backup supplier qualification: ISM survey ($50K-$250K general; $500K-$2M semiconductor)
#   - Dual-sourcing premium: McKinsey/BCG supply chain studies (10-30% over single-source)
#   - Freight multipliers: Freightos/IATA (air 4-6x sea; up to 10-12x bulk)
#   - SLA penalties: Automotive OEM contracts ($22K/min line stoppage per CAR study)
#   - Hedge premium: Options market 5-8% of notional for commodity hedges

# Action cost parameters
BACKUP_QUALIFICATION_COST = 150_000.0  # ISM: $50K-$250K; midpoint for electronics
BACKUP_PREMIUM_RATE = 0.12  # McKinsey: dual-sourcing adds 10-30%; 12% conservative
REROUTE_COST_PER_PORT = 35_000.0  # Industry avg $25K-$50K per alternate port call
CARRYING_COST_RATE = 0.25  # CSCMP benchmark: 25% of inventory value/year
HEDGE_PREMIUM_RATE = 0.06  # Commodity options premium: 5-8% of notional
SLA_PENALTY_PER_DAY = 25_000.0  # Electronics OEM: 2-10% of PO value/week ≈ $25K/day

# Buyer procurement share: fraction of supplier's total annual_spend that represents
# the buyer's procurement for this supply chain. Major suppliers (TSMC, Bosch, CATL)
# have $10B-$80B total revenue; a single product line typically procures 0.5-2%.
# Using 1% aligns with McKinsey/BCG supply chain concentration studies.
BUYER_PROCUREMENT_SHARE = 0.01

# Expedite cost multipliers (Freightos/IATA 2024 data)
# Air freight Shanghai-LA: ~$4.50/kg vs sea ~$0.45/kg = ~10x
# Rail China-Europe: 2-3x sea cost, 50% faster
# Express sea (premium slot): ~2x standard ocean freight
EXPEDITE_MULTIPLIERS = {
    "air": 10.0,    # IATA: air freight 4-12x sea; 10x for electronics
    "rail": 2.5,    # China-Europe rail: 2-3x sea (DB Cargo, China Railway Express)
    "express_sea": 2.0,  # Premium ocean slot: ~2x standard
}


class FinancialEngine:
    """
    Tracks all financial state for a supply chain simulation episode.

    Maintains budget, cumulative costs, revenue losses, SLA penalties,
    and commodity price multipliers.
    """

    def __init__(self, budget: float) -> None:
        self.budget_total: float = budget
        self.budget_remaining: float = budget
        self.cumulative_cost_incurred: float = 0.0
        self.cumulative_revenue_lost: float = 0.0
        self.cumulative_penalty_fees: float = 0.0

        # Commodity prices as multipliers (1.0 = baseline, 1.5 = 50% increase)
        self.commodity_prices: dict[str, float] = {
            "semiconductors": 1.0,
            "rare_earths": 1.0,
            "shipping_container_40ft": 1.0,
            "crude_oil_barrel": 1.0,
            "steel": 1.0,
            "aluminum": 1.0,
            "copper": 1.0,
            "lithium": 1.0,
        }

        # Track daily revenue loss history for grading
        self.daily_loss_history: list[float] = []

        # Track backup supplier ongoing premium costs
        self._active_backup_premiums: dict[str, float] = {}

    def process_action_cost(
        self, action: SupplyMindAction, graph: SupplyChainGraph
    ) -> float:
        """
        Calculate and deduct the cost of an action from the budget.

        Returns the cost incurred. Returns 0.0 if budget insufficient.
        """
        cost = self._calculate_action_cost(action, graph)

        if cost > self.budget_remaining:
            return -1.0  # Signal insufficient budget

        self.budget_remaining -= cost
        self.cumulative_cost_incurred += cost
        return cost

    def _calculate_action_cost(
        self, action: SupplyMindAction, graph: SupplyChainGraph
    ) -> float:
        """Calculate the cost of an action without applying it."""
        if action.action_type == "do_nothing":
            return 0.0

        if action.action_type == "issue_supplier_alert":
            return 0.0

        if action.action_type == "activate_backup_supplier":
            return self._calc_backup_cost(action, graph)

        if action.action_type == "reroute_shipment":
            return self._calc_reroute_cost(action)

        if action.action_type == "increase_safety_stock":
            return self._calc_stock_cost(action, graph)

        if action.action_type == "expedite_order":
            return self._calc_expedite_cost(action, graph)

        if action.action_type == "hedge_commodity":
            return self._calc_hedge_cost(action)

        return 0.0

    def _calc_backup_cost(
        self, action: SupplyMindAction, graph: SupplyChainGraph
    ) -> float:
        """$150K qualification + 12% dual-sourcing premium on buyer's procurement share.

        Note: graph node 'annual_spend' represents the supplier's total company
        revenue (e.g., $18B for TSMC). The dual-sourcing premium applies only to
        the buyer's procurement share — typically 0.5-2% of a major supplier's
        total business for a single product line (McKinsey/BCG).
        """
        qualification = BACKUP_QUALIFICATION_COST

        # Calculate ongoing premium based on buyer's procurement share
        target = action.target_node_id
        if target and target in graph.G:
            annual_spend = graph.G.nodes[target].get("annual_spend", 100_000_000)
            # Buyer procurement share: use buyer_procurement if set, otherwise
            # estimate as 1% of supplier's total business (typical for a single
            # product line from a major semiconductor/auto supplier).
            buyer_procurement = graph.G.nodes[target].get(
                "buyer_procurement", annual_spend * BUYER_PROCUREMENT_SHARE
            )
            daily_premium = buyer_procurement * BACKUP_PREMIUM_RATE / 365.0
            # Charge first week upfront
            premium = daily_premium * 7

            # Store for ongoing charges
            if action.backup_supplier_id:
                self._active_backup_premiums[action.backup_supplier_id] = daily_premium
        else:
            premium = 0.0

        return qualification + premium

    def _calc_reroute_cost(self, action: SupplyMindAction) -> float:
        """$35K per port change (industry avg $25K-$50K per alternate port call)."""
        if action.reroute_via:
            return REROUTE_COST_PER_PORT * len(action.reroute_via)
        return REROUTE_COST_PER_PORT

    def _calc_stock_cost(
        self, action: SupplyMindAction, graph: SupplyChainGraph
    ) -> float:
        """units * cost_per_unit * (0.25/365) * additional_days."""
        target = action.target_node_id
        days = action.additional_stock_days or 7

        if not target or target not in graph.G:
            return 0.0

        node_data = graph.G.nodes[target]
        daily_rate = node_data.get("daily_consumption_rate", 100)
        units = daily_rate * days

        # Get cost per unit from inbound edges
        cost_per_unit = 45.0  # default
        for src, _ in graph.G.in_edges(target):
            edge_data = graph.G.edges[src, target]
            if "cost_per_unit" in edge_data:
                cost_per_unit = edge_data["cost_per_unit"]
                break

        return units * cost_per_unit * (CARRYING_COST_RATE / 365.0) * days

    def _calc_expedite_cost(
        self, action: SupplyMindAction, graph: SupplyChainGraph
    ) -> float:
        """base_shipping_cost * multiplier * crisis_surcharge."""
        mode = action.expedite_mode or "air"
        multiplier = EXPEDITE_MULTIPLIERS.get(mode, 8.0)

        # Calculate base shipping cost from graph edges
        target = action.target_node_id
        base_cost = 2_500.0  # default

        if target and target in graph.G:
            for src, _ in graph.G.in_edges(target):
                edge_data = graph.G.edges[src, target]
                if "cost_per_unit" in edge_data:
                    qty = edge_data.get("quantity", 100)
                    base_cost = edge_data["cost_per_unit"] * qty
                    break

        # Crisis surcharge: during active shipping disruptions, freight rates spike
        # (Freightos: spot rates rose 2-5x during Suez/Red Sea crises)
        shipping_price = self.commodity_prices.get("shipping_container_40ft", 1.0)
        crisis_factor = 1.0 + max(0.0, shipping_price - 1.0) * 0.3

        return base_cost * multiplier * crisis_factor

    def _calc_hedge_cost(self, action: SupplyMindAction) -> float:
        """6% of hedge notional amount (commodity options premium: 5-8%)."""
        if action.hedge_amount_usd:
            return action.hedge_amount_usd * HEDGE_PREMIUM_RATE
        return 0.0

    def calculate_daily_revenue_loss(self, graph: SupplyChainGraph) -> float:
        """
        Calculate daily revenue loss from disrupted supply paths.

        daily_loss = annual_revenue_at_risk / 365 * severity
        """
        revenue_at_risk = graph.get_total_revenue_at_risk()
        daily_loss = revenue_at_risk / 365.0

        # Apply commodity price effects (higher prices = more loss)
        commodity_multiplier = 1.0
        active_commodities = [p for p in self.commodity_prices.values() if p > 1.0]
        if active_commodities:
            commodity_multiplier = max(active_commodities)

        daily_loss *= commodity_multiplier

        # Subtract hedge protection
        for commodity, hedge_amount in graph._active_hedges.items():
            price_change = self.commodity_prices.get(commodity, 1.0)
            if price_change > 1.0:
                # Hedge offsets some loss
                protection = min(daily_loss * 0.5, hedge_amount * (price_change - 1.0) / 365.0)
                daily_loss = max(0.0, daily_loss - protection)

        self.cumulative_revenue_lost += daily_loss
        self.daily_loss_history.append(daily_loss)

        return daily_loss

    def calculate_sla_penalties(self, graph: SupplyChainGraph) -> float:
        """
        Calculate SLA penalty fees for customers beyond their SLA window.

        $25K per customer per day beyond SLA (electronics OEM benchmark:
        2-10% of PO value per week late; automotive line stops cost $1.3M/hr
        per Center for Automotive Research).
        """
        total_penalty = 0.0

        for cust_id, cust_data in graph.G.nodes(data=True):
            if cust_data.get("node_type", "").lower() != "customer":
                continue

            sla_days = cust_data.get("sla_days", 14)
            delay = graph._customer_delays.get(cust_id, 0.0)

            if delay > sla_days:
                days_over = delay - sla_days
                # Compounding penalty: $25K * 1.4^(day-1) per day over SLA
                # Day 1: $25K, Day 3: $49K, Day 5: $96K — reflects escalating
                # contract penalties and production line stoppage costs
                penalty = sum(
                    SLA_PENALTY_PER_DAY * (1.4 ** d) for d in range(int(days_over))
                )
                total_penalty += penalty

        self.cumulative_penalty_fees += total_penalty
        return total_penalty

    def apply_daily_backup_premiums(self) -> float:
        """Apply ongoing daily premium costs for active backup suppliers."""
        daily_total = sum(self._active_backup_premiums.values())
        if daily_total > 0:
            self.budget_remaining -= daily_total
            self.cumulative_cost_incurred += daily_total
        return daily_total

    def apply_commodity_price_change(self, commodity: str, multiplier: float) -> None:
        """
        Apply a commodity price change.

        Args:
            commodity: Name of the commodity
            multiplier: Price multiplier (1.0 = no change, 1.5 = 50% increase)
        """
        self.commodity_prices[commodity] = max(0.1, multiplier)

    def get_snapshot(self, graph: SupplyChainGraph) -> FinancialSnapshot:
        """Build a FinancialSnapshot from current state."""
        return FinancialSnapshot(
            total_revenue_at_risk=graph.get_total_revenue_at_risk(),
            budget_remaining=max(0.0, self.budget_remaining),
            budget_total=self.budget_total,
            cumulative_cost_incurred=self.cumulative_cost_incurred,
            cumulative_revenue_lost=self.cumulative_revenue_lost,
            cumulative_penalty_fees=self.cumulative_penalty_fees,
            supply_chain_health_score=graph.get_health_score(),
            monte_carlo_p50_loss=0.0,  # Filled in by simulation engine
            monte_carlo_p95_loss=0.0,  # Filled in by simulation engine
            commodity_price_changes={
                k: v for k, v in self.commodity_prices.items() if v != 1.0
            },
        )

    def has_budget_for(self, cost: float) -> bool:
        """Check if there is sufficient budget for a given cost."""
        return self.budget_remaining >= cost

    def reset(self, budget: float) -> None:
        """Reset financial state for a new episode."""
        self.budget_total = budget
        self.budget_remaining = budget
        self.cumulative_cost_incurred = 0.0
        self.cumulative_revenue_lost = 0.0
        self.cumulative_penalty_fees = 0.0
        self.daily_loss_history.clear()
        self._active_backup_premiums.clear()
        for k in self.commodity_prices:
            self.commodity_prices[k] = 1.0
