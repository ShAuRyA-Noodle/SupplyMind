"""
SupplyMind Supply Chain Graph

Core domain model using NetworkX DiGraph. Represents the supply chain as a
directed graph with 5 node types and 4 edge types. Supports disruption
propagation, inventory tracking, and action application.
"""
from __future__ import annotations

import json
import copy
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from models import SupplyMindAction, ActionResult, SupplierStatus


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

NODE_TYPES = {"supplier", "warehouse", "port", "factory", "customer"}

EDGE_TYPES = {"supplies", "ships_via", "stores_at", "delivers_to"}

# Severity decay per hop in BFS propagation
SEVERITY_DECAY_PER_HOP = 0.20

# Expedite mode cost multipliers
EXPEDITE_MULTIPLIERS: dict[str, float] = {
    "air": 8.0,
    "rail": 3.0,
    "express_sea": 2.0,
}


class SupplyChainGraph:
    """
    Directed graph model of a supply chain network.

    Nodes represent supply chain entities (suppliers, warehouses, ports,
    factories, customers). Edges represent relationships (supplies, ships_via,
    stores_at, delivers_to).
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._raw_data: dict[str, Any] = {}
        # Track which nodes have been disrupted during the episode
        self._ever_offline: set[str] = set()
        # Track delivery delays per customer for SLA
        self._customer_delays: dict[str, float] = {}
        # Track hedges placed
        self._active_hedges: dict[str, float] = {}
        # Track alerted suppliers
        self._alerted_suppliers: set[str] = set()
        # Track rerouted shipments
        self._rerouted_edges: list[tuple[str, str]] = []

    # ──────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────

    def load_from_json(self, filepath: str) -> None:
        """Load supply chain graph from a JSON file."""
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)

        self._raw_data = data
        self.G.clear()

        # Load nodes
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            node_type = node_data["node_type"].lower()
            assert node_type in NODE_TYPES, f"Invalid node type: {node_type}"
            attrs = {k: v for k, v in node_data.items() if k != "id"}
            # Ensure defaults
            if node_type == "supplier":
                attrs.setdefault("is_operational", True)
                attrs.setdefault("risk_score", 0.0)
                attrs.setdefault("backup_supplier_ids", [])
                attrs.setdefault("single_source", False)
                attrs.setdefault("components", [])
                attrs.setdefault("tier", 1)
                attrs.setdefault("lead_time_days", 14)
                attrs.setdefault("annual_spend", 0.0)
            elif node_type == "warehouse":
                attrs.setdefault("inventory_days_cover", 30.0)
                attrs.setdefault("capacity_units", 10000)
                attrs.setdefault("current_inventory_units", 5000)
                attrs.setdefault("daily_consumption_rate", 100)
            elif node_type == "port":
                attrs.setdefault("is_operational", True)
                attrs.setdefault("port_type", "sea")
                attrs.setdefault("avg_dwell_time_hours", 48)
                attrs.setdefault("congestion_score", 0.2)
            elif node_type == "factory":
                attrs.setdefault("is_operational", True)
                attrs.setdefault("production_capacity_daily", 1000)
                attrs.setdefault("utilization_pct", 0.85)
            elif node_type == "customer":
                attrs.setdefault("revenue_contribution", 0.0)
                attrs.setdefault("sla_days", 14)

            self.G.add_node(node_id, **attrs)

        # Load edges
        for edge_data in data.get("edges", []):
            src = edge_data["source"]
            tgt = edge_data["target"]
            edge_type = edge_data["edge_type"].lower()
            assert edge_type in EDGE_TYPES, f"Invalid edge type: {edge_type}"
            attrs = {k: v for k, v in edge_data.items()
                     if k not in ("source", "target")}
            attrs.setdefault("is_active", True)
            self.G.add_edge(src, tgt, **attrs)

        # Initialize customer delay tracking
        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type", "").lower() == "customer":
                self._customer_delays[nid] = 0.0

    def deep_copy(self) -> SupplyChainGraph:
        """Create a deep copy of this graph for simulations."""
        new_graph = SupplyChainGraph()
        new_graph.G = copy.deepcopy(self.G)
        new_graph._raw_data = copy.deepcopy(self._raw_data)
        new_graph._ever_offline = copy.copy(self._ever_offline)
        new_graph._customer_delays = copy.copy(self._customer_delays)
        new_graph._active_hedges = copy.copy(self._active_hedges)
        new_graph._alerted_suppliers = copy.copy(self._alerted_suppliers)
        new_graph._rerouted_edges = copy.copy(self._rerouted_edges)
        return new_graph

    # ──────────────────────────────────────────────
    # Disruption Propagation (BFS)
    # ──────────────────────────────────────────────

    def propagate_disruption(
        self,
        node_id: str,
        severity: float,
        duration_days: float,
    ) -> dict[str, dict[str, float]]:
        """
        Propagate a disruption from a source node through the graph using BFS.

        Severity decays by SEVERITY_DECAY_PER_HOP per hop. Inventory buffers
        at warehouses absorb delay (reducing effective severity downstream).

        Returns:
            Dict of {node_id: {delay_days, severity, revenue_at_risk, time_to_impact}}
        """
        if node_id not in self.G:
            return {}

        affected: dict[str, dict[str, float]] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, float, int, float]] = deque()

        # (node_id, current_severity, hop_count, cumulative_delay_days)
        queue.append((node_id, severity, 0, 0.0))

        while queue:
            current_id, current_sev, hops, cumulative_delay = queue.popleft()

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_sev < 0.05:
                continue

            node_data = self.G.nodes[current_id]
            node_type = node_data.get("node_type", "").lower()

            # Calculate revenue at risk for this node
            revenue_at_risk = 0.0
            if node_type == "customer":
                revenue_at_risk = node_data.get("revenue_contribution", 0.0) * current_sev
            elif node_type == "supplier":
                # Calculate downstream revenue at risk
                revenue_at_risk = self._downstream_revenue(current_id) * current_sev

            # Calculate time to impact based on edge lead times
            time_to_impact = cumulative_delay * 24.0  # convert days to hours

            affected[current_id] = {
                "delay_days": cumulative_delay + duration_days * current_sev,
                "severity": current_sev,
                "revenue_at_risk": revenue_at_risk,
                "time_to_impact": time_to_impact,
            }

            # Only the directly affected node (hop 0) goes offline
            # Downstream nodes get risk scores and delays, not shutdown
            if hops == 0 and current_sev >= 0.5:
                if node_type in ("supplier", "port", "factory"):
                    node_data["is_operational"] = False
                    self._ever_offline.add(current_id)

            # Update risk score for suppliers
            if node_type == "supplier":
                node_data["risk_score"] = max(
                    node_data.get("risk_score", 0.0), current_sev
                )

            # BFS to downstream nodes
            for _, neighbor in self.G.out_edges(current_id):
                if neighbor in visited:
                    continue

                edge_data = self.G.edges[current_id, neighbor]
                edge_lead_time = edge_data.get("lead_time_days",
                                               edge_data.get("transit_time_days", 1))

                # Calculate severity for next hop
                next_sev = current_sev - SEVERITY_DECAY_PER_HOP

                # Inventory buffer absorption at warehouses
                # A warehouse with 30 days of cover fully absorbs a 10-day disruption
                neighbor_data = self.G.nodes[neighbor]
                if neighbor_data.get("node_type", "").lower() == "warehouse":
                    inv_cover = neighbor_data.get("inventory_days_cover", 0.0)
                    disruption_remaining = max(1.0, duration_days * current_sev)
                    # Cap at 80%: even full inventory can't block all disruption signal
                    # (lead-time uncertainty, quality issues, etc.)
                    absorption = min(0.8, inv_cover / disruption_remaining)
                    next_sev *= (1.0 - absorption)

                next_delay = cumulative_delay + edge_lead_time

                if next_sev > 0.05:
                    queue.append((neighbor, next_sev, hops + 1, next_delay))

        return affected

    def _find_downstream_of_type(
        self, node_id: str, target_types: set[str]
    ) -> list[str]:
        """Find all downstream nodes of given types reachable from node_id."""
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        result: list[str] = []

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for _, neighbor in self.G.out_edges(current):
                if neighbor in visited:
                    continue
                neighbor_type = self.G.nodes[neighbor].get("node_type", "").lower()
                if neighbor_type in target_types:
                    result.append(neighbor)
                queue.append(neighbor)

        return result

    def _downstream_revenue(self, node_id: str) -> float:
        """Calculate total downstream customer revenue reachable from a node."""
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        total_revenue = 0.0

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            node_data = self.G.nodes[current]
            if node_data.get("node_type", "").lower() == "customer":
                total_revenue += node_data.get("revenue_contribution", 0.0)

            for _, neighbor in self.G.out_edges(current):
                if neighbor not in visited:
                    queue.append(neighbor)

        return total_revenue

    # ──────────────────────────────────────────────
    # Inventory
    # ──────────────────────────────────────────────

    def inventory_cover(self, node_id: str, disrupted_supplier_ids: list[str]) -> float:
        """
        Calculate days of inventory cover at a node given disrupted suppliers.

        For warehouses: inventory_days_cover based on current inventory and
        consumption rate, adjusted for disrupted inbound supply.

        For other nodes: returns 0.0 (no inventory concept).
        """
        if node_id not in self.G:
            return 0.0

        node_data = self.G.nodes[node_id]
        node_type = node_data.get("node_type", "").lower()

        if node_type != "warehouse":
            return 0.0

        current_inv = node_data.get("current_inventory_units", 0)
        daily_consumption = node_data.get("daily_consumption_rate", 1)

        if daily_consumption <= 0:
            return float("inf") if current_inv > 0 else 0.0

        # Check how many inbound suppliers are disrupted
        inbound_edges = list(self.G.in_edges(node_id, data=True))
        total_inbound_capacity = 0.0
        disrupted_capacity = 0.0

        for src, _, edata in inbound_edges:
            qty = edata.get("quantity", daily_consumption)
            total_inbound_capacity += qty
            if src in disrupted_supplier_ids:
                disrupted_capacity += qty

        # If all supply is disrupted, days = current_inventory / consumption
        if total_inbound_capacity > 0 and disrupted_capacity > 0:
            disruption_fraction = disrupted_capacity / total_inbound_capacity
            effective_consumption = daily_consumption * (1.0 + disruption_fraction * 0.5)
            # Some supply still coming in
            net_daily_drain = effective_consumption - (
                daily_consumption * (1.0 - disruption_fraction)
            )
            if net_daily_drain <= 0:
                return float("inf")
            return current_inv / net_daily_drain

        return current_inv / daily_consumption

    def apply_lead_time_variance(self, rng) -> None:
        """
        Apply ±15% normal variance to edge transit times each step.

        Real-world shipping has natural day-to-day variability due to
        weather, port congestion, customs processing, etc.
        """
        for u, v, edata in self.G.edges(data=True):
            base_key = "_base_lead_time"
            lt_key = "lead_time_days"
            tt_key = "transit_time_days"

            # Store base value on first call
            if base_key not in edata:
                edata[base_key] = edata.get(lt_key, edata.get(tt_key, 1))

            base = edata[base_key]
            variance = rng.normal(0.0, 0.15) * base
            new_lt = max(1.0, base + variance)

            if lt_key in edata:
                edata[lt_key] = new_lt
            if tt_key in edata:
                edata[tt_key] = new_lt

    def deplete_inventory(self, disrupted_supplier_ids: list[str]) -> None:
        """
        Deplete warehouse inventory by one day for disrupted supply paths.

        Applies a bullwhip multiplier (1.2x consumption) when upstream
        suppliers are disrupted, reflecting real-world panic ordering
        and safety-stock drawdown acceleration.
        """
        BULLWHIP_FACTOR = 1.2  # 20% demand amplification per MIT Beer Game studies

        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type", "").lower() != "warehouse":
                continue

            daily_rate = ndata.get("daily_consumption_rate", 0)
            if daily_rate <= 0:
                continue

            # Check if any inbound supplier is disrupted
            inbound_disrupted = False
            for src, _ in self.G.in_edges(nid):
                if src in disrupted_supplier_ids:
                    inbound_disrupted = True
                    break

            if inbound_disrupted:
                # Bullwhip effect: demand amplification during disruptions
                effective_rate = daily_rate * BULLWHIP_FACTOR
                current = ndata.get("current_inventory_units", 0)
                new_inv = max(0, current - effective_rate)
                ndata["current_inventory_units"] = new_inv
                if daily_rate > 0:
                    ndata["inventory_days_cover"] = new_inv / daily_rate
                else:
                    ndata["inventory_days_cover"] = 0.0

    # ──────────────────────────────────────────────
    # Action Application
    # ──────────────────────────────────────────────

    def apply_action(self, action: SupplyMindAction) -> ActionResult:
        """
        Apply a SupplyMindAction to the graph and return the result.

        Validates the action, modifies graph state, and returns cost/effect.
        """
        if action.action_type == "do_nothing":
            return ActionResult(
                success=True,
                message="No action taken.",
                cost=0.0,
                effect_description="Agent chose to wait and observe.",
            )

        if action.action_type == "issue_supplier_alert":
            return self._apply_supplier_alert(action)

        if action.action_type == "activate_backup_supplier":
            return self._apply_activate_backup(action)

        if action.action_type == "reroute_shipment":
            return self._apply_reroute(action)

        if action.action_type == "increase_safety_stock":
            return self._apply_increase_stock(action)

        if action.action_type == "expedite_order":
            return self._apply_expedite(action)

        if action.action_type == "hedge_commodity":
            return self._apply_hedge(action)

        return ActionResult(
            success=False,
            message=f"Unknown action type: {action.action_type}",
            cost=0.0,
            effect_description="",
        )

    def _apply_supplier_alert(self, action: SupplyMindAction) -> ActionResult:
        """Issue a supplier alert (free, information-only)."""
        target = action.target_node_id
        if not target or target not in self.G:
            return ActionResult(
                success=False,
                message=f"Target node '{target}' not found in graph.",
                cost=0.0,
                effect_description="",
            )

        self._alerted_suppliers.add(target)
        node_data = self.G.nodes[target]
        name = node_data.get("name", target)
        is_op = node_data.get("is_operational", True)
        risk = node_data.get("risk_score", 0.0)

        return ActionResult(
            success=True,
            message=f"Alert issued to {name}.",
            cost=0.0,
            effect_description=(
                f"Supplier alert sent to {name}. "
                f"Status: {'operational' if is_op else 'OFFLINE'}. "
                f"Risk score: {risk:.2f}. "
                f"Response will provide updated status information."
            ),
        )

    def _apply_activate_backup(self, action: SupplyMindAction) -> ActionResult:
        """Activate a backup supplier."""
        target = action.target_node_id
        backup_id = action.backup_supplier_id

        if not target or target not in self.G:
            return ActionResult(
                success=False,
                message=f"Target node '{target}' not found.",
                cost=0.0,
                effect_description="",
            )

        if not backup_id or backup_id not in self.G:
            return ActionResult(
                success=False,
                message=f"Backup supplier '{backup_id}' not found.",
                cost=0.0,
                effect_description="",
            )

        target_data = self.G.nodes[target]
        backup_data = self.G.nodes[backup_id]

        # Verify backup is in the backup list
        valid_backups = target_data.get("backup_supplier_ids", [])
        if backup_id not in valid_backups:
            return ActionResult(
                success=False,
                message=f"'{backup_id}' is not a valid backup for '{target}'.",
                cost=0.0,
                effect_description="",
            )

        # Check if backup supplier is itself disrupted
        backup_operational = backup_data.get("is_operational", True)
        backup_risk = backup_data.get("risk_score", 0.0)
        if not backup_operational or backup_risk > 0.5:
            backup_name = backup_data.get("name", backup_id)
            return ActionResult(
                success=False,
                message=(
                    f"Backup supplier '{backup_name}' is currently disrupted "
                    f"(operational={backup_operational}, risk={backup_risk:.0%}). "
                    f"Cannot activate a disrupted backup. Wait for recovery or "
                    f"choose a different backup."
                ),
                cost=0.0,
                effect_description="Backup activation rejected: supplier under active disruption.",
            )

        qualification_cost = 150_000.0  # ISM: $50K-$250K; matches financial.py

        # Activate: connect backup to target's downstream nodes
        # First, activate any existing edges from backup
        for _, downstream in list(self.G.out_edges(backup_id)):
            self.G.edges[backup_id, downstream]["is_active"] = True

        # Copy target's outbound edges to backup
        for _, downstream in list(self.G.out_edges(target)):
            edge_data = dict(self.G.edges[target, downstream])
            if not self.G.has_edge(backup_id, downstream):
                new_edge = edge_data.copy()
                new_edge["is_active"] = True
                if "lead_time_days" in new_edge:
                    new_edge["lead_time_days"] = int(
                        new_edge["lead_time_days"] * 1.2
                    )
                if "cost_per_unit" in new_edge:
                    new_edge["cost_per_unit"] = new_edge["cost_per_unit"] * 1.2
                self.G.add_edge(backup_id, downstream, **new_edge)

        # Backup supplier uses the target's existing supply paths (copied above
        # with 20% lead-time/cost premium). No instant bypass edges — real backup
        # activation requires routing through the existing logistics network.

        # Mark backup as operational
        backup_data["is_operational"] = True
        backup_data["risk_score"] = max(0.0, backup_data.get("risk_score", 0.0) - 0.2)

        total_cost = qualification_cost
        backup_name = backup_data.get("name", backup_id)
        target_name = target_data.get("name", target)

        return ActionResult(
            success=True,
            message=f"Backup supplier {backup_name} activated to replace {target_name}.",
            cost=total_cost,
            effect_description=(
                f"Activated {backup_name} as backup for {target_name}. "
                f"Qualification cost: ${qualification_cost:,.0f}. "
                f"Ongoing premium: {12}% dual-sourcing on buyer procurement share. "
                f"New supply path established with ~20% longer lead time."
            ),
        )

    def _apply_reroute(self, action: SupplyMindAction) -> ActionResult:
        """Reroute shipment through alternative ports."""
        target = action.target_node_id
        reroute_via = action.reroute_via

        if not target or target not in self.G:
            return ActionResult(
                success=False,
                message=f"Target node '{target}' not found.",
                cost=0.0,
                effect_description="",
            )

        if not reroute_via or len(reroute_via) == 0:
            return ActionResult(
                success=False,
                message="No reroute ports specified.",
                cost=0.0,
                effect_description="",
            )

        # Validate all reroute nodes exist and are ports
        for port_id in reroute_via:
            if port_id not in self.G:
                return ActionResult(
                    success=False,
                    message=f"Reroute port '{port_id}' not found.",
                    cost=0.0,
                    effect_description="",
                )

        # Check reroute port operational status — warn and degrade if disrupted
        degraded_ports: list[tuple[str, bool, float]] = []
        for port_id in reroute_via:
            port_data = self.G.nodes[port_id]
            port_op = port_data.get("is_operational", True)
            port_risk = port_data.get("risk_score", 0.0)
            if not port_op or port_risk > 0.5:
                degraded_ports.append((port_id, port_op, port_risk))
        degraded_ids = {p[0] for p in degraded_ports}

        # Deactivate old SHIPS_VIA edges from target
        old_routes = []
        for _, neighbor in list(self.G.out_edges(target)):
            edge_data = self.G.edges[target, neighbor]
            if edge_data.get("edge_type", "").lower() == "ships_via":
                edge_data["is_active"] = False
                old_routes.append(neighbor)

        # Also check inbound SHIPS_VIA edges to target
        for predecessor, _ in list(self.G.in_edges(target)):
            edge_data = self.G.edges[predecessor, target]
            if edge_data.get("edge_type", "").lower() == "ships_via":
                edge_data["is_active"] = False
                old_routes.append(predecessor)

        # Create new edges through reroute ports
        # Connect target to first reroute port, chain ports, connect last port to downstream
        port_change_count = len(reroute_via)
        cost_per_change = 35_000.0  # Industry avg $25K-$50K; matches financial.py
        total_cost = port_change_count * cost_per_change

        # Find downstream nodes from target
        downstream_nodes = []
        for _, neighbor in self.G.out_edges(target):
            edge_data = self.G.edges[target, neighbor]
            if edge_data.get("edge_type", "").lower() != "ships_via":
                downstream_nodes.append(neighbor)

        # Create new shipping path — prefer activating existing dormant edges
        # over synthesizing new ones (degraded ports get 2x transit time)
        for port_id in reroute_via:
            is_degraded = port_id in degraded_ids
            inbound_transit = 10 if is_degraded else 5
            outbound_transit = 14 if is_degraded else 7

            if self.G.has_edge(target, port_id):
                # Activate existing dormant edge
                edge = self.G.edges[target, port_id]
                edge["is_active"] = True
                if is_degraded:
                    edge["transit_time_days"] = max(edge.get("transit_time_days", inbound_transit), inbound_transit)
            else:
                self.G.add_edge(target, port_id,
                                edge_type="ships_via",
                                transit_time_days=inbound_transit,
                                carrier="rerouted",
                                is_active=True)
                self._rerouted_edges.append((target, port_id))

            # Connect reroute port to downstream warehouses/factories
            for downstream in downstream_nodes:
                if self.G.has_edge(port_id, downstream):
                    edge = self.G.edges[port_id, downstream]
                    edge["is_active"] = True
                    if is_degraded:
                        edge["transit_time_days"] = max(edge.get("transit_time_days", outbound_transit), outbound_transit)
                else:
                    self.G.add_edge(port_id, downstream,
                                    edge_type="ships_via",
                                    transit_time_days=outbound_transit,
                                    carrier="rerouted",
                                    is_active=True)
                    self._rerouted_edges.append((port_id, downstream))

        target_name = self.G.nodes[target].get("name", target)
        port_names = [self.G.nodes[p].get("name", p) for p in reroute_via]

        # Build degradation warning if any reroute ports are disrupted
        warning_suffix = ""
        if degraded_ports:
            port_warnings = [
                f"{self.G.nodes[p[0]].get('name', p[0])} "
                f"(operational={p[1]}, risk={p[2]:.0%})"
                for p in degraded_ports
            ]
            warning_suffix = (
                f" WARNING: Degraded reroute ports: {'; '.join(port_warnings)}. "
                f"Transit times doubled for disrupted ports."
            )

        return ActionResult(
            success=True,
            message=f"Shipment rerouted via {', '.join(port_names)}.{warning_suffix}",
            cost=total_cost,
            effect_description=(
                f"Rerouted shipments from {target_name} via "
                f"{', '.join(port_names)}. "
                f"Cost: ${total_cost:,.0f} ({port_change_count} port changes). "
                f"{'Transit times increased due to port disruption.' if degraded_ports else 'May add 2-5 days transit time.'}"
            ),
        )

    def _apply_increase_stock(self, action: SupplyMindAction) -> ActionResult:
        """Increase safety stock at a warehouse."""
        target = action.target_node_id
        extra_days = action.additional_stock_days

        if not target or target not in self.G:
            return ActionResult(
                success=False,
                message=f"Target node '{target}' not found.",
                cost=0.0,
                effect_description="",
            )

        node_data = self.G.nodes[target]
        if node_data.get("node_type", "").lower() != "warehouse":
            return ActionResult(
                success=False,
                message=f"Node '{target}' is not a warehouse.",
                cost=0.0,
                effect_description="",
            )

        if not extra_days or extra_days <= 0:
            return ActionResult(
                success=False,
                message="additional_stock_days must be positive.",
                cost=0.0,
                effect_description="",
            )

        daily_rate = node_data.get("daily_consumption_rate", 100)
        extra_units = daily_rate * extra_days

        # Get cost per unit from inbound supply edges
        cost_per_unit = 45.0  # default
        for src, _ in self.G.in_edges(target):
            edge_data = self.G.edges[src, target]
            if "cost_per_unit" in edge_data:
                cost_per_unit = edge_data["cost_per_unit"]
                break

        # Carrying cost: units * cost_per_unit * (0.25/365) * days
        carrying_cost = extra_units * cost_per_unit * (0.25 / 365.0) * extra_days

        # Actually increase inventory
        current_inv = node_data.get("current_inventory_units", 0)
        capacity = node_data.get("capacity_units", float("inf"))
        new_inv = min(current_inv + extra_units, capacity)
        node_data["current_inventory_units"] = new_inv
        if daily_rate > 0:
            node_data["inventory_days_cover"] = new_inv / daily_rate

        target_name = node_data.get("name", target)

        return ActionResult(
            success=True,
            message=f"Safety stock increased at {target_name} by {extra_days} days.",
            cost=carrying_cost,
            effect_description=(
                f"Added {extra_units:,.0f} units ({extra_days} days cover) "
                f"to {target_name}. "
                f"Carrying cost: ${carrying_cost:,.0f}. "
                f"New inventory: {new_inv:,.0f} units "
                f"({node_data.get('inventory_days_cover', 0):.0f} days cover)."
            ),
        )

    def _apply_expedite(self, action: SupplyMindAction) -> ActionResult:
        """Expedite an order by upgrading transport mode."""
        target = action.target_node_id
        mode = action.expedite_mode

        if not target or target not in self.G:
            return ActionResult(
                success=False,
                message=f"Target node '{target}' not found.",
                cost=0.0,
                effect_description="",
            )

        if not mode or mode not in EXPEDITE_MULTIPLIERS:
            return ActionResult(
                success=False,
                message=f"Invalid expedite mode: {mode}.",
                cost=0.0,
                effect_description="",
            )

        multiplier = EXPEDITE_MULTIPLIERS[mode]

        # Find the supply edge and calculate cost
        base_shipping_cost = 2_500.0  # default base shipping cost
        for src, _ in self.G.in_edges(target):
            edge_data = self.G.edges[src, target]
            if "cost_per_unit" in edge_data:
                # Use edge quantity * cost as base
                qty = edge_data.get("quantity", 100)
                base_shipping_cost = edge_data["cost_per_unit"] * qty
                break

        total_cost = base_shipping_cost * multiplier

        # Reduce lead times on inbound edges
        lead_time_reductions = {
            "air": 0.2,        # 80% reduction
            "rail": 0.5,       # 50% reduction
            "express_sea": 0.7, # 30% reduction
        }
        reduction = lead_time_reductions.get(mode, 0.5)

        for src, _ in self.G.in_edges(target):
            edge_data = self.G.edges[src, target]
            if "lead_time_days" in edge_data:
                edge_data["lead_time_days"] = max(
                    1, int(edge_data["lead_time_days"] * reduction)
                )
            if "transit_time_days" in edge_data:
                edge_data["transit_time_days"] = max(
                    1, int(edge_data["transit_time_days"] * reduction)
                )
            edge_data["transport_mode"] = mode

        target_name = self.G.nodes[target].get("name", target)

        return ActionResult(
            success=True,
            message=f"Order to {target_name} expedited via {mode}.",
            cost=total_cost,
            effect_description=(
                f"Expedited delivery to {target_name} via {mode} freight. "
                f"Cost: ${total_cost:,.0f} ({multiplier}x base). "
                f"Lead time reduced by {int((1 - reduction) * 100)}%."
            ),
        )

    def _apply_hedge(self, action: SupplyMindAction) -> ActionResult:
        """Hedge commodity price risk."""
        commodity = action.commodity
        hedge_amount = action.hedge_amount_usd

        if not commodity:
            return ActionResult(
                success=False,
                message="No commodity specified for hedge.",
                cost=0.0,
                effect_description="",
            )

        if not hedge_amount or hedge_amount <= 0:
            return ActionResult(
                success=False,
                message="Hedge amount must be positive.",
                cost=0.0,
                effect_description="",
            )

        # Premium is 3% of notional
        premium = hedge_amount * 0.06  # 5-8% options premium; matches financial.py
        self._active_hedges[commodity] = self._active_hedges.get(commodity, 0.0) + hedge_amount

        return ActionResult(
            success=True,
            message=f"Hedged {commodity} for ${hedge_amount:,.0f}.",
            cost=premium,
            effect_description=(
                f"Placed hedge on {commodity} with ${hedge_amount:,.0f} notional. "
                f"Option premium: ${premium:,.0f} (3% of notional). "
                f"This protects against price increases for the hedged amount."
            ),
        )

    # ──────────────────────────────────────────────
    # Query Methods
    # ──────────────────────────────────────────────

    def get_total_revenue_at_risk(self) -> float:
        """Sum of revenue_contribution for all disrupted downstream customers."""
        total = 0.0
        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type", "").lower() != "customer":
                continue
            # Check if any upstream path has a disrupted node
            if self._has_disrupted_upstream(nid):
                total += ndata.get("revenue_contribution", 0.0)
        return total

    def _has_disrupted_upstream(self, customer_id: str) -> bool:
        """
        Check if all supply paths to a customer are disrupted.

        Returns True only if every path from suppliers to this customer
        passes through at least one non-operational node. If there is at
        least one fully operational path, returns False (customer is served).
        """
        # Find all tier-1 suppliers reachable upstream from this customer
        # and check if at least one operational path exists
        return not self._has_operational_path_to(customer_id)

    def _has_operational_path_to(self, node_id: str) -> bool:
        """
        Check if there is at least one operational supply path reaching
        this node by traversing backwards through the graph.

        A path is operational if all supplier/port/factory nodes on it
        are operational and all edges are active.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        has_any_supplier_upstream = False

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for predecessor, _ in self.G.in_edges(current):
                if predecessor in visited:
                    continue

                # Skip inactive edges
                edge_data = self.G.edges[predecessor, current]
                if not edge_data.get("is_active", True):
                    continue

                pred_data = self.G.nodes[predecessor]
                node_type = pred_data.get("node_type", "").lower()

                if node_type in ("supplier", "port", "factory"):
                    if pred_data.get("is_operational", True):
                        if node_type == "supplier":
                            # Found an operational supplier via active path
                            return True
                        # Operational intermediate node - keep tracing
                        queue.append(predecessor)
                    else:
                        has_any_supplier_upstream = True
                        # Don't traverse through offline nodes
                else:
                    queue.append(predecessor)

        # If we found no operational supplier but there are suppliers
        # upstream, all paths are disrupted
        return not has_any_supplier_upstream

    def get_health_score(self) -> float:
        """
        Composite health score 0-100.

        Components:
        - 40% operational nodes fraction
        - 30% average inventory cover (normalized to 30 days = 1.0)
        - 30% inverse of average risk score
        """
        if len(self.G.nodes) == 0:
            return 100.0

        # Operational fraction
        operational_types = {"supplier", "port", "factory"}
        total_ops = 0
        online_ops = 0
        for _, ndata in self.G.nodes(data=True):
            ntype = ndata.get("node_type", "").lower()
            if ntype in operational_types:
                total_ops += 1
                if ndata.get("is_operational", True):
                    online_ops += 1

        ops_fraction = online_ops / max(1, total_ops)

        # Average inventory cover
        inv_scores = []
        for _, ndata in self.G.nodes(data=True):
            if ndata.get("node_type", "").lower() == "warehouse":
                cover = ndata.get("inventory_days_cover", 0.0)
                inv_scores.append(min(1.0, cover / 30.0))

        avg_inv = sum(inv_scores) / max(1, len(inv_scores)) if inv_scores else 1.0

        # Average risk score (inverted: low risk = high health)
        risk_scores = []
        for _, ndata in self.G.nodes(data=True):
            if ndata.get("node_type", "").lower() == "supplier":
                risk_scores.append(ndata.get("risk_score", 0.0))

        avg_risk = sum(risk_scores) / max(1, len(risk_scores)) if risk_scores else 0.0
        risk_health = 1.0 - avg_risk

        score = (0.40 * ops_fraction + 0.30 * avg_inv + 0.30 * risk_health) * 100.0
        return round(max(0.0, min(100.0, score)), 1)

    def get_sla_compliance(self) -> float:
        """
        Fraction of customers whose delivery is within SLA.

        Returns float 0.0-1.0.
        """
        customers = [
            (nid, ndata) for nid, ndata in self.G.nodes(data=True)
            if ndata.get("node_type", "").lower() == "customer"
        ]

        if not customers:
            return 1.0

        compliant = 0
        for cust_id, cust_data in customers:
            sla_days = cust_data.get("sla_days", 14)
            delay = self._customer_delays.get(cust_id, 0.0)
            if delay <= sla_days:
                compliant += 1

        return compliant / len(customers)

    def update_customer_delays(self, disrupted_supplier_ids: list[str]) -> None:
        """Update customer delivery delays based on disrupted supply paths."""
        for cust_id, cust_data in self.G.nodes(data=True):
            if cust_data.get("node_type", "").lower() != "customer":
                continue

            # Check upstream for disruptions
            max_delay = 0.0
            for predecessor, _ in self.G.in_edges(cust_id):
                pred_data = self.G.nodes[predecessor]
                pred_type = pred_data.get("node_type", "").lower()

                if pred_type == "warehouse":
                    inv_cover = pred_data.get("inventory_days_cover", 0.0)
                    if inv_cover <= 0:
                        max_delay = max(max_delay, 1.0)
                elif pred_type in ("factory", "port"):
                    if not pred_data.get("is_operational", True):
                        max_delay = max(max_delay, 3.0)

            self._customer_delays[cust_id] = self._customer_delays.get(cust_id, 0.0) + max_delay

    def get_node_statuses(self) -> list[SupplierStatus]:
        """Build list of SupplierStatus for all nodes in the graph."""
        statuses = []
        for nid, ndata in self.G.nodes(data=True):
            node_type = ndata.get("node_type", "").lower()

            # Determine inventory days cover
            inv_cover = 0.0
            if node_type == "warehouse":
                inv_cover = ndata.get("inventory_days_cover", 0.0)

            # Determine operational status
            # Warehouses CAN go offline (e.g. Thailand floods); customers cannot
            is_operational = ndata.get("is_operational", True)
            if node_type == "customer":
                is_operational = True

            # Get active disruptions
            active_disruptions = ndata.get("active_disruption_ids", [])

            statuses.append(SupplierStatus(
                node_id=nid,
                name=ndata.get("name", nid),
                node_type=node_type,
                tier=ndata.get("tier", 0),
                country=ndata.get("country", ""),
                is_operational=is_operational,
                current_risk_score=ndata.get("risk_score", 0.0),
                inventory_days_cover=inv_cover,
                has_backup=len(ndata.get("backup_supplier_ids", [])) > 0,
                backup_supplier_ids=ndata.get("backup_supplier_ids", []),
                active_disruption_ids=active_disruptions,
                revenue_contribution=ndata.get("revenue_contribution", 0.0),
            ))

        return statuses

    def find_backup_suppliers(self, node_id: str) -> list[str]:
        """Return backup supplier IDs for a given node."""
        if node_id not in self.G:
            return []
        return self.G.nodes[node_id].get("backup_supplier_ids", [])

    def get_disrupted_node_ids(self) -> list[str]:
        """Get IDs of all currently non-operational nodes."""
        result = []
        for nid, ndata in self.G.nodes(data=True):
            ntype = ndata.get("node_type", "").lower()
            if ntype in ("supplier", "port", "factory"):
                if not ndata.get("is_operational", True):
                    result.append(nid)
        return result

    def get_customer_ids(self) -> list[str]:
        """Get all customer node IDs."""
        return [
            nid for nid, ndata in self.G.nodes(data=True)
            if ndata.get("node_type", "").lower() == "customer"
        ]

    def get_warehouse_ids(self) -> list[str]:
        """Get all warehouse node IDs."""
        return [
            nid for nid, ndata in self.G.nodes(data=True)
            if ndata.get("node_type", "").lower() == "warehouse"
        ]

    def get_supplier_ids(self) -> list[str]:
        """Get all supplier node IDs."""
        return [
            nid for nid, ndata in self.G.nodes(data=True)
            if ndata.get("node_type", "").lower() == "supplier"
        ]

    def restore_node(self, node_id: str) -> None:
        """Restore a node to operational status."""
        if node_id in self.G:
            self.G.nodes[node_id]["is_operational"] = True
            self.G.nodes[node_id]["risk_score"] = max(
                0.0, self.G.nodes[node_id].get("risk_score", 0.0) - 0.3
            )

    def set_node_disruption(self, node_id: str, signal_id: str) -> None:
        """Mark a node as affected by a disruption signal."""
        if node_id in self.G:
            active_ids = self.G.nodes[node_id].get("active_disruption_ids", [])
            if signal_id not in active_ids:
                active_ids.append(signal_id)
            self.G.nodes[node_id]["active_disruption_ids"] = active_ids

    def clear_node_disruption(self, node_id: str, signal_id: str) -> None:
        """Remove a disruption signal from a node."""
        if node_id in self.G:
            active_ids = self.G.nodes[node_id].get("active_disruption_ids", [])
            if signal_id in active_ids:
                active_ids.remove(signal_id)
            self.G.nodes[node_id]["active_disruption_ids"] = active_ids
            # If no more disruptions, restore
            if not active_ids:
                self.G.nodes[node_id]["risk_score"] = max(
                    0.0, self.G.nodes[node_id].get("risk_score", 0.0) - 0.2
                )

    def total_annual_revenue(self) -> float:
        """Sum of all customer revenue contributions."""
        return sum(
            ndata.get("revenue_contribution", 0.0)
            for _, ndata in self.G.nodes(data=True)
            if ndata.get("node_type", "").lower() == "customer"
        )

    def count_ever_offline(self) -> int:
        """Count nodes that went offline at any point during the episode."""
        return len(self._ever_offline)
