"""
SupplyMind Simulation Engine

Core step loop that orchestrates graph, disruptions, financial, rewards,
and Monte Carlo engines. This is the heart of the environment.
"""
from __future__ import annotations

from typing import Optional

from models import (
    SupplyMindAction,
    SupplyMindObservation,
    ActionResult,
    DisruptionSignal,
    FinancialSnapshot,
)
from server.engine.graph import SupplyChainGraph
from server.engine.disruptions import DisruptionEngine
from server.engine.financial import FinancialEngine
from server.engine.rewards import RewardCalculator, StepState
from server.engine.monte_carlo import MonteCarloEngine


class SimulationEngine:
    """
    Core simulation engine that orchestrates all sub-engines through the
    step loop.

    Each call to step() advances the simulation by one day:
    1. Validate and apply the agent's action to the graph
    2. Process action cost via financial engine
    3. Advance the day counter
    4. Advance disruption lifecycles and apply effects to the graph
    5. Update commodity prices, deplete inventory, update customer delays
    6. Calculate revenue loss, SLA penalties, and backup premiums
    7. Run Monte Carlo projection
    8. Compute dense reward
    9. Build and return the observation

    Attributes:
        graph: The supply chain graph model.
        financial: Financial state tracker (budget, costs, losses).
        disruption_engine: Manages disruption scenario lifecycles.
        reward_calculator: Computes 7-component dense reward per step.
        monte_carlo: Probabilistic loss estimator.
    """

    def __init__(
        self,
        graph_file: str,
        disruption_file: str,
        budget: float,
        max_steps: int,
        min_episode_days: int,
        seed: int = 42,
        jitter_enabled: bool = False,
    ) -> None:
        """
        Initialize the simulation engine for a new episode.

        Args:
            graph_file: Path to the supply chain graph JSON file.
            disruption_file: Path to the disruption scenarios JSON file.
            budget: Total budget available for mitigation actions (USD).
            max_steps: Maximum number of steps (days) in the episode.
            min_episode_days: Minimum days before the episode can end early.
            seed: RNG seed for Monte Carlo — derived per-episode for variance
                  while maintaining within-episode determinism.
            jitter_enabled: If True, apply seed-based jitter to disruption
                  scenarios for episode variation. Default False preserves
                  backward-compatible deterministic behavior.
        """
        # Load supply chain graph
        self.graph = SupplyChainGraph()
        self.graph.load_from_json(graph_file)

        # Load disruption scenarios
        self.disruption_engine = DisruptionEngine()
        self.disruption_engine.load_scenarios(disruption_file)

        # Apply scenario jitter if enabled (seed controls the variation)
        if jitter_enabled:
            self.disruption_engine.apply_jitter(seed, self.graph)

        # Create financial engine with starting budget
        self.financial = FinancialEngine(budget)

        # Create reward calculator with total annual revenue from graph
        total_revenue = self.graph.total_annual_revenue()
        self.reward_calculator = RewardCalculator(total_revenue, episode_length=max_steps)

        # Create Monte Carlo engine with per-episode seed for variance
        # while maintaining within-episode determinism for grading
        self.monte_carlo = MonteCarloEngine(seed=seed)

        # Episode parameters
        self.max_steps: int = max_steps
        self.min_episode_days: int = min_episode_days
        self.current_step: int = 0

        # Track state for reward computation
        self._prev_reward_state: Optional[StepState] = None
        self._last_action_result: Optional[ActionResult] = None
        self._last_mc_results: dict[str, float] = {}
        self._any_stockout_occurred: bool = False

        # Track consecutive offline days per node for emergent cascades
        self._offline_durations: dict[str, int] = {}
        self._injected_cascade_ids: set[str] = set()

    # ──────────────────────────────────────────────
    # Public Interface
    # ──────────────────────────────────────────────

    def get_initial_observation(self) -> SupplyMindObservation:
        """
        Build and return the initial observation for day 0 (before any action).

        Advances disruptions to day 0 so any pre-existing warning signals
        are visible, captures baseline reward state, and runs an initial
        Monte Carlo simulation.

        Returns:
            The initial SupplyMindObservation for the episode.
        """
        # Advance disruptions to day 0 (may produce warning signals)
        active_signals = self.disruption_engine.advance_day(self.current_step)
        new_signals = self.disruption_engine.get_new_signals()

        # Apply any day-0 disruption effects (typically just risk score bumps)
        self.disruption_engine.apply_to_graph(self.graph)

        # Run initial Monte Carlo
        self._last_mc_results = self.monte_carlo.run_quick_simulation(
            self.graph, active_signals
        )

        # Capture baseline reward state
        self._prev_reward_state = self.reward_calculator.capture_state(
            self.graph, self.financial, active_signals
        )

        # Build financial snapshot with MC data
        financials = self._build_financial_snapshot(active_signals)

        node_statuses = self.graph.get_node_statuses()

        return self._build_observation(
            active_signals=active_signals,
            new_signals=new_signals,
            financials=financials,
            action_result=None,
            reward=0.0,
            done=False,
            node_statuses=node_statuses,
        )

    def step(self, action: SupplyMindAction) -> SupplyMindObservation:
        """
        Execute one simulation step.

        This is the core loop that processes the agent's action and advances
        the world state by one day.

        Args:
            action: The action chosen by the agent for this step.

        Returns:
            The observation after the action and world update.
        """
        # ── 1. Validate action ──
        action = self._validate_action(action)

        # ── 2. Apply action to graph → get ActionResult ──
        action_result = self.graph.apply_action(action)

        # ── 3. Process action cost via financial engine ──
        if action_result.success and action.action_type not in (
            "do_nothing",
            "issue_supplier_alert",
        ):
            cost = self.financial.process_action_cost(action, self.graph)
            if cost == -1.0:
                # Budget insufficient -- action fails
                action_result = ActionResult(
                    success=False,
                    message=(
                        f"Insufficient budget for {action.action_type}. "
                        f"Budget remaining: ${self.financial.budget_remaining:,.0f}."
                    ),
                    cost=0.0,
                    effect_description="Action rejected due to budget constraints.",
                )
            else:
                action_result.cost = cost

        self._last_action_result = action_result

        # ── 4. Advance day ──
        self.current_step += 1

        # ── 5. Advance disruptions ──
        active_signals = self.disruption_engine.advance_day(self.current_step)
        new_signals = self.disruption_engine.get_new_signals()

        # ── 6. Apply disruptions to graph ──
        self.disruption_engine.apply_to_graph(self.graph)

        # ── 7. Update commodity prices from disruption effects ──
        commodity_effects = self.disruption_engine.get_commodity_effects()
        for commodity, multiplier in commodity_effects.items():
            self.financial.apply_commodity_price_change(commodity, multiplier)

        # ── 7b. Apply lead-time variance (±15% normal noise per step) ──
        self.graph.apply_lead_time_variance(self.monte_carlo._rng)

        # ── 8. Deplete inventory for disrupted suppliers ──
        disrupted_ids = self.disruption_engine.get_disrupted_node_ids()
        self.graph.deplete_inventory(disrupted_ids)

        # ── 8b. Check for emergent cascades (inventory buffer exhaustion) ──
        self._check_emergent_cascades()

        # ── 9. Update customer delays ──
        self.graph.update_customer_delays(disrupted_ids)

        # ── 10. Calculate daily revenue loss ──
        daily_loss = self.financial.calculate_daily_revenue_loss(self.graph)

        # ── 11. Calculate SLA penalties ──
        sla_penalties = self.financial.calculate_sla_penalties(self.graph)

        # ── 12. Apply daily backup premiums ──
        backup_premiums = self.financial.apply_daily_backup_premiums()

        # ── 13. Run quick Monte Carlo simulation ──
        self._last_mc_results = self.monte_carlo.run_quick_simulation(
            self.graph, active_signals
        )

        # ── 14. Capture reward state and compute step reward ──
        current_reward_state = self.reward_calculator.capture_state(
            self.graph, self.financial, active_signals
        )

        reward = self.reward_calculator.compute_step_reward(
            prev_state=self._prev_reward_state,
            current_state=current_reward_state,
            action=action,
            action_result=action_result,
        )

        self._prev_reward_state = current_reward_state

        # Track stockout occurrence
        self._check_stockout()

        # ── 15. Check if done ──
        done = self._check_done()

        # ── 16. Build and return observation ──
        financials = self._build_financial_snapshot(active_signals)
        node_statuses = self.graph.get_node_statuses()

        return self._build_observation(
            active_signals=active_signals,
            new_signals=new_signals,
            financials=financials,
            action_result=action_result,
            reward=reward,
            done=done,
            node_statuses=node_statuses,
        )

    # ──────────────────────────────────────────────
    # Grader-Accessible Methods
    # ──────────────────────────────────────────────

    def calculate_max_possible_loss(self) -> float:
        """
        Estimate the worst-case revenue loss if no mitigation actions are taken.

        Uses the actual cumulative loss as a floor (since the do-nothing agent
        experiences this), plus a margin. This ensures the grader's
        revenue-preservation score is meaningful.

        Returns:
            Estimated maximum loss in USD.
        """
        total_revenue = self.graph.total_annual_revenue()
        if total_revenue <= 0:
            return self.financial.budget_total

        # Find the total disruption window across all scenarios
        max_disruption_days = 0
        total_disruption_days = 0
        for scenario in self.disruption_engine.scenarios:
            duration = scenario.resolved_day - scenario.trigger_day
            max_disruption_days = max(max_disruption_days, duration)
            total_disruption_days += duration

        # Use the larger of: total disruption window or episode length
        effective_days = max(total_disruption_days, self.max_steps)

        # Daily revenue at risk (full revenue / 365) * effective disruption days
        # Use a higher multiplier to account for cascading effects
        max_loss = total_revenue * (effective_days / 365.0)

        # Add potential SLA penalties
        num_customers = len(self.graph.get_customer_ids())
        sla_penalty_estimate = num_customers * 10_000.0 * effective_days * 0.5

        # Ensure max_loss is at least as large as actual cumulative loss
        # so the score is always in [0, 1]
        actual_loss = (
            self.financial.cumulative_revenue_lost
            + self.financial.cumulative_penalty_fees
        )
        max_loss = max(max_loss + sla_penalty_estimate, actual_loss * 1.25)

        return max_loss

    def calculate_max_cascade_nodes(self) -> int:
        """
        Count the maximum number of nodes that could go offline.

        Returns the count of all non-customer nodes (suppliers, ports,
        factories) since customers don't go offline, only experience delays.

        Returns:
            Maximum cascade node count.
        """
        count = 0
        for _, ndata in self.graph.G.nodes(data=True):
            ntype = ndata.get("node_type", "").lower()
            if ntype in ("supplier", "port", "factory"):
                count += 1
        return max(1, count)

    def count_nodes_that_went_offline(self) -> int:
        """
        Count the number of nodes that went offline at any point during
        the episode.

        Returns:
            Number of nodes that were ever non-operational.
        """
        return self.graph.count_ever_offline()

    def any_customer_experienced_stockout(self) -> bool:
        """
        Check if any warehouse serving customers hit zero inventory
        during the episode.

        Returns:
            True if any stockout occurred, False otherwise.
        """
        return self._any_stockout_occurred

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    def _validate_action(self, action: SupplyMindAction) -> SupplyMindAction:
        """
        Validate and sanitize an incoming action.

        Ensures the action type is recognized and required parameters are
        present. Returns the action unchanged if valid, or converts it to
        a do_nothing action if invalid.

        Args:
            action: The raw action from the agent.

        Returns:
            A validated SupplyMindAction.
        """
        valid_types = {
            "do_nothing",
            "activate_backup_supplier",
            "reroute_shipment",
            "increase_safety_stock",
            "expedite_order",
            "hedge_commodity",
            "issue_supplier_alert",
        }

        if action.action_type not in valid_types:
            return SupplyMindAction(action_type="do_nothing")

        # Validate that target_node_id exists when required
        needs_target = {
            "activate_backup_supplier",
            "reroute_shipment",
            "increase_safety_stock",
            "expedite_order",
            "issue_supplier_alert",
        }
        if action.action_type in needs_target and not action.target_node_id:
            return SupplyMindAction(action_type="do_nothing")

        # Validate that target_node_id actually exists in the graph
        if (
            action.target_node_id
            and action.action_type in needs_target
            and action.target_node_id not in self.graph.G
        ):
            # Demote to do_nothing but graph.apply_action will report the
            # unknown node_id in the ActionResult message for the agent to see.
            pass  # Let apply_action handle it and return a clear error message

        return action

    def _check_done(self) -> bool:
        """
        Determine if the episode should end.

        The episode ends when:
        - current_step >= max_steps (hard limit), OR
        - All disruptions are resolved AND current_step >= min_episode_days
          (early termination if the crisis is fully over)

        Returns:
            True if the episode is done.
        """
        if self.current_step >= self.max_steps:
            return True

        if (
            self.current_step >= self.min_episode_days
            and self.disruption_engine.all_resolved()
        ):
            return True

        return False

    def _check_stockout(self) -> None:
        """Check all warehouses for zero inventory and record stockout."""
        if self._any_stockout_occurred:
            return  # Already recorded

        for nid, ndata in self.graph.G.nodes(data=True):
            if ndata.get("node_type", "").lower() != "warehouse":
                continue
            inv = ndata.get("current_inventory_units", 0)
            if inv <= 0:
                self._any_stockout_occurred = True
                return

    def _check_emergent_cascades(self) -> None:
        """
        Check for inventory buffer exhaustion and inject emergent cascading
        disruptions on downstream nodes.

        When a supplier/port stays offline longer than the downstream
        warehouse's inventory buffer, a supply shortage cascade is triggered
        on further downstream nodes. This creates emergent behavior on top
        of the pre-scripted disruption scenarios.
        """
        from server.engine.disruptions import DisruptionScenario

        # Update offline durations
        for nid, ndata in self.graph.G.nodes(data=True):
            ntype = ndata.get("node_type", "").lower()
            if ntype in ("supplier", "port", "factory"):
                if not ndata.get("is_operational", True):
                    self._offline_durations[nid] = self._offline_durations.get(nid, 0) + 1
                else:
                    self._offline_durations[nid] = 0

        # Check if any offline node has exhausted downstream inventory buffer
        for nid, days_offline in self._offline_durations.items():
            if days_offline < 3:  # Need at least 3 days offline to cascade
                continue

            for _, downstream in self.graph.G.out_edges(nid):
                down_data = self.graph.G.nodes[downstream]
                if down_data.get("node_type", "").lower() != "warehouse":
                    continue

                inv_cover = down_data.get("inventory_days_cover", 30.0)
                # Cascade triggers when offline duration exceeds inventory buffer
                # AND inventory is critically low
                if days_offline > inv_cover and inv_cover < 3:
                    self._inject_cascade(nid, downstream, days_offline)

    def _inject_cascade(
        self, source_id: str, warehouse_id: str, days_offline: int
    ) -> None:
        """
        Inject an emergent supply shortage cascade downstream of an exhausted
        warehouse.

        Args:
            source_id: The offline supplier/port causing the cascade.
            warehouse_id: The warehouse whose inventory buffer was exhausted.
            days_offline: How many consecutive days the source has been offline.
        """
        from server.engine.disruptions import DisruptionScenario

        cascade_id = f"CASCADE_{source_id}_{warehouse_id}"
        if cascade_id in self._injected_cascade_ids:
            return  # Already injected this cascade

        # Find downstream nodes from the warehouse
        downstream_nodes = [n for _, n in self.graph.G.out_edges(warehouse_id)]
        if not downstream_nodes:
            return

        # Calculate cascade severity proportional to dependency
        total_inbound_qty = sum(
            self.graph.G.edges[src, warehouse_id].get("quantity", 100)
            for src, _ in self.graph.G.in_edges(warehouse_id)
        )
        source_qty = self.graph.G.edges.get(
            (source_id, warehouse_id), {}
        ).get("quantity", 100)
        dependency_ratio = source_qty / max(1, total_inbound_qty)
        cascade_severity = min(0.6, 0.3 + dependency_ratio * 0.3)

        source_name = self.graph.G.nodes[source_id].get("name", source_id)
        wh_name = self.graph.G.nodes[warehouse_id].get("name", warehouse_id)

        cascade_data = {
            "signal_id": cascade_id,
            "disruption_type": "supply_shortage",
            "trigger_day": self.current_step,
            "warning_severity": cascade_severity * 0.5,
            "warning_confidence": 0.9,
            "peak_severity": cascade_severity,
            "impact_day": self.current_step + 1,
            "recovery_start_day": self.current_step + 5,
            "resolved_day": self.current_step + 10,
            "affected_region": "Cascading",
            "affected_node_ids": downstream_nodes,
            "estimated_duration_days": 10,
            "description": (
                f"Supply shortage cascade: {source_name} offline for {days_offline} "
                f"days exhausted inventory buffer at {wh_name}. "
                f"Downstream nodes experiencing supply disruption."
            ),
        }

        self.disruption_engine.scenarios.append(DisruptionScenario(cascade_data))
        self._injected_cascade_ids.add(cascade_id)

    def _build_financial_snapshot(
        self, active_signals: list[DisruptionSignal]
    ) -> FinancialSnapshot:
        """
        Build a FinancialSnapshot enriched with Monte Carlo projections.

        Args:
            active_signals: Currently active disruption signals.

        Returns:
            Complete FinancialSnapshot with MC P50/P95 projections.
        """
        snapshot = self.financial.get_snapshot(self.graph)

        # Enrich with Monte Carlo projections
        snapshot.monte_carlo_p50_loss = self._last_mc_results.get("p50_loss", 0.0)
        snapshot.monte_carlo_p95_loss = self._last_mc_results.get("p95_loss", 0.0)

        return snapshot

    def _build_observation(
        self,
        active_signals: list[DisruptionSignal],
        new_signals: list[DisruptionSignal],
        financials: FinancialSnapshot,
        action_result: Optional[ActionResult],
        reward: float,
        done: bool,
        node_statuses: Optional[list] = None,
    ) -> SupplyMindObservation:
        """
        Assemble a complete SupplyMindObservation from current state.

        Args:
            active_signals: All currently active disruption signals.
            new_signals: Signals that appeared this step only.
            financials: Current financial snapshot.
            action_result: Result of the agent's last action (None for day 0).
            reward: Reward for this step.
            done: Whether the episode is over.
            node_statuses: Pre-computed node statuses (avoids double computation).

        Returns:
            A fully populated SupplyMindObservation.
        """
        if node_statuses is None:
            node_statuses = self.graph.get_node_statuses()

        situation_summary = self._generate_situation_summary(
            active_signals=active_signals,
            new_signals=new_signals,
            financials=financials,
            node_statuses=node_statuses,
            action_result=action_result,
        )

        compact_summary = self._generate_compact_summary(
            active_signals, financials, node_statuses
        )

        info: dict = {
            "reward_components": self.reward_calculator.get_last_components(),
            "monte_carlo": self._last_mc_results,
        }

        return SupplyMindObservation(
            current_day=self.current_step,
            days_remaining=max(0, self.max_steps - self.current_step),
            active_signals=active_signals,
            new_signals=new_signals,
            node_statuses=node_statuses,
            financials=financials,
            last_action_result=action_result,
            situation_summary=situation_summary,
            compact_summary=compact_summary,
            reward=reward,
            done=done,
            info=info,
        )

    def _generate_compact_summary(
        self,
        active_signals: list[DisruptionSignal],
        financials: FinancialSnapshot,
        node_statuses: list,
    ) -> str:
        """
        Generate a compact summary (≤500 tokens) for token-constrained LLM agents.

        Includes only the most critical information: day/budget, disruption
        count, top 3 at-risk nodes, and the single most urgent action.
        """
        parts: list[str] = []

        # Day and budget
        days_remaining = max(0, self.max_steps - self.current_step)
        budget_pct = (
            financials.budget_remaining / financials.budget_total * 100
            if financials.budget_total > 0 else 0
        )
        parts.append(
            f"Day {self.current_step}/{self.max_steps} ({days_remaining} left) | "
            f"Budget: ${financials.budget_remaining:,.0f} ({budget_pct:.0f}%) | "
            f"Health: {financials.supply_chain_health_score:.0f}/100"
        )

        # Disruption summary
        if active_signals:
            max_sev = max(s.severity for s in active_signals)
            warning_count = sum(1 for s in active_signals if s.lifecycle_phase == "warning")
            active_count = sum(1 for s in active_signals if s.lifecycle_phase == "active")
            parts.append(
                f"Disruptions: {len(active_signals)} total "
                f"({warning_count} warning, {active_count} active, "
                f"max severity {max_sev:.0%})"
            )
        else:
            parts.append("No active disruptions.")

        # Top 3 at-risk nodes (sorted by risk)
        at_risk = sorted(
            [n for n in node_statuses if n.current_risk_score > 0.1],
            key=lambda n: n.current_risk_score,
            reverse=True,
        )[:3]
        if at_risk:
            risk_strs = [
                f"{n.node_id}({n.current_risk_score:.0%}"
                f"{', OFFLINE' if not n.is_operational else ''}"
                f"{', backup:' + n.backup_supplier_ids[0] if n.backup_supplier_ids else ''}"
                f")"
                for n in at_risk
            ]
            parts.append(f"Top risks: {', '.join(risk_strs)}")

        # Most urgent suggested action
        offline_with_backup = [
            n for n in node_statuses
            if not n.is_operational and n.backup_supplier_ids
        ]
        low_inv = [
            n for n in node_statuses
            if n.node_type == "warehouse" and n.inventory_days_cover < 5
        ]
        warning_sigs = [s for s in active_signals if s.lifecycle_phase == "warning"]

        if offline_with_backup:
            n = offline_with_backup[0]
            parts.append(
                f"URGENT: Activate backup {n.backup_supplier_ids[0]} for offline {n.node_id}"
            )
        elif low_inv:
            n = low_inv[0]
            parts.append(
                f"URGENT: Increase stock at {n.node_id} ({n.inventory_days_cover:.0f}d remaining)"
            )
        elif warning_sigs:
            sig = warning_sigs[0]
            parts.append(
                f"PREPARE: {sig.disruption_type} impact in {sig.time_to_impact_hours:.0f}h "
                f"on {', '.join(sig.affected_node_ids[:2])}"
            )

        # Commodity alerts (only significant spikes)
        spikes = {
            k: v for k, v in financials.commodity_price_changes.items() if v >= 1.3
        }
        if spikes:
            spike_str = ", ".join(f"{k} +{(v-1)*100:.0f}%" for k, v in spikes.items())
            parts.append(f"Commodities spiking: {spike_str}")

        return " | ".join(parts)

    def _generate_situation_summary(
        self,
        active_signals: list[DisruptionSignal],
        new_signals: list[DisruptionSignal],
        financials: FinancialSnapshot,
        node_statuses: list,
        action_result: Optional[ActionResult],
    ) -> str:
        """
        Generate a rich natural language summary of the current situation
        for LLM-based agents.

        Includes current day, active disruptions, key metrics, recent
        action results, Monte Carlo projections, and actionable insights.

        Args:
            active_signals: Currently active disruption signals.
            new_signals: New signals this step.
            financials: Current financial snapshot.
            node_statuses: Current node statuses.
            action_result: Result of last action (may be None).

        Returns:
            Multi-paragraph situation summary string.
        """
        lines: list[str] = []

        # ── Header ──
        days_remaining = max(0, self.max_steps - self.current_step)
        lines.append(
            f"=== DAY {self.current_step} of {self.max_steps} "
            f"({days_remaining} days remaining) ==="
        )
        lines.append("")

        # ── New signals alert ──
        if new_signals:
            lines.append("** NEW DISRUPTION SIGNALS **")
            for sig in new_signals:
                lines.append(
                    f"  - [{sig.lifecycle_phase.upper()}] {sig.disruption_type}: "
                    f"{sig.description} "
                    f"(Severity: {sig.severity:.0%}, "
                    f"Confidence: {sig.confidence:.0%})"
                )
                if sig.time_to_impact_hours > 0:
                    lines.append(
                        f"    Time to impact: {sig.time_to_impact_hours:.0f} hours"
                    )
            lines.append("")

        # ── Active disruptions ──
        if active_signals:
            lines.append(f"ACTIVE DISRUPTIONS ({len(active_signals)}):")
            for sig in active_signals:
                affected_count = len(sig.affected_node_ids)
                lines.append(
                    f"  - {sig.signal_id} [{sig.lifecycle_phase.upper()}]: "
                    f"{sig.disruption_type} in {sig.affected_region} "
                    f"(Severity: {sig.severity:.0%}, "
                    f"{affected_count} nodes affected)"
                )
            lines.append("")
        else:
            lines.append("No active disruptions.")
            lines.append("")

        # ── Key metrics ──
        lines.append("KEY METRICS:")
        lines.append(
            f"  Revenue at risk: ${financials.total_revenue_at_risk:,.0f}"
        )
        lines.append(
            f"  Budget remaining: ${financials.budget_remaining:,.0f} "
            f"of ${financials.budget_total:,.0f} "
            f"({financials.budget_remaining / financials.budget_total * 100:.0f}% remaining)"
            if financials.budget_total > 0
            else f"  Budget remaining: ${financials.budget_remaining:,.0f}"
        )
        lines.append(
            f"  Cumulative revenue lost: ${financials.cumulative_revenue_lost:,.0f}"
        )
        lines.append(
            f"  Cumulative costs incurred: ${financials.cumulative_cost_incurred:,.0f}"
        )
        if financials.cumulative_penalty_fees > 0:
            lines.append(
                f"  SLA penalty fees: ${financials.cumulative_penalty_fees:,.0f}"
            )
        lines.append(
            f"  Supply chain health: {financials.supply_chain_health_score:.1f}/100"
        )
        lines.append("")

        # ── Commodity prices ──
        if financials.commodity_price_changes:
            lines.append("COMMODITY PRICE CHANGES:")
            for commodity, change in financials.commodity_price_changes.items():
                pct = (change - 1.0) * 100
                direction = "UP" if pct > 0 else "DOWN"
                lines.append(
                    f"  - {commodity}: {direction} {abs(pct):.1f}% "
                    f"(multiplier: {change:.2f}x)"
                )
            lines.append("")

        # ── Node status summary ──
        offline_nodes = [n for n in node_statuses if not n.is_operational]
        low_inventory = [
            n
            for n in node_statuses
            if n.node_type == "warehouse" and n.inventory_days_cover < 7
        ]
        high_risk = [
            n for n in node_statuses if n.current_risk_score >= 0.5
        ]

        if offline_nodes or low_inventory or high_risk:
            lines.append("CRITICAL NODES:")
            for node in offline_nodes:
                lines.append(
                    f"  [OFFLINE] {node.name} ({node.node_type}, "
                    f"{node.country})"
                )
                if node.backup_supplier_ids:
                    lines.append(
                        f"    Backups available: "
                        f"{', '.join(node.backup_supplier_ids)}"
                    )
            for node in low_inventory:
                lines.append(
                    f"  [LOW INVENTORY] {node.name}: "
                    f"{node.inventory_days_cover:.1f} days remaining"
                )
            for node in high_risk:
                if node.is_operational:
                    lines.append(
                        f"  [HIGH RISK] {node.name}: "
                        f"risk score {node.current_risk_score:.2f}"
                    )
            lines.append("")

        # ── Last action result ──
        if action_result is not None:
            status = "SUCCESS" if action_result.success else "FAILED"
            lines.append(f"LAST ACTION [{status}]: {action_result.message}")
            if action_result.effect_description:
                lines.append(f"  Effect: {action_result.effect_description}")
            if action_result.cost > 0:
                lines.append(f"  Cost: ${action_result.cost:,.0f}")
            lines.append("")

        # ── Monte Carlo projections ──
        mc = self._last_mc_results
        if mc and mc.get("p50_loss", 0) > 0:
            lines.append("RISK PROJECTIONS (Monte Carlo):")
            lines.append(f"  P50 projected loss: ${mc.get('p50_loss', 0):,.0f}")
            lines.append(f"  P95 projected loss: ${mc.get('p95_loss', 0):,.0f}")
            avg_nodes = mc.get("avg_nodes_affected", 0)
            if avg_nodes > 0:
                lines.append(
                    f"  Avg nodes affected: {avg_nodes:.1f}"
                )
            max_delay = mc.get("max_delay_days", 0)
            if max_delay > 0:
                lines.append(
                    f"  P95 max delay: {max_delay:.1f} days"
                )
            lines.append("")

        # ── Actionable insights ──
        insights = self._generate_insights(
            active_signals, financials, node_statuses, offline_nodes, low_inventory
        )
        if insights:
            lines.append("RECOMMENDED ACTIONS:")
            for insight in insights:
                lines.append(f"  -> {insight}")
            lines.append("")

        return "\n".join(lines)

    def _generate_insights(
        self,
        active_signals: list[DisruptionSignal],
        financials: FinancialSnapshot,
        node_statuses: list,
        offline_nodes: list,
        low_inventory: list,
    ) -> list[str]:
        """
        Generate actionable insights based on current state.

        Returns a list of insight strings that suggest specific actions
        the agent should consider.
        """
        insights: list[str] = []

        # Warning phase signals -- suggest proactive action
        warning_signals = [
            s for s in active_signals if s.lifecycle_phase == "warning"
        ]
        for sig in warning_signals:
            hours = sig.time_to_impact_hours
            insights.append(
                f"PROACTIVE: {sig.disruption_type} impact in ~{hours:.0f}h. "
                f"Consider activating backups or increasing safety stock for "
                f"affected nodes: {', '.join(sig.affected_node_ids[:3])}"
            )

        # Offline nodes with backups available
        for node in offline_nodes:
            if node.backup_supplier_ids:
                insights.append(
                    f"Activate backup supplier for offline node {node.name}. "
                    f"Available backups: {', '.join(node.backup_supplier_ids)}"
                )

        # Low inventory warehouses
        for node in low_inventory:
            if node.inventory_days_cover <= 0:
                insights.append(
                    f"CRITICAL: {node.name} stockout! "
                    f"Expedite orders immediately."
                )
            elif node.inventory_days_cover < 3:
                insights.append(
                    f"URGENT: {node.name} has only "
                    f"{node.inventory_days_cover:.1f} days of inventory. "
                    f"Increase safety stock."
                )

        # Commodity price spikes -- suggest hedging
        for commodity, multiplier in financials.commodity_price_changes.items():
            if multiplier >= 1.2:
                pct = (multiplier - 1.0) * 100
                insights.append(
                    f"Hedge {commodity} -- price up {pct:.0f}%. "
                    f"Consider commodity hedge to protect margins."
                )

        # Budget warning
        budget_pct = (
            financials.budget_remaining / financials.budget_total
            if financials.budget_total > 0
            else 0
        )
        if budget_pct < 0.15:
            insights.append(
                f"Budget critically low ({budget_pct:.0%} remaining). "
                f"Prioritize only highest-impact actions."
            )
        elif budget_pct < 0.30:
            insights.append(
                f"Budget running low ({budget_pct:.0%} remaining). "
                f"Be selective with mitigation spending."
            )

        # If no disruptions and nothing to do
        if not active_signals and not offline_nodes and not low_inventory:
            insights.append(
                "No active threats. Consider issuing supplier alerts for "
                "situational awareness or doing nothing to conserve budget."
            )

        return insights
