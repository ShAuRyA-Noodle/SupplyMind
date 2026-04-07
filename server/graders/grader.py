"""
SupplyMind Episode Graders

Deterministic, multi-component graders that score completed episodes on
a 0.0-1.0 scale. Each difficulty level has its own grading function with
different component weights.

CRITICAL INVARIANTS:
- Deterministic: same episode history always produces the same score
- Discriminating: different strategies MUST produce different scores
- Do-nothing agent scores ~0.15-0.35 (some baseline revenue is naturally preserved)
- Optimal agent scores ~0.85-0.95 (perfection is unrealistic)
"""

from __future__ import annotations

from typing import Any


class EpisodeGrader:
    """
    Grades a completed SupplyMind episode based on task-specific criteria.

    The grader examines the full episode history (list of (action, observation)
    tuples) and the engine's final state to produce a weighted composite score.

    Usage:
        grader = EpisodeGrader("easy_typhoon_response")
        score = grader.grade(episode_history, engine)
        breakdown = grader.get_breakdown()
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.breakdown: dict[str, dict[str, float]] = {}

    def grade(self, episode_history: list[tuple[Any, Any]], engine: Any) -> float:
        """
        Grade a completed episode.

        Args:
            episode_history: List of (SupplyMindAction, SupplyMindObservation) tuples
                for every step in the episode.
            engine: The SimulationEngine instance with final state accessible via
                engine.financial, engine.graph, etc.

        Returns:
            Score between 0.0 and 1.0.

        Raises:
            ValueError: If the task_id is unknown.
        """
        # Guard: empty history means no steps were taken — score 0.0
        if not episode_history:
            self.breakdown = {"no_steps": {"score": 0.0, "weight": 1.0}}
            return 0.0

        if self.task_id == "easy_typhoon_response":
            return self._grade_easy(episode_history, engine)
        elif self.task_id == "medium_multi_front":
            return self._grade_medium(episode_history, engine)
        elif self.task_id == "hard_cascading_crisis":
            return self._grade_hard(episode_history, engine)
        raise ValueError(
            f"Unknown task_id for grading: '{self.task_id}'. "
            f"Expected one of: easy_typhoon_response, medium_multi_front, hard_cascading_crisis"
        )

    def get_breakdown(self) -> dict[str, dict[str, float]]:
        """
        Return the scoring breakdown from the last grade() call.

        Returns:
            Dict mapping component name to {"score": float, "weight": float}.
        """
        return self.breakdown

    # ------------------------------------------------------------------
    # Easy: Typhoon Response
    # Revenue preserved (30%) + timeliness (25%) + action coverage (20%)
    # + cost efficiency (15%) + stockout prevention (10%)
    #
    # Weight rationale:
    #   Revenue (30%): Primary business objective — protecting revenue is
    #     the whole point of supply chain risk management.
    #   Timeliness (25%): The easy task's core lesson is proactive response
    #     to early warning signals. High weight rewards acting early.
    #   Action coverage (20%): Ensures do-nothing agents score low (~0.1-0.2).
    #     Agents must take meaningful cost-bearing mitigation actions.
    #   Cost efficiency (15%): Single-disruption task has ample budget, so
    #     cost discipline matters but is secondary to correct action.
    #   Stockout prevention (10%): Binary outcome (stockout or not) makes
    #     this lower-weight; partial credit via timing offset.
    # ------------------------------------------------------------------

    def _grade_easy(self, history: list[tuple[Any, Any]], engine: Any) -> float:
        """
        Grade the easy typhoon response task.

        Components:
        - revenue_preserved (30%): How much revenue was saved vs. do-nothing
        - timeliness (25%): Did the agent act before or after the disruption hit?
        - cost_efficiency (15%): Was the budget used wisely (sweet spot 10-30%)?
        - stockout_prevention (10%): Did any customer experience stockout?
        - action_coverage (20%): Did the agent take meaningful mitigation actions?
          (critical: prevents do-nothing from scoring > 0.2)
        """
        import math

        # Component 1: Revenue Preserved (30%)
        max_possible_loss = self._get_max_possible_loss(engine)
        actual_loss = engine.financial.cumulative_revenue_lost
        if max_possible_loss > 0:
            revenue_preserved = 1.0 - (actual_loss / max_possible_loss)
        else:
            revenue_preserved = 1.0
        revenue_score = _clamp(revenue_preserved)

        # Component 2: Timeliness (25%)
        # Did the agent act BEFORE impact day (day 5 in easy scenario)?
        # The typhoon warning comes on day 2, impact on day 5.
        first_meaningful_action_day = self._find_first_meaningful_action_day(history)
        if first_meaningful_action_day is None:
            # Agent never took a meaningful action
            timeliness_score = 0.0
        elif first_meaningful_action_day <= 3:
            # Acted during warning phase (days 2-3) -- excellent
            timeliness_score = 1.0
        elif first_meaningful_action_day <= 5:
            # Acted right at impact -- decent but not proactive
            timeliness_score = 0.6
        elif first_meaningful_action_day <= 8:
            # Acted during active disruption -- reactive
            timeliness_score = 0.3
        else:
            # Very late action
            timeliness_score = 0.1

        # Component 3: Cost Efficiency (15%)
        # Smooth Gaussian curve centered on ideal spend ratio (0.20 for easy).
        total_cost = engine.financial.cumulative_cost_incurred
        budget = engine.financial.budget_total
        cost_ratio = total_cost / budget if budget > 0 else 0.0

        ideal_ratio = 0.20  # Easy task: ~20% of budget is efficient
        sigma = 0.20
        # Special case: near-zero spend means agent did nothing useful
        if cost_ratio < 0.02:
            cost_score = 0.1
        else:
            cost_score = max(0.1, math.exp(-0.5 * ((cost_ratio - ideal_ratio) / sigma) ** 2))

        # Component 4: Stockout Prevention (10%)
        stockout_occurred = self._check_any_stockout(history, engine)
        if not stockout_occurred:
            stockout_score = 1.0
        else:
            stockout_day = self._find_first_stockout_day(history)
            total_days = len(history) if len(history) > 0 else 1
            stockout_score = max(0.0, min(0.4, (stockout_day / total_days) * 0.4))

        # Component 5: Action Coverage (20%)
        # Measures whether the agent took meaningful mitigation actions.
        # A do-nothing agent scores 0.0. An agent that takes 3+ targeted
        # cost-bearing actions during disruption scores 1.0.
        meaningful_actions = sum(
            1 for a, _ in history
            if a.action_type not in ("do_nothing", "issue_supplier_alert")
        )
        if meaningful_actions == 0:
            action_coverage_score = 0.0
        elif meaningful_actions == 1:
            action_coverage_score = 0.4
        elif meaningful_actions == 2:
            action_coverage_score = 0.7
        else:
            action_coverage_score = 1.0

        # Assemble breakdown
        self.breakdown = {
            "revenue_preserved": {"score": round(revenue_score, 4), "weight": 0.30},
            "timeliness": {"score": round(timeliness_score, 4), "weight": 0.25},
            "cost_efficiency": {"score": round(cost_score, 4), "weight": 0.15},
            "stockout_prevention": {"score": round(stockout_score, 4), "weight": 0.10},
            "action_coverage": {"score": round(action_coverage_score, 4), "weight": 0.20},
        }

        final = sum(v["score"] * v["weight"] for v in self.breakdown.values())
        return round(_clamp(final), 4)

    # ------------------------------------------------------------------
    # Medium: Multi-Front Crisis
    # Financial impact (30%) + triage quality (25%) + budget utilization (20%)
    # + SLA compliance (15%) + proactive score (10%)
    #
    # Weight rationale:
    #   Financial impact (30%): Still the top objective, but reduced from
    #     40% because triage skill matters more with 3 concurrent crises.
    #   Triage quality (25%): The medium task's core lesson — budget covers
    #     ~2 of 3 disruptions, so prioritization is critical.
    #   Budget utilization (20%): Tight budget means overspending on one
    #     crisis starves others. Efficient allocation is rewarded.
    #   SLA compliance (15%): Customer delivery commitments matter more
    #     here because multiple supply paths are disrupted simultaneously.
    #   Proactive score (10%): Lower weight than easy because the agent
    #     has less warning time and more to juggle.
    # ------------------------------------------------------------------

    def _grade_medium(self, history: list[tuple[Any, Any]], engine: Any) -> float:
        """
        Grade the medium multi-front crisis task.

        Components:
        - financial_impact (30%): Revenue loss + penalties minimized
        - triage_quality (25%): Were highest-impact disruptions addressed first?
        - budget_utilization (20%): Budget spent in the efficient range
        - sla_compliance (15%): Fraction of customers within SLA
        - proactive_score (10%): Actions taken before disruptions escalate
        """
        # Component 1: Financial Impact (30%)
        max_loss = self._get_max_possible_loss(engine)
        actual_loss = (
            engine.financial.cumulative_revenue_lost
            + engine.financial.cumulative_penalty_fees
        )
        if max_loss > 0:
            financial_score = 1.0 - (actual_loss / max_loss)
        else:
            financial_score = 1.0
        financial_score = _clamp(financial_score)

        # Component 2: Triage Quality (25%)
        # Evaluate whether the agent addressed disruptions in priority order.
        # In the medium scenario:
        #   - Port strike (Day 7): highest immediate revenue impact
        #   - Thailand flood (Day 9): moderate impact, Tier 2 suppliers
        #   - Sanctions (Day 18): slower onset, can be hedged
        action_targets = []
        action_days = []
        for action, obs in history:
            if (
                action.action_type not in ("do_nothing", "issue_supplier_alert")
                and action.target_node_id is not None
            ):
                action_targets.append(action.target_node_id)
                action_days.append(obs.current_day)

        triage_score = self._evaluate_triage_order(action_targets, action_days, engine)

        # Component 3: Budget Utilization (20%)
        spent = engine.financial.cumulative_cost_incurred
        budget = engine.financial.budget_total
        utilization = spent / budget if budget > 0 else 0.0

        # Ideal: 20-60% utilization for medium difficulty
        if utilization < 0.05:
            budget_score = 0.15  # Did almost nothing
        elif 0.20 <= utilization <= 0.60:
            budget_score = 1.0  # Sweet spot
        elif utilization < 0.20:
            # Under-spent: linear interpolation from 0.15 to 1.0
            budget_score = 0.15 + (utilization / 0.20) * 0.85
        elif utilization <= 0.80:
            # Moderate overspend
            budget_score = 1.0 - ((utilization - 0.60) / 0.20) * 0.4
        else:
            # Heavy overspend
            budget_score = max(0.1, 0.6 - (utilization - 0.80))

        budget_score = _clamp(budget_score)

        # Component 4: SLA Compliance (15%)
        sla_score = _clamp(engine.graph.get_sla_compliance())

        # Component 5: Proactive Score (10%)
        # How many cost-bearing actions were taken before Day 7 (first disruption)?
        # Port strike starts Day 7, so acting before shows foresight.
        # Excludes free actions (do_nothing, issue_supplier_alert) to prevent gaming.
        early_actions = sum(
            1
            for action, obs in history
            if action.action_type not in ("do_nothing", "issue_supplier_alert")
            and obs.current_day < 7
        )
        # 3 early cost-bearing actions is excellent
        proactive_score = _clamp(min(1.0, early_actions / 3.0))

        # Assemble breakdown
        self.breakdown = {
            "financial_impact": {"score": round(financial_score, 4), "weight": 0.30},
            "triage_quality": {"score": round(triage_score, 4), "weight": 0.25},
            "budget_utilization": {"score": round(budget_score, 4), "weight": 0.20},
            "sla_compliance": {"score": round(sla_score, 4), "weight": 0.15},
            "proactive_score": {"score": round(proactive_score, 4), "weight": 0.10},
        }

        final = sum(v["score"] * v["weight"] for v in self.breakdown.values())
        return round(_clamp(final), 4)

    # ------------------------------------------------------------------
    # Hard: Cascading Crisis
    # Loss minimized (20%) + active mitigation (20%) + cascade containment (15%)
    # + budget ROI (15%) + information efficiency (10%) + resilience (10%)
    # + customer impact (10%)
    #
    # Weight rationale:
    #   Loss minimized (20%): Lower than easy/medium because perfect loss
    #     prevention is impossible in a cascading crisis.
    #   Active mitigation (20%): Ensures do-nothing agents score low.
    #     Hard task requires 8+ cost-bearing actions across multiple fronts.
    #   Cascade containment (15%): The hard task's defining mechanic.
    #     Preventing secondary failures is the key skill being tested.
    #   Budget ROI (15%): Very tight budget ($10M vs $2B+ exposure) means
    #     every dollar must count. Rewards smart allocation over brute force.
    #   Information efficiency (10%): Scouting (supplier alerts) before
    #     committing budget is valuable but secondary to action.
    #   Resilience (10%): End-state network health measures whether the
    #     agent preserved long-term supply chain viability.
    #   Customer impact (10%): Some customer impact is unavoidable in a
    #     cascading crisis; lower weight avoids penalizing good-but-imperfect
    #     strategies.
    # ------------------------------------------------------------------

    def _grade_hard(self, history: list[tuple[Any, Any]], engine: Any) -> float:
        """
        Grade the hard cascading crisis task.

        Designed so that even a well-executed GPT-4o strategy lands at
        ~0.60-0.70. A perfect score requires suppressing ALL cascade stages,
        maintaining 90%+ health, AND spending budget with surgical precision.

        Components:
        - loss_minimized (15%): Total financial losses vs. worst case
        - cascade_containment (20%): Strict — penalizes ANY node going offline
        - information_efficiency (10%): Quality of information gathering (alerts)
        - budget_roi (15%): Return on investment for mitigation spending
        - resilience (10%): Final network health score (raised bar to 90%)
        - customer_impact (10%): SLA compliance across all customers
        - active_mitigation (10%): Cost-bearing actions taken
        - cascade_stage_suppression (10%): Did agent prevent each cascade stage?
        """
        # Component 1: Loss Minimized (15%)
        # Use stricter scoring: quadratic penalty for losses
        max_loss = self._get_max_possible_loss(engine)
        actual_total = (
            engine.financial.cumulative_revenue_lost
            + engine.financial.cumulative_cost_incurred
            + engine.financial.cumulative_penalty_fees
        )
        if max_loss > 0:
            loss_ratio = actual_total / max_loss
            # Quadratic: small losses are fine, large losses punished hard
            loss_score = _clamp(1.0 - loss_ratio ** 0.7)
        else:
            loss_score = 1.0

        # Component 2: Cascade Containment (20%) — STRICT
        # Every node that goes offline costs the score heavily.
        # In a 40-node hard graph, even 5 nodes offline = severe penalty.
        max_cascade = self._get_max_cascade_nodes(engine)
        actual_cascade = self._count_nodes_went_offline(engine)
        if max_cascade > 0:
            # Stricter curve: losing even 20% of nodes = score near 0
            offline_fraction = actual_cascade / max_cascade
            cascade_score = _clamp(max(0.0, 1.0 - (offline_fraction * 3.0)))
        else:
            cascade_score = 1.0

        # Component 3: Information Efficiency (10%)
        total_alerts = sum(
            1 for a, _ in history if a.action_type == "issue_supplier_alert"
        )
        total_non_idle = sum(
            1 for a, _ in history if a.action_type != "do_nothing"
        )

        if total_non_idle == 0:
            info_score = 0.0
        else:
            alert_ratio = total_alerts / total_non_idle
            if 0.15 <= alert_ratio <= 0.45:
                info_score = 1.0
            elif alert_ratio < 0.15:
                info_score = max(0.1, alert_ratio / 0.15)
            else:
                info_score = max(0.2, 1.0 - (alert_ratio - 0.45) / 0.55)
        info_score = _clamp(info_score)

        # Component 4: Budget ROI (15%)
        spent = engine.financial.cumulative_cost_incurred
        losses_with_actions = (
            engine.financial.cumulative_revenue_lost
            + engine.financial.cumulative_penalty_fees
        )
        if spent <= 0:
            roi_score = 0.0
        else:
            saved = max(0.0, max_loss - losses_with_actions - spent)
            if saved > 0:
                roi = saved / spent
                # Require ROI of 15x+ for perfect score (very hard)
                roi_score = _clamp(min(1.0, roi / 15.0))
            else:
                roi_score = 0.1
        roi_score = _clamp(roi_score)

        # Component 5: Resilience (10%)
        # Raised bar: need 90+ health for full score, linear below
        final_health = engine.graph.get_health_score()
        if final_health >= 90.0:
            resilience_score = 1.0
        else:
            resilience_score = _clamp(final_health / 90.0)

        # Component 6: Customer Impact (10%)
        customer_score = _clamp(engine.graph.get_sla_compliance())

        # Component 7: Active Mitigation (10%)
        # Requires 12+ cost-bearing actions for full score (hard has 60 steps)
        cost_bearing_actions = sum(
            1 for a, _ in history
            if a.action_type not in ("do_nothing", "issue_supplier_alert")
        )
        if cost_bearing_actions == 0:
            mitigation_score = 0.0
        elif cost_bearing_actions <= 3:
            mitigation_score = 0.2
        elif cost_bearing_actions <= 6:
            mitigation_score = 0.4
        elif cost_bearing_actions <= 9:
            mitigation_score = 0.6
        elif cost_bearing_actions <= 12:
            mitigation_score = 0.8
        else:
            mitigation_score = 1.0

        # Component 8: Cascade Stage Suppression (10%)
        # The hard scenario has 5 cascade stages. Agent must address EACH stage.
        # Score is fraction of distinct disruption types that had mitigation.
        cascade_stage_score = self._evaluate_cascade_stage_coverage(history, engine)

        # Assemble breakdown
        self.breakdown = {
            "loss_minimized": {"score": round(loss_score, 4), "weight": 0.15},
            "cascade_containment": {"score": round(cascade_score, 4), "weight": 0.20},
            "information_efficiency": {"score": round(info_score, 4), "weight": 0.10},
            "budget_roi": {"score": round(roi_score, 4), "weight": 0.15},
            "resilience": {"score": round(resilience_score, 4), "weight": 0.10},
            "customer_impact": {"score": round(customer_score, 4), "weight": 0.10},
            "active_mitigation": {"score": round(mitigation_score, 4), "weight": 0.10},
            "cascade_stage_suppression": {"score": round(cascade_stage_score, 4), "weight": 0.10},
        }

        final = sum(v["score"] * v["weight"] for v in self.breakdown.values())
        return round(_clamp(final), 4)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_max_possible_loss(self, engine: Any) -> float:
        """
        Estimate the maximum possible revenue loss (do-nothing scenario).

        Uses the engine's calculate_max_possible_loss() if available,
        otherwise falls back to a heuristic based on total revenue at risk
        and episode length.
        """
        if hasattr(engine, "calculate_max_possible_loss"):
            val = engine.calculate_max_possible_loss()
            if val > 0:
                return val

        # Fallback: estimate from current financial state and episode config
        # Total revenue at risk * fraction of episode with active disruptions
        total_revenue = 0.0
        if hasattr(engine, "graph"):
            total_revenue = engine.graph.get_total_revenue_at_risk()
        if total_revenue <= 0 and hasattr(engine, "financial"):
            total_revenue = engine.financial.budget_total * 2.0

        # Assume ~60% of episode has active disruptions in worst case
        return max(total_revenue * 0.6, engine.financial.budget_total)

    def _get_max_cascade_nodes(self, engine: Any) -> float:
        """Get the maximum number of nodes that could go offline."""
        if hasattr(engine, "calculate_max_cascade_nodes"):
            return engine.calculate_max_cascade_nodes()
        # Fallback: count all non-customer nodes
        if hasattr(engine, "graph") and hasattr(engine.graph, "G"):
            g = engine.graph.G
            return max(1, len([
                n for n, d in g.nodes(data=True)
                if d.get("node_type", "").lower() in ("supplier", "port", "factory")
            ]))
        return 40.0  # hard task has 40 nodes

    def _count_nodes_went_offline(self, engine: Any) -> float:
        """Count nodes that went offline during the episode."""
        if hasattr(engine, "count_nodes_that_went_offline"):
            return engine.count_nodes_that_went_offline()
        # Fallback: count currently non-operational nodes
        if hasattr(engine, "graph"):
            statuses = engine.graph.get_node_statuses()
            return sum(1 for s in statuses if not s.is_operational)
        return 0

    @staticmethod
    def _find_first_meaningful_action_day(
        history: list[tuple[Any, Any]],
    ) -> int | None:
        """
        Find the day of the first meaningful (non-idle, non-alert) action
        that targets a node affected by an active disruption signal.

        Returns None if the agent never took a relevant meaningful action.
        """
        for action, obs in history:
            if action.action_type not in ("do_nothing", "issue_supplier_alert"):
                # Verify the action targets a node under threat
                if action.target_node_id is None:
                    # Untargeted actions (hedge_commodity) count as meaningful
                    return obs.current_day
                for sig in obs.active_signals:
                    if action.target_node_id in sig.affected_node_ids:
                        return obs.current_day
        return None

    @staticmethod
    def _check_any_stockout(history: list[tuple[Any, Any]], engine: Any) -> bool:
        """Check if any customer experienced a stockout during the episode."""
        # First check the engine method if available
        if hasattr(engine, "any_customer_experienced_stockout"):
            return engine.any_customer_experienced_stockout()

        # Fallback: check observation history for nodes with zero inventory
        for _, obs in history:
            for node in obs.node_statuses:
                if (
                    node.node_type in ("customer", "warehouse")
                    and node.inventory_days_cover <= 0
                    and node.is_operational
                ):
                    return True
        return False

    @staticmethod
    def _find_first_stockout_day(history: list[tuple[Any, Any]]) -> int:
        """Find the first day a stockout occurred. Returns episode length if none."""
        for _, obs in history:
            for node in obs.node_statuses:
                if (
                    node.node_type in ("customer", "warehouse")
                    and node.inventory_days_cover <= 0
                    and node.is_operational
                ):
                    return obs.current_day
        return len(history)

    def _evaluate_triage_order(
        self,
        action_targets: list[str],
        action_days: list[int],
        engine: Any,
    ) -> float:
        """
        Evaluate whether the agent addressed disruptions in priority order.

        For the medium task, the priority order should be:
        1. Port-related nodes (port strike has highest immediate impact)
        2. Thailand supplier nodes (flooding is second)
        3. Chinese supplier nodes (sanctions are slowest-onset)

        The scoring checks whether the first few meaningful actions targeted
        the highest-priority disruption zones.
        """
        if not action_targets:
            return 0.0  # No actions taken at all

        # Classify action targets by disruption zone
        port_nodes = set()
        flood_nodes = set()
        sanctions_nodes = set()

        if hasattr(engine, "graph") and hasattr(engine.graph, "G"):
            g = engine.graph.G
            for node_id, data in g.nodes(data=True):
                node_type = data.get("node_type", "")
                country = data.get("country", "")
                if node_type == "port" or "port" in node_id.lower():
                    port_nodes.add(node_id)
                if country in ("TH", "Thailand"):
                    flood_nodes.add(node_id)
                if country in ("CN", "China"):
                    sanctions_nodes.add(node_id)

        # Score based on action ordering
        # First 3 actions targeting port nodes = optimal triage
        score = 0.0
        early_targets = action_targets[:5]  # Look at first 5 actions

        if not port_nodes and not flood_nodes and not sanctions_nodes:
            # Cannot classify nodes -- give partial credit based on action diversity
            unique_targets = len(set(early_targets))
            return _clamp(min(1.0, unique_targets / 3.0) * 0.6)

        # Points for addressing port nodes first (highest priority)
        for i, target in enumerate(early_targets):
            weight = 1.0 - (i * 0.15)  # Earlier actions worth more
            if target in port_nodes:
                score += 0.25 * weight
            elif target in flood_nodes:
                score += 0.15 * weight
            elif target in sanctions_nodes:
                score += 0.10 * weight
            else:
                score += 0.05 * weight  # Some credit for any action

        # Bonus: Did the agent address all three zones?
        targeted_zones = set()
        for target in action_targets:
            if target in port_nodes:
                targeted_zones.add("port")
            elif target in flood_nodes:
                targeted_zones.add("flood")
            elif target in sanctions_nodes:
                targeted_zones.add("sanctions")

        coverage_bonus = len(targeted_zones) * 0.1
        score += coverage_bonus

        return _clamp(score)


    def _evaluate_cascade_stage_coverage(
        self,
        history: list[tuple[Any, Any]],
        engine: Any,
    ) -> float:
        """
        Evaluate whether the agent addressed all cascade stages.

        The hard scenario has multiple disruption types that cascade.
        The agent must take cost-bearing actions targeting nodes affected
        by EACH distinct disruption type. Score is the fraction of
        disruption types that received at least one targeted action.
        """
        # Collect all distinct disruption types from active signals
        all_disruption_types: set[str] = set()
        disruption_to_nodes: dict[str, set[str]] = {}

        for _, obs in history:
            for sig in obs.active_signals:
                dtype = sig.disruption_type
                all_disruption_types.add(dtype)
                if dtype not in disruption_to_nodes:
                    disruption_to_nodes[dtype] = set()
                disruption_to_nodes[dtype].update(sig.affected_node_ids)

        if not all_disruption_types:
            return 1.0  # No disruptions = nothing to suppress

        # Check which disruption types had cost-bearing actions targeting
        # their affected nodes
        addressed_types: set[str] = set()
        for action, obs in history:
            if action.action_type in ("do_nothing", "issue_supplier_alert"):
                continue
            # Untargeted actions (hedge_commodity) count for commodity-related disruptions
            if action.action_type == "hedge_commodity":
                # Credit for addressing sanctions/material_shortage types
                for dtype in all_disruption_types:
                    if dtype in ("sanctions", "material_shortage", "supplier_financial"):
                        addressed_types.add(dtype)
                continue
            if action.target_node_id:
                for dtype, nodes in disruption_to_nodes.items():
                    if action.target_node_id in nodes:
                        addressed_types.add(dtype)

        coverage = len(addressed_types) / len(all_disruption_types)
        return _clamp(coverage)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
