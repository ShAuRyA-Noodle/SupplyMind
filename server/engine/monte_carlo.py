"""
SupplyMind Monte Carlo Engine

Runs N simulations with randomized disruption parameters to estimate
the probability distribution of financial losses. Uses Beta distributions
for severity and lognormal for duration.

Results (P50/P95/P99) are included in the observation to help the agent
make informed risk decisions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from models import DisruptionSignal

if TYPE_CHECKING:
    from server.engine.graph import SupplyChainGraph


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for probabilistic loss estimation.

    For each simulation run:
    1. Randomize severity using Beta distribution centered on current severity
    2. Randomize duration using lognormal distribution centered on expected duration
    3. Run disruption propagation with randomized parameters
    4. Sum revenue at risk across all affected nodes
    5. Compute percentile estimates from all runs
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def run_simulation(
        self,
        graph: SupplyChainGraph,
        active_disruptions: list[DisruptionSignal],
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """
        Run Monte Carlo simulation to estimate loss distribution.

        Args:
            graph: Current supply chain graph state
            active_disruptions: Currently active disruption signals
            n_simulations: Number of simulation runs (default 500)

        Returns:
            Dictionary with keys: p50_loss, p95_loss, p99_loss,
            avg_nodes_affected, max_delay_days
        """
        if not active_disruptions:
            return {
                "p50_loss": 0.0,
                "p95_loss": 0.0,
                "p99_loss": 0.0,
                "avg_nodes_affected": 0.0,
                "max_delay_days": 0.0,
            }

        losses: list[float] = []
        nodes_affected_counts: list[int] = []
        max_delays: list[float] = []

        for _ in range(n_simulations):
            sim_loss = 0.0
            sim_nodes_affected = 0
            sim_max_delay = 0.0

            for disruption in active_disruptions:
                # Randomize severity using Beta distribution
                rand_severity = self._randomize_severity(disruption.severity)

                # Randomize duration using lognormal with severity correlation:
                # more severe disruptions tend to last longer (0.6 factor)
                severity_factor = 1.0 + 0.6 * (rand_severity - disruption.severity)
                correlated_base = disruption.estimated_duration_days * max(0.5, severity_factor)
                rand_duration = self._randomize_duration(correlated_base)

                # Run propagation on a lightweight copy
                for node_id in disruption.affected_node_ids:
                    if node_id not in graph.G:
                        continue

                    affected = self._simulate_propagation(
                        graph, node_id, rand_severity, rand_duration
                    )

                    sim_nodes_affected += len(affected)

                    for info in affected.values():
                        sim_loss += info.get("revenue_at_risk", 0.0)
                        sim_max_delay = max(
                            sim_max_delay, info.get("delay_days", 0.0)
                        )

            losses.append(sim_loss)
            nodes_affected_counts.append(sim_nodes_affected)
            max_delays.append(sim_max_delay)

        # Compute percentiles
        losses_arr = np.array(losses)
        nodes_arr = np.array(nodes_affected_counts, dtype=float)
        delays_arr = np.array(max_delays)

        return {
            "p50_loss": float(np.percentile(losses_arr, 50)),
            "p95_loss": float(np.percentile(losses_arr, 95)),
            "p99_loss": float(np.percentile(losses_arr, 99)),
            "avg_nodes_affected": float(np.mean(nodes_arr)),
            "max_delay_days": float(np.percentile(delays_arr, 95)),
        }

    def _randomize_severity(self, base_severity: float) -> float:
        """
        Randomize severity using a Beta distribution.

        The Beta distribution is parameterized so that its mean equals
        base_severity. We use concentration parameter kappa=10 to control
        spread.
        """
        if base_severity <= 0.0:
            return 0.0
        if base_severity >= 1.0:
            return 1.0

        # Beta distribution with mean = base_severity
        kappa = 10.0
        alpha = base_severity * kappa
        beta = (1.0 - base_severity) * kappa

        # Ensure alpha, beta > 0
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)

        sample = self._rng.beta(alpha, beta)
        return float(max(0.0, min(1.0, sample)))

    def _randomize_duration(self, base_duration: float) -> float:
        """
        Randomize duration using a lognormal distribution.

        Mean of the lognormal is base_duration, with sigma=0.3.
        """
        if base_duration <= 0:
            return 0.0

        mu = np.log(max(0.1, base_duration))
        sigma = 0.3

        sample = self._rng.lognormal(mu, sigma)
        return float(max(1.0, sample))

    def _simulate_propagation(
        self,
        graph: SupplyChainGraph,
        node_id: str,
        severity: float,
        duration: float,
    ) -> dict[str, dict[str, float]]:
        """
        Lightweight propagation simulation without modifying the actual graph.

        Uses BFS similar to graph.propagate_disruption but reads node data
        without modifying operational status.
        """
        from collections import deque

        if node_id not in graph.G:
            return {}

        affected: dict[str, dict[str, float]] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, float, float]] = deque()
        queue.append((node_id, severity, 0.0))

        severity_decay = 0.20

        while queue:
            current_id, current_sev, cumulative_delay = queue.popleft()

            if current_id in visited or current_sev < 0.05:
                continue
            visited.add(current_id)

            node_data = graph.G.nodes[current_id]
            node_type = node_data.get("node_type", "").lower()

            # Calculate revenue at risk
            revenue_at_risk = 0.0
            if node_type == "customer":
                revenue_at_risk = (
                    node_data.get("revenue_contribution", 0.0) * current_sev
                )

            affected[current_id] = {
                "delay_days": cumulative_delay + duration * current_sev,
                "severity": current_sev,
                "revenue_at_risk": revenue_at_risk,
            }

            # Traverse downstream
            for _, neighbor in graph.G.out_edges(current_id):
                if neighbor in visited:
                    continue

                edge_data = graph.G.edges[current_id, neighbor]
                edge_lead = edge_data.get(
                    "lead_time_days", edge_data.get("transit_time_days", 1)
                )

                next_sev = current_sev - severity_decay

                # Warehouse buffer absorption
                neighbor_data = graph.G.nodes[neighbor]
                if neighbor_data.get("node_type", "").lower() == "warehouse":
                    inv_cover = neighbor_data.get("inventory_days_cover", 0.0)
                    if inv_cover > duration * current_sev:
                        next_sev *= 0.3
                    else:
                        absorption = inv_cover / max(1.0, duration * current_sev)
                        next_sev *= 1.0 - absorption * 0.5

                if next_sev > 0.05:
                    queue.append(
                        (neighbor, next_sev, cumulative_delay + edge_lead)
                    )

        return affected

    def run_quick_simulation(
        self,
        graph: SupplyChainGraph,
        active_disruptions: list[DisruptionSignal],
    ) -> dict[str, float]:
        """
        Quick simulation with fewer runs (N=500) for per-step use.

        Suitable for real-time simulation stepping where speed matters.
        """
        return self.run_simulation(graph, active_disruptions, n_simulations=500)
