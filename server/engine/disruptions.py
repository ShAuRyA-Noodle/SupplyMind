"""
SupplyMind Disruption Lifecycle Engine

Manages pre-scripted disruption scenarios through their lifecycle phases:
WARNING -> ACTIVE -> RECOVERY -> RESOLVED.

Scenarios are loaded from JSON and optionally jittered via a seed for episode
variation. Same seed = same jitter = reproducible episodes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from models import DisruptionSignal

if TYPE_CHECKING:
    from server.engine.graph import SupplyChainGraph


class DisruptionScenario:
    """A single pre-scripted disruption with lifecycle parameters."""

    def __init__(self, data: dict) -> None:
        self.signal_id: str = data["signal_id"]
        self.disruption_type: str = data["disruption_type"]
        self.trigger_day: int = data["trigger_day"]
        self.warning_severity: float = data["warning_severity"]
        self.warning_confidence: float = data.get("warning_confidence", 0.6)
        self.peak_severity: float = data["peak_severity"]
        self.impact_day: int = data["impact_day"]
        self.recovery_start_day: int = data["recovery_start_day"]
        self.resolved_day: int = data["resolved_day"]
        self.affected_region: str = data["affected_region"]
        self.affected_node_ids: list[str] = data["affected_node_ids"]
        self.estimated_duration_days: float = data.get(
            "estimated_duration_days",
            float(self.resolved_day - self.trigger_day),
        )
        self.description: str = data["description"]

        # Optional: commodity price effects
        self.commodity_effects: dict[str, float] = data.get("commodity_effects", {})

    def get_phase(self, current_day: int) -> str | None:
        """
        Determine the lifecycle phase for a given day.

        Returns: "warning", "active", "recovery", "resolved", or None
        """
        if current_day < self.trigger_day:
            return None
        if current_day < self.impact_day:
            return "warning"
        if current_day < self.recovery_start_day:
            return "active"
        if current_day < self.resolved_day:
            return "recovery"
        return "resolved"

    def get_severity(self, current_day: int) -> float:
        """
        Calculate severity for the current day using realistic curves.

        - Warning: sigmoid escalation (slow start, rapid ramp near impact)
        - Active: bell curve with sustained peak (real disruptions aren't flat-top)
        - Recovery: exponential decay (fast initial improvement, long tail)
        """
        import math
        phase = self.get_phase(current_day)

        if phase is None:
            return 0.0

        if phase == "warning":
            warning_duration = self.impact_day - self.trigger_day
            if warning_duration <= 0:
                return self.warning_severity
            progress = (current_day - self.trigger_day) / warning_duration
            # Sigmoid escalation: slow start, accelerates toward impact
            return self.warning_severity * (1.0 / (1.0 + math.exp(-6.0 * (progress - 0.5))))

        if phase == "active":
            active_duration = self.recovery_start_day - self.impact_day
            if active_duration <= 0:
                return self.peak_severity
            progress = (current_day - self.impact_day) / active_duration
            # Bell curve: peak in the middle, slight dip at edges
            return self.peak_severity * (1.0 - 0.3 * (2.0 * progress - 1.0) ** 2)

        if phase == "recovery":
            recovery_duration = self.resolved_day - self.recovery_start_day
            if recovery_duration <= 0:
                return 0.0
            progress = (current_day - self.recovery_start_day) / recovery_duration
            # Exponential decay: fast initial recovery, long tail
            return self.peak_severity * math.exp(-3.0 * progress)

        # resolved
        return 0.0

    def get_confidence(self, current_day: int) -> float:
        """Calculate confidence for the current day."""
        phase = self.get_phase(current_day)

        if phase is None:
            return 0.0
        if phase == "warning":
            # Confidence increases as impact day approaches
            warning_duration = self.impact_day - self.trigger_day
            if warning_duration <= 0:
                return self.warning_confidence
            days_into_warning = current_day - self.trigger_day
            progress = days_into_warning / warning_duration
            return self.warning_confidence + (1.0 - self.warning_confidence) * progress * 0.6
        if phase == "active":
            return 1.0
        if phase == "recovery":
            return 0.9
        return 1.0

    def get_time_to_impact_hours(self, current_day: int) -> float:
        """Calculate hours until impact."""
        phase = self.get_phase(current_day)

        if phase is None or phase in ("active", "recovery", "resolved"):
            return 0.0
        if phase == "warning":
            days_until = self.impact_day - current_day
            return max(0.0, days_until * 24.0)
        return 0.0

    def to_signal(self, current_day: int) -> DisruptionSignal | None:
        """
        Convert this scenario to a DisruptionSignal for the given day.

        Returns None if the disruption has not started or is resolved.
        """
        phase = self.get_phase(current_day)

        if phase is None or phase == "resolved":
            return None

        return DisruptionSignal(
            signal_id=self.signal_id,
            disruption_type=self.disruption_type,
            severity=self.get_severity(current_day),
            confidence=self.get_confidence(current_day),
            affected_region=self.affected_region,
            affected_node_ids=self.affected_node_ids,
            time_to_impact_hours=self.get_time_to_impact_hours(current_day),
            estimated_duration_days=self.estimated_duration_days,
            description=self._get_description_for_phase(phase, current_day),
            lifecycle_phase=phase,
        )

    def _get_description_for_phase(self, phase: str, current_day: int) -> str:
        """Generate a phase-appropriate description."""
        base = self.description

        if phase == "warning":
            hours = self.get_time_to_impact_hours(current_day)
            return f"[WARNING] {base} Expected impact in {hours:.0f} hours."
        if phase == "active":
            severity = self.get_severity(current_day)
            return (
                f"[ACTIVE] {base} "
                f"Severity: {severity:.0%}. "
                f"Nodes affected: {', '.join(self.affected_node_ids)}."
            )
        if phase == "recovery":
            severity = self.get_severity(current_day)
            return (
                f"[RECOVERY] {base} "
                f"Severity decreasing: {severity:.0%}. "
                f"Expected resolution by day {self.resolved_day}."
            )
        return base


class DisruptionEngine:
    """
    Manages disruption lifecycles for a simulation episode.

    Loads pre-scripted scenarios from JSON and advances them day by day,
    producing DisruptionSignal objects for the observation.
    """

    def __init__(self) -> None:
        self.scenarios: list[DisruptionScenario] = []
        self._current_day: int = 0
        self._previous_active_ids: set[str] = set()
        self._new_signal_ids: set[str] = set()

    def load_scenarios(self, filepath: str) -> None:
        """Load disruption scenarios from a JSON file."""
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)

        self.scenarios = [
            DisruptionScenario(d) for d in data.get("disruptions", [])
        ]

    def apply_jitter(self, seed: int, graph: SupplyChainGraph) -> None:
        """
        Apply seed-based jitter to loaded scenarios for episode variation.

        Same seed always produces the same jitter (reproducible).
        Jitters: trigger/impact/recovery/resolved days (±0-2), peak severity
        (±0-0.08), and occasionally swaps one affected node with a same-type
        graph neighbor (30% chance per scenario).

        Args:
            seed: RNG seed for deterministic jitter.
            graph: Supply chain graph (used for neighbor lookups during node swap).
        """
        rng = random.Random(seed)

        for scenario in self.scenarios:
            # Jitter timing: shift all phase boundaries by the same offset
            # to preserve phase durations
            day_offset = rng.randint(0, 2)
            scenario.trigger_day += day_offset
            scenario.impact_day += day_offset
            scenario.recovery_start_day += day_offset
            scenario.resolved_day += day_offset

            # Jitter peak severity ±0.08
            sev_jitter = rng.uniform(-0.08, 0.08)
            scenario.peak_severity = max(0.1, min(1.0, scenario.peak_severity + sev_jitter))

            # Occasionally swap one affected node with a same-type neighbor
            if scenario.affected_node_ids and rng.random() < 0.3:
                idx = rng.randint(0, len(scenario.affected_node_ids) - 1)
                node_id = scenario.affected_node_ids[idx]
                if node_id in graph.G:
                    node_type = graph.G.nodes[node_id].get("node_type", "")
                    # Collect same-type neighbors (both successors and predecessors)
                    neighbors = list(graph.G.successors(node_id)) + list(graph.G.predecessors(node_id))
                    same_type = [
                        n for n in neighbors
                        if graph.G.nodes[n].get("node_type") == node_type
                    ]
                    if same_type:
                        scenario.affected_node_ids[idx] = rng.choice(same_type)

    def advance_day(self, current_day: int) -> list[DisruptionSignal]:
        """
        Advance to the given day and return all active signals.

        Also tracks which signals are new this step.
        """
        self._current_day = current_day

        current_active_ids: set[str] = set()
        self._new_signal_ids = set()
        signals: list[DisruptionSignal] = []

        for scenario in self.scenarios:
            signal = scenario.to_signal(current_day)
            if signal is not None:
                signals.append(signal)
                current_active_ids.add(scenario.signal_id)

                # Track new signals
                if scenario.signal_id not in self._previous_active_ids:
                    self._new_signal_ids.add(scenario.signal_id)

        self._previous_active_ids = current_active_ids
        return signals

    def get_active_signals(self) -> list[DisruptionSignal]:
        """Get all currently active disruption signals."""
        signals = []
        for scenario in self.scenarios:
            signal = scenario.to_signal(self._current_day)
            if signal is not None:
                signals.append(signal)
        return signals

    def get_new_signals(self) -> list[DisruptionSignal]:
        """Get signals that appeared this step only."""
        signals = []
        for scenario in self.scenarios:
            if scenario.signal_id in self._new_signal_ids:
                signal = scenario.to_signal(self._current_day)
                if signal is not None:
                    signals.append(signal)
        return signals

    def apply_to_graph(self, graph: SupplyChainGraph) -> None:
        """
        Update graph node operational status based on active disruptions.

        - ACTIVE phase: sets affected nodes as non-operational, propagates
        - RECOVERY phase: gradually restores nodes
        - RESOLVED: fully restores nodes
        - WARNING: marks risk scores but doesn't disable
        """
        for scenario in self.scenarios:
            phase = scenario.get_phase(self._current_day)
            severity = scenario.get_severity(self._current_day)

            if phase is None:
                continue

            for node_id in scenario.affected_node_ids:
                if node_id not in graph.G:
                    continue

                node_data = graph.G.nodes[node_id]
                node_type = node_data.get("node_type", "").lower()

                if phase == "warning":
                    # Increase risk score but don't disable
                    current_risk = node_data.get("risk_score", 0.0)
                    node_data["risk_score"] = max(
                        current_risk, scenario.warning_severity * 0.7
                    )
                    graph.set_node_disruption(node_id, scenario.signal_id)

                elif phase == "active":
                    # Set as non-operational if severity is high enough
                    if severity >= 0.5 and node_type in ("supplier", "port", "factory"):
                        node_data["is_operational"] = False
                        graph._ever_offline.add(node_id)
                    node_data["risk_score"] = max(
                        node_data.get("risk_score", 0.0), severity
                    )
                    graph.set_node_disruption(node_id, scenario.signal_id)

                    # Propagate disruption through the graph
                    graph.propagate_disruption(
                        node_id, severity, scenario.estimated_duration_days
                    )

                elif phase == "recovery":
                    # Gradually restore
                    if severity < 0.3 and node_type in ("supplier", "port", "factory"):
                        node_data["is_operational"] = True
                    node_data["risk_score"] = max(0.0, severity)
                    graph.set_node_disruption(node_id, scenario.signal_id)

                elif phase == "resolved":
                    # Fully restore
                    if node_type in ("supplier", "port", "factory"):
                        node_data["is_operational"] = True
                    node_data["risk_score"] = max(
                        0.0, node_data.get("risk_score", 0.0) - 0.3
                    )
                    graph.clear_node_disruption(node_id, scenario.signal_id)

        # Apply commodity price effects for active disruptions
        self._update_commodity_effects()

    def _update_commodity_effects(self) -> dict[str, float]:
        """Calculate commodity price effects from active disruptions."""
        effects: dict[str, float] = {}

        for scenario in self.scenarios:
            phase = scenario.get_phase(self._current_day)
            if phase in ("active", "recovery"):
                severity = scenario.get_severity(self._current_day)
                for commodity, max_multiplier in scenario.commodity_effects.items():
                    current = effects.get(commodity, 1.0)
                    effect = 1.0 + (max_multiplier - 1.0) * severity
                    effects[commodity] = max(current, effect)

        return effects

    def get_commodity_effects(self) -> dict[str, float]:
        """Get current commodity price effects from all active disruptions."""
        return self._update_commodity_effects()

    def all_resolved(self) -> bool:
        """Check if all disruptions have been resolved."""
        for scenario in self.scenarios:
            phase = scenario.get_phase(self._current_day)
            if phase is not None and phase != "resolved":
                return False
        return True

    def get_disrupted_node_ids(self) -> list[str]:
        """Get all node IDs currently affected by active disruptions."""
        node_ids: set[str] = set()
        for scenario in self.scenarios:
            phase = scenario.get_phase(self._current_day)
            if phase in ("active", "recovery"):
                node_ids.update(scenario.affected_node_ids)
        return list(node_ids)

    def get_max_severity(self) -> float:
        """Get the maximum severity across all active disruptions."""
        max_sev = 0.0
        for scenario in self.scenarios:
            severity = scenario.get_severity(self._current_day)
            max_sev = max(max_sev, severity)
        return max_sev
