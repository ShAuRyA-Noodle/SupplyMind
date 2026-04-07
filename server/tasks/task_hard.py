"""
SupplyMind Hard Task: Cascading Crisis

A cascading geopolitical crisis across a 40-node global automotive supply
chain spanning 6 countries and 3 tiers. A Taiwan Strait escalation triggers
a chain reaction: shipping disruption, semiconductor cutoff, commodity
price spikes, and a cyber attack on logistics systems.
"""

from __future__ import annotations

from server.tasks.registry import TaskDefinition, TaskRegistry


def register_hard_task() -> None:
    """Register the hard 'Cascading Crisis' task."""
    TaskRegistry.register(
        TaskDefinition(
            task_id="hard_cascading_crisis",
            name="Cascading Crisis",
            difficulty="hard",
            description=(
                "Navigate a cascading geopolitical crisis in a global automotive "
                "supply chain with 40 nodes spanning 3 tiers and 6 countries "
                "(Taiwan, Korea, Japan, Germany, India, US).\n\n"
                "SCENARIO: A Taiwan Strait escalation triggers an 8-event "
                "cascade over 30 days:\n"
                "1. Military exercises near Taiwan (Day 2) - warning signals\n"
                "2. Shipping lanes restricted (Day 5) - transit delays begin\n"
                "3. Naval blockade announced (Day 8) - Taiwan ports close\n"
                "4. TSMC production halted (Day 10) - semiconductor cutoff\n"
                "5. Samsung delays from Korean caution (Day 12) - backup limited\n"
                "6. Commodity price spike (Day 15) - rare earths +80%, chips +120%\n"
                "7. Cyber attack on logistics (Day 20) - warehouse systems down\n"
                "8. Partial reopening signals (Day 30) - slow recovery begins\n\n"
                "KEY CHALLENGE: Each disruption amplifies the next. Early "
                "containment prevents cascade amplification, but the $10M "
                "budget is tight relative to the 40-node network. Information "
                "gathering (supplier alerts) is critical for seeing the cascade "
                "before it hits. The agent must balance immediate firefighting "
                "with strategic positioning for the next wave.\n\n"
                "OPTIMAL STRATEGY: Heavy information gathering in early days, "
                "pre-position safety stock before blockade, diversify away from "
                "Taiwan suppliers before cutoff, hedge commodities before spike, "
                "and maintain budget reserves for the cyber attack response.\n\n"
                "SCORING: Total loss minimized (25%), cascade containment (20%), "
                "information efficiency (15%), budget ROI (15%), network "
                "resilience (15%), customer impact (10%)."
            ),
            episode_length=60,
            budget=10_000_000.0,
            graph_file="server/data/graphs/hard_graph.json",
            disruption_file="server/data/disruptions/hard_scenarios.json",
            min_episode_days=45,
        )
    )
