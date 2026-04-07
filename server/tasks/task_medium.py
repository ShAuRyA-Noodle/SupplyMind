"""
SupplyMind Medium Task: Multi-Front Crisis

Three concurrent disruptions across a 25-node multi-region electronics
supply chain. The agent must triage between a US port strike, Thailand
flooding, and Chinese supplier sanctions -- the budget only covers
mitigation for approximately two of three crises.
"""

from __future__ import annotations

from server.tasks.registry import TaskDefinition, TaskRegistry


def register_medium_task() -> None:
    """Register the medium 'Multi-Front Crisis' task."""
    TaskRegistry.register(
        TaskDefinition(
            task_id="medium_multi_front",
            name="Multi-Front Crisis",
            difficulty="medium",
            description=(
                "Triage three concurrent disruptions across a multi-region "
                "electronics supply chain with 25 nodes spanning 3 supplier "
                "tiers and 5 countries (Taiwan, Korea, Thailand, China, US).\n\n"
                "SCENARIO: Three crises hit in rapid succession:\n"
                "1. US West Coast port strike (Day 7) - Long Beach and Oakland "
                "ports shut down, blocking inbound shipments from Asia\n"
                "2. Thailand flooding (Day 9) - Monsoon flooding disrupts Tier 2 "
                "component suppliers in the Ayutthaya industrial zone\n"
                "3. Chinese rare earth sanctions (Day 18) - Export controls on "
                "rare earth materials affect Chinese suppliers\n\n"
                "KEY CHALLENGE: Your $8M budget can only fully mitigate ~2 of "
                "the 3 crises. You must decide which disruptions to address "
                "aggressively and which to accept partial losses on. Triage "
                "quality -- addressing the highest-impact disruptions first -- "
                "is critical.\n\n"
                "OPTIMAL STRATEGY: Prioritize port strike rerouting (highest "
                "immediate revenue impact), pre-position safety stock before "
                "Thailand floods peak, and hedge rare earth exposure rather "
                "than trying to find alternative suppliers.\n\n"
                "SCORING: Financial impact minimized (30%), triage quality "
                "(25%), budget utilization (20%), SLA compliance (15%), "
                "proactive actions (10%)."
            ),
            episode_length=45,
            budget=8_000_000.0,
            graph_file="server/data/graphs/medium_graph.json",
            disruption_file="server/data/disruptions/medium_scenarios.json",
            min_episode_days=35,
        )
    )
