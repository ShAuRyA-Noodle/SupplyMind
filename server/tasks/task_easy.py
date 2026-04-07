"""
SupplyMind Easy Task: Typhoon Response

Single-disruption scenario with a 12-node semiconductor supply chain.
A Category 3 typhoon approaches Taiwan, threatening TSMC (single-source
chip supplier). The agent has a 72-hour warning window to activate backup
suppliers and expedite critical orders before impact.
"""

from __future__ import annotations

from server.tasks.registry import TaskDefinition, TaskRegistry


def register_easy_task() -> None:
    """Register the easy 'Typhoon Response' task."""
    TaskRegistry.register(
        TaskDefinition(
            task_id="easy_typhoon_response",
            name="Typhoon Response",
            difficulty="easy",
            description=(
                "Manage a semiconductor supply chain through a single typhoon "
                "disruption affecting Taiwan. Your network has 12 supply chain "
                "nodes across 2 tiers, centered on the Taiwan-Korea-US corridor.\n\n"
                "SCENARIO: A Category 3 typhoon is approaching the Taiwan "
                "manufacturing corridor. You receive warning signals 72 hours "
                "before impact. TSMC Fab 14 (your single-source chip supplier, "
                "$500M annual spend) and the Port of Kaohsiung are directly in "
                "the storm path.\n\n"
                "KEY CHALLENGE: TSMC is a single-source dependency. If it goes "
                "offline without preparation, downstream customers (Apple, Dell, "
                "HP) face stockouts within 15-20 days. Samsung (Korea) is "
                "available as a backup but requires activation cost and has a "
                "20% cost premium.\n\n"
                "OPTIMAL STRATEGY: Issue supplier alert during warning phase, "
                "activate Samsung as backup before impact, expedite critical "
                "orders via air freight, and increase safety stock at US "
                "warehouses. Budget of $5M is sufficient if spent wisely.\n\n"
                "SCORING: Revenue preserved (40%), timeliness of response (25%), "
                "cost efficiency (20%), stockout prevention (15%)."
            ),
            episode_length=30,
            budget=5_000_000.0,
            graph_file="server/data/graphs/easy_graph.json",
            disruption_file="server/data/disruptions/easy_scenarios.json",
            min_episode_days=20,
        )
    )
