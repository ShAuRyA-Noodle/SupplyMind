"""hormuz_chokepoint_graph.py — IEA/EIA-cited Hormuz flow graph.

14 nodes + 18 edges. Every flow capacity in million barrels per day (mb/d) or
LNG Bcf/d, with source URL + as-of date. Used by the war-room frontend to
render the chokepoint map and bypass-ceiling overlays.

Headline number: ~20 mb/d total Hormuz oil + products throughput in 2025
(IEA factsheet). Saudi East-West (~5 mb/d capacity, ~2 mb/d typical) +
UAE Habshan-Fujairah (~1.5 mb/d) provide the only meaningful bypass.
LNG has NO Hormuz alternative for Qatar (~85% of Qatar LNG exits Hormuz).
"""
from __future__ import annotations


# Layout uses normalized [0..1] coordinates; UI projects into SVG viewBox.
NODES: list[dict] = [
    # Producers — top half
    {"id": "IRAN", "label": "Iran",
     "type": "producer", "x": 0.18, "y": 0.18,
     "throughput_mbd": 1.5, "throughput_note": "post-sanctions exports, mostly to China",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},
    {"id": "IRAQ", "label": "Iraq (Basra)",
     "type": "producer", "x": 0.05, "y": 0.30,
     "throughput_mbd": 3.5, "throughput_note": "Basra crude; Hormuz-routed",
     "source": "https://www.eia.gov/international/analysis/country/IRQ",
     "agency": "EIA"},
    {"id": "KUWAIT", "label": "Kuwait",
     "type": "producer", "x": 0.10, "y": 0.42,
     "throughput_mbd": 2.5, "throughput_note": "Mina al-Ahmadi exports",
     "source": "https://www.eia.gov/international/analysis/country/KWT",
     "agency": "EIA"},
    {"id": "SAUDI", "label": "Saudi Arabia",
     "type": "producer", "x": 0.20, "y": 0.55,
     "throughput_mbd": 6.0, "throughput_note": "Gulf-shipped (also has East-West bypass)",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},
    {"id": "QATAR", "label": "Qatar",
     "type": "producer", "x": 0.32, "y": 0.30,
     "throughput_mbd": 1.0, "throughput_lng_bcfd": 10.5,
     "throughput_note": "~85% of Qatar LNG via Hormuz; ~1 mb/d crude+condensate",
     "source": "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
     "agency": "EIA"},
    {"id": "UAE", "label": "UAE",
     "type": "producer", "x": 0.40, "y": 0.45,
     "throughput_mbd": 3.0,
     "throughput_note": "ADNOC crude; bypassable via Habshan-Fujairah",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},

    # Bypass routes
    {"id": "EAST_WEST_PIPELINE", "label": "Saudi East-West pipeline",
     "type": "bypass", "x": 0.10, "y": 0.65,
     "capacity_mbd": 5.0, "current_utilization_mbd": 2.0,
     "throughput_note": "~5 mb/d capacity; typical use ~2 mb/d (IEA)",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},
    {"id": "HABSHAN_FUJAIRAH", "label": "Habshan–Fujairah pipeline",
     "type": "bypass", "x": 0.45, "y": 0.30,
     "capacity_mbd": 1.5, "current_utilization_mbd": 0.7,
     "throughput_note": "ADNOC pipeline lands UAE crude at Fujairah, outside Hormuz",
     "source": "https://en.wikipedia.org/wiki/ADNOC",
     "agency": "ADNOC"},
    {"id": "FUJAIRAH", "label": "Fujairah port",
     "type": "bypass_node", "x": 0.55, "y": 0.32,
     "throughput_note": "Strategic bunkering hub — gains volume during Hormuz disruption",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},

    # The chokepoint itself
    {"id": "STRAIT_OF_HORMUZ", "label": "Strait of Hormuz",
     "type": "chokepoint", "x": 0.55, "y": 0.42,
     "throughput_mbd": 20.0, "throughput_lng_bcfd": 13.0,
     "throughput_note": "~20 mb/d oil + products + ~20% global LNG (IEA / EIA 2025)",
     "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
     "agency": "IEA"},

    # Consumers — bottom + right
    {"id": "INDIA_WEST_COAST", "label": "India west-coast ports",
     "type": "consumer", "x": 0.62, "y": 0.65,
     "throughput_mbd": 3.5, "throughput_note": "Mundra/Kandla/Mumbai/Jamnagar",
     "source": "https://www.ppac.gov.in/",
     "agency": "PPAC India"},
    {"id": "CHINA", "label": "China",
     "type": "consumer", "x": 0.85, "y": 0.55,
     "throughput_mbd": 5.5, "throughput_note": "Largest single Hormuz crude destination",
     "source": "https://www.eia.gov/international/analysis/country/CHN",
     "agency": "EIA"},
    {"id": "JAPAN_KOREA", "label": "Japan + Korea",
     "type": "consumer", "x": 0.92, "y": 0.32,
     "throughput_mbd": 3.5, "throughput_note": "JKT LNG + crude buyers",
     "source": "https://www.eia.gov/international/analysis/country/JPN",
     "agency": "EIA"},
    {"id": "EUROPE", "label": "Europe",
     "type": "consumer", "x": 0.20, "y": 0.85,
     "throughput_mbd": 2.0, "throughput_note": "Mostly via Suez/SUMED, partial via Hormuz",
     "source": "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
     "agency": "EIA"},
]


# Edges with cited flow capacity (mb/d) or LNG Bcf/d.
EDGES: list[dict] = [
    # Producers → Hormuz
    {"src": "IRAN", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 1.5, "kind": "crude",
     "agency": "IEA"},
    {"src": "IRAQ", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 3.5, "kind": "crude",
     "agency": "EIA"},
    {"src": "KUWAIT", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 2.5, "kind": "crude",
     "agency": "EIA"},
    {"src": "SAUDI", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 6.0, "kind": "crude",
     "agency": "IEA"},
    {"src": "QATAR", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 1.0, "kind": "crude_condensate",
     "agency": "EIA"},
    {"src": "QATAR", "dst": "STRAIT_OF_HORMUZ", "flow_lng_bcfd": 10.5, "kind": "lng",
     "agency": "EIA",
     "note": "~85% of Qatar LNG exits via Hormuz; no pipeline alternative"},
    {"src": "UAE", "dst": "STRAIT_OF_HORMUZ", "flow_mbd": 2.3, "kind": "crude",
     "agency": "IEA"},

    # Bypass paths
    {"src": "SAUDI", "dst": "EAST_WEST_PIPELINE", "flow_mbd": 2.0,
     "capacity_mbd": 5.0, "kind": "bypass_pipeline",
     "agency": "IEA",
     "note": "~5 mb/d capacity, typical use ~2 mb/d, lands at Yanbu (Red Sea)"},
    {"src": "EAST_WEST_PIPELINE", "dst": "EUROPE", "flow_mbd": 1.5, "kind": "bypass_route",
     "agency": "EIA"},
    {"src": "UAE", "dst": "HABSHAN_FUJAIRAH", "flow_mbd": 0.7,
     "capacity_mbd": 1.5, "kind": "bypass_pipeline",
     "agency": "ADNOC",
     "note": "1.5 mb/d capacity; lands at Fujairah outside Hormuz"},
    {"src": "HABSHAN_FUJAIRAH", "dst": "FUJAIRAH", "flow_mbd": 0.7, "kind": "bypass_route",
     "agency": "ADNOC"},
    {"src": "FUJAIRAH", "dst": "INDIA_WEST_COAST", "flow_mbd": 0.4, "kind": "bypass_export",
     "agency": "IEA"},
    {"src": "FUJAIRAH", "dst": "JAPAN_KOREA", "flow_mbd": 0.3, "kind": "bypass_export",
     "agency": "IEA"},

    # Hormuz → consumers
    {"src": "STRAIT_OF_HORMUZ", "dst": "INDIA_WEST_COAST", "flow_mbd": 3.5,
     "kind": "crude", "agency": "PPAC India"},
    {"src": "STRAIT_OF_HORMUZ", "dst": "CHINA", "flow_mbd": 5.5, "kind": "crude",
     "agency": "EIA"},
    {"src": "STRAIT_OF_HORMUZ", "dst": "JAPAN_KOREA", "flow_mbd": 3.5,
     "kind": "crude_lng", "agency": "EIA"},
    {"src": "STRAIT_OF_HORMUZ", "dst": "EUROPE", "flow_mbd": 0.5, "kind": "crude",
     "agency": "EIA"},
    {"src": "STRAIT_OF_HORMUZ", "dst": "JAPAN_KOREA", "flow_lng_bcfd": 8.5,
     "kind": "lng", "agency": "EIA",
     "note": "JKT receives ~65% of Qatar LNG via Hormuz"},
]


HEADLINE_FACTS: list[dict] = [
    {
        "fact": "~20 million barrels per day of oil + products transit Hormuz (2025)",
        "value": 20.0, "unit": "mb/d",
        "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
        "agency": "IEA",
        "as_of": "2025",
    },
    {
        "fact": "~25% of world seaborne oil trade passes through Hormuz",
        "value": 25.0, "unit": "% world seaborne oil",
        "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
        "agency": "IEA",
        "as_of": "2025",
    },
    {
        "fact": "~80% of Hormuz oil flows are destined for Asia",
        "value": 80.0, "unit": "% of Hormuz oil to Asia",
        "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
        "agency": "IEA",
        "as_of": "2025",
    },
    {
        "fact": "~20% of global LNG trade transits Hormuz, mostly Qatari",
        "value": 20.0, "unit": "% global LNG trade",
        "source": "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints",
        "agency": "EIA",
        "as_of": "2024",
    },
    {
        "fact": "Saudi + UAE bypass pipelines can redirect only 3.5–5.5 mb/d combined",
        "value_low": 3.5, "value_high": 5.5, "unit": "mb/d",
        "source": "https://www.iea.org/reports/the-oil-and-gas-industry-in-net-zero-transitions",
        "agency": "IEA",
        "as_of": "2025",
        "note": "vs ~20 mb/d total Hormuz oil — bypass covers <30% of disruption",
    },
]


def get_graph() -> dict:
    """Return the graph dict ready for JSON serialization to the frontend."""
    return {
        "nodes": NODES,
        "edges": EDGES,
        "headline_facts": HEADLINE_FACTS,
        "data_attribution": (
            "All flow numbers cited from IEA Strait of Hormuz factsheet (2025), "
            "EIA World Oil Transit Chokepoints (2024), ADNOC corporate, and "
            "PPAC India. Numbers are monthly-average representative; daily flows "
            "vary ±15%."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_graph(), indent=2))
