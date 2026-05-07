"""test_spof_v2.py — G8 fix regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.spof_v2 import (
    benchmark, benchmark_all_graphs, detect_spofs_v2,
)


def test_spofs_detected_on_all_graphs():
    for g in ("easy_graph", "medium_graph", "hard_graph"):
        spofs = detect_spofs_v2(PROJECT_ROOT / "server" / "data" / "graphs" / f"{g}.json")
        assert len(spofs) >= 1, f"{g} should have at least 1 SPOF"
        # Every SPOF must increase components by >= 1
        assert all(s.increases_components_by >= 1 for s in spofs)


def test_benchmark_easy_graph_f1_perfect():
    result = benchmark("easy_graph")
    assert result["v2_articulation"]["f1"] == 1.0, \
        f"v2 articulation F1 must be 1.0 by construction, got {result}"


def test_benchmark_all_graphs_v2_beats_v1():
    result = benchmark_all_graphs()
    assert result["summary"]["v2_mean_f1"] >= result["summary"]["v1_mean_f1"], \
        "v2 must dominate v1 on mean F1"
    assert result["summary"]["v2_mean_f1"] >= 0.99, \
        "v2 mean F1 must be essentially perfect"
