"""test_gcn_attention_viz.py — F7 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.gcn_attention_viz import (
    compute_edge_importance, GRAPHS_DIR,
)


def test_edge_importance_easy_graph_non_trivial():
    imps = compute_edge_importance(GRAPHS_DIR / "easy_graph.json")
    assert len(imps) >= 10
    # Top edge must have materially higher importance than median
    vals = sorted([e.gradient_magnitude for e in imps], reverse=True)
    assert vals[0] > vals[len(vals) // 2], \
        "top edge should exceed the median importance"
    # All values in [0, 1]
    assert all(0.0 <= e.gradient_magnitude <= 1.0 for e in imps)


def test_edge_importance_respects_target_node():
    imps_a = compute_edge_importance(GRAPHS_DIR / "medium_graph.json",
                                     target_node_id="FAC_SUZHOU")
    imps_b = compute_edge_importance(GRAPHS_DIR / "medium_graph.json",
                                     target_node_id="FAC_GUADALAJARA")
    # Different targets should give at least SOME different top-edge ordering
    top_a = {(e.source, e.target) for e in imps_a[:5]}
    top_b = {(e.source, e.target) for e in imps_b[:5]}
    # Allow full overlap only on tiny graphs; medium (25 nodes) should diverge
    assert top_a != top_b or len(imps_a) < 10


def test_empty_graph_gracefully():
    # Construct an ad-hoc empty-ish graph
    import tempfile
    import json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"nodes": [{"id": "X", "node_type": "supplier"}], "edges": []}, f)
        p = Path(f.name)
    try:
        imps = compute_edge_importance(p)
        assert imps == []
    finally:
        p.unlink(missing_ok=True)
