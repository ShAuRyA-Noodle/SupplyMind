"""test_rag_provenance.py — F8 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.rag_provenance import (
    Chunk, build_graph, build_provenance, classify_document, demo_run,
)


def test_tier_classifier():
    assert classify_document("https://www.sec.gov/edgar/apple-10k")[0] == 1
    assert classify_document("bis.org/publ")[0] == 2
    assert classify_document("https://en.wikipedia.org/wiki/Taiwan_Strait")[0] == 3
    assert classify_document("https://semianalysis.com/tsmc")[0] == 4
    assert classify_document("https://randomblog.example/post")[0] == 5


def test_build_provenance_and_score():
    chunks = [
        Chunk(id="a", text="hi", doc_url="https://sec.gov/x", doc_name="SEC 10-K", score=0.9),
        Chunk(id="b", text="hi", doc_url="https://wikipedia.org/y", doc_name="Wiki", score=0.8),
    ]
    prov = build_provenance("why", chunks)
    assert prov.provenance_score > 0
    # Weighted toward tier-1 SEC (higher retrieval weight + tier-1 trust)
    assert prov.provenance_score > 0.5  # weighted-mean of (1.0*0.9 + 0.333*0.8)/(0.9+0.8) = 0.685
    assert len(prov.documents) == 2


def test_build_graph_structure():
    chunks = [
        Chunk(id="c1", text="t", doc_url="https://sec.gov/x", doc_name="SEC", score=0.9),
    ]
    prov = build_provenance("q", chunks)
    G = build_graph(prov)
    # Must have query + document + chunk = 3 nodes
    kinds = {G.nodes[n].get("kind") for n in G.nodes()}
    assert "query" in kinds and "document" in kinds and "chunk" in kinds


def test_demo_runs_without_crash():
    result = demo_run()
    assert result["n_chunks"] >= 5
    assert result["n_documents"] >= 4
    assert 0 <= result["provenance_score"] <= 1
