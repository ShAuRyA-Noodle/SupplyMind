"""
rag_provenance.py — F8. RAG provenance graph with clickable citations.

Given (query, top_k_chunks), produce:
    - NetworkX graph: query_node -> document_nodes -> chunk_nodes
    - Interactive Plotly HTML dashboard (optional)
    - JSON summary with URLs + trust scores

Each chunk in the v3 corpus (6,483 total) has a document of origin. We classify
documents into 5 trust tiers:
    tier_1_regulatory (SEC 10-K, gov policy PDFs)
    tier_2_academic   (peer-reviewed papers, BIS / FRBSF / FRBNY)
    tier_3_reference  (Wikipedia articles with citations)
    tier_4_industry   (analyst reports, trade pubs)
    tier_5_other      (unclassified)

Trust score = 1.0 / tier_number (e.g. tier 1 = 1.0, tier 3 = 0.33). An answer's
provenance_score = weighted mean of its top-k source trust scores.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "provenance"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# --- Trust tier classifier ---------------------------------------------------

TIER_PATTERNS = [
    (1, "tier_1_regulatory", ["sec.gov", "10-K", "EDGAR", "10k", "10K"]),
    (2, "tier_2_academic",
     ["bis.org", "frbsf", "frbny", "imf.org", "worldbank.org",
      "arxiv.org", "nature.com", "science.org", "pubmed", "arxiv"]),
    (3, "tier_3_reference", ["wikipedia", "wiki"]),
    (4, "tier_4_industry",
     ["semianalysis", "gartner", "mckinsey", "bcg", "cscmp",
      "lloydslist", "freightos", "drewry", "alixpartners", "susquehanna"]),
    (5, "tier_5_other", []),
]


def classify_document(url_or_name: str) -> tuple[int, str, float]:
    """Return (tier_number, tier_label, trust_score)."""
    low = (url_or_name or "").lower()
    for tier_n, label, patterns in TIER_PATTERNS:
        if not patterns:
            continue
        if any(p.lower() in low for p in patterns):
            return tier_n, label, round(1.0 / tier_n, 3)
    return 5, "tier_5_other", round(1.0 / 5, 3)


@dataclass
class Chunk:
    id: str
    text: str
    doc_url: str
    doc_name: str
    score: float = 0.0       # retrieval score (cosine sim)


@dataclass
class Provenance:
    query: str
    chunks: list[Chunk]
    documents: dict = field(default_factory=dict)   # url -> {name, tier, trust}
    provenance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "n_chunks": len(self.chunks),
            "chunks": [{
                "id": c.id, "doc_url": c.doc_url, "doc_name": c.doc_name,
                "score": round(c.score, 4), "text_preview": c.text[:200],
            } for c in self.chunks],
            "documents": self.documents,
            "provenance_score": round(self.provenance_score, 3),
        }


def build_provenance(query: str, chunks: list[Chunk]) -> Provenance:
    docs: dict[str, dict] = {}
    weighted_trust_num = 0.0
    weighted_trust_den = 0.0
    for c in chunks:
        url = c.doc_url or c.doc_name or "unknown"
        tier_n, label, trust = classify_document(url)
        if url not in docs:
            docs[url] = {
                "name": c.doc_name or url,
                "tier": tier_n,
                "tier_label": label,
                "trust_score": trust,
                "chunk_ids": [],
            }
        docs[url]["chunk_ids"].append(c.id)
        # Weighted by retrieval score
        w = max(0.001, c.score)
        weighted_trust_num += trust * w
        weighted_trust_den += w

    prov = Provenance(query=query, chunks=chunks, documents=docs)
    prov.provenance_score = (weighted_trust_num / weighted_trust_den) if weighted_trust_den else 0.0
    return prov


def build_graph(prov: Provenance) -> nx.DiGraph:
    G = nx.DiGraph()
    q_id = "QUERY"
    G.add_node(q_id, kind="query", label=prov.query[:80])
    for url, meta in prov.documents.items():
        G.add_node(url, kind="document", label=meta["name"][:60],
                   tier=meta["tier"], trust=meta["trust_score"])
        G.add_edge(q_id, url, kind="retrieves_from",
                   score=sum(c.score for c in prov.chunks if c.doc_url == url))
    for c in prov.chunks:
        G.add_node(c.id, kind="chunk", label=c.text[:80], score=c.score)
        G.add_edge(c.doc_url or c.doc_name or "unknown", c.id, kind="contains")
    return G


def render_html(prov: Provenance, G: nx.DiGraph, out_path: Path) -> None:
    """Export an interactive Plotly HTML visualization (optional, graceful no-op)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.info("[rag_provenance] plotly not installed; skipping HTML render")
        return

    pos = nx.spring_layout(G, seed=42, k=1.5)
    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(color="#888", width=1),
                            hoverinfo="none")
    # Nodes — color by kind, size by type
    node_x, node_y, colors, sizes, texts, hovers = [], [], [], [], [], []
    color_map = {"query": "red", "document": "steelblue", "chunk": "lightgrey"}
    size_map = {"query": 30, "document": 22, "chunk": 12}
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        data = G.nodes[n]
        kind = data.get("kind", "chunk")
        colors.append(color_map.get(kind, "lightgrey"))
        sizes.append(size_map.get(kind, 12))
        texts.append(data.get("label", str(n))[:40])
        hovers.append(f"{kind}: {n}<br>{data.get('label', '')[:200]}")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=texts,
                            textposition="top center", textfont=dict(size=9),
                            marker=dict(color=colors, size=sizes, line=dict(width=1)),
                            hovertext=hovers, hoverinfo="text")

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=f"RAG Provenance — {prov.query[:80]}<br><sub>provenance score = {prov.provenance_score:.3f}</sub>",
        showlegend=False, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=700,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info("[rag_provenance] wrote %s", out_path)


# --- Demo: load a real slice of the v3 RAG corpus ----------------------------

def demo_run() -> dict:
    """Run a miniature provenance demo using manually-crafted chunks from v3 sources."""
    chunks = [
        Chunk(id="c1",
              text="TSMC produces 54% of global foundry revenue and 92% of <7nm advanced logic.",
              doc_url="https://www.semianalysis.com/tsmc-market-share",
              doc_name="SemiAnalysis — TSMC market share 2024", score=0.91),
        Chunk(id="c2",
              text="Section 1A Risk Factors: Concentration in a single Taiwanese manufacturing partner exposes Company to geopolitical risk.",
              doc_url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K",
              doc_name="Apple Inc. 10-K Fiscal 2024", score=0.88),
        Chunk(id="c3",
              text="Taiwan Strait tensions reached highest level since 1996 during PLA exercises August 2022.",
              doc_url="https://en.wikipedia.org/wiki/2022_Chinese_military_exercises_around_Taiwan",
              doc_name="Wikipedia — 2022 Taiwan Strait exercises", score=0.85),
        Chunk(id="c4",
              text="Global supply chain disruption costs exceeded $184 billion in 2023 per BCI analysis.",
              doc_url="https://www.bis.org/publ/qtrpdf/r_qt2312.htm",
              doc_name="BIS Quarterly Review — Dec 2023", score=0.82),
        Chunk(id="c5",
              text="Lead times for advanced nodes reached 52+ weeks at peak 2021 chip shortage.",
              doc_url="https://www.alixpartners.com/semi-report",
              doc_name="AlixPartners Semi Shortage 2021", score=0.79),
    ]
    prov = build_provenance(
        query="Why is TSMC a supply-chain single point of failure for advanced semiconductors?",
        chunks=chunks,
    )
    G = build_graph(prov)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "demo.json").write_text(json.dumps(prov.to_dict(), indent=2))
    render_html(prov, G, OUT_DIR / "demo.html")
    return {
        "provenance_score": prov.provenance_score,
        "n_chunks": len(chunks),
        "n_documents": len(prov.documents),
        "tier_distribution": {
            meta["tier_label"]: 1 for meta in prov.documents.values()
        },
        "html_path": str(OUT_DIR / "demo.html"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        result = demo_run()
        print(json.dumps(result, indent=2))
