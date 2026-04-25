"""het_temporal_gat.py — Heterogeneous Temporal Graph Attention Network.

Upgrades the v1 3-layer plain GCN cascade predictor to:

  1. **Heterogeneous edges** — separate attention per edge_type
     (SHIPS_TO, SUPPLIES, ROUTES_VIA, ALTERNATE_TO, ...).
  2. **Multi-head attention** — Velickovic-style GAT with K=4 heads.
  3. **Temporal gating** — GRU fuses node embedding at step t with the
     embedding at step t-1, so cascades that build over multiple
     simulation days are tracked properly.
  4. **Node typing** — separate input projections per node_type
     (PORT, WAREHOUSE, SUPPLIER, CUSTOMER) to capture role-specific
     features.

Forward usage:

    cfg = HetGATConfig()
    model = HetTemporalGAT(cfg)
    h_t = model(node_feats, node_types, edge_index, edge_types,
                  prev_h=h_t_minus_1)

`prev_h` is the previous-timestep hidden state (None on day 0). The
model returns the new hidden state which becomes prev_h on the next
call. This GRU memory is what makes cascades tractable.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HetGATConfig:
    in_dim: int = 16
    hidden_dim: int = 64
    out_dim: int = 32
    n_heads: int = 4
    n_node_types: int = 4   # PORT / WAREHOUSE / SUPPLIER / CUSTOMER
    n_edge_types: int = 4   # SHIPS_TO / SUPPLIES / ROUTES_VIA / ALTERNATE_TO
    n_layers: int = 2
    dropout: float = 0.2
    use_temporal_gru: bool = True


# ---------------------------------------------------------------------
# Het-attention layer (Velickovic GAT × edge-type attention)
# ---------------------------------------------------------------------

class HetGATLayer(nn.Module):
    """One layer of multi-head GAT with edge-type-conditional weights.

    Shapes:
      x         : (N, in_dim)            node features
      node_type : (N,) ints               in [0, n_node_types)
      edge_index: (2, E) ints             [src; dst]
      edge_type : (E,) ints                in [0, n_edge_types)
    Returns:
      h         : (N, n_heads * head_dim) updated node features
    """

    def __init__(self, cfg: HetGATConfig, in_dim: int, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = out_dim // cfg.n_heads
        assert out_dim % cfg.n_heads == 0
        # Per-node-type input projection
        self.in_proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(cfg.n_node_types)
        ])
        # Edge-type-conditional attention weights:
        #   e_ij = LeakyReLU(a^T_t [Wq * h_i || Wk * h_j])  where t = edge_type
        self.attn_a = nn.Parameter(
            torch.empty(cfg.n_edge_types, cfg.n_heads, 2 * self.head_dim)
        )
        nn.init.xavier_uniform_(self.attn_a)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)
        # Per-type linear projection
        h = torch.zeros(N, self.n_heads * self.head_dim, device=x.device, dtype=x.dtype)
        for t in range(self.cfg.n_node_types):
            mask = node_type == t
            if mask.any():
                h[mask] = self.in_proj[t](x[mask])
        h = h.view(N, self.n_heads, self.head_dim)

        if edge_index.size(1) == 0:
            # No edges — return self-loop projection
            return h.reshape(N, -1)

        src, dst = edge_index[0], edge_index[1]
        # Build attention-input: [h_dst || h_src] selected per-edge
        h_src = h[src]  # (E, n_heads, head_dim)
        h_dst = h[dst]  # (E, n_heads, head_dim)
        cat = torch.cat([h_dst, h_src], dim=-1)  # (E, n_heads, 2*head_dim)

        # Per-edge-type attention vector: a_t (n_heads, 2*head_dim)
        a_per_edge = self.attn_a[edge_type]  # (E, n_heads, 2*head_dim)
        # e_ij = sum over last dim of a * cat
        scores = (a_per_edge * cat).sum(dim=-1)  # (E, n_heads)
        scores = self.act(scores)

        # Softmax over incoming edges per dst node, per head
        # Group-softmax via index_add stabilization
        scores_max = torch.full((N, self.n_heads), float("-inf"),
                                 device=scores.device, dtype=scores.dtype)
        scores_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, self.n_heads),
                                    scores, reduce="amax", include_self=False)
        scores_max = torch.where(torch.isinf(scores_max),
                                  torch.zeros_like(scores_max), scores_max)
        scores_shifted = scores - scores_max[dst]
        exp_scores = torch.exp(scores_shifted)
        denom = torch.zeros_like(scores_max)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.n_heads), exp_scores)
        denom = denom.clamp(min=1e-12)
        alpha = exp_scores / denom[dst]
        alpha = self.dropout(alpha)

        # Aggregate: out[i] = sum over neighbors j of alpha_ij * h_j
        # h_src is shape (E, n_heads, head_dim); alpha is (E, n_heads).
        weighted = h_src * alpha.unsqueeze(-1)  # (E, n_heads, head_dim)
        out = torch.zeros_like(h)
        out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim),
            weighted,
        )
        return out.reshape(N, -1)


# ---------------------------------------------------------------------
# Full Het-GAT with temporal GRU
# ---------------------------------------------------------------------

class HetTemporalGAT(nn.Module):
    """Heterogeneous GAT × temporal GRU cascade predictor."""

    def __init__(self, cfg: HetGATConfig | None = None):
        super().__init__()
        self.cfg = cfg or HetGATConfig()
        # Stack of het-GAT layers
        layers: list[nn.Module] = []
        d_in = self.cfg.in_dim
        for li in range(self.cfg.n_layers):
            d_out = self.cfg.hidden_dim if li < self.cfg.n_layers - 1 else self.cfg.out_dim
            layers.append(HetGATLayer(self.cfg, in_dim=d_in, out_dim=d_out))
            d_in = d_out
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(self.cfg.out_dim)

        # Temporal GRU cell — fuses prev_h with current_h
        if self.cfg.use_temporal_gru:
            self.gru = nn.GRUCell(self.cfg.out_dim, self.cfg.out_dim)
        else:
            self.gru = None

        # Output head: scalar per-node "expected disruption magnitude"
        self.disruption_head = nn.Linear(self.cfg.out_dim, 1)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        prev_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (per_node_disruption_score, new_hidden_state)."""
        h = node_feats
        for li, layer in enumerate(self.layers):
            h = layer(h, node_types, edge_index, edge_types)
            if li < len(self.layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.cfg.dropout, training=self.training)

        h = self.layer_norm(h)

        if self.gru is not None:
            if prev_h is None:
                prev_h = torch.zeros_like(h)
            h = self.gru(h, prev_h)

        disruption = self.disruption_head(h).squeeze(-1)  # (N,)
        return disruption, h

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------
# Loader: convert server/data/graphs/*.json into Het-GAT inputs
# ---------------------------------------------------------------------

NODE_TYPES = {
    "PORT": 0, "PORT_PRIMARY": 0, "PORT_BACKUP": 0,
    "WAREHOUSE": 1, "WAREHOUSE_PRIMARY": 1, "WH": 1,
    "SUPPLIER": 2, "SUPPLIER_PRIMARY": 2, "SUPPLIER_BACKUP": 2,
    "CUSTOMER": 3, "RETAILER": 3, "FACTORY": 3,
}
EDGE_TYPES = {
    "SHIPS_TO": 0, "SUPPLIES": 1, "ROUTES_VIA": 2, "ALTERNATE_TO": 3,
}


def _classify_node(node: dict) -> int:
    raw = (node.get("type") or node.get("node_type") or node.get("kind") or "").upper()
    if raw in NODE_TYPES:
        return NODE_TYPES[raw]
    nid = str(node.get("id") or "").upper()
    for prefix, t in [("PORT", 0), ("WH", 1), ("WAREHOUSE", 1),
                       ("SUPPLIER", 2), ("RETAILER", 3), ("FACTORY", 3),
                       ("CUSTOMER", 3)]:
        if prefix in nid:
            return t
    return 0


def _classify_edge(edge: dict) -> int:
    raw = (edge.get("type") or edge.get("edge_type") or "SHIPS_TO").upper()
    return EDGE_TYPES.get(raw, 0)


def graph_json_to_tensors(path: Path,
                          *, in_dim: int = 16) -> tuple[torch.Tensor, torch.Tensor,
                                                         torch.Tensor, torch.Tensor,
                                                         list[str]]:
    """Load a server/data/graphs/*.json into Het-GAT input tensors.

    Returns (node_feats, node_types, edge_index, edge_types, node_id_list).
    """
    g = json.loads(path.read_text(encoding="utf-8"))
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    id_to_idx = {n.get("id"): i for i, n in enumerate(nodes)}

    # Per-node feature vector (in_dim=16): synthetic but DETERMINISTIC
    # based on real fields — capacity, tier, geographic coords, etc.
    node_feats = torch.zeros(len(nodes), in_dim, dtype=torch.float32)
    node_types = torch.zeros(len(nodes), dtype=torch.long)
    node_ids: list[str] = []
    for i, n in enumerate(nodes):
        node_ids.append(n.get("id", f"node_{i}"))
        node_types[i] = _classify_node(n)
        # Pack real fields into the 16-dim feature vector
        node_feats[i, 0] = float(n.get("capacity", 1.0)) / 100.0
        node_feats[i, 1] = float(n.get("tier", 1))
        node_feats[i, 2] = float(n.get("latitude") or n.get("lat") or 0) / 90.0
        node_feats[i, 3] = float(n.get("longitude") or n.get("lon") or 0) / 180.0
        node_feats[i, 4] = float(n.get("storage_days", 7)) / 30.0
        node_feats[i, 5] = float(n.get("cost_per_unit", 100)) / 1000.0
        # 6-15: hash-derived stable embedding from node id
        nid = str(n.get("id", ""))
        for k in range(min(10, len(nid))):
            node_feats[i, 6 + k] = (ord(nid[k]) % 100) / 100.0

    # Edges
    src_list, dst_list, type_list = [], [], []
    for e in edges:
        s = id_to_idx.get(e.get("source"))
        d = id_to_idx.get(e.get("target"))
        if s is None or d is None: continue
        src_list.append(s)
        dst_list.append(d)
        type_list.append(_classify_edge(e))
    if not src_list:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_types = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_types = torch.tensor(type_list, dtype=torch.long)
    return node_feats, node_types, edge_index, edge_types, node_ids


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

def HET_GAT_smoke_test(graph_path: Path | None = None) -> dict:
    """Forward pass on a real supply-chain graph + temporal rollout."""
    if graph_path is None:
        for cand in (REPO_ROOT / "server" / "data" / "graphs").glob("*.json"):
            graph_path = cand; break
    if graph_path is None:
        return {"error": "no graph json found"}

    feats, types, edges, etypes, ids = graph_json_to_tensors(graph_path)
    cfg = HetGATConfig()
    model = HetTemporalGAT(cfg)
    model.eval()

    # 5-day rollout simulating a cascading shock
    h_t = None
    history: list[list[float]] = []
    with torch.no_grad():
        for day in range(5):
            disruption, h_t = model(feats, types, edges, etypes, prev_h=h_t)
            history.append([round(float(d), 4) for d in disruption.tolist()])

    return {
        "graph": str(graph_path.relative_to(REPO_ROOT)),
        "n_nodes": int(feats.size(0)),
        "n_edges": int(edges.size(1)),
        "node_id_sample": ids[:5],
        "node_type_distribution": {
            k: int((types == v).sum())
            for k, v in {"PORT": 0, "WH": 1, "SUPPLIER": 2, "CUSTOMER": 3}.items()
        },
        "edge_type_distribution": {
            k: int((etypes == v).sum())
            for k, v in EDGE_TYPES.items()
        },
        "n_parameters": model.n_parameters(),
        "5_day_disruption_per_node": history,
        "config": cfg.__dict__,
    }


if __name__ == "__main__":
    import json as _json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = HET_GAT_smoke_test()
    print(_json.dumps(out, indent=2))
