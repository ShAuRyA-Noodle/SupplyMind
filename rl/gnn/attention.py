"""
GNN Attention Visualization + Link Prediction for SupplyMind.

Uses GATConv (Graph Attention Networks) to learn which supply chain
edges are most critical. Extracts attention weights for dashboard visualization.

Architecture:
    2 GATConv layers (4 heads -> 2 heads)
    Predictor: Linear(64->32)->ReLU->Linear(32->1)->Sigmoid -> failure_prob per node

If PyG is not installed, falls back to a pure-PyTorch MLP version.

NOTE: this module has no training entrypoint and no GAT checkpoint ships with
the repo. get_attention_edges() therefore requires a caller-supplied trained
model or a checkpoint on disk — it will NOT fabricate attention weights from an
untrained network. Until a GAT is trained, the dashboard attention feed is off.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
GAT_CHECKPOINT = CHECKPOINT_DIR / "gat_best.pt"

# Check if PyG is available
_HAS_PYG = False
try:
    import torch_geometric  # noqa: F401
    from torch_geometric.nn import GATConv
    _HAS_PYG = True
    logger.info("PyTorch Geometric available (v%s)", torch_geometric.__version__)
except ImportError:
    logger.info("PyTorch Geometric not installed — using MLP fallback")


class SupplyChainGAT(nn.Module):
    """Graph Attention Network for supply chain link prediction.

    Uses 2 GATConv layers when PyG is available, otherwise falls back
    to an MLP that treats the flattened adjacency as input.

    Args:
        node_features:  Per-node feature dimension (10).
        hidden_dim:     Hidden dimension after first GAT layer (32).
        n_heads_1:      Attention heads in first layer (4).
        n_heads_2:      Attention heads in second layer (2).
        out_dim:        Output embedding per node (32).
    """

    def __init__(
        self,
        node_features: int = 10,
        hidden_dim: int = 32,
        n_heads_1: int = 4,
        n_heads_2: int = 2,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.use_pyg = _HAS_PYG

        if self.use_pyg:
            self.conv1 = GATConv(node_features, hidden_dim, heads=n_heads_1, concat=True)
            self.conv2 = GATConv(hidden_dim * n_heads_1, out_dim, heads=n_heads_2, concat=False)
        else:
            # MLP fallback
            self.mlp = nn.Sequential(
                nn.Linear(node_features, hidden_dim * n_heads_1),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * n_heads_1, out_dim),
                nn.ReLU(inplace=True),
            )

        # Link predictor: node pair -> failure probability
        self.predictor = nn.Sequential(
            nn.Linear(out_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Node failure predictor
        self.node_predictor = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass.

        Args:
            x:          (n_nodes, node_features)
            edge_index: (2, n_edges) long tensor (PyG format), or None for MLP.

        Returns:
            node_embeddings: (n_nodes, out_dim)
            info: dict with attention weights if PyG available.
        """
        info = {}

        if self.use_pyg and edge_index is not None:
            h, (edge_idx_1, attn_1) = self.conv1(x, edge_index, return_attention_weights=True)
            h = F.elu(h)
            h, (edge_idx_2, attn_2) = self.conv2(h, edge_index, return_attention_weights=True)
            info["attention_weights_l1"] = attn_1.detach()
            info["attention_weights_l2"] = attn_2.detach()
            info["edge_index_l1"] = edge_idx_1.detach()
        else:
            h = self.mlp(x)

        return h, info

    def predict_link_failure(
        self, embeddings: torch.Tensor, src: int, dst: int,
    ) -> float:
        """Predict failure probability for a specific link."""
        pair = torch.cat([embeddings[src], embeddings[dst]])
        return self.predictor(pair.unsqueeze(0)).item()

    def predict_node_failures(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict failure probability for all nodes."""
        return self.node_predictor(embeddings).squeeze(-1)


def load_graph_as_tensors(graph_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load supply chain graph JSON into PyTorch tensors.

    Returns:
        x:          (n_nodes, 10) node feature matrix.
        edge_index: (2, n_edges) edge index.
        node_ids:   List of node ID strings.
    """
    with open(graph_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Node features (10 per node, matching gym_env encoding)
    node_type_map = {"supplier": 0, "warehouse": 1, "port": 2, "factory": 3, "customer": 4}
    x = np.zeros((len(nodes), 10), dtype=np.float32)
    for i, n in enumerate(nodes):
        x[i, 0] = 1.0 if n.get("is_operational", True) else 0.0
        x[i, 1] = n.get("risk_score", 0.0)
        x[i, 2] = 0.5  # inventory normalized placeholder
        x[i, 3] = 1.0 if n.get("backup_supplier_ids") else 0.0
        nt = node_type_map.get(n.get("node_type", ""), 0)
        x[i, 4 + nt] = 1.0
        x[i, 9] = n.get("annual_spend", 0) / 1e10  # revenue normalized

    # Edges
    edges = data.get("edges", [])
    src_list, dst_list = [], []
    for e in edges:
        s = id_to_idx.get(e["source"])
        d = id_to_idx.get(e["target"])
        if s is not None and d is not None:
            src_list.append(s)
            dst_list.append(d)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    return torch.from_numpy(x), edge_index, node_ids


def get_attention_edges(
    graph_path: str | Path,
    model: SupplyChainGAT | None = None,
) -> list[dict[str, Any]]:
    """Get edge attention weights for dashboard visualization.

    Returns list of {source, target, attention_weight} for rendering
    edge thickness proportional to attention.

    Requires a TRAINED model: either passed in via `model`, or loaded from
    GAT_CHECKPOINT on disk. Raises RuntimeError if neither is available —
    an untrained randomly-initialized GAT's attention is meaningless noise and
    must never be served as edge criticality.
    """
    x, edge_index, node_ids = load_graph_as_tensors(graph_path)

    if model is None:
        if not GAT_CHECKPOINT.exists():
            raise RuntimeError(
                f"no trained GAT checkpoint (looked for {GAT_CHECKPOINT}). "
                "Refusing to serve attention weights from an untrained network. "
                "Train and save a SupplyChainGAT, or pass a trained `model`."
            )
        model = SupplyChainGAT()
        from rl.checkpoint_security import safe_load
        state = safe_load(GAT_CHECKPOINT, map_location="cpu")
        model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)

    model.eval()
    with torch.no_grad():
        embeddings, info = model(x, edge_index)

    # Extract attention weights
    edges = []
    if "attention_weights_l2" in info:
        attn = info["attention_weights_l2"].numpy()
        eidx = edge_index.numpy()
        for i in range(eidx.shape[1]):
            edges.append({
                "source": node_ids[eidx[0, i]],
                "target": node_ids[eidx[1, i]],
                "attention_weight": float(attn[i].mean()) if i < len(attn) else 0.5,
            })
    else:
        # Fallback: uniform attention
        eidx = edge_index.numpy()
        for i in range(eidx.shape[1]):
            edges.append({
                "source": node_ids[eidx[0, i]],
                "target": node_ids[eidx[1, i]],
                "attention_weight": 0.5,
            })

    return edges
