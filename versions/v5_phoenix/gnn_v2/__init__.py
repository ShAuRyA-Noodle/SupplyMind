"""gnn_v2 — Heterogeneous Temporal Graph Attention Network.

Replaces the v1 3-layer GCN with edge-type-conditional attention +
temporal gated updates. Pass-7 C13.
"""
from .het_temporal_gat import HetTemporalGAT, HetGATConfig, HET_GAT_smoke_test

__all__ = ["HetTemporalGAT", "HetGATConfig", "HET_GAT_smoke_test"]
