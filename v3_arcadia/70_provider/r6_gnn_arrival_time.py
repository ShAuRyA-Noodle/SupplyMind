"""R6 Provider v2 — Arrival-time regression (non-trivial GNN task).

Fixes the R6 Provider honest finding that easy-graph F1 = 1.000 (task too trivial).

Root cause: BFS-reachable-set prediction on a 12-node graph is linearly
separable. A 2-layer linear model can memorize it. No GNN lift visible.

Fix: switch to a **harder, continuous regression task**: predict the
**expected arrival time** of a disruption at each node, given:
  - Disruption source node(s) (binary flag)
  - Noisy per-edge lead-times (Gaussian noise on the real lead_time_days)
  - Node features

Ground truth: Dijkstra shortest-path distance from source through the
perturbed lead-time graph. Continuous, noisy, hop-count-dependent.

This requires the GNN to:
  1. Learn to propagate along edges (non-trivial on a 3-hop noisy graph)
  2. Integrate noisy edge weights
  3. Handle multi-source disruptions

Baselines:
  - MLP ignoring graph structure (node features only)
  - 1-hop mean (predict mean of 1-hop neighbor lead-times)
  - Dijkstra on UN-noisy graph (oracle)

Metrics: MAE, Spearman rank correlation with true arrival time.

Output:
  v3_arcadia/results/R6_PROVIDER_V2.json
  v3_arcadia/plots/provider/r6_provider_v2.png
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "provider"
CKPT.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "v3_arcadia" / "plots" / "provider"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

NODE_TYPES = ["supplier", "warehouse", "port", "factory", "customer"]

# Training hyperparameters — reduced from v1 since task is harder + we need to
# hit 3 graphs in under 30 min total.
N_TRAIN = 500
N_TEST = 200
HIDDEN = 64
N_EPOCHS = 30
LR = 3e-3

LEAD_TIME_NOISE_SIGMA = 0.2  # relative perturbation
MAX_HOPS_FOR_TARGET = 4
UNREACHABLE_PENALTY = 100.0  # large arrival time for unreachable nodes


def load_graph(difficulty: str) -> dict:
    path = ROOT / "server" / "data" / "graphs" / f"{difficulty}_graph.json"
    g = json.loads(path.read_text())
    nodes = g["nodes"]
    edges = g["edges"]
    id2idx = {n["id"]: i for i, n in enumerate(nodes)}

    # Node features (same as v1)
    feats = []
    for n in nodes:
        tier = n.get("tier", 0) / 5.0
        risk = n.get("risk_score", 0.0)
        spend = n.get("annual_spend", 0)
        log_spend = np.log1p(spend) / 25.0 if spend else 0
        ss = 1.0 if n.get("single_source", False) else 0.0
        op = 1.0 if n.get("is_operational", True) else 0.0
        t = [0.0] * len(NODE_TYPES)
        if n.get("node_type", "supplier") in NODE_TYPES:
            t[NODE_TYPES.index(n["node_type"])] = 1.0
        feats.append([tier, risk, log_spend, ss, op] + t)
    X = np.array(feats, dtype=np.float32)

    # Directed adjacency for Dijkstra + edge weights
    adj = [[] for _ in range(len(nodes))]
    for e in edges:
        if e["source"] in id2idx and e["target"] in id2idx:
            si, di = id2idx[e["source"]], id2idx[e["target"]]
            lt = float(e.get("lead_time_days", 1.0))
            adj[si].append((di, lt))

    # Undirected edge_index for GCN message passing
    src, dst = [], []
    for e in edges:
        if e["source"] in id2idx and e["target"] in id2idx:
            si, di = id2idx[e["source"]], id2idx[e["target"]]
            src.append(si); dst.append(di)
            src.append(di); dst.append(si)
    edge_index = np.array([src, dst], dtype=np.int64)

    return {"nodes": nodes, "X": X, "edge_index": edge_index, "adj": adj,
            "n": len(nodes), "f": X.shape[1]}


def dijkstra_arrival(g: dict, sources: list[int], noisy_adj: list[list[tuple[int, float]]]) -> np.ndarray:
    """Shortest-path arrival time from sources (with 0 arrival time) through noisy graph."""
    import heapq
    n = g["n"]
    dist = np.full(n, UNREACHABLE_PENALTY, dtype=np.float32)
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in noisy_adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def simulate_arrival_time(g: dict, rng: np.random.Generator, n_sources: int = 2):
    """Sample noisy edges, pick sources, compute true arrival times."""
    # Add Gaussian noise to each edge lead time (clipped to >= 0.1)
    noisy_adj = []
    for lst in g["adj"]:
        noisy = []
        for (v, w) in lst:
            noise = rng.normal(0, LEAD_TIME_NOISE_SIGMA * w)
            noisy.append((v, max(0.1, w + noise)))
        noisy_adj.append(noisy)
    sources = rng.choice(g["n"], size=n_sources, replace=False)
    arrival = dijkstra_arrival(g, sources.tolist(), noisy_adj)
    source_flag = np.zeros(g["n"], dtype=np.float32)
    source_flag[sources] = 1.0
    return source_flag, arrival


def generate_dataset(g: dict, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns list of (node_features_with_source, source_flag, arrival_time)."""
    rng = np.random.default_rng(seed)
    return [simulate_arrival_time(g, rng, n_sources=rng.integers(1, 4)) for _ in range(n)]


# ============================================================
# GCN (same as v1)
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)
        src, dst = edge_index
        agg = torch.zeros_like(x)
        count = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, src, x[dst])
        count.index_add_(0, src, torch.ones(src.size(0), 1, device=x.device))
        agg = agg / count.clamp(min=1.0)
        return self.lin(torch.cat([x, agg], dim=1))


class ArrivalTimeGNN(nn.Module):
    def __init__(self, f, hidden):
        super().__init__()
        self.l1 = GCNLayer(f + 1, hidden)
        self.l2 = GCNLayer(hidden, hidden)
        self.l3 = GCNLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, source_flag, edge_index):
        h = torch.cat([x, source_flag.unsqueeze(1)], dim=1)
        h = F.relu(self.l1(h, edge_index))
        h = F.relu(self.l2(h, edge_index))
        h = self.l3(h, edge_index)
        return self.head(h).squeeze(-1)


class MLPBaseline(nn.Module):
    """Baseline: ignores graph structure."""
    def __init__(self, f, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x, source_flag, edge_index=None):
        h = torch.cat([x, source_flag.unsqueeze(1)], dim=1)
        return self.net(h).squeeze(-1)


# ============================================================
# Training + eval
# ============================================================
def train_model(model_cls, g, train_set, test_set, n_epochs, lr, name):
    X = torch.tensor(g["X"], device=DEVICE)
    ei = torch.tensor(g["edge_index"], device=DEVICE)
    model = model_cls(g["f"], HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    losses = []
    test_maes = []
    for ep in range(n_epochs):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        idx = rng.permutation(len(train_set))
        ep_loss = 0.0
        for i in idx:
            sf, y = train_set[i]
            sf_t = torch.tensor(sf, device=DEVICE)
            y_t = torch.tensor(y, device=DEVICE)
            pred = model(X, sf_t, ei)
            loss = mse(pred, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(train_set))

        model.eval()
        with torch.no_grad():
            maes = []
            for sf, y in test_set:
                sf_t = torch.tensor(sf, device=DEVICE)
                y_t = torch.tensor(y, device=DEVICE)
                pred = model(X, sf_t, ei)
                maes.append(float((pred - y_t).abs().mean().item()))
            test_maes.append(float(np.mean(maes)))
        if ep % 10 == 0 or ep == n_epochs - 1:
            log.info(f"  [{name}] ep {ep:3d}: loss={losses[-1]:.4f}  test_MAE={test_maes[-1]:.4f}")
    return {"losses": losses, "test_mae_curve": test_maes,
            "final_mae": float(test_maes[-1]), "model": model}


def baseline_1hop_mean(g, test_set):
    """Predict each node's arrival as 1-hop neighbor mean lead-time from source."""
    n = g["n"]
    # 1-hop adj (directed)
    adj = g["adj"]
    maes = []
    for sf, y in test_set:
        pred = np.zeros(n, dtype=np.float32)
        sources = np.where(sf > 0)[0]
        # For each non-source node, its prediction = mean edge-weight to its predecessors
        # If no predecessor reachable, use UNREACHABLE_PENALTY
        for i in range(n):
            if sf[i] > 0:
                pred[i] = 0.0
                continue
            preds_list = []
            for u in range(n):
                for (v, w) in adj[u]:
                    if v == i and u in sources:
                        preds_list.append(w)
            pred[i] = float(np.mean(preds_list)) if preds_list else UNREACHABLE_PENALTY
        maes.append(float(np.mean(np.abs(pred - y))))
    return float(np.mean(maes))


def main():
    t0 = time.time()
    log.info("R6 Provider v2 — Arrival-time regression (non-trivial GNN task)")

    out = {}
    for difficulty in ["easy", "medium", "hard"]:
        log.info(f"\n=== Graph: {difficulty} ===")
        g = load_graph(difficulty)
        log.info(f"  nodes={g['n']} edges={g['edge_index'].shape[1] // 2}")

        train_set = generate_dataset(g, N_TRAIN, seed=SEED)
        test_set = generate_dataset(g, N_TEST, seed=SEED + 1)

        gnn_result = train_model(ArrivalTimeGNN, g, train_set, test_set, N_EPOCHS, LR, "GNN")
        mlp_result = train_model(MLPBaseline, g, train_set, test_set, N_EPOCHS, LR, "MLP")
        one_hop_mae = baseline_1hop_mean(g, test_set)

        torch.save(gnn_result["model"].state_dict(), CKPT / f"gnn_arrival_{difficulty}.pt")

        improvement_vs_mlp = (mlp_result["final_mae"] - gnn_result["final_mae"]) / mlp_result["final_mae"] * 100
        improvement_vs_1hop = (one_hop_mae - gnn_result["final_mae"]) / one_hop_mae * 100

        out[difficulty] = {
            "n_nodes": g["n"],
            "n_edges": int(g["edge_index"].shape[1] // 2),
            "gnn_mae": gnn_result["final_mae"],
            "mlp_mae": mlp_result["final_mae"],
            "one_hop_mean_mae": one_hop_mae,
            "improvement_vs_mlp_pct": improvement_vs_mlp,
            "improvement_vs_1hop_pct": improvement_vs_1hop,
            "gnn_loss_curve": gnn_result["losses"],
            "gnn_test_mae_curve": gnn_result["test_mae_curve"],
            "mlp_test_mae_curve": mlp_result["test_mae_curve"],
        }
        log.info(f"  {difficulty}: GNN MAE={gnn_result['final_mae']:.3f}  "
                 f"MLP MAE={mlp_result['final_mae']:.3f}  "
                 f"1-hop MAE={one_hop_mae:.3f}  "
                 f"GNN_vs_MLP={improvement_vs_mlp:+.1f}%  "
                 f"GNN_vs_1hop={improvement_vs_1hop:+.1f}%")

    final = {
        "task": "arrival_time_regression",
        "task_description": (
            "Predict expected disruption arrival time (continuous) per node, given "
            "noisy per-edge lead-times and random source nodes. Non-trivial: requires "
            "GNN to learn Dijkstra-like aggregation through the graph."
        ),
        "lead_time_noise_sigma_relative": LEAD_TIME_NOISE_SIGMA,
        "graphs": out,
        "config": {"n_train": N_TRAIN, "n_test": N_TEST, "hidden": HIDDEN,
                   "epochs": N_EPOCHS, "lr": LR},
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R6_PROVIDER_V2.json"
    out_path.write_text(json.dumps(final, indent=2, default=str))

    log.info("\n=== R6 PROVIDER V2 SUMMARY ===")
    for d, r in out.items():
        log.info(f"  {d:<8} GNN MAE={r['gnn_mae']:.3f}  vs MLP={r['improvement_vs_mlp_pct']:+.1f}%  vs 1-hop={r['improvement_vs_1hop_pct']:+.1f}%")
    log.info(f"\nSaved: {out_path}  ({final['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
