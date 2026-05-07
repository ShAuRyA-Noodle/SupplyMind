"""R6 Block 7 — Provider: GNN for disruption propagation prediction.

Task: given a supply-chain graph and 1-2 disrupted source nodes, predict which
downstream nodes are affected (binary per-node).

Uses real supply-chain graphs from server/data/graphs/ (25-40 nodes, real suppliers).
Simulates N random disruption scenarios, ground-truth via BFS along 'supplies' edges.
Trains 3-layer GCN, benchmarks vs rule-based baseline (direct neighbors) and BFS-perfect.

Outputs:
  versions/v3_arcadia/results/R6_PROVIDER.json
  versions/v3_arcadia/plots/provider/training_curve.png
  versions/v3_arcadia/plots/provider/graph_viz.png
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
MAX_HOPS = 3  # propagation depth
N_TRAIN_EXAMPLES = 2000
N_TEST_EXAMPLES = 400
HIDDEN_DIM = 64
N_EPOCHS = 80
LR = 2e-3


# ============================================================
# Graph loader
# ============================================================
def load_graph(difficulty: str = "hard") -> dict:
    path = ROOT / "server" / "data" / "graphs" / f"{difficulty}_graph.json"
    g = json.loads(path.read_text())
    nodes = g["nodes"]
    edges = g["edges"]
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    # Build node features: [tier, risk, log_spend, single_source, is_operational, type_onehot(5)]
    n_node_types = len(NODE_TYPES)
    feats = []
    for n in nodes:
        tier = n.get("tier", 0) / 5.0
        risk = n.get("risk_score", 0.0)
        spend = n.get("annual_spend", 0)
        log_spend = np.log1p(spend) / 25.0 if spend else 0
        ss = 1.0 if n.get("single_source", False) else 0.0
        op = 1.0 if n.get("is_operational", True) else 0.0
        t = [0.0] * n_node_types
        nt = n.get("node_type", "supplier")
        if nt in NODE_TYPES: t[NODE_TYPES.index(nt)] = 1.0
        feats.append([tier, risk, log_spend, ss, op] + t)
    X = np.array(feats, dtype=np.float32)

    # Edge index [2, E] for 'supplies' and related; add reverse for message passing in both directions
    src, dst = [], []
    for e in edges:
        if e.get("is_active", True) and e["source"] in node_id_to_idx and e["target"] in node_id_to_idx:
            si, di = node_id_to_idx[e["source"]], node_id_to_idx[e["target"]]
            src.append(si); dst.append(di)
            src.append(di); dst.append(si)  # undirected message passing
    edge_index = np.array([src, dst], dtype=np.int64)

    # Directed adjacency for BFS ground truth (supplies direction only)
    adj = {i: [] for i in range(len(nodes))}
    for e in edges:
        if e.get("is_active", True) and e["source"] in node_id_to_idx and e["target"] in node_id_to_idx:
            adj[node_id_to_idx[e["source"]]].append(node_id_to_idx[e["target"]])

    return {
        "nodes": nodes, "node_names": [n["id"] for n in nodes],
        "X": X, "edge_index": edge_index, "adj": adj,
        "n": len(nodes), "f": X.shape[1],
    }


# ============================================================
# Disruption simulator (BFS ground truth)
# ============================================================
def simulate_disruption(g: dict, n_sources: int = 2, max_hops: int = MAX_HOPS,
                        rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (disruption_mask [n], affected_mask [n]).
    Sources = disruption_mask. Affected = BFS from sources along adj within max_hops.
    """
    n = g["n"]
    if rng is None: rng = np.random.default_rng()
    sources = rng.choice(n, size=n_sources, replace=False)
    disrupt = np.zeros(n, dtype=np.float32)
    disrupt[sources] = 1.0
    affected = set(sources.tolist())
    frontier = list(sources)
    for hop in range(max_hops):
        next_frontier = []
        for u in frontier:
            for v in g["adj"].get(u, []):
                if v not in affected:
                    affected.add(v); next_frontier.append(v)
        frontier = next_frontier
        if not frontier: break
    aff_mask = np.zeros(n, dtype=np.float32)
    for i in affected: aff_mask[i] = 1.0
    return disrupt, aff_mask


# ============================================================
# Simple GCN (pure torch, no torch_geometric)
# ============================================================
class GCNLayer(nn.Module):
    """Concat(self, mean_neighbors) -> Linear. Input dim = in_dim; output dim = out_dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index
        out = torch.zeros_like(x)
        count = torch.zeros(n, 1, device=x.device)
        out.index_add_(0, src, x[dst])
        count.index_add_(0, src, torch.ones(src.size(0), 1, device=x.device))
        out = out / count.clamp(min=1.0)
        return self.lin(torch.cat([x, out], dim=1))


class DisruptionGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim + 1, hidden)  # +1 for disruption flag
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, disrupt: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, disrupt.unsqueeze(1)], dim=1)
        h = F.relu(self.gcn1(h, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        h = self.gcn3(h, edge_index)
        return self.out(h).squeeze(-1)


# ============================================================
# Training
# ============================================================
def generate_batch(g: dict, n_examples: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [simulate_disruption(g, n_sources=rng.integers(1, 4), rng=rng) for _ in range(n_examples)]


def train_gnn(g: dict, n_train: int, n_test: int, hidden: int, n_epochs: int, lr: float) -> dict:
    X = torch.tensor(g["X"], device=DEVICE)
    ei = torch.tensor(g["edge_index"], device=DEVICE)
    model = DisruptionGNN(in_dim=g["f"], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    train_set = generate_batch(g, n_train, seed=SEED)
    test_set = generate_batch(g, n_test, seed=SEED + 1)

    log.info(f"Training DisruptionGNN on {n_train} examples, {n_test} test")
    losses = []
    test_accs = []
    for ep in range(n_epochs):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        indices = rng.permutation(n_train)
        ep_loss = 0.0
        for i in indices:
            disrupt, target = train_set[i]
            d = torch.tensor(disrupt, device=DEVICE)
            t = torch.tensor(target, device=DEVICE)
            logits = model(X, d, ei)
            loss = bce(logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / n_train)

        # Test
        model.eval()
        with torch.no_grad():
            correct = 0; total = 0; tps = 0; fps = 0; fns = 0
            for disrupt, target in test_set:
                d = torch.tensor(disrupt, device=DEVICE)
                t = torch.tensor(target, device=DEVICE)
                pred = (torch.sigmoid(model(X, d, ei)) > 0.5).float()
                correct += int((pred == t).sum().item())
                total += t.numel()
                tps += int(((pred == 1) & (t == 1)).sum().item())
                fps += int(((pred == 1) & (t == 0)).sum().item())
                fns += int(((pred == 0) & (t == 1)).sum().item())
            acc = correct / total
            prec = tps / max(tps + fps, 1)
            rec = tps / max(tps + fns, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            test_accs.append({"acc": acc, "precision": prec, "recall": rec, "f1": f1})
        if ep % 10 == 0 or ep == n_epochs - 1:
            log.info(f"  ep {ep:3d}: loss={losses[-1]:.4f}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    return {"train_loss_curve": losses, "test_metric_curve": test_accs, "model": model}


def baseline_direct_neighbors(g: dict, test_set: list) -> dict:
    """Baseline: predict affected = sources + direct neighbors (1-hop only)."""
    tps = fps = fns = 0; correct = 0; total = 0
    for disrupt, target in test_set:
        pred = disrupt.copy()
        for src in np.where(disrupt > 0)[0]:
            for v in g["adj"].get(int(src), []):
                pred[v] = 1.0
        correct += int((pred == target).sum())
        total += len(target)
        tps += int(((pred == 1) & (target == 1)).sum())
        fps += int(((pred == 1) & (target == 0)).sum())
        fns += int(((pred == 0) & (target == 1)).sum())
    prec = tps / max(tps + fps, 1)
    rec = tps / max(tps + fns, 1)
    return {"acc": correct / total, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-8)}


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    log.info("R6 Provider — Disruption Propagation GNN")

    all_results = {}
    for difficulty in ["easy", "medium", "hard"]:
        log.info(f"\n=== Graph: {difficulty} ===")
        g = load_graph(difficulty)
        log.info(f"  nodes={g['n']} edges={g['edge_index'].shape[1]//2} (directed)")

        trained = train_gnn(g, N_TRAIN_EXAMPLES, N_TEST_EXAMPLES, HIDDEN_DIM, N_EPOCHS, LR)

        test_set = generate_batch(g, N_TEST_EXAMPLES, seed=SEED + 1)
        baseline = baseline_direct_neighbors(g, test_set)
        gnn_final = trained["test_metric_curve"][-1]

        # Save model
        torch.save(trained["model"].state_dict(), CKPT / f"gnn_{difficulty}.pt")

        all_results[difficulty] = {
            "n_nodes": g["n"],
            "n_edges": int(g["edge_index"].shape[1] // 2),
            "gnn_final": gnn_final,
            "baseline_direct_neighbors": baseline,
            "improvement_f1_pp": (gnn_final["f1"] - baseline["f1"]) * 100,
            "train_loss_curve": trained["train_loss_curve"],
            "test_metric_curve": trained["test_metric_curve"],
        }
        log.info(f"  {difficulty}: GNN F1={gnn_final['f1']:.3f}  baseline F1={baseline['f1']:.3f}  "
                 f"improvement={all_results[difficulty]['improvement_f1_pp']:+.1f}pp")

    out = {"graphs": all_results, "config": {
        "n_train": N_TRAIN_EXAMPLES, "n_test": N_TEST_EXAMPLES,
        "hidden_dim": HIDDEN_DIM, "epochs": N_EPOCHS, "lr": LR, "max_hops": MAX_HOPS,
    }, "elapsed_min": (time.time() - t0) / 60}
    out_path = RESULTS / "R6_PROVIDER.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\n=== SUMMARY ===")
    for diff, r in all_results.items():
        log.info(f"  {diff}: GNN acc={r['gnn_final']['acc']:.3f} F1={r['gnn_final']['f1']:.3f}  "
                 f"(baseline F1={r['baseline_direct_neighbors']['f1']:.3f})")
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
