"""train_hetgat.py — train HetTemporalGAT on the R6_PROVIDER cascade task.

Replicates the exact "arrival_time_regression" task from R6 (predict expected
disruption arrival time per node given noisy per-edge lead-times) on each
of easy/medium/hard supply-chain graphs, then reports HetGAT MAE
head-to-head against the published v1 GCN MAE in R6_PROVIDER_V2.json.

Usage:
    python -m versions.v5_phoenix.gnn_v2.train_hetgat --graph easy --epochs 200
    python -m versions.v5_phoenix.gnn_v2.train_hetgat --all
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .het_temporal_gat import (HetGATConfig, HetTemporalGAT,
                                 graph_json_to_tensors)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DIR = REPO_ROOT / "server" / "data" / "graphs"
R6_RESULTS = REPO_ROOT / "v3_arcadia" / "results" / "R6_PROVIDER_V2.json"
OUT_DIR = REPO_ROOT / "versions/v5_phoenix" / "experiments" / "hetgat_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def synthesize_arrival_dataset(
    feats: torch.Tensor,
    edge_index: torch.Tensor,
    n_samples: int = 256,
    noise_sigma_rel: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate the R6_PROVIDER arrival-time-regression synthetic task.

    For each sample, we:
      1. Draw a "true" per-edge lead-time t_e ~ Uniform[1, 10] days
      2. Compute the per-node arrival time = shortest-path distance from
         a designated source node (node 0) using true lead-times
      3. Add per-edge Gaussian noise (sigma = noise_sigma_rel * t_e) to
         get the noisy observed lead-times — these become the input edge
         features (encoded into node_feats[5] = mean incoming noisy lead).
      4. The supervised target is the noise-free arrival time per node.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    N = feats.size(0)
    src_nodes, dst_nodes = edge_index[0].tolist(), edge_index[1].tolist()
    E = len(src_nodes)

    samples_in: list[torch.Tensor] = []
    samples_target: list[torch.Tensor] = []

    # Treat node 0 as the disruption source for all samples
    source = 0

    for s in range(n_samples):
        # 1. True edge times
        true_times = rng.uniform(1.0, 10.0, size=E).astype(np.float32)
        # 2. Shortest-path arrival times (Dijkstra-lite)
        dist = np.full(N, np.inf, dtype=np.float32)
        dist[source] = 0.0
        # Simple relaxation rounds (graph is small)
        for _ in range(N):
            updated = False
            for k, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
                if dist[u] + true_times[k] < dist[v]:
                    dist[v] = dist[u] + true_times[k]
                    updated = True
            if not updated:
                break
        dist[np.isinf(dist)] = 0.0  # unreachable -> 0 (test-time fallback)

        # 3. Noisy observed edge times (input perturbation)
        noisy_times = true_times + rng.normal(0, noise_sigma_rel * true_times)
        noisy_times = np.maximum(0.1, noisy_times)

        # 4. Encode noisy edge means into node_feats[5] (cost_per_unit slot)
        node_feats_s = feats.clone()
        # Compute mean incoming noisy lead per node
        incoming_sum = np.zeros(N, dtype=np.float32)
        incoming_n = np.zeros(N, dtype=np.float32)
        for k, dst in enumerate(dst_nodes):
            incoming_sum[dst] += noisy_times[k]
            incoming_n[dst] += 1
        mean_incoming = incoming_sum / np.maximum(1.0, incoming_n)
        node_feats_s[:, 5] = torch.from_numpy(mean_incoming / 10.0)
        # Also encode source-distance (1-hop estimate) into slot[6]
        is_source = torch.zeros(N, dtype=torch.float32)
        is_source[source] = 1.0
        node_feats_s[:, 6] = is_source

        samples_in.append(node_feats_s)
        samples_target.append(torch.from_numpy(dist.astype(np.float32)))

    X = torch.stack(samples_in)         # (n_samples, N, in_dim)
    Y = torch.stack(samples_target)     # (n_samples, N)
    return X, Y


def train_one_graph(
    graph_path: Path,
    *,
    epochs: int = 200,
    n_train: int = 256,
    n_test: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train HetGAT on the arrival-time task for one graph; report test MAE."""
    feats, types, edges, etypes, ids = graph_json_to_tensors(graph_path)
    N = feats.size(0)
    E = edges.size(1)
    logger.info("[hetgat:%s] %d nodes, %d edges", graph_path.stem, N, E)

    if E == 0:
        return {"graph": graph_path.stem, "skipped": "no_edges"}

    Xtr, Ytr = synthesize_arrival_dataset(feats, edges, n_samples=n_train, seed=seed)
    Xte, Yte = synthesize_arrival_dataset(feats, edges, n_samples=n_test, seed=seed + 999)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xte, Yte = Xte.to(device), Yte.to(device)
    types_d = types.to(device)
    edges_d = edges.to(device)
    etypes_d = etypes.to(device)

    cfg = HetGATConfig(in_dim=16, hidden_dim=64, out_dim=32, n_layers=2,
                        n_heads=4, dropout=0.15)
    model = HetTemporalGAT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history: list[dict] = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        # Mini-batch over samples
        perm = torch.randperm(Xtr.size(0))
        epoch_loss = 0.0
        for i in range(0, Xtr.size(0), 32):
            batch_idx = perm[i:i + 32]
            losses_b = []
            for j in batch_idx:
                pred, _ = model(Xtr[j], types_d, edges_d, etypes_d)
                # Mask unreachable targets (Y==0 except source itself)
                mask = (Ytr[j] > 0) | (torch.arange(N, device=device) == 0)
                loss = F.smooth_l1_loss(pred[mask], Ytr[j][mask])
                losses_b.append(loss)
            batch_loss = torch.stack(losses_b).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(batch_loss.item())
        epoch_loss /= max(1, (Xtr.size(0) // 32))

        # Eval
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_mae = 0.0
                for j in range(Xte.size(0)):
                    pred, _ = model(Xte[j], types_d, edges_d, etypes_d)
                    mask = (Yte[j] > 0) | (torch.arange(N, device=device) == 0)
                    test_mae += float(F.l1_loss(pred[mask], Yte[j][mask]).item())
                test_mae /= max(1, Xte.size(0))
            history.append({"epoch": ep, "train_loss": epoch_loss,
                             "test_mae": test_mae,
                             "elapsed_s": round(time.time() - t0, 2)})
            logger.info("[hetgat:%s] ep %d  train_loss=%.4f  test_mae=%.4f",
                        graph_path.stem, ep, epoch_loss, test_mae)

    final_test_mae = history[-1]["test_mae"] if history else float("nan")

    # Compare to R6 baseline
    r6_baseline = None
    if R6_RESULTS.exists():
        r6 = json.loads(R6_RESULTS.read_text(encoding="utf-8"))
        gid = graph_path.stem.replace("_graph", "")
        if gid in r6.get("graphs", {}):
            r6_baseline = r6["graphs"][gid]

    out = {
        "graph": graph_path.stem,
        "n_nodes": N, "n_edges": E,
        "epochs": epochs, "n_train": n_train, "n_test": n_test,
        "n_parameters": model.n_parameters(),
        "hetgat_test_mae_final": final_test_mae,
        "elapsed_s": round(time.time() - t0, 2),
        "history": history,
    }
    if r6_baseline:
        out["r6_v1_gcn_baseline"] = {
            "gnn_mae": r6_baseline.get("gnn_mae"),
            "mlp_mae": r6_baseline.get("mlp_mae"),
            "one_hop_mae": r6_baseline.get("one_hop_mean_mae"),
            "v1_improvement_vs_mlp_pct": r6_baseline.get("improvement_vs_mlp_pct"),
        }
        v1_mae = r6_baseline.get("gnn_mae")
        if v1_mae and v1_mae > 0:
            out["hetgat_vs_v1_gcn_pct"] = round(
                100.0 * (v1_mae - final_test_mae) / v1_mae, 2)

    # Save weights
    weight_path = OUT_DIR / f"hetgat_{graph_path.stem}.pt"
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "result": out}, weight_path)
    out["weights_path"] = str(weight_path)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="easy",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.graph == "all":
        graphs = ["easy_graph", "medium_graph", "hard_graph"]
    else:
        graphs = [f"{args.graph}_graph"]

    all_results: list[dict] = []
    for gname in graphs:
        gpath = GRAPH_DIR / f"{gname}.json"
        if not gpath.exists():
            logger.warning("[hetgat] graph not found: %s", gpath)
            continue
        result = train_one_graph(gpath, epochs=args.epochs,
                                   n_train=args.n_train, n_test=args.n_test)
        all_results.append(result)

    # Aggregate report
    report_path = OUT_DIR / "report.json"
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_graphs_trained": len(all_results),
        "results_per_graph": all_results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str),
                            encoding="utf-8")
    logger.info("[hetgat] report saved to %s", report_path)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
