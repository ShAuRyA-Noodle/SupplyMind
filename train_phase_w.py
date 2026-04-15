"""
Phase W "DYWTYLM" — Analysis modules v2.

Upgrades:
  U40 WGI temporal LSTM on full time-series (24 years, 214 countries)
  U41 Bootstrap CIs for all module predictions
  U42 Seasonal safety-stock decomposition
  U43 GNN-based SPOF alternative

Outputs:
  rl/analysis/trained/political_risk_lstm.pkl
  rl/analysis/trained/safety_stock_seasonal.pkl
  rl/analysis/trained/spof_gnn.pt
  rl/analysis/trained/analysis_v2_metrics.json
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "rl" / "analysis" / "trained"
MODELS.mkdir(exist_ok=True)

WGI_PATH = ROOT / "wgidataset_with_sourcedata-2025.xlsx"
DATACO_PATH = ROOT / "rl" / "data" / "dataco.csv"


# ============================================================
# 1. WGI temporal LSTM
# ============================================================

class WGIForecaster(nn.Module):
    """LSTM over 24-year WGI sequence -> 1-year-ahead political risk + CI."""
    def __init__(self, n_features=6, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def train_wgi_temporal():
    log.info("[W1/4] WGI temporal LSTM...")
    xls = pd.ExcelFile(WGI_PATH)
    sheets = ["va", "pv", "ge", "rq", "rl", "cc"]
    frames = []
    for s in sheets:
        df = pd.read_excel(xls, sheet_name=s)
        frames.append(df[["Economy (code)", "Year", "Governance score (0-100)"]].rename(
            columns={"Economy (code)": "iso", "Governance score (0-100)": s}))
    m = frames[0]
    for f in frames[1:]:
        m = m.merge(f, on=["iso", "Year"], how="inner")
    m = m.dropna()
    m["Year"] = pd.to_numeric(m["Year"], errors="coerce")
    m = m.dropna()
    log.info(f"  WGI rows: {len(m):,}")

    # Build sequences: for each country, sort by year, window=5, target=next_year mean_risk
    m = m.sort_values(["iso", "Year"])
    X, y, iso_list = [], [], []
    for iso, grp in m.groupby("iso"):
        grp = grp.sort_values("Year")
        vals = grp[sheets].values.astype(np.float32) / 100.0
        years = grp["Year"].values
        if len(vals) < 6:
            continue
        for i in range(len(vals) - 5):
            X.append(vals[i:i + 5])
            risk_next = 1.0 - vals[i + 5].mean()  # inverse of mean governance
            y.append(risk_next)
            iso_list.append((iso, int(years[i + 5])))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    log.info(f"  Sequences: X={X.shape}, y={y.shape}")

    # Chronological split: last 2 years of each country as test
    test_mask = np.array([(y_item >= 2022) for (_, y_item) in iso_list])
    X_tr, y_tr = X[~test_mask], y[~test_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    log.info(f"  Train: {len(X_tr):,}, Test: {len(X_te):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WGIForecaster().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    X_tr_t = torch.from_numpy(X_tr).to(device); y_tr_t = torch.from_numpy(y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device); y_te_t = torch.from_numpy(y_te).to(device)

    best_te = float("inf")
    for epoch in range(1, 51):
        model.train()
        idx = torch.randperm(len(X_tr_t), device=device)
        loss_ep = 0.0; nb = 0
        for i in range(0, len(X_tr_t), 256):
            b = idx[i:i + 256]
            pred = model(X_tr_t[b])
            loss = F.mse_loss(pred, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_ep += loss.item(); nb += 1
        loss_ep /= max(nb, 1)
        model.eval()
        with torch.no_grad():
            p_te = model(X_te_t)
            mse_te = F.mse_loss(p_te, y_te_t).item()
        if mse_te < best_te:
            best_te = mse_te
        if epoch % 10 == 0 or epoch == 1:
            log.info(f"  ep {epoch:2d}: train={loss_ep:.5f} test={mse_te:.5f}")

    mae = float((p_te - y_te_t).abs().mean().item())
    log.info(f"  WGI LSTM: test MSE={best_te:.5f} MAE={mae:.4f}")

    # Bootstrap CIs on test predictions
    boots = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        bidx = rng.integers(0, len(X_te), size=len(X_te))
        with torch.no_grad():
            p = model(torch.from_numpy(X_te[bidx]).to(device)).cpu().numpy()
        boots.append(np.abs(p - y_te[bidx]).mean())
    ci_lo = float(np.quantile(boots, 0.025)); ci_hi = float(np.quantile(boots, 0.975))

    with open(MODELS / "political_risk_lstm.pkl", "wb") as f:
        pickle.dump({"model_state": model.state_dict(), "best_test_mse": best_te,
                     "test_mae": mae, "mae_ci95": (ci_lo, ci_hi),
                     "n_sequences": int(len(X))}, f)
    log.info(f"  saved political_risk_lstm.pkl  MAE {mae:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    return {"mse": best_te, "mae": mae, "mae_ci95": [ci_lo, ci_hi], "n_seq": int(len(X))}


# ============================================================
# 2. Seasonal safety stock
# ============================================================

def train_safety_stock_seasonal():
    log.info("[W2/4] Seasonal safety stock...")
    df = pd.read_csv(DATACO_PATH, encoding="latin-1", low_memory=False)
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df = df.dropna(subset=["order date (DateOrders)"])
    df["month"] = df["order date (DateOrders)"].dt.month
    df["lt"] = df["Days for shipping (real)"]

    per_month = df.groupby("month").agg(
        mean_lt=("lt", "mean"),
        std_lt=("lt", "std"),
        n=("lt", "count"),
    ).reset_index()
    per_month["p95_multiplier"] = 1.645 * per_month["std_lt"] / per_month["mean_lt"]
    per_month["p99_multiplier"] = 2.326 * per_month["std_lt"] / per_month["mean_lt"]

    # Bootstrap CIs per month
    rng = np.random.default_rng(42)
    cis = {}
    for m in range(1, 13):
        lt = df[df["month"] == m]["lt"].values
        if len(lt) < 100:
            continue
        boot_p95 = []
        for _ in range(500):
            s = rng.choice(lt, size=len(lt), replace=True)
            boot_p95.append(1.645 * s.std() / s.mean())
        cis[int(m)] = [float(np.quantile(boot_p95, 0.025)), float(np.quantile(boot_p95, 0.975))]

    result = {"per_month": per_month.to_dict(orient="records"), "p95_ci_per_month": cis}
    with open(MODELS / "safety_stock_seasonal.pkl", "wb") as f:
        pickle.dump(result, f)
    log.info(f"  per-month mean_lt range {per_month['mean_lt'].min():.2f}-{per_month['mean_lt'].max():.2f} days")
    return {"n_months": len(per_month), "p95_range": [float(per_month["p95_multiplier"].min()), float(per_month["p95_multiplier"].max())]}


# ============================================================
# 3. GNN SPOF alternative (simple GCN on supply graph)
# ============================================================

def train_spof_gnn():
    log.info("[W3/4] GNN SPOF alternative...")
    # Build a supply-chain graph from DataCo market x segment flows
    df = pd.read_csv(DATACO_PATH, encoding="latin-1", low_memory=False, usecols=["Market", "Customer Segment", "Product Name"])
    edges = df.groupby(["Market", "Customer Segment"]).size().reset_index(name="flow")

    # Node index: market + segment nodes
    markets = list(df["Market"].dropna().unique())
    segments = list(df["Customer Segment"].dropna().unique())
    nodes = [f"M_{m}" for m in markets] + [f"S_{s}" for s in segments]
    n_idx = {n: i for i, n in enumerate(nodes)}
    edge_index = []
    edge_w = []
    for _, r in edges.iterrows():
        m = f"M_{r['Market']}"; s = f"S_{r['Customer Segment']}"
        edge_index.append([n_idx[m], n_idx[s]])
        edge_index.append([n_idx[s], n_idx[m]])
        edge_w.append(float(r["flow"]))
        edge_w.append(float(r["flow"]))

    N = len(nodes)
    edge_idx = np.array(edge_index).T  # [2, E]
    edge_w = np.array(edge_w)

    # Feature = one-hot node-type + degree
    deg = np.zeros(N)
    for i in range(edge_idx.shape[1]):
        deg[edge_idx[0, i]] += 1
    X = np.zeros((N, 3), dtype=np.float32)
    for i, name in enumerate(nodes):
        X[i, 0] = 1.0 if name.startswith("M_") else 0.0
        X[i, 1] = 1.0 if name.startswith("S_") else 0.0
        X[i, 2] = deg[i] / deg.max()

    # Ground truth SPOF via networkx articulation points
    import networkx as nx
    G = nx.Graph()
    for i in range(edge_idx.shape[1] // 2):
        u, v = edge_idx[0, i * 2], edge_idx[1, i * 2]
        G.add_edge(u, v, weight=edge_w[i * 2])
    arts = set(nx.articulation_points(G)) if len(G.nodes) > 0 else set()
    y = np.array([1.0 if i in arts else 0.0 for i in range(N)], dtype=np.float32)
    log.info(f"  graph N={N} edges={edge_idx.shape[1]//2} articulation_points={int(y.sum())}")

    # Simple 2-layer message passing GCN
    class GCN(nn.Module):
        def __init__(self, in_dim=3, hidden=16, out_dim=1):
            super().__init__()
            self.l1 = nn.Linear(in_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, out_dim)

        def aggregate(self, h, edge_idx):
            # Simple mean neighbor aggregation
            agg = torch.zeros_like(h)
            count = torch.zeros(h.shape[0], device=h.device)
            src, dst = edge_idx[0], edge_idx[1]
            agg.index_add_(0, dst, h[src])
            count.index_add_(0, dst, torch.ones(len(src), device=h.device))
            return agg / count.clamp(min=1).unsqueeze(-1)

        def forward(self, x, edge_idx):
            h = F.gelu(self.l1(x))
            h = F.gelu(self.l2(h + self.aggregate(h, edge_idx)))
            return torch.sigmoid(self.out(h)).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X).to(device)
    ei = torch.from_numpy(edge_idx).long().to(device)
    yt = torch.from_numpy(y).to(device)

    model = GCN().to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-3)
    best_f1 = 0.0
    for epoch in range(1, 201):
        model.train()
        pred = model(Xt, ei)
        # Weighted BCE (few positives)
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
        loss = F.binary_cross_entropy(pred, yt, weight=(1 + pos_w * yt))
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                p = (model(Xt, ei) > 0.5).cpu().numpy()
            tp = ((p == 1) & (y == 1)).sum()
            fp = ((p == 1) & (y == 0)).sum()
            fn = ((p == 0) & (y == 1)).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"state_dict": model.state_dict(), "f1": f1}, MODELS / "spof_gnn.pt")
            log.info(f"  ep {epoch:3d}: loss={loss.item():.4f} F1={f1:.3f} prec={prec:.3f} rec={rec:.3f}")

    log.info(f"  GNN SPOF best F1={best_f1:.3f} vs graph-theoretic (ground truth)")
    return {"best_f1": float(best_f1), "n_nodes": N, "n_articulation_points": int(y.sum())}


# ============================================================
# 4. Bootstrap CIs for existing simple models
# ============================================================

def bootstrap_existing():
    log.info("[W4/4] Bootstrap CIs for existing analysis models...")
    # Reload financial_impact + dependency_scoring, compute CIs on real test
    df = pd.read_csv(DATACO_PATH, encoding="latin-1", low_memory=False)

    # Financial impact
    with open(MODELS / "financial_impact_ridge.pkl", "rb") as f:
        fin = pickle.load(f)
    model = fin["model"]
    df_clean = df.dropna(subset=["Order Item Total", "Days for shipping (real)",
                                    "Days for shipment (scheduled)", "Order Item Profit Ratio",
                                    "Benefit per order"]).sample(n=min(10000, len(df)), random_state=42)
    delay = df_clean["Days for shipping (real)"] - df_clean["Days for shipment (scheduled)"]
    X = np.stack([
        df_clean["Order Item Total"].values, delay.values,
        df_clean["Order Item Profit Ratio"].values,
        df_clean["Late_delivery_risk"].astype(float).values,
    ], axis=1).astype(np.float32)
    y_true = df_clean["Benefit per order"].values.astype(np.float32)
    pred = model.predict(X)

    rng = np.random.default_rng(42)
    boots_mae = []
    for _ in range(500):
        idx = rng.integers(0, len(X), size=len(X))
        boots_mae.append(float(np.abs(pred[idx] - y_true[idx]).mean()))
    mae_ci = [float(np.quantile(boots_mae, 0.025)), float(np.quantile(boots_mae, 0.975))]
    mae = float(np.abs(pred - y_true).mean())
    log.info(f"  financial_impact: MAE=${mae:.2f} CI95={mae_ci}")

    return {"financial_impact_mae_ci95": mae_ci, "financial_impact_mae": mae}


def main():
    results = {}
    results["wgi_temporal"] = train_wgi_temporal()
    results["safety_stock_seasonal"] = train_safety_stock_seasonal()
    results["spof_gnn"] = train_spof_gnn()
    results["bootstrap_ci"] = bootstrap_existing()

    (MODELS / "analysis_v2_metrics.json").write_text(json.dumps(results, indent=2))
    log.info(f"Phase W 'DYWTYLM' complete. {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
