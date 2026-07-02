"""Real federated learning (FedAvg) for SupplyMind late-delivery risk.

Rebuild of R12 (see REBUILD_BACKLOG.md). The previous version was an inline toy:
it loaded the RL offline buffer, classified 280-way *actions*, and its module
header asserted a "23%" federated improvement that was never actually measured.
This version runs the real experiment on the real DataCo order dataset and
reports whatever the numbers turn out to be — including the honest possibility
that federation does not beat a single client.

Experiment
----------
Task     : predict ``Late_delivery_risk`` (binary) at ORDER-PLACEMENT time.
Data     : ``rl/data/dataco.csv`` — 180,519 real DataCo supply-chain orders.
Clients  : K = 5 federated clients, one per ``Market`` region
           (LATAM, Europe, Pacific Asia, USCA, Africa). A client's raw rows
           never leave the client; only model weights are shared each round.
Model    : logistic regression (a single ``nn.Linear`` -> 1 logit). It is
           convex, so FedAvg has a clean theoretical target (the centralized
           optimum) and the comparison is not confounded by optimizer luck.
Compare  : (a) each client trained ALONE, (b) FedAvg across all K clients,
           (c) CENTRALIZED on pooled raw data — the privacy-violating upper bound.
Metric   : ROC-AUC / accuracy / F1 on a held-out GLOBAL test set (the union of
           each market's stratified 20 % test split), plus per-market test sets.

No leakage
----------
Features use only fields known when an order is placed. The columns that DEFINE
the label — ``Days for shipping (real)`` and ``Delivery Status`` (which literally
contains the value "Late delivery") — are excluded, as are other post-hoc fields.
``Late_delivery_risk`` in DataCo is 1 iff real shipping days exceeded the
scheduled days, so leaking either would make the task trivial and dishonest.

Federated assumption
--------------------
The feature schema (categorical vocabulary) and the standardization statistics
are agreed on up front and shared across clients — this is standard in a
federated deployment (a data contract). Only the raw rows stay local and only
model weights are exchanged during training.

Usage
-----
    python rl/federated/fedavg.py
    python -m rl.federated.fedavg
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

DATACO_CSV = _PROJECT_ROOT / "rl" / "data" / "dataco.csv"
RECEIPT_PATH = _PROJECT_ROOT / "tests" / "receipts" / "fedavg_REAL.json"

LABEL_COL = "Late_delivery_risk"
SPLIT_COL = "Market"

# Order-placement-time features only (verified no NaN in the CSV).
NUMERIC_COLS = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Benefit per order",
    "Order Item Total",
    "Product Price",
]
CATEGORICAL_COLS = ["Shipping Mode", "Type", "Customer Segment", "Department Name"]

# Excluded because they encode the label or are only known after fulfilment.
LEAKAGE_COLS = [
    "Days for shipping (real)",   # label = (real > scheduled); pure leakage
    "Delivery Status",            # literally contains "Late delivery"
    "Order Status",               # post-hoc (COMPLETE / CANCELED / ...)
    "shipping date (DateOrders)",  # post-hoc timestamp
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataco() -> "tuple[Any, Any, Any, Any, list[str]]":
    """Load DataCo, build a shared feature matrix, split by market.

    Returns ``(X, y, market, feature_names)`` where X is float32 [N, D],
    y is int64 [N], market is an object array of the split column, and
    feature_names documents each column of X.
    """
    import pandas as pd

    if not DATACO_CSV.exists():
        raise FileNotFoundError(
            f"DataCo dataset not found at {DATACO_CSV}. This experiment requires "
            "the real 96MB order file — refusing to fabricate data."
        )

    usecols = NUMERIC_COLS + CATEGORICAL_COLS + [LABEL_COL, SPLIT_COL]
    df = pd.read_csv(DATACO_CSV, encoding="latin-1", usecols=usecols)
    logger.info("Loaded %d real DataCo orders", len(df))

    # One-hot the categoricals on the full (pooled) vocabulary — the agreed
    # feature schema. This does NOT leak the label; it is a data contract.
    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS], prefix_sep="=", dtype=np.float32)
    feature_names = list(NUMERIC_COLS) + list(cat_dummies.columns)

    X = np.concatenate(
        [df[NUMERIC_COLS].to_numpy(dtype=np.float32), cat_dummies.to_numpy(dtype=np.float32)],
        axis=1,
    )
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    market = df[SPLIT_COL].to_numpy()
    return X, y, market, feature_names


def stratified_split(
    y: np.ndarray, market: np.ndarray, test_frac: float, seed: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Per-(market, label) stratified train/test index split.

    Guarantees each market's test set preserves its own class balance, so the
    global test set is a clean union of per-market test sets.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for m in np.unique(market):
        for cls in (0, 1):
            idx = np.where((market == m) & (y == cls))[0]
            rng.shuffle(idx)
            n_test = int(round(len(idx) * test_frac))
            test_idx.append(idx[:n_test])
            train_idx.append(idx[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def standardize(
    X_train: np.ndarray, X_all: np.ndarray, n_numeric: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Z-score the numeric block using TRAIN statistics only (shared schema)."""
    mean = X_train[:, :n_numeric].mean(axis=0)
    std = X_train[:, :n_numeric].std(axis=0)
    std[std == 0] = 1.0
    Xn = X_all.copy()
    Xn[:, :n_numeric] = (Xn[:, :n_numeric] - mean) / std
    return Xn, np.stack([mean, std])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LogisticRegression(nn.Module):
    """Single linear layer -> one logit. Convex binary classifier."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _train_local(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> nn.Module:
    """Full-batch SGD on one client's local data (the FedAvg local step)."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, X: torch.Tensor, y: np.ndarray, device: str) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval().to(device)
    probs = torch.sigmoid(model(X.to(device))).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "n": int(len(y)),
    }


def _fedavg(state_dicts: list[dict], weights: np.ndarray) -> dict:
    """Sample-count-weighted parameter average (McMahan et al. 2017)."""
    w = torch.tensor(weights / weights.sum(), dtype=torch.float32)
    avg = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        shape = [len(state_dicts)] + [1] * (stacked.dim() - 1)
        avg[key] = (stacked * w.view(shape)).sum(dim=0)
    return avg


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
class FederatedExperiment:
    def __init__(
        self,
        n_rounds: int = 60,
        local_epochs: int = 3,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        test_frac: float = 0.2,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_frac = test_frac
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X, y, market, feature_names = load_dataco()
        n_numeric = len(NUMERIC_COLS)
        in_dim = X.shape[1]

        train_idx, test_idx = stratified_split(y, market, self.test_frac, self.seed)
        Xn, _stats = standardize(X[train_idx], X, n_numeric)

        markets = sorted(np.unique(market).tolist())
        # Build tensors per client (train) and per market (test) + a global test.
        client_train: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        market_test: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        for m in markets:
            m_train = np.array([i for i in np.where(market == m)[0] if i in train_set])
            m_test = np.array([i for i in np.where(market == m)[0] if i in test_set])
            Xt = torch.tensor(Xn[m_train], dtype=torch.float32, device=self.device)
            yt = torch.tensor(y[m_train], dtype=torch.float32, device=self.device)
            client_train[m] = (Xt, yt, len(m_train))
            market_test[m] = (
                torch.tensor(Xn[m_test], dtype=torch.float32),
                y[m_test],
            )
        X_test_g = torch.tensor(Xn[test_idx], dtype=torch.float32)
        y_test_g = y[test_idx]
        X_train_g = torch.tensor(Xn[train_idx], dtype=torch.float32, device=self.device)
        y_train_g = torch.tensor(y[train_idx], dtype=torch.float32, device=self.device)

        logger.info(
            "Split: %d train / %d test | %d features | device=%s",
            len(train_idx), len(test_idx), in_dim, self.device,
        )
        for m in markets:
            logger.info("  client %-13s train=%6d", m, client_train[m][2])

        # --- (a) Each client trained ALONE ---------------------------------
        client_alone: dict[str, Any] = {}
        total_steps = self.n_rounds * self.local_epochs
        for m in markets:
            torch.manual_seed(self.seed)
            model = LogisticRegression(in_dim)
            Xt, yt, _ = client_train[m]
            model = _train_local(model, Xt, yt, total_steps, self.lr, self.weight_decay, self.device)
            client_alone[m] = {
                "on_global_test": _evaluate(model, X_test_g, y_test_g, self.device),
                "on_own_test": _evaluate(model, market_test[m][0], market_test[m][1], self.device),
            }
            logger.info(
                "  ALONE %-13s global AUC=%.4f  own AUC=%.4f",
                m, client_alone[m]["on_global_test"]["auc"],
                client_alone[m]["on_own_test"]["auc"],
            )

        # --- (b) FedAvg across all K clients -------------------------------
        torch.manual_seed(self.seed)
        global_model = LogisticRegression(in_dim)
        client_weights = np.array([client_train[m][2] for m in markets], dtype=np.float64)
        fed_rounds: list[dict] = []
        for rnd in range(self.n_rounds):
            local_states = []
            for m in markets:
                local = copy.deepcopy(global_model)
                Xt, yt, _ = client_train[m]
                local = _train_local(local, Xt, yt, self.local_epochs, self.lr, self.weight_decay, self.device)
                local_states.append({k: v.cpu() for k, v in local.state_dict().items()})
            global_model.load_state_dict(_fedavg(local_states, client_weights))
            if (rnd + 1) % 10 == 0 or rnd == 0:
                ev = _evaluate(global_model, X_test_g, y_test_g, self.device)
                fed_rounds.append({"round": rnd + 1, **ev})
                logger.info("  FedAvg round %2d/%d  global AUC=%.4f", rnd + 1, self.n_rounds, ev["auc"])
        fed_global = _evaluate(global_model, X_test_g, y_test_g, self.device)
        fed_per_market = {m: _evaluate(global_model, market_test[m][0], market_test[m][1], self.device) for m in markets}

        # --- (c) CENTRALIZED (pooled raw data — upper bound) ---------------
        torch.manual_seed(self.seed)
        central = LogisticRegression(in_dim)
        central = _train_local(central, X_train_g, y_train_g, total_steps, self.lr, self.weight_decay, self.device)
        central_global = _evaluate(central, X_test_g, y_test_g, self.device)
        central_per_market = {m: _evaluate(central, market_test[m][0], market_test[m][1], self.device) for m in markets}

        # --- Honest deltas --------------------------------------------------
        best_alone_market = max(markets, key=lambda m: client_alone[m]["on_global_test"]["auc"])
        best_alone_auc = client_alone[best_alone_market]["on_global_test"]["auc"]
        mean_alone_auc = float(np.mean([client_alone[m]["on_global_test"]["auc"] for m in markets]))

        # Small-client story: does FedAvg beat that client trained alone, on its OWN market test?
        smallest = min(markets, key=lambda m: client_train[m][2])
        small_alone = client_alone[smallest]["on_own_test"]["auc"]
        small_fed = fed_per_market[smallest]["auc"]

        findings = {
            "fedavg_vs_best_single_client_auc_delta": round(fed_global["auc"] - best_alone_auc, 5),
            "fedavg_vs_mean_single_client_auc_delta": round(fed_global["auc"] - mean_alone_auc, 5),
            "fedavg_vs_centralized_auc_delta": round(fed_global["auc"] - central_global["auc"], 5),
            "best_single_client": best_alone_market,
            "smallest_client": smallest,
            "smallest_client_alone_own_auc": round(small_alone, 5),
            "smallest_client_fedavg_own_auc": round(small_fed, 5),
            "smallest_client_fedavg_gain_auc": round(small_fed - small_alone, 5),
        }

        return {
            "experiment": "fedavg_dataco_late_delivery",
            "dataset": {
                "path": str(DATACO_CSV.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
                "n_orders": int(X.shape[0]),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "base_rate_late": round(float(y.mean()), 4),
                "n_features": in_dim,
                "n_clients": len(markets),
                "clients_by_market": {m: client_train[m][2] for m in markets},
            },
            "config": {
                "model": "logistic_regression (nn.Linear -> 1 logit, BCEWithLogits)",
                "optimizer": f"SGD(lr={self.lr}, weight_decay={self.weight_decay})",
                "n_rounds": self.n_rounds,
                "local_epochs": self.local_epochs,
                "total_local_steps_per_client": total_steps,
                "test_frac": self.test_frac,
                "seed": self.seed,
                "device": self.device,
                "fedavg_weighting": "by client sample count (McMahan 2017)",
                "leakage_excluded": LEAKAGE_COLS,
            },
            "results": {
                "client_alone": client_alone,
                "fedavg": {
                    "global_test": fed_global,
                    "per_market_test": fed_per_market,
                    "round_history": fed_rounds,
                },
                "centralized": {
                    "global_test": central_global,
                    "per_market_test": central_per_market,
                },
            },
            "findings": findings,
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Real FedAvg on DataCo late-delivery risk")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-receipt", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    exp = FederatedExperiment(n_rounds=args.rounds, local_epochs=args.local_epochs, seed=args.seed)
    result = exp.run()
    result["elapsed_s"] = round(time.time() - t0, 2)

    try:
        result["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT)
        ).decode().strip()
    except Exception:
        result["git_sha"] = "unknown"
    result["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    f = result["findings"]
    print("\n" + "=" * 72)
    print("REAL FedAvg on DataCo late-delivery risk — honest result")
    print("=" * 72)
    print(f"  Centralized (upper bound)   global AUC = {result['results']['centralized']['global_test']['auc']:.4f}")
    print(f"  FedAvg (K={result['dataset']['n_clients']} markets)         global AUC = {result['results']['fedavg']['global_test']['auc']:.4f}")
    print(f"  Best single client ({f['best_single_client']}) global AUC = {result['results']['client_alone'][f['best_single_client']]['on_global_test']['auc']:.4f}")
    print("-" * 72)
    print(f"  FedAvg - best single client : {f['fedavg_vs_best_single_client_auc_delta']:+.5f} AUC")
    print(f"  FedAvg - centralized        : {f['fedavg_vs_centralized_auc_delta']:+.5f} AUC")
    print(f"  Smallest client ({f['smallest_client']}) on its own market:")
    print(f"    alone  AUC = {f['smallest_client_alone_own_auc']:.4f}")
    print(f"    FedAvg AUC = {f['smallest_client_fedavg_own_auc']:.4f}  ({f['smallest_client_fedavg_gain_auc']:+.5f})")
    print("=" * 72)

    if not args.no_receipt:
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Receipt written: {RECEIPT_PATH}")


if __name__ == "__main__":
    main()
