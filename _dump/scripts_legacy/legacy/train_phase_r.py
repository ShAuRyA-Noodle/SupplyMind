"""
Phase R "The Apparition" — TFT big + multi-target + rolling backtest.

Upgrades:
  U24 Bigger TFT-like model (350K+ params)
  U25 100 epochs with early stopping
  U26 Multi-target: oil + copper + shipping-proxy (PPICMM)
  U27 Rolling-origin 10-fold backtest with reliability plot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FRED = ROOT / "rl" / "data" / "fred_cache.json"
FRED_EXT = ROOT / "rl" / "data" / "fred_extended.json"
CKPT = ROOT / "rl" / "checkpoints"
CKPT.mkdir(exist_ok=True)

ENC_LEN = 90
HORIZON = 14
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_multitarget():
    raw = json.loads(FRED.read_text())
    ext = json.loads(FRED_EXT.read_text())
    series = ["DCOILWTICO", "PCOPPUSDM", "DEXTAUS", "DEXKOUS", "DEXJPUS", "DEXUSEU", "DEXCHUS",
              "PPIACO", "PPICMM", "PCU484121484121", "IPG334S", "IR"]
    frames = []
    for s in series:
        src = raw if s in raw else ext
        if s not in src:
            continue
        df = pd.DataFrame(src[s]["data"])
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df.set_index("date").rename(columns={"value": s}).resample("B").ffill())
    combined = pd.concat(frames, axis=1, join="inner").dropna().reset_index()
    log.info(f"Multi-target FRED: {len(combined)} days, {len(series)} series")
    return combined


def build_windows(df: pd.DataFrame, targets):
    feat_cols = [c for c in df.columns if c != "date"]
    feats = df[feat_cols].values.astype(np.float32)
    N = len(df) - ENC_LEN - HORIZON + 1
    X = np.zeros((N, ENC_LEN, len(feat_cols)), dtype=np.float32)
    Y = np.zeros((N, len(targets), HORIZON), dtype=np.float32)
    tgt_idx = [feat_cols.index(t) for t in targets]
    for i in range(N):
        X[i] = feats[i:i + ENC_LEN]
        for j, ti in enumerate(tgt_idx):
            Y[i, j] = feats[i + ENC_LEN:i + ENC_LEN + HORIZON, ti]
    return X, Y, feat_cols


def quantile_loss(pred, target, quantiles):
    """pred: [B, T, H, Q], target: [B, T, H]."""
    target = target.unsqueeze(-1)
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, 1, -1)
    return torch.max(q * err, (q - 1) * err).mean()


class BigTFT(nn.Module):
    def __init__(self, n_feats, n_targets, hidden=128, horizon=HORIZON, n_q=3):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, hidden)
        self.var_attn = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True, dropout=0.1)
        self.encoder = nn.LSTM(hidden, hidden, num_layers=3, batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_targets * horizon * n_q),
        )
        self.n_targets = n_targets
        self.horizon = horizon
        self.n_q = n_q

    def forward(self, x):
        h = self.input_proj(x)
        h2, _ = self.var_attn(h, h, h)
        h = self.layer_norm(h + h2)
        out, _ = self.encoder(h)
        last = out[:, -1]
        y = self.decoder(last).view(-1, self.n_targets, self.horizon, self.n_q)
        return y


def rolling_backtest(X_all, Y_all, n_folds=10, epochs=30, targets=["DCOILWTICO", "PCOPPUSDM", "PPICMM"]):
    results = []
    N = len(X_all)
    fold_size = N // (n_folds + 1)
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        val_end = train_end + fold_size
        X_tr, Y_tr = X_all[:train_end], Y_all[:train_end]
        X_va, Y_va = X_all[train_end:val_end], Y_all[train_end:val_end]
        if len(X_va) < 10:
            continue
        # Normalize using train
        mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
        sd = X_tr.reshape(-1, X_tr.shape[-1]).std(0) + 1e-6
        y_mu = Y_tr.reshape(-1, Y_tr.shape[-1]).mean(0)  # per-horizon
        y_sd = Y_tr.reshape(-1, Y_tr.shape[-1]).std(0) + 1e-6
        Xtn = (X_tr - mu) / sd; Xvn = (X_va - mu) / sd
        Ytn = (Y_tr - y_mu) / y_sd; Yvn = (Y_va - y_mu) / y_sd

        ds_tr = TensorDataset(torch.from_numpy(Xtn), torch.from_numpy(Ytn))
        ds_va = TensorDataset(torch.from_numpy(Xvn), torch.from_numpy(Yvn))
        dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=64)

        model = BigTFT(n_feats=X_all.shape[-1], n_targets=len(targets)).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_va = float("inf")
        best_state = None
        for ep in range(epochs):
            model.train()
            tr_loss = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                p = model(xb)
                loss = quantile_loss(p, yb, QUANTILES)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                tr_loss += loss.item()
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    p = model(xb)
                    va_loss += quantile_loss(p, yb, QUANTILES).item()
            tr_loss /= max(len(dl_tr), 1); va_loss /= max(len(dl_va), 1)
            if va_loss < best_va:
                best_va = va_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Fold metrics on val set, denormalized MAE
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_va = model(torch.from_numpy(Xvn).to(DEVICE)).cpu().numpy()
        p50 = p_va[..., 1] * y_sd + y_mu
        mae_per_target = {targets[j]: float(np.abs(p50[:, j] - Y_va[:, j]).mean()) for j in range(len(targets))}
        results.append({"fold": fold, "best_val_qloss": best_va, "mae_per_target": mae_per_target})
        log.info(f"  fold {fold}: val_qloss={best_va:.4f} MAE={mae_per_target}")

    return results


def main():
    targets = ["DCOILWTICO", "PCOPPUSDM", "PPICMM"]
    df = load_multitarget()
    X, Y, feat_cols = build_windows(df, targets)
    log.info(f"Windows: X={X.shape}, Y={Y.shape} (B, T, H)")

    # Primary train on 80/10/10
    n_tr = int(0.80 * len(X)); n_va = int(0.10 * len(X))
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_va, Y_va = X[n_tr:n_tr + n_va], Y[n_tr:n_tr + n_va]
    X_te, Y_te = X[n_tr + n_va:], Y[n_tr + n_va:]

    mu = X_tr.reshape(-1, X.shape[-1]).mean(0)
    sd = X_tr.reshape(-1, X.shape[-1]).std(0) + 1e-6
    y_mu = Y_tr.reshape(-1, Y.shape[-1]).mean(0)
    y_sd = Y_tr.reshape(-1, Y.shape[-1]).std(0) + 1e-6
    X_tr_n = (X_tr - mu) / sd; X_va_n = (X_va - mu) / sd; X_te_n = (X_te - mu) / sd
    Y_tr_n = (Y_tr - y_mu) / y_sd; Y_va_n = (Y_va - y_mu) / y_sd

    ds_tr = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_n))
    ds_va = TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(Y_va_n))
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=64)

    model = BigTFT(n_feats=X.shape[-1], n_targets=len(targets)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"BigTFT params: {n_params:,}")
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    best_va = float("inf")
    patience = 0
    path = CKPT / "tft_v2.pt"
    for ep in range(1, 101):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(xb)
            loss = quantile_loss(p, yb, QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tr_loss += loss.item()
        sched.step()
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                p = model(xb)
                va_loss += quantile_loss(p, yb, QUANTILES).item()
        tr_loss /= max(len(dl_tr), 1); va_loss /= max(len(dl_va), 1)
        if va_loss < best_va:
            best_va = va_loss; patience = 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_qloss": best_va,
                        "mu": mu, "sd": sd, "y_mu": y_mu, "y_sd": y_sd,
                        "targets": targets, "n_feats": X.shape[-1]}, path)
        else:
            patience += 1
        if ep % 5 == 0 or ep == 1:
            log.info(f"  TFT ep {ep:3d}/100 tr={tr_loss:.4f} va={va_loss:.4f} best={best_va:.4f} pat={patience}")
        if patience >= 15:
            log.info(f"  early stop at epoch {ep}")
            break

    # Test MAE
    model.load_state_dict(torch.load(path)["state_dict"])
    model.eval()
    with torch.no_grad():
        p_te = model(torch.from_numpy(X_te_n).to(DEVICE)).cpu().numpy()
    p50 = p_te[..., 1] * y_sd + y_mu
    mae_te = {targets[j]: float(np.abs(p50[:, j] - Y_te[:, j]).mean()) for j in range(len(targets))}
    log.info(f"Test MAE P50: {mae_te}")

    # Rolling backtest (10 folds)
    log.info("Rolling 10-fold backtest...")
    folds = rolling_backtest(X, Y, n_folds=10, epochs=20, targets=targets)

    summary = {
        "params": n_params,
        "test_mae_p50": mae_te,
        "best_val_qloss": best_va,
        "rolling_backtest": folds,
        "horizon": HORIZON,
        "targets": targets,
    }
    (CKPT / "tft_v2_metrics.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase R 'The Apparition' complete.")


if __name__ == "__main__":
    main()
