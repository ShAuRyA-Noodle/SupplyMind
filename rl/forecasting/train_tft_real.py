"""
Phase D — Real commodity-price forecaster on FRED data.

Pure-PyTorch Temporal Fusion-style model:
  - Encoder: 2-layer LSTM over 60-day history
  - Variable selection: multi-head attention on covariates
  - Head: quantile regression on [0.1, 0.5, 0.9] for horizon=14 days
  - Target: WTI crude oil daily price
  - Covariates: copper, 5 FX rates

Pure-PyTorch avoids pytorch-forecasting/lightning version drift.
Full chronological split (no look-ahead leakage).

Output: rl/checkpoints/tft_real.pt + tft_real_metrics.json
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

ROOT = Path(__file__).resolve().parent.parent.parent
FRED_PATH = ROOT / "rl" / "data" / "fred_cache.json"
CKPT_DIR = ROOT / "rl" / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

ENC_LEN = 60
HORIZON = 14
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_fred_df() -> pd.DataFrame:
    raw = json.loads(FRED_PATH.read_text())
    series = ["DCOILWTICO", "PCOPPUSDM", "DEXTAUS", "DEXKOUS", "DEXJPUS", "DEXUSEU", "DEXCHUS"]
    frames = []
    for s in series:
        if s not in raw:
            continue
        df = pd.DataFrame(raw[s]["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={"value": s})
        frames.append(df.resample("B").ffill())
    combined = pd.concat(frames, axis=1, join="inner").dropna().reset_index()
    return combined


def build_windows(df: pd.DataFrame, target: str):
    """Return (X, y) where X is [N, ENC_LEN, F] covariate history and y is [N, HORIZON] future target."""
    feat_cols = ["DCOILWTICO", "PCOPPUSDM", "DEXTAUS", "DEXKOUS", "DEXJPUS", "DEXUSEU", "DEXCHUS"]
    feats = df[feat_cols].values.astype(np.float32)
    tgt = df[target].values.astype(np.float32)

    # Z-normalize features on training portion only (done after split below)
    N = len(df) - ENC_LEN - HORIZON + 1
    X = np.zeros((N, ENC_LEN, len(feat_cols)), dtype=np.float32)
    y = np.zeros((N, HORIZON), dtype=np.float32)
    for i in range(N):
        X[i] = feats[i:i + ENC_LEN]
        y[i] = tgt[i + ENC_LEN:i + ENC_LEN + HORIZON]
    return X, y


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """pred: [B, H, Q], target: [B, H]."""
    target = target.unsqueeze(-1)  # [B, H, 1]
    errors = target - pred  # [B, H, Q]
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    loss = torch.max(q * errors, (q - 1) * errors).mean()
    return loss


class TFTLike(nn.Module):
    def __init__(self, n_feats: int, hidden: int = 64, horizon: int = HORIZON, n_quantiles: int = 3):
        super().__init__()
        self.var_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.input_proj = nn.Linear(n_feats, hidden)
        self.encoder = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, horizon * n_quantiles),
        )
        self.horizon = horizon
        self.n_q = n_quantiles

    def forward(self, x):
        # x: [B, T, F]
        h = self.input_proj(x)
        attn_out, _ = self.var_attn(h, h, h)
        enc_out, _ = self.encoder(attn_out)
        last = enc_out[:, -1]  # [B, H]
        out = self.decoder(last).view(-1, self.horizon, self.n_q)
        return out


def main():
    df = load_fred_df()
    log.info(f"FRED df: {len(df)} business days {df['date'].min().date()} -> {df['date'].max().date()}")

    X, y = build_windows(df, target="DCOILWTICO")
    log.info(f"Windows: X={X.shape}, y={y.shape}")

    # Chronological split (no leakage)
    n_train = int(0.80 * len(X))
    n_val = int(0.10 * len(X))
    X_tr, X_va, X_te = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_tr, y_va, y_te = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    # Normalize using train stats
    mu = X_tr.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    X_te = (X_te - mu) / sd
    y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-6
    y_tr_n = (y_tr - y_mu) / y_sd
    y_va_n = (y_va - y_mu) / y_sd

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_n))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va_n))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=64, num_workers=0)

    model = TFTLike(n_feats=X.shape[-1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,} | device={DEVICE}")

    best_val = float("inf")
    best_path = CKPT_DIR / "tft_real.pt"
    for epoch in range(1, 21):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = quantile_loss(pred, yb, QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                va_loss += quantile_loss(pred, yb, QUANTILES).item()
        va_loss /= len(va_dl)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "state_dict": model.state_dict(),
                "mu": mu, "sd": sd, "y_mu": y_mu, "y_sd": y_sd,
                "quantiles": QUANTILES, "horizon": HORIZON, "enc_len": ENC_LEN,
                "n_feats": X.shape[-1],
            }, best_path)

        if epoch % 2 == 0 or epoch == 1:
            log.info(f"  epoch {epoch:2d}: train={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f}")

    # Test set MAE on P50
    model.load_state_dict(torch.load(best_path)["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()
    # De-normalize
    p50 = pred_te[..., 1] * y_sd + y_mu
    mae = float(np.abs(p50 - y_te).mean())
    rmse = float(np.sqrt(((p50 - y_te) ** 2).mean()))
    log.info(f"Test MAE (P50): ${mae:.3f}  RMSE: ${rmse:.3f}  best_val_qloss={best_val:.4f}")

    metrics = {
        "mae_p50_usd": mae,
        "rmse_p50_usd": rmse,
        "best_val_quantile_loss": best_val,
        "params": n_params,
        "n_train_windows": len(X_tr),
        "n_val_windows": len(X_va),
        "n_test_windows": len(X_te),
        "enc_len": ENC_LEN,
        "horizon": HORIZON,
        "quantiles": QUANTILES,
        "target": "DCOILWTICO",
    }
    (CKPT_DIR / "tft_real_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("TFT (TFT-like pure PyTorch) training complete.")


if __name__ == "__main__":
    main()
