"""
Temporal Fusion Transformer for commodity price forecasting.

Data from FRED (oil, copper, gas) + Baltic Dry Index CSV, 2015-present.
Produces 30-day ahead P10/P50/P90 commodity price forecasts.

Config:
  - pytorch-forecasting TimeSeriesDataSet
  - TemporalFusionTransformer with hidden_size=16, attention_head_size=1
  - QuantileLoss([0.1, 0.5, 0.9])
  - max_encoder_length=90, max_prediction_length=30
  - ~20 min for 100 epochs on GPU

CHECKPOINTS: the pytorch-forecasting train_tft() below writes tft_best.ckpt,
but the checkpoint that actually ships and is consumed at inference is the
pure-PyTorch tft_real.pt produced by train_tft_real.py (avoids lightning/
pytorch-forecasting version drift). predict_tft() loads tft_real.pt and errors
loudly if it is absent.

Usage:
    python -m rl.forecasting.train_tft_real   # the real trainer (-> tft_real.pt)
    python -m rl.forecasting.tft --predict    # forecast from tft_real.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_fred_series() -> pd.DataFrame:
    """Load FRED cache and build time-series DataFrame for TFT."""
    cache_path = DATA_DIR / "fred_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"FRED cache not found at {cache_path}. Run dataset.py first.")

    with open(cache_path) as f:
        fred_data = json.load(f)

    # Build per-series DataFrames
    series_dfs = {}
    for series_id in ["DCOILWTICO", "PCOPPUSDM"]:
        if series_id not in fred_data or not fred_data[series_id]["data"]:
            logger.warning("Missing FRED series: %s", series_id)
            continue
        records = fred_data[series_id]["data"]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={"value": series_id})
        df = df.resample("B").ffill()  # Business day frequency, forward fill
        series_dfs[series_id] = df

    if not series_dfs:
        raise ValueError("No FRED series available for TFT training")

    # Merge all series
    combined = pd.concat(series_dfs.values(), axis=1, join="inner")
    combined = combined.dropna()
    combined = combined.reset_index()
    combined.rename(columns={"index": "date"}, inplace=True)

    # Add time features
    combined["day_of_week"] = combined["date"].dt.dayofweek
    combined["month"] = combined["date"].dt.month
    combined["year"] = combined["date"].dt.year
    combined["time_idx"] = np.arange(len(combined))
    combined["group"] = "commodities"  # Single group for TFT

    logger.info("TFT data: %d rows, columns=%s", len(combined), list(combined.columns))
    return combined


def train_tft(
    max_epochs: int = 100,
    max_encoder_length: int = 90,
    max_prediction_length: int = 30,
    hidden_size: int = 16,
    attention_head_size: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Path:
    """Train Temporal Fusion Transformer on commodity price data."""
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TFT Training (Temporal Fusion Transformer)")
    logger.info("  Epochs: %d | Encoder: %d | Prediction: %d | Hidden: %d",
                max_epochs, max_encoder_length, max_prediction_length, hidden_size)
    logger.info("  Quantiles: [0.1, 0.5, 0.9] | Device: %s", device)
    logger.info("=" * 60)

    df = load_fred_series()

    # Target column (crude oil as primary)
    target = "DCOILWTICO"

    # Split: last 200 days for validation
    training_cutoff = df["time_idx"].max() - 200

    # pytorch-forecasting TimeSeriesDataSet
    time_varying_known = ["day_of_week", "month"]
    time_varying_unknown = [c for c in ["DCOILWTICO", "PCOPPUSDM"] if c in df.columns]

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx > training_cutoff],
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=4, pin_memory=True,
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=4, pin_memory=True,
    )

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        learning_rate=lr,
        log_interval=10,
        reduce_on_plateau_patience=5,
    )

    logger.info("TFT parameters: %s", f"{sum(p.numel() for p in tft.parameters()):,}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if "cuda" in device else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=False,
        default_root_dir=str(CHECKPOINT_DIR / "tft_logs"),
    )

    start = time.time()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start

    # Save
    save_path = CHECKPOINT_DIR / "tft_best.ckpt"
    trainer.save_checkpoint(str(save_path))

    logger.info("=" * 60)
    logger.info("TFT training done in %.1f min. Saved: %s", elapsed / 60, save_path)
    logger.info("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()
    return save_path


def predict_tft(
    checkpoint_path: Path | None = None,
) -> dict:
    """Generate a real commodity-price forecast from the trained TFT model.

    Loads the pure-PyTorch checkpoint produced by train_tft_real.py
    (rl/checkpoints/tft_real.pt) — the only TFT checkpoint that actually
    exists — reconstructs the TFTLike model, runs the most recent encoder
    window through it, and returns de-normalized P10/P50/P90 forecasts for
    the target series (WTI crude, DCOILWTICO) over the model's native horizon.

    Raises FileNotFoundError if no trained checkpoint is present. There is no
    silent fabrication: callers that want a heuristic band must call
    _fallback_forecast() explicitly.
    """
    from rl.forecasting.train_tft_real import TFTLike, load_fred_df

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "tft_real.pt"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained TFT checkpoint at {checkpoint_path}. "
            f"Train one with: python -m rl.forecasting.train_tft_real"
        )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    required = {"state_dict", "mu", "sd", "y_mu", "y_sd", "quantiles", "horizon", "enc_len", "n_feats"}
    missing = required - set(ckpt)
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a train_tft_real TFTLike checkpoint "
            f"(missing keys: {sorted(missing)}). predict_tft only supports that format."
        )

    horizon = int(ckpt["horizon"])
    enc_len = int(ckpt["enc_len"])
    quantiles = list(ckpt["quantiles"])
    n_feats = int(ckpt["n_feats"])

    model = TFTLike(n_feats=n_feats, horizon=horizon, n_quantiles=len(quantiles))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Most recent encoder window of the exact 7 feature columns used in training.
    df = load_fred_df()
    if len(df) < enc_len:
        raise ValueError(
            f"Only {len(df)} FRED business days available; need >= enc_len={enc_len}."
        )
    feat_cols = ["DCOILWTICO", "PCOPPUSDM", "DEXTAUS", "DEXKOUS", "DEXJPUS", "DEXUSEU", "DEXCHUS"]
    window = df[feat_cols].values[-enc_len:].astype(np.float32)
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    x = (window - mu) / sd
    x_t = torch.from_numpy(x).unsqueeze(0)  # [1, enc_len, n_feats]

    with torch.no_grad():
        pred = model(x_t).squeeze(0).numpy()  # [horizon, n_quantiles]

    # De-normalize to USD.
    y_mu = float(ckpt["y_mu"])
    y_sd = float(ckpt["y_sd"])
    pred = pred * y_sd + y_mu
    q_index = {q: i for i, q in enumerate(quantiles)}
    last_date = pd.to_datetime(df["date"].values[-1])
    last_observed = float(df["DCOILWTICO"].values[-1])

    def _band(q: float) -> list[float]:
        return [round(float(v), 2) for v in pred[:, q_index[q]]]

    return {
        "status": "trained",
        "model": "TFTLike (pure-PyTorch, train_tft_real.py)",
        "checkpoint": checkpoint_path.name,
        "target": "DCOILWTICO",
        "target_label": "Crude Oil (WTI)",
        "horizon_days": horizon,
        "quantiles": quantiles,
        "last_observed_date": str(last_date.date()),
        "last_observed": round(last_observed, 2),
        "forecast": {
            "p10": _band(min(quantiles)),
            "p50": _band(sorted(quantiles)[len(quantiles) // 2]),
            "p90": _band(max(quantiles)),
        },
    }


def _fallback_forecast(horizon: int = 30) -> dict:
    """Fallback forecast when TFT hasn't been trained yet.

    Uses last observed FRED values with simple volatility scaling.
    This is NOT synthetic -- it's the last real observed value + historical vol.
    """
    cache_path = DATA_DIR / "fred_cache.json"
    if not cache_path.exists():
        return {"status": "no_data", "horizon_days": horizon}

    with open(cache_path) as f:
        fred_data = json.load(f)

    forecasts = {}
    for series_id in ["DCOILWTICO", "PCOPPUSDM"]:
        if series_id not in fred_data or not fred_data[series_id]["data"]:
            continue
        data = fred_data[series_id]["data"]
        values = [d["value"] for d in data[-90:]]  # last 90 days
        if not values:
            continue

        last_val = values[-1]
        # Historical volatility (annualized std of daily returns)
        if len(values) > 1:
            returns = np.diff(np.log(np.array(values, dtype=np.float64)))
            vol = float(np.std(returns) * np.sqrt(252))
        else:
            vol = 0.2  # default 20% annual vol

        # Simple forecast: last value +/- vol-scaled band
        daily_vol = vol / np.sqrt(252)
        days = np.arange(1, horizon + 1)
        p50 = [last_val] * horizon
        p10 = [last_val * (1 - 1.28 * daily_vol * np.sqrt(d)) for d in days]
        p90 = [last_val * (1 + 1.28 * daily_vol * np.sqrt(d)) for d in days]

        forecasts[series_id] = {
            "label": fred_data[series_id]["label"],
            "last_observed": last_val,
            "annual_volatility": round(vol, 4),
            "p10": [round(v, 2) for v in p10],
            "p50": [round(v, 2) for v in p50],
            "p90": [round(v, 2) for v in p90],
        }

    return {
        "status": "fallback_from_historical_vol",
        "horizon_days": horizon,
        "forecasts": forecasts,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="TFT Commodity Forecasting")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.train:
        train_tft(device=args.device)
    if args.predict:
        result = predict_tft()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
