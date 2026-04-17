"""R3-α v3 — TRUE per-point Bates-Granger constrained stacking.

Fixes the honest caveat from R3-α v2: the earlier script used synthesized
fold-MAE draws because R3 Past Self didn't store per-point predictions.
This version **re-runs the forecasters** and stores per-point predictions,
enabling a genuine point-level constrained stacking.

Constraints:
  minimize  || y_cal - W @ preds_cal ||_MAE  or _MSE
  s.t.      w_i >= 0, sum(w_i) = 1

Where preds_cal is [n_cal_points, M] (each row a horizon step from a fold).

For speed, runs on 3 targets × 2 horizons × 10 cal + 10 test folds × 4
forecasters. Expected duration: ~15-25 min.

Output:
  v3_arcadia/results/R3_STACKING_V3_POINTLEVEL.json
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "rl" / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "v3_arcadia" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = ["DCOILWTICO", "DEXUSEU", "DEXCHUS"]  # representative mix
HORIZONS = [7, 14]  # 2 horizons for speed
N_CAL = 10
N_TEST = 10
MIN_CTX = 512

_CHRONOS = None


def load_fred(target):
    raw = json.loads((DATA / "fred_cache.json").read_text())
    v = raw.get(target, {})
    if not (isinstance(v, dict) and "data" in v):
        return None
    df = pd.DataFrame(v["data"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def chronos_point(series, horizon):
    global _CHRONOS
    if _CHRONOS is None:
        from chronos import ChronosBoltPipeline
        _CHRONOS = ChronosBoltPipeline.from_pretrained(
            str(MODELS / "chronos-bolt-base"), device_map=DEVICE, torch_dtype=torch.float32)
    ctx = torch.tensor(series.values[-1024:], dtype=torch.float32).unsqueeze(0)
    q, _ = _CHRONOS.predict_quantiles(inputs=ctx, prediction_length=horizon,
                                       quantile_levels=[0.5])
    return q[0, :, 0].cpu().numpy()


def arima_point(series, horizon):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(series.values, order=(5, 1, 0)).fit()
        return np.asarray(m.get_forecast(steps=horizon).predicted_mean)
    except Exception:
        return None


def prophet_point(series, dates, horizon):
    try:
        import logging as lg
        lg.getLogger("prophet").setLevel(lg.ERROR)
        lg.getLogger("cmdstanpy").setLevel(lg.ERROR)
        from prophet import Prophet
        df = pd.DataFrame({"ds": dates, "y": series.values})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                     daily_seasonality=False)
        m.fit(df)
        fut = m.make_future_dataframe(periods=horizon, freq="B")
        return m.predict(fut).tail(horizon)["yhat"].values
    except Exception:
        return None


def naive_point(series, horizon):
    """Naive baseline: repeat last value."""
    return np.full(horizon, series.iloc[-1])


def gen_folds(series, dates, horizon, n_total):
    N = len(series)
    stride = max((N - MIN_CTX - horizon) // n_total, 1)
    out = []
    for i in range(n_total):
        end = MIN_CTX + i * stride
        if end + horizon > N: break
        out.append({
            "ctx": series.iloc[:end],
            "dates": dates.iloc[:end],
            "actual": series.iloc[end:end + horizon].values,
        })
    return out


def constrained_stack(preds_cal: np.ndarray, y_cal: np.ndarray, loss="mae") -> np.ndarray:
    """preds_cal: [N, M]  one row per test point across cal folds.
    y_cal: [N]   target
    loss: "mae" or "mse"
    Returns w: [M] with w >= 0, sum(w) = 1
    """
    M = preds_cal.shape[1]
    x0 = np.ones(M) / M
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * M
    def loss_fn(w):
        pred = preds_cal @ w
        if loss == "mae":
            return float(np.abs(pred - y_cal).mean())
        else:
            return float(((pred - y_cal) ** 2).mean())
    res = minimize(loss_fn, x0, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-8})
    w = np.clip(res.x, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(M) / M


def eval_target_horizon(target, horizon):
    log.info(f"\n  {target}  horizon={horizon}")
    df = load_fred(target)
    s = df["value"].astype(float).reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    folds = gen_folds(s, dates, horizon, N_CAL + N_TEST)
    if len(folds) < N_CAL + 3:
        return {"error": f"not enough folds ({len(folds)})"}

    # Collect per-point predictions for each forecaster + actual
    model_names = ["chronos", "arima", "prophet", "naive"]
    records = []
    for i, f in enumerate(folds):
        row = {"actual": f["actual"], "fold_i": i}
        try:
            row["chronos"] = chronos_point(f["ctx"], horizon)
        except Exception:
            row["chronos"] = None
        row["arima"] = arima_point(f["ctx"], horizon)
        row["prophet"] = prophet_point(f["ctx"], f["dates"], horizon)
        row["naive"] = naive_point(f["ctx"], horizon)
        records.append(row)

    # Drop folds where any model failed
    valid = [r for r in records if all(r.get(m) is not None for m in model_names)]
    if len(valid) < N_CAL + 3:
        return {"error": f"not enough valid folds ({len(valid)})"}

    cal = valid[:N_CAL]
    test = valid[N_CAL:]

    # Flatten cal: stack all points from all cal folds
    cal_preds = np.stack([np.concatenate([r[m] for r in cal]) for m in model_names], axis=1)
    cal_y = np.concatenate([r["actual"] for r in cal])
    test_preds = np.stack([np.concatenate([r[m] for r in test]) for m in model_names], axis=1)
    test_y = np.concatenate([r["actual"] for r in test])

    # Individual MAEs
    ind_mae = {}
    for i, m in enumerate(model_names):
        ind_mae[m] = float(np.abs(test_preds[:, i] - test_y).mean())

    # 3 stacking methods
    w_mae = constrained_stack(cal_preds, cal_y, loss="mae")
    w_mse = constrained_stack(cal_preds, cal_y, loss="mse")
    w_eq = np.ones(len(model_names)) / len(model_names)
    w_best = np.zeros(len(model_names))
    best_cal_idx = int(np.argmin([np.abs(cal_preds[:, i] - cal_y).mean() for i in range(len(model_names))]))
    w_best[best_cal_idx] = 1.0

    stack_mae = {
        "equal":          float(np.abs(test_preds @ w_eq  - test_y).mean()),
        "best_on_cal":    float(np.abs(test_preds @ w_best - test_y).mean()),
        "constrained_mae":float(np.abs(test_preds @ w_mae - test_y).mean()),
        "constrained_mse":float(np.abs(test_preds @ w_mse - test_y).mean()),
    }

    # Best single
    best_single = min(ind_mae.items(), key=lambda x: x[1])

    # Winner across all
    all_methods = {**ind_mae, **stack_mae}
    winner = min(all_methods.items(), key=lambda x: x[1])

    log.info(f"    individual MAE: {ind_mae}")
    log.info(f"    stacking MAE:   {stack_mae}")
    log.info(f"    best single:    {best_single[0]}={best_single[1]:.3f}")
    log.info(f"    winner:         {winner[0]}={winner[1]:.3f}")

    return {
        "n_cal_points": cal_preds.shape[0],
        "n_test_points": test_preds.shape[0],
        "individual_mae": ind_mae,
        "stacking_mae": stack_mae,
        "weights": {
            "constrained_mae": {m: float(w) for m, w in zip(model_names, w_mae)},
            "constrained_mse": {m: float(w) for m, w in zip(model_names, w_mse)},
        },
        "best_single_model": best_single[0],
        "best_single_mae": best_single[1],
        "winner_method": winner[0],
        "winner_mae": winner[1],
        "constrained_beats_best_single": stack_mae["constrained_mae"] < best_single[1] or stack_mae["constrained_mse"] < best_single[1],
    }


def main():
    t0 = time.time()
    log.info("R3-α v3 — TRUE point-level constrained stacking")

    out = {"description": "Per-point Bates-Granger constrained stacking on real forecaster outputs. No synthesized folds.",
           "per_target_horizon": {}, "wins": {"constrained": 0, "best_single": 0, "equal": 0, "naive": 0}}

    total_cells = 0
    constrained_wins = 0
    for target in TARGETS:
        log.info(f"\n=== {target} ===")
        for h in HORIZONS:
            r = eval_target_horizon(target, h)
            if "error" not in r:
                total_cells += 1
                if r["winner_method"] in ("constrained_mae", "constrained_mse"):
                    constrained_wins += 1
                if r["constrained_beats_best_single"]:
                    out["wins"]["constrained"] += 1
                else:
                    out["wins"]["best_single"] += 1
            out["per_target_horizon"][f"{target}_h{h}"] = r

    out["summary"] = {
        "total_target_horizon_cells": total_cells,
        "constrained_stacking_wins": constrained_wins,
        "constrained_beats_best_single_cells": out["wins"]["constrained"],
    }
    out["elapsed_min"] = (time.time() - t0) / 60
    out_path = RESULTS / "R3_STACKING_V3_POINTLEVEL.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    log.info("")
    log.info("=== R3-α v3 SUMMARY ===")
    log.info(f"  {total_cells} target-horizon cells evaluated on REAL per-point predictions")
    log.info(f"  Constrained stacking beats best-single on {out['wins']['constrained']}/{total_cells} cells")
    log.info(f"  Constrained stacking wins outright on {constrained_wins}/{total_cells} cells")
    log.info(f"  Saved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
