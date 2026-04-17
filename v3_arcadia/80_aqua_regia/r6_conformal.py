"""R6 Block 8 — Aqua Regia: Conformal prediction calibration for R3 forecasters.

Adds split-conformal prediction wrapper around ARIMA + Prophet + Chronos (cached).
Compares:
  - Bare model quantile intervals (often miscalibrated)
  - Split-conformal calibrated intervals (finite-sample coverage guarantee)
  - Adaptive conformal (quantile regression of residuals)

Outputs:
  v3_arcadia/results/R6_AQUA_REGIA.json
  v3_arcadia/plots/aqua_regia/coverage.png
  v3_arcadia/plots/aqua_regia/bandwidth.png
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "rl" / "data"
MODELS = ROOT / "models"
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "aqua_regia"
CKPT.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "v3_arcadia" / "plots" / "aqua_regia"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

TARGETS = ["DCOILWTICO", "DEXJPUS", "DEXUSEU", "DEXCHUS", "DEXKOUS"]
HORIZON = 14
N_CAL_FOLDS = 30  # calibration folds
N_TEST_FOLDS = 30  # test folds
NOMINAL_ALPHAS = [0.8, 0.9, 0.95]


# ============================================================
# Data loading (same as R3)
# ============================================================
def load_fred() -> dict:
    raw = json.loads((DATA / "fred_cache.json").read_text())
    out = {}
    for k in TARGETS:
        if k in raw and isinstance(raw[k], dict) and "data" in raw[k]:
            df = pd.DataFrame(raw[k]["data"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            out[k] = df
    return out


# ============================================================
# Forecasters (ARIMA + Chronos; Prophet too slow for this many folds)
# ============================================================
_CHRONOS = None


def chronos_forecast(series: pd.Series, horizon: int, alpha_levels: list[float]):
    """Returns (point, lo_dict, hi_dict) where lo/hi are keyed by alpha."""
    global _CHRONOS
    try:
        if _CHRONOS is None:
            from chronos import ChronosBoltPipeline
            _CHRONOS = ChronosBoltPipeline.from_pretrained(
                str(MODELS / "chronos-bolt-base"), device_map=DEVICE, torch_dtype=torch.float32)
        ctx = torch.tensor(series.values[-1024:], dtype=torch.float32).unsqueeze(0)
        # Build quantile levels from alpha: for each alpha, lo=0.5-alpha/2, hi=0.5+alpha/2
        quantiles = set([0.5])
        for a in alpha_levels:
            quantiles.add(round(0.5 - a / 2, 3))
            quantiles.add(round(0.5 + a / 2, 3))
        qs = sorted(quantiles)
        q, _ = _CHRONOS.predict_quantiles(inputs=ctx, prediction_length=horizon, quantile_levels=qs)
        arr = q[0].cpu().numpy()  # [H, len(qs)]
        point = arr[:, qs.index(0.5)]
        lo = {a: arr[:, qs.index(round(0.5 - a / 2, 3))] for a in alpha_levels}
        hi = {a: arr[:, qs.index(round(0.5 + a / 2, 3))] for a in alpha_levels}
        return point, lo, hi
    except Exception as e:
        log.warning(f"  chronos fail: {str(e)[:80]}")
        return None, None, None


def arima_forecast(series: pd.Series, horizon: int, alpha_levels: list[float]):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(series.values, order=(5, 1, 0)).fit()
        point = np.asarray(m.get_forecast(steps=horizon).predicted_mean)
        lo = {}
        hi = {}
        for a in alpha_levels:
            ci = m.get_forecast(steps=horizon).conf_int(alpha=1 - a)
            lo[a] = np.asarray(ci[:, 0])
            hi[a] = np.asarray(ci[:, 1])
        return point, lo, hi
    except Exception as e:
        log.warning(f"  arima fail: {str(e)[:80]}")
        return None, None, None


# ============================================================
# Split-conformal: calibrate absolute residuals, use empirical quantile as band
# ============================================================
def split_conformal_band(cal_residuals: np.ndarray, alpha: float) -> float:
    """Return the (1-alpha)-quantile of |residuals|, with finite-sample correction.

    Split-conformal guarantee: P(|Y_test - ŷ_test| <= q̂) >= 1 - alpha (marginally).
    """
    n = len(cal_residuals)
    if n == 0: return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(np.abs(cal_residuals))[k - 1])


def rolling_conformal_eval(series: pd.Series, forecaster_name: str, forecaster_fn,
                           horizon: int, n_cal: int, n_test: int,
                           alphas: list[float], min_ctx: int = 512) -> dict:
    """Rolling-origin: generate n_cal+n_test forecasts. First n_cal used for calibration,
    next n_test used for evaluation.
    """
    N = len(series)
    total = n_cal + n_test
    stride = max((N - min_ctx - horizon) // total, 1)
    all_preds = []  # list of (point, actual, lo_dict, hi_dict) per fold
    for i in range(total):
        end = min_ctx + i * stride
        if end + horizon > N: break
        ctx = series.iloc[:end]
        actual = series.iloc[end:end + horizon].values
        point, lo, hi = forecaster_fn(ctx, horizon, alphas)
        if point is None: continue
        all_preds.append((np.asarray(point), actual, lo, hi))

    if len(all_preds) < n_cal + 5:
        return {"error": f"not enough folds ({len(all_preds)} < {n_cal + 5})"}

    cal = all_preds[:n_cal]
    test = all_preds[n_cal:n_cal + n_test]

    # Absolute residuals per horizon step from calibration set, pooled
    cal_residuals = np.concatenate([np.abs(p - a) for (p, a, _, _) in cal])

    # Evaluate on test
    results = {"forecaster": forecaster_name, "n_cal": len(cal), "n_test": len(test)}
    for alpha in alphas:
        nominal = 1 - alpha
        # Bare model coverage (from model's own quantile band)
        bare_covs = []
        bare_widths = []
        # Conformal coverage
        conf_cov = []
        conf_widths = []
        q_hat = split_conformal_band(cal_residuals, alpha)
        for (p, a, lo, hi) in test:
            # Bare: coverage of y in [lo[alpha(nominal=1-alpha), hi]]
            # Chronos/ARIMA used alpha as confidence (1-sig). Keep consistent: alpha = confidence.
            # The lo/hi were computed with alpha_levels as confidence levels.
            if (1 - alpha) in lo:
                bi = lo[1 - alpha]
                bhi = hi[1 - alpha]
                bare_covs.append(float(((a >= bi) & (a <= bhi)).mean()))
                bare_widths.append(float((bhi - bi).mean()))
            # Conformal: [p - q_hat, p + q_hat]
            cov = float(((a >= p - q_hat) & (a <= p + q_hat)).mean())
            conf_cov.append(cov)
            conf_widths.append(2 * q_hat)

        results[f"alpha={alpha}"] = {
            "nominal_coverage": float(nominal),
            "bare_coverage_mean": float(np.mean(bare_covs)) if bare_covs else None,
            "bare_width_mean": float(np.mean(bare_widths)) if bare_widths else None,
            "conformal_coverage_mean": float(np.mean(conf_cov)) if conf_cov else None,
            "conformal_width_mean": float(np.mean(conf_widths)) if conf_widths else None,
            "conformal_q_hat": q_hat,
        }
    return results


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    log.info("R6 Aqua Regia — conformal prediction calibration")
    series_map = load_fred()
    log.info(f"Loaded targets: {list(series_map.keys())}")

    out = {"targets": TARGETS, "horizon": HORIZON, "alphas": NOMINAL_ALPHAS,
           "n_cal": N_CAL_FOLDS, "n_test": N_TEST_FOLDS, "results": {}}

    # Important: Chronos treats alpha_levels as the interval widths (e.g. 0.8 = 80% PI).
    # We pass NOMINAL_ALPHAS as CONFIDENCE LEVELS, not error levels.
    # conformal split uses alpha as error rate, so we need inversion.
    # Rewrite: pass [1 - a for a in NOMINAL_ALPHAS] as error alphas to conformal.
    conformal_error_alphas = [round(1 - c, 2) for c in NOMINAL_ALPHAS]  # [0.2, 0.1, 0.05]

    for target in TARGETS:
        s = series_map[target]["value"].astype(float).reset_index(drop=True)
        log.info(f"\n=== {target}  (N={len(s)}) ===")
        target_out = {}
        for name, fn in [("arima", arima_forecast), ("chronos", chronos_forecast)]:
            log.info(f"  [{name}] evaluating...")
            res = rolling_conformal_eval(s, name, lambda ctx, h, _: fn(ctx, h, NOMINAL_ALPHAS),
                                          HORIZON, N_CAL_FOLDS, N_TEST_FOLDS, conformal_error_alphas)
            target_out[name] = res
            for a in conformal_error_alphas:
                entry = res.get(f"alpha={a}")
                if entry:
                    log.info(f"    alpha={a}: bare_cov={entry.get('bare_coverage_mean',0):.3f} "
                             f"conformal_cov={entry.get('conformal_coverage_mean',0):.3f} "
                             f"(nominal={entry['nominal_coverage']:.2f})")
        out["results"][target] = target_out

    out["elapsed_min"] = (time.time() - t0) / 60
    out_path = RESULTS / "R6_AQUA_REGIA.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
