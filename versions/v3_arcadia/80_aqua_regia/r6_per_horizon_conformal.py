"""R6 Aqua Regia v2 — Per-horizon split-conformal prediction intervals.

Fixes the honest R6 Aqua Regia finding that pooled-residual conformal under-covered
high-variance series (oil).

Root cause: residual magnitude grows monotonically with horizon step. Pooling
residuals across all steps produces a single q-hat that is too small for late
steps and too large for early steps — on average, under-covers for heavy-tailed
series because the distribution is skewed right.

Fix: compute separate q_hat_1, q_hat_2, ..., q_hat_H from calibration residuals
at each horizon step independently. This is standard practice in conformal
prediction literature (Foygel Barber et al., Lei et al.).

Expected result: empirical coverage within ±2pp of nominal across all targets.

Reuses same FRED data and forecaster wrappers as the v1 conformal script.

Outputs:
  versions/v3_arcadia/results/R6_AQUA_REGIA_V2.json
  versions/v3_arcadia/plots/aqua_regia/r6_aqua_regia_v2.png
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "rl" / "data"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "v3_arcadia" / "results"
PLOTS = ROOT / "v3_arcadia" / "plots" / "aqua_regia"
PLOTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

TARGETS = ["DCOILWTICO", "DEXJPUS", "DEXUSEU", "DEXCHUS", "DEXKOUS"]
HORIZON = 14
N_CAL_FOLDS = 30
N_TEST_FOLDS = 30
NOMINAL_CONFS = [0.8, 0.9, 0.95]  # these are CONFIDENCE levels
ERROR_ALPHAS = [round(1 - c, 2) for c in NOMINAL_CONFS]  # error-rate for conformal

_CHRONOS = None


def load_fred() -> dict:
    raw = json.loads((DATA / "fred_cache.json").read_text())
    out = {}
    for k in TARGETS:
        v = raw.get(k)
        if isinstance(v, dict) and "data" in v:
            df = pd.DataFrame(v["data"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            out[k] = df
    return out


def chronos_forecast(series, horizon, confs):
    global _CHRONOS
    try:
        if _CHRONOS is None:
            from chronos import ChronosBoltPipeline
            _CHRONOS = ChronosBoltPipeline.from_pretrained(
                str(MODELS_DIR / "chronos-bolt-base"),
                device_map=DEVICE, torch_dtype=torch.float32)
        ctx = torch.tensor(series.values[-1024:], dtype=torch.float32).unsqueeze(0)
        qlevels = set([0.5])
        for c in confs:
            qlevels.add(round(0.5 - c / 2, 3))
            qlevels.add(round(0.5 + c / 2, 3))
        qs = sorted(qlevels)
        q, _ = _CHRONOS.predict_quantiles(inputs=ctx, prediction_length=horizon,
                                           quantile_levels=qs)
        arr = q[0].cpu().numpy()
        point = arr[:, qs.index(0.5)]
        lo = {c: arr[:, qs.index(round(0.5 - c / 2, 3))] for c in confs}
        hi = {c: arr[:, qs.index(round(0.5 + c / 2, 3))] for c in confs}
        return point, lo, hi
    except Exception as e:
        log.warning(f"  chronos fail: {e}")
        return None, None, None


def arima_forecast(series, horizon, confs):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(series.values, order=(5, 1, 0)).fit()
        fc = m.get_forecast(steps=horizon)
        point = np.asarray(fc.predicted_mean)
        lo, hi = {}, {}
        for c in confs:
            ci = fc.conf_int(alpha=1 - c)
            lo[c] = np.asarray(ci[:, 0])
            hi[c] = np.asarray(ci[:, 1])
        return point, lo, hi
    except Exception as e:
        log.warning(f"  arima fail: {e}")
        return None, None, None


def gen_folds(series, horizon, n_total, min_ctx=512):
    N = len(series)
    stride = max((N - min_ctx - horizon) // n_total, 1)
    out = []
    for i in range(n_total):
        end = min_ctx + i * stride
        if end + horizon > N:
            break
        out.append({"ctx_end": end, "ctx": series.iloc[:end],
                    "actual": series.iloc[end:end + horizon].values})
    return out


def per_horizon_conformal_band(cal_residuals: np.ndarray, alpha: float) -> np.ndarray:
    """cal_residuals: [n_cal, H]  |y - yhat| at each horizon step per fold.
    Returns q_hat: [H]  finite-sample conformal quantile per horizon step.
    """
    n, H = cal_residuals.shape
    q_hat = np.zeros(H)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    for h in range(H):
        q_hat[h] = float(np.sort(np.abs(cal_residuals[:, h]))[k - 1])
    return q_hat


def pooled_conformal_band(cal_residuals: np.ndarray, alpha: float) -> float:
    """Single q_hat across all horizon steps (v1 method, kept for comparison)."""
    flat = np.abs(cal_residuals).reshape(-1)
    n = len(flat)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(flat)[k - 1])


def eval_target(target, series, forecaster, forecaster_fn):
    log.info(f"  [{forecaster}] {target}: generating folds...")
    folds = gen_folds(series, HORIZON, N_CAL_FOLDS + N_TEST_FOLDS, min_ctx=512)
    preds = []
    for f in folds:
        p, lo, hi = forecaster_fn(f["ctx"], HORIZON, NOMINAL_CONFS)
        if p is None:
            continue
        preds.append({"point": np.asarray(p), "actual": f["actual"],
                      "lo": lo, "hi": hi})
    if len(preds) < N_CAL_FOLDS + 5:
        return {"error": f"not enough valid folds ({len(preds)})"}

    cal = preds[:N_CAL_FOLDS]
    test = preds[N_CAL_FOLDS:]

    # Residuals per horizon step on calibration set
    cal_residuals = np.array([p["actual"] - p["point"] for p in cal])  # [n_cal, H]

    out = {"forecaster": forecaster, "n_cal": len(cal), "n_test": len(test)}

    for conf in NOMINAL_CONFS:
        alpha = 1 - conf
        q_per_horizon = per_horizon_conformal_band(cal_residuals, alpha)  # [H]
        q_pooled = pooled_conformal_band(cal_residuals, alpha)

        # Test coverage
        bare_covs = []
        bare_widths = []
        perh_covs = []
        perh_widths = []
        pool_covs = []
        pool_widths = []
        for p in test:
            actual = p["actual"]
            point = p["point"]
            # Bare model interval
            lo_bare = p["lo"].get(conf)
            hi_bare = p["hi"].get(conf)
            if lo_bare is not None and hi_bare is not None:
                bare_covs.append(float(((actual >= lo_bare) & (actual <= hi_bare)).mean()))
                bare_widths.append(float(np.mean(hi_bare - lo_bare)))
            # Per-horizon conformal
            perh_lo = point - q_per_horizon
            perh_hi = point + q_per_horizon
            perh_covs.append(float(((actual >= perh_lo) & (actual <= perh_hi)).mean()))
            perh_widths.append(float(np.mean(perh_hi - perh_lo)))
            # Pooled conformal (v1 for reference)
            pool_lo = point - q_pooled
            pool_hi = point + q_pooled
            pool_covs.append(float(((actual >= pool_lo) & (actual <= pool_hi)).mean()))
            pool_widths.append(float(2 * q_pooled))

        out[f"conf={conf}"] = {
            "nominal_coverage": conf,
            "bare_coverage_mean": float(np.mean(bare_covs)) if bare_covs else None,
            "bare_width_mean": float(np.mean(bare_widths)) if bare_widths else None,
            "perhorizon_coverage_mean": float(np.mean(perh_covs)),
            "perhorizon_width_mean": float(np.mean(perh_widths)),
            "pooled_coverage_mean": float(np.mean(pool_covs)),
            "pooled_width_mean": float(np.mean(pool_widths)),
            "q_per_horizon": q_per_horizon.tolist(),
            "q_pooled": q_pooled,
        }
    return out


def main():
    t0 = time.time()
    log.info("R6 Aqua Regia v2 — per-horizon split-conformal")

    series_map = load_fred()
    log.info(f"Loaded {list(series_map.keys())}")

    out = {"targets": TARGETS, "horizon": HORIZON,
           "confs": NOMINAL_CONFS, "n_cal": N_CAL_FOLDS, "n_test": N_TEST_FOLDS,
           "results": {}}

    for target in TARGETS:
        s = series_map[target]["value"].astype(float).reset_index(drop=True)
        log.info(f"\n=== {target}  (N={len(s)}) ===")
        tgt_out = {}
        for fc_name, fc_fn in [("arima", arima_forecast), ("chronos", chronos_forecast)]:
            r = eval_target(target, s, fc_name, fc_fn)
            tgt_out[fc_name] = r
            if "error" not in r:
                for c in NOMINAL_CONFS:
                    e = r[f"conf={c}"]
                    bc = e.get("bare_coverage_mean")
                    pc = e.get("perhorizon_coverage_mean")
                    poc = e.get("pooled_coverage_mean")
                    log.info(f"    conf={c}: bare={bc:.3f}  pooled={poc:.3f}  per-horizon={pc:.3f}  (nominal={c})")
        out["results"][target] = tgt_out

    out["elapsed_min"] = (time.time() - t0) / 60
    out_path = RESULTS / "R6_AQUA_REGIA_V2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)")

    # Summary: how close to nominal does per-horizon get?
    log.info("\n=== v2 SUMMARY: deviation from nominal coverage ===")
    for target in TARGETS:
        for fc in ["arima", "chronos"]:
            r = out["results"][target].get(fc, {})
            if "error" in r:
                continue
            for c in NOMINAL_CONFS:
                e = r.get(f"conf={c}", {})
                ph_dev = abs(e.get("perhorizon_coverage_mean", 0) - c)
                pool_dev = abs(e.get("pooled_coverage_mean", 0) - c)
                better = "✓" if ph_dev < pool_dev else " "
                log.info(f"  {target:<12} {fc:<8} conf={c}  per-h dev={ph_dev:.3f}  pool dev={pool_dev:.3f}  {better}")


if __name__ == "__main__":
    main()
