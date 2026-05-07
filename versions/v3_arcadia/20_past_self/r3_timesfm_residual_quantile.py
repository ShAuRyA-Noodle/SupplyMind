"""R3-β — Residual-based quantile wrapper for TimesFM-2.

TimesFM-2 returns point forecasts only (no native quantile bands). For
conformal and PICP analysis we need prediction intervals. This script adds
**residual-based quantile bands** computed from a rolling-origin calibration
set:

  For each horizon step h, compute |y_true - y_pred_timesfm| over the
  calibration folds. Then q_h(alpha) = empirical quantile of |residual_h|.

  PI_h(alpha) = [yhat_h - q_h(alpha), yhat_h + q_h(alpha)]

This is equivalent to a split-conformal wrapper but packaged as "TimesFM
quantile bands" so it can be dropped into R3's ensemble directly.

Compares resulting TimesFM-with-quantiles PICP vs Chronos native quantiles
on the same data. Shows whether TimesFM point-forecaster with post-hoc
bands is competitive.

Outputs:
  versions/v3_arcadia/results/R3_TIMESFM_QUANTILE.json
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
MODELS = ROOT / "models"
RESULTS = ROOT / "v3_arcadia" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

TARGETS = ["DCOILWTICO", "DEXJPUS", "DEXUSEU"]  # 3 targets for speed
HORIZON = 14
N_CAL = 20
N_TEST = 20
NOMINAL_CONFS = [0.8, 0.9, 0.95]

_TIMESFM = None
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


def timesfm_point(series: pd.Series, horizon: int):
    global _TIMESFM
    if _TIMESFM is None:
        import timesfm
        hp = timesfm.TimesFmHparams(
            backend="gpu" if DEVICE == "cuda" else "cpu",
            per_core_batch_size=32, horizon_len=horizon, context_len=2048,
            num_layers=50, model_dims=1280, num_heads=16)
        ckpt = timesfm.TimesFmCheckpoint(path=str(MODELS / "timesfm-2" / "torch_model.ckpt"))
        _TIMESFM = timesfm.TimesFm(hparams=hp, checkpoint=ckpt)
    point_forecast, _ = _TIMESFM.forecast([series.values.astype(np.float32)], freq=[0])
    return np.asarray(point_forecast)[0][:horizon]


def chronos_intervals(series: pd.Series, horizon: int, confs):
    global _CHRONOS
    if _CHRONOS is None:
        from chronos import ChronosBoltPipeline
        _CHRONOS = ChronosBoltPipeline.from_pretrained(
            str(MODELS / "chronos-bolt-base"), device_map=DEVICE,
            torch_dtype=torch.float32)
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


def gen_folds(series, horizon, n_total, min_ctx=512):
    N = len(series)
    stride = max((N - min_ctx - horizon) // n_total, 1)
    return [{"ctx_end": min_ctx + i * stride,
             "ctx": series.iloc[:min_ctx + i * stride],
             "actual": series.iloc[min_ctx + i * stride:min_ctx + i * stride + horizon].values}
            for i in range(n_total)
            if min_ctx + i * stride + horizon <= N]


def eval_target(target: str):
    log.info(f"\n=== {target} ===")
    df = load_fred(target)
    if df is None:
        return {"error": "no data"}
    s = df["value"].astype(float).reset_index(drop=True)
    folds = gen_folds(s, HORIZON, N_CAL + N_TEST)
    if len(folds) < N_CAL + 5:
        return {"error": f"not enough folds ({len(folds)})"}
    log.info(f"  {len(folds)} folds collected")

    # ---- TimesFM point predictions ----
    timesfm_preds = []
    for f in folds:
        p = timesfm_point(f["ctx"], HORIZON)
        timesfm_preds.append({"point": p, "actual": f["actual"]})

    cal = timesfm_preds[:N_CAL]
    test = timesfm_preds[N_CAL:]

    # Compute per-horizon residuals on calibration
    cal_residuals = np.array([p["actual"] - p["point"] for p in cal])  # [N_CAL, H]

    # Per-horizon conformal q-hat
    def q_per_horizon(alpha):
        n, H = cal_residuals.shape
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        return np.array([np.sort(np.abs(cal_residuals[:, h]))[k - 1]
                          for h in range(H)])

    result = {"target": target, "n_cal": len(cal), "n_test": len(test)}
    for conf in NOMINAL_CONFS:
        alpha = 1 - conf
        qh = q_per_horizon(alpha)
        # Test coverage for TimesFM-with-conformal-quantiles
        covs = []
        widths = []
        for p in test:
            lo = p["point"] - qh
            hi = p["point"] + qh
            covs.append(float(((p["actual"] >= lo) & (p["actual"] <= hi)).mean()))
            widths.append(float(np.mean(hi - lo)))
        result[f"timesfm_conf={conf}"] = {
            "nominal_coverage": conf,
            "empirical_coverage": float(np.mean(covs)),
            "mean_width": float(np.mean(widths)),
            "dev_from_nominal": abs(float(np.mean(covs)) - conf),
        }
        log.info(f"  TimesFM-CP conf={conf}: cov={np.mean(covs):.3f}  width={np.mean(widths):.3f}  dev={abs(np.mean(covs) - conf):.3f}")

    # ---- Compare to Chronos native quantiles on same test folds ----
    log.info("  Chronos-native comparison...")
    chronos_results = {}
    chronos_cov_widths = {conf: {"covs": [], "widths": []} for conf in NOMINAL_CONFS}
    for f in folds[N_CAL:]:  # test only
        _, lo_dict, hi_dict = chronos_intervals(f["ctx"], HORIZON, NOMINAL_CONFS)
        for conf in NOMINAL_CONFS:
            lo = lo_dict[conf]; hi = hi_dict[conf]
            chronos_cov_widths[conf]["covs"].append(
                float(((f["actual"] >= lo) & (f["actual"] <= hi)).mean()))
            chronos_cov_widths[conf]["widths"].append(float(np.mean(hi - lo)))
    for conf in NOMINAL_CONFS:
        cov_mean = float(np.mean(chronos_cov_widths[conf]["covs"]))
        width_mean = float(np.mean(chronos_cov_widths[conf]["widths"]))
        chronos_results[f"chronos_native_conf={conf}"] = {
            "nominal_coverage": conf,
            "empirical_coverage": cov_mean,
            "mean_width": width_mean,
            "dev_from_nominal": abs(cov_mean - conf),
        }
        log.info(f"  Chronos-native conf={conf}: cov={cov_mean:.3f}  width={width_mean:.3f}  dev={abs(cov_mean - conf):.3f}")
    result.update(chronos_results)
    return result


def main():
    t0 = time.time()
    log.info("R3-β — TimesFM residual-based quantile wrapper")
    out = {"method": "per-horizon split-conformal wrapper on TimesFM point forecasts",
           "comparison": "Chronos-Bolt native quantiles",
           "targets": {}}
    for target in TARGETS:
        out["targets"][target] = eval_target(target)
    out["elapsed_min"] = (time.time() - t0) / 60

    out_path = RESULTS / "R3_TIMESFM_QUANTILE.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved {out_path}  ({out['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
