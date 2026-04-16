"""Fix R2 Task 4: Benefit per order regression with RAW target + MAE-optimized quantile regression.

Previous bug: log1p-signed transform amplified errors on inverse transform.
Fix: train directly on raw target with MAE objective, no transform.
Also try quantile regression P50 as primary predictor.
"""
from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "rl" / "data"
MODELS = ROOT / "models"
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "caramel"
RESULTS = ROOT / "v3_arcadia" / "results"

import sys
sys.path.insert(0, str(ROOT / "v3_arcadia" / "10_caramel"))
from train_caramel import build_features, bootstrap_ci, SEED, TABPFN_REG

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


def main():
    t0 = time.time()
    print("Benefit/order regression FIX (raw target + MAE objective + quantile)")
    df = pd.read_csv(DATA / "dataco.csv", encoding="latin-1", low_memory=False).reset_index(drop=True)
    X, meta = build_features(df, "benefit")
    y = pd.to_numeric(df["Benefit per order"], errors="coerce").fillna(0).values.astype(np.float32)
    print(f"  features: {X.shape[1]}, y stats: min={y.min():.1f} max={y.max():.1f} mean={y.mean():.1f} std={y.std():.1f}")

    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.1764, random_state=SEED)
    print(f"  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

    # Baseline: predict train mean
    baseline_mae = float(np.abs(y_te - y_tr.mean()).mean())
    baseline_rmse = float(np.sqrt(((y_te - y_tr.mean()) ** 2).mean()))
    print(f"  BASELINE (predict mean): MAE=${baseline_mae:.2f}  RMSE=${baseline_rmse:.2f}")

    results = {"baseline": {"mae": baseline_mae, "rmse": baseline_rmse}}

    # ---- XGB with MAE objective ----
    print("\n  XGB (MAE objective)...")
    m_xgb = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8,
                              subsample=0.85, colsample_bytree=0.85,
                              tree_method="hist", device="cuda", verbosity=0,
                              objective="reg:absoluteerror",
                              early_stopping_rounds=50)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p = m_xgb.predict(X_te)
    mae, lo, hi = bootstrap_ci(y_te, p, mean_absolute_error)
    r2, r2_lo, r2_hi = bootstrap_ci(y_te, p, r2_score)
    rmse = float(np.sqrt(mean_squared_error(y_te, p)))
    results["xgb_mae"] = {"mae": mae, "mae_ci95": [lo, hi], "r2": r2,
                          "r2_ci95": [r2_lo, r2_hi], "rmse": rmse}
    print(f"    XGB(MAE): MAE=${mae:.2f} [{lo:.2f},{hi:.2f}]  R2={r2:.4f}  RMSE=${rmse:.2f}")
    with open(CKPT / "benefit_xgb_mae.pkl", "wb") as f: pickle.dump(m_xgb, f)

    # ---- LGB with L1 objective ----
    print("\n  LGB (L1 / regression_l1)...")
    m_lgb = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                               subsample=0.85, colsample_bytree=0.85,
                               min_child_samples=20, objective="regression_l1",
                               metric="mae", verbosity=-1)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m_lgb.predict(X_te)
    mae, lo, hi = bootstrap_ci(y_te, p, mean_absolute_error)
    r2, r2_lo, r2_hi = bootstrap_ci(y_te, p, r2_score)
    rmse = float(np.sqrt(mean_squared_error(y_te, p)))
    results["lgb_l1"] = {"mae": mae, "mae_ci95": [lo, hi], "r2": r2,
                         "r2_ci95": [r2_lo, r2_hi], "rmse": rmse}
    print(f"    LGB(L1):  MAE=${mae:.2f} [{lo:.2f},{hi:.2f}]  R2={r2:.4f}  RMSE=${rmse:.2f}")
    with open(CKPT / "benefit_lgb_l1.pkl", "wb") as f: pickle.dump(m_lgb, f)

    # ---- LGB quantile P10/P50/P90 ----
    print("\n  LGB quantile (P10/P50/P90)...")
    quantile_preds = {}
    for alpha in [0.1, 0.5, 0.9]:
        m_q = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                 subsample=0.85, colsample_bytree=0.85,
                                 min_child_samples=20, objective="quantile", alpha=alpha,
                                 metric="quantile", verbosity=-1)
        m_q.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        quantile_preds[alpha] = m_q.predict(X_te)
        if alpha == 0.5:
            with open(CKPT / "benefit_lgb_p50.pkl", "wb") as f: pickle.dump(m_q, f)

    p50 = quantile_preds[0.5]
    mae, lo, hi = bootstrap_ci(y_te, p50, mean_absolute_error)
    r2, r2_lo, r2_hi = bootstrap_ci(y_te, p50, r2_score)
    # PICP (coverage) at 80% nominal = fraction of y_te in [p10, p90]
    picp_80 = float(((y_te >= quantile_preds[0.1]) & (y_te <= quantile_preds[0.9])).mean())
    results["lgb_quantile_p50"] = {"mae": mae, "mae_ci95": [lo, hi], "r2": r2,
                                    "r2_ci95": [r2_lo, r2_hi], "picp_80": picp_80}
    print(f"    LGB(P50): MAE=${mae:.2f} [{lo:.2f},{hi:.2f}]  R2={r2:.4f}  PICP@80%={picp_80:.3f}")

    # ---- CatBoost with MAE loss ----
    print("\n  CatBoost (MAE)...")
    m_cat = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8,
                               loss_function="MAE", eval_metric="MAE",
                               early_stopping_rounds=50, random_seed=SEED,
                               task_type="CPU", thread_count=-1, verbose=False)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    p = m_cat.predict(X_te)
    mae, lo, hi = bootstrap_ci(y_te, p, mean_absolute_error)
    r2, r2_lo, r2_hi = bootstrap_ci(y_te, p, r2_score)
    rmse = float(np.sqrt(mean_squared_error(y_te, p)))
    results["cat_mae"] = {"mae": mae, "mae_ci95": [lo, hi], "r2": r2,
                          "r2_ci95": [r2_lo, r2_hi], "rmse": rmse}
    print(f"    CAT(MAE): MAE=${mae:.2f} [{lo:.2f},{hi:.2f}]  R2={r2:.4f}  RMSE=${rmse:.2f}")

    # ---- TabPFN-v2-reg (small subsample) ----
    print("\n  TabPFN-v2-reg...")
    try:
        from tabpfn import TabPFNRegressor
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), size=min(10_000, len(X_tr)), replace=False)
        m_tp = TabPFNRegressor(device="cuda", model_path=str(TABPFN_REG),
                                n_estimators=2, ignore_pretraining_limits=True)
        m_tp.fit(X_tr.iloc[idx].values, y_tr[idx])
        p = m_tp.predict(X_te.values)
        mae, lo, hi = bootstrap_ci(y_te, p, mean_absolute_error)
        r2, r2_lo, r2_hi = bootstrap_ci(y_te, p, r2_score)
        results["tabpfn_reg"] = {"mae": mae, "mae_ci95": [lo, hi], "r2": r2,
                                  "r2_ci95": [r2_lo, r2_hi]}
        print(f"    TabPFN:   MAE=${mae:.2f} [{lo:.2f},{hi:.2f}]  R2={r2:.4f}")
    except Exception as e:
        print(f"    TabPFN failed: {str(e)[:160]}")
        results["tabpfn_reg"] = {"error": str(e)[:200]}

    # Summary
    print("\n=== SUMMARY ===")
    best = None
    best_mae = float("inf")
    for k, v in results.items():
        if isinstance(v, dict) and "mae" in v:
            if v["mae"] < best_mae:
                best_mae = v["mae"]
                best = k
    print(f"  BEST: {best}  MAE=${best_mae:.2f}")
    if best != "baseline":
        improvement = (baseline_mae - best_mae) / baseline_mae * 100
        print(f"  Improvement over baseline (predict mean): {improvement:.1f}%")

    out = RESULTS / "R2_BENEFIT_FIX.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
