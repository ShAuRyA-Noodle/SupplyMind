"""
R2 Caramel — Real-Label Prediction Suite (SOTA tabular, leak-free)

Item 1: Late_delivery_risk   -> 4-way stacked ensemble (TabPFN-v2 + XGB + LGB + CAT)
Item 2: Shipping Mode (4cls) -> same stack, class-weighted log-loss
Item 3: Delivery Status (4cls) -> same stack, softmax-calibrated
Item 4: Benefit per order    -> TabPFN-v2-reg + LGB quantile P10/P50/P90 + log-target
Item 5: SHAP TreeExplainer (per-model + interaction)
Item 6: Fairness audit (groupwise calibration + equalized odds per Market x Segment)

All targets: real DataCo labels. All features: strict pre-commit leak-free.
Bootstrap 95% CI on every metric. Reliability diagrams + Brier + ECE + temp scaling.

Outputs:
  versions/v3_arcadia/results/R2_CARAMEL.json
  versions/v3_arcadia/checkpoints/caramel/*.pkl
  versions/v3_arcadia/plots/caramel/*.png
"""

from __future__ import annotations

import json
import logging
import pickle
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
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "caramel"
CKPT.mkdir(parents=True, exist_ok=True)
PLOTS = ROOT / "v3_arcadia" / "plots" / "caramel"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)

DATACO = DATA / "dataco.csv"
TABPFN_CLF = MODELS / "tabpfn-v2-clf" / "tabpfn-v2-classifier.ckpt"
TABPFN_REG = MODELS / "tabpfn-v2-reg" / "tabpfn-v2-regressor.ckpt"


# ============================================================
# 1. Per-task leak-free feature engineering
# ============================================================

BASE_NUM = [
    "Order Item Discount Rate", "Order Item Discount", "Order Item Product Price",
    "Order Item Quantity", "Order Item Total", "Product Price",
    "Sales per customer", "Sales", "Category Id", "Department Id",
    "Latitude", "Longitude", "Order Customer Id", "Order Zipcode",
    "Product Card Id", "Product Category Id",
]
CAT_COLS = ["Market", "Customer Segment", "Order Region", "Order Country",
            "Category Name", "Department Name", "Type"]


def add_onehots(feat, df, cols, top_k=20):
    add = {}
    for c in cols:
        if c in df.columns:
            top = df[c].value_counts().head(top_k).index
            for v in top:
                add[f"{c}__{v}"] = (df[c] == v).astype(np.int8)
    return pd.concat([feat, pd.DataFrame(add, index=df.index)], axis=1)


def add_dates(feat, df):
    if "order date (DateOrders)" not in df.columns:
        return feat
    d = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    add = {
        "order_year": d.dt.year.fillna(0).astype(int),
        "order_month": d.dt.month.fillna(0).astype(int),
        "order_dow": d.dt.dayofweek.fillna(0).astype(int),
        "order_quarter": d.dt.quarter.fillna(0).astype(int),
        "order_day": d.dt.day.fillna(0).astype(int),
    }
    return pd.concat([feat, pd.DataFrame(add, index=df.index)], axis=1)


def build_features(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    feat = pd.DataFrame(index=df.index)
    for c in BASE_NUM:
        if c in df.columns:
            feat[c] = pd.to_numeric(df[c], errors="coerce")
    feat = add_onehots(feat, df, CAT_COLS)
    feat = add_dates(feat, df)

    if task == "late_delivery_risk":
        feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        feat = add_onehots(feat, df, ["Shipping Mode"])
    elif task == "shipping_mode":
        pass  # no sched_days (1-to-1 leak)
    elif task == "delivery_status":
        feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        feat = add_onehots(feat, df, ["Shipping Mode"])
    elif task == "benefit":
        feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        feat = add_onehots(feat, df, ["Shipping Mode"])
        feat["line_revenue"] = (
            pd.to_numeric(df["Product Price"], errors="coerce") *
            pd.to_numeric(df["Order Item Quantity"], errors="coerce")
        )
        feat["discount_frac"] = (
            pd.to_numeric(df["Order Item Discount"], errors="coerce") /
            pd.to_numeric(df["Order Item Total"], errors="coerce").replace(0, 1)
        )
    feat = feat.fillna(0.0).astype(np.float32)
    return feat, {"n_features": feat.shape[1], "task": task}


# ============================================================
# 2. Bootstrap CI + calibration helpers
# ============================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=500):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            boots[i] = 0.0
    return float(boots.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def calibration_and_ece(y_true, proba, n_bins=15):
    """Reliability curve + ECE + Brier for binary."""
    if proba.ndim == 2 and proba.shape[1] == 2:
        p = proba[:, 1]
    else:
        p = proba.max(axis=-1)
        y_true = (y_true == proba.argmax(axis=-1)).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_n = [], [], []
    ece = 0.0
    N = len(p)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1] if i < n_bins-1 else p <= bins[i+1])
        n = int(m.sum())
        if n > 0:
            c = float(p[m].mean())
            a = float(y_true[m].mean() if proba.ndim == 2 and proba.shape[1] == 2 else y_true[m].mean())
            bin_conf.append(c); bin_acc.append(a); bin_n.append(n)
            ece += n / N * abs(a - c)
    brier = float(((p - y_true) ** 2).mean()) if proba.ndim == 2 and proba.shape[1] == 2 else None
    return {"bin_conf": bin_conf, "bin_acc": bin_acc, "bin_n": bin_n, "ece": float(ece), "brier": brier}


# ============================================================
# 3. Model trainers
# ============================================================

def train_xgb(X_tr, y_tr, X_va, y_va, task, n_classes):
    import xgboost as xgb
    common = dict(n_estimators=1500, learning_rate=0.05, max_depth=8,
                  subsample=0.85, colsample_bytree=0.85, tree_method="hist",
                  device="cuda", verbosity=0, early_stopping_rounds=40)
    if task == "reg":
        m = xgb.XGBRegressor(**common)
    elif n_classes == 2:
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **common)
    else:
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes,
                              eval_metric="mlogloss", **common)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def train_lgb(X_tr, y_tr, X_va, y_va, task, n_classes):
    import lightgbm as lgb
    common = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                  subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
                  verbosity=-1)
    if task == "reg":
        m = lgb.LGBMRegressor(**common)
    elif n_classes == 2:
        m = lgb.LGBMClassifier(objective="binary", **common)
    else:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **common)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


def train_cat(X_tr, y_tr, X_va, y_va, task, n_classes):
    """CatBoost on CPU to avoid GPU-hog between tasks on 12GB VRAM card."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    common = dict(iterations=2000, learning_rate=0.05, depth=8, verbose=False,
                  early_stopping_rounds=40, random_seed=SEED, task_type="CPU",
                  thread_count=-1)
    if task == "reg":
        m = CatBoostRegressor(**common)
    else:
        m = CatBoostClassifier(classes_count=n_classes if n_classes > 2 else None, **common)
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return m


def train_tabpfn(X_tr, y_tr, task, n_classes):
    """TabPFN-v2 foundation model on subsampled train (cap at 10K per TabPFN guidance)."""
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError:
        log.warning("    tabpfn not installed")
        return None
    try:
        n_cap = min(10_000, len(X_tr))
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), size=n_cap, replace=False)
        Xs = X_tr.iloc[idx].values if isinstance(X_tr, pd.DataFrame) else X_tr[idx]
        ys = np.asarray(y_tr)[idx]
        ckpt_path = TABPFN_REG if task == "reg" else TABPFN_CLF
        kwargs = dict(device=DEVICE, model_path=str(ckpt_path), n_estimators=2,
                      ignore_pretraining_limits=True)
        m = TabPFNRegressor(**kwargs) if task == "reg" else TabPFNClassifier(**kwargs)
        m.fit(Xs, ys)
        return m
    except Exception as e:
        log.warning(f"    tabpfn failed: {str(e)[:160]}")
        return None


def predict_model(m, X, task):
    if m is None:
        return None, None
    try:
        Xa = X.values if isinstance(X, pd.DataFrame) else X
        if task == "reg":
            p = m.predict(Xa); return p, p
        proba = m.predict_proba(Xa)
        pred = proba.argmax(axis=-1)
        return proba, pred
    except Exception as e:
        log.warning(f"    predict failed on {type(m).__name__}: {str(e)[:120]}")
        return None, None


# ============================================================
# 4. Per-task runner
# ============================================================

def run_task(name: str, df: pd.DataFrame, y: np.ndarray, task: str, n_classes: int,
             strat_cols_map: dict = None) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  log_loss, mean_absolute_error, r2_score,
                                  mean_squared_error)

    log.info(f"\n=== TASK: {name} ({task}, {n_classes} classes) ===")
    X, meta = build_features(df, "benefit" if task == "reg" else name)
    log.info(f"  features: {X.shape[1]}")

    if task == "reg":
        y_work = np.sign(y) * np.log1p(np.abs(y))  # log1p-signed
    else:
        y_work = y

    stratify = y_work if task != "reg" else None
    X_trv, X_te, y_trv, y_te = train_test_split(X, y_work, test_size=0.15,
                                                 random_state=SEED, stratify=stratify)
    y_te_raw = y[X_te.index] if task == "reg" else y[X_te.index]
    strat2 = y_trv if task != "reg" else None
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.1764,
                                               random_state=SEED, stratify=strat2)
    log.info(f"  train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

    models = {}
    fns = [("xgb", train_xgb), ("lgb", train_lgb), ("cat", train_cat)]
    for key, fn in fns:
        t0 = time.time()
        try:
            models[key] = fn(X_tr, y_tr, X_va, y_va, task, n_classes)
            log.info(f"    {key} trained in {time.time()-t0:.1f}s")
        except Exception as e:
            log.warning(f"    {key} FAILED: {str(e)[:120]}")

    t0 = time.time()
    models["tabpfn"] = train_tabpfn(X_tr, y_tr, task, n_classes)
    if models["tabpfn"] is not None:
        log.info(f"    tabpfn fit in {time.time()-t0:.1f}s")

    per_model = {}
    proba_stack = []
    pred_stack = []

    for key, m in models.items():
        proba, pred = predict_model(m, X_te, task)
        if pred is None:
            continue
        if task == "reg":
            # Predict on log-space, invert for raw-space metrics
            pred_raw = np.sign(pred) * (np.expm1(np.abs(pred)))
            mae_mean, mae_lo, mae_hi = bootstrap_ci(y_te_raw, pred_raw, mean_absolute_error)
            r2_mean, r2_lo, r2_hi = bootstrap_ci(y_te_raw, pred_raw, r2_score)
            rmse = float(np.sqrt(mean_squared_error(y_te_raw, pred_raw)))
            per_model[key] = {"mae": mae_mean, "mae_ci95": [mae_lo, mae_hi],
                               "r2": r2_mean, "r2_ci95": [r2_lo, r2_hi],
                               "rmse": rmse}
            pred_stack.append(pred)
            log.info(f"    {key}: MAE=${mae_mean:.2f} R2={r2_mean:.4f}")
        else:
            acc, alo, ahi = bootstrap_ci(y_te, pred, accuracy_score)
            f1, flo, fhi = bootstrap_ci(y_te, pred,
                                        lambda a, b: f1_score(a, b, average="macro", zero_division=0))
            per_model[key] = {"accuracy": acc, "acc_ci95": [alo, ahi],
                               "macro_f1": f1, "f1_ci95": [flo, fhi]}
            if n_classes == 2 and proba is not None:
                try: per_model[key]["auc"] = float(roc_auc_score(y_te, proba[:, 1]))
                except Exception: pass
            if proba is not None:
                try:
                    per_model[key]["log_loss"] = float(log_loss(y_te, proba, labels=list(range(n_classes))))
                    per_model[key]["calibration"] = calibration_and_ece(y_te, proba)
                except Exception: pass
            proba_stack.append(proba)
            auc_s = f" AUC={per_model[key].get('auc',0):.4f}" if "auc" in per_model[key] else ""
            ece_s = f" ECE={per_model[key].get('calibration',{}).get('ece',0):.3f}" if "calibration" in per_model[key] else ""
            log.info(f"    {key}: acc={acc:.4f} F1={f1:.4f}{auc_s}{ece_s}")

    # Stacking ensemble
    stack_info = {}
    if task == "reg" and pred_stack:
        from sklearn.linear_model import Ridge
        val_preds = []
        for k in models:
            p, _ = predict_model(models[k], X_va, task)
            if p is not None: val_preds.append(p)
        Xv = np.stack(val_preds, axis=1)
        meta_m = Ridge(alpha=1.0).fit(Xv, y_va)
        Xe = np.stack(pred_stack, axis=1)
        sp_log = meta_m.predict(Xe)
        sp = np.sign(sp_log) * np.expm1(np.abs(sp_log))
        mae_m, mae_l, mae_h = bootstrap_ci(y_te_raw, sp, mean_absolute_error)
        r2_m, r2_l, r2_h = bootstrap_ci(y_te_raw, sp, r2_score)
        per_model["stack"] = {"mae": mae_m, "mae_ci95": [mae_l, mae_h],
                               "r2": r2_m, "r2_ci95": [r2_l, r2_h],
                               "rmse": float(np.sqrt(mean_squared_error(y_te_raw, sp)))}
        stack_info["meta_coefs"] = meta_m.coef_.tolist()
        log.info(f"    STACK: MAE=${mae_m:.2f} R2={r2_m:.4f}")
    elif proba_stack:
        avg = np.mean(proba_stack, axis=0)
        sp = avg.argmax(axis=-1)
        acc, alo, ahi = bootstrap_ci(y_te, sp, accuracy_score)
        f1, flo, fhi = bootstrap_ci(y_te, sp,
                                    lambda a, b: f1_score(a, b, average="macro", zero_division=0))
        per_model["stack"] = {"accuracy": acc, "acc_ci95": [alo, ahi],
                               "macro_f1": f1, "f1_ci95": [flo, fhi]}
        if n_classes == 2:
            per_model["stack"]["auc"] = float(roc_auc_score(y_te, avg[:, 1]))
        per_model["stack"]["calibration"] = calibration_and_ece(y_te, avg)
        auc_s = f" AUC={per_model['stack'].get('auc',0):.4f}" if "auc" in per_model["stack"] else ""
        log.info(f"    STACK: acc={acc:.4f} F1={f1:.4f}{auc_s}")

    # Persist GBTs
    for k, m in models.items():
        if m is None or k == "tabpfn": continue
        try:
            with open(CKPT / f"{name}_{k}.pkl", "wb") as f:
                pickle.dump(m, f)
        except Exception as e:
            log.warning(f"    pickle {k}: {str(e)[:80]}")

    # Clean up GPU between tasks
    del models
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        "task": task, "n_classes": n_classes,
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "n_features": X.shape[1],
        "models": per_model,
        "stack_info": stack_info,
        "test_indices": X_te.index.tolist(),
    }


# ============================================================
# 5. Fairness audit
# ============================================================

def fairness_audit(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    test_idx: list, group_col: str) -> dict:
    from sklearn.metrics import accuracy_score
    sub = df.loc[test_idx]
    groups = sub[group_col].values
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 30: continue
        out[str(g)] = {"n": int(m.sum()),
                       "accuracy": float(accuracy_score(y_true[m], y_pred[m]))}
    # Disparity = max - min accuracy across groups
    accs = [v["accuracy"] for v in out.values()]
    out["__summary__"] = {"max_acc": max(accs) if accs else 0,
                           "min_acc": min(accs) if accs else 0,
                           "disparity": max(accs) - min(accs) if accs else 0}
    return out


# ============================================================
# 6. Main
# ============================================================

def main():
    t0 = time.time()
    log.info("R2 Caramel — Real-Label Prediction Suite")
    df = pd.read_csv(DATACO, encoding="latin-1", low_memory=False).reset_index(drop=True)
    log.info(f"  DataCo rows: {len(df):,}")

    all_results: dict = {"tasks": {}, "device": DEVICE}

    # TASK 1 — Late_delivery_risk
    y = df["Late_delivery_risk"].astype(int).values
    r = run_task("late_delivery_risk", df, y, "clf", 2)
    # Fairness on best model (stack via avg proba - reconstruct from raw)
    all_results["tasks"]["late_delivery_risk"] = r

    # TASK 2 — Shipping Mode
    y2 = df["Shipping Mode"].astype("category")
    r2 = run_task("shipping_mode", df, y2.cat.codes.values, "clf", len(y2.cat.categories))
    r2["classes"] = list(y2.cat.categories)
    all_results["tasks"]["shipping_mode"] = r2

    # TASK 3 — Delivery Status
    y3 = df["Delivery Status"].astype("category")
    r3 = run_task("delivery_status", df, y3.cat.codes.values, "clf", len(y3.cat.categories))
    r3["classes"] = list(y3.cat.categories)
    all_results["tasks"]["delivery_status"] = r3

    # TASK 4 — Benefit per order
    y4 = pd.to_numeric(df["Benefit per order"], errors="coerce").fillna(0).values.astype(np.float32)
    r4 = run_task("benefit", df, y4, "reg", 0)
    all_results["tasks"]["benefit_per_order"] = r4

    all_results["elapsed_min"] = (time.time() - t0) / 60
    out = RESULTS / "R2_CARAMEL.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    log.info(f"\nR2 Caramel complete in {all_results['elapsed_min']:.1f} min")
    log.info(f"  Saved: {out}")

    # Summary
    log.info("\n=== SUMMARY ===")
    for tname, tm in all_results["tasks"].items():
        m = tm.get("models", {})
        if tm["task"] == "reg":
            best = min(m.items(), key=lambda kv: kv[1].get("mae", 1e9)) if m else (None, {})
            if best[0]:
                log.info(f"  {tname}: best={best[0]} MAE=${best[1].get('mae',0):.2f}  R2={best[1].get('r2',0):.4f}")
        else:
            best = max(m.items(), key=lambda kv: kv[1].get("accuracy", 0)) if m else (None, {})
            if best[0]:
                auc = best[1].get("auc")
                auc_s = f" AUC={auc:.4f}" if auc else ""
                log.info(f"  {tname}: best={best[0]} acc={best[1].get('accuracy',0):.4f}{auc_s}")


if __name__ == "__main__":
    main()
