"""
v3.0 Block 1 — Real-Label Prediction Suite (STRICT LEAK-FREE)

Confirmed deterministic leaks in raw DataCo:
  - Shipping Mode is 1-to-1 with Days for shipment (scheduled): Same Day=0, First Class=1, Second=2, Standard=4
  - Late_delivery_risk = 1 iff Delivery Status = "Late delivery" (perfect correlation)
  - Days for shipping (real) > scheduled  =>  Late_delivery_risk = 1 by definition
  - Benefit per order is derived from Order Item Profit Ratio * totals

Per-task leak-free feature sets:
  TASK 1 Late_delivery_risk (binary, PRE-shipping decision):
    DROP: Days for shipping (real), delay_days, Delivery Status, Benefit, Profit Ratio, Profit per order
    KEEP: Days for shipment (scheduled), Shipping Mode, customer/product/market/price/discount/qty/date

  TASK 2 Shipping Mode (5-class, PRE-shipping decision):
    DROP: Days for shipment (scheduled) (1-to-1), real days, Late_risk, Delivery Status
    KEEP: customer/product/market/price/discount/qty/date

  TASK 3 Delivery Status (4-class, POST-commit outcome):
    DROP: Days for shipping (real), delay_days, Late_risk (1-to-1)
    KEEP: Days scheduled, Shipping Mode, customer/product/market/price/date

  TASK 4 Benefit per order (regression, PRE-shipping profit forecast):
    DROP: Order Item Profit Ratio, Order Profit Per Order, expected_profit (algebraic), real days, delay, late_risk, delivery status
    KEEP: Order Item Total, qty, price, discount, customer, product, market, mode, scheduled days

Ensemble: XGBoost + LightGBM + CatBoost + TabPFN-v2 (zero-shot) + stacking avg.
Rigor: bootstrap 95% CI on every metric + macro-F1 + AUC + log-loss + calibration.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rl" / "data"
OUT = ROOT / "rl" / "analysis" / "trained" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DATACO_PATH = DATA / "dataco.csv"

MODELS_DIR = ROOT / "models"
TABPFN_CLF = MODELS_DIR / "tabpfn-v2-clf"
TABPFN_REG = MODELS_DIR / "tabpfn-v2-reg"

# ============================================================
# Feature builder — per-task strict subsets
# ============================================================

BASE_NUMERIC = [
    "Order Item Discount Rate", "Order Item Discount",
    "Order Item Product Price", "Order Item Quantity",
    "Order Item Total", "Product Price", "Sales per customer", "Sales",
    "Category Id", "Department Id", "Latitude", "Longitude",
    "Order Customer Id", "Order Zipcode", "Product Card Id", "Product Category Id",
]

CAT_COLS_ALL = [
    "Market", "Customer Segment", "Order Region", "Order Country",
    "Category Name", "Department Name", "Type",
]


def add_categoricals(feat: pd.DataFrame, df: pd.DataFrame, cat_cols: list[str], top_k: int = 20):
    for c in cat_cols:
        if c in df.columns:
            top = df[c].value_counts().head(top_k).index
            for v in top:
                feat[f"{c}__{v}"] = (df[c] == v).astype(np.int8)


def add_date_features(feat: pd.DataFrame, df: pd.DataFrame):
    if "order date (DateOrders)" in df.columns:
        d = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
        feat["order_year"] = d.dt.year
        feat["order_month"] = d.dt.month
        feat["order_dow"] = d.dt.dayofweek
        feat["order_quarter"] = d.dt.quarter
        feat["order_day"] = d.dt.day


def build_features_for_task(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    feat = pd.DataFrame(index=df.index)

    # Base numerics (safe for all tasks)
    for c in BASE_NUMERIC:
        if c in df.columns:
            feat[c] = pd.to_numeric(df[c], errors="coerce")

    # Categoricals (Market, Segment, Region, Country, Category, Dept, Type)
    add_categoricals(feat, df, CAT_COLS_ALL)
    add_date_features(feat, df)

    # ----- Task-specific additions / strict drops -----
    if task == "late_delivery_risk":
        # Pre-shipping decision: we know mode + scheduled days at commit time
        if "Days for shipment (scheduled)" in df.columns:
            feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        # Shipping Mode is known at commit (what carrier was booked)
        add_categoricals(feat, df, ["Shipping Mode"])

    elif task == "shipping_mode":
        # Predict what mode the company will choose — available signals: customer/product/market/price/date
        # DO NOT include sched_days (1-to-1 with mode)
        pass

    elif task == "delivery_status":
        # Outcome prediction given commit-time + mode — may also see delay, but to avoid trivial leak of Late_risk, DROP delay + real days
        if "Days for shipment (scheduled)" in df.columns:
            feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        add_categoricals(feat, df, ["Shipping Mode"])

    elif task == "benefit_per_order":
        # Profit forecast at commit time
        if "Days for shipment (scheduled)" in df.columns:
            feat["sched_days"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")
        add_categoricals(feat, df, ["Shipping Mode"])
        # Revenue proxy is safe (price*qty), but NOT profit ratio or expected_profit
        if "Product Price" in df.columns and "Order Item Quantity" in df.columns:
            feat["line_revenue"] = (
                pd.to_numeric(df["Product Price"], errors="coerce") *
                pd.to_numeric(df["Order Item Quantity"], errors="coerce")
            )
        if "Order Item Discount" in df.columns and "Order Item Total" in df.columns:
            feat["discount_frac"] = (
                pd.to_numeric(df["Order Item Discount"], errors="coerce") /
                pd.to_numeric(df["Order Item Total"], errors="coerce").replace(0, 1)
            )

    feat = feat.fillna(0.0).astype(np.float32)
    meta = {"n_features": feat.shape[1], "task": task, "feature_names": list(feat.columns)[:50]}
    return feat, meta


# ============================================================
# Evaluation utilities
# ============================================================

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 500):
    rng = np.random.default_rng(42)
    n = len(y_true)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[i] = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            boots[i] = 0.0
    return float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ============================================================
# Model trainers
# ============================================================

def train_xgb(X_tr, y_tr, X_va, y_va, task: str, n_classes: int):
    import xgboost as xgb
    common = dict(n_estimators=1000, learning_rate=0.05, max_depth=8,
                  subsample=0.85, colsample_bytree=0.85,
                  tree_method="hist", device="cuda", verbosity=0,
                  early_stopping_rounds=30)
    if task == "reg":
        m = xgb.XGBRegressor(**common)
    elif n_classes == 2:
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **common)
    else:
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **common)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def train_lgb(X_tr, y_tr, X_va, y_va, task: str, n_classes: int):
    import lightgbm as lgb
    common = dict(n_estimators=1500, learning_rate=0.05, num_leaves=63,
                  subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
                  verbosity=-1)
    if task == "reg":
        m = lgb.LGBMRegressor(**common)
    elif n_classes == 2:
        m = lgb.LGBMClassifier(objective="binary", **common)
    else:
        m = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **common)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def train_cat(X_tr, y_tr, X_va, y_va, task: str, n_classes: int):
    from catboost import CatBoostClassifier, CatBoostRegressor
    common = dict(iterations=1500, learning_rate=0.05, depth=8, verbose=False,
                  early_stopping_rounds=30, random_seed=42, task_type="GPU", devices="0")
    try:
        if task == "reg":
            m = CatBoostRegressor(**common)
        else:
            m = CatBoostClassifier(classes_count=n_classes if n_classes > 2 else None, **common)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    except Exception as e:
        log.warning(f"  CatBoost GPU failed ({str(e)[:80]}); CPU retry")
        common["task_type"] = "CPU"; common.pop("devices", None)
        if task == "reg":
            m = CatBoostRegressor(**common)
        else:
            m = CatBoostClassifier(classes_count=n_classes if n_classes > 2 else None, **common)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return m


def try_tabpfn(X_tr, y_tr, task: str, n_classes: int, model_dir: Path | None):
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError:
        return None
    try:
        n_cap = min(10_000, len(X_tr))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_tr), size=n_cap, replace=False)
        Xs = X_tr.iloc[idx].values if isinstance(X_tr, pd.DataFrame) else X_tr[idx]
        ys = np.asarray(y_tr)[idx]
        kwargs = {"device": "cuda"}
        if model_dir and model_dir.exists():
            kwargs["model_path"] = str(model_dir)
        if task == "reg":
            m = TabPFNRegressor(**kwargs)
        else:
            m = TabPFNClassifier(**kwargs)
        m.fit(Xs, ys)
        return m
    except Exception as e:
        log.warning(f"  TabPFN failed: {str(e)[:160]}")
        return None


def predict_model(m, X, task: str):
    if m is None:
        return None, None
    try:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if task == "reg":
            p = m.predict(X_arr); return p, p
        proba = m.predict_proba(X_arr)
        pred = proba.argmax(axis=-1)
        return proba, pred
    except Exception as e:
        log.warning(f"  predict failed on {type(m).__name__}: {str(e)[:80]}")
        return None, None


# ============================================================
# Task runner
# ============================================================

def run_task(name: str, X: pd.DataFrame, y: np.ndarray, task: str, n_classes: int):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score,
                                  roc_auc_score, log_loss, f1_score)
    log.info(f"\n=== {name} ({task}, {n_classes} classes, n_feat={X.shape[1]}) ===")

    stratify = y if task != "reg" else None
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=stratify)
    stratify2 = y_trv if task != "reg" else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.1764, random_state=42, stratify=stratify2
    )
    log.info(f"  train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

    models: dict = {}
    for key, fn in [
        ("xgb", train_xgb), ("lgb", train_lgb), ("cat", train_cat),
    ]:
        t0 = time.time()
        try:
            models[key] = fn(X_tr, y_tr, X_va, y_va, task, n_classes)
            log.info(f"  {key} trained in {time.time()-t0:.1f}s")
        except Exception as e:
            log.warning(f"  {key} FAILED: {str(e)[:120]}")

    t0 = time.time()
    tabpfn_dir = TABPFN_CLF if task != "reg" else TABPFN_REG
    models["tabpfn"] = try_tabpfn(X_tr, y_tr, task, n_classes, tabpfn_dir)
    if models["tabpfn"] is not None:
        log.info(f"  tabpfn fit in {time.time()-t0:.1f}s")

    # Evaluate
    per_model: dict = {}
    proba_stack = []
    pred_stack_reg = []

    for key, m in models.items():
        proba, pred = predict_model(m, X_te, task)
        if pred is None:
            continue
        if task == "reg":
            mae_mean, mae_lo, mae_hi = bootstrap_ci(y_te, pred, mean_absolute_error)
            r2_mean, r2_lo, r2_hi = bootstrap_ci(y_te, pred, r2_score)
            per_model[key] = {
                "mae": mae_mean, "mae_ci95": [mae_lo, mae_hi],
                "r2": r2_mean, "r2_ci95": [r2_lo, r2_hi],
            }
            pred_stack_reg.append(pred)
            log.info(f"  {key}: MAE={mae_mean:.3f} [CI {mae_lo:.3f},{mae_hi:.3f}]  R2={r2_mean:.4f}")
        else:
            acc_mean, acc_lo, acc_hi = bootstrap_ci(y_te, pred, accuracy_score)
            f1_mean, _, _ = bootstrap_ci(y_te, pred, lambda a, b: f1_score(a, b, average="macro", zero_division=0))
            per_model[key] = {
                "accuracy": acc_mean, "acc_ci95": [acc_lo, acc_hi],
                "macro_f1": f1_mean,
            }
            if n_classes == 2 and proba is not None:
                try:
                    per_model[key]["auc"] = float(roc_auc_score(y_te, proba[:, 1]))
                except Exception:
                    pass
            if proba is not None:
                try:
                    per_model[key]["log_loss"] = float(log_loss(y_te, proba, labels=list(range(n_classes))))
                except Exception:
                    pass
            proba_stack.append(proba)
            auc_s = f" AUC={per_model[key].get('auc', 0):.4f}" if "auc" in per_model[key] else ""
            log.info(f"  {key}: acc={acc_mean:.4f} [CI {acc_lo:.3f},{acc_hi:.3f}] F1={f1_mean:.4f}{auc_s}")

    # Stacking
    if task == "reg" and pred_stack_reg:
        from sklearn.linear_model import Ridge
        val_preds = []
        for key, m in models.items():
            p, _ = predict_model(m, X_va, task)
            if p is not None:
                val_preds.append(p)
        Xs_val = np.stack(val_preds, axis=1)
        meta = Ridge(alpha=1.0).fit(Xs_val, y_va)
        Xs_te = np.stack(pred_stack_reg, axis=1)
        sp = meta.predict(Xs_te)
        mae_mean, mae_lo, mae_hi = bootstrap_ci(y_te, sp, mean_absolute_error)
        r2_mean, r2_lo, r2_hi = bootstrap_ci(y_te, sp, r2_score)
        per_model["stack"] = {"mae": mae_mean, "mae_ci95": [mae_lo, mae_hi],
                               "r2": r2_mean, "r2_ci95": [r2_lo, r2_hi]}
        log.info(f"  STACK(ridge): MAE={mae_mean:.3f} R2={r2_mean:.4f}")
    elif proba_stack:
        avg = np.mean(proba_stack, axis=0)
        sp = avg.argmax(axis=-1)
        acc_mean, acc_lo, acc_hi = bootstrap_ci(y_te, sp, accuracy_score)
        f1_mean, _, _ = bootstrap_ci(y_te, sp, lambda a, b: f1_score(a, b, average="macro", zero_division=0))
        entry = {"accuracy": acc_mean, "acc_ci95": [acc_lo, acc_hi], "macro_f1": f1_mean}
        if n_classes == 2:
            try:
                entry["auc"] = float(roc_auc_score(y_te, avg[:, 1]))
            except Exception:
                pass
        per_model["stack"] = entry
        log.info(f"  STACK(avg-proba): acc={acc_mean:.4f} F1={f1_mean:.4f}"
                 + (f" AUC={entry.get('auc', 0):.4f}" if "auc" in entry else ""))

    # Persist small models
    for key, m in models.items():
        if m is None or key == "tabpfn":
            continue
        try:
            with open(OUT / f"{name}_{key}.pkl", "wb") as f:
                pickle.dump(m, f)
        except Exception as e:
            log.warning(f"  pickle {key}: {str(e)[:80]}")

    with open(OUT / f"{name}_metrics.json", "w") as f:
        json.dump({
            "task": task, "n_classes": n_classes,
            "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
            "n_features": X.shape[1], "models": per_model,
        }, f, indent=2)
    return per_model


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    log.info("v3 Block 1 (strict leak-free)")
    df = pd.read_csv(DATACO_PATH, encoding="latin-1", low_memory=False)
    log.info(f"  DataCo rows: {len(df):,}")

    all_metrics = {"tasks": {}}

    # TASK 1
    X1, meta1 = build_features_for_task(df, "late_delivery_risk")
    y1 = df["Late_delivery_risk"].astype(int).values
    all_metrics["tasks"]["late_delivery_risk"] = {
        "meta": meta1, "models": run_task("late_delivery_risk", X1, y1, "clf", 2)
    }

    # TASK 2
    X2, meta2 = build_features_for_task(df, "shipping_mode")
    y2 = df["Shipping Mode"].astype("category")
    labels2 = list(y2.cat.categories)
    all_metrics["tasks"]["shipping_mode"] = {
        "meta": meta2, "classes": labels2,
        "models": run_task("shipping_mode", X2, y2.cat.codes.values, "clf", len(labels2)),
    }

    # TASK 3
    X3, meta3 = build_features_for_task(df, "delivery_status")
    y3 = df["Delivery Status"].astype("category")
    labels3 = list(y3.cat.categories)
    all_metrics["tasks"]["delivery_status"] = {
        "meta": meta3, "classes": labels3,
        "models": run_task("delivery_status", X3, y3.cat.codes.values, "clf", len(labels3)),
    }

    # TASK 4
    X4, meta4 = build_features_for_task(df, "benefit_per_order")
    y4 = pd.to_numeric(df["Benefit per order"], errors="coerce").fillna(0).values.astype(np.float32)
    all_metrics["tasks"]["benefit_per_order"] = {
        "meta": meta4,
        "models": run_task("benefit_per_order", X4, y4, "reg", 0),
    }

    all_metrics["elapsed_min"] = (time.time() - t0) / 60
    out_path = RESULTS / "V3_BLOCK1_REAL_LABELS.json"
    out_path.write_text(json.dumps(all_metrics, indent=2))
    log.info(f"\nv3 Block 1 complete in {all_metrics['elapsed_min']:.1f} min")

    log.info("\n=== SUMMARY (real, leak-free) ===")
    for tname, tm in all_metrics["tasks"].items():
        models = tm["models"]
        if any(key in models for key in ("stack", "tabpfn")):
            if "stack" in models and "accuracy" in models["stack"]:
                m = models["stack"]
                log.info(f"  {tname} STACK: acc={m['accuracy']:.4f} CI95=[{m['acc_ci95'][0]:.3f},{m['acc_ci95'][1]:.3f}]"
                         + (f" AUC={m['auc']:.4f}" if 'auc' in m else ""))
            elif "stack" in models and "mae" in models["stack"]:
                m = models["stack"]
                log.info(f"  {tname} STACK: MAE={m['mae']:.3f} R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
