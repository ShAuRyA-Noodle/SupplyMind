"""
Phase J — Analysis modules as TRAINED models, not formulas.

Replaces 5 hand-coded formulas with data-driven models:
  1. political_risk      — Gradient boosting on WGI 6 governance dimensions
  2. dependency_scoring  — MLP on DataCo supplier features -> late_delivery_risk
  3. financial_impact    — Regression on DataCo -> realized profit loss
  4. confidence          — Isotonic regression calibrated on real prediction outcomes
  5. safety_stock        — Empirical lead-time multiplier from DataCo distribution

(`spof` remains graph-theoretic — already real, no formula to replace.)

Legacy formula-only code preserved in:
  rl/legacy/fallbacks/analysis_formulas.py

Run:
    python -m rl.analysis.trained_models
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "trained"
MODELS_DIR.mkdir(exist_ok=True)

WGI_PATH = ROOT / "external_data" / "wgi" / "wgidataset_with_sourcedata-2025.xlsx"
DATACO_PATH = ROOT / "rl" / "data" / "dataco.csv"


# ============================================================
# 1. Political Risk — Gradient Boosting on WGI 6 dimensions
# ============================================================

WGI_SHEETS = ["va", "pv", "ge", "rq", "rl", "cc"]
# va=Voice/Accountability, pv=Political Stability/Absence of Violence,
# ge=Gov Effectiveness, rq=Regulatory Quality, rl=Rule of Law,
# cc=Control of Corruption

# Non-circular target: predict the held-out Political Stability dimension (pv)
# from the OTHER five governance indicators. pv is not a function of the
# feature set, so the reported R2 is a genuine cross-indicator predictability
# metric rather than the tautology y = f(X) that a mean-of-features target gives.
POL_FEATURE_SHEETS = ["va", "ge", "rq", "rl", "cc"]
POL_TARGET_SHEET = "pv"


def train_political_risk():
    log.info("[1/5] political_risk: training on WGI (predict held-out pv from other 5 dims)...")
    xls = pd.ExcelFile(WGI_PATH)
    frames = {}
    for sheet in WGI_SHEETS:
        df = pd.read_excel(xls, sheet_name=sheet)
        frames[sheet] = df[["Economy (code)", "Year", "Governance score (0-100)"]].rename(
            columns={"Economy (code)": "iso", "Governance score (0-100)": sheet}
        )
    # Merge by iso+year
    merged = frames[WGI_SHEETS[0]]
    for sheet in WGI_SHEETS[1:]:
        merged = merged.merge(frames[sheet], on=["iso", "Year"], how="inner")
    # Keep most recent year per country
    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce")
    merged = merged.dropna()
    latest = merged.sort_values("Year").groupby("iso").tail(1).reset_index(drop=True)
    log.info(f"  WGI merged: {len(latest)} countries, most recent year={int(latest['Year'].max())}")

    # Features: 5 governance dims (excluding the target). Target: held-out pv.
    X = latest[POL_FEATURE_SHEETS].values
    y = latest[POL_TARGET_SHEET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    log.info(f"  political_risk: predicting pv, MAE={mae:.4f}, R2={r2:.4f}")

    # Country risk lookup grounded in ACTUAL Political Stability (pv): lower
    # stability -> higher political risk. This is real WGI data, not the model's
    # self-prediction.
    lookup = {row["iso"]: float((100 - row[POL_TARGET_SHEET]) / 100.0) for _, row in latest.iterrows()}
    with open(MODELS_DIR / "political_risk_gbr.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": POL_FEATURE_SHEETS,
                "target": POL_TARGET_SHEET,
                "risk_definition": "(100 - actual_pv) / 100",
                "country_lookup": lookup,
            },
            f,
        )
    log.info(f"  saved political_risk_gbr.pkl, {len(lookup)} countries")

    return {"mae": mae, "r2": r2, "target": POL_TARGET_SHEET, "countries": len(lookup)}


# ============================================================
# 2. Dependency Scoring — MLP on DataCo supplier features
# ============================================================

def train_dependency_scoring(df: pd.DataFrame):
    log.info("[2/5] dependency_scoring: training MLP...")
    # Order-time features only. Late_delivery_risk is defined by realized
    # (real - scheduled) shipping days, so "Days for shipping (real)" and the
    # realized "Order Item Profit Ratio" are label leakage and are EXCLUDED.
    # These features are all knowable at order placement.
    feat_cols = [
        "Days for shipment (scheduled)",
        "Order Item Discount Rate",
        "Order Item Quantity",
        "Sales per customer",
    ]
    df2 = df.dropna(subset=feat_cols + ["Late_delivery_risk"])
    X = df2[feat_cols].values.astype(np.float32)
    y = df2["Late_delivery_risk"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=42, early_stopping=True)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    log.info(f"  dependency_scoring: test accuracy={acc:.4f}")

    with open(MODELS_DIR / "dependency_scoring_mlp.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feat_cols}, f)
    return {"accuracy": acc, "n_train": len(X_tr)}


# ============================================================
# 3. Financial Impact — Ridge regression on DataCo
# ============================================================

def train_financial_impact(df: pd.DataFrame):
    log.info("[3/5] financial_impact: training Ridge regression...")
    df2 = df.dropna(subset=[
        "Order Item Total", "Days for shipping (real)", "Days for shipment (scheduled)",
        "Order Item Profit Ratio", "Benefit per order",
    ])
    delay = df2["Days for shipping (real)"] - df2["Days for shipment (scheduled)"]
    X = np.stack([
        df2["Order Item Total"].values,
        delay.values,
        df2["Order Item Profit Ratio"].values,
        df2["Late_delivery_risk"].astype(float).values,
    ], axis=1).astype(np.float32)
    y = df2["Benefit per order"].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    log.info(f"  financial_impact: MAE=${mae:.2f}, R2={r2:.4f}")

    with open(MODELS_DIR / "financial_impact_ridge.pkl", "wb") as f:
        pickle.dump({"model": model, "features": ["order_total", "delay_days", "profit_ratio", "late_risk"]}, f)
    return {"mae": float(mae), "r2": float(r2)}


# ============================================================
# 4. Confidence — Isotonic regression calibration
# ============================================================

def train_confidence(df: pd.DataFrame):
    log.info("[4/5] confidence: isotonic calibration on real outcomes...")
    # Proxy: predict-probability = profit_ratio shifted+scaled; actual = late_delivery_risk
    df2 = df.dropna(subset=["Order Item Profit Ratio", "Late_delivery_risk"])
    raw_score = -df2["Order Item Profit Ratio"].clip(-1, 1).values  # higher = more risk
    # Normalize to [0,1]
    raw_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
    actual = df2["Late_delivery_risk"].astype(int).values

    # Calibrate
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_score, actual)

    # Reliability
    calibrated = iso.predict(raw_score)
    bins = np.linspace(0, 1, 11)
    reliability = []
    for i in range(10):
        mask = (calibrated >= bins[i]) & (calibrated < bins[i + 1])
        if mask.sum() > 0:
            reliability.append((float((bins[i] + bins[i + 1]) / 2), float(actual[mask].mean())))
    ece = np.mean([abs(a - b) for a, b in reliability])
    log.info(f"  confidence: ECE={ece:.4f}")

    with open(MODELS_DIR / "confidence_isotonic.pkl", "wb") as f:
        pickle.dump({"model": iso, "reliability": reliability}, f)
    return {"ece": float(ece)}


# ============================================================
# 5. Safety Stock — Empirical multiplier from DataCo lead-time
# ============================================================

def train_safety_stock(df: pd.DataFrame):
    log.info("[5/5] safety_stock: fitting empirical multiplier...")
    df2 = df.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)"])
    lead_time = df2["Days for shipping (real)"].values
    mean_lt = float(np.mean(lead_time))
    std_lt = float(np.std(lead_time))
    # Service-level multipliers (z-scores): 95%=1.645, 99%=2.326
    multipliers = {
        "p90": 1.282 * std_lt / max(mean_lt, 1e-6),
        "p95": 1.645 * std_lt / max(mean_lt, 1e-6),
        "p99": 2.326 * std_lt / max(mean_lt, 1e-6),
    }
    log.info(f"  safety_stock: mean_lt={mean_lt:.2f}, std_lt={std_lt:.2f}, mult_p95={multipliers['p95']:.3f}")

    with open(MODELS_DIR / "safety_stock_empirical.pkl", "wb") as f:
        pickle.dump({"mean_lead_time": mean_lt, "std_lead_time": std_lt, "multipliers": multipliers}, f)
    return {"mean_lead_time": mean_lt, "std_lead_time": std_lt, **multipliers}


# ============================================================
# Main
# ============================================================

def main():
    log.info("Phase J: training all 5 analysis modules on real data")

    results = {}
    results["political_risk"] = train_political_risk()

    log.info("Loading DataCo for modules 2-5...")
    df = pd.read_csv(DATACO_PATH, encoding="latin-1", low_memory=False)
    log.info(f"DataCo: {len(df)} orders")

    results["dependency_scoring"] = train_dependency_scoring(df)
    results["financial_impact"] = train_financial_impact(df)
    results["confidence"] = train_confidence(df)
    results["safety_stock"] = train_safety_stock(df)

    report = MODELS_DIR / "phase_j_results.json"
    report.write_text(json.dumps(results, indent=2))
    log.info(f"Phase J complete. Report: {report}")
    log.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
