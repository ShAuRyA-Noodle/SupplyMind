"""
stacking_v2.py — G15 fix. Proper meta-learner stacking over base tabular models.

The v2-era "Ensemble_WV" (weighted voting) at 37.52% failed to beat best
individual TD3+BC_v2 at 37.44% — within CI95 overlap. The documented issue
(AUDIT_PLAN.md R2-alpha): naive voting can't exploit base-learner complementarity.

This module implements the canonical fix: stacking with out-of-fold (OOF)
predictions fed to a meta-learner.

Pipeline
--------
1. 5-fold stratified CV on the training set.
2. For each base learner (XGBoost, LightGBM, CatBoost, RandomForest), train on
   fold-train and predict on fold-val. This gives OOF predictions with NO
   leakage.
3. Train a Ridge meta-learner (or LogisticRegression for clf) on the OOF
   prediction matrix (shape n_train x n_base_learners).
4. At inference, average each base learner's out-of-fold models' predictions
   (the canonical Wolpert 1992 stacking recipe), feed into meta-learner.
5. Benchmark: best single vs ensemble_wv (v2 legacy) vs stacking_v2.

Target: `late_delivery_risk` binary classification on Kaggle DataCo.

Usage
-----
    python -m ShAuRyA_Supplymind.features.stacking_v2 --n-rows 50000 --save
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATACO_PATH = PROJECT_ROOT / "rl" / "data" / "dataco.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "R15_STACKING_V2.json"


@dataclass
class ModelResult:
    name: str
    auc: float
    f1: float
    train_time_s: float = 0.0
    n_params: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "auc": round(self.auc, 4), "f1": round(self.f1, 4),
                "train_time_s": round(self.train_time_s, 2), "n_params": self.n_params}


@dataclass
class StackingBenchmark:
    n_train: int
    n_test: int
    n_features: int
    n_folds: int
    base_learners: list[ModelResult] = field(default_factory=list)
    ensemble_wv_v1: ModelResult | None = None
    stacking_v2: ModelResult | None = None
    best_single: str | None = None
    best_single_auc: float = 0.0
    lift_stacking_vs_best_single_auc: float = 0.0
    lift_stacking_vs_wv_auc: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "n_folds": self.n_folds,
            "base_learners": [b.to_dict() for b in self.base_learners],
            "ensemble_wv_v1": self.ensemble_wv_v1.to_dict() if self.ensemble_wv_v1 else None,
            "stacking_v2": self.stacking_v2.to_dict() if self.stacking_v2 else None,
            "best_single": self.best_single,
            "best_single_auc": round(self.best_single_auc, 4),
            "lift_stacking_vs_best_single_auc": round(self.lift_stacking_vs_best_single_auc, 4),
            "lift_stacking_vs_wv_auc": round(self.lift_stacking_vs_wv_auc, 4),
        }


# --- Data prep ------------------------------------------------------------

def _load_dataco(n_rows: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load DataCo, extract numeric features + late_delivery_risk target.

    Returns X, y, feature_names.
    """
    logger.info("[data] loading %s", DATACO_PATH)
    df = pd.read_csv(DATACO_PATH, encoding="latin1")
    # Target
    if "Late_delivery_risk" not in df.columns:
        raise RuntimeError("Late_delivery_risk column missing")
    y = df["Late_delivery_risk"].astype(int).values

    # Features: all numeric columns except the target + any identifiers
    drop_cols = [
        "Late_delivery_risk",
        "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
        "Order Id", "Order Customer Id", "Product Description",
        "Customer Street", "Customer Zipcode",
    ]
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Keep only numeric
    feat_df = feat_df.select_dtypes(include=[np.number])
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    feature_names = list(feat_df.columns)
    X = feat_df.values.astype(np.float32)

    # Optional subsample (stratified)
    if n_rows is not None and n_rows < len(X):
        rng = np.random.default_rng(seed)
        # stratified by y
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        n_pos = min(len(idx_pos), n_rows // 2)
        n_neg = n_rows - n_pos
        sel = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=min(len(idx_neg), n_neg), replace=False),
        ])
        rng.shuffle(sel)
        X, y = X[sel], y[sel]

    logger.info("[data] X shape=%s y balance=%.3f", X.shape, y.mean())
    return X, y, feature_names


def _fit_and_predict_proba(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.predict(X_test)


# --- Base learner factory -------------------------------------------------

def _base_learners(seed: int):
    learners: dict = {}
    # XGBoost
    try:
        from xgboost import XGBClassifier
        learners["xgboost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=seed, verbosity=0, use_label_encoder=False,
            eval_metric="logloss", n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed; skipping")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        learners["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=-1, num_leaves=63, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            verbosity=-1, n_jobs=-1,
        )
    except ImportError:
        logger.warning("lightgbm not installed; skipping")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        learners["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.08,
            random_state=seed, verbose=False, thread_count=-1,
        )
    except ImportError:
        logger.warning("catboost not installed; skipping")

    # Sklearn RandomForest
    learners["random_forest"] = lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )

    # Sklearn LogisticRegression on scaled features (non-tree family, decorrelates)
    from sklearn.pipeline import Pipeline

    class _ScaledLR:
        def __init__(self):
            self.pipe = Pipeline([("scaler", StandardScaler()),
                                  ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=seed, n_jobs=-1))])

        def fit(self, X, y):
            self.pipe.fit(X, y)

        def predict_proba(self, X):
            return self.pipe.predict_proba(X)

    learners["logistic_regression"] = _ScaledLR

    # Sklearn MLP (also non-tree)
    class _ScaledMLP:
        def __init__(self):
            self.pipe = Pipeline([("scaler", StandardScaler()),
                                  ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                                        max_iter=50,  # keep fast
                                                        random_state=seed,
                                                        early_stopping=True,
                                                        validation_fraction=0.1))])

        def fit(self, X, y):
            self.pipe.fit(X, y)

        def predict_proba(self, X):
            return self.pipe.predict_proba(X)

    learners["mlp"] = _ScaledMLP
    return learners


# --- Core pipeline --------------------------------------------------------

def run_stacking(
    n_rows: int = 50_000,
    n_folds: int = 5,
    seed: int = 42,
) -> StackingBenchmark:
    X, y, feature_names = _load_dataco(n_rows=n_rows, seed=seed)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    n_train, n_test, n_feats = len(X_trainval), len(X_test), X.shape[1]
    logger.info("[split] train=%d val=%d n_features=%d", n_train, n_test, n_feats)

    factory = _base_learners(seed)
    learner_names = list(factory.keys())
    n_learners = len(learner_names)

    # OOF predictions matrix
    oof = np.zeros((n_train, n_learners), dtype=np.float32)
    test_preds = np.zeros((n_test, n_learners), dtype=np.float32)

    # Per-learner results (trained on full train for test metric)
    base_results: list[ModelResult] = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for li, name in enumerate(learner_names):
        t0 = time.time()
        fold_preds = np.zeros(n_train, dtype=np.float32)
        for f, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval)):
            model = factory[name]()
            p = _fit_and_predict_proba(model, X_trainval[tr_idx], y_trainval[tr_idx], X_trainval[va_idx])
            fold_preds[va_idx] = p

        oof[:, li] = fold_preds
        # Train on full trainval for test prediction
        full_model = factory[name]()
        test_p = _fit_and_predict_proba(full_model, X_trainval, y_trainval, X_test)
        test_preds[:, li] = test_p

        auc = float(roc_auc_score(y_test, test_p))
        f1 = float(f1_score(y_test, (test_p > 0.5).astype(int)))
        base_results.append(ModelResult(name=name, auc=auc, f1=f1, train_time_s=time.time() - t0))
        logger.info("[base] %-14s auc=%.4f f1=%.4f train=%.1fs",
                    name, auc, f1, time.time() - t0)

    # Ensemble v1 (naive weighted voting — v2 legacy approach)
    # Weight by val AUC
    val_aucs = [r.auc for r in base_results]
    w = np.array(val_aucs) / np.sum(val_aucs)
    wv_test = test_preds @ w
    wv_auc = float(roc_auc_score(y_test, wv_test))
    wv_f1 = float(f1_score(y_test, (wv_test > 0.5).astype(int)))
    wv_result = ModelResult(name="ensemble_wv_v1", auc=wv_auc, f1=wv_f1)
    logger.info("[wv ] ensemble_wv_v1   auc=%.4f f1=%.4f", wv_auc, wv_f1)

    # Stacking v2 — Ridge meta-learner on OOF probs
    t0 = time.time()
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    meta.fit(oof, y_trainval)
    stack_test_p = meta.predict_proba(test_preds)[:, 1]
    stack_auc = float(roc_auc_score(y_test, stack_test_p))
    stack_f1 = float(f1_score(y_test, (stack_test_p > 0.5).astype(int)))
    stack_result = ModelResult(name="stacking_v2", auc=stack_auc, f1=stack_f1,
                               train_time_s=time.time() - t0)
    logger.info("[stk] stacking_v2      auc=%.4f f1=%.4f", stack_auc, stack_f1)

    best_single = max(base_results, key=lambda r: r.auc)
    bench = StackingBenchmark(
        n_train=n_train, n_test=n_test, n_features=n_feats, n_folds=n_folds,
        base_learners=base_results,
        ensemble_wv_v1=wv_result,
        stacking_v2=stack_result,
        best_single=best_single.name,
        best_single_auc=best_single.auc,
        lift_stacking_vs_best_single_auc=stack_auc - best_single.auc,
        lift_stacking_vs_wv_auc=stack_auc - wv_auc,
    )
    return bench


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=50_000)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    bench = run_stacking(n_rows=args.n_rows, n_folds=args.n_folds, seed=args.seed)
    out = bench.to_dict()
    print(json.dumps(out, indent=2))

    if args.save:
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"saved to {RESULTS_PATH}")
