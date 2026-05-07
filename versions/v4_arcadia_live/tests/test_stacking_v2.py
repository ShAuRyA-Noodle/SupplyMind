"""test_stacking_v2.py — G15 regression test.

We do NOT re-run the full 30K-row DataCo pipeline in tests (too slow). Instead
we validate the STACKING FRAMEWORK on sklearn's make_classification so CI stays
fast and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.stacking_v2 import (
    StackingBenchmark, _base_learners, _fit_and_predict_proba,
)


def _tiny_stack(X, y, seed: int = 42):
    """Run the same stacking recipe on a tiny synthetic dataset."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=seed)
    factory = _base_learners(seed)
    # Restrict to 2 fast learners for test speed
    names = [n for n in ("logistic_regression", "random_forest") if n in factory]
    oof = np.zeros((len(X_tr), len(names)), dtype=np.float32)
    test_preds = np.zeros((len(X_te), len(names)), dtype=np.float32)
    base_aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for li, name in enumerate(names):
        fold_preds = np.zeros(len(X_tr), dtype=np.float32)
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = factory[name]()
            fold_preds[va_idx] = _fit_and_predict_proba(m, X_tr[tr_idx], y_tr[tr_idx], X_tr[va_idx])
        oof[:, li] = fold_preds
        full = factory[name]()
        test_preds[:, li] = _fit_and_predict_proba(full, X_tr, y_tr, X_te)
        base_aucs.append(roc_auc_score(y_te, test_preds[:, li]))
    # Weighted voting
    w = np.array(base_aucs) / max(1e-9, sum(base_aucs))
    wv_auc = roc_auc_score(y_te, test_preds @ w)
    # Stacking meta
    meta = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    meta.fit(oof, y_tr)
    stack_auc = roc_auc_score(y_te, meta.predict_proba(test_preds)[:, 1])
    return base_aucs, wv_auc, stack_auc


def test_stacking_framework_runs_and_returns_valid_auc():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    base_aucs, wv_auc, stack_auc = _tiny_stack(X, y, seed=42)
    assert all(0.5 <= a <= 1.0 for a in base_aucs)
    assert 0.5 <= wv_auc <= 1.0
    assert 0.5 <= stack_auc <= 1.0


def test_stacking_benchmark_dataclass_serializable():
    bench = StackingBenchmark(
        n_train=1000, n_test=300, n_features=20, n_folds=3,
        best_single="xgboost", best_single_auc=0.85,
        lift_stacking_vs_best_single_auc=0.005,
        lift_stacking_vs_wv_auc=0.002,
    )
    d = bench.to_dict()
    assert d["n_train"] == 1000
    assert d["best_single_auc"] == 0.85
    assert "lift_stacking_vs_best_single_auc" in d


@pytest.mark.parametrize("n_informative,n_redundant", [(10, 5), (15, 2)])
def test_stacking_is_at_least_as_good_as_wv_on_mixed_family(n_informative, n_redundant):
    """With mixed-family base learners (tree + linear), stacking should
    at worst match WV and typically beat it on synthetic data."""
    X, y = make_classification(
        n_samples=2500, n_features=25, n_informative=n_informative,
        n_redundant=n_redundant, random_state=1,
    )
    base_aucs, wv_auc, stack_auc = _tiny_stack(X, y, seed=1)
    # Stacking should not be materially worse than WV
    assert stack_auc + 0.02 >= wv_auc, \
        f"stacking AUC {stack_auc:.4f} should be within 0.02 of WV AUC {wv_auc:.4f}"
