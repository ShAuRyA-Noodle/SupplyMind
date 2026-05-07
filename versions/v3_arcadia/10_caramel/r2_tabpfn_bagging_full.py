"""R2-α FULL — TabPFN bagging at scale (10 disjoint bags × 12K samples).

Unlike r2_tabpfn_bagging.py (3-bag demo), this runs the full-scale bagging
designed to extract genuine ensemble lift. 10 bags × 12K samples = 120K
effective coverage of the 126K train set.

Output:
  versions/v3_arcadia/results/R2_TABPFN_BAGGING_FULL.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "rl" / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "v3_arcadia" / "results"

TABPFN_CLF = MODELS / "tabpfn-v2-clf" / "tabpfn-v2-classifier.ckpt"

SEED = 42
N_BAGS = 10
BAG_SIZE = 12_000

sys.path.insert(0, str(ROOT / "v3_arcadia" / "10_caramel"))
from train_caramel import build_features


def load_dataco():
    return pd.read_csv(DATA / "dataco.csv", encoding="latin-1", low_memory=False).reset_index(drop=True)


def train_tabpfn_bag(X_tr, y_tr, seed):
    from tabpfn import TabPFNClassifier
    import torch
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=min(BAG_SIZE, len(X_tr)), replace=False)
    Xs = X_tr.iloc[idx].values if hasattr(X_tr, "iloc") else X_tr[idx]
    ys = np.asarray(y_tr)[idx]
    dev = "cpu" if os.environ.get("R2_FORCE_CPU", "") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")
    m = TabPFNClassifier(device=dev, model_path=str(TABPFN_CLF),
                          n_estimators=2, ignore_pretraining_limits=True)
    m.fit(Xs, ys)
    return m


def main():
    t0 = time.time()
    log.info(f"R2-α FULL — {N_BAGS} bags × {BAG_SIZE} samples TabPFN bagging")

    df = load_dataco()
    from sklearn.model_selection import train_test_split
    X, meta = build_features(df, "late_delivery_risk")
    y = df["Late_delivery_risk"].astype(int).values
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.1764, random_state=SEED, stratify=y_trv)
    log.info(f"Train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

    X_te_arr = X_te.values if hasattr(X_te, "values") else X_te
    proba_accum = np.zeros((len(X_te), 2))
    bag_accs = []
    bag_times = []
    ok = 0
    for i in range(N_BAGS):
        log.info(f"\nBag {i + 1}/{N_BAGS}...")
        bt = time.time()
        try:
            m = train_tabpfn_bag(X_tr, y_tr, seed=SEED + i * 31)
            proba = m.predict_proba(X_te_arr)
            proba_accum += proba
            ok += 1
            pred = proba.argmax(axis=-1)
            acc = float((pred == y_te).mean())
            bag_accs.append(acc)
            bag_times.append(time.time() - bt)
            log.info(f"  accuracy={acc:.4f}  elapsed={bag_times[-1]:.1f}s")
        except Exception as e:
            log.warning(f"  Bag failed: {str(e)[:200]}")
            bag_accs.append(None)
            bag_times.append(time.time() - bt)

    if ok == 0:
        log.error("All bags failed; not writing result file.")
        return
    proba_mean = proba_accum / ok
    pred_mean = proba_mean.argmax(axis=-1)
    bagged_acc = float((pred_mean == y_te).mean())
    single_mean = float(np.mean([a for a in bag_accs if a is not None]))
    lift_pp = (bagged_acc - single_mean) * 100

    out = {
        "task": "late_delivery_risk binary classification",
        "n_bags_succeeded": ok,
        "n_bags_requested": N_BAGS,
        "bag_size": BAG_SIZE,
        "per_bag_accuracy": bag_accs,
        "per_bag_elapsed_s": bag_times,
        "single_bag_mean_accuracy": single_mean,
        "bagged_accuracy": bagged_acc,
        "bagging_lift_pp": lift_pp,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = RESULTS / "R2_TABPFN_BAGGING_FULL.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"\nSaved: {out_path}  ({out['elapsed_min']:.1f} min)  lift={lift_pp:+.2f}pp")


if __name__ == "__main__":
    main()
