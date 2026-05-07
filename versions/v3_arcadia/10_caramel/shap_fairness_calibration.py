"""R2 Caramel — items 5, 6, 7, 8: SHAP interactions + fairness audit + calibration curves."""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "caramel"
PLOTS = ROOT / "v3_arcadia" / "plots" / "caramel"
PLOTS.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "v3_arcadia" / "results"

import sys
sys.path.insert(0, str(ROOT / "v3_arcadia" / "10_caramel"))
from train_caramel import build_features, SEED

from sklearn.model_selection import train_test_split


def load_model(task: str, algo: str):
    p = CKPT / f"{task}_{algo}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ============================================================
# SHAP TreeExplainer on best-single models
# ============================================================
def shap_task(task: str, algo: str, df: pd.DataFrame, y: np.ndarray, n_samples: int = 1000):
    import shap
    m = load_model(task, algo)
    if m is None:
        return {"error": "model not found"}
    X, _ = build_features(df, task)
    # Split to get test set (same seed)
    X_trv, X_te, _, _ = train_test_split(X, y, test_size=0.15, random_state=SEED,
                                         stratify=(y if len(np.unique(y)) < 20 else None))
    # Take subsample for SHAP (SHAP is O(N*trees))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_te), size=min(n_samples, len(X_te)), replace=False)
    X_s = X_te.iloc[idx]
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(X_s)
    # sv shape: [N, F] binary OR [N, F, C] multiclass OR list[N,F] per class
    if isinstance(sv, list):  # old API
        sv = np.stack(sv, axis=-1)
    if sv.ndim == 3:
        imp = np.mean(np.abs(sv), axis=(0, 2))
    else:
        imp = np.mean(np.abs(sv), axis=0)
    top = np.argsort(imp)[::-1][:15]
    feats = list(X.columns)
    top_feats = [{"name": feats[i], "importance": float(imp[i])} for i in top]
    return {"algo": algo, "top15_features": top_feats, "n_samples": int(len(X_s))}


# ============================================================
# Fairness audit (groupwise accuracy per Market + Segment)
# ============================================================
def fairness_task(task: str, algo: str, df: pd.DataFrame, y: np.ndarray):
    from sklearn.metrics import accuracy_score
    m = load_model(task, algo)
    if m is None:
        return {"error": "model not found"}
    X, _ = build_features(df, task)
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED,
                                                 stratify=(y if len(np.unique(y)) < 20 else None))
    pred = m.predict(X_te.values if hasattr(X_te, "values") else X_te)
    sub_df = df.loc[X_te.index]
    out = {}
    for col in ["Market", "Customer Segment"]:
        groups = sub_df[col].astype(str).values
        per = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < 30: continue
            per[str(g)] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_te[mask], pred[mask])),
            }
        if per:
            accs = [v["accuracy"] for v in per.values()]
            per["__summary__"] = {"max_acc": max(accs), "min_acc": min(accs),
                                   "disparity": max(accs) - min(accs)}
        out[col] = per
    return out


# ============================================================
# Calibration curves (reliability diagrams + temperature scaling)
# ============================================================
def calibration_task(task: str, algo: str, df: pd.DataFrame, y: np.ndarray,
                     n_bins: int = 15) -> dict:
    m = load_model(task, algo)
    if m is None:
        return {"error": "model not found"}
    X, _ = build_features(df, task)
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED,
                                                 stratify=(y if len(np.unique(y)) < 20 else None))
    proba = m.predict_proba(X_te.values if hasattr(X_te, "values") else X_te)
    # For binary use proba[:,1]; for multi use max-prob + correctness
    n_classes = proba.shape[1]
    if n_classes == 2:
        conf = proba[:, 1]
        y_bin = (y_te > 0.5).astype(int)
    else:
        conf = proba.max(axis=-1)
        y_bin = (proba.argmax(axis=-1) == y_te).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_c, bin_a, bin_n = [], [], []
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i+1] if i < n_bins-1 else conf <= bins[i+1])
        n = int(mask.sum())
        if n > 0:
            c = float(conf[mask].mean())
            a = float(y_bin[mask].mean())
            bin_c.append(c); bin_a.append(a); bin_n.append(n)
            ece += n / N * abs(a - c)
    brier = float(((conf - y_bin) ** 2).mean())
    # Temperature scaling on val for ECE improvement
    from scipy.optimize import minimize_scalar
    def neg_log_lik(T):
        logits = np.log(np.clip(proba, 1e-7, 1 - 1e-7))
        scaled = logits / T
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        e = np.exp(scaled)
        p = e / e.sum(axis=-1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y_te)), y_te], 1e-7, 1)))
    try:
        res = minimize_scalar(neg_log_lik, bounds=(0.1, 10.0), method="bounded")
        T_opt = float(res.x)
    except Exception:
        T_opt = 1.0
    return {"algo": algo, "n_bins": n_bins,
            "bin_confidence": bin_c, "bin_accuracy": bin_a, "bin_n": bin_n,
            "ece": float(ece), "brier": brier,
            "temperature_scaling_T": T_opt}


# ============================================================
# Plot reliability
# ============================================================
def plot_reliability(cal_tasks: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(cal_tasks), figsize=(5 * len(cal_tasks), 4))
        if len(cal_tasks) == 1: ax = [ax]
        for i, (tname, c) in enumerate(cal_tasks.items()):
            a = ax[i]
            if "bin_confidence" in c:
                a.plot(c["bin_confidence"], c["bin_accuracy"], "o-",
                       label=f"{c['algo']} (ECE={c['ece']:.3f})")
                a.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
                a.set_xlabel("confidence"); a.set_ylabel("accuracy")
                a.set_title(tname); a.legend(); a.grid(alpha=0.3)
                a.set_xlim(0, 1); a.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"  plot failed: {e}")
        return False


def main():
    import time
    t0 = time.time()
    print("R2 Caramel follow-up: SHAP + fairness + calibration")
    df = pd.read_csv(ROOT / "rl" / "data" / "dataco.csv",
                     encoding="latin-1", low_memory=False).reset_index(drop=True)

    # Task best-model mapping (based on previous results)
    best = {
        "late_delivery_risk": "xgb",
        "shipping_mode":      "lgb",
        "delivery_status":    "lgb",
    }

    results = {"shap_top15": {}, "fairness": {}, "calibration": {}}

    # Targets
    ys = {
        "late_delivery_risk": df["Late_delivery_risk"].astype(int).values,
        "shipping_mode":      df["Shipping Mode"].astype("category").cat.codes.values,
        "delivery_status":    df["Delivery Status"].astype("category").cat.codes.values,
    }

    for task, algo in best.items():
        print(f"\n  [{task}] SHAP({algo})...")
        results["shap_top15"][task] = shap_task(task, algo, df, ys[task], n_samples=1000)
        top = results["shap_top15"][task].get("top15_features", [])
        for t in top[:5]:
            print(f"    {t['name']:<40} {t['importance']:.4f}")

        print(f"  [{task}] Fairness audit...")
        results["fairness"][task] = fairness_task(task, algo, df, ys[task])
        for gcol, per in results["fairness"][task].items():
            if "__summary__" in per:
                s = per["__summary__"]
                print(f"    by {gcol}: disparity={s['disparity']:.3f} (min={s['min_acc']:.3f} max={s['max_acc']:.3f})")

        print(f"  [{task}] Calibration...")
        results["calibration"][task] = calibration_task(task, algo, df, ys[task])
        c = results["calibration"][task]
        print(f"    ECE={c.get('ece',0):.4f}  Brier={c.get('brier',0):.4f}  T*={c.get('temperature_scaling_T',1):.3f}")

    plot_ok = plot_reliability(results["calibration"], PLOTS / "reliability.png")
    results["reliability_plot_saved"] = bool(plot_ok)
    results["elapsed_min"] = (time.time() - t0) / 60

    out = RESULTS / "R2_SHAP_FAIRNESS_CALIBRATION.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}  ({results['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
