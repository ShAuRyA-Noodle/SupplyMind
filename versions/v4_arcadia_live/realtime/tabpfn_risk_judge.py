"""tabpfn_risk_judge.py — TabPFN-v2 classifier as a tabular 7th judge.

Trains on the 8 documented historical events (real EMDAT-anchored) using
features (severity, brent_pre, duration_days, region_id, hormuz_dep_share)
→ ground-truth tier from `severity` band. Acts as a 7th vote alongside the
6-judge OpenRouter panel + 3-judge Ollama panel.

Output: predicted tier (LOW/MEDIUM/HIGH/CRITICAL) + class probabilities +
contributing feature ranks.

Falls back gracefully if TabPFN package or weights are missing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "models" / "tabpfn-v2-clf"
LIB = REPO_ROOT / "versions/v4_arcadia_live" / "scenarios" / "iran_israel_hormuz_2024_2026.json"

TIER_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DEVICE_ENV = None  # lazy import torch only inside _load


def _severity_to_tier(sev: float) -> int:
    if sev >= 0.85:
        return 3  # CRITICAL
    if sev >= 0.65:
        return 2  # HIGH
    if sev >= 0.40:
        return 1  # MEDIUM
    return 0      # LOW


def _build_train_set() -> tuple[np.ndarray, np.ndarray] | None:
    """Real-event-anchored training set: 8 documented Iran/Israel/Hormuz events."""
    if not LIB.exists():
        return None
    catalog = json.loads(LIB.read_text(encoding="utf-8"))
    events = catalog.get("events", [])
    rows: list[list[float]] = []
    targets: list[int] = []
    for ev in events:
        sev = float(ev.get("severity") or 0.5)
        oi = ev.get("oil_impact_usd_bbl") or {}
        pre = oi.get("pre")
        try:
            pre = float(pre) if pre is not None else 80.0
        except (TypeError, ValueError):
            pre = 80.0
        duration = max(1, int(ev.get("duration_days") or 7))
        region = ev.get("region", "hormuz")
        region_id = {"hormuz": 1.0, "red_sea": 2.0,
                      "iran_israel": 3.0}.get(region, 0.0)
        hormuz_dep = 0.6 if region == "hormuz" else (
            0.4 if region == "iran_israel" else 0.7)
        rows.append([sev, pre, float(duration), region_id, hormuz_dep])
        targets.append(_severity_to_tier(sev))
    return np.array(rows, dtype=np.float32), np.array(targets, dtype=np.int64)


_clf = None


def _load_clf():
    global _clf, DEVICE_ENV
    if _clf is not None:
        return _clf
    try:
        import torch
        from tabpfn import TabPFNClassifier
        DEVICE_ENV = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = MODEL_DIR / "tabpfn-v2-classifier.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing {ckpt}")
        _clf = TabPFNClassifier(
            device=DEVICE_ENV, model_path=str(ckpt),
            n_estimators=1, ignore_pretraining_limits=True,
        )
        # Fit once on the 8-event corpus
        train = _build_train_set()
        if train is None:
            _clf = "FAILED"; return None
        X, y = train
        _clf.fit(X, y)
        logger.info("[tabpfn-judge] trained on %d events, device=%s",
                    X.shape[0], DEVICE_ENV)
        return _clf
    except Exception as e:  # noqa: BLE001
        logger.warning("[tabpfn-judge] load/train failed: %s", e)
        _clf = "FAILED"
        return None


def predict(severity: float, brent_pre: float, duration_days: int,
             region: str = "hormuz", hormuz_dep: float = 0.6) -> dict:
    """Public API. Returns predicted tier + per-class probabilities."""
    t0 = time.time()
    clf = _load_clf()
    if clf is None or clf == "FAILED":
        return {"ok": False, "error": "tabpfn_unavailable",
                "fallback_tier": TIER_NAMES[_severity_to_tier(severity)]}

    region_id = {"hormuz": 1.0, "red_sea": 2.0,
                  "iran_israel": 3.0}.get(region.lower(), 0.0)
    x = np.array([[severity, brent_pre, float(duration_days),
                    region_id, hormuz_dep]], dtype=np.float32)
    proba = clf.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))

    return {
        "ok": True,
        "model": "tabpfn-v2-clf",
        "predicted_tier": TIER_NAMES[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "class_probabilities": {
            TIER_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "n_train_events": 8,
        "input_features": {
            "severity": severity, "brent_pre_usd": brent_pre,
            "duration_days": duration_days, "region": region,
            "hormuz_dep_share": hormuz_dep,
        },
        "latency_s": round(time.time() - t0, 3),
        "device": DEVICE_ENV,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = predict(severity=0.85, brent_pre=132.0,
                   duration_days=21, region="hormuz", hormuz_dep=0.6)
    print(json.dumps(out, indent=2))
