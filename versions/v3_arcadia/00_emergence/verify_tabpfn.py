"""Verify TabPFN-v2-clf and TabPFN-v2-reg load from local checkpoints + produce predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"
OUT_PATH = ROOT / "v3_arcadia" / "results" / "tabpfn_verify.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

result: dict = {"cuda_available": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu"}

# -------- Classifier --------
try:
    from tabpfn import TabPFNClassifier
    ckpt = MODELS / "tabpfn-v2-clf" / "tabpfn-v2-classifier.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing classifier ckpt: {ckpt}")
    clf = TabPFNClassifier(device=result["device"], model_path=str(ckpt), n_estimators=1, ignore_pretraining_limits=True)
    rng = np.random.default_rng(42)
    Xc = rng.standard_normal((200, 12)).astype(np.float32)
    yc = rng.integers(0, 2, 200)
    clf.fit(Xc, yc)
    pc = clf.predict_proba(Xc[:10])
    result["tabpfn_clf"] = {
        "status": "OK",
        "ckpt": str(ckpt),
        "n_train_rows": 200,
        "proba_shape": list(pc.shape),
        "sample_pred": pc[0].tolist(),
    }
    print(f"TabPFN-clf OK: shape={pc.shape}")
except Exception as e:
    result["tabpfn_clf"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"TabPFN-clf FAIL: {e}", file=sys.stderr)

# -------- Regressor --------
try:
    from tabpfn import TabPFNRegressor
    ckpt = MODELS / "tabpfn-v2-reg" / "tabpfn-v2-regressor.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing regressor ckpt: {ckpt}")
    reg = TabPFNRegressor(device=result["device"], model_path=str(ckpt), n_estimators=1, ignore_pretraining_limits=True)
    Xr = rng.standard_normal((200, 12)).astype(np.float32)
    yr = Xr.sum(axis=1).astype(np.float32) + rng.standard_normal(200) * 0.1
    reg.fit(Xr, yr)
    pr = reg.predict(Xr[:10])
    result["tabpfn_reg"] = {
        "status": "OK",
        "ckpt": str(ckpt),
        "sample_pred": pr.tolist(),
    }
    print(f"TabPFN-reg OK: preds={pr[:3]}")
except Exception as e:
    result["tabpfn_reg"] = {"status": "FAIL", "error": str(e)[:300]}
    print(f"TabPFN-reg FAIL: {e}", file=sys.stderr)

OUT_PATH.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT_PATH}")
