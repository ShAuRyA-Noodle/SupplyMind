"""Verify TimesFM-2 via Google's timesfm pkg on local weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL = ROOT / "models" / "timesfm-2"
OUT = ROOT / "v3_arcadia" / "results" / "timesfm_verify.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
result: dict = {"device": DEVICE, "local_dir": str(LOCAL)}

try:
    import timesfm

    # TimesFM-2.0 uses 50 layers, 1280 hidden, 1 head, 2048 context
    hp = timesfm.TimesFmHparams(
        backend="gpu" if DEVICE == "cuda" else "cpu",
        per_core_batch_size=32,
        horizon_len=14,
        context_len=2048,
        num_layers=50,
        model_dims=1280,
        num_heads=16,
    )
    ckpt = timesfm.TimesFmCheckpoint(path=str(LOCAL / "torch_model.ckpt"))
    tfm = timesfm.TimesFm(hparams=hp, checkpoint=ckpt)

    # Synthetic sine series
    ts = np.sin(np.linspace(0, 20, 256)).astype(np.float32)
    forecast_input = [ts]
    freq_input = [0]  # 0=high freq daily, 1=medium weekly/monthly, 2=low quarterly/yearly
    point_forecast, quantile_forecast = tfm.forecast(forecast_input, freq=freq_input)
    result["timesfm_2"] = {
        "status": "OK",
        "point_shape": list(np.asarray(point_forecast).shape),
        "quantile_shape": list(np.asarray(quantile_forecast).shape),
        "sample_forecast": point_forecast[0][:5].tolist(),
    }
    print(f"TimesFM-2 OK: point shape={np.asarray(point_forecast).shape}, sample={point_forecast[0][:3].tolist()}")
except Exception as e:
    import traceback
    traceback.print_exc()
    result["timesfm_2"] = {"status": "FAIL", "error": str(e)[:500]}

OUT.write_text(json.dumps(result, indent=2))
print(f"\nSaved {OUT}")
