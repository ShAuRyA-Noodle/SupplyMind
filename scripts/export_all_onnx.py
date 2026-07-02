"""Export every non-LLM SupplyMind model with REAL trained weights to ONNX.

Produces a self-contained inference bundle in
`versions/v3_arcadia/checkpoints/onnx_bundle/` that runs without PyTorch on
pure onnxruntime-cpu.

Only artifacts backed by real trained weights are exported. Models without a
trained checkpoint on disk are recorded in the manifest's `skipped` list with
the reason — a random-initialized network is never shipped under a model name.

Currently exported:
  - PPO policies (easy/medium/hard) — real MaskablePPO exports produced by
    versions/v3_arcadia/50_gethsemane/export_v3_ppo_onnx.py

Skipped (no trained-weight checkpoint present):
  - GCN arrival-time regressor
  - Ridge stacker
  - TFT v1

Output: versions/v3_arcadia/checkpoints/onnx_bundle/{ppo_*.onnx}
        versions/v3_arcadia/results/ONNX_BUNDLE_MANIFEST.json
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V3 = ROOT / "versions" / "v3_arcadia"
OUT = V3 / "checkpoints" / "onnx_bundle"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = V3 / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

manifest = {"exported": [], "skipped": []}


def copy_ppo_onnx():
    log.info("(1/1) PPO ONNX — copying existing trained exports into bundle")
    src = V3 / "checkpoints" / "gethsemane"
    count = 0
    for name in ("ppo_easy_typhoon_response", "ppo_medium_multi_front", "ppo_hard_cascading_crisis"):
        s = src / f"{name}.onnx"
        if s.exists():
            d = OUT / f"{name}.onnx"
            shutil.copy2(s, d)
            count += 1
            manifest["exported"].append({
                "name": name + " (MaskablePPO)",
                "file": d.name,
                "size_kb": int(d.stat().st_size / 1024),
                "input_shape": [1, 408],
                "output_shape": [1, 280],
                "source": "versions/v3_arcadia/50_gethsemane/export_v3_ppo_onnx.py",
                "weights": "trained",
            })
        else:
            manifest["skipped"].append({
                "name": name + " (MaskablePPO)",
                "reason": f"trained ONNX not found at {s.relative_to(ROOT)}",
            })
    log.info(f"  {count}/3 trained PPO ONNX included")


def record_untrained_skips():
    """Do NOT export random-initialized networks as named models. Record the
    non-LLM models that lack a trained checkpoint so the manifest is honest."""
    manifest["skipped"].extend([
        {
            "name": "GCN arrival-time regressor",
            "reason": ("no trained state_dict on disk — refusing to export a "
                       "random-initialized GCN as a named model. Train via "
                       "versions/v3_arcadia/70_provider/r6_gnn_arrival_time.py "
                       "and save its state_dict, then re-export."),
        },
        {
            "name": "Ridge stacker",
            "reason": ("no fitted estimator on disk — refusing to export a "
                       "model fit on random dummy data. Train via "
                       "versions/v3_arcadia/10_caramel/train_caramel.py and "
                       "persist the fitted Ridge, then re-export."),
        },
        {
            "name": "TFT v1",
            "reason": ("pytorch-forecasting TimeSeriesDataSet is required at "
                       "inference; ONNX export needs a wrapper packaging the "
                       "normalizer scaler + encoder/decoder split. Deferred."),
        },
    ])


def main():
    t0 = time.time()
    log.info("ONNX bundle export — pure onnxruntime-cpu surface for trained non-LLM SupplyMind models")

    copy_ppo_onnx()
    record_untrained_skips()

    manifest["elapsed_s"] = time.time() - t0
    manifest["bundle_dir"] = str(OUT.relative_to(ROOT))
    manifest["total_bundle_size_kb"] = sum(
        int((OUT / e["file"]).stat().st_size / 1024) for e in manifest["exported"]
    )
    manifest["policy"] = ("Only real trained weights are exported. Untrained/"
                          "dummy-fit models are listed in skipped, never shipped.")
    out_p = RESULTS / "ONNX_BUNDLE_MANIFEST.json"
    out_p.write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"\nBundle: {len(manifest['exported'])} exported, {len(manifest['skipped'])} skipped")
    log.info(f"Total size: {manifest['total_bundle_size_kb']} KB")
    log.info(f"Saved manifest: {out_p}")


if __name__ == "__main__":
    main()
