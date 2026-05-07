"""R6 Gethsemane v3 — Export MaskablePPO policies to ONNX for production deployment.

The three trained PPO checkpoints (easy/medium/hard) need to be export-ready for
inference in production (e.g. via FastAPI /rl/act or mobile deployment). ONNX
provides a language-agnostic, runtime-optimized format.

Exports the *actor* subnetwork (observation -> action logits). Action masking is
applied at inference time outside the ONNX graph (simple post-processing).

Outputs:
  versions/v3_arcadia/checkpoints/gethsemane/ppo_<task>.onnx
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "v3_arcadia" / "checkpoints" / "gethsemane"

OBS_DIM = 408
N_ACTIONS = 280  # 7 action types × 40 target nodes, flattened


class PPOActor(nn.Module):
    """Pure-PyTorch actor wrapper around SB3 MaskablePPO's policy net.

    SB3's MlpPolicy stores the shared net + action_net. We re-pack to a
    forward function: obs -> logits.
    """
    def __init__(self, mlp_extractor_policy_net: nn.Module, action_net: nn.Module):
        super().__init__()
        self.mlp_extractor = mlp_extractor_policy_net
        self.action_net = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.mlp_extractor(obs)
        logits = self.action_net(features)
        return logits


def export_task(task: str) -> dict:
    from sb3_contrib import MaskablePPO
    ckpt_path = CKPT / f"ppo_{task}.zip"
    if not ckpt_path.exists():
        return {"task": task, "error": "checkpoint not found"}

    log.info(f"Loading {ckpt_path.name}...")
    model = MaskablePPO.load(str(ckpt_path), device="cpu")
    policy = model.policy
    # MaskablePPO MlpPolicy: features_extractor is Flatten; mlp_extractor has policy_net + value_net
    # We want: obs -> features_extractor -> mlp_extractor.policy_net -> action_net
    features_extractor = policy.features_extractor
    mlp_policy = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    class FullActor(nn.Module):
        def __init__(self, fe, mlp, an):
            super().__init__()
            self.fe = fe
            self.mlp = mlp
            self.an = an

        def forward(self, obs):
            x = self.fe(obs)
            x = self.mlp(x)
            return self.an(x)

    actor = FullActor(features_extractor, mlp_policy, action_net).eval()

    # Sanity check on a random obs
    dummy = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        logits = actor(dummy)
    assert logits.shape == (1, N_ACTIONS), f"Expected (1,{N_ACTIONS}) got {logits.shape}"
    log.info(f"  actor forward OK: logits shape {tuple(logits.shape)}")

    # Export to ONNX
    onnx_path = CKPT / f"ppo_{task}.onnx"
    torch.onnx.export(
        actor, dummy, str(onnx_path),
        input_names=["observation"], output_names=["action_logits"],
        dynamic_axes={"observation": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
    )
    log.info(f"  exported {onnx_path.name}  ({onnx_path.stat().st_size/1e6:.2f} MB)")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        test = np.random.randn(1, OBS_DIM).astype(np.float32)
        out = sess.run(None, {"observation": test})
        log.info(f"  onnxruntime verified: output shape {out[0].shape}")
        # Compare against torch
        with torch.no_grad():
            torch_out = actor(torch.tensor(test)).numpy()
        diff = float(np.abs(out[0] - torch_out).max())
        log.info(f"  max torch vs onnx diff: {diff:.2e}")
        return {"task": task, "onnx_path": str(onnx_path),
                "size_mb": float(onnx_path.stat().st_size / 1e6),
                "verified": True, "max_diff": diff}
    except ImportError:
        log.warning("  onnxruntime not installed; skipping verification")
        return {"task": task, "onnx_path": str(onnx_path),
                "size_mb": float(onnx_path.stat().st_size / 1e6),
                "verified": False}


def main():
    from v3_arcadia.results import export_summary
    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    results = [export_task(t) for t in tasks]
    out_path = ROOT / "v3_arcadia" / "results" / "R6_GETHSEMANE_ONNX_EXPORT.json"
    out_path.write_text(json.dumps({"exports": results}, indent=2, default=str))
    log.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    results = [export_task(t) for t in tasks]
    out_path = ROOT / "v3_arcadia" / "results" / "R6_GETHSEMANE_ONNX_EXPORT.json"
    out_path.write_text(json.dumps({"exports": results}, indent=2, default=str))
    log.info(f"\nSaved {out_path}")
