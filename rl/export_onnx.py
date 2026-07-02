"""
ONNX export for SupplyMind RL policy.

Exports the QR-DQN policy to ONNX format for production deployment.
opset_version=17 for broad compatibility.

Usage:
    python -m rl.export_onnx
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def export_to_onnx(
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Export QR-DQN policy to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint (default: qrdqn_best_easy.pt).
        output_path:     Path for ONNX output (default: checkpoints/supplymind_policy.onnx).

    Returns:
        Path to saved ONNX file.
    """
    from rl.distributional.qr_dqn import QRDQNNetwork

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "qrdqn_best_easy.pt"
    if output_path is None:
        output_path = CHECKPOINT_DIR / "supplymind_policy.onnx"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        logger.warning("Checkpoint not found at %s. Creating from untrained model.", checkpoint_path)
        model = QRDQNNetwork(408, 280, 51)
    else:
        from rl.checkpoint_security import safe_load
        ckpt = safe_load(checkpoint_path, map_location="cpu")
        cfg = {k: v for k, v in ckpt["config"].items() if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
        model = QRDQNNetwork(**cfg)
        model.load_state_dict(ckpt["state_dict"])

    model.eval()

    dummy_input = torch.randn(1, 408)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["state"],
        output_names=["quantile_values"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "quantile_values": {0: "batch_size"},
        },
    )

    logger.info("ONNX model exported to %s (%.1f MB)",
                output_path, output_path.stat().st_size / 1e6)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    export_to_onnx()


if __name__ == "__main__":
    main()
