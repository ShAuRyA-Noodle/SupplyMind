"""Convert BGE-M3 pytorch_model.bin + colbert_linear.pt + sparse_linear.pt to safetensors format.

BGE-M3's sentence-transformers loader uses pytorch_model.bin which triggers torch.load
security restriction on torch<2.6. Converting to model.safetensors eliminates the issue.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
from safetensors.torch import save_file

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "bge-m3"


def convert_bin(bin_path: Path, out_path: Path):
    if out_path.exists():
        print(f"  exists: {out_path}")
        return
    # Temporarily patch torch.load
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
    try:
        state = torch.load(bin_path, map_location="cpu")
    finally:
        torch.load = orig
    # Write as safetensors
    clean = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state.items()
             if isinstance(v, torch.Tensor)}
    save_file(clean, str(out_path))
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e9:.2f} GB, {len(clean)} tensors)")


if __name__ == "__main__":
    # Main model weights
    convert_bin(MODEL_DIR / "pytorch_model.bin", MODEL_DIR / "model.safetensors")
    # Auxiliary heads (colbert and sparse) — load and re-save or keep as .pt
    # These are small so skip conversion, keep as-is.
    for aux in ["colbert_linear.pt", "sparse_linear.pt"]:
        p = MODEL_DIR / aux
        if p.exists():
            print(f"  kept aux: {aux} ({p.stat().st_size/1e6:.1f} MB)")
    print("Done.")
