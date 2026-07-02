"""
cuda_kernel_verify.py — G14. Verify + benchmark the custom CUDA action-mask kernel.

The v2 era added `rl/cuda/action_mask_kernel.cu` + compiled `.obj` with a
PyTorch-fallback wrapper at `rl/cuda/action_mask_kernel.py`. The kernel was
never loaded as a `.dll` — the fallback path was always used.

This module:
    1. Tries to JIT-compile the .cu via torch.utils.cpp_extension (CUDA + MSVC
       Build Tools required).
    2. If compile succeeds: benchmarks the custom kernel vs PyTorch fallback
       vs a naive Python loop. Numerical-equivalence check between all three.
    3. If compile fails: benchmarks the PyTorch fallback only and documents why
       the JIT compile failed (usually MSVC missing on Windows).

Result JSON saved to F14_CUDA_KERNEL.json — documents whether the kernel is
compilable in the current environment AND reports the speed comparison.

Honest finding (from preliminary runs): PyTorch's scatter_add and masked_fill
are already hand-optimized and run in <1ms for our batch sizes (B<=1000 x
n_actions=280). The custom kernel was worth writing pedagogically but the
fallback is fast enough for production. We do NOT claim the CUDA kernel is
the secret sauce.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CUDA_SRC = PROJECT_ROOT / "rl" / "cuda" / "action_mask_kernel.cu"
RESULTS_PATH = Path(__file__).resolve().parent / "F14_CUDA_KERNEL.json"


def _torch_fallback_mask(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The fallback used in rl/cuda/action_mask_kernel.py."""
    result = q.clone()
    result[~mask] = float("-inf")
    return result


def _naive_python_mask(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pure Python loop (reference for correctness + slowest baseline)."""
    out = q.clone()
    q_np = q.cpu().numpy()
    m_np = mask.cpu().numpy()
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if not m_np[i, j]:
                out[i, j] = float("-inf")
    return out


def _try_jit_compile() -> tuple[bool, str, object]:
    """Attempt to JIT compile the CUDA kernel. Returns (ok, message, module)."""
    if not CUDA_SRC.exists():
        return False, f"CUDA source missing at {CUDA_SRC}", None
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() == False", None
    # Check for MSVC on Windows
    if platform.system() == "Windows":
        if not shutil.which("cl.exe") and not shutil.which("cl"):
            return False, "MSVC (cl.exe) not on PATH; install Visual Studio Build Tools", None

    try:
        from torch.utils.cpp_extension import load
        # Minimal inline wrapper as a .cpp file referencing the .cu kernel
        wrapper_cpp = CUDA_SRC.parent / "_action_mask_pytorch_wrapper.cpp"
        wrapper_cpp.write_text(
            '#include <torch/extension.h>\n'
            'torch::Tensor apply_mask_cuda(torch::Tensor q, torch::Tensor mask);\n'
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
            '  m.def("apply_mask", &apply_mask_cuda, "action mask apply (CUDA)");\n'
            '}\n'
        )
        module = load(
            name="action_mask_jit",
            sources=[str(wrapper_cpp), str(CUDA_SRC)],
            verbose=False,
        )
        return True, "compiled via torch.utils.cpp_extension.load", module
    except Exception as e:  # noqa: BLE001
        return False, f"JIT compile failed: {str(e)[:300]}", None


def _bench(fn, q, mask, warmup=5, iters=50, device="cuda") -> float:
    for _ in range(warmup):
        _ = fn(q, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms per call


def run_benchmark(
    batch_sizes: tuple[int, ...] = (32, 256, 1024, 8192),
    n_actions: int = 280,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try JIT compile
    jit_ok, jit_msg, jit_module = _try_jit_compile()
    logger.info("[cuda] JIT compile: %s — %s", jit_ok, jit_msg)

    results = []
    for bs in batch_sizes:
        q = torch.randn(bs, n_actions, device=device)
        mask = torch.rand(bs, n_actions, device=device) > 0.3
        # Ensure at least one valid action per row
        for i in range(bs):
            if not mask[i].any():
                mask[i, 0] = True

        # Reference: PyTorch fallback
        ref = _torch_fallback_mask(q, mask)

        ms_fallback = _bench(_torch_fallback_mask, q, mask, device=device)

        jit_ms = None
        jit_equal = None
        if jit_ok and jit_module is not None:
            try:
                jit_out = jit_module.apply_mask(q, mask)
                jit_equal = bool(torch.equal(ref, jit_out) or
                                 torch.allclose(ref, jit_out, atol=1e-5, equal_nan=True))
                jit_ms = _bench(jit_module.apply_mask, q, mask, device=device)
            except Exception as e:  # noqa: BLE001
                logger.warning("[cuda] JIT apply failed at bs=%d: %s", bs, e)

        # Naive python only for small batches (O(b*n))
        naive_ms = None
        naive_equal = None
        if bs <= 1024:
            naive_out = _naive_python_mask(q, mask)
            naive_equal = bool(torch.equal(ref, naive_out) or
                               torch.allclose(ref, naive_out, atol=1e-5, equal_nan=True))
            naive_ms = _bench(_naive_python_mask, q, mask, warmup=1, iters=3, device=device)

        results.append({
            "batch_size": bs,
            "n_actions": n_actions,
            "pytorch_fallback_ms": round(ms_fallback, 4),
            "jit_cuda_ms": round(jit_ms, 4) if jit_ms is not None else None,
            "jit_matches_pytorch": jit_equal,
            "naive_python_ms": round(naive_ms, 4) if naive_ms is not None else None,
            "naive_matches_pytorch": naive_equal,
            "speedup_jit_over_fallback": (round(ms_fallback / jit_ms, 2)
                                          if jit_ms else None),
            "speedup_fallback_over_naive": (round(naive_ms / ms_fallback, 2)
                                             if naive_ms else None),
        })

    out = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "jit_compile": {"ok": jit_ok, "message": jit_msg},
        "benchmarks": results,
        "conclusion": _conclude(jit_ok, results),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    return out


def _conclude(jit_ok: bool, results: list[dict]) -> str:
    if not results:
        return "no benchmark rows"
    ms_at_1024 = next((r["pytorch_fallback_ms"] for r in results if r["batch_size"] == 1024), None)
    if ms_at_1024 is None:
        return "partial results"
    if jit_ok:
        speedups = [r["speedup_jit_over_fallback"] for r in results if r["speedup_jit_over_fallback"]]
        if speedups:
            mean_speedup = sum(speedups) / len(speedups)
            return (f"CUDA JIT compiled. Mean speedup over PyTorch fallback: "
                    f"{mean_speedup:.2f}x. Fallback is already fast ({ms_at_1024:.3f}ms "
                    f"at batch=1024), so kernel is optional for our scale.")
    return (f"CUDA JIT compile failed; using PyTorch fallback only "
            f"({ms_at_1024:.3f}ms at batch=1024 — fast enough for production).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 256, 1024, 8192])
    args = parser.parse_args()

    out = run_benchmark(batch_sizes=tuple(args.batches))
    print(json.dumps(out, indent=2))
