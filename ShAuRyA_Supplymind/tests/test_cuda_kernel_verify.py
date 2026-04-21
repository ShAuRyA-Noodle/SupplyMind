"""test_cuda_kernel_verify.py — G14 regression (PyTorch fallback path)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features.cuda_kernel_verify import (
    _bench, _naive_python_mask, _torch_fallback_mask, run_benchmark,
)


def test_fallback_produces_minus_inf_on_masked():
    q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    out = _torch_fallback_mask(q, mask)
    assert out[0, 1].item() == float("-inf")
    assert out[1, 0].item() == float("-inf")
    # Valid entries preserved
    assert out[0, 0].item() == 1.0 and out[1, 1].item() == 5.0


def test_naive_matches_fallback():
    torch.manual_seed(1)
    q = torch.randn(4, 8)
    mask = torch.rand(4, 8) > 0.3
    # Ensure >=1 valid per row
    for i in range(4):
        if not mask[i].any():
            mask[i, 0] = True
    a = _torch_fallback_mask(q, mask)
    b = _naive_python_mask(q, mask)
    assert torch.equal(a, b) or torch.allclose(a, b, atol=1e-6, equal_nan=True)


def test_benchmark_returns_structured_result():
    out = run_benchmark(batch_sizes=(32,))
    assert "benchmarks" in out and len(out["benchmarks"]) == 1
    bench = out["benchmarks"][0]
    assert bench["pytorch_fallback_ms"] > 0
    assert "conclusion" in out
    assert bench.get("naive_matches_pytorch") is True
