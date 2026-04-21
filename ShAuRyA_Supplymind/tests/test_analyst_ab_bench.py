"""test_analyst_ab_bench.py — G9 fix regression test (Ollama-optional)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features import analyst_ab_bench as ab


def test_ten_scenarios_defined():
    assert len(ab.SCENARIOS) == 10
    for s in ab.SCENARIOS:
        assert s.correct_risk in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert len(s.required_evidence) >= 2


def test_modelfile_v5_exists_and_non_empty():
    mf = PROJECT_ROOT / "ShAuRyA_Supplymind" / "features" / "Modelfile.analyst_v5"
    assert mf.exists()
    content = mf.read_text(encoding="utf-8")
    # Must have at least 5 MESSAGE examples
    assert content.count("MESSAGE user") >= 5
    assert "SupplyMind Analyst v5" in content
    assert "CALIBRATION RULES" in content


def test_rubric_scoring_functions():
    # Exact match
    score = ab._score_response(
        {"risk_level": "HIGH", "evidence": ["TSMC backup", "typhoon forecast"],
         "decision": "activate backup", "counterfactual": "no-op"},
        ab.SCENARIOS[2],  # typhoon_72h_warning -> HIGH
    )
    assert score["parsed"] is True
    assert score["exact"] == 1
    assert score["ev_coverage"] > 0.5

    # Off by one
    s2 = ab._score_response(
        {"risk_level": "MEDIUM", "evidence": []},
        ab.SCENARIOS[2],  # correct HIGH -> predicted MEDIUM is off-by-one
    )
    assert s2["exact"] == 0
    assert s2["one_off"] == 1


def test_benchmark_reports_when_ollama_down():
    """Without Ollama the benchmark should report status cleanly, not crash."""
    # Force-fail Ollama detection by pointing at a bad URL
    import importlib
    ab._ollama_up_original = ab._ollama_up
    ab._ollama_up = lambda: False
    try:
        result = ab.benchmark("supplymind-analyst:v5", "qwen2.5:14b")
        assert result["status"] == "ollama_down"
    finally:
        ab._ollama_up = ab._ollama_up_original
