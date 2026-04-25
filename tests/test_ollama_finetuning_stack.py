from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ollama_finetuning_stack_verifier_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_ollama_finetuning_stack.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["n_passed"] == payload["n_checks"]


def test_v5_modelfile_requires_calibrated_strict_json() -> None:
    text = (ROOT / "ShAuRyA_Supplymind/features/Modelfile.analyst_v5").read_text(
        encoding="utf-8"
    )
    assert "Not every news headline is CRITICAL" in text
    assert '"risk_level": "LOW|MEDIUM|HIGH|CRITICAL"' in text
    assert "NO prose outside JSON" in text
    assert text.count("MESSAGE user") >= 8


def test_dpo_preference_pairs_are_real_r4_derived() -> None:
    path = ROOT / "ShAuRyA_Phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 21
    for row in rows:
        assert {"prompt", "chosen", "rejected", "meta"}.issubset(row)
        assert row["meta"]["gt_risk"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert row["meta"]["quality_gap"] >= 2
