"""test_lora_train.py — G7 regression (dry-run path only)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from versions.v4_arcadia_live.features.lora_train import (
    LoRAConfig, TrainingExample, _format_example, build_dataset, dry_run,
)


def test_dataset_builds_at_least_10_examples():
    examples = build_dataset()
    assert len(examples) >= 10
    risks = {ex.correct_risk_level for ex in examples}
    assert {"LOW", "MEDIUM", "HIGH", "CRITICAL"} & risks  # at least one each


def test_lora_config_defaults_reasonable():
    cfg = LoRAConfig()
    assert cfg.rank in (8, 16, 32, 64)
    assert cfg.learning_rate > 0 and cfg.learning_rate < 0.01
    assert cfg.n_epochs >= 1
    assert len(cfg.target_modules) >= 2


def test_format_example_contains_chat_template_tokens():
    ex = TrainingExample(
        scenario="Test scenario",
        correct_risk_level="HIGH",
        rationale="test rationale",
    )
    text = _format_example(ex)
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    assert "<|im_end|>" in text
    assert "HIGH" in text


def test_dry_run_reports_success():
    result = dry_run(LoRAConfig())
    assert result["status"] == "dry_run_ok"
    assert result["n_examples"] >= 10
    assert "sample_text" in result
    assert "config" in result
