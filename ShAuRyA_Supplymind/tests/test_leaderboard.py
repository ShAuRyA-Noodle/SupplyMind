"""test_leaderboard.py — F5 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features import leaderboard as lb


def test_reference_submissions_defined():
    assert lb.SUBMISSION_DO_NOTHING
    assert lb.SUBMISSION_RANDOM_VALID
    assert lb.SUBMISSION_ALERT_THEN_DO_NOTHING


def test_load_submission_valid():
    act = lb._load_submission("def act(obs, m): return 0")
    assert callable(act)
    assert act([0.0], [True]) == 0


def test_load_submission_missing_act_raises():
    import pytest
    with pytest.raises(RuntimeError, match="must define"):
        lb._load_submission("def wrong(x): pass")


def test_bootstrap_ci95_stable_on_known_input():
    ci = lb._bootstrap_ci95_lower([0.5] * 20)
    # All-same input -> bootstrap mean = 0.5 always, CI lower ~= 0.5
    assert 0.45 < ci <= 0.5


def test_render_leaderboard_includes_header():
    md = lb.render_leaderboard_markdown()
    assert "Rank" in md and "CI95 lower" in md


def test_read_leaderboard_returns_list_even_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "LEADERBOARD_PATH", tmp_path / "nope.jsonl")
    assert lb.read_leaderboard() == []
