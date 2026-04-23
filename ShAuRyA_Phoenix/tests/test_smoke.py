"""test_smoke.py — ultra-light smoke tests for every Phoenix module.

All tests here must run in under 10 seconds combined, with no GPU, no
Ollama, and no live APIs. They verify imports, core function invocations
on mock inputs, and file-system state.

Heavier tests (GPU-backed training, live-API integration, full arena
evaluation) live in test_*_integration.py and are skipped by default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PHOENIX = ROOT / "ShAuRyA_Phoenix"


def test_phoenix_skeleton_exists():
    for sub in ["roll_integration", "supplymind_skills", "arena",
                "counterfactual_twin", "autoresearch_fixed", "receipts_v2",
                "experiments", "docs", "tests", "server", "upstream_prs",
                "scripts", "realtime_v5"]:
        assert (PHOENIX / sub).is_dir(), f"missing {sub}/"


def test_receipts_indexed():
    idx = PHOENIX / "receipts_v2" / "INDEX.md"
    idx_json = PHOENIX / "receipts_v2" / "INDEX.json"
    assert idx.exists(), "INDEX.md missing; run receipts_v2.register --stub"
    assert idx_json.exists(), "INDEX.json missing"
    rows = json.loads(idx_json.read_text())
    assert len(rows) >= 20, f"expected 20+ receipts, got {len(rows)}"


def test_autoresearch_state_coherent():
    state = PHOENIX / "autoresearch_fixed" / "state.json"
    assert state.exists(), "run rebuild_state.py"
    s = json.loads(state.read_text())
    assert s["best"] is not None, "no best experiment after rebuild"
    # Post-Phoenix-rerun: s3_curriculum_learning beats s2 (+0.097 CI95 lower delta).
    # Pre-rerun state (v4): s2_higher_entropy was best. Accept either.
    assert s["best"]["experiment_name"] in ("s2_higher_entropy", "s3_curriculum_learning"), \
        f"unexpected best: {s['best']['experiment_name']}"
    # The current best (whichever it is) must have CI95 lower >= 0.45.
    best = [h for h in s["history"] if h["experiment_name"] == s["best"]["experiment_name"]]
    assert best, "best experiment missing from history"
    assert best[0]["metric_ci95_lower"] >= 0.45


def test_replay_cache_built():
    cache = PHOENIX / "realtime_v5" / "replay_cache_latest.json"
    assert cache.exists(), "run freeze_cache.py"
    c = json.loads(cache.read_text())
    assert "events" in c
    assert len(c["events"]) >= 8, "crisis library should yield 8 events"


def test_skill_pack_complete():
    skills = PHOENIX / "supplymind_skills"
    assert (skills / "plugin.json").exists()
    plugin = json.loads((skills / "plugin.json").read_text())
    assert len(plugin.get("skills", [])) == 3
    for s in plugin["skills"]:
        sk_path = skills / s["path"]
        assert sk_path.exists(), f"missing SKILL.md for {s['name']}"
        content = sk_path.read_text()
        assert content.startswith("---"), f"{s['name']}: missing YAML frontmatter"
        assert "description:" in content


def test_receipt_framework_importable():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.receipts_v2.framework import Receipt
    r = Receipt(
        claim_id="test_dummy",
        claim="python --version returns something parseable",
        command="python --version",
        extraction="python --version",
        expected="Python",
        comparator="regex",
        expected_regex=r"Python \d+\.\d+",
    )
    r.run()
    # `python --version` must succeed; pass-5 tightened from (0, 1) to == 0
    # after audit flagged the permissive assertion.
    assert r.exit_code == 0, f"unexpected exit_code: {r.exit_code}"
    assert r.match, f"expected python --version to match 'Python \\d+\\.\\d+', got {r.extracted!r}"


def test_arena_leaderboard_importable():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.arena import leaderboard
    b = leaderboard.rebuild()
    assert b["n_baselines"] >= 6
    assert any(r["policy_name"].startswith("MaskablePPO") for r in b["rows"])


def test_arena_runner_importable():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.arena import runner  # noqa: F401
    # Just check the module compiles; running takes GPU + trained policy
    assert hasattr(runner, "evaluate_policy")
    assert hasattr(runner, "TaskResult")
    assert hasattr(runner, "ArenaResult")


def test_twin_importable():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.counterfactual_twin import twin  # noqa: F401
    assert hasattr(twin, "run_twin")
    assert hasattr(twin, "TwinReport")


def test_dpo_judge_preference_builder_importable():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.roll_integration.dpo_judge import prepare_preference_data
    assert hasattr(prepare_preference_data, "build_pairs")
    # Don't actually build; that requires real GT files


def test_roll_env_wrapper_importable():
    sys.path.insert(0, str(ROOT))
    # Env wrapper imports from rl.gym_env; skip gracefully if missing
    try:
        from ShAuRyA_Phoenix.roll_integration.env import supplymind_roll_env  # noqa: F401
        assert hasattr(supplymind_roll_env, "SupplyMindRollEnv")
    except Exception as e:
        pytest.skip(f"dependencies not in env: {e}")


def test_reward_bridge_importable_without_roll():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.roll_integration.reward_bridge import supplymind_judge_worker
    # Class exists even when ROLL isn't installed (fallback stub)
    assert hasattr(supplymind_judge_worker, "SupplyMind3JudgeRewardWorker")


def test_replay_adapter_status():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.realtime_v5 import replay_adapter
    s = replay_adapter.status()
    assert s["cache_exists"] is True
    assert s["n_events"] >= 8


def test_phoenix_app_builds():
    sys.path.insert(0, str(ROOT))
    from ShAuRyA_Phoenix.server import phoenix_app
    # App exists; if any router failed to mount it still comes up with the rest
    assert phoenix_app.app is not None
    status_route = [r for r in phoenix_app.app.routes if getattr(r, "path", "") == "/phoenix/status"]
    assert status_route, "phoenix status endpoint missing"


def test_upstream_pr_drafts_present():
    for p in [PHOENIX / "upstream_prs" / "meta_openenv" / "PR.md",
              PHOENIX / "upstream_prs" / "alibaba_roll" / "PR.md"]:
        assert p.exists(), f"missing upstream PR draft: {p}"
        assert p.read_text().strip(), f"{p} is empty"


def test_docs_suite_complete():
    for fn in ["README_V5_OPENENV_FIRST.md", "PREPRINT_V5.md",
               "PITCH_DECK_V5.md", "DEMO_VIDEO_SCRIPT_V5.md", "JUDGES_V5.md"]:
        p = PHOENIX / "docs" / fn
        assert p.exists(), f"missing {fn}"
        assert len(p.read_text()) > 500, f"{fn} is too short"
