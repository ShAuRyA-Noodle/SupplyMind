"""Behavioral contract checks for the optional ROLL/GEM adapter."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.v5_phoenix.roll_integration.env import SupplyMindRollEnv, register_env


def test_roll_adapter_runs_real_episode_with_dense_rewards() -> None:
    env = SupplyMindRollEnv(task_id="easy_typhoon_response", max_steps=2)
    first, info = env.reset(seed=23)
    assert json.loads(first)["day"] == 0
    assert "action_type" in info["env_instruction"]

    second, reward, terminated, truncated, details = env.step(
        '{"action_type":"do_nothing"}'
    )
    assert isinstance(reward, float)
    assert details["metrics"]["action_is_valid"] is True
    assert json.loads(second)["day"] > 0
    assert not terminated and not truncated

    _, penalty, terminated, truncated, details = env.step("invalid json")
    assert penalty <= env.format_penalty
    assert details["metrics"]["action_is_valid"] is False
    assert not terminated and truncated
    assert 0 <= details["episode_score"] <= 1
    with pytest.raises(RuntimeError):
        env.step('{"action_type":"do_nothing"}')
    env.close()


def test_roll_adapter_resets_deterministically() -> None:
    env = SupplyMindRollEnv()
    first, _ = env.reset(seed=17)
    env.step('{"action_type":"do_nothing"}')
    again, _ = env.reset(seed=17)
    assert first == again


def test_gem_registration_round_trip_when_available() -> None:
    gem = pytest.importorskip("gem")
    register_env()
    register_env()
    env = gem.make("supplymind_crisis", max_steps=1)
    observation, _ = env.reset(seed=23)
    assert json.loads(observation)["day"] == 0
    _, reward, _, truncated, _ = env.step('{"action_type":"do_nothing"}')
    assert isinstance(reward, float) and truncated
    env.close()
