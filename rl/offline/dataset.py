"""
Offline dataset generation for SupplyMind RL.

Generates (state, action, reward, next_state, done, returns_to_go) tuples by
running episodes with a scripted agent (good actions) and a random agent
(exploration). Injects real commodity prices from FRED API.

Usage:
    python -m rl.offline.dataset                        # default 10K episodes
    python -m rl.offline.dataset --episodes 5000        # custom count
    python -m rl.offline.dataset --scripted-only         # only scripted agent
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import SupplyMindAction
from rl.gym_env import (
    ACTION_TYPES,
    MAX_NODES,
    NUM_ACTION_TYPES,
    SupplyMindGymnasiumEnv,
)
from scripted_agent import choose_action as scripted_choose_action

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED API — Real commodity price injection
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "DCOILWTICO": "Crude Oil (WTI)",
    "PCOPPUSDM": "Copper",
    "DEXTAUS": "TWD/USD Exchange Rate",
    "DEXKOUS": "KRW/USD Exchange Rate",
    "DEXJPUS": "JPY/USD Exchange Rate",
    "DEXUSEU": "EUR/USD Exchange Rate",
    "DEXCHUS": "CNY/USD Exchange Rate",
}

FRED_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "fred_cache.json"


def _load_env_api_key() -> str | None:
    """Load FRED_API_KEY from .env file."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("FRED_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def fetch_fred_data(force_refresh: bool = False) -> dict[str, Any]:
    """Fetch FRED series and cache to JSON. Returns cached data if available."""
    if FRED_CACHE_PATH.exists() and not force_refresh:
        logger.info("Loading cached FRED data from %s", FRED_CACHE_PATH)
        return json.loads(FRED_CACHE_PATH.read_text())

    api_key = _load_env_api_key()
    if not api_key:
        logger.warning("FRED_API_KEY not found in .env — using fallback data")
        return _fallback_fred_data()

    try:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        result: dict[str, Any] = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

        for series_id, label in FRED_SERIES.items():
            logger.info("Fetching FRED series: %s (%s)", series_id, label)
            try:
                data = fred.get_series(series_id, observation_start="2015-01-01")
                # Convert to list of (date_str, value) pairs, drop NaN
                records = []
                for date, val in data.items():
                    if not np.isnan(val):
                        records.append({
                            "date": str(date.date()),
                            "value": round(float(val), 4),
                        })
                result[series_id] = {
                    "label": label,
                    "count": len(records),
                    "data": records,
                }
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", series_id, e)
                result[series_id] = {"label": label, "count": 0, "data": [], "error": str(e)}

        # Cache
        FRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FRED_CACHE_PATH.write_text(json.dumps(result, indent=2))
        logger.info("FRED data cached to %s", FRED_CACHE_PATH)
        return result

    except ImportError:
        logger.warning("fredapi not installed — using fallback data")
        return _fallback_fred_data()


def _fallback_fred_data() -> dict[str, Any]:
    """Hardcoded commodity price fallback when the live FRED API is unavailable.

    These are a handful of hand-picked historical points, NOT live data. The
    returned dict carries "source": "hardcoded_fallback" so nothing downstream
    can present it as a live fetch, and a warning is always logged.
    """
    logger.warning(
        "Serving HARDCODED FRED fallback data (source=hardcoded_fallback) — "
        "not a live FRED fetch. Set FRED_API_KEY and install fredapi for real data."
    )
    # Real approximate values from public FRED data
    return {
        "fetched_at": "fallback",
        "source": "hardcoded_fallback",
        "DCOILWTICO": {
            "label": "Crude Oil (WTI)",
            "count": 5,
            "data": [
                {"date": "2024-01-02", "value": 70.38},
                {"date": "2024-04-01", "value": 83.17},
                {"date": "2024-07-01", "value": 81.54},
                {"date": "2024-10-01", "value": 68.17},
                {"date": "2025-01-02", "value": 73.96},
            ],
        },
        "PCOPPUSDM": {
            "label": "Copper",
            "count": 5,
            "data": [
                {"date": "2024-01-01", "value": 8536.0},
                {"date": "2024-04-01", "value": 9437.0},
                {"date": "2024-07-01", "value": 9243.0},
                {"date": "2024-10-01", "value": 9674.0},
                {"date": "2025-01-01", "value": 8891.0},
            ],
        },
        "DEXTAUS": {"label": "TWD/USD Exchange Rate", "count": 0, "data": []},
        "DEXKOUS": {"label": "KRW/USD Exchange Rate", "count": 0, "data": []},
        "DEXJPUS": {"label": "JPY/USD Exchange Rate", "count": 0, "data": []},
        "DEXUSEU": {"label": "EUR/USD Exchange Rate", "count": 0, "data": []},
        "DEXCHUS": {"label": "CNY/USD Exchange Rate", "count": 0, "data": []},
    }


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------


def random_choose_action(obs, env: SupplyMindGymnasiumEnv, rng: np.random.Generator) -> np.ndarray:
    """Choose a random valid action using action masks."""
    mask = env._compute_action_mask()
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return np.array([0, 0], dtype=np.int64)  # do_nothing
    flat_idx = rng.choice(valid_indices)
    action_type = flat_idx // MAX_NODES
    node_idx = flat_idx % MAX_NODES
    return np.array([action_type, node_idx], dtype=np.int64)


def scripted_to_multidiscrete(
    sm_action: SupplyMindAction,
    node_id_to_idx: dict[str, int],
) -> np.ndarray:
    """Convert SupplyMindAction to MultiDiscrete([7, 40]) array."""
    action_type_idx = ACTION_TYPES.index(sm_action.action_type)
    target_idx = 0
    if sm_action.target_node_id and sm_action.target_node_id in node_id_to_idx:
        target_idx = node_id_to_idx[sm_action.target_node_id]
    return np.array([action_type_idx, target_idx], dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

TASK_IDS = [
    "easy_typhoon_response",
    "medium_multi_front",
    "hard_cascading_crisis",
]


def generate_dataset(
    n_scripted: int = 5000,
    n_random: int = 5000,
    output_path: Path | None = None,
) -> Path:
    """Generate offline RL dataset.

    Returns path to saved .npz file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "offline_buffer.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch FRED data first
    fred_data = fetch_fred_data()
    logger.info("FRED data ready (%d series)", len([k for k in fred_data if k != "fetched_at"]))

    rng = np.random.default_rng(42)
    total_episodes = n_scripted + n_random

    # Pre-allocate lists
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_next_states: list[np.ndarray] = []
    all_dones: list[bool] = []
    all_returns_to_go: list[float] = []

    episode_rewards: list[list[float]] = []  # Per-episode reward sequences

    start_time = time.time()

    for ep_idx in range(total_episodes):
        is_scripted = ep_idx < n_scripted
        task_id = TASK_IDS[ep_idx % len(TASK_IDS)]

        env = SupplyMindGymnasiumEnv(task_id=task_id)
        seed = int(rng.integers(0, 100_000))
        obs, info = env.reset(seed=seed)

        ep_states = [obs.copy()]
        ep_actions = []
        ep_rewards = []
        step = 0

        while True:
            if is_scripted:
                raw_obs = info["raw_obs"]
                sm_action = scripted_choose_action(raw_obs, step)
                action = scripted_to_multidiscrete(sm_action, env._node_id_to_idx)
            else:
                action = random_choose_action(info["raw_obs"], env, rng)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_actions.append(action.copy())
            ep_rewards.append(reward)
            ep_states.append(next_obs.copy())
            step += 1

            if done:
                break

        # Compute returns-to-go (reverse cumulative sum)
        returns = []
        rtg = 0.0
        for r in reversed(ep_rewards):
            rtg += r
            returns.append(rtg)
        returns.reverse()

        # Store transitions
        for t in range(len(ep_actions)):
            all_states.append(ep_states[t])
            all_actions.append(ep_actions[t])
            all_rewards.append(ep_rewards[t])
            all_next_states.append(ep_states[t + 1])
            all_dones.append(t == len(ep_actions) - 1)
            all_returns_to_go.append(returns[t])

        episode_rewards.append(ep_rewards)

        # Progress logging
        if (ep_idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep_idx + 1) / elapsed
            eta = (total_episodes - ep_idx - 1) / max(eps_per_sec, 0.01)
            agent_type = "scripted" if is_scripted else "random"
            avg_reward = np.mean([sum(r) for r in episode_rewards[-500:]])
            logger.info(
                "[%d/%d] %s | transitions=%d | avg_ep_reward=%.3f | %.1f eps/s | ETA %.0fs",
                ep_idx + 1, total_episodes, agent_type,
                len(all_states), avg_reward, eps_per_sec, eta,
            )

    # Convert to arrays and save
    dataset = {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int64),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "next_states": np.array(all_next_states, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.bool_),
        "returns_to_go": np.array(all_returns_to_go, dtype=np.float32),
    }

    np.savez_compressed(output_path, **dataset)
    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Dataset saved to %s", output_path)
    logger.info("Total transitions: %d", len(all_states))
    logger.info("Total episodes: %d (scripted=%d, random=%d)", total_episodes, n_scripted, n_random)
    logger.info("File size: %.1f MB", output_path.stat().st_size / 1e6)
    logger.info("Time: %.1f minutes", total_time / 60)
    logger.info("=" * 60)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate offline RL dataset")
    parser.add_argument("--episodes", type=int, default=10000, help="Total episodes (split 50/50)")
    parser.add_argument("--scripted-only", action="store_true", help="Only scripted agent")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    args = parser.parse_args()

    if args.scripted_only:
        n_scripted = args.episodes
        n_random = 0
    else:
        n_scripted = args.episodes // 2
        n_random = args.episodes - n_scripted

    output = Path(args.output) if args.output else None
    generate_dataset(n_scripted=n_scripted, n_random=n_random, output_path=output)


if __name__ == "__main__":
    main()
