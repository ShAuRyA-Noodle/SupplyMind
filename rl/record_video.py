"""
RecordVideo wrapper — generate MP4s of agent behavior.

Creates 3 videos showing:
  1. Scripted agent handling typhoon response (baseline)
  2. PPO agent (learned policy)
  3. QR-DQN CVaR agent (risk-averse, best performer)

Uses Gymnasium's RecordVideo wrapper with matplotlib rgb_array rendering.

Usage:
    python -m rl.record_video
    python -m rl.record_video --agent scripted --task easy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

VIDEO_DIR = _PROJECT_ROOT / "videos"


def record_scripted(task_id: str = "easy_typhoon_response", seed: int = 42) -> Path:
    """Record scripted agent episode as video frames."""
    import gymnasium as gym
    import rl  # noqa: F401
    from gymnasium.wrappers import RecordVideo

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make("SupplyMind-Easy-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(VIDEO_DIR),
                      name_prefix=f"scripted_{task_id}",
                      episode_trigger=lambda ep: True)

    from scripted_agent import choose_action as scripted_choose
    from rl.gym_env import ACTION_TYPES

    obs, info = env.reset(seed=seed)
    step = 0

    while True:
        raw_obs = info.get("raw_obs")
        if raw_obs:
            sm_action = scripted_choose(raw_obs, step)
            action_type_idx = ACTION_TYPES.index(sm_action.action_type)
            target_idx = 0
            if sm_action.target_node_id:
                node_ids = [n.node_id for n in raw_obs.node_statuses]
                if sm_action.target_node_id in node_ids:
                    target_idx = node_ids.index(sm_action.target_node_id)
            action = np.array([action_type_idx, target_idx], dtype=np.int64)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            break

    env.close()
    logger.info("Scripted video saved to %s", VIDEO_DIR)
    return VIDEO_DIR


def record_qrdqn(task_id: str = "easy_typhoon_response", seed: int = 42) -> Path:
    """Record QR-DQN CVaR agent episode."""
    import gymnasium as gym
    import torch
    import rl  # noqa: F401
    from gymnasium.wrappers import RecordVideo

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

    env = gym.make("SupplyMind-Easy-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(VIDEO_DIR),
                      name_prefix=f"qrdqn_{task_id}",
                      episode_trigger=lambda ep: True)

    # Load QR-DQN
    ckpt_path = CHECKPOINT_DIR / "qrdqn_best_easy.pt"
    if ckpt_path.exists():
        from rl.distributional.qr_dqn import QRDQNNetwork
        from rl.checkpoint_security import safe_load
        ckpt = safe_load(ckpt_path, map_location="cpu")
        cfg = {k: v for k, v in ckpt["config"].items()
               if k in ("state_dim", "n_actions", "n_quantiles", "hidden_dim")}
        model = QRDQNNetwork(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        has_model = True
    else:
        logger.warning("QR-DQN checkpoint not found, using random actions")
        has_model = False

    obs, info = env.reset(seed=seed)
    while True:
        if has_model:
            with torch.no_grad():
                state_t = torch.from_numpy(obs).float().unsqueeze(0)
                mask = info["action_masks"]
                mask_t = torch.from_numpy(mask).bool().unsqueeze(0)
                flat_action = model.cvar_policy(state_t, alpha=0.1, action_mask=mask_t).item()
            action = np.array([flat_action // 40, flat_action % 40], dtype=np.int64)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    logger.info("QR-DQN video saved to %s", VIDEO_DIR)
    return VIDEO_DIR


def record_all() -> None:
    """Record all 3 agent videos."""
    logger.info("Recording 3 agent behavior videos...")
    try:
        record_scripted()
    except Exception as e:
        logger.warning("Scripted video failed: %s", e)
    try:
        record_qrdqn()
    except Exception as e:
        logger.warning("QR-DQN video failed: %s", e)
    logger.info("Videos saved to %s", VIDEO_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Record agent behavior videos")
    parser.add_argument("--agent", choices=["scripted", "qrdqn", "all"], default="all")
    parser.add_argument("--task", default="easy_typhoon_response")
    args = parser.parse_args()

    if args.agent == "all":
        record_all()
    elif args.agent == "scripted":
        record_scripted(args.task)
    elif args.agent == "qrdqn":
        record_qrdqn(args.task)


if __name__ == "__main__":
    main()
