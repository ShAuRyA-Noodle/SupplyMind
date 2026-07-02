"""router.py — FastAPI router for OpenEnv Arena.

Endpoints:
    POST /arena/run           upload policy.pt, return ArenaResult (sync; sized for 50 ep x 3 tasks ~ 1-3 min)
    GET  /arena/leaderboard   current leaderboard
    GET  /arena/health        liveness

Mounted under /arena by server/phoenix_app.py.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import leaderboard, runner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["arena"])


class HealthOut(BaseModel):
    ok: bool
    arena_dir: str
    n_submitted: int
    n_baselines: int


class LeaderboardRow(BaseModel):
    rank: int
    policy_name: str
    submitted_at: str
    overall_reward_mean: float
    overall_ci95: list[float | None]
    total_violations: float | int
    source: str


class LeaderboardOut(BaseModel):
    generated_at: str
    n_submissions: int
    n_baselines: int
    rows: list[LeaderboardRow]


class ArenaRunOut(BaseModel):
    policy_name: str
    submitted_at: str
    per_task: dict
    overall_reward_mean: float
    overall_ci95: list[float]
    total_violations: int
    rank_against_baseline: str


@router.get("/health", response_model=HealthOut)
def health():
    b = leaderboard.rebuild()
    return HealthOut(ok=True, arena_dir=str(leaderboard.ARENA_DIR),
                     n_submitted=b["n_submissions"], n_baselines=b["n_baselines"])


@router.get("/leaderboard", response_model=LeaderboardOut)
def get_leaderboard():
    b = leaderboard.rebuild()
    return LeaderboardOut(**b)


@router.post("/run", response_model=ArenaRunOut)
async def run(
    policy: UploadFile = File(..., description="PyTorch policy (.pt / .zip / .pth)"),
    name: str | None = Form(None, description="Display name for the leaderboard"),
    episodes: int = Form(50, ge=1, le=200, description="Episodes per task"),
):
    """Evaluate a submitted PyTorch policy on 3 SupplyMind tasks.

    Runtime scales ~linearly: 50 ep x 3 tasks ~ 1-3 min on RTX 4080 Laptop.
    Accepts sb3_contrib.MaskablePPO, stable_baselines3.PPO, or a raw torch.nn.Module.
    """
    if not policy.filename.endswith((".pt", ".zip", ".pth")):
        raise HTTPException(400, "policy must be .pt / .zip / .pth")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(policy.filename).suffix) as tmp:
        tmp.write(await policy.read())
        tmp_path = Path(tmp.name)

    try:
        display = name or Path(policy.filename).stem
        start = time.time()
        result = runner.evaluate_policy(tmp_path, n_episodes_per_task=episodes, policy_name=display)
        elapsed = time.time() - start
        logger.info("[arena] %s evaluated in %.1fs", display, elapsed)

        out_path = leaderboard.ARENA_DIR / f"{display}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_dict().__repr__())  # will be valid JSON via .to_dict()

        # proper JSON write
        import json as _json
        out_path.write_text(_json.dumps(result.to_dict(), indent=2))

        leaderboard.rebuild()
        return ArenaRunOut(**result.to_dict())
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
