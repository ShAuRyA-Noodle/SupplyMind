"""router.py — FastAPI router for the Counterfactual Digital Twin.

    POST /twin/run
        body: {severity: float, brent_usd: float, task_id: str, n_rollouts: int}
        returns: TwinReport.to_dict()

    GET  /twin/health

Mounted under /twin by server/phoenix_app.py.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from . import twin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["twin"])


class TwinRequest(BaseModel):
    severity: float = Field(0.85, ge=0.0, le=1.0)
    brent_usd: float = Field(85.0, ge=0.0)
    task_id: str = Field("hard_cascading_crisis")
    n_rollouts: int = Field(100, ge=10, le=500)


class HealthOut(BaseModel):
    ok: bool
    default_task: str
    default_rollouts: int


@router.get("/health", response_model=HealthOut)
def health():
    return HealthOut(ok=True, default_task=twin.DEFAULT_TASK, default_rollouts=twin.N_ROLLOUTS)


@router.post("/run")
def run(req: TwinRequest):
    rep = twin.run_twin(
        severity=req.severity,
        brent_usd=req.brent_usd,
        task_id=req.task_id,
        n_rollouts=req.n_rollouts,
    )
    return rep.to_dict()
