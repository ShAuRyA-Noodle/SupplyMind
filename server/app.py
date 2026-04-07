"""
SupplyMind FastAPI Application

Thin HTTP layer over SupplyMindEnvironment. All game logic lives in
supply_environment.py -- this file only handles request/response mapping,
error handling, and endpoint definitions.

Required OpenEnv endpoints:
    GET  /health  -- Health check
    POST /reset   -- Reset environment with optional task_id
    POST /step    -- Execute one action
    GET  /state   -- Return current episode metadata
    GET  /tasks   -- List available tasks and action schema
    POST /grader  -- Grade a completed episode
    POST /baseline -- Run baseline inference on all tasks
"""

from __future__ import annotations

import logging
import traceback

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from pydantic import BaseModel

from models import SupplyMindAction
from server.supply_environment import SupplyMindEnvironment


class ResetRequest(BaseModel):
    """Optional body for POST /reset."""
    task_id: Optional[str] = "easy_typhoon_response"
    seed: Optional[int] = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown events. Pre-loads data for fast first request."""
    from server.tasks.registry import TaskRegistry

    logger.info("SupplyMind environment server started.")
    task_ids = [t.task_id for t in TaskRegistry.list_tasks()]
    logger.info("Available tasks: %s", task_ids)

    # Pre-warm: load all graph and disruption JSONs into memory so the
    # first /reset request doesn't pay a cold-start penalty.
    try:
        warm_env = SupplyMindEnvironment()
        for tid in task_ids:
            warm_env.reset(task_id=tid)
        logger.info("Pre-warmed all %d tasks.", len(task_ids))
    except Exception as e:
        logger.warning("Pre-warm failed (non-fatal): %s", e)

    yield


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SupplyMind",
    description=(
        "Supply chain risk management OpenEnv environment. "
        "An AI agent manages a global supply chain through real-world disruptions "
        "(typhoons, port strikes, sanctions, cascading crises) to minimize "
        "financial impact."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow CORS for browser-based clients and HF Spaces iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment pool keyed by session_id for concurrent isolation.
# OpenEnv evaluation typically runs sequentially, but this supports
# multiple concurrent sessions (e.g., multiple judges or demo users).
# A global lock protects the session registry; each session gets its
# own SupplyMindEnvironment instance.
import asyncio

_sessions: dict[str, SupplyMindEnvironment] = {}
_sessions_lock = asyncio.Lock()
_DEFAULT_SESSION = "default"

# Max sessions to prevent memory exhaustion
_MAX_SESSIONS = 20


async def _get_env(session_id: str | None = None) -> SupplyMindEnvironment:
    """Get or create an environment for the given session."""
    sid = session_id or _DEFAULT_SESSION
    async with _sessions_lock:
        if sid not in _sessions:
            if len(_sessions) >= _MAX_SESSIONS:
                # Evict oldest session (first key)
                oldest = next(iter(_sessions))
                del _sessions[oldest]
            _sessions[sid] = SupplyMindEnvironment()
        return _sessions[sid]


# Keep a module-level reference for backward compat with /baseline
env = SupplyMindEnvironment()
_env_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Health check endpoint. Returns 200 if the server is running."""
    return {
        "status": "healthy",
        "environment": "supplymind",
        "version": "1.0.0",
    }


@app.get("/metadata")
async def metadata() -> dict:
    """
    Return environment metadata.

    Required by the OpenEnv runtime validation contract.
    """
    return {
        "name": "supplymind",
        "description": (
            "Supply chain risk management environment. An AI agent manages a "
            "global supply chain through real-world disruptions (typhoons, port "
            "strikes, sanctions, cascading crises) to minimize financial impact."
        ),
        "version": "1.0.0",
        "mode": "simulation",
        "tags": ["openenv", "supply-chain", "risk-management"],
    }


@app.get("/schema")
async def schema() -> dict:
    """
    Return JSON schemas for action, observation, and state models.

    Required by the OpenEnv runtime validation contract.
    """
    from models import SupplyMindObservation, SupplyMindState

    return {
        "action": SupplyMindAction.model_json_schema(),
        "observation": SupplyMindObservation.model_json_schema(),
        "state": SupplyMindState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp_handler(request: dict = {}) -> dict:
    """
    Model Context Protocol (MCP) JSON-RPC 2.0 endpoint.

    Required by the OpenEnv runtime validation contract. Supports
    'initialize' and 'tools/list' methods for tool discovery.
    """
    method = request.get("method", "")
    req_id = request.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "supplymind", "version": "1.0.0"},
                "capabilities": {"tools": {"listChanged": False}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment with a task_id",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "enum": [
                                        "easy_typhoon_response",
                                        "medium_multi_front",
                                        "hard_cascading_crisis",
                                    ],
                                }
                            },
                        },
                    },
                    {
                        "name": "step",
                        "description": "Execute one action in the environment",
                        "inputSchema": SupplyMindAction.model_json_schema(),
                    },
                    {
                        "name": "state",
                        "description": "Get current episode metadata",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "grade",
                        "description": "Grade the current episode",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
        }

    # Default: return server capabilities
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "supplymind", "version": "1.0.0"},
            "capabilities": {"tools": {"listChanged": False}},
        },
    }



@app.post("/reset")
async def reset(
    request: Optional[ResetRequest] = None,
    task_id: str = Query(
        default="easy_typhoon_response",
        description="Task ID. One of: easy_typhoon_response, medium_multi_front, hard_cascading_crisis",
    ),
    x_session_id: Optional[str] = Header(default=None),
) -> dict:
    """
    Reset the environment for a new episode.

    Accepts task_id as either:
    - Query parameter: POST /reset?task_id=easy_typhoon_response
    - Request body: POST /reset with JSON body {"task_id": "easy_typhoon_response", "seed": 42}

    Optional parameters:
    - seed (int): Episode variation seed. Same seed = identical episode for reproducibility.
      Different seeds produce different disruption timings/severities via jitter.
      Omit for default deterministic behavior.
    - X-Session-Id header: Per-session isolation for concurrent users.

    Returns the initial observation of the supply chain state.
    """
    # Body takes precedence over query param
    effective_task_id = (request.task_id if request and request.task_id else task_id) or "easy_typhoon_response"
    effective_seed = request.seed if request else None
    session_env = await _get_env(x_session_id)
    try:
        obs = session_env.reset(task_id=effective_task_id, seed=effective_seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data file not found: {e}. Ensure server/data/ files exist.",
        )
    except Exception as e:
        logger.error("Reset failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(
    action: SupplyMindAction,
    x_session_id: Optional[str] = Header(default=None),
) -> dict:
    """
    Execute one action in the environment.

    The agent submits a single action per step. The simulation advances
    one day, applies disruptions, updates financials, and returns the
    new observation with reward and done flag.
    """
    session_env = await _get_env(x_session_id)
    if session_env.engine is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    try:
        obs = session_env.step(action)
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Step failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state(
    x_session_id: Optional[str] = Header(default=None),
) -> dict:
    """
    Return current episode metadata.

    Includes episode_id, step count, task info, cumulative reward,
    and whether the episode is done.
    """
    session_env = await _get_env(x_session_id)
    return session_env.state.model_dump()


@app.get("/tasks")
async def list_tasks() -> dict:
    """
    List all available tasks and the action schema.

    Returns task definitions (id, name, difficulty, description, episode
    length, budget) and the JSON schema for SupplyMindAction.
    """
    from server.tasks.registry import TaskRegistry

    tasks = TaskRegistry.list_tasks()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "episode_length": t.episode_length,
                "budget": t.budget,
            }
            for t in tasks
        ],
        "action_schema": SupplyMindAction.model_json_schema(),
    }


@app.post("/grader")
async def grade(
    x_session_id: Optional[str] = Header(default=None),
) -> dict:
    """
    Grade the current or most recent episode.

    Returns a score in [0.0, 1.0] with a per-component breakdown.
    Can be called during an episode (partial grade) or after it ends.
    """
    session_env = await _get_env(x_session_id)
    if session_env.engine is None:
        raise HTTPException(
            status_code=400,
            detail="No episode to grade. Call POST /reset and run an episode first.",
        )
    try:
        result = session_env.grade()
        return result
    except Exception as e:
        logger.error("Grading failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.post("/baseline")
async def run_baseline() -> dict:
    """
    Run the baseline inference agent on all 3 tasks.

    Requires at least one of HF_TOKEN, API_KEY, or OPENAI_API_KEY to be set.
    Uses the model specified by MODEL_NAME (default: gpt-4o) with
    temperature=0.1 for reproducible scores.

    Returns scores for all 3 tasks and an average score.
    """
    import os
    api_key = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "API key not set",
                "message": (
                    "Set HF_TOKEN (or API_KEY / OPENAI_API_KEY) environment variable "
                    "to run baseline inference."
                ),
                "instructions": (
                    "docker run -e HF_TOKEN=hf_... -e MODEL_NAME=gpt-4o "
                    "-p 8000:8000 supplymind"
                ),
            },
        )
    async with _env_lock:
        try:
            from baseline import run_all_baselines
            results = run_all_baselines(env)
            return results
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="baseline.py not found. Ensure openai>=1.0 is installed.",
            )
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error("Baseline failed: %s\n%s", e, traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Baseline inference failed: {str(e)}",
            )


# ---------------------------------------------------------------------------
# Run directly with: python -m server.app
# Or via entry point: supplymind-server
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the SupplyMind environment server."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

