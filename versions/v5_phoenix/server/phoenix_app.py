"""phoenix_app.py — Phoenix v5 FastAPI entry point.

Imports v4's `server.app:app` (frozen — no edits), then mounts:
    /arena     (versions.v5_phoenix.arena.router)
    /twin      (versions.v5_phoenix.counterfactual_twin.router)
    /replay    (versions.v5_phoenix.realtime_v5.replay_adapter)
    /phoenix   (status + version metadata)

Run:
    uvicorn versions.v5_phoenix.server.phoenix_app:app --host 0.0.0.0 --port 8000

Environment variables:
    FORCE_REPLAY=1   intercept /live/hormuz-closure with replay cache
    PHOENIX_VERSION  overrides the version string shown at /phoenix/status

Everything degrades gracefully: if any v5 router has a missing dependency
the import fails quietly and the rest keep working. You can always fall
back to running v4's server directly via `uvicorn server.app:app ...`.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from server.app import app  # v4 FastAPI, frozen
    logger.info("[phoenix] mounted v4 server.app")
except Exception as e:  # noqa: BLE001
    logger.warning("[phoenix] could not import server.app: %s", e)
    from fastapi import FastAPI
    app = FastAPI(title="SupplyMind Phoenix v5 (v4 app unavailable)")


def _try_mount(path: str, import_path: str, attr: str = "router") -> bool:
    try:
        mod = __import__(import_path, fromlist=[attr])
        router = getattr(mod, attr)
        app.include_router(router, prefix=path, tags=[path.strip("/")])
        logger.info("[phoenix] mounted %s -> %s", path, import_path)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("[phoenix] skipping %s (%s): %s", path, import_path, e)
        return False


# Mount v5 routers
_arena_ok = _try_mount("/arena", "versions.v5_phoenix.arena.router")
_twin_ok = _try_mount("/twin", "versions.v5_phoenix.counterfactual_twin.router")
_replay_ok = _try_mount("/replay", "versions.v5_phoenix.realtime_v5.replay_adapter")
_war_room_ok = _try_mount("/demo", "versions.v5_phoenix.war_room.router")


@app.get("/phoenix/status", tags=["phoenix"])
def phoenix_status():
    return {
        "version": os.environ.get("PHOENIX_VERSION", "v5.0-phoenix-ascensionism"),
        "force_replay_enabled": os.environ.get("FORCE_REPLAY") == "1",
        "mounted": {
            "arena": _arena_ok,
            "twin": _twin_ok,
            "replay": _replay_ok,
            "war_room": _war_room_ok,
        },
        "underlying_v4_app": getattr(app, "title", "unknown"),
    }


@app.get("/phoenix/routes", tags=["phoenix"])
def phoenix_routes():
    return sorted([
        {"path": getattr(r, "path", str(r)), "name": getattr(r, "name", None)}
        for r in app.routes
    ], key=lambda d: d["path"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("versions.v5_phoenix.server.phoenix_app:app", host="0.0.0.0", port=8000, reload=False)
