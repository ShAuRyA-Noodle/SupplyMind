"""Phoenix FastAPI entry point.

`phoenix_app.py` imports v4's `server.app` (frozen) and mounts v5 routers
(`/arena`, `/twin`, `/replay`) without touching any v4 code. Judges run a
single `uvicorn` invocation and get every endpoint at once.
"""
