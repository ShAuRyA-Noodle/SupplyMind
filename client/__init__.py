"""client — thin HTTP client for the SupplyMind OpenEnv.

This package provides a typed client for remote SupplyMind environments
(either a local uvicorn server or the live HuggingFace Space). It is
intentionally decoupled from `server/`: no module in `client/` imports
from `server/` — the only shared surface is the JSON schema published
by the server's `/schema` endpoint.

Usage:
    from client import SupplyMindClient
    env = SupplyMindClient("http://localhost:8000")
    obs = env.reset(task_id="easy_typhoon_response", seed=42)
    # step() takes a flat SupplyMindAction payload (see models.SupplyMindAction):
    obs = env.step({"action_type": "issue_supplier_alert", "target_node_id": "SUP001"})
    print(env.state())
"""
from .supplymind_client import SupplyMindClient

__all__ = ["SupplyMindClient"]
