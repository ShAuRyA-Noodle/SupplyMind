"""supplymind_client.py — typed HTTP client for a remote SupplyMind OpenEnv.

Design principles (hackathon judge doc §"Engineer it cleanly"):
  * Respects client/server separation — no `from server import ...` anywhere.
  * Uses only stdlib + `httpx` (already a core dependency).
  * Thin: only HTTP transport + lightweight validation. No business logic.
  * Works against either a local `uvicorn server.app:app` or the live HF Space.

Example — against the live Space, no local install needed:

    from client import SupplyMindClient
    env = SupplyMindClient("https://shaurya-noodle-supplymind.hf.space")
    obs = env.reset(task_id="easy_typhoon_response", seed=42)
    while not obs.get("done"):
        action = {"task_id": env.current_task_id,
                  "action_type": "NO_OP", "target": 0, "magnitude": 0.0}
        obs = env.step(action)
    print(env.grade())
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class SupplyMindClient:
    """Thin HTTP client for a remote SupplyMind OpenEnv server.

    Args:
        base_url: URL of the server (e.g. "http://localhost:8000" or the HF Space).
        session_id: Optional session identifier for concurrent-session isolation.
        timeout_s: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout_s = timeout_s
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)
        self.current_task_id: str | None = None
        self.current_episode_id: str | None = None

    # --- OpenEnv gym-style API -------------------------------------------------

    def reset(
        self,
        task_id: str = "easy_typhoon_response",
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """POST /reset — start a new episode."""
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if self.session_id:
            payload["session_id"] = self.session_id
        r = self._client.post("/reset", json=payload)
        r.raise_for_status()
        obs = r.json()
        self.current_task_id = task_id
        self.current_episode_id = obs.get("episode_id") or episode_id
        return obs

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """POST /step — apply an action and return the next observation."""
        payload: dict[str, Any] = {"action": action}
        if self.session_id:
            payload["session_id"] = self.session_id
        r = self._client.post("/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict[str, Any]:
        """GET /state — current episode metadata."""
        params = {"session_id": self.session_id} if self.session_id else None
        r = self._client.get("/state", params=params)
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict[str, Any]:
        """POST /grader — score the current episode against the task rubric."""
        payload = {"session_id": self.session_id} if self.session_id else {}
        r = self._client.post("/grader", json=payload)
        r.raise_for_status()
        return r.json()

    # --- OpenEnv introspection -------------------------------------------------

    def schema(self) -> dict[str, Any]:
        """GET /schema — action + observation JSON schemas."""
        r = self._client.get("/schema")
        r.raise_for_status()
        return r.json()

    def metadata(self) -> dict[str, Any]:
        """GET /metadata — env metadata (name, version, task list)."""
        r = self._client.get("/metadata")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> list[dict[str, Any]]:
        """GET /tasks — list of available task definitions."""
        r = self._client.get("/tasks")
        r.raise_for_status()
        body = r.json()
        return body.get("tasks", body) if isinstance(body, dict) else body

    def health(self) -> bool:
        """GET /health — liveness probe."""
        try:
            r = self._client.get("/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # --- Episode helper --------------------------------------------------------

    def rollout(
        self,
        policy,
        task_id: str = "easy_typhoon_response",
        seed: int | None = None,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Run one full episode with a callable `policy(observation) -> action`.

        Returns the grade dict plus a trajectory log. Matches the OpenEnv
        reset/step loop exactly — suitable for use as an RL reward oracle.
        """
        obs = self.reset(task_id=task_id, seed=seed)
        trajectory: list[dict[str, Any]] = []
        cumulative_reward = 0.0
        for _ in range(max_steps):
            action = policy(obs)
            obs = self.step(action)
            trajectory.append({"action": action, "observation": obs})
            cumulative_reward += float(obs.get("reward", 0.0))
            if obs.get("done"):
                break
        grade = self.grade()
        grade["cumulative_reward"] = cumulative_reward
        grade["n_steps"] = len(trajectory)
        grade["trajectory"] = trajectory
        return grade

    # --- Context-manager plumbing ---------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupplyMindClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def __main__() -> None:  # pragma: no cover
    """Smoke test — hit /health on the live HF Space."""
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://shaurya-noodle-supplymind.hf.space"
    with SupplyMindClient(url) as env:
        print(json.dumps({"base_url": url, "health": env.health(),
                          "metadata": env.metadata()}, indent=2)[:600])


if __name__ == "__main__":
    __main__()
