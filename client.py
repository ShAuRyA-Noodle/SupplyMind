"""
SupplyMind Client

HTTP client for interacting with the SupplyMind environment server.
Uses httpx for synchronous HTTP calls with automatic JSON serialization
and Pydantic model parsing.

Usage:
    from client import SupplyMindClient
    from models import SupplyMindAction

    client = SupplyMindClient("http://localhost:8000")
    obs = client.reset("easy_typhoon_response")

    action = SupplyMindAction(
        action_type="activate_backup_supplier",
        target_node_id="SUP001",
        backup_supplier_id="SUP002",
    )
    obs = client.step(action)
    print(obs.situation_summary)

    result = client.grade()
    print(f"Score: {result['score']}")
    client.close()
"""

from __future__ import annotations

from typing import Any

import httpx

from models import SupplyMindAction, SupplyMindObservation, SupplyMindState


class SupplyMindClient:
    """
    Synchronous HTTP client for the SupplyMind environment server.

    Provides typed methods that match the OpenEnv interface:
    reset, step, state, tasks, grade, and close.

    Args:
        base_url: Server URL (default: http://localhost:8000).
        timeout: Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    def health(self) -> dict[str, Any]:
        """
        Check server health.

        Returns:
            Dict with status, environment name, and version.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
        """
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str = "easy_typhoon_response") -> SupplyMindObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Task to run. One of:
                - "easy_typhoon_response"
                - "medium_multi_front"
                - "hard_cascading_crisis"

        Returns:
            Initial observation of the supply chain state.

        Raises:
            httpx.HTTPStatusError: If the server returns an error (e.g., unknown task_id).
        """
        resp = self.client.post("/reset", params={"task_id": task_id})
        resp.raise_for_status()
        return SupplyMindObservation(**resp.json())

    def step(self, action: SupplyMindAction) -> SupplyMindObservation:
        """
        Execute one action in the environment.

        Args:
            action: The action to take this step. Use SupplyMindAction with
                the appropriate action_type and parameters.

        Returns:
            Observation after the action, including reward and done flag.

        Raises:
            httpx.HTTPStatusError: If the server returns an error
                (e.g., episode not started, episode done).
        """
        resp = self.client.post(
            "/step",
            json=action.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return SupplyMindObservation(**resp.json())

    def state(self) -> SupplyMindState:
        """
        Get current episode metadata.

        Returns:
            Episode state with step count, task info, cumulative reward, and done flag.

        Raises:
            httpx.HTTPStatusError: If the server returns an error.
        """
        resp = self.client.get("/state")
        resp.raise_for_status()
        return SupplyMindState(**resp.json())

    def tasks(self) -> dict[str, Any]:
        """
        List all available tasks and the action schema.

        Returns:
            Dict with "tasks" (list of task definitions) and
            "action_schema" (JSON schema for SupplyMindAction).

        Raises:
            httpx.HTTPStatusError: If the server returns an error.
        """
        resp = self.client.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> dict[str, Any]:
        """
        Grade the current or most recent episode.

        Returns:
            Dict with score (0.0-1.0), task info, steps taken,
            cumulative reward, and per-component breakdown.

        Raises:
            httpx.HTTPStatusError: If the server returns an error
                (e.g., no episode has been run).
        """
        resp = self.client.post("/grader")
        resp.raise_for_status()
        return resp.json()

    def run_baseline(self) -> dict[str, Any]:
        """
        Trigger baseline inference on all 3 tasks.

        Requires OPENAI_API_KEY to be set on the server.

        Returns:
            Dict with baseline scores for each task.

        Raises:
            httpx.HTTPStatusError: If baseline fails (e.g., no API key).
        """
        resp = self.client.post("/baseline")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        self.client.close()

    def __enter__(self) -> SupplyMindClient:
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close client on context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"SupplyMindClient(base_url='{self.base_url}')"
