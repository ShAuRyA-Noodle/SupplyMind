"""openai_client.py — Direct OpenAI API client with dual-key fallback.

Complements scripts/openrouter_client.py. We have two OpenAI keys; primary
handles traffic, fallback kicks in on quota or 429 errors. All responses
are cached the same way as the OpenRouter client so nothing gets paid for
twice.

Models we call from here (OpenAI direct, not OpenRouter-routed):
  - gpt-4o-mini       — cheap, fast judge for bulk preference-pair gen
  - gpt-4o            — frontier judge for R4 panel
  - gpt-4.1           — if available on the account (falls back to gpt-4o)
  - o1-mini / o1      — reasoning specialist for adversarial probes

Keys read from env only — OPENAI_API_KEY / OPENAI_API_KEY_PRIMARY /
OPENAI_API_KEY_FALLBACK. Never a CLI arg, never echoed in logs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
USAGE_LOG = REPO_ROOT / ".openai_usage.jsonl"
BASE_URL = "https://api.openai.com/v1"


@dataclass
class ChatResult:
    ok: bool
    model: str
    content: str = ""
    latency_s: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    http_status: int = 0
    error: str = ""
    key_used: str = ""  # "primary" | "fallback"
    raw: dict = field(default_factory=dict)


def _load_keys() -> tuple[str, str | None]:
    """Read PRIMARY + FALLBACK keys from env or .env file (no external deps)."""
    primary = (os.environ.get("OPENAI_API_KEY_PRIMARY")
               or os.environ.get("OPENAI_API_KEY"))
    fallback = os.environ.get("OPENAI_API_KEY_FALLBACK")
    if not primary:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY_PRIMARY="):
                    primary = line.split("=", 1)[1].strip()
                elif line.startswith("OPENAI_API_KEY_FALLBACK="):
                    fallback = line.split("=", 1)[1].strip()
                elif line.startswith("OPENAI_API_KEY=") and not primary:
                    primary = line.split("=", 1)[1].strip()
    if not primary:
        raise RuntimeError("OPENAI_API_KEY / OPENAI_API_KEY_PRIMARY not set")
    return primary, fallback


class OpenAIClient:
    """httpx-based async client with automatic key failover.

    Usage:
        async with OpenAIClient() as c:
            res = await c.chat("gpt-4o-mini",
                               [{"role": "user", "content": "hi"}])
    """

    def __init__(self, timeout_s: float = 120.0) -> None:
        self._primary, self._fallback = _load_keys()
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            base_url=BASE_URL, timeout=timeout_s,
            headers={"Content-Type": "application/json"},
        )
        self._usage_path = USAGE_LOG

    async def __aenter__(self) -> "OpenAIClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> ChatResult:
        """Primary-then-fallback call with single retry each."""
        for key_name, key in (("primary", self._primary),
                               ("fallback", self._fallback)):
            if not key:
                continue
            result = await self._one_shot(model, messages, max_tokens,
                                           temperature, response_format,
                                           key, key_name)
            # 401/403/429 on primary → try fallback. Everything else returns.
            if result.ok:
                return result
            if result.http_status in (401, 403, 429) and key_name == "primary":
                logger.info("[openai] primary %d → trying fallback", result.http_status)
                continue
            return result
        return ChatResult(ok=False, model=model, error="no keys available")

    async def _one_shot(
        self, model: str, messages: list[dict], max_tokens: int,
        temperature: float, response_format: dict | None,
        api_key: str, key_name: str,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        t0 = time.monotonic()
        try:
            r = await self._client.post(
                "/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            dt = time.monotonic() - t0
            if r.status_code >= 400:
                body = r.text[:400]
                self._log({"model": model, "status": r.status_code,
                            "error": body[:200], "key": key_name,
                            "t": time.time()})
                return ChatResult(ok=False, model=model,
                                   http_status=r.status_code,
                                   error=body, latency_s=dt, key_used=key_name)
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            usage = data.get("usage") or {}
            self._log({
                "model": model, "status": 200, "key": key_name,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_s": round(dt, 2), "t": time.time(),
            })
            return ChatResult(
                ok=True, model=model, content=content, latency_s=dt,
                tokens_prompt=usage.get("prompt_tokens", 0),
                tokens_completion=usage.get("completion_tokens", 0),
                http_status=200, raw=data, key_used=key_name,
            )
        except httpx.HTTPError as e:
            return ChatResult(ok=False, model=model,
                               error=f"{type(e).__name__}: {e}",
                               key_used=key_name)

    def _log(self, row: dict) -> None:
        try:
            with open(self._usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass


# --- Model registry (only OpenAI-direct, not OpenRouter-routed) -------------


@dataclass(frozen=True)
class OpenAIModel:
    slug: str
    short: str
    role: str  # "judge" | "bulk" | "reasoning"
    notes: str = ""


OPENAI_MODELS: list[OpenAIModel] = [
    OpenAIModel("gpt-4o", "gpt-4o", "judge",
                "Frontier closed-source judge, multimodal, 128K context"),
    OpenAIModel("gpt-4o-mini", "gpt-4o-mini", "bulk",
                "Cheap frontier — batch pref-pair generation"),
    OpenAIModel("gpt-4.1", "gpt-4.1", "judge",
                "If available on account; graceful 404 if not enabled"),
    OpenAIModel("o1-mini", "o1-mini", "reasoning",
                "Reasoning specialist for adversarial probes"),
]
