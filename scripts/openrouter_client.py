"""openrouter_client.py — rate-limited async OpenRouter client for SupplyMind.

Usage:
    from scripts.openrouter_client import OpenRouterClient, MODELS

    async with OpenRouterClient() as c:
        out = await c.chat("nvidia/nemotron-3-super", [{"role":"user","content":"..."}])

Rate-limit policy: 20 req/min, 1000 req/day (free tier). The client enforces
a local token-bucket at 18 req/min (conservative), retries on 429 with
exponential backoff, and logs every call to disk so we can audit usage.

Keys are read from env var OPENROUTER_API_KEY only — never a CLI arg,
never a literal, never echoed in logs.
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
USAGE_LOG = REPO_ROOT / ".openrouter_usage.jsonl"  # .gitignore this
BASE_URL = "https://openrouter.ai/api/v1"

# --- Model registry ---------------------------------------------------------
# Curated free-tier reasoning models verified from OpenRouter 2026-04.
# Ordered roughly by judge-quality for risk-assessment; use this order for
# the Frontier Panel v2 run.


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    short: str
    params_desc: str
    context: int
    role: str  # "judge" | "red-team" | "vision" | "ocr" | "utility"
    notes: str = ""


MODELS: list[ModelSpec] = [
    # --- top-tier judges (correct slugs verified against OpenRouter API) -
    ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", "nemotron3-super",
              "120B MoE / 12B active", 262_000, "judge",
              "1M-capable, multi-agent"),
    ModelSpec("inclusionai/ling-2.6-1t:free", "ling-2.6-1t",
              "1T params", 262_000, "judge",
              "GOING AWAY 2026-04-30 — use urgently"),
    ModelSpec("nousresearch/hermes-3-llama-3.1-405b:free", "hermes-3-405b",
              "405B", 131_000, "judge", "Frontier agentic"),
    ModelSpec("openai/gpt-oss-120b:free", "gpt-oss-120b",
              "117B MoE / 5.1B active", 131_000, "judge",
              "OpenAI open reasoning, native tool use"),
    ModelSpec("google/gemma-4-31b-it:free", "gemma-4-31b",
              "30.7B dense", 262_000, "judge",
              "Latest Google open, thinking mode"),
    ModelSpec("google/gemma-4-26b-a4b-it:free", "gemma-4-26b-a4b",
              "25.2B MoE / 3.8B active", 256_000, "judge",
              "Multimodal text+image+video, thinking"),
    ModelSpec("qwen/qwen3-next-80b-a3b-instruct:free", "qwen3-next-80b",
              "80B MoE / 3B active", 262_000, "judge", "Stable reasoning"),
    ModelSpec("z-ai/glm-4.5-air:free", "glm-4.5-air",
              "MoE w/ thinking", 131_000, "judge",
              "Configurable reasoning depth"),
    ModelSpec("meta-llama/llama-3.3-70b-instruct:free", "llama-3.3-70b",
              "70B dense", 66_000, "judge", "Meta SOTA baseline"),
    ModelSpec("nvidia/nemotron-3-nano-30b-a3b:free", "nemotron3-nano-30b",
              "30B MoE", 256_000, "judge", "Agentic mid-tier"),
    ModelSpec("minimax/minimax-m2.5:free", "minimax-m2.5",
              "large MoE", 197_000, "judge",
              "Real-world productivity, agent env specialist"),
    ModelSpec("nvidia/nemotron-nano-9b-v2:free", "nemotron-nano-9b",
              "9B", 128_000, "judge", "Cheap reasoning-trace generator"),
    # --- red-team / code -------------------------------------------------
    ModelSpec("qwen/qwen3-coder:free", "qwen3-coder-480b",
              "480B MoE / 35B active", 262_000, "red-team",
              "Adversarial reward-hack generator"),
    # --- vision / multimodal ---------------------------------------------
    ModelSpec("nvidia/nemotron-nano-12b-v2-vl:free", "nemotron-12b-vl",
              "12B multimodal", 131_000, "vision",
              "Port imagery + document understanding"),
    ModelSpec("google/gemma-3-12b-it:free", "gemma-3-12b",
              "12B vision+text", 33_000, "vision", "Port imagery fallback"),
    ModelSpec("google/gemma-3-4b-it:free", "gemma-3-4b",
              "4B vision+text", 33_000, "vision", "Tiny fast vision"),
    # --- utility tier ----------------------------------------------------
    ModelSpec("meta-llama/llama-3.2-3b-instruct:free", "llama-3.2-3b",
              "3B", 131_000, "utility", "Cheap text"),
    ModelSpec("openai/gpt-oss-20b:free", "gpt-oss-20b",
              "21B MoE / 3.6B active", 131_000, "utility",
              "Light tool-use judge"),
]


JUDGE_SLUGS = [m.slug for m in MODELS if m.role == "judge"]
REDTEAM_SLUGS = [m.slug for m in MODELS if m.role == "red-team"]
VISION_SLUGS = [m.slug for m in MODELS if m.role == "vision"]


# --- Rate limiter -----------------------------------------------------------


class _RateLimiter:
    """Simple async token-bucket: max N requests per window seconds.

    Configured for OpenRouter free tier: 18 req/min (conservative of their 20).
    Also enforces a daily budget: 950 req/day (conservative of their 1000).
    """

    def __init__(self, per_minute: int = 18, per_day: int = 950) -> None:
        self._per_min = per_minute
        self._per_day = per_day
        self._min_slots: list[float] = []
        self._day_slots: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # prune windows
            self._min_slots = [t for t in self._min_slots if now - t < 60.0]
            self._day_slots = [t for t in self._day_slots if now - t < 86400.0]
            # day check
            if len(self._day_slots) >= self._per_day:
                wait = 86400.0 - (now - self._day_slots[0]) + 0.1
                raise RuntimeError(
                    f"OpenRouter daily budget exhausted "
                    f"({self._per_day} req). Need new key. Reset in ~{int(wait/3600)}h"
                )
            # minute check
            if len(self._min_slots) >= self._per_min:
                wait = 60.0 - (now - self._min_slots[0]) + 0.2
                logger.info("[rate] waiting %.1fs for per-minute budget", wait)
                await asyncio.sleep(wait)
                now = time.monotonic()
                self._min_slots = [t for t in self._min_slots if now - t < 60.0]
            self._min_slots.append(now)
            self._day_slots.append(now)

    def remaining(self) -> dict[str, int]:
        now = time.monotonic()
        mins = [t for t in self._min_slots if now - t < 60.0]
        days = [t for t in self._day_slots if now - t < 86400.0]
        return {
            "per_min_used": len(mins),
            "per_min_budget": self._per_min,
            "per_day_used": len(days),
            "per_day_budget": self._per_day,
        }


# --- Client -----------------------------------------------------------------


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
    raw: dict = field(default_factory=dict)


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        # Read key from env; never accept as string arg in production.
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            # Try loading .env file manually (no python-dotenv dependency)
            env_path = REPO_ROOT / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in env or .env")
        self._key = key
        self._limiter = _RateLimiter()
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {self._key}",
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_SITE_URL",
                    "https://huggingface.co/spaces/Shaurya-Noodle/Supplymind",
                ),
                "X-Title": os.environ.get(
                    "OPENROUTER_APP_NAME", "SupplyMind-Hackathon-Finals-2026"
                ),
            },
        )
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> "OpenRouterClient":
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
        retries: int = 2,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        last_err: str = ""
        for attempt in range(retries + 1):
            await self._limiter.acquire()
            t0 = time.monotonic()
            try:
                r = await self._client.post("/chat/completions", json=payload)
                dt = time.monotonic() - t0
                if r.status_code == 429:
                    last_err = f"429 rate-limit: {r.text[:200]}"
                    await asyncio.sleep(2 ** attempt * 3)
                    continue
                if r.status_code >= 400:
                    body = r.text
                    self._log({"model": model, "status": r.status_code,
                                "error": body[:300], "t": time.time()})
                    return ChatResult(ok=False, model=model, http_status=r.status_code,
                                      error=body[:400], latency_s=dt)
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content", "")
                usage = data.get("usage") or {}
                self._log({"model": model, "status": 200,
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "latency_s": round(dt, 2), "t": time.time()})
                return ChatResult(
                    ok=True, model=model, content=content, latency_s=dt,
                    tokens_prompt=usage.get("prompt_tokens", 0),
                    tokens_completion=usage.get("completion_tokens", 0),
                    http_status=200, raw=data,
                )
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(2 ** attempt)
        return ChatResult(ok=False, model=model, error=last_err)

    def _log(self, row: dict) -> None:
        try:
            with open(USAGE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def budget_remaining(self) -> dict[str, int]:
        return self._limiter.remaining()
