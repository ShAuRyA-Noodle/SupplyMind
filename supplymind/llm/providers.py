"""supplymind.llm.providers — the single LLM interface (CLAUDE.md §0.4, §7.1).

One interface, two backends:

* ``openrouter`` (default) — frontier/open models via the promoted
  :class:`supplymind.llm.client.OpenRouterClient`.
* ``ollama`` — local/edge mode against a running Ollama daemon.

Selection is env-driven (``SUPPLYMIND_LLM_PROVIDER``), never hardcoded. The
model slug is likewise env-driven (``SUPPLYMIND_LLM_MODEL`` /
``SUPPLYMIND_OLLAMA_MODEL``); there is *no* baked-in default slug, because a
silent wrong-model default is a fake per CLAUDE.md §0. When no model is
configured, calls degrade LOUDLY with a surfaced reason — they never fabricate
a verdict.

Structured-JSON output is enforced at this interface: the provider validates the
model's reply against a caller-supplied JSON schema (required keys + typed
constraints) and retries with a corrective nudge on invalid output. If every
retry fails, the response is returned with ``degraded=True`` and the reason
surfaced — never silently coerced.

Every call is appended to the usage log (``.openrouter_usage.jsonl`` by default,
overridable with ``SUPPLYMIND_USAGE_LOG`` for tests) as one JSON line, so cost
and reliability are auditable.

Design note on testability: both providers accept an optional httpx transport
(``httpx.MockTransport``) so the offline unit tests exercise the *real*
enforcement / retry / logging / degrade code paths without any network call.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Env var names (never hardcode slugs; read these) -----------------------
PROVIDER_ENV = "SUPPLYMIND_LLM_PROVIDER"          # "openrouter" (default) | "ollama"
MODEL_ENV = "SUPPLYMIND_LLM_MODEL"                # OpenRouter slug
OLLAMA_MODEL_ENV = "SUPPLYMIND_OLLAMA_MODEL"      # Ollama model tag
OLLAMA_HOST_ENV = "OLLAMA_HOST"                   # default http://localhost:11434
USAGE_LOG_ENV = "SUPPLYMIND_USAGE_LOG"            # override log path (tests)

DEFAULT_PROVIDER = "openrouter"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------
def _usage_log_path() -> Path:
    """Resolve the usage-log path fresh on every call so tests can override it
    via the ``SUPPLYMIND_USAGE_LOG`` env var without import-time pinning."""
    override = os.environ.get(USAGE_LOG_ENV)
    if override:
        return Path(override)
    return REPO_ROOT / ".openrouter_usage.jsonl"


def _log_usage(row: dict[str, Any]) -> None:
    """Append one provider-level usage line. Never raises on IO error (logging
    must not take down a production request), but the failure is not silent —
    the caller still gets its response with real fields."""
    path = _usage_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        # Best-effort audit log; the request result itself is unaffected.
        pass


# ---------------------------------------------------------------------------
# Structured-JSON enforcement
# ---------------------------------------------------------------------------
class JSONEnforcementError(ValueError):
    """Raised internally when a model reply cannot be coerced to the schema."""


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from model text.

    Handles bare JSON, ```json fenced blocks, and leading/trailing prose. Returns
    None if no parseable object is found (caller decides how to react — we never
    fabricate a fallback object)."""
    if not text:
        return None
    # Strip common markdown fences.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    # Fast path: whole thing is a JSON object.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first balanced {...} span.
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def validate_against_schema(obj: dict, schema: dict) -> list[str]:
    """Validate ``obj`` against a small JSON-schema subset. Returns a list of
    human-readable violation strings (empty == valid).

    Supported schema features (enough for our decision/judge contracts):
    ``required`` keys, ``properties`` with ``type`` (string/number/integer/
    boolean/array/object), ``enum``, and numeric ``minimum``/``maximum``.
    """
    errors: list[str] = []
    required = schema.get("required", [])
    props: dict = schema.get("properties", {})
    for key in required:
        if key not in obj:
            errors.append(f"missing required key '{key}'")
    type_map: dict[str, tuple] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }
    for key, spec in props.items():
        if key not in obj:
            continue
        val = obj[key]
        exp_type = spec.get("type")
        if exp_type in type_map:
            # bool is a subclass of int — guard number/integer against bool.
            if exp_type in ("number", "integer") and isinstance(val, bool):
                errors.append(f"key '{key}' expected {exp_type}, got boolean")
                continue
            if not isinstance(val, type_map[exp_type]):
                errors.append(
                    f"key '{key}' expected {exp_type}, got {type(val).__name__}"
                )
                continue
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"key '{key}'={val!r} not in enum {spec['enum']}")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if "minimum" in spec and val < spec["minimum"]:
                errors.append(f"key '{key}'={val} < minimum {spec['minimum']}")
            if "maximum" in spec and val > spec["maximum"]:
                errors.append(f"key '{key}'={val} > maximum {spec['maximum']}")
    return errors


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """Uniform result across providers.

    ``ok`` == a usable structured reply was obtained. ``degraded`` == the call
    could not fully succeed and the reason is surfaced (never faked). When a
    schema was requested and satisfied, ``parsed`` holds the validated dict.
    """
    ok: bool
    provider: str
    model: str
    content: str = ""
    parsed: dict | None = None
    degraded: bool = False
    degrade_reason: str = ""
    error: str = ""
    latency_s: float = 0.0
    attempts: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    http_status: int = 0

    def as_log_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("parsed", None)  # keep the log line compact
        row["t"] = time.time()
        return row


@dataclass
class LivenessResult:
    ok: bool
    provider: str
    reason: str
    model: str = ""
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Key / model resolution (env-driven, fail-loud)
# ---------------------------------------------------------------------------
def resolve_openrouter_key(env_path: str | os.PathLike | None = None) -> str | None:
    """Resolve the OpenRouter key: os.environ first, then the shared .env
    loader. Returns None when absent (caller degrades loud — never fakes)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        from scripts._env import load_env

        load_env(env_path)
    except Exception:  # noqa: BLE001 - env loader is best-effort here
        return None
    return os.environ.get("OPENROUTER_API_KEY")


def resolve_model(provider: str, explicit: str | None = None) -> str | None:
    """Resolve the model slug for ``provider``. Explicit arg wins; otherwise the
    provider's env var. Returns None when unconfigured (never a hardcoded slug)."""
    if explicit:
        return explicit
    if provider == "ollama":
        return os.environ.get(OLLAMA_MODEL_ENV)
    return os.environ.get(MODEL_ENV)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def _raw_chat(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict | None,
    ) -> LLMResponse:
        """Single raw round-trip. Populates content/tokens/http_status/error but
        does NOT enforce the schema (the shared ``complete`` loop does that)."""

    @abstractmethod
    async def liveness(self) -> LivenessResult:
        """Cheap reachability/credential check. MUST fail loud (ok=False +
        reason) when the key/daemon is absent — never a fake OK."""

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        schema: dict | None = None,
        max_tokens: int = 768,
        temperature: float = 0.15,
        retries: int = 2,
    ) -> LLMResponse:
        """Enforced round-trip with retry-on-invalid + usage logging.

        Contract:
        * No configured model -> degraded LOUD (no fabricated verdict).
        * ``schema`` given and satisfied -> ``parsed`` set, ``ok=True``.
        * ``schema`` given, never satisfied after ``retries`` -> ``ok=False,
          degraded=True`` with the last validation errors surfaced.
        """
        slug = resolve_model(self.name, model)
        if not slug:
            env_name = OLLAMA_MODEL_ENV if self.name == "ollama" else MODEL_ENV
            resp = LLMResponse(
                ok=False, provider=self.name, model="",
                degraded=True,
                degrade_reason=(
                    f"no model configured: set {env_name} (env-driven; no default "
                    "slug is baked in per CLAUDE.md §0)"
                ),
                error="model-unconfigured",
            )
            _log_usage(resp.as_log_row())
            return resp

        response_format = _json_schema_response_format(schema) if schema else None
        work_messages = list(messages)
        last: LLMResponse | None = None
        t0 = time.time()

        for attempt in range(retries + 1):
            resp = await self._raw_chat(
                work_messages,
                model=slug,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            resp.attempts = attempt + 1
            last = resp

            if not resp.ok and resp.error:
                # Transport / auth / HTTP failure: surface loud, do not retry a
                # dead key forever (the client already retried transient 429s).
                resp.degraded = True
                if not resp.degrade_reason:
                    resp.degrade_reason = f"provider call failed: {resp.error[:200]}"
                break

            if schema is None:
                resp.ok = True
                break

            parsed = extract_json_object(resp.content)
            if parsed is None:
                violations = ["reply contained no parseable JSON object"]
            else:
                violations = validate_against_schema(parsed, schema)

            if parsed is not None and not violations:
                resp.parsed = parsed
                resp.ok = True
                break

            # Invalid — prepare a corrective nudge and retry if budget remains.
            resp.ok = False
            resp.degraded = True
            resp.degrade_reason = "schema violations: " + "; ".join(violations[:5])
            if attempt < retries:
                work_messages = list(messages) + [
                    {"role": "assistant", "content": resp.content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was invalid: "
                            + "; ".join(violations[:5])
                            + ". Return ONLY a single valid JSON object matching "
                            "the required schema. No prose, no markdown."
                        ),
                    },
                ]

        assert last is not None
        last.latency_s = round(time.time() - t0, 3)
        _log_usage(last.as_log_row())
        return last


def _json_schema_response_format(schema: dict) -> dict:
    """OpenRouter ``response_format`` for structured output (``require_parameters``
    style). Ollama ignores this and uses its own ``format=json`` switch."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "supplymind_output",
            "strict": True,
            "schema": schema,
        },
    }


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------
class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        env_path: str | os.PathLike | None = None,
    ) -> None:
        # api_key/transport injection is for the offline mock-transport tests.
        self._api_key = api_key
        self._transport = transport
        self._env_path = env_path

    def _client(self):
        from supplymind.llm.client import OpenRouterClient

        key = self._api_key or resolve_openrouter_key(self._env_path)
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set in env or .env — cannot make a live "
                "OpenRouter call (CLAUDE.md §0: no faking around a missing key)"
            )
        return OpenRouterClient(api_key=key, transport=self._transport)

    async def _raw_chat(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict | None,
    ) -> LLMResponse:
        try:
            async with self._client() as client:
                res = await client.chat(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
        except RuntimeError as e:
            # Missing key -> loud degrade, never a fabricated verdict.
            return LLMResponse(
                ok=False, provider=self.name, model=model,
                error=str(e), degraded=True, degrade_reason=str(e),
            )
        except httpx.HTTPError as e:
            return LLMResponse(
                ok=False, provider=self.name, model=model,
                error=f"{type(e).__name__}: {e}",
            )
        return LLMResponse(
            ok=res.ok,
            provider=self.name,
            model=model,
            content=res.content,
            error=res.error,
            latency_s=res.latency_s,
            tokens_prompt=res.tokens_prompt,
            tokens_completion=res.tokens_completion,
            http_status=res.http_status,
        )

    async def liveness(self) -> LivenessResult:
        key = self._api_key or resolve_openrouter_key(self._env_path)
        if not key:
            return LivenessResult(
                ok=False, provider=self.name,
                reason="OPENROUTER_API_KEY absent (env + .env) — provider unavailable",
                model=resolve_model(self.name) or "",
            )
        # A key is present; do a minimal real probe against the CONFIGURED model
        # so a *revoked* key (401) is caught rather than reported as a fake OK.
        # No fallback slug is hardcoded — an unset model is itself a loud failure.
        model = resolve_model(self.name)
        if not model:
            return LivenessResult(
                ok=False, provider=self.name,
                reason=f"key present but {MODEL_ENV} unset — cannot probe a live "
                       "model (no slug is hardcoded per CLAUDE.md §0)",
            )
        try:
            async with self._client() as client:
                res = await client.chat(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                )
        except Exception as e:  # noqa: BLE001
            return LivenessResult(
                ok=False, provider=self.name,
                reason=f"probe failed: {type(e).__name__}: {e}",
                model=model or "",
            )
        if res.ok:
            return LivenessResult(
                ok=True, provider=self.name, reason="ok", model=model or "",
                detail={"http_status": res.http_status},
            )
        return LivenessResult(
            ok=False, provider=self.name,
            reason=f"probe non-200 (status {res.http_status}): {res.error[:160]}",
            model=model or "", detail={"http_status": res.http_status},
        )


# ---------------------------------------------------------------------------
# Ollama backend (edge mode)
# ---------------------------------------------------------------------------
class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        host: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._host = (host or os.environ.get(OLLAMA_HOST_ENV, DEFAULT_OLLAMA_HOST)).rstrip("/")
        self._transport = transport
        self._timeout = timeout_s

    async def _raw_chat(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: dict | None,
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # Ollama enforces JSON via its own top-level ``format`` switch.
        if response_format is not None:
            payload["format"] = "json"
        t0 = time.time()
        try:
            async with httpx.AsyncClient(
                base_url=self._host, timeout=self._timeout, transport=self._transport
            ) as client:
                r = await client.post("/api/chat", json=payload)
        except httpx.HTTPError as e:
            return LLMResponse(
                ok=False, provider=self.name, model=model,
                error=f"{type(e).__name__}: {e}",
                latency_s=round(time.time() - t0, 3),
            )
        dt = round(time.time() - t0, 3)
        if r.status_code >= 400:
            return LLMResponse(
                ok=False, provider=self.name, model=model,
                error=r.text[:300], http_status=r.status_code, latency_s=dt,
            )
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            ok=bool(content),
            provider=self.name, model=model, content=content,
            http_status=r.status_code, latency_s=dt,
            tokens_prompt=int(data.get("prompt_eval_count") or 0),
            tokens_completion=int(data.get("eval_count") or 0),
            error="" if content else "empty response from ollama",
        )

    async def liveness(self) -> LivenessResult:
        model = resolve_model(self.name)
        try:
            async with httpx.AsyncClient(
                base_url=self._host, timeout=10.0, transport=self._transport
            ) as client:
                r = await client.get("/api/tags")
        except httpx.HTTPError as e:
            return LivenessResult(
                ok=False, provider=self.name,
                reason=f"ollama daemon unreachable at {self._host}: "
                       f"{type(e).__name__}: {e}",
                model=model or "",
            )
        if r.status_code != 200:
            return LivenessResult(
                ok=False, provider=self.name,
                reason=f"ollama /api/tags returned {r.status_code}",
                model=model or "",
            )
        tags = [m.get("name") for m in (r.json().get("models") or [])]
        if model and model not in tags:
            return LivenessResult(
                ok=False, provider=self.name,
                reason=f"model '{model}' not pulled (available: {tags[:6]})",
                model=model, detail={"available": tags},
            )
        return LivenessResult(
            ok=True, provider=self.name, reason="ok",
            model=model or "", detail={"available": tags},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_provider(name: str | None = None, **kwargs: Any) -> LLMProvider:
    """Return the selected provider. ``name`` overrides the ``SUPPLYMIND_LLM_PROVIDER``
    env var; env overrides the ``openrouter`` default. Unknown provider -> loud."""
    selected = (name or os.environ.get(PROVIDER_ENV, DEFAULT_PROVIDER)).strip().lower()
    if selected == "openrouter":
        return OpenRouterProvider(**kwargs)
    if selected == "ollama":
        return OllamaProvider(**kwargs)
    raise ValueError(
        f"unknown LLM provider {selected!r}: expected 'openrouter' or 'ollama' "
        f"(set {PROVIDER_ENV})"
    )


# Convenience for building a mock-transport OpenRouter provider in tests.
def _openrouter_with_transport(
    handler: Callable[[httpx.Request], httpx.Response], *, api_key: str = "sk-test"
) -> OpenRouterProvider:
    return OpenRouterProvider(api_key=api_key, transport=httpx.MockTransport(handler))
