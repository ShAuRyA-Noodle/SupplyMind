"""Offline unit tests for the supplymind.llm provider layer (WP6.1).

MOCK TRANSPORT ONLY — zero live network calls, so these run in CI with no key
and no daemon. They exercise the REAL enforcement / retry / logging / degrade
code paths via httpx.MockTransport, and are falsifiable (each asserts a specific
behaviour that breaks if the code regresses).

pytest-asyncio is NOT installed in this repo (asyncio_mode is an unknown config
option), so async coroutines are driven explicitly with asyncio.run().
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from supplymind.llm import analyst as analyst_mod
from supplymind.llm import panel as panel_mod
from supplymind.llm import providers as prov


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Mock-transport handler factories
# ---------------------------------------------------------------------------
def openrouter_reply(content: str, *, status: int = 200,
                     prompt_tokens: int = 11, completion_tokens: int = 7):
    """A single fixed OpenRouter /chat/completions response."""
    def handler(request: httpx.Request) -> httpx.Response:
        if status >= 400:
            return httpx.Response(status, text="mock error body")
        return httpx.Response(status, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens},
        })
    return handler


def openrouter_sequence(contents: list[str]):
    """Return the i-th content on the i-th call (clamped to the last)."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(state["i"], len(contents) - 1)
        state["i"] += 1
        return httpx.Response(200, json={
            "choices": [{"message": {"content": contents[i]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        })
    return handler


VALID_DECISION = json.dumps({
    "decision": "Hedge oil exposure $4.2M.",
    "evidence": ["Brent +12% DoD", "Hormuz closure announced", "bookings halted"],
    "counterfactual": "Unhedged -> P50 COGS impact $80-120M.",
    "precedent": "2019 Abqaiq: Brent +14%.",
    "risk_level": "CRITICAL",
    "confidence": 0.9,
})

VALID_JUDGE = json.dumps({"risk_level": "HIGH", "confidence": 0.8,
                          "reason": "active chokepoint risk"})


def or_provider(handler):
    return prov.OpenRouterProvider(api_key="sk-test",
                                   transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# 1. Provider selection / switch
# ---------------------------------------------------------------------------
def test_get_provider_default_is_openrouter(monkeypatch):
    monkeypatch.delenv(prov.PROVIDER_ENV, raising=False)
    p = prov.get_provider()
    assert isinstance(p, prov.OpenRouterProvider)
    assert p.name == "openrouter"


def test_get_provider_env_switch_to_ollama(monkeypatch):
    monkeypatch.setenv(prov.PROVIDER_ENV, "ollama")
    p = prov.get_provider()
    assert isinstance(p, prov.OllamaProvider)
    assert p.name == "ollama"


def test_get_provider_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(prov.PROVIDER_ENV, "ollama")
    assert isinstance(prov.get_provider("openrouter"), prov.OpenRouterProvider)


def test_get_provider_unknown_is_loud(monkeypatch):
    monkeypatch.delenv(prov.PROVIDER_ENV, raising=False)
    with pytest.raises(ValueError, match="unknown LLM provider"):
        prov.get_provider("gpt5-turbo-max")


# ---------------------------------------------------------------------------
# 2. JSON schema enforcement helpers (pure functions)
# ---------------------------------------------------------------------------
def test_extract_json_bare_fenced_and_prose():
    assert prov.extract_json_object('{"a": 1}') == {"a": 1}
    assert prov.extract_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert prov.extract_json_object('Here is my answer: {"a": 3}. Thanks!') == {"a": 3}
    assert prov.extract_json_object("no json here") is None
    assert prov.extract_json_object("") is None


def test_validate_against_schema_catches_violations():
    schema = analyst_mod.DECISION_SCHEMA
    good = json.loads(VALID_DECISION)
    assert prov.validate_against_schema(good, schema) == []

    bad_enum = dict(good, risk_level="APOCALYPTIC")
    errs = prov.validate_against_schema(bad_enum, schema)
    assert any("enum" in e for e in errs)

    bad_range = dict(good, confidence=1.7)
    errs = prov.validate_against_schema(bad_range, schema)
    assert any("maximum" in e for e in errs)

    missing = {k: v for k, v in good.items() if k != "precedent"}
    errs = prov.validate_against_schema(missing, schema)
    assert any("precedent" in e for e in errs)

    # bool must not satisfy a numeric field
    bad_bool = dict(good, confidence=True)
    errs = prov.validate_against_schema(bad_bool, schema)
    assert any("confidence" in e for e in errs)


# ---------------------------------------------------------------------------
# 3. complete(): happy path with schema
# ---------------------------------------------------------------------------
def test_complete_valid_schema_sets_parsed(monkeypatch, tmp_path):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    p = or_provider(openrouter_reply(VALID_DECISION))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="test/slug", schema=analyst_mod.DECISION_SCHEMA))
    assert resp.ok is True
    assert resp.degraded is False
    assert resp.parsed["risk_level"] == "CRITICAL"
    assert resp.attempts == 1
    assert resp.tokens_prompt == 11 and resp.tokens_completion == 7


def test_complete_no_schema_returns_content():
    p = or_provider(openrouter_reply("free-form text answer"))
    resp = run(p.complete([{"role": "user", "content": "x"}], model="test/slug"))
    assert resp.ok is True
    assert resp.parsed is None
    assert resp.content == "free-form text answer"


# ---------------------------------------------------------------------------
# 4. Retry-on-invalid
# ---------------------------------------------------------------------------
def test_retry_recovers_on_second_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    handler = openrouter_sequence(["totally not json", VALID_DECISION])
    p = or_provider(handler)
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="test/slug", schema=analyst_mod.DECISION_SCHEMA,
                          retries=2))
    assert resp.ok is True
    assert resp.attempts == 2
    assert resp.parsed["confidence"] == 0.9


def test_retry_exhausted_degrades_loud(monkeypatch, tmp_path):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    # every reply violates the schema (wrong enum)
    bad = json.dumps(dict(json.loads(VALID_DECISION), risk_level="MAYBE"))
    p = or_provider(openrouter_reply(bad))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="test/slug", schema=analyst_mod.DECISION_SCHEMA,
                          retries=2))
    assert resp.ok is False
    assert resp.degraded is True
    assert resp.parsed is None
    assert resp.attempts == 3  # 1 + 2 retries
    assert "schema violations" in resp.degrade_reason


# ---------------------------------------------------------------------------
# 5. Usage logging
# ---------------------------------------------------------------------------
def test_usage_log_written(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(log))
    p = or_provider(openrouter_reply(VALID_DECISION))
    run(p.complete([{"role": "user", "content": "x"}],
                   model="test/slug", schema=analyst_mod.DECISION_SCHEMA))
    assert log.exists()
    lines = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    row = lines[0]
    assert row["provider"] == "openrouter"
    assert row["model"] == "test/slug"
    assert row["ok"] is True
    assert "parsed" not in row  # log stays compact
    assert "t" in row


def test_usage_log_records_degraded_no_model(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(log))
    monkeypatch.delenv(prov.MODEL_ENV, raising=False)
    p = or_provider(openrouter_reply(VALID_DECISION))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          schema=analyst_mod.DECISION_SCHEMA))
    assert resp.ok is False and resp.degraded is True
    assert "no model configured" in resp.degrade_reason
    row = json.loads(log.read_text().splitlines()[0])
    assert row["degraded"] is True


# ---------------------------------------------------------------------------
# 6. Degrade-loud on missing key
# ---------------------------------------------------------------------------
def test_liveness_no_key_fails_loud(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    empty_env = tmp_path / "nonexistent.env"
    p = prov.OpenRouterProvider(api_key=None, env_path=str(empty_env))
    live = run(p.liveness())
    assert live.ok is False
    assert "OPENROUTER_API_KEY" in live.reason


def test_liveness_key_present_but_model_unset_is_loud(monkeypatch):
    # key present (injected), but no configured model -> loud, no hardcoded slug,
    # no network probe.
    monkeypatch.delenv(prov.MODEL_ENV, raising=False)
    p = prov.OpenRouterProvider(api_key="sk-test")
    live = run(p.liveness())
    assert live.ok is False
    assert prov.MODEL_ENV in live.reason


def test_complete_no_key_degrades_not_fakes(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    empty_env = tmp_path / "nonexistent.env"
    p = prov.OpenRouterProvider(api_key=None, env_path=str(empty_env))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="test/slug", schema=analyst_mod.DECISION_SCHEMA))
    assert resp.ok is False
    assert resp.degraded is True
    assert resp.parsed is None  # NEVER a fabricated verdict
    assert "OPENROUTER_API_KEY" in resp.degrade_reason


def test_http_error_status_is_surfaced(monkeypatch, tmp_path):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    p = or_provider(openrouter_reply("", status=401))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="test/slug", schema=analyst_mod.DECISION_SCHEMA))
    assert resp.ok is False
    assert resp.degraded is True
    assert resp.http_status == 401
    assert resp.parsed is None


# ---------------------------------------------------------------------------
# 7. Ollama backend via mock transport
# ---------------------------------------------------------------------------
def ollama_chat_reply(content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["format"] == "json"  # schema -> json mode enforced
        return httpx.Response(200, json={
            "message": {"content": content},
            "prompt_eval_count": 20, "eval_count": 12,
        })
    return handler


def test_ollama_complete_parses_json(tmp_path, monkeypatch):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    p = prov.OllamaProvider(host="http://localhost:11434",
                            transport=httpx.MockTransport(ollama_chat_reply(VALID_JUDGE)))
    resp = run(p.complete([{"role": "user", "content": "x"}],
                          model="qwen2.5:7b", schema=panel_mod.JUDGE_SCHEMA))
    assert resp.ok is True
    assert resp.provider == "ollama"
    assert resp.parsed["risk_level"] == "HIGH"
    assert resp.tokens_completion == 12


def test_ollama_liveness_ok_and_down():
    def tags_ok(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})

    up = prov.OllamaProvider(transport=httpx.MockTransport(tags_ok))
    assert run(up.liveness()).ok is True

    def tags_down(request):
        return httpx.Response(503, text="daemon down")
    down = prov.OllamaProvider(transport=httpx.MockTransport(tags_down))
    live = run(down.liveness())
    assert live.ok is False
    assert "503" in live.reason


# ---------------------------------------------------------------------------
# 8. Analyst integration over mock transport
# ---------------------------------------------------------------------------
def test_analyst_assess_returns_valid_decision(tmp_path, monkeypatch):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    a = analyst_mod.Analyst(provider=or_provider(openrouter_reply(VALID_DECISION)),
                            model="test/slug")
    resp = run(a.assess(analyst_mod.HOLDOUT_SCENARIOS[0]["prompt"]))
    assert resp.ok is True
    assert resp.parsed["risk_level"] in analyst_mod.RISK_LEVELS
    assert set(resp.parsed) == set(analyst_mod.DECISION_SCHEMA["required"])


def test_analyst_message_structure_has_8_fewshots():
    msgs = analyst_mod.build_messages("PROBE")
    assert msgs[0]["role"] == "system"
    # 1 system + 8*(user+assistant) + 1 user = 18
    assert len(msgs) == 18
    assert msgs[-1]["content"] == "PROBE"
    assistant_turns = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant_turns) == 8
    for t in assistant_turns:
        assert set(json.loads(t["content"])) == set(analyst_mod.DECISION_SCHEMA["required"])


def test_analyst_structural_report_matches_spec():
    r = analyst_mod.structural_report()
    assert r["few_shot_count"] == 8
    assert r["few_shots_all_valid_json_keys"] is True
    assert r["system_prompt_declares_all_output_keys"] is True
    assert set(r["few_shot_level_coverage"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# 9. Panel scaffold over mock transport
# ---------------------------------------------------------------------------
def test_panel_requires_configured_judges(monkeypatch):
    monkeypatch.delenv(panel_mod.PANEL_MODELS_ENV, raising=False)
    with pytest.raises(ValueError, match="no panel judges configured"):
        panel_mod.JudgePanel()


def test_panel_agreement_stats(tmp_path, monkeypatch):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    # three judges: two agree HIGH, one says CRITICAL; one extra returns garbage
    replies = {
        "j/agree1": json.dumps({"risk_level": "HIGH", "confidence": 0.8, "reason": "a"}),
        "j/agree2": json.dumps({"risk_level": "HIGH", "confidence": 0.7, "reason": "b"}),
        "j/dissent": json.dumps({"risk_level": "CRITICAL", "confidence": 0.9, "reason": "c"}),
        "j/broken": "I cannot produce JSON, sorry.",
    }

    def route(request):
        body = json.loads(request.content)
        model = body["model"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": replies[model]}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4},
        })

    provider = or_provider(route)
    pnl = panel_mod.JudgePanel(models=list(replies), provider=provider)
    res = run(pnl.run("Hormuz closure scenario"))
    assert res.n_ok == 3
    assert res.n_unparseable == 1  # broken judge counted separately, NOT faked
    assert res.consensus_risk == "HIGH"  # median of HIGH,HIGH,CRITICAL
    assert res.krippendorff_alpha_ordinal is not None
    assert 0.0 <= res.mean_confidence <= 1.0


def test_krippendorff_edge_cases():
    assert panel_mod.krippendorff_alpha_ordinal(["HIGH", "HIGH", "HIGH"]) == 1.0
    assert panel_mod.krippendorff_alpha_ordinal(["HIGH"]) is None  # <2 -> undefined
    assert panel_mod.krippendorff_alpha_ordinal([]) is None
    # disagreement -> alpha below perfect
    a = panel_mod.krippendorff_alpha_ordinal(["LOW", "CRITICAL"])
    assert a is not None and a < 1.0


def test_panel_all_fail_reports_none_agreement(tmp_path, monkeypatch):
    monkeypatch.setenv(prov.USAGE_LOG_ENV, str(tmp_path / "usage.jsonl"))
    p = or_provider(openrouter_reply("", status=500))
    pnl = panel_mod.JudgePanel(models=["j/a", "j/b"], provider=p)
    res = run(pnl.run("scenario"))
    assert res.n_ok == 0
    assert res.n_failed == 2
    assert res.consensus_risk is None
    assert res.krippendorff_alpha_ordinal is None  # undefined, not faked
