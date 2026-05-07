"""coder_action_critic.py — Qwen-2.5-Coder-14B as JSON validator + semantic
critic on war-room recommended_actions output.

Closes the war-room loop: after `_recommend_actions()` produces a list of
typed action dicts, this module passes them to the local Coder model with
a prompt that asks for (a) JSON-schema validity, (b) semantic plausibility
(do the action_type + parameters make sense for the scenario?), (c) cost
sanity (estimated_cost_usd vs estimated_loss_avoided_usd).

Returns per-action critique + overall plan score. Honest about Ollama
unavailability — if Coder model isn't reachable, returns a deterministic
JSON-schema-only check.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = "qwen25-coder-local:latest"


REQUIRED_KEYS = {"action_type", "reason"}
ALLOWED_ACTION_TYPES = {
    "do_nothing", "activate_backup", "reroute_shipment",
    "increase_safety_stock", "expedite_shipment", "hedge_commodity",
    "issue_supplier_alert",
}


def _json_schema_check(actions: list[dict]) -> list[dict]:
    """Pure-Python schema check — no LLM needed. Always runs first."""
    out: list[dict] = []
    for i, a in enumerate(actions):
        errs: list[str] = []
        missing = REQUIRED_KEYS - set(a.keys())
        if missing:
            errs.append(f"missing required keys: {sorted(missing)}")
        atype = a.get("action_type")
        if atype not in ALLOWED_ACTION_TYPES:
            errs.append(f"action_type '{atype}' not in allowed set")
        cost = a.get("estimated_cost_usd")
        save = a.get("estimated_loss_avoided_usd")
        if cost is not None and save is not None:
            try:
                cost = float(cost); save = float(save)
                if cost > 0 and save / cost < 0.5:
                    errs.append(f"poor cost/save ratio {save/cost:.2f}")
            except (TypeError, ValueError):
                errs.append("cost/save not numeric")
        out.append({
            "idx": i,
            "action_type": atype,
            "schema_pass": not errs,
            "schema_errors": errs,
        })
    return out


def _coder_critique(actions: list[dict], scenario_text: str) -> dict | None:
    """Ask Qwen-Coder-14B for a semantic + plausibility critique."""
    try:
        import requests
    except ImportError:
        return None

    prompt = (
        "You are a code reviewer. Review the JSON action plan below for an "
        "AI agent in a supply-chain disruption scenario.\n\n"
        f"Scenario: {scenario_text[:500]}\n\n"
        f"Action plan ({len(actions)} actions):\n"
        f"{json.dumps(actions, indent=2)[:3000]}\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"plan_score_0_to_1": 0.XX, "issues": ["..."], '
        '"missing_action_types": ["..."], "verdict": "<approve|revise|reject>"}\n\n'
        "Plan score considers: schema validity, semantic fit to scenario, "
        "cost/benefit ratio, action diversity (don't recommend 5 hedges)."
    )
    try:
        t0 = time.time()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json", "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        parsed = json.loads(content)
        return {
            "model": MODEL_NAME,
            "plan_score_0_to_1": float(parsed.get("plan_score_0_to_1", 0.5)),
            "issues": parsed.get("issues", [])[:8],
            "missing_action_types": parsed.get("missing_action_types", [])[:5],
            "verdict": parsed.get("verdict", "revise"),
            "latency_s": round(time.time() - t0, 2),
            "ollama_available": True,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[coder-critic] failed: %s", str(e)[:200])
        return None


def critique_action_plan(actions: list[dict],
                          scenario_text: str = "") -> dict:
    """Public entry. Combines fast schema check + optional Coder LLM review.

    Always returns a dict with `schema_results` (per-action) + `coder_review`
    (None if Ollama unreachable).
    """
    schema_results = _json_schema_check(actions)
    schema_pass_rate = (sum(1 for r in schema_results if r["schema_pass"])
                          / max(1, len(schema_results)))

    coder = _coder_critique(actions, scenario_text)

    overall_score = schema_pass_rate
    if coder is not None:
        overall_score = 0.4 * schema_pass_rate + 0.6 * coder["plan_score_0_to_1"]

    return {
        "n_actions_reviewed": len(actions),
        "schema_pass_rate": round(schema_pass_rate, 3),
        "schema_results": schema_results,
        "coder_review": coder,
        "overall_score": round(overall_score, 3),
        "verdict": (coder.get("verdict") if coder
                     else ("approve" if schema_pass_rate >= 0.95 else "revise")),
        "data_source": ("coder_llm + schema" if coder else "schema_only_no_ollama"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_actions = [
        {"action_type": "reroute_shipment", "target": "IN_TRANSIT_TANKERS",
         "parameters": {"via": ["cape_of_good_hope"], "delay_days": 12},
         "reason": "Hormuz CRITICAL; Cape adds 12d but eliminates exposure.",
         "estimated_cost_usd": 2_160_000,
         "estimated_loss_avoided_usd": 1_339_534_884},
        {"action_type": "hedge_commodity", "target": None,
         "parameters": {"commodity": "oil", "hedge_amount_usd": 3_570_000},
         "reason": "Brent projection $132/bbl under analog scenario.",
         "estimated_cost_usd": 214_200,
         "estimated_loss_avoided_usd": 33_660_000},
        {"action_type": "issue_supplier_alert",
         "target": "ALL_TIER1_SUPPLIERS", "parameters": {},
         "reason": "Zero-cost; request continuity plan.",
         "estimated_cost_usd": 0,
         "estimated_loss_avoided_usd": None},
    ]
    res = critique_action_plan(
        test_actions,
        scenario_text="Iran-Israel-US escalation restricts Hormuz",
    )
    print(json.dumps(res, indent=2, default=str))
