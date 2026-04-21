"""
counterfactual_explainer.py — F3. LLM-driven counterfactual explanations.

Given (state, action_taken, outcome) produce a structured counterfactual:
    - "If you had done NOTHING instead, P50 loss would have been $X."
    - "If you had done the OPPOSITE action, P50 loss would have been $Y."
    - "Nearest historical analog + what that org actually did."

Two modes:
    - llm: call Ollama (Qwen-14B) with a strict JSON schema (preferred)
    - template: deterministic formula using crisis_library analogs (fallback)

Caching: JSON file at `counterfactual_cache.json` keyed by SHA256 of
(state, action) so the same scenario returns instantly on repeat calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent / "counterfactual_cache.json"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


@dataclass
class Counterfactual:
    action_taken: dict
    no_action_delta_usd: float
    opposite_action_delta_usd: float
    rationale: str
    historical_analog: str = ""
    historical_outcome_usd: float = 0.0
    source: str = "template"              # "llm" | "template" | "cache"
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_taken": self.action_taken,
            "no_action_delta_usd": round(self.no_action_delta_usd, 0),
            "opposite_action_delta_usd": round(self.opposite_action_delta_usd, 0),
            "rationale": self.rationale,
            "historical_analog": self.historical_analog,
            "historical_outcome_usd": round(self.historical_outcome_usd, 0),
            "source": self.source,
            "latency_s": round(self.latency_s, 2),
            "meta": self.meta,
        }


def _cache_key(state: dict, action: dict) -> str:
    payload = json.dumps({"state": state, "action": action}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Template fallback — uses crisis library + action-cost heuristics
# ---------------------------------------------------------------------------


def _template_counterfactual(state: dict, action: dict) -> Counterfactual:
    """Deterministic counterfactual using analog + simple cost model.

    Loss model (order-of-magnitude, per v4 pipeline):
        base_exposure_usd = severity * duration_days * 1_500_000
        action_save = estimated_loss_avoided_usd (if provided)
        opposite_cost = same magnitude as action_save but negative
    """
    try:
        from ShAuRyA_Supplymind.realtime.crisis_library import find_analogs
    except Exception:
        find_analogs = None

    severity = float(state.get("severity", 0.3))
    duration = float(state.get("duration_days", 14))
    base = severity * duration * 1_500_000

    action_type = action.get("action_type", "do_nothing")
    # Assumed loss avoided by action type
    SAVE_FACTOR = {
        "do_nothing": 0.0,
        "activate_backup_supplier": 0.35,
        "reroute_shipment": 0.60,
        "increase_safety_stock": 0.25,
        "expedite_order": 0.20,
        "hedge_commodity": 0.40,
        "issue_supplier_alert": 0.05,
    }
    save_frac = SAVE_FACTOR.get(action_type, 0.15)
    no_action_delta = base                     # doing nothing costs `base`
    opposite_frac = 1.0 - save_frac            # opposite: unwinds the saving
    opposite_delta = base * opposite_frac

    # Analog lookup
    analog_name = ""
    analog_outcome = 0.0
    if find_analogs is not None:
        q = state.get("scenario_text") or f"{action_type} severity {severity}"
        try:
            analogs = find_analogs(q, k=1, mode="tfidf")
            if analogs:
                analog_name = f"{analogs[0].name} ({analogs[0].date})"
                rec = analogs[0].full_record
                impact = (rec.get("oil_impact_usd_bbl") or {}).get("peak", 80.0)
                duration_a = rec.get("duration_days", 14)
                analog_outcome = float(impact * 800_000 * duration_a / 30)
        except Exception as e:  # noqa: BLE001
            logger.debug("analog lookup failed: %s", e)

    rationale = (
        f"Template counterfactual (no LLM): action '{action_type}' saves an estimated "
        f"{save_frac * 100:.0f}% of the P50 base exposure of ${base:,.0f} under "
        f"severity {severity:.2f} × {duration:.0f}-day duration. Doing nothing would "
        f"cost ${no_action_delta:,.0f}. An opposite-family action (less mitigating) "
        f"would cost ${opposite_delta:,.0f}."
    )
    return Counterfactual(
        action_taken=action,
        no_action_delta_usd=no_action_delta,
        opposite_action_delta_usd=opposite_delta,
        rationale=rationale,
        historical_analog=analog_name,
        historical_outcome_usd=analog_outcome,
        source="template",
    )


# ---------------------------------------------------------------------------
# LLM-driven counterfactual (Qwen-14B JSON mode)
# ---------------------------------------------------------------------------


LLM_PROMPT = """You are a supply-chain risk counterfactual analyst. Given a state
and an action, produce a STRICT JSON counterfactual explaining what would happen
if the action had NOT been taken or if the OPPOSITE action had been taken.

State: {state}
Action taken: {action}
Template baseline (use as anchor, refine if needed): {template}

Respond with JSON ONLY. Schema:
{{
  "no_action_delta_usd": <float: P50 loss if action had NOT been taken>,
  "opposite_action_delta_usd": <float: P50 loss if opposite action had been taken>,
  "rationale": <string: 2-3 sentences explaining the counterfactual>,
  "historical_analog": <string: 1-line real historical event with outcome>,
  "historical_outcome_usd": <float: real loss of the analog event, 0 if unknown>
}}"""


def _llm_counterfactual(state: dict, action: dict, template: Counterfactual) -> Counterfactual:
    start = time.time()
    prompt = LLM_PROMPT.format(
        state=json.dumps(state, default=str)[:2000],
        action=json.dumps(action, default=str),
        template=json.dumps(template.to_dict(), default=str),
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": "qwen2.5:14b",
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 8192},
            },
            timeout=90,
        )
        r.raise_for_status()
        parsed = json.loads(r.json()["message"]["content"])
        return Counterfactual(
            action_taken=action,
            no_action_delta_usd=float(parsed.get("no_action_delta_usd", template.no_action_delta_usd)),
            opposite_action_delta_usd=float(parsed.get("opposite_action_delta_usd",
                                                       template.opposite_action_delta_usd)),
            rationale=str(parsed.get("rationale", template.rationale)),
            historical_analog=str(parsed.get("historical_analog", template.historical_analog)),
            historical_outcome_usd=float(parsed.get("historical_outcome_usd", 0.0)),
            source="llm",
            latency_s=time.time() - start,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("llm counterfactual failed: %s; falling back to template", e)
        template.source = "template (llm_failed)"
        template.latency_s = time.time() - start
        return template


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def explain_counterfactual(
    state: dict,
    action: dict,
    use_cache: bool = True,
    use_llm: bool = True,
) -> Counterfactual:
    key = _cache_key(state, action)
    cache = _load_cache() if use_cache else {}
    if use_cache and key in cache:
        cached = cache[key]
        return Counterfactual(
            action_taken=cached["action_taken"],
            no_action_delta_usd=cached["no_action_delta_usd"],
            opposite_action_delta_usd=cached["opposite_action_delta_usd"],
            rationale=cached["rationale"],
            historical_analog=cached.get("historical_analog", ""),
            historical_outcome_usd=cached.get("historical_outcome_usd", 0.0),
            source="cache",
            latency_s=0.001,
        )

    template = _template_counterfactual(state, action)
    if use_llm and _ollama_up():
        cf = _llm_counterfactual(state, action, template)
    else:
        cf = template

    if use_cache:
        cache[key] = cf.to_dict()
        _save_cache(cache)
    return cf


# ---------------------------------------------------------------------------
# CLI / batch pre-warm for the demo
# ---------------------------------------------------------------------------


DEMO_SCENARIOS = [
    {
        "name": "hormuz_hedge",
        "state": {"severity": 0.82, "duration_days": 30, "scenario_text": "Iran threatens Hormuz closure"},
        "action": {"action_type": "hedge_commodity", "commodity": "oil", "hedge_amount_usd": 4_200_000},
    },
    {
        "name": "red_sea_reroute",
        "state": {"severity": 0.85, "duration_days": 60, "scenario_text": "Houthi Red Sea attacks ongoing"},
        "action": {"action_type": "reroute_shipment", "via": ["cape_of_good_hope"]},
    },
    {
        "name": "typhoon_backup",
        "state": {"severity": 0.65, "duration_days": 7, "scenario_text": "Typhoon forecast to hit TSMC"},
        "action": {"action_type": "activate_backup_supplier", "backup_supplier_id": "SUP_SAMSUNG"},
    },
    {
        "name": "haifa_reroute",
        "state": {"severity": 0.6, "duration_days": 14, "scenario_text": "Hezbollah rockets hit Haifa port"},
        "action": {"action_type": "reroute_shipment", "via": ["ASHDOD"]},
    },
    {
        "name": "panama_buffer",
        "state": {"severity": 0.45, "duration_days": 90, "scenario_text": "Panama Canal drought low water"},
        "action": {"action_type": "increase_safety_stock", "additional_stock_days": 14},
    },
    {
        "name": "quiet_monitor",
        "state": {"severity": 0.1, "duration_days": 30, "scenario_text": "Routine operations, no disruption"},
        "action": {"action_type": "do_nothing"},
    },
]


def prewarm_cache(use_llm: bool = True) -> dict:
    """Pre-compute counterfactuals for the 6 demo scenarios + write cache."""
    results = {}
    for sc in DEMO_SCENARIOS:
        cf = explain_counterfactual(sc["state"], sc["action"],
                                    use_cache=True, use_llm=use_llm)
        results[sc["name"]] = cf.to_dict()
        logger.info("[prewarm] %s -> source=%s no_action=$%,.0f opposite=$%,.0f",
                    sc["name"], cf.source, cf.no_action_delta_usd, cf.opposite_action_delta_usd)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prewarm", action="store_true", help="Pre-compute all demo scenarios")
    parser.add_argument("--no-llm", action="store_true", help="Force template fallback")
    parser.add_argument("--scenario", type=str, default=None, help="JSON file with state + action")
    args = parser.parse_args()

    if args.prewarm:
        results = prewarm_cache(use_llm=not args.no_llm)
        print(json.dumps(results, indent=2, default=str))
    elif args.scenario:
        payload = json.loads(Path(args.scenario).read_text())
        cf = explain_counterfactual(payload["state"], payload["action"],
                                    use_llm=not args.no_llm)
        print(json.dumps(cf.to_dict(), indent=2, default=str))
    else:
        print("usage: --prewarm | --scenario <file.json>")
