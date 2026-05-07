"""
analyst_ab_bench.py — G9 fix benchmark. Compares supplymind-analyst v5 vs base
Qwen-2.5-14B-Instruct on 10 fixed scenarios, judged by a deterministic rubric.

This is the A/B harness that the original v3 version lost 12% on.

Judge is DETERMINISTIC rubric (not another LLM) — makes the result reproducible
and independent of judge-LLM noise. Every scenario has a correct risk_level
anchor + a list of required evidence keywords.

Usage:
    # Ensure both models are built via:
    #   ollama create supplymind-analyst:v5 -f versions/v4_arcadia_live/features/Modelfile.analyst_v5
    # Then run:
    python -m versions.v4_arcadia_live.features.analyst_ab_bench --save

If Ollama is down, the benchmark skips gracefully and returns a synthetic stub
so CI doesn't break.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OUTPUT_PATH = Path(__file__).resolve().parent / "R9_ANALYST_AB_V5.json"


@dataclass
class Scenario:
    id: str
    prompt: str
    correct_risk: str                        # one of LOW|MEDIUM|HIGH|CRITICAL
    required_evidence: list[str]             # lower-case substrings required in rationale
    category: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        id="hormuz_2026_04",
        prompt=("STATE: Iran threatened full closure of Strait of Hormuz. "
                "Brent crude $123/bbl DoD +3.5%. Carriers pause Persian Gulf bookings. "
                "Health 72/100. What is the supply-chain risk level and what action?"),
        correct_risk="CRITICAL",
        required_evidence=["hormuz", "brent", "123"],
        category="kinetic_conflict",
    ),
    Scenario(
        id="routine_q3_report",
        prompt=("STATE: Q3 internal supplier report arrived on time. All operational. "
                "No active disruption signals. Health 95/100. What is the risk level?"),
        correct_risk="LOW",
        required_evidence=["no active", "routine"],
        category="baseline",
    ),
    Scenario(
        id="typhoon_72h_warning",
        prompt=("STATE: NOAA warns Category 3 typhoon tracking toward Kaohsiung, 72h ETA. "
                "TSMC in projected path. Samsung backup qualified. What action?"),
        correct_risk="HIGH",
        required_evidence=["tsmc", "backup", "typhoon"],
        category="weather",
    ),
    Scenario(
        id="minor_fx_move",
        prompt=("STATE: Turkish Lira -2.5% overnight. No supplier/shipping issues. "
                "Health 93/100. What is the risk level?"),
        correct_risk="LOW",
        required_evidence=["fx", "no operational", "2.5"],
        category="fx_noise",
    ),
    Scenario(
        id="red_sea_campaign",
        prompt=("STATE: Houthi Red Sea attacks ongoing 60+ days. 100+ vessel attacks. "
                "Maersk + MSC rerouting via Cape of Good Hope. Brent +7% WoW. "
                "What action for in-transit vessels?"),
        correct_risk="CRITICAL",
        required_evidence=["cape", "houthi", "red sea"],
        category="route_closure",
    ),
    Scenario(
        id="small_earthquake",
        prompt=("STATE: M4.8 earthquake Japan Pacific logged. No tsunami advisory. "
                "No damage reports. All suppliers operational. What action?"),
        correct_risk="LOW",
        required_evidence=["m4", "no damage", "no tsunami"],
        category="benign_event",
    ),
    Scenario(
        id="panama_drought",
        prompt=("STATE: Panama Canal water levels -25%. Transit slots -30%. Shanghai-East US "
                "freight +18% WoW. What is the risk and action?"),
        correct_risk="MEDIUM",
        required_evidence=["panama", "slot", "18"],
        category="route_capacity",
    ),
    Scenario(
        id="chinese_sanctions_rumor",
        prompt=("STATE: Unconfirmed rumor on social media that China may impose rare-earth "
                "export controls in Q4. No official announcement. Stock prices stable. "
                "What action?"),
        correct_risk="MEDIUM",
        required_evidence=["rumor", "unconfirmed", "rare-earth"],
        category="unverified_signal",
    ),
    Scenario(
        id="iran_israel_missile",
        prompt=("STATE: Iran launched 180 ballistic missiles at Israel (2024-10-01 "
                "True Promise II analog). Haifa port intermittent closures. Lloyd's "
                "war-risk premium +50bp East Med. What action for Haifa shipments?"),
        correct_risk="HIGH",
        required_evidence=["haifa", "reroute", "missile"],
        category="kinetic_conflict",
    ),
    Scenario(
        id="quiet_day",
        prompt=("STATE: Day 4 of 30. All suppliers operational. No disruption signals. "
                "Brent +0.2% DoD. Health 96/100. What action?"),
        correct_risk="LOW",
        required_evidence=["no disruption", "do nothing", "monitor"],
        category="baseline",
    ),
]


@dataclass
class AnalystResult:
    model: str
    scenarios: list[dict] = field(default_factory=list)
    exact_risk_match: int = 0
    one_off_risk_match: int = 0             # off by one level, partial credit
    evidence_coverage_sum: float = 0.0       # sum across scenarios
    parse_rate: int = 0                      # JSON parse success count
    total_latency_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.scenarios)

    def to_dict(self) -> dict:
        n = max(self.n, 1)
        return {
            "model": self.model,
            "n": self.n,
            "exact_risk_acc": round(self.exact_risk_match / n, 3),
            "partial_risk_acc": round((self.exact_risk_match + 0.5 * self.one_off_risk_match) / n, 3),
            "evidence_coverage_mean": round(self.evidence_coverage_sum / n, 3),
            "parse_rate": round(self.parse_rate / n, 3),
            "total_latency_s": round(self.total_latency_s, 2),
            "scenarios": self.scenarios,
        }


def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _list_models() -> set[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        return {m["name"] for m in r.get("models", [])}
    except Exception:
        return set()


def _call_model(model: str, prompt: str) -> tuple[dict, float]:
    """Return (parsed_json, latency_s). Empty dict on failure."""
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.15, "num_ctx": 16384},
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["message"]["content"]
        parsed = json.loads(text)
        return parsed, time.time() - start
    except Exception as e:  # noqa: BLE001
        logger.warning("model=%s failed: %s", model, e)
        return {}, time.time() - start


LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _off_by_one(predicted: str, correct: str) -> bool:
    try:
        return abs(LEVEL_ORDER.index(predicted) - LEVEL_ORDER.index(correct)) == 1
    except ValueError:
        return False


def _evidence_coverage(rationale_text: str, required: list[str]) -> float:
    low = (rationale_text or "").lower()
    hits = sum(1 for k in required if k in low)
    return hits / max(1, len(required))


def _score_response(resp: dict, sc: Scenario) -> dict:
    if not resp:
        return {"parsed": False, "exact": 0, "one_off": 0, "ev_coverage": 0.0}
    predicted = str(resp.get("risk_level", "")).upper()
    exact = int(predicted == sc.correct_risk)
    one_off = 0 if exact else int(_off_by_one(predicted, sc.correct_risk))
    # Rationale = evidence list + decision + counterfactual
    rationale = " ".join([
        " ".join(resp.get("evidence", []) if isinstance(resp.get("evidence"), list) else []),
        str(resp.get("decision", "")),
        str(resp.get("counterfactual", "")),
    ])
    ev = _evidence_coverage(rationale, sc.required_evidence)
    return {"parsed": True, "exact": exact, "one_off": one_off, "ev_coverage": ev}


def benchmark(v5_model: str, base_model: str) -> dict:
    if not _ollama_up():
        return {"status": "ollama_down", "note": "start Ollama + build v5 model"}
    tags = _list_models()
    if v5_model not in tags:
        return {"status": "v5_not_built",
                "hint": f"ollama create {v5_model} -f versions/v4_arcadia_live/features/Modelfile.analyst_v5",
                "available_models": sorted(tags),
                "note": "run the ollama create command, then re-run this benchmark."}

    results: dict[str, AnalystResult] = {
        v5_model: AnalystResult(model=v5_model),
        base_model: AnalystResult(model=base_model),
    }

    for sc in SCENARIOS:
        for m in (v5_model, base_model):
            parsed, lat = _call_model(m, sc.prompt)
            score = _score_response(parsed, sc)
            ar = results[m]
            ar.scenarios.append({
                "id": sc.id, "correct_risk": sc.correct_risk,
                "predicted_risk": parsed.get("risk_level", "") if parsed else "",
                "exact": score["exact"], "one_off": score["one_off"],
                "evidence_coverage": score["ev_coverage"],
                "parsed": score["parsed"],
                "latency_s": round(lat, 2),
                "response": parsed,
            })
            ar.exact_risk_match += score["exact"]
            ar.one_off_risk_match += score["one_off"]
            ar.evidence_coverage_sum += score["ev_coverage"]
            ar.parse_rate += int(score["parsed"])
            ar.total_latency_s += lat

    v5_d = results[v5_model].to_dict()
    base_d = results[base_model].to_dict()
    summary = {
        "v5_exact_acc": v5_d["exact_risk_acc"],
        "base_exact_acc": base_d["exact_risk_acc"],
        "exact_acc_lift": round(v5_d["exact_risk_acc"] - base_d["exact_risk_acc"], 3),
        "v5_partial_acc": v5_d["partial_risk_acc"],
        "base_partial_acc": base_d["partial_risk_acc"],
        "partial_acc_lift": round(v5_d["partial_risk_acc"] - base_d["partial_risk_acc"], 3),
        "v5_evidence_mean": v5_d["evidence_coverage_mean"],
        "base_evidence_mean": base_d["evidence_coverage_mean"],
    }
    return {
        "status": "ok",
        "v5": v5_d,
        "base": base_d,
        "summary": summary,
        "note": ("v5 target: exact_acc_lift > 0 AND evidence_coverage_mean > base. "
                 "Historical v3 A/B win rate was only 12% vs base Qwen (per "
                 "docs/legacy/AUTORESEARCH_SUMMARY.md + docs/v3/EXECUTIVE_SUMMARY.md §supplymind-analyst)."),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", default="supplymind-analyst:v5")
    parser.add_argument("--base", default="qwen2.5:14b")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    result = benchmark(args.v5, args.base)
    print(json.dumps(result, indent=2))

    if args.save:
        OUTPUT_PATH.write_text(json.dumps(result, indent=2))
        print(f"saved to {OUTPUT_PATH}")
