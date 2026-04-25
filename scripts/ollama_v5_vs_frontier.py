"""ollama_v5_vs_frontier.py — head-to-head benchmark of the locally
fine-tuned ``supplymind-analyst:v5`` (Ollama, Qwen-2.5-14B + 8 hard-negative
few-shots) against the 6-model OpenRouter frontier judge panel.

Scope: 15 disaster-severity scenarios = 8 documented Iran/Israel/Hormuz
events + first 7 documented EMDAT events from the v2 1500-event library.

For each scenario every judge predicts a 4-tier risk level
(LOW / MEDIUM / HIGH / CRITICAL). We report:

  * exact_tier_accuracy (per judge, vs ground truth)
  * soft_accuracy_within_1_tier
  * Krippendorff α (ordinal) of each judge against ground-truth
  * mean latency and consensus-with-panel rate

If Ollama is not running we gracefully skip the v5 leg and still produce a
frontier-only benchmark.

Run:    python scripts/ollama_v5_vs_frontier.py
Cost:   <$0.20 (frontier judges are :free; local Ollama is GPU-only).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger("ollama_v5_vs_frontier")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "supplymind-analyst:v5"

FRONTIER_JUDGES = [
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-4.5-air:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
]

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
INV_ORDER = {v: k for k, v in RISK_ORDER.items()}

IRAN_PATH = ROOT / "ShAuRyA_Supplymind/scenarios/iran_israel_hormuz_2024_2026.json"
V2_PATH = ROOT / "ShAuRyA_Supplymind/scenarios/crisis_library_v2.json"
RECEIPT_PATH = ROOT / "tests/receipts/ollama_v5_vs_frontier.json"

SYSTEM_PROMPT = (
    "You are a senior supply-chain risk analyst. The user describes an event. "
    "Score it on the ordinal 4-tier scale LOW/MEDIUM/HIGH/CRITICAL based on "
    "expected disruption to global supply chains. Respond with ONLY a JSON "
    "object."
)

USER_TEMPLATE = (
    "Scenario: {scenario}\n\n"
    'Respond with JSON: {{"risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>", '
    '"confidence": 0.0-1.0, "rationale": "<one sentence>"}}'
)


# ---------------------------------------------------------------------------
# Ground truth + scenario assembly
# ---------------------------------------------------------------------------


def _sev_to_tier(sev: float) -> str:
    if sev is None:
        return "MEDIUM"
    if sev >= 0.85:
        return "CRITICAL"
    if sev >= 0.65:
        return "HIGH"
    if sev >= 0.40:
        return "MEDIUM"
    return "LOW"


def _scenario_text_iran(ev: dict) -> str:
    parts = [
        f"{ev.get('name','(unnamed)')} ({ev.get('date','?')}, "
        f"region={ev.get('region','?')}).",
        ev.get("summary", "")[:600],
    ]
    routes = ev.get("affected_routes") or []
    if routes:
        parts.append(f"Affected routes: {', '.join(routes)}.")
    nodes = ev.get("supply_chain_nodes_affected") or []
    if nodes:
        parts.append(f"Nodes affected: {', '.join(nodes[:5])}.")
    oil = ev.get("oil_impact_usd_bbl") or {}
    if oil:
        parts.append(
            f"Brent: pre={oil.get('pre','?')}, peak={oil.get('peak','?')}, "
            f"7d-post={oil.get('post_7d','?')} USD/bbl."
        )
    return " ".join(p for p in parts if p)


def _scenario_text_v2(ev: dict) -> str:
    return (
        f"Disaster: {ev.get('disaster_type','?')} / "
        f"{ev.get('disaster_subtype','?')}. "
        f"Country: {ev.get('country','?')}. Region: {ev.get('region','?')}. "
        f"Year: {ev.get('year','?')}. Location: {ev.get('location','?')}. "
        f"Magnitude: {ev.get('magnitude','?')}. "
        f"Total deaths: {ev.get('deaths','?')}. "
        f"Damage USD: {ev.get('damage_usd','?')}. "
        f"Total affected: {ev.get('total_affected','?')}."
    )


def load_scenarios() -> list[dict]:
    rows: list[dict] = []
    iran = json.loads(IRAN_PATH.read_text(encoding="utf-8"))
    for ev in iran.get("events", []):
        rows.append({
            "id": ev["id"],
            "source": "iran_israel_hormuz_2024_2026",
            "scenario_text": _scenario_text_iran(ev),
            "ground_truth_tier": _sev_to_tier(ev.get("severity")),
            "severity_raw": ev.get("severity"),
        })
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    take = 0
    for ev in v2.get("events", []):
        tier = ev.get("severity_tier_emdat")
        if tier not in RISK_ORDER:
            continue
        rows.append({
            "id": ev["event_id"],
            "source": "crisis_library_v2",
            "scenario_text": _scenario_text_v2(ev),
            "ground_truth_tier": tier,
            "severity_raw": None,
        })
        take += 1
        if take >= 7:
            break
    return rows


# ---------------------------------------------------------------------------
# Risk parsing + metrics
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_risk(text: str | None) -> str | None:
    if not text:
        return None
    up = str(text).upper().strip()
    if up in RISK_ORDER:
        return up
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", up):
            return level
    return None


def _krippendorff_alpha_ordinal(pairs: list[tuple[str, str]]) -> float:
    """Multi-item Krippendorff α on a 2-rater (judge vs ground-truth) matrix.

    pairs: list of (judge_pred, ground_truth) for the same N items.
    """
    valid = [(a, b) for a, b in pairs if a in RISK_ORDER and b in RISK_ORDER]
    if len(valid) < 2:
        return 0.0
    # Build full set of values (each item has 2 raters)
    values: list[int] = []
    item_idxs: list[list[int]] = []
    for a, b in valid:
        ia, ib = RISK_ORDER[a], RISK_ORDER[b]
        item_idxs.append([ia, ib])
        values.extend([ia, ib])
    # Observed disagreement: average squared distance within each item-pair
    D_o_num = 0.0
    D_o_den = 0
    for ia, ib in item_idxs:
        # only one within-item pair (since 2 raters per item)
        D_o_num += (ia - ib) ** 2
        D_o_den += 1
    D_o = D_o_num / max(1, D_o_den)
    # Expected disagreement: squared distance over the marginal distribution
    pair_sum = 0.0
    pair_n = 0
    for x, y in combinations(values, 2):
        pair_sum += (x - y) ** 2
        pair_n += 1
    if pair_n == 0:
        return 0.0
    D_e = pair_sum / pair_n
    if D_o == 0 and D_e == 0:
        return 1.0
    if D_e == 0:
        return 0.0
    return round(1.0 - (D_o / D_e), 4)


# ---------------------------------------------------------------------------
# Ollama (local) judge
# ---------------------------------------------------------------------------


def _check_ollama() -> tuple[bool, str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            return False, f"http_{r.status_code}"
        tags = [m.get("name") for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in tags:
            return False, f"model_not_pulled: have={tags[:5]}..."
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _ollama_judge(scenario_text: str) -> dict:
    user = USER_TEMPLATE.format(scenario=scenario_text[:1200])
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 120,
                    "num_ctx": 8192,
                },
            },
            timeout=180,
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        obj = _extract_json(content) or {}
        risk = _normalize_risk(obj.get("risk_level"))
        return {
            "ok": risk is not None,
            "risk_level": risk,
            "confidence": float(obj.get("confidence") or 0.5),
            "rationale": str(obj.get("rationale") or "")[:300],
            "latency_s": round(time.time() - t0, 2),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "risk_level": None,
            "error": f"{type(e).__name__}: {e}"[:200],
            "latency_s": round(time.time() - t0, 2),
        }


# ---------------------------------------------------------------------------
# OpenRouter (frontier) judges
# ---------------------------------------------------------------------------


async def _openrouter_one(client: OpenRouterClient, model: str,
                          scenario_text: str) -> dict:
    user = USER_TEMPLATE.format(scenario=scenario_text[:1200])
    t0 = time.time()
    try:
        res = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=120, temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": model,
                "error": f"{type(e).__name__}: {e}"[:200],
                "latency_s": round(time.time() - t0, 2)}
    if not res.ok:
        return {"ok": False, "model": model,
                "error": (res.error or "unknown")[:200],
                "latency_s": round(res.latency_s, 2)}
    obj = _extract_json(res.content) or {}
    risk = _normalize_risk(obj.get("risk_level"))
    return {
        "ok": risk is not None,
        "model": model,
        "risk_level": risk,
        "confidence": float(obj.get("confidence") or 0.5),
        "rationale": str(obj.get("rationale") or "")[:300],
        "latency_s": round(res.latency_s, 2),
        "tokens_prompt": res.tokens_prompt,
        "tokens_completion": res.tokens_completion,
    }


async def _frontier_panel_for_scenario(client: OpenRouterClient,
                                        scenario_text: str) -> dict[str, dict]:
    results = await asyncio.gather(
        *[_openrouter_one(client, m, scenario_text) for m in FRONTIER_JUDGES],
        return_exceptions=False,
    )
    return {r["model"]: r for r in results}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _consensus_panel(predictions: list[str]) -> str:
    valid = [p for p in predictions if p in RISK_ORDER]
    if not valid:
        return "MEDIUM"
    idxs = sorted(RISK_ORDER[p] for p in valid)
    return INV_ORDER[idxs[len(idxs) // 2]]


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scenarios = load_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]
    n = len(scenarios)
    logger.info("loaded %d scenarios (%d iran/israel + %d v2)", n,
                sum(1 for s in scenarios if s["source"].startswith("iran")),
                sum(1 for s in scenarios if s["source"] == "crisis_library_v2"))

    ok_ollama, ollama_status = _check_ollama()
    logger.info("Ollama %s @ %s -> %s", OLLAMA_MODEL, OLLAMA_URL, ollama_status)

    # 1. Ollama (sequential)
    ollama_preds: list[dict] = []
    if ok_ollama:
        for i, sc in enumerate(scenarios, 1):
            logger.info("[ollama %d/%d] %s", i, n, sc["id"])
            ollama_preds.append(_ollama_judge(sc["scenario_text"]))
    else:
        ollama_preds = [{"ok": False, "skipped": True,
                          "reason": ollama_status} for _ in scenarios]

    # 2. OpenRouter (per scenario, parallel within scenario)
    or_preds: list[dict[str, dict]] = []
    async with OpenRouterClient() as client:
        for i, sc in enumerate(scenarios, 1):
            logger.info("[frontier %d/%d] %s", i, n, sc["id"])
            or_preds.append(await _frontier_panel_for_scenario(
                client, sc["scenario_text"]))
        budget_remaining = client.budget_remaining()

    # 3. Per-event records + per-judge metrics
    judges = ([OLLAMA_MODEL] if ok_ollama else []) + FRONTIER_JUDGES
    per_event: list[dict] = []
    per_judge_pairs: dict[str, list[tuple[str, str]]] = {j: [] for j in judges}
    per_judge_latencies: dict[str, list[float]] = {j: [] for j in judges}
    per_judge_succeeded: dict[str, int] = {j: 0 for j in judges}

    for sc, op, op_set in zip(scenarios, ollama_preds, or_preds):
        gt = sc["ground_truth_tier"]
        record: dict[str, Any] = {
            "id": sc["id"],
            "source": sc["source"],
            "ground_truth_tier": gt,
            "predictions": {},
        }
        if ok_ollama:
            risk = op.get("risk_level")
            record["predictions"][OLLAMA_MODEL] = {
                "risk_level": risk,
                "confidence": op.get("confidence"),
                "latency_s": op.get("latency_s"),
                "ok": op.get("ok", False),
                "error": op.get("error"),
            }
            if risk in RISK_ORDER:
                per_judge_pairs[OLLAMA_MODEL].append((risk, gt))
                per_judge_succeeded[OLLAMA_MODEL] += 1
            if op.get("latency_s") is not None:
                per_judge_latencies[OLLAMA_MODEL].append(op["latency_s"])
        for jm in FRONTIER_JUDGES:
            r = op_set.get(jm, {})
            risk = r.get("risk_level")
            record["predictions"][jm] = {
                "risk_level": risk,
                "confidence": r.get("confidence"),
                "latency_s": r.get("latency_s"),
                "ok": r.get("ok", False),
                "error": r.get("error"),
            }
            if risk in RISK_ORDER:
                per_judge_pairs[jm].append((risk, gt))
                per_judge_succeeded[jm] += 1
            if r.get("latency_s") is not None:
                per_judge_latencies[jm].append(r["latency_s"])
        # consensus across panel (excludes the judge being measured)
        per_event.append(record)

    # 4. Compute per-judge metrics
    per_judge_out: dict[str, dict] = {}
    for j in judges:
        pairs = per_judge_pairs[j]
        if not pairs:
            per_judge_out[j] = {
                "exact_tier_accuracy": 0.0,
                "soft_accuracy_within_1_tier": 0.0,
                "n_succeeded": 0,
                "krippendorff_alpha_against_ground_truth": 0.0,
                "mean_latency_s": (
                    round(sum(per_judge_latencies[j])
                          / max(1, len(per_judge_latencies[j])), 2)
                    if per_judge_latencies[j] else 0.0),
                "consensus_with_panel": 0.0,
            }
            continue
        exact = sum(1 for a, b in pairs if a == b) / len(pairs)
        soft = sum(1 for a, b in pairs
                    if abs(RISK_ORDER[a] - RISK_ORDER[b]) <= 1) / len(pairs)
        alpha = _krippendorff_alpha_ordinal(pairs)
        per_judge_out[j] = {
            "exact_tier_accuracy": round(exact, 4),
            "soft_accuracy_within_1_tier": round(soft, 4),
            "n_succeeded": per_judge_succeeded[j],
            "krippendorff_alpha_against_ground_truth": alpha,
            "mean_latency_s": round(
                sum(per_judge_latencies[j]) / len(per_judge_latencies[j]), 2),
            "consensus_with_panel": 0.0,  # filled below
        }

    # 5. Consensus-with-panel: per-item, build panel-ex-self consensus
    for j in judges:
        agree = 0
        denom = 0
        for ev in per_event:
            preds = ev["predictions"]
            self_pred = preds.get(j, {}).get("risk_level")
            if self_pred not in RISK_ORDER:
                continue
            others = [p["risk_level"] for k, p in preds.items()
                      if k != j and p.get("risk_level") in RISK_ORDER]
            if not others:
                continue
            cons = _consensus_panel(others)
            denom += 1
            if cons == self_pred:
                agree += 1
        per_judge_out[j]["consensus_with_panel"] = (
            round(agree / denom, 4) if denom else 0.0)

    # 6. Headline numbers
    v5 = per_judge_out.get(OLLAMA_MODEL)
    v5_acc = v5["exact_tier_accuracy"] if v5 else None
    frontier_accs = [per_judge_out[m]["exact_tier_accuracy"]
                     for m in FRONTIER_JUDGES if m in per_judge_out]
    frontier_mean = round(sum(frontier_accs) / len(frontier_accs), 4) if frontier_accs else 0.0
    v5_beats = bool(v5_acc is not None and v5_acc > frontier_mean)

    receipt = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_scenarios": n,
        "judges": judges,
        "ollama_status": ollama_status,
        "ollama_model": OLLAMA_MODEL,
        "frontier_judges": FRONTIER_JUDGES,
        "per_judge": per_judge_out,
        "headline": {
            "v5_exact_acc": v5_acc,
            "frontier_panel_mean_exact_acc": frontier_mean,
            "v5_beats_frontier": v5_beats,
            "v5_skipped": (not ok_ollama),
            "v5_skip_reason": (None if ok_ollama else ollama_status),
        },
        "openrouter_budget_remaining": budget_remaining,
        "per_event_predictions": per_event,
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info("\nWrote %s", RECEIPT_PATH)
    logger.info("v5_exact_acc                  = %s", v5_acc)
    logger.info("frontier_panel_mean_exact_acc = %s", frontier_mean)
    logger.info("v5_beats_frontier             = %s", v5_beats)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of scenarios (debug)")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
