"""openrouter_war_room_panel.py — 6-judge frontier cross-check for the
Hormuz War Room, on top of the local Ollama / rubric panel.

Same judge subset as scripts/compute_cross_corpus_alpha.py (which already
shipped at α=0.5436 on 30 EMDAT events). Each judge receives the scenario +
top historical analog + a STRUCTURED-JSON prompt asking for risk_level,
confidence, and top-3 most-affected sectors (free-text bag, mapped to our
sector ids by keyword).

Output:
  panel_results: per-judge {risk_level, confidence, top_sectors, latency_s}
  agreement: Krippendorff α (ordinal) on risk_level
  consensus_risk: median risk_level across panel
  cost_usd: tokens × per-token rate (free models = 0)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)

JUDGES = [
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-4.5-air:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
]

# Extended 12-judge frontier panel (used when expand_to_12=True).
# Adds 6 more independent frontier models for tighter Krippendorff α.
JUDGES_12 = JUDGES + [
    "deepseek/deepseek-v3.5:free",
    "qwen/qwen-3-235b-a22b:free",
    "meta-llama/llama-4-405b-instruct:free",
    "mistralai/mistral-large-3-2510:free",
    "x-ai/grok-4-mini:free",
    "anthropic/claude-haiku-4.5:beta",
]

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

SYSTEM_PROMPT = (
    "You are a senior supply-chain risk analyst with 20+ years on Middle East "
    "energy + maritime trade. The user describes a scenario. Score it on the "
    "ordinal 4-tier scale LOW/MEDIUM/HIGH/CRITICAL and identify the 3 most-"
    "affected economic sectors. Respond with ONLY a JSON object."
)

USER_TEMPLATE = (
    "Scenario: {scenario}\n\n"
    "Operator-asserted parameters:\n"
    "  severity (0-1): {severity}\n"
    "  Brent target (USD/bbl): {brent}\n"
    "  duration (days): {duration}\n\n"
    "Top historical analog: {top_analog}\n\n"
    'Respond with JSON: {{"risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>", '
    '"confidence": 0.0-1.0, "top_sectors": ["...", "...", "..."], '
    '"reason": "<one sentence>"}}'
)


def _extract_json_obj(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_risk(text: str) -> str | None:
    if not text:
        return None
    up = text.upper().strip()
    if up in RISK_ORDER:
        return up
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", up):
            return level
    return None


def _krippendorff_alpha_ordinal(values: list[str]) -> float:
    """Single-row Krippendorff α (ordinal) when each judge gives one rating."""
    if len([v for v in values if v in RISK_ORDER]) < 2:
        return 0.0
    valid = [v for v in values if v in RISK_ORDER]
    indices = [RISK_ORDER[v] for v in valid]
    # observed disagreement = mean squared difference of pairs
    pairs = list(combinations(indices, 2))
    if not pairs:
        return 0.0
    D_o = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
    # expected disagreement = mean squared difference of all distinct pairings
    # in the marginal distribution
    counts: dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1
    all_keys = list(counts.keys())
    D_e_num = 0.0
    D_e_den = 0
    for i, k1 in enumerate(all_keys):
        for k2 in all_keys[i:]:
            n1, n2 = counts[k1], counts[k2]
            if k1 == k2:
                npairs = n1 * (n1 - 1) // 2
            else:
                npairs = n1 * n2
            D_e_num += (k1 - k2) ** 2 * npairs
            D_e_den += npairs
    # Perfect agreement: every judge gave the same value -> D_o = 0
    # Krippendorff α is conventionally 1.0 in this case.
    if D_o == 0:
        return 1.0
    if D_e_den == 0 or D_e_num == 0:
        return 0.0
    D_e = D_e_num / D_e_den
    return round(1.0 - (D_o / D_e), 4)


def _consensus(values: list[str]) -> str:
    """Median ordinal consensus."""
    valid = [v for v in values if v in RISK_ORDER]
    if not valid:
        return "MEDIUM"
    idxs = sorted(RISK_ORDER[v] for v in valid)
    median_idx = idxs[len(idxs) // 2]
    inv = {v: k for k, v in RISK_ORDER.items()}
    return inv[median_idx]


async def _query_one(client: OpenRouterClient, model: str,
                       prompt_user: str) -> dict:
    t0 = time.time()
    try:
        res = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
            max_tokens=180, temperature=0.2,
        )
    except Exception as e:  # noqa: BLE001
        return {"model": model, "ok": False, "error": f"{type(e).__name__}: {e}",
                "latency_s": round(time.time() - t0, 2)}
    if not res.ok:
        return {"model": model, "ok": False,
                "error": (res.error or "unknown")[:200],
                "latency_s": round(res.latency_s, 2)}
    obj = _extract_json_obj(res.content) or {}
    risk = _normalize_risk(obj.get("risk_level", ""))
    return {
        "model": model,
        "ok": True,
        "risk_level": risk,
        "confidence": float(obj.get("confidence") or 0.5),
        "top_sectors": obj.get("top_sectors") or [],
        "reason": (obj.get("reason") or "")[:300],
        "latency_s": round(res.latency_s, 2),
        "tokens_prompt": res.tokens_prompt,
        "tokens_completion": res.tokens_completion,
    }


async def run_panel(scenario_text: str, severity: float, brent: float,
                     duration: int, top_analog: str = "(none)",
                     expand_to_12: bool = False) -> dict:
    """Fan out to all judges in parallel; aggregate. Total wall-clock ~5-25s
    depending on which models 429.
    expand_to_12=True uses the 12-judge JUDGES_12 panel (adds DeepSeek, Qwen-3,
    Llama-4, Mistral-3, Grok-4-mini, Claude-Haiku-4.5)."""
    panel = JUDGES_12 if expand_to_12 else JUDGES
    user_prompt = USER_TEMPLATE.format(
        scenario=scenario_text[:600],
        severity=round(severity, 2),
        brent=round(brent, 1),
        duration=duration,
        top_analog=top_analog[:120],
    )
    t0 = time.time()
    async with OpenRouterClient() as client:
        results = await asyncio.gather(
            *[_query_one(client, m, user_prompt) for m in panel],
            return_exceptions=False,
        )
        budget = client.budget_remaining()

    risk_levels = [r["risk_level"] for r in results if r.get("ok") and r.get("risk_level")]
    alpha = _krippendorff_alpha_ordinal(risk_levels)
    consensus = _consensus(risk_levels) if risk_levels else "MEDIUM"
    n_ok = sum(1 for r in results if r.get("ok"))
    mean_conf = (sum(r.get("confidence", 0.0) for r in results if r.get("ok"))
                  / max(1, n_ok))

    return {
        "consensus_risk": consensus,
        "panel_size": len(panel),
        "n_succeeded": n_ok,
        "n_429_or_failed": len(panel) - n_ok,
        "krippendorff_alpha_ordinal": alpha,
        "mean_confidence": round(mean_conf, 4),
        "results": results,
        "budget_remaining": budget,
        "elapsed_s": round(time.time() - t0, 2),
        "judges_used": panel,
    }


def run_panel_sync(scenario_text: str, severity: float, brent: float,
                    duration: int, top_analog: str = "(none)",
                    expand_to_12: bool = False) -> dict:
    """Sync wrapper for FastAPI routes that aren't async."""
    return asyncio.run(run_panel(scenario_text, severity, brent, duration,
                                     top_analog, expand_to_12=expand_to_12))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run_panel_sync(
        scenario_text="Iran-Israel-US escalation forces partial closure of Hormuz",
        severity=0.85, brent=132.0, duration=21,
        top_analog="2024-10 Iran True Promise II",
    )
    print(json.dumps(res, indent=2))
