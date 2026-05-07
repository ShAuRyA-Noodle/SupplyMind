"""compute_cross_corpus_alpha.py — extend Krippendorff α to v2 library.

Pass-5g computed α on the 26 R4 scenarios. Pass-6 cooked a 1500-event
EMDAT v2 library. This script extends the panel by running 6 frontier
judges on a stratified sample of 30 v2 library events (5 per severity tier
× 4 tiers + 10 random) and reports α stratified by tier + cross-corpus
stability vs the original R4 α.

Cost estimate: 6 models × 30 events ≈ 180 calls × ~$0.0001 = $0.02.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openrouter_client import OpenRouterClient  # noqa: E402

logger = logging.getLogger(__name__)

LIBRARY_JSON = ROOT / "versions/v4_arcadia_live" / "scenarios" / "crisis_library_v2.json"
OUT_RECEIPT = ROOT / "tests" / "receipts" / "cross_corpus_alpha.json"
CACHE_DIR = ROOT / ".openrouter_cache" / "cross_corpus"

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Subset of cheapest reliable judges from pass 5/6 panel
JUDGES = [
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "z-ai/glm-4.5-air:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
]

SYSTEM_PROMPT = (
    "You are a supply-chain risk analyst. Score the disaster scenario severity on the "
    "ordinal 4-tier scale: LOW / MEDIUM / HIGH / CRITICAL. Respond with JSON only, "
    "format: {\"risk_level\": \"<LOW|MEDIUM|HIGH|CRITICAL>\", \"confidence\": 0.0-1.0}."
)


def stratified_sample(events: list[dict], k_per_tier: int = 5,
                       k_random: int = 10, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_tier: dict[str, list[dict]] = {"LOW": [], "MEDIUM": [], "HIGH": [], "CRITICAL": []}
    for e in events:
        t = e.get("severity_tier_emdat", "LOW")
        if t in by_tier:
            by_tier[t].append(e)
    out: list[dict] = []
    for tier, lst in by_tier.items():
        if lst:
            out.extend(rng.sample(lst, min(k_per_tier, len(lst))))
    # Add k_random more random events
    pool = [e for e in events if e not in out]
    if pool:
        out.extend(rng.sample(pool, min(k_random, len(pool))))
    return out


def _extract_risk_level(text: str) -> str | None:
    """Robust JSON extractor + regex fallback for risk_level."""
    # Try JSON first
    m = re.search(r"\{[^}]*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("risk_level", "")).upper().strip()
            if v in RISK_ORDER:
                return v
        except json.JSONDecodeError:
            pass
    # Regex fallback
    up = (text or "").upper()
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", up):
            return level
    return None


def _krippendorff_alpha_ordinal(table: dict[str, dict[str, str]]) -> float:
    """Same impl as scripts/compute_panel_agreement.py."""
    D_o = 0.0; n_o = 0
    value_counts: dict[str, int] = {}
    for sid, judges in table.items():
        valid = [v for v in judges.values() if v in RISK_ORDER]
        for v in valid:
            value_counts[v] = value_counts.get(v, 0) + 1
        for a, b in combinations(valid, 2):
            ia, ib = RISK_ORDER[a], RISK_ORDER[b]
            D_o += (ia - ib) ** 2
            n_o += 1
    values = list(value_counts.keys())
    D_e = 0.0; n_e = 0
    for i, v1 in enumerate(values):
        for v2 in values[i:]:
            n1 = value_counts[v1]; n2 = value_counts[v2]
            pairs = (n1 * (n1 - 1) // 2) if v1 == v2 else (n1 * n2)
            ia, ib = RISK_ORDER[v1], RISK_ORDER[v2]
            D_e += (ia - ib) ** 2 * pairs
            n_e += pairs
    if n_o == 0 or n_e == 0 or D_e == 0:
        return 0.0
    return round(1.0 - (D_o / n_o) / (D_e / n_e), 4)


async def query_judge(client: OpenRouterClient, model: str,
                       scenario: dict) -> str | None:
    """Run one model on one scenario. Returns the predicted tier or None."""
    user_msg = (
        f"Scenario: {scenario.get('title', '?')}. "
        f"Country: {scenario.get('country', '?')}. "
        f"Year: {scenario.get('year', '?')}. "
        f"Disaster type: {scenario.get('disaster_type', '?')} / "
        f"{scenario.get('disaster_subtype', '')}. "
        f"Total deaths: {scenario.get('deaths', 0)}. "
        f"Total damage USD: {scenario.get('damage_usd', 0):,.0f}. "
        f"Total affected: {scenario.get('total_affected', 0)}.\n\n"
        "Output JSON: {\"risk_level\": \"...\", \"confidence\": 0.0-1.0}"
    )
    res = await client.chat(model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ], max_tokens=80, temperature=0.2)
    if not res.ok:
        return None
    return _extract_risk_level(res.content)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load library
    if not LIBRARY_JSON.exists():
        logger.error("library v2 not cooked: %s", LIBRARY_JSON)
        return
    catalog = json.loads(LIBRARY_JSON.read_text(encoding="utf-8"))
    events = catalog.get("events", [])
    logger.info("[cross_corpus] loaded %d v2 library events", len(events))

    # Stratified sample
    sample = stratified_sample(events, k_per_tier=5, k_random=10)
    logger.info("[cross_corpus] sampled %d events for cross-corpus α", len(sample))

    # Query each judge × each event
    table: dict[str, dict[str, str]] = {}
    t0 = time.time()
    async with OpenRouterClient() as client:
        for ev_idx, ev in enumerate(sample):
            scen_id = ev.get("event_id", f"scen_{ev_idx}")
            table[scen_id] = {}
            for model in JUDGES:
                try:
                    pred = await query_judge(client, model, ev)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[cross_corpus] %s/%s failed: %s",
                                    model[:30], scen_id[:20], str(e)[:60])
                    pred = None
                if pred:
                    table[scen_id][model] = pred
            if (ev_idx + 1) % 5 == 0:
                logger.info("[cross_corpus] %d/%d events done, elapsed %.1fs",
                            ev_idx + 1, len(sample), time.time() - t0)
        budget = client.budget_remaining()

    # Compute α on each tier sub-table + overall
    overall_alpha = _krippendorff_alpha_ordinal(table)

    # Per-tier α
    per_tier_alpha: dict[str, float] = {}
    per_tier_count: dict[str, int] = {}
    for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        sub = {sid: judges for sid, judges in table.items()
               if next((e for e in sample if e.get("event_id") == sid), {})
                   .get("severity_tier_emdat") == tier}
        per_tier_count[tier] = len(sub)
        if len(sub) >= 2:
            per_tier_alpha[tier] = _krippendorff_alpha_ordinal(sub)

    # Compare each judge's verdicts to ground-truth tier
    accuracy_per_judge: dict[str, float] = {}
    for model in JUDGES:
        hits = 0; tot = 0
        for sid, judges in table.items():
            v = judges.get(model)
            if not v: continue
            gt = next((e["severity_tier_emdat"] for e in sample
                       if e.get("event_id") == sid), None)
            if gt:
                tot += 1
                if v == gt: hits += 1
        if tot > 0:
            accuracy_per_judge[model.split("/")[-1]] = round(hits / tot, 4)

    # Assemble receipt
    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_events_sampled": len(sample),
        "n_judges": len(JUDGES),
        "judges": JUDGES,
        "ground_truth_source": ("v2 library deterministic severity rule on "
                                 "real EMDAT death/damage/affected counts"),
        "krippendorff_alpha_ordinal": {
            "overall": overall_alpha,
            "per_tier": per_tier_alpha,
            "per_tier_n_events": per_tier_count,
        },
        "accuracy_per_judge_vs_emdat_gt": accuracy_per_judge,
        "elapsed_s": round(time.time() - t0, 2),
        "openrouter_budget": budget,
        "n_calls_attempted": len(sample) * len(JUDGES),
        "n_calls_succeeded": sum(len(j) for j in table.values()),
        "table": table,
        "comparison_to_pass5g_R4_alpha_local_only": 0.2097,
        "comparison_to_pass5g_R4_alpha_frontier_only": 0.5669,
        "inference_type": "cross_corpus_panel_v2_library_stratified",
    }

    OUT_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    logger.info("[cross_corpus] receipt: %s", OUT_RECEIPT)
    print(json.dumps({
        "n_events": receipt["n_events_sampled"],
        "n_judges": receipt["n_judges"],
        "alpha_overall": overall_alpha,
        "alpha_per_tier": per_tier_alpha,
        "accuracy_per_judge": accuracy_per_judge,
        "openrouter_spend_s": receipt["elapsed_s"],
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
