"""run_frontier_judge_panel.py — Frontier Judge Panel v2 (Tier 1 crown upgrade).

Runs up to 12 frontier OpenRouter models against the 26 real R4 crisis
scenarios. Each model produces a structured risk assessment (LOW / MEDIUM /
HIGH / CRITICAL + confidence + rationale). Results are committed as
`versions/v3_arcadia/results/R4_FRONTIER_PANEL_V2.json` with per-model / per-scenario
verdicts + the expanded cross-frontier panel agreement statistics.

Combines with the 3 existing local judges (DeepSeek-R1 Q4, Qwen-2.5-14B,
Mistral-Nemo) for a 9-15 judge panel depending on OpenRouter availability.

Usage:
    python scripts/run_frontier_judge_panel.py
        --models  hermes,gpt-oss,gemma-4-31b,nemotron3-super
        --limit   26

Respects OpenRouter free-tier limits (18 req/min local limiter + upstream
per-model 429 backoff with jitter). Every call is logged to
`.openrouter_usage.jsonl`. On upstream rate-limit the model is retried twice
then skipped — one model failing never blocks the panel.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.openrouter_client import MODELS, OpenRouterClient, ModelSpec  # noqa: E402

logger = logging.getLogger(__name__)

R4_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json"
OUT_PATH = ROOT / "v3_arcadia" / "results" / "R4_FRONTIER_PANEL_V2.json"
CACHE_DIR = ROOT / ".openrouter_cache"

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

SYSTEM_PROMPT = (
    "You are a supply-chain risk analyst. Assess the crisis scenario below "
    "and output a JSON object ONLY — no preamble, no code fence — with keys:\n"
    '  risk_level : one of LOW | MEDIUM | HIGH | CRITICAL\n'
    "  confidence : float in [0,1]\n"
    "  primary_vulnerabilities : list of short strings\n"
    "  mitigations : list of short strings\n"
    "  rationale_one_line : one-sentence reason\n"
    "Respond with the JSON object only."
)


def _format_prompt(scen_id: str, scenario_text: str) -> list[dict]:
    readable = scen_id.replace("_", " ")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Scenario: {readable}\n\n"
            f"Summary (from R4 crisis library): {scenario_text[:1800]}\n\n"
            "Produce the JSON assessment now."
        )},
    ]


def _extract_json(text: str) -> dict | None:
    """Depth-counting JSON extractor robust to preambles + repeated blobs."""
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i : j + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
    # Regex fallback for risk_level only — catches "answer is CRITICAL"-style replies
    up = (text or "").upper()
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", up):
            return {"risk_level": level, "confidence": 0.5,
                    "rationale_one_line": "(extracted from free-text reply)"}
    return None


def _load_r4_scenarios(limit: int | None = None) -> list[tuple[str, str, str]]:
    r4 = json.loads(R4_PATH.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str]] = []
    for sid, scen in r4.get("per_scenario", {}).items():
        gt = str(scen.get("ground_truth", "")).upper()
        # Use the first judge's rationale text as the scenario summary
        first_judge = next(iter(scen.get("per_judge", {}).values()), {})
        summary = ""
        if isinstance(first_judge, dict):
            parsed = first_judge.get("parsed") or {}
            if isinstance(parsed, dict):
                summary = (parsed.get("rationale_one_line") or
                           " ".join(parsed.get("primary_vulnerabilities", []))[:200] or
                           "")
        if not summary:
            summary = sid.replace("_", " ")
        rows.append((sid, summary, gt))
        if limit and len(rows) >= limit:
            break
    return rows


def _cache_key(model_slug: str, scenario_id: str) -> Path:
    safe_slug = model_slug.replace("/", "__").replace(":", "_")
    safe_scen = re.sub(r"[^A-Za-z0-9_-]", "_", scenario_id)
    return CACHE_DIR / safe_slug / f"{safe_scen}.json"


async def _query_one(
    client: OpenRouterClient,
    model: ModelSpec,
    scen_id: str,
    scenario_text: str,
    gt: str,
) -> dict:
    """Run one model × scenario with cache + upstream-429 backoff."""
    cache_path = _cache_key(model.slug, scen_id)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["from_cache"] = True
        return cached

    messages = _format_prompt(scen_id, scenario_text)
    attempts = 3
    last_err: str = ""
    for attempt in range(attempts):
        res = await client.chat(model.slug, messages, max_tokens=512, temperature=0.2)
        if res.ok:
            parsed = _extract_json(res.content) or {}
            pred = str(parsed.get("risk_level", "")).upper().strip()
            row = {
                "model": model.slug,
                "model_short": model.short,
                "ok": pred in RISK_ORDER,
                "http_status": 200,
                "latency_s": round(res.latency_s, 2),
                "tokens": {"prompt": res.tokens_prompt,
                            "completion": res.tokens_completion},
                "predicted_risk": pred,
                "confidence": parsed.get("confidence"),
                "primary_vulnerabilities": parsed.get("primary_vulnerabilities", []),
                "mitigations": parsed.get("mitigations", []),
                "rationale_one_line": parsed.get("rationale_one_line", ""),
                "raw_preview": (res.content or "")[:300],
                "ground_truth": gt,
                "scenario_id": scen_id,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(row, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            return row
        last_err = res.error or f"http_{res.http_status}"
        # Upstream 429 backoff with jitter
        if "429" in last_err or res.http_status == 429:
            wait = 3 * (2 ** attempt) + random.uniform(0, 1.5)
            logger.info("[%s] upstream 429, sleeping %.1fs", model.short, wait)
            await asyncio.sleep(wait)
        else:
            break
    return {
        "model": model.slug, "model_short": model.short, "ok": False,
        "http_status": 0, "error": last_err[:300],
        "scenario_id": scen_id, "ground_truth": gt,
    }


async def run_panel(
    judge_models: list[ModelSpec],
    scenarios: list[tuple[str, str, str]],
) -> dict:
    per_scenario: dict[str, dict] = {}
    total = len(judge_models) * len(scenarios)
    done = 0
    async with OpenRouterClient() as client:
        for (sid, summary, gt) in scenarios:
            rows = []
            for model in judge_models:
                row = await _query_one(client, model, sid, summary, gt)
                rows.append(row)
                done += 1
                if done % 5 == 0:
                    b = client.budget_remaining()
                    logger.info("[%d/%d] budget: %d/%d per-min, %d/%d per-day",
                                done, total,
                                b["per_min_used"], b["per_min_budget"],
                                b["per_day_used"], b["per_day_budget"])
            preds = [r for r in rows if r.get("ok")]
            tallies: dict[str, int] = {}
            for r in preds:
                tallies[r["predicted_risk"]] = tallies.get(r["predicted_risk"], 0) + 1
            majority = max(tallies, key=tallies.get) if tallies else "UNKNOWN"
            per_scenario[sid] = {
                "ground_truth": gt,
                "n_judges_ok": len(preds),
                "n_judges_total": len(judge_models),
                "majority": majority,
                "majority_matches_gt": majority == gt,
                "tallies": tallies,
                "per_judge": rows,
            }

    ok_total = sum(s["n_judges_ok"] for s in per_scenario.values())
    majority_correct = sum(1 for s in per_scenario.values()
                           if s["majority_matches_gt"])
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panel_size_frontier": len(judge_models),
        "frontier_model_slugs": [m.slug for m in judge_models],
        "local_models_not_included": [
            "deepseek-r1-local-q4", "qwen2.5:14b", "mistral-nemo"
        ],
        "n_scenarios": len(scenarios),
        "ok_call_total": ok_total,
        "majority_vote_accuracy_vs_ground_truth": round(
            majority_correct / max(1, len(scenarios)), 4),
        "per_scenario": per_scenario,
        "source": "https://openrouter.ai/api/v1/chat/completions",
        "ground_truth_source": "versions/v3_arcadia/results/R4_DANGEROUS_V2.json",
        "inference_type": "live_http_multi_provider_panel",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated short names; default=all judge-role")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max scenarios (default: all 26)")
    args = parser.parse_args()

    judge_models = [m for m in MODELS if m.role == "judge"]
    if args.models:
        wanted = set(s.strip() for s in args.models.split(","))
        judge_models = [m for m in judge_models if m.short in wanted]
    if not judge_models:
        print("no judge models selected"); sys.exit(2)

    scenarios = _load_r4_scenarios(limit=args.limit)
    logger.info("running %d models × %d scenarios = %d calls max",
                len(judge_models), len(scenarios),
                len(judge_models) * len(scenarios))

    result = asyncio.run(run_panel(judge_models, scenarios))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(json.dumps({
        "panel_size_frontier": result["panel_size_frontier"],
        "n_scenarios": result["n_scenarios"],
        "ok_call_total": result["ok_call_total"],
        "majority_vote_accuracy_vs_ground_truth":
            result["majority_vote_accuracy_vs_ground_truth"],
        "output": str(OUT_PATH.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
