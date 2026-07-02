"""Pass 27 U20 — Auto-extract scenario params from news headlines (OpenRouter live).

Closes HONEST_LIMITATIONS L11 (war-room scenario params operator-asserted).

Pipeline: 5 historical news headlines -> OpenRouter gpt-4o-mini -> JSON
{severity, brent_price_usd, duration_days} -> compare to ground-truth params
captured in war_room_validation receipt.

Uses live OpenRouter key from .env. Records sha256 of every response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = ROOT / "FINAL_SUBMIT" / "receipts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._env import load_env  # noqa: E402
load_env()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write(name: str, payload: dict) -> tuple[Path, str]:
    payload["_pass"] = 27
    payload["_generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = RECEIPTS / name
    raw = json.dumps(payload, indent=2, default=str).encode()
    out.write_bytes(raw)
    return out, _sha(raw)


# Ground-truth scenario params from documented historical events
HEADLINES = [
    {
        "id": "suez_2021",
        "headline": "EVERGREEN container ship runs aground blocking Suez Canal in both directions",
        "ground_truth": {"severity": 0.9, "brent_price_usd": 64, "duration_days": 6},
        "context": "March 2021 Suez blockage, 6 days, ~$9.6B/day disruption",
    },
    {
        "id": "houthi_red_sea_2024",
        "headline": "Houthi forces strike commercial vessels in Red Sea, major shipping lines reroute around Africa",
        "ground_truth": {"severity": 0.7, "brent_price_usd": 78, "duration_days": 90},
        "context": "Late 2023 - 2024 Red Sea Houthi attacks, ongoing rerouting",
    },
    {
        "id": "tohoku_2011",
        "headline": "9.0 magnitude earthquake and tsunami hits Tohoku Japan, Fukushima nuclear plant damaged",
        "ground_truth": {"severity": 1.0, "brent_price_usd": 110, "duration_days": 60},
        "context": "March 2011 Tohoku earthquake, 60-day acute phase",
    },
    {
        "id": "thailand_floods_2011",
        "headline": "Severe monsoon flooding inundates 7 industrial parks in Ayutthaya Thailand, hard drive supply collapses",
        "ground_truth": {"severity": 0.6, "brent_price_usd": 110, "duration_days": 45},
        "context": "Q4 2011 Thai floods, semiconductor + HDD supply hit",
    },
    {
        "id": "iran_sanctions_2018",
        "headline": "US re-imposes sanctions on Iran oil exports, secondary sanctions threatened against buyers",
        "ground_truth": {"severity": 0.5, "brent_price_usd": 75, "duration_days": 180},
        "context": "Nov 2018 Iran nuclear deal withdrawal, sanctions effective",
    },
]


def call_openrouter(headline: str) -> dict:
    """Call OpenRouter with strict JSON schema prompt."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx_missing"}

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return {"error": "no_OPENROUTER_API_KEY"}

    prompt = (
        "You are a supply-chain disruption analyst. Read this news headline and extract "
        "three numerical parameters in STRICT JSON format. Do NOT include any text outside the JSON.\n\n"
        f"HEADLINE: {headline}\n\n"
        "Output JSON with exactly these keys:\n"
        '  - severity (float, 0.0 = minor / 1.0 = catastrophic)\n'
        '  - brent_price_usd (int, expected Brent crude USD/barrel during disruption)\n'
        '  - duration_days (int, expected disruption duration in days)\n\n'
        'Output ONLY the JSON object, nothing else.'
    )

    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://supplymind.dev"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "SupplyMind"),
    }

    try:
        t0 = time.time()
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                       json=body, headers=headers, timeout=30)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"error": f"http_{r.status_code}", "body": r.text[:300]}
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        # Extract JSON from response
        m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if not m:
            return {"error": "no_json_in_response", "raw": content[:300]}
        try:
            extracted = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"error": f"json_decode: {e}", "raw": content[:300]}
        return {
            "ok": True,
            "extracted": extracted,
            "elapsed_s": round(elapsed, 3),
            "response_sha256": _sha(r.content),
            "tokens_used": data.get("usage", {}),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def evaluate(extracted: dict, ground: dict) -> dict:
    """Numeric error metrics."""
    out = {}
    for key in ["severity", "brent_price_usd", "duration_days"]:
        gt = ground.get(key)
        ex = extracted.get(key)
        if ex is None or gt is None:
            out[key] = {"error": "missing"}
            continue
        try:
            ex = float(ex)
            gt = float(gt)
            abs_err = abs(ex - gt)
            rel_err = abs_err / max(abs(gt), 0.01) * 100
            out[key] = {
                "extracted": ex,
                "ground_truth": gt,
                "abs_err": round(abs_err, 3),
                "rel_err_pct": round(rel_err, 2),
                "within_25pct": rel_err <= 25.0,
            }
        except (TypeError, ValueError) as e:
            out[key] = {"error": f"type_error: {e}"}
    return out


def main():
    print("=" * 78)
    print("PASS 27 U20 — Scenario auto-extract from news headlines (OpenRouter live)")
    print("=" * 78)

    results = []
    for h in HEADLINES:
        print(f"\n[{h['id']}] {h['headline'][:60]}...")
        api_resp = call_openrouter(h["headline"])
        if not api_resp.get("ok"):
            print(f"  FAIL: {api_resp.get('error', 'unknown')}")
            results.append({
                "id": h["id"],
                "headline": h["headline"],
                "ground_truth": h["ground_truth"],
                "api_response": api_resp,
                "skipped": True,
            })
            continue
        extracted = api_resp["extracted"]
        evals = evaluate(extracted, h["ground_truth"])
        n_within_25 = sum(1 for v in evals.values() if v.get("within_25pct"))
        print(f"  extracted: {extracted}")
        print(f"  within_25%: {n_within_25}/3")
        results.append({
            "id": h["id"],
            "headline": h["headline"],
            "ground_truth": h["ground_truth"],
            "extracted": extracted,
            "field_evaluation": evals,
            "n_fields_within_25pct": n_within_25,
            "elapsed_s": api_resp.get("elapsed_s"),
            "response_sha256": api_resp.get("response_sha256"),
        })

    # Aggregate
    n_total = len(results) * 3
    n_correct = sum(r.get("n_fields_within_25pct", 0) for r in results)
    n_skipped = sum(1 for r in results if r.get("skipped"))

    payload = {
        "name": "scenario_auto_extract_v1",
        "closes_honest_limitation": "L11 (war-room scenario params operator-asserted)",
        "model": "openai/gpt-4o-mini via OpenRouter (live)",
        "n_headlines_tested": len(results),
        "n_skipped_api_error": n_skipped,
        "field_accuracy_within_25pct": f"{n_correct}/{n_total - n_skipped*3}",
        "field_accuracy_pct": round(n_correct / max(n_total - n_skipped*3, 1) * 100, 1),
        "results": results,
    }
    out, sha = _write("pass27_U20_scenario_extractor.json", payload)
    print(f"\nReceipt: {out.name}  sha={sha[:24]}")
    print(f"Field accuracy within 25%: {n_correct}/{n_total - n_skipped*3} = {payload['field_accuracy_pct']}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
