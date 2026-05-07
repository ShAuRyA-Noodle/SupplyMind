"""freeze_cache.py — build an offline replay cache for the Hormuz demo.

Two modes:

    1. --from-crisis-library (default, offline-safe)
        Pulls the 8 canonical 2024-2026 Iran/Israel/Hormuz events from
        versions/v4_arcadia_live/scenarios/iran_israel_hormuz_2024_2026.json and
        synthesizes a plausible /live/hormuz-closure response for each.
        This cache works even if all live APIs are down.

    2. --from-live-ingestor (requires NewsAPI/FRED/GDELT keys)
        Calls versions.v4_arcadia_live.realtime.ingestor with --once and captures
        the output. Produces the most realistic cache but only works today
        when the APIs answer.

Output: versions/v5_phoenix/realtime_v5/replay_cache_<timestamp>.json
       + symlink/copy as replay_cache_latest.json

Use in demos:
    curl -X POST http://localhost:8000/live/hormuz-closure?replay=1 ...
    # or: set env FORCE_REPLAY=1 before starting the server
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LIVE_CRISES = ROOT / "versions/v4_arcadia_live" / "scenarios" / "iran_israel_hormuz_2024_2026.json"
OUT_DIR = Path(__file__).resolve().parent


def _severity_to_level(sev: float) -> str:
    if sev >= 0.85: return "CRITICAL"
    if sev >= 0.65: return "HIGH"
    if sev >= 0.35: return "MEDIUM"
    return "LOW"


def _severity_to_escalation(sev: float) -> str:
    if sev >= 0.85: return "C_SUITE_IMMEDIATE"
    if sev >= 0.70: return "C_SUITE_REVIEW"
    if sev >= 0.55: return "OPS_DIRECTOR_4H"
    if sev >= 0.35: return "OPS_DIRECTOR_24H"
    return "FYI_DASHBOARD"


def _actions_for(event: dict) -> list[str]:
    affected = set(event.get("affected_routes", []))
    sev = float(event.get("severity", 0.5))
    actions = []
    if "strait_of_hormuz" in affected:
        actions.append("Hedge Brent crude exposure +30% via Q3 futures")
        actions.append("Activate Iraq alt-oil backup corridor (7d lead time)")
    if "red_sea" in affected or "suez_canal" in affected:
        actions.append("Reroute 60% of Asia-Europe TEU via Cape of Good Hope (+12d)")
        actions.append("Pre-book 4 wk of air-freight capacity for tier-1 SKUs")
    if "semiconductor" in (event.get("supply_chain_nodes_affected", []) or ""):
        actions.append("Pull forward 8 wk of TSMC N5 wafer orders")
    if sev >= 0.7:
        actions.append("Alert C-suite + legal for potential insurance claim filing")
    if sev >= 0.85:
        actions.append("Trigger dual-source contingency plan (budget authority $25M)")
    while len(actions) < 5:
        actions.append("Maintain real-time situational awareness; re-assess in 24h")
    return actions[:5]


def _counterfactual_for(event: dict) -> dict:
    sev = float(event.get("severity", 0.5))
    base_no_action = 400_000_000 * sev  # $0 to $400M depending on severity
    mitigation_savings = 0.80 if sev >= 0.85 else 0.60 if sev >= 0.7 else 0.40
    with_plan = base_no_action * (1.0 - mitigation_savings)
    return {
        "no_action_loss_usd": int(round(base_no_action)),
        "with_plan_loss_usd": int(round(with_plan)),
        "savings_usd": int(round(base_no_action - with_plan)),
        "savings_pct": round(100.0 * mitigation_savings, 1),
    }


def build_from_crisis_library() -> dict:
    if not LIVE_CRISES.exists():
        raise FileNotFoundError(f"crisis library missing: {LIVE_CRISES}")
    blob = json.loads(LIVE_CRISES.read_text())
    cache: dict[str, dict] = {}
    for event in blob.get("events", []):
        sev = float(event.get("severity", 0.5))
        synth = {
            "scenario_input": {
                "scenario_text": f"{event['name']}. {event.get('summary', '')}",
                "region": event.get("region", "hormuz"),
            },
            "top_analog": {
                "id": event["id"],
                "name": event["name"],
                "similarity": 0.99,                       # exact match for cached events
                "date": event.get("date"),
                "duration_days": event.get("duration_days"),
            },
            "risk_level": _severity_to_level(sev),
            "confidence": round(sev, 2),
            "recommended_actions": _actions_for(event),
            "escalation_tier": _severity_to_escalation(sev),
            "counterfactual": _counterfactual_for(event),
            "oil_impact_usd_bbl": event.get("oil_impact_usd_bbl"),
            "judges": {
                "qwen25_14b": {"risk_level": _severity_to_level(sev), "confidence": round(sev, 2)},
                "mistral_nemo": {"risk_level": _severity_to_level(sev), "confidence": round(sev - 0.05, 2)},
                "deepseek_r1": {"risk_level": _severity_to_level(sev - 0.1), "confidence": round(sev - 0.15, 2)},
            },
            "judges_agreement": "2_of_3_CRITICAL" if sev >= 0.85 else "2_of_3_HIGH",
            "replay_source": "crisis_library_v1",
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        cache[event["id"]] = synth
    return {
        "schema_version": "1.0",
        "source": "versions/v4_arcadia_live/scenarios/iran_israel_hormuz_2024_2026.json",
        "build_mode": "offline_from_crisis_library",
        "n_events": len(cache),
        "events": cache,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_from_live_ingestor() -> dict:
    """Shell out to v4 ingestor and capture its responses."""
    import subprocess
    logger.info("[freeze] running live ingestor -- requires NEWSAPI_KEY/FRED_API_KEY in env")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "versions.v4_arcadia_live.realtime.ingestor", "--once",
             "--skip", "marinetraffic", "--json-out"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            logger.error("[freeze] ingestor failed rc=%d stderr=%s", proc.returncode, proc.stderr[:500])
            raise RuntimeError("live ingestor failed")
        live_payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except Exception as e:  # noqa: BLE001
        logger.error("[freeze] live ingestor path errored: %s", e)
        raise
    return {
        "schema_version": "1.0",
        "source": "versions.v4_arcadia_live.realtime.ingestor --once",
        "build_mode": "live_api_capture",
        "events": live_payload.get("events", []),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-crisis-library", action="store_true", default=True,
                        help="Offline-safe path; synthesizes from iran_israel_hormuz_2024_2026.json")
    parser.add_argument("--from-live-ingestor", action="store_true",
                        help="Requires API keys in .env; captures fresh live responses")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.from_live_ingestor:
        cache = build_from_live_ingestor()
    else:
        cache = build_from_crisis_library()

    out = args.out or OUT_DIR / f"replay_cache_{time.strftime('%Y_%m_%d')}.json"
    out.write_text(json.dumps(cache, indent=2))

    latest = OUT_DIR / "replay_cache_latest.json"
    latest.write_text(json.dumps(cache, indent=2))

    logger.info("[freeze] wrote %s (%d events)", out, len(cache.get("events", [])))
    logger.info("[freeze] wrote %s (pointer)", latest)
    print(f"[freeze] n_events={len(cache.get('events', {}))} mode={cache['build_mode']}")


if __name__ == "__main__":
    main()
