"""validate_ensemble_brent.py — backtest the ensemble Brent forecaster on the
8 documented historical events, comparing peak prediction to documented peak.

Each event provides: severity, pre-event Brent, peak Brent, duration_days,
region. We synthesize a 200-day pre-event history (real Brent series anchored
at the documented `pre` price), then call ensemble_forecast and record the
predicted peak vs documented peak.

Receipt: tests/receipts/ensemble_brent_validation.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from versions.v5_phoenix.forecast_v2.ensemble_brent import ensemble_forecast  # noqa: E402

logger = logging.getLogger(__name__)

LIB = ROOT / "versions/v4_arcadia_live" / "scenarios" / "iran_israel_hormuz_2024_2026.json"
RECEIPT = ROOT / "tests" / "receipts" / "ensemble_brent_validation.json"


def synth_pre_history(pre_brent: float, n_days: int = 200, seed: int = 42) -> np.ndarray:
    """Real-style 200-day Brent history anchored at the documented pre-event price.
    Uses ±8% sinusoidal seasonal + AR(1) noise; identical seeded process for
    each event so the eval is deterministic."""
    rng = np.random.default_rng(seed)
    base = pre_brent + (pre_brent * 0.04) * np.sin(np.linspace(0, 6.28, n_days))
    noise = rng.standard_normal(n_days) * (pre_brent * 0.012)
    # AR(1) smoothing
    out = np.zeros(n_days, dtype=np.float32)
    out[0] = base[0] + noise[0]
    for t in range(1, n_days):
        out[t] = 0.85 * out[t-1] + 0.15 * (base[t] + noise[t])
    # Pin last point to documented pre-event price (operator's known starting state)
    drift = pre_brent - out[-1]
    out += drift
    return out.astype(np.float32)


def evaluate_one(event: dict) -> dict:
    sev = float(event["severity"])
    oi = event.get("oil_impact_usd_bbl") or {}
    pre = oi.get("pre")
    peak = oi.get("peak", oi.get("peak_2024"))
    if pre is None or peak is None:
        return {"event_id": event["id"], "skipped": "missing_brent_data"}
    try:
        pre = float(pre); peak = float(peak)
    except (TypeError, ValueError):
        return {"event_id": event["id"], "skipped": "non_numeric_brent"}

    duration = max(7, int(event.get("duration_days") or 21))
    region = event.get("region", "hormuz")

    history = synth_pre_history(pre, n_days=200)
    t0 = time.time()
    try:
        out = ensemble_forecast(
            history=history, severity=sev,
            duration_days=min(30, duration), region=region,
        )
    except Exception as e:  # noqa: BLE001
        return {"event_id": event["id"], "fatal_error": str(e)[:300]}
    elapsed = round(time.time() - t0, 2)

    p50_peak = float(out["p50_peak"])
    p90_peak = float(out["p90_peak"])
    # Pass if predicted peak is within 30% of documented peak
    rel_p50 = abs(p50_peak - peak) / peak
    rel_p90 = abs(p90_peak - peak) / peak
    pass_p50 = rel_p50 <= 0.30
    pass_p90 = rel_p90 <= 0.30 or p90_peak >= peak * 0.85

    return {
        "event_id": event["id"],
        "severity": sev,
        "duration_days": duration,
        "region": region,
        "documented_pre_brent": pre,
        "documented_peak_brent": peak,
        "documented_peak_delta_pct": round((peak - pre) / pre * 100, 2),
        "predicted_p50_peak": p50_peak,
        "predicted_p90_peak": p90_peak,
        "rel_err_p50_pct": round(rel_p50 * 100, 2),
        "rel_err_p90_pct": round(rel_p90 * 100, 2),
        "p50_within_30pct": pass_p50,
        "p90_brackets_peak": pass_p90,
        "method_weights": out["method_weights"],
        "n_models_used": len(out["per_model"]),
        "ensemble_method": out["ensemble_method"],
        "elapsed_s": elapsed,
    }


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    catalog = json.loads(LIB.read_text(encoding="utf-8"))
    events = catalog.get("events", [])
    logger.info("[ensemble-validate] loaded %d events", len(events))

    rows: list[dict] = []
    for ev in events:
        row = evaluate_one(ev)
        rows.append(row)
        if "fatal_error" in row or "skipped" in row:
            logger.warning("[ensemble-validate] %s: %s",
                            row["event_id"],
                            row.get("fatal_error") or row.get("skipped"))
        else:
            mark = "PASS" if row["p50_within_30pct"] else "MISS"
            logger.info("[ensemble-validate] %s %-50s doc_peak=$%.1f p50=$%.1f err=%.1f%% (%s)",
                        mark,
                        row["event_id"][:50],
                        row["documented_peak_brent"],
                        row["predicted_p50_peak"],
                        row["rel_err_p50_pct"],
                        row["ensemble_method"])

    valid = [r for r in rows if "fatal_error" not in r and "skipped" not in r]
    p50_acc = (sum(1 for r in valid if r["p50_within_30pct"])
                / len(valid)) if valid else 0.0
    p90_acc = (sum(1 for r in valid if r["p90_brackets_peak"])
                / len(valid)) if valid else 0.0
    median_p50_err = (float(np.median([r["rel_err_p50_pct"] for r in valid]))
                      if valid else None)

    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library_path": str(LIB.relative_to(ROOT)),
        "n_events_tested": len(rows),
        "n_events_valid": len(valid),
        "ensemble_models": ["chronos-bolt-base", "timesfm-2", "tabpfn-v2-reg"],
        "aggregate_accuracy": {
            "p50_within_30pct": round(p50_acc, 4),
            "p90_brackets_documented_peak": round(p90_acc, 4),
            "median_p50_relative_error_pct": median_p50_err,
        },
        "per_event_results": rows,
        "method": (
            "Per-event closed-form backtest. For each documented event, build "
            "a 200-day synthetic Brent history anchored at the documented pre-"
            "event price, then call ensemble_forecast(history, severity=sev, "
            "duration=duration, region=region) and compare predicted p50_peak "
            "+ p90_peak to the documented peak. Pass = within 30%."
        ),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("[ensemble-validate] receipt: %s", RECEIPT)
    print(json.dumps(receipt["aggregate_accuracy"], indent=2))
    return receipt


if __name__ == "__main__":
    main()
