"""validate_ensemble_brent.py — backtest the ensemble Brent forecaster on the
documented historical events, comparing peak prediction to documented peak.

For each event we slice the REAL FRED DCOILBRENTEU daily series (the actual
observations ending on/before the event date) as the pre-event history, then
call ensemble_forecast and record the predicted peak vs documented peak.

The Brent series is loaded from a local cache or fetched live from FRED
(FRED_API_KEY in .env) and cached. If neither is available the script FAILS
LOUD — it never synthesizes a price history.

Receipt: tests/receipts/ensemble_brent_validation.json
"""
from __future__ import annotations

import json
import logging
import os
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

FRED_SERIES = "DCOILBRENTEU"
BRENT_CACHE = ROOT / "rl" / "data" / "brent_daily_fred_cache.json"


def load_brent_series() -> list[tuple[str, float]]:
    """Return the REAL FRED DCOILBRENTEU daily series as a sorted list of
    (ISO-date, price). Resolution order: local cache -> live FRED fetch (then
    cached). Raises RuntimeError if neither is available — no synthetic data."""
    if BRENT_CACHE.exists():
        try:
            cached = json.loads(BRENT_CACHE.read_text(encoding="utf-8"))
            series = [(o["date"], float(o["value"]))
                      for o in cached.get("observations", [])
                      if o.get("value") not in (None, ".", "")]
            if series:
                return sorted(series)
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            pass  # fall through to a live fetch

    from scripts._env import load_env
    load_env()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            f"No real Brent history: FRED_API_KEY unset and no cache at "
            f"{BRENT_CACHE.relative_to(ROOT)}. Refusing to synthesize prices."
        )
    import httpx
    r = httpx.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"api_key": key, "file_type": "json", "series_id": FRED_SERIES,
                "observation_start": "2010-01-01"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"FRED fetch for {FRED_SERIES} failed: HTTP {r.status_code} "
            f"{r.text[:200]}")
    obs = r.json().get("observations", [])
    series = [(o["date"], float(o["value"])) for o in obs
              if o.get("value") not in (None, ".", "")]
    if not series:
        raise RuntimeError(f"FRED returned no usable {FRED_SERIES} observations")
    series.sort()
    BRENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BRENT_CACHE.write_text(json.dumps({
        "series_id": FRED_SERIES,
        "source": "FRED https://api.stlouisfed.org (DCOILBRENTEU)",
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_observations": len(series),
        "observations": [{"date": d, "value": v} for d, v in series],
    }, indent=2), encoding="utf-8")
    return series


def _pre_event_history(series: list[tuple[str, float]], event_date: str,
                        n_days: int = 200, min_obs: int = 20):
    """Real Brent observations on/before event_date, last n_days. ISO dates
    sort lexically. Returns (np.ndarray, n_obs) or (None, n_obs) if too few."""
    window = [v for d, v in series if event_date and d <= event_date]
    if len(window) < min_obs:
        return None, len(window)
    hist = window[-n_days:]
    return np.asarray(hist, dtype=np.float32), len(hist)


def evaluate_one(event: dict, series: list[tuple[str, float]]) -> dict:
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
    event_date = event.get("date") or event.get("event_date") or ""

    history, n_real = _pre_event_history(series, event_date, n_days=200)
    if history is None:
        return {"event_id": event["id"],
                "skipped": "insufficient_real_brent_history",
                "event_date": event_date, "n_real_obs_before_event": n_real}

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
        "brent_history_source": f"FRED {FRED_SERIES} (real observations)",
        "pre_event_window_end": event_date,
        "n_real_pre_event_obs": n_real,
        "real_history_last3": [round(float(x), 3) for x in history[-3:]],
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
    series = load_brent_series()  # fails loud if no real data
    logger.info("[ensemble-validate] loaded %d events, %d real Brent observations",
                len(events), len(series))

    rows: list[dict] = []
    for ev in events:
        row = evaluate_one(ev, series)
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

    n_skipped = sum(1 for r in rows if "skipped" in r or "fatal_error" in r)
    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library_path": str(LIB.relative_to(ROOT)),
        "brent_series": FRED_SERIES,
        "brent_series_source": "FRED api.stlouisfed.org (real daily observations)",
        "n_brent_observations": len(series),
        "n_events_tested": len(rows),
        "n_events_valid": len(valid),
        "n_events_skipped": n_skipped,
        "ensemble_models": ["chronos-bolt-base", "timesfm-2", "tabpfn-v2-reg"],
        "aggregate_accuracy": {
            "p50_within_30pct": round(p50_acc, 4),
            "p90_brackets_documented_peak": round(p90_acc, 4),
            "median_p50_relative_error_pct": median_p50_err,
        },
        "per_event_results": rows,
        "method": (
            "Per-event backtest on REAL data. For each documented event, slice "
            "the actual FRED DCOILBRENTEU daily observations ending on/before "
            "the event date (last 200) as the pre-event history, then call "
            "ensemble_forecast(history, severity, duration, region) and compare "
            "predicted p50_peak + p90_peak to the documented peak. Pass = "
            "within 30%. Events without enough real pre-event history are "
            "skipped (never synthesized)."
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
