"""validate_ensemble_brent.py — REAL walk-forward backtest of the Chronos +
TimesFM + TabPFN Brent ensemble on the actual FRED DCOILBRENTEU daily series.

This is R10 of the rebuild backlog. The old version built a 200-day SYNTHETIC
sinusoid "Real-style" history and scored the ensemble against hand-authored
"documented peak" numbers — a fake. This version is fully real:

  * History and ground truth both come from the REAL FRED daily Brent series
    (external_data/fred_brent_daily.csv, ~9.9k real observations 1987-2026).
  * For each documented crisis event (real dates), we slice the real prices
    ending on/before the event date as the pre-event context, forecast the
    next H trading days, and compare the forecast to the REAL realized prices
    over that same window (the actual observations that followed).
  * We report MAPE and within-+/-10% hit-rate for the ENSEMBLE and for EACH
    member (chronos / timesfm / tabpfn) — a genuine "ensemble vs members"
    comparison over real history.
  * TabPFN is trained leave-one-out (the event under test is dropped from its
    training set) so it cannot memorise its own documented outcome.

If no real Brent series is available the script FAILS LOUD — it never
synthesizes a price history.

Receipt: tests/receipts/ensemble_brent_REAL.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplymind.phoenix.forecast_v2.ensemble_brent import ensemble_forecast  # noqa: E402

logger = logging.getLogger(__name__)

LIB = ROOT / "supplymind" / "warroom" / "scenarios" / "iran_israel_hormuz_2024_2026.json"
RECEIPT = ROOT / "tests" / "receipts" / "ensemble_brent_REAL.json"

FRED_SERIES = "DCOILBRENTEU"
BRENT_CSV = ROOT / "external_data" / "fred_brent_daily.csv"
BRENT_CACHE = ROOT / "rl" / "data" / "brent_daily_fred_cache.json"

HORIZON = 30            # forecast/evaluation horizon in trading days
CONTEXT_DAYS = 200      # pre-event context window
WITHIN_PCT = 0.10       # +/-10% hit-rate band


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_brent_series() -> tuple[list[tuple[str, float]], dict]:
    """Return the REAL FRED DCOILBRENTEU daily series as a sorted list of
    (ISO-date, price) plus a provenance dict.

    Resolution order:
      1. external_data/fred_brent_daily.csv (the sanctioned real series on disk)
      2. local JSON cache
      3. live FRED fetch (only if FRED_API_KEY is set) -> cached

    Raises RuntimeError if none is available. Never synthesizes prices."""
    if BRENT_CSV.exists():
        series: list[tuple[str, float]] = []
        with open(BRENT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # header e.g. ['observation_date', 'DCOILBRENTEU']
            for row in reader:
                if len(row) < 2:
                    continue
                d, v = row[0].strip(), row[1].strip()
                if not d or v in (".", "", "None"):
                    continue
                try:
                    series.append((d, float(v)))
                except ValueError:
                    continue
        if series:
            series.sort()
            prov = {
                "source": "external_data/fred_brent_daily.csv (FRED DCOILBRENTEU, real daily observations)",
                "resolution": "local_csv",
                "csv_sha256": _sha256(BRENT_CSV),
                "n_observations": len(series),
                "date_range": [series[0][0], series[-1][0]],
                "header": header,
            }
            return series, prov

    if BRENT_CACHE.exists():
        try:
            cached = json.loads(BRENT_CACHE.read_text(encoding="utf-8"))
            series = [(o["date"], float(o["value"]))
                      for o in cached.get("observations", [])
                      if o.get("value") not in (None, ".", "")]
            if series:
                series.sort()
                return series, {
                    "source": "rl/data/brent_daily_fred_cache.json (FRED DCOILBRENTEU cache)",
                    "resolution": "local_cache",
                    "n_observations": len(series),
                    "date_range": [series[0][0], series[-1][0]],
                }
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            pass

    from scripts._env import load_env
    load_env()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            f"No real Brent history: {BRENT_CSV.relative_to(ROOT)} missing, no "
            f"cache at {BRENT_CACHE.relative_to(ROOT)}, and FRED_API_KEY unset. "
            f"Refusing to synthesize prices."
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
    return series, {
        "source": "FRED api.stlouisfed.org live fetch (DCOILBRENTEU)",
        "resolution": "live_fred",
        "n_observations": len(series),
        "date_range": [series[0][0], series[-1][0]],
    }


def _split_history_actuals(series: list[tuple[str, float]], event_date: str):
    """Split the real series at ``event_date`` into (context history, realized
    actuals). ISO dates sort lexically.

    Returns (hist_prices, hist_dates, actual_prices, actual_dates)."""
    before = [(d, v) for d, v in series if event_date and d <= event_date]
    after = [(d, v) for d, v in series if event_date and d > event_date]
    hist = before[-CONTEXT_DAYS:]
    actuals = after[:HORIZON]
    return (
        np.asarray([v for _, v in hist], dtype=np.float32),
        [d for d, _ in hist],
        np.asarray([v for _, v in actuals], dtype=np.float32),
        [d for d, _ in actuals],
    )


def _mape(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute percentage error over the overlap length."""
    n = min(len(forecast), len(actual))
    if n == 0:
        return float("nan")
    f = np.asarray(forecast[:n], dtype=np.float64)
    a = np.asarray(actual[:n], dtype=np.float64)
    return float(np.mean(np.abs(f - a) / np.abs(a)) * 100.0)


def _within_pct(forecast: np.ndarray, actual: np.ndarray, band: float) -> float:
    n = min(len(forecast), len(actual))
    if n == 0:
        return float("nan")
    f = np.asarray(forecast[:n], dtype=np.float64)
    a = np.asarray(actual[:n], dtype=np.float64)
    return float(np.mean(np.abs(f - a) / np.abs(a) <= band))


def evaluate_one(event: dict, series: list[tuple[str, float]]) -> dict:
    sev = float(event.get("severity", 0.5))
    region = event.get("region", "hormuz")
    event_id = event.get("id", "unknown")
    event_date = event.get("date") or event.get("event_date") or ""

    hist, hist_dates, actuals, actual_dates = _split_history_actuals(
        series, event_date)
    if hist.size < 30:
        return {"event_id": event_id, "skipped": "insufficient_context_history",
                "event_date": event_date, "n_context_obs": int(hist.size)}
    if actuals.size < 5:
        return {"event_id": event_id,
                "skipped": "insufficient_realized_actuals",
                "event_date": event_date, "n_actuals": int(actuals.size)}

    t0 = time.time()
    try:
        out = ensemble_forecast(
            history=hist, severity=sev, duration_days=HORIZON, region=region,
            exclude_event_id=event_id,   # leave-one-out: no self-leakage
        )
    except Exception as e:  # noqa: BLE001
        return {"event_id": event_id, "fatal_error": str(e)[:300]}
    elapsed = round(time.time() - t0, 2)

    eval_len = min(len(out["p50"]), len(actuals))
    ens_p50 = np.asarray(out["p50"][:eval_len], dtype=np.float64)
    act = actuals[:eval_len]

    # Per-member metrics (real forecast vs real realized prices)
    per_method: dict[str, dict] = {}
    ens_metrics = {
        "mape_pct": round(_mape(ens_p50, act), 3),
        "within_10pct": round(_within_pct(ens_p50, act, WITHIN_PCT), 3),
    }
    per_method["ensemble"] = ens_metrics
    for m, mo in out["per_model"].items():
        fp50 = np.asarray(mo["p50"][:eval_len], dtype=np.float64)
        per_method[m] = {
            "mape_pct": round(_mape(fp50, act), 3),
            "within_10pct": round(_within_pct(fp50, act, WITHIN_PCT), 3),
        }

    realized_peak = float(np.max(act))
    ens_pred_peak = float(np.max(ens_p50))

    return {
        "event_id": event_id,
        "event_date": event_date,
        "severity": sev,
        "region": region,
        "n_context_obs": int(hist.size),
        "context_last_date": hist_dates[-1] if hist_dates else None,
        "context_last_price": round(float(hist[-1]), 3),
        "eval_horizon_days": eval_len,
        "actual_window": [actual_dates[0], actual_dates[eval_len - 1]] if eval_len else [],
        "realized_prices_first3": [round(float(x), 3) for x in act[:3]],
        "realized_peak": round(realized_peak, 3),
        "ensemble_pred_peak": round(ens_pred_peak, 3),
        "peak_rel_err_pct": round(abs(ens_pred_peak - realized_peak) / realized_peak * 100, 3),
        "metrics_vs_realized": per_method,
        "method_weights": out["method_weights"],
        "n_models_used": len(out["per_model"]),
        "ensemble_method": out["ensemble_method"],
        "elapsed_s": elapsed,
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _aggregate(valid: list[dict]) -> dict:
    """Aggregate MAPE + within-10% across events, per method."""
    methods = ["ensemble", "chronos", "timesfm", "tabpfn"]
    agg: dict[str, dict] = {}
    for meth in methods:
        mapes = [r["metrics_vs_realized"][meth]["mape_pct"]
                 for r in valid if meth in r["metrics_vs_realized"]]
        withins = [r["metrics_vs_realized"][meth]["within_10pct"]
                   for r in valid if meth in r["metrics_vs_realized"]]
        if not mapes:
            continue
        agg[meth] = {
            "n_events": len(mapes),
            "mean_mape_pct": round(float(np.mean(mapes)), 3),
            "median_mape_pct": round(float(np.median(mapes)), 3),
            "mean_within_10pct": round(float(np.mean(withins)), 3),
        }
    return agg


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    catalog = json.loads(LIB.read_text(encoding="utf-8"))
    events = catalog.get("events", [])
    series, provenance = load_brent_series()  # fails loud if no real data
    logger.info("[ensemble-validate] %d events, %d REAL Brent obs (%s..%s) via %s",
                len(events), len(series), series[0][0], series[-1][0],
                provenance["resolution"])

    rows: list[dict] = []
    for ev in events:
        row = evaluate_one(ev, series)
        rows.append(row)
        if "fatal_error" in row or "skipped" in row:
            logger.warning("[ensemble-validate] %-45s SKIP/ERR: %s",
                            row["event_id"][:45],
                            row.get("fatal_error") or row.get("skipped"))
        else:
            em = row["metrics_vs_realized"]["ensemble"]
            logger.info(
                "[ensemble-validate] %-45s ens MAPE=%.2f%% within10=%.0f%% "
                "(chronos=%.2f timesfm=%.2f tabpfn=%.2f) H=%dd",
                row["event_id"][:45], em["mape_pct"], em["within_10pct"] * 100,
                row["metrics_vs_realized"].get("chronos", {}).get("mape_pct", float("nan")),
                row["metrics_vs_realized"].get("timesfm", {}).get("mape_pct", float("nan")),
                row["metrics_vs_realized"].get("tabpfn", {}).get("mape_pct", float("nan")),
                row["eval_horizon_days"])

    valid = [r for r in rows if "fatal_error" not in r and "skipped" not in r]
    n_skipped = sum(1 for r in rows if "skipped" in r or "fatal_error" in r)
    agg = _aggregate(valid)

    receipt = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "library_path": str(LIB.relative_to(ROOT)).replace("\\", "/"),
        "brent_series": FRED_SERIES,
        "brent_provenance": provenance,
        "forecast_horizon_days": HORIZON,
        "context_window_days": CONTEXT_DAYS,
        "within_band_pct": WITHIN_PCT * 100,
        "ensemble_models": ["chronos-bolt-base", "timesfm-2", "tabpfn-v2-reg"],
        "n_events_tested": len(rows),
        "n_events_valid": len(valid),
        "n_events_skipped": n_skipped,
        "aggregate_metrics_vs_realized": agg,
        "per_event_results": rows,
        "method": (
            "REAL walk-forward backtest. For each documented crisis event (real "
            "date), slice the actual FRED DCOILBRENTEU daily observations ending "
            "on/before the event date (last %d) as pre-event context, forecast "
            "the next %d trading days with the Chronos+TimesFM+TabPFN ensemble, "
            "and compare BOTH the ensemble and each individual member to the REAL "
            "realized Brent prices over that window (the observations that "
            "actually followed). Metrics: MAPE (mean abs %% error) and within-"
            "+/-%d%% hit-rate. TabPFN is trained leave-one-out so the event under "
            "test never trains on its own documented outcome (no label leakage). "
            "History and ground truth are both real FRED data; nothing is "
            "synthesized."
        ) % (CONTEXT_DAYS, HORIZON, int(WITHIN_PCT * 100)),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("[ensemble-validate] receipt: %s", RECEIPT)
    print(json.dumps({
        "n_events_valid": len(valid),
        "n_events_skipped": n_skipped,
        "aggregate_metrics_vs_realized": agg,
    }, indent=2))
    return receipt


if __name__ == "__main__":
    main()
