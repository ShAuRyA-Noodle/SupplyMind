"""Pass 28 K1-K3 — LIVE real-data ingest with new keys (no Colab needed).

K1 FRED Brent backfill 8 events (closes L9)
K2 NewsAPI live ingest 5 queries (closes G4)
K3 NOAA CDO live (closes M typhoon-response)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
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
    payload["_pass"] = 28
    payload["_generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = RECEIPTS / name
    raw = json.dumps(payload, indent=2, default=str).encode()
    out.write_bytes(raw)
    return out, _sha(raw)


# ---------------------------------------------------------------------------
# K1 — FRED DCOILBRENTEU 200d pre-event slices for 8 events
# ---------------------------------------------------------------------------
def k1_fred_brent_real() -> dict:
    import httpx
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return {"skipped": "no FRED_API_KEY"}

    EVENTS = [
        ("iran_sanctions_2018",    "2018-05-08", 75.43),
        ("israel_hamas_2023",      "2023-10-07", 88.50),
        ("hormuz_tanker_2019",     "2019-06-13", 61.31),
        ("houthi_red_sea_2023",    "2023-11-19", 81.30),
        ("suez_2021",              "2021-03-23", 64.41),
        ("taiwan_tension_2022",    "2022-08-02", 100.54),
        ("thailand_floods_2011",   "2011-10-15", 110.20),
        ("tohoku_2011",            "2011-03-11", 113.84),
    ]

    fred_results = []
    for ev_id, ev_date, anchor in EVENTS:
        end = datetime.strptime(ev_date, "%Y-%m-%d")
        start = end - timedelta(days=300)
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "api_key": key, "file_type": "json",
            "series_id": "DCOILBRENTEU",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        t0 = time.time()
        r = httpx.get(url, params=params, timeout=30)
        elapsed = time.time() - t0
        if r.status_code != 200:
            fred_results.append({"event_id": ev_id, "error": f"http_{r.status_code}",
                                  "body": r.text[:200]})
            continue
        d = r.json()
        obs = [(o["date"], float(o["value"])) for o in d.get("observations", [])
               if o["value"] not in (".", None)]
        if not obs:
            fred_results.append({"event_id": ev_id, "error": "no_observations"})
            continue
        prices = [o[1] for o in obs]
        fred_results.append({
            "event_id": ev_id, "event_date": ev_date,
            "anchor_value_published_usd": anchor,
            "n_pre_event_obs": len(obs),
            "pre_event_brent_first_3": obs[:3],
            "pre_event_brent_last_3": obs[-3:],
            "pre_event_brent_mean": round(sum(prices) / len(prices), 4),
            "pre_event_brent_min": round(min(prices), 4),
            "pre_event_brent_max": round(max(prices), 4),
            "rel_err_anchor_vs_mean_pct": round(abs(anchor - sum(prices)/len(prices)) / anchor * 100, 2),
            "elapsed_s": round(elapsed, 3),
            "sha256_response_first_2k": _sha(r.content[:2048]),
        })

    n_with_data = sum(1 for r in fred_results if "n_pre_event_obs" in r)
    return {
        "name": "K1_fred_brent_real_backfill",
        "closes": "HONEST_LIMITATIONS L9 (synthetic Brent pre-history)",
        "series": "DCOILBRENTEU (real FRED)",
        "n_events_total": len(EVENTS),
        "n_events_with_real_data": n_with_data,
        "events": fred_results,
        "no_synthetic_substitution": True,
    }


# ---------------------------------------------------------------------------
# K2 — NewsAPI live ingest 5 queries
# ---------------------------------------------------------------------------
def k2_newsapi_ingest() -> dict:
    import httpx
    key = os.environ.get("NEWS_API_KEY")
    if not key:
        return {"skipped": "no NEWS_API_KEY"}

    QUERIES = [
        "Strait of Hormuz",
        "Suez Canal",
        "Red Sea Houthi",
        "Iran sanctions oil",
        "Taiwan strait shipping",
    ]
    news_results = []
    for q in QUERIES:
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "apiKey": key, "sortBy": "publishedAt",
                  "pageSize": 10, "language": "en"}
        t0 = time.time()
        r = httpx.get(url, params=params, timeout=20)
        elapsed = time.time() - t0
        if r.status_code != 200:
            news_results.append({"query": q, "error": f"http_{r.status_code}",
                                  "body": r.text[:200]})
            continue
        d = r.json()
        articles = d.get("articles", [])
        news_results.append({
            "query": q,
            "totalResults": d.get("totalResults", 0),
            "n_articles_returned": len(articles),
            "top_3_titles": [a.get("title", "")[:140] for a in articles[:3]],
            "top_3_publishedAt": [a.get("publishedAt") for a in articles[:3]],
            "top_3_sources": [a.get("source", {}).get("name") for a in articles[:3]],
            "elapsed_s": round(elapsed, 3),
            "sha256_response_first_2k": _sha(r.content[:2048]),
        })

    n_ok = sum(1 for r in news_results if "totalResults" in r)
    return {
        "name": "K2_newsapi_live_ingest_5queries",
        "closes": "G4 (NewsAPI substitute path)",
        "queries": news_results,
        "n_queries_successful": n_ok,
    }


# ---------------------------------------------------------------------------
# K3 — NOAA CDO live datasets endpoint
# ---------------------------------------------------------------------------
def k3_noaa_cdo() -> dict:
    import httpx
    token = os.environ.get("NOAA_TOKEN")
    if not token:
        return {"skipped": "no NOAA_TOKEN"}

    headers = {"token": token}
    out = {"name": "K3_noaa_cdo_live", "closes": "M typhoon-response real data",
            "endpoints": []}

    for ep_name, ep_path in [
        ("datasets", "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"),
        ("locationcategories", "https://www.ncdc.noaa.gov/cdo-web/api/v2/locationcategories"),
        ("datatypes", "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes?limit=10"),
    ]:
        t0 = time.time()
        r = httpx.get(ep_path, headers=headers, timeout=20)
        elapsed = time.time() - t0
        ep_result = {
            "endpoint": ep_name, "url": ep_path,
            "status_code": r.status_code, "elapsed_s": round(elapsed, 3),
        }
        if r.status_code == 200:
            d = r.json()
            ep_result["n_results"] = len(d.get("results", []))
            ep_result["sample_results"] = d.get("results", [])[:3]
            ep_result["sha256_response_first_2k"] = _sha(r.content[:2048])
        else:
            ep_result["body"] = r.text[:200]
        out["endpoints"].append(ep_result)

    out["n_endpoints_200_OK"] = sum(1 for ep in out["endpoints"] if ep.get("status_code") == 200)
    return out


def main():
    print("=" * 78)
    print("PASS 28 K1-K3 LIVE INGEST -- 3 new keys, no Colab needed")
    print("=" * 78)

    blocks = [
        ("K1", "fred_brent_real", k1_fred_brent_real, "pass28_K1_fred_brent_real.json"),
        ("K2", "newsapi_live_ingest", k2_newsapi_ingest, "pass28_K2_newsapi_live_ingest.json"),
        ("K3", "noaa_cdo_live", k3_noaa_cdo, "pass28_K3_noaa_cdo_live.json"),
    ]

    summary = {"pass": 28, "tier": "K1-K3 keys ingest", "blocks": []}
    for letter, name, fn, receipt_name in blocks:
        print(f"\n[{letter}] {name}...")
        t0 = time.time()
        try:
            payload = fn()
            elapsed = round(time.time() - t0, 2)
            payload["_block_id"] = letter
            payload["_block_name"] = name
            payload["_block_elapsed_s"] = elapsed
            out, sha = _write(receipt_name, payload)
            print(f"  [OK] {receipt_name}  sha={sha[:16]}...  elapsed={elapsed}s")
            summary["blocks"].append({
                "id": letter, "name": name, "receipt": receipt_name,
                "sha256_24": sha[:24], "elapsed_s": elapsed,
            })
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}  elapsed={elapsed}s")
            import traceback
            traceback.print_exc()
            summary["blocks"].append({
                "id": letter, "name": name,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed_s": elapsed,
            })

    out, sha = _write("pass28_keys_ingest_master_summary.json", summary)
    print(f"\nMaster: {out.name}  sha={sha[:24]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
