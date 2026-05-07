"""wiki_pageviews.py — Wikipedia REST pageview API.

Real-time public-attention signal. No auth. Free.

When "Strait_of_Hormuz" pageviews spike 5-10x baseline, something happened.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ._common import cached_get, http_get_json, standard_event

logger = logging.getLogger(__name__)

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

# Articles whose pageview spikes correlate with supply-chain disruption
WATCH_LIST = [
    "Strait_of_Hormuz", "Suez_Canal", "Bab-el-Mandeb",
    "Panama_Canal", "Singapore_Strait",
    "Supply_chain", "Houthi_movement",
    "TSMC", "Iranian_drone_attacks_on_Israel",
    "Russian_invasion_of_Ukraine",
]


def fetch_pageview_pulses(days_back: int = 7) -> list[dict]:
    """For each watch-list article, return the most recent pageview spike."""
    cache_key = f"pulses_d{days_back}_list"

    def _fetch():
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days_back + 30)  # extra for baseline
        out: list[dict] = []
        for article in WATCH_LIST:
            url = (f"{BASE}/en.wikipedia/all-access/all-agents/"
                   f"{article}/daily/"
                   f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}")
            try:
                data = http_get_json(url, timeout=20,
                                      headers={"User-Agent": "supplymind/1.0"})
            except Exception as e:  # noqa: BLE001
                logger.warning("[wiki_pageviews] %s failed: %s", article, str(e)[:80])
                continue
            items = data.get("items") or []
            if len(items) < 8:
                continue
            recent_window = items[-days_back:]
            baseline_window = items[-days_back - 21:-days_back]
            if not recent_window or not baseline_window:
                continue
            recent_max = max((it.get("views") or 0) for it in recent_window)
            baseline_med = sorted(it.get("views") or 0
                                   for it in baseline_window)[len(baseline_window) // 2]
            if baseline_med <= 0:
                continue
            spike = recent_max / max(1.0, baseline_med)
            spike_day = max(recent_window, key=lambda i: i.get("views") or 0)
            sev = min(1.0, (spike - 1.0) / 9.0)  # 10x baseline -> sev 1.0
            out.append(standard_event(
                source="wiki_pageviews",
                event_id=f"pageview_{article}_{spike_day.get('timestamp')}",
                title=f"{article.replace('_',' ')} pageview spike {spike:.2f}x baseline",
                description=(f"Wikipedia pageviews for {article}: "
                             f"recent_max={recent_max}, "
                             f"30d_baseline_median={baseline_med}, "
                             f"spike_ratio={spike:.2f}. "
                             f"Spike date: {spike_day.get('timestamp')}."),
                occurred_at_utc=_iso_from_ts(spike_day.get("timestamp")),
                raw_url=f"https://pageviews.wmcloud.org/?project=en.wikipedia.org&pages={article}",
                severity_proxy=sev,
                extra={
                    "article": article,
                    "spike_ratio": round(spike, 2),
                    "recent_max_views": recent_max,
                    "baseline_median": baseline_med,
                },
            ))
        # Sort by spike strength descending
        out.sort(key=lambda e: e["severity_proxy"] or 0, reverse=True)
        logger.info("[wiki_pageviews] returned %d pulses across watch list", len(out))
        return out

    return cached_get("wiki_pageviews", cache_key, _fetch, ttl=3600)


def _iso_from_ts(ts: str | None) -> str | None:
    if not ts or len(ts) < 10:
        return ts
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:00:00Z"


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = fetch_pageview_pulses(days_back=7)
    print(json.dumps(e[:5], indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(e)} pageview pulses")
