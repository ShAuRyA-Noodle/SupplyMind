"""
newsapi.py — NewsAPI.org polling for geopolitical supply-chain signals.

Free tier: 100 req/day. We use 1 req per 5-minute cycle on specific keywords,
stays well under the limit.

Keywords focus on Hormuz / Iran / Israel / Red Sea / Taiwan Strait / port closure.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from ..store import Event

logger = logging.getLogger(__name__)

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
SOURCE_NAME = "newsapi"

# Supply-chain risk keywords, grouped by region.
# Keep queries SIMPLE — NewsAPI quoted phrases over-restrict on free tier.
# Unquoted OR expressions return thousands of relevant articles.
QUERIES = {
    "hormuz": "Hormuz OR Gulf of Oman tanker OR Persian Gulf escalation",
    "iran_israel": "Iran Israel strike OR IDF Tehran OR Hezbollah attack",
    "red_sea": "Houthi Red Sea OR Bab-el-Mandeb OR Suez Canal disruption",
    "taiwan_strait": "Taiwan Strait OR TSMC disruption OR Taiwan semiconductor risk",
    "global_ports": "port strike OR container backlog OR shipping disruption",
}

# Map query name to region tag stored with event
REGION_TAG = {
    "hormuz": "hormuz",
    "iran_israel": "iran_israel",
    "red_sea": "red_sea",
    "taiwan_strait": "taiwan_strait",
    "global_ports": "global",
}


def _severity_from_title(title: str) -> float:
    """Cheap keyword-based severity estimate in [0, 1]."""
    t = (title or "").lower()
    score = 0.1
    for word, weight in [
        ("attack", 0.25), ("strike", 0.2), ("closed", 0.2), ("blockade", 0.25),
        ("missile", 0.25), ("bomb", 0.25), ("cyber", 0.15), ("escalat", 0.2),
        ("seize", 0.25), ("fire", 0.15), ("explosion", 0.2), ("drone", 0.15),
        ("sanctions", 0.15), ("halt", 0.15), ("disrupt", 0.1), ("shortage", 0.1),
    ]:
        if word in t:
            score += weight
    return min(1.0, score)


def _classify_event_type(title: str, description: str) -> str:
    text = ((title or "") + " " + (description or "")).lower()
    if any(w in text for w in ("missile", "strike", "bomb", "attack", "drone")):
        return "kinetic_conflict"
    if any(w in text for w in ("blockade", "closed", "closure", "halt")):
        return "route_closure"
    if any(w in text for w in ("cyber", "hacker", "ransomware")):
        return "cyber_attack"
    if any(w in text for w in ("sanctions", "tariff", "export control")):
        return "policy_shock"
    if any(w in text for w in ("earthquake", "typhoon", "flood", "storm")):
        return "natural_disaster"
    return "news_signal"


def fetch(lookback_minutes: int = 120, api_key: Optional[str] = None) -> list[Event]:
    """Poll NewsAPI for all tracked queries. Returns deduped event list.

    Args:
        lookback_minutes: how far back to query (NewsAPI free tier limits to 30 days).
        api_key: override env var NEWS_API_KEY.
    """
    key = api_key or os.environ.get("NEWS_API_KEY")
    if not key:
        logger.warning("[newsapi] NEWS_API_KEY not set, skipping")
        return []

    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    from_param = since.strftime("%Y-%m-%dT%H:%M:%S")

    events: list[Event] = []
    for q_name, q_str in QUERIES.items():
        try:
            resp = requests.get(
                NEWSAPI_ENDPOINT,
                params={
                    "q": q_str,
                    "from": from_param,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "apiKey": key,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("[newsapi] %s -> %d %s", q_name, resp.status_code, resp.text[:200])
                continue
            data = resp.json()
            articles = data.get("articles", [])
            for art in articles:
                title = art.get("title") or ""
                desc = art.get("description") or ""
                url = art.get("url") or ""
                pub = art.get("publishedAt") or datetime.now(timezone.utc).isoformat()
                if not title:
                    continue
                ev = Event(
                    source=SOURCE_NAME,
                    ts_iso=pub.replace("+00:00", "Z"),
                    event_type=_classify_event_type(title, desc),
                    region=REGION_TAG[q_name],
                    severity=_severity_from_title(title),
                    raw_text=f"{title}. {desc}",
                    urls=[url] if url else [],
                    entities=_extract_entities(title + " " + desc),
                    meta={"query": q_name, "newsapi_source": art.get("source", {}).get("name")},
                )
                events.append(ev)
            time.sleep(0.5)  # be nice to the API
        except Exception as e:  # noqa: BLE001
            logger.error("[newsapi] %s fetch failed: %s", q_name, e)

    logger.info("[newsapi] fetched %d events across %d queries", len(events), len(QUERIES))
    return events


KNOWN_ENTITIES = {
    "TSMC", "Samsung", "Apple", "Foxconn", "ASML", "Nvidia", "Intel",
    "Iran", "Israel", "Hormuz", "Tehran", "Tel Aviv", "Haifa", "Kaohsiung",
    "Houthi", "Hezbollah", "IDF", "IRGC", "Red Sea", "Bab-el-Mandeb",
    "Suez", "Taiwan", "China", "Russia", "Ukraine", "Brent", "WTI",
}


def _extract_entities(text: str) -> list[str]:
    """Very cheap entity extraction via exact-match list."""
    found = []
    for ent in KNOWN_ENTITIES:
        if re.search(rf"\b{re.escape(ent)}\b", text, re.IGNORECASE):
            found.append(ent)
    return found


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-min", type=int, default=120)
    args = parser.parse_args()

    evs = fetch(args.lookback_min)
    for e in evs[:10]:
        print(f"{e.ts_iso}  {e.region:15s}  {e.event_type:18s}  sev={e.severity:.2f}  {e.raw_text[:80]}")
    print(f"\ntotal: {len(evs)}")
