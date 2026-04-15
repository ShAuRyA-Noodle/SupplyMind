"""
Phase G — Build real RAG corpus with Ollama nomic-embed-text.

Real sources indexed:
  1. Crisis library (5 JSON scenarios: Tohoku 2011, Suez 2021, Red Sea 2023, ...)
  2. NOAA IBTRACS historical storm narratives (top 200 by wind)
  3. USGS earthquake records
  4. DataCo delivery-risk pattern summaries (by market × segment × risk)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from rl.rag.indexer import CrisisRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CRISIS_LIB = ROOT / "benchmark" / "crisis_library"
NOAA = ROOT / "rl" / "data" / "ibtracs_wp.csv"
USGS = ROOT / "rl" / "data" / "usgs_m55_30days.csv"
DATACO = ROOT / "rl" / "data" / "dataco.csv"


def index_crisis_library(rag: CrisisRAG) -> int:
    total = 0
    if not CRISIS_LIB.exists():
        log.warning("No crisis_library at %s", CRISIS_LIB)
        return 0
    for path in CRISIS_LIB.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            text = json.dumps(data, indent=2)
            total += rag.index_text(text, source=f"crisis_library/{path.name}")
        except Exception as e:
            log.warning("Skip %s: %s", path.name, e)
    return total


def index_noaa(rag: CrisisRAG, top_n: int = 200) -> int:
    if not NOAA.exists():
        return 0
    df = pd.read_csv(NOAA, low_memory=False, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    wind_col = "WMO_WIND" if "WMO_WIND" in df.columns else "USA_WIND"
    name_col = "NAME" if "NAME" in df.columns else None
    if wind_col not in df.columns:
        return 0

    # Aggregate per storm: max wind, year, name
    key_col = "SID" if "SID" in df.columns else "NUMBER"
    agg = df.groupby(key_col).agg({
        wind_col: "max",
        "SEASON": "first",
        name_col: "first" if name_col else wind_col,
    }).reset_index()
    agg[wind_col] = pd.to_numeric(agg[wind_col], errors="coerce")
    agg = agg.dropna(subset=[wind_col]).sort_values(wind_col, ascending=False).head(top_n)

    count = 0
    for _, row in agg.iterrows():
        name = row[name_col] if name_col else "Unnamed"
        text = (
            f"Storm {name} (SID {row[key_col]}) in season {int(row['SEASON'])}: "
            f"peak sustained winds {row[wind_col]:.0f} knots, Western Pacific basin. "
            f"Typhoons of this magnitude typically cause port closures in Taiwan, Japan, Philippines, "
            f"disrupting semiconductor and electronics supply chains for 3-14 days. "
            f"Historical precedent for NOAA-driven supply-chain risk forecasting."
        )
        count += rag.index_text(text, source=f"NOAA_IBTRACS/{row[key_col]}")
    return count


def index_usgs(rag: CrisisRAG) -> int:
    if not USGS.exists():
        return 0
    df = pd.read_csv(USGS)
    count = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("mag")):
            continue
        text = (
            f"Earthquake M{row['mag']:.1f} on {row['time']} at depth {row.get('depth', 0):.0f} km. "
            f"Location: {row.get('place', 'unknown')}. Magnitude >5.5 events historically "
            f"disrupt semiconductor fabs (TSMC, Samsung), port operations, and ground logistics "
            f"for 24-72 hours. Risk factor for Tier-1 electronics suppliers."
        )
        count += rag.index_text(text, source=f"USGS/{row.get('id','unknown')}")
    return count


def index_dataco_patterns(rag: CrisisRAG) -> int:
    if not DATACO.exists():
        return 0
    df = pd.read_csv(DATACO, encoding="latin-1", low_memory=False)
    # Group by Market x Segment x late_delivery_risk, summarize
    grp = df.groupby(["Market", "Customer Segment", "Late_delivery_risk"]).agg({
        "Order Item Profit Ratio": "mean",
        "Days for shipping (real)": "mean",
        "Days for shipment (scheduled)": "mean",
        "Sales per customer": "mean",
        "Order Id": "count",
    }).reset_index()
    grp.rename(columns={"Order Id": "order_count"}, inplace=True)

    count = 0
    for _, row in grp.iterrows():
        delay = row["Days for shipping (real)"] - row["Days for shipment (scheduled)"]
        text = (
            f"DataCo pattern: Market={row['Market']}, Segment={row['Customer Segment']}, "
            f"late_risk={int(row['Late_delivery_risk'])}. "
            f"n={int(row['order_count'])} orders. Avg profit ratio {row['Order Item Profit Ratio']:.3f}, "
            f"avg shipping delay {delay:.2f} days, avg sales per customer ${row['Sales per customer']:.0f}. "
            f"This pattern indicates {'high risk — activate mitigation' if row['Late_delivery_risk'] == 1 else 'baseline operations'}."
        )
        count += rag.index_text(text, source="DataCo_pattern")
    return count


def main():
    rag = CrisisRAG()
    log.info("Initial count: %d", rag.count())

    totals = {}
    log.info("Indexing crisis library...")
    totals["crisis_lib"] = index_crisis_library(rag)
    log.info("Indexing NOAA (top 200 storms)...")
    totals["noaa"] = index_noaa(rag, top_n=200)
    log.info("Indexing USGS earthquakes...")
    totals["usgs"] = index_usgs(rag)
    log.info("Indexing DataCo patterns...")
    totals["dataco"] = index_dataco_patterns(rag)

    final = rag.count()
    log.info("Final count: %d. Breakdown: %s", final, totals)

    # Sanity query
    test = rag.retrieve_precedents("Taiwan earthquake semiconductor TSMC fab disruption", n=3)
    log.info("Sanity retrieval (%d hits):", len(test))
    for p in test:
        log.info("  [%.3f] %s: %s", p["relevance_score"], p["source"], p["text"][:120])


if __name__ == "__main__":
    main()
