"""
DataCo Supply Chain Dataset Integration.

Real data from Kaggle: 180,519 orders, 20,652 customers, 2015-2017.
Source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

This module:
  1. Extracts real supply chain patterns (delay distributions, customer segments,
     profit margins, shipping mode preferences)
  2. Calibrates our environment's SLA penalties, lead times, and customer delays
     to match real-world distributions
  3. Provides a "real-data benchmark" — agents evaluated on whether their actions
     would have mitigated actual historical disruptions
  4. Creates a disruption signal generator based on real late-delivery patterns

Usage:
    from rl.dataco_integration import DataCoAnalyzer
    analyzer = DataCoAnalyzer()
    stats = analyzer.get_delay_statistics()
    signals = analyzer.generate_real_signals(n=100)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATACO_PATH = DATA_DIR / "dataco.csv"
DATACO_STATS_PATH = DATA_DIR / "dataco_statistics.json"


class DataCoAnalyzer:
    """Extract and use real supply chain patterns from DataCo dataset."""

    def __init__(self) -> None:
        self.df = None
        self.stats: dict[str, Any] = {}

    def load(self) -> None:
        """Load DataCo CSV. Heavy operation (~3 sec)."""
        try:
            import pandas as pd
            self.df = pd.read_csv(str(DATACO_PATH), encoding="latin-1")
            logger.info("DataCo loaded: %d rows, %d columns", len(self.df), len(self.df.columns))
        except Exception as e:
            logger.error("Failed to load DataCo: %s", e)

    def compute_statistics(self) -> dict[str, Any]:
        """Extract all relevant statistics from DataCo. Caches to JSON."""
        if DATACO_STATS_PATH.exists():
            logger.info("Loading cached DataCo stats")
            self.stats = json.loads(DATACO_STATS_PATH.read_text())
            return self.stats

        if self.df is None:
            self.load()

        df = self.df
        stats: dict[str, Any] = {
            "source": "DataCo Smart Supply Chain (Kaggle)",
            "url": "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
            "n_orders": len(df),
            "n_customers": int(df["Customer Id"].nunique()),
            "n_products": int(df["Product Card Id"].nunique()),
            "n_countries": int(df["Order Country"].nunique()),
            "date_range": {
                "start": str(df["order date (DateOrders)"].min()),
                "end": str(df["order date (DateOrders)"].max()),
            },
        }

        # --- Delay/SLA statistics ---
        real_ship = df["Days for shipping (real)"].dropna()
        sched_ship = df["Days for shipment (scheduled)"].dropna()
        delay = real_ship - sched_ship
        late_mask = delay > 0

        stats["shipping"] = {
            "real_days_mean": float(real_ship.mean()),
            "real_days_p50": float(real_ship.median()),
            "real_days_p95": float(real_ship.quantile(0.95)),
            "scheduled_days_mean": float(sched_ship.mean()),
            "late_delivery_rate": float(late_mask.mean()),
            "avg_delay_days_when_late": float(delay[late_mask].mean()) if late_mask.any() else 0,
            "max_delay_days": float(delay.max()),
        }

        # --- Financial statistics ---
        benefit = df["Benefit per order"].dropna()
        sales = df["Sales per customer"].dropna()
        profit_ratio = df["Order Item Profit Ratio"].dropna()

        stats["financial"] = {
            "avg_benefit_per_order_usd": float(benefit.mean()),
            "benefit_std": float(benefit.std()),
            "loss_making_order_rate": float((benefit < 0).mean()),
            "avg_sales_per_customer_usd": float(sales.mean()),
            "avg_profit_ratio": float(profit_ratio.mean()),
        }

        # --- Customer segment distribution ---
        segment_counts = df["Customer Segment"].value_counts(normalize=True)
        stats["customer_segments"] = {k: float(v) for k, v in segment_counts.items()}

        # --- Shipping mode patterns ---
        mode_stats = df.groupby("Shipping Mode").agg({
            "Days for shipping (real)": "mean",
            "Benefit per order": "mean",
            "Late_delivery_risk": "mean",
        }).round(3)
        stats["shipping_modes"] = {
            mode: {
                "avg_days": float(row["Days for shipping (real)"]),
                "avg_benefit": float(row["Benefit per order"]),
                "late_risk": float(row["Late_delivery_risk"]),
            }
            for mode, row in mode_stats.iterrows()
        }

        # --- Market distribution ---
        market_counts = df["Market"].value_counts(normalize=True)
        stats["markets"] = {k: float(v) for k, v in market_counts.items()}

        # --- Order status (like disruption types) ---
        status_counts = df["Order Status"].value_counts(normalize=True)
        stats["order_status_distribution"] = {k: float(v) for k, v in status_counts.items()}

        # --- Delivery status distribution ---
        delivery_counts = df["Delivery Status"].value_counts(normalize=True)
        stats["delivery_status_distribution"] = {k: float(v) for k, v in delivery_counts.items()}

        # --- Category-level statistics ---
        cat_stats = df.groupby("Category Name").agg({
            "Sales": "mean",
            "Late_delivery_risk": "mean",
            "Order Item Profit Ratio": "mean",
        }).round(3)
        stats["top_categories_by_sales"] = []
        for cat, row in cat_stats.nlargest(10, "Sales").iterrows():
            stats["top_categories_by_sales"].append({
                "category": cat,
                "avg_sales_usd": float(row["Sales"]),
                "late_risk": float(row["Late_delivery_risk"]),
                "profit_ratio": float(row["Order Item Profit Ratio"]),
            })

        # Cache
        DATACO_STATS_PATH.write_text(json.dumps(stats, indent=2))
        logger.info("DataCo statistics cached to %s", DATACO_STATS_PATH)
        self.stats = stats
        return stats

    def get_calibration_parameters(self) -> dict[str, float]:
        """Get parameters to calibrate our environment to match DataCo reality."""
        if not self.stats:
            self.compute_statistics()

        return {
            # From 180K real orders
            "real_late_delivery_rate": self.stats["shipping"]["late_delivery_rate"],
            "real_avg_delay_when_late": self.stats["shipping"]["avg_delay_days_when_late"],
            "real_sla_buffer_days": self.stats["shipping"]["scheduled_days_mean"],
            "real_loss_making_rate": self.stats["financial"]["loss_making_order_rate"],
            "real_avg_profit_ratio": self.stats["financial"]["avg_profit_ratio"],
            "source_records": self.stats["n_orders"],
        }

    def generate_real_signals(self, n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
        """Generate disruption signals based on real DataCo late-delivery patterns.

        Returns list of (region, severity, confidence) triples sampled from real data.
        """
        if self.df is None:
            self.load()

        rng = np.random.default_rng(seed)
        late = self.df[self.df["Late_delivery_risk"] == 1].copy()
        late = late.dropna(subset=["Days for shipping (real)", "Days for shipment (scheduled)"])

        signals = []
        for _ in range(n):
            row = late.iloc[rng.integers(0, len(late))]
            delay_days = row["Days for shipping (real)"] - row["Days for shipment (scheduled)"]
            severity = min(1.0, max(0.1, delay_days / 14))

            signals.append({
                "market": row["Market"],
                "order_country": row["Order Country"],
                "category": row["Category Name"],
                "delay_days": int(delay_days),
                "severity": float(severity),
                "shipping_mode": row["Shipping Mode"],
                "order_status": row["Order Status"],
                "source": "DataCo real order",
            })

        return signals

    def backtest_agent_on_real_data(
        self, agent_fn, n_orders: int = 100, seed: int = 42,
    ) -> dict[str, float]:
        """Backtest a late-delivery predictor on a mixed sample of real orders.

        Draws ``n_orders`` orders uniformly from the FULL DataCo table (both
        late and on-time — not late-only), builds an order-time feature signal
        for each (no realized-delay leakage), and calls ``agent_fn(signal)`` to
        obtain a prediction. ``agent_fn`` may return a bool or a probability/
        score (thresholded at 0.5). Predictions are scored against the actual
        outcome ``is_late = (real - scheduled) > 0``.

        Returns accuracy plus precision/recall/F1 and the sample's base late
        rate, so a trivial always-late predictor cannot look good by accuracy.
        """
        if self.df is None:
            self.load()
        if self.df is None:
            raise RuntimeError(
                "DataCo dataframe is unavailable (load failed) — cannot backtest."
            )

        df = self.df.dropna(
            subset=["Days for shipping (real)", "Days for shipment (scheduled)"]
        ).reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No DataCo orders with shipping columns available.")

        rng = np.random.default_rng(seed)
        n = min(n_orders, len(df))
        idx = rng.choice(len(df), size=n, replace=False)

        tp = fp = tn = fn = 0
        n_actual_late = 0
        for i in idx:
            row = df.iloc[int(i)]
            scheduled = float(row["Days for shipment (scheduled)"])
            delay = float(row["Days for shipping (real)"]) - scheduled
            is_late = delay > 0
            n_actual_late += int(is_late)

            # Order-time features only — the realized delay is NOT exposed.
            signal = {
                "market": row.get("Market"),
                "order_country": row.get("Order Country"),
                "category": row.get("Category Name"),
                "shipping_mode": row.get("Shipping Mode"),
                "order_status": row.get("Order Status"),
                "scheduled_days": scheduled,
                "sales_per_customer": float(row.get("Sales per customer", 0.0) or 0.0),
                "order_item_quantity": float(row.get("Order Item Quantity", 0.0) or 0.0),
                "source": "DataCo historical order",
            }

            pred = agent_fn(signal)
            if isinstance(pred, (bool, np.bool_)):
                predicted_late = bool(pred)
            else:
                predicted_late = float(pred) >= 0.5

            if predicted_late and is_late:
                tp += 1
            elif predicted_late and not is_late:
                fp += 1
            elif not predicted_late and not is_late:
                tn += 1
            else:
                fn += 1

        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "n_orders": int(n),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "actual_late_rate": float(n_actual_late / n),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "source": "DataCo historical orders (mixed late + on-time sample)",
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    analyzer = DataCoAnalyzer()
    stats = analyzer.compute_statistics()

    print("\n" + "=" * 60)
    print("DATACO SUPPLY CHAIN DATASET ANALYSIS")
    print("=" * 60)
    print(f"Orders: {stats['n_orders']:,}")
    print(f"Customers: {stats['n_customers']:,}")
    print(f"Products: {stats['n_products']:,}")
    print(f"Countries: {stats['n_countries']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print()
    print("SHIPPING REALITY:")
    s = stats["shipping"]
    print(f"  Late delivery rate: {s['late_delivery_rate']*100:.1f}%")
    print(f"  Avg scheduled days: {s['scheduled_days_mean']:.1f}")
    print(f"  Avg real days: {s['real_days_mean']:.1f}")
    print(f"  Avg delay when late: {s['avg_delay_days_when_late']:.1f} days")
    print()
    print("FINANCIAL REALITY:")
    f = stats["financial"]
    print(f"  Avg benefit per order: ${f['avg_benefit_per_order_usd']:.2f}")
    print(f"  Loss-making order rate: {f['loss_making_order_rate']*100:.1f}%")
    print(f"  Avg profit ratio: {f['avg_profit_ratio']*100:.1f}%")
    print()
    print("CUSTOMER SEGMENTS:")
    for seg, pct in stats["customer_segments"].items():
        print(f"  {seg}: {pct*100:.1f}%")


if __name__ == "__main__":
    main()
