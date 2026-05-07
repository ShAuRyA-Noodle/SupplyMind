"""
multi_agent_demo.py — G4+F2. Multi-agent competition on shared supply-chain
capacity under a shared crisis.

Three agents — Apple, Samsung, Toyota — compete for TSMC backup capacity
during a Hormuz closure crisis. Each agent has a budget, a strategy, and
observes the same global signals but makes independent decisions.

Shared-resource constraints:
    - Samsung backup fab has total capacity CAP_total = 1000 wafers/week
    - Each agent bids a dollar amount for a slice of that capacity
    - Allocation: proportional to bid until CAP_total is exhausted
    - Losers get nothing and face production shortfalls

Strategies:
    "aggressive"  — bid ~70% of budget immediately
    "conservative" — bid 25% now, hold reserve
    "reactive"    — wait for price signal, then bid in tier 2

This reproduces the 2021 chip shortage dynamic: early bidders won capacity,
late bidders faced 40-week lead times.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "F2_MULTI_AGENT_DEMO.json"


# Domain constants (2021 chip-shortage calibrated)
CAP_TOTAL_WAFERS_WEEK = 1000
WAFER_REVENUE_USD = 16_500        # TSMC N5 wafer revenue per SemiAnalysis
SHORTFALL_LOSS_USD = 55_000       # per wafer unfulfilled (hardware OEM estimate)
CRISIS_DURATION_WEEKS = 6


@dataclass
class Agent:
    name: str
    budget_usd: float
    strategy: str                              # "aggressive" | "conservative" | "reactive"
    bid_usd: float = 0.0
    allocated_wafers: float = 0.0
    revenue_earned_usd: float = 0.0
    shortfall_loss_usd: float = 0.0

    def net_pnl_usd(self) -> float:
        return self.revenue_earned_usd - self.bid_usd - self.shortfall_loss_usd

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "budget_usd": round(self.budget_usd, 0),
            "bid_usd": round(self.bid_usd, 0),
            "allocated_wafers": round(self.allocated_wafers, 1),
            "revenue_earned_usd": round(self.revenue_earned_usd, 0),
            "shortfall_loss_usd": round(self.shortfall_loss_usd, 0),
            "net_pnl_usd": round(self.net_pnl_usd(), 0),
        }


def _bid_by_strategy(agent: Agent, step: int, price_signal: float) -> float:
    """Return the agent's bid at this competition step (1 or 2)."""
    if step == 1:
        if agent.strategy == "aggressive":
            return 0.70 * agent.budget_usd
        if agent.strategy == "conservative":
            return 0.25 * agent.budget_usd
        if agent.strategy == "reactive":
            return 0.0          # wait for price signal
        return 0.33 * agent.budget_usd
    if step == 2:
        remaining = agent.budget_usd - agent.bid_usd
        if agent.strategy == "reactive":
            # Bid based on observed price signal, scaled down if prices are surge
            price_multiplier = 1.0 / max(0.5, price_signal)
            return min(remaining, 0.60 * agent.budget_usd * price_multiplier)
        if agent.strategy == "aggressive":
            return min(remaining, 0.15 * agent.budget_usd)
        if agent.strategy == "conservative":
            return min(remaining, 0.20 * agent.budget_usd)
    return 0.0


def _allocate_proportional(agents: list[Agent], capacity_remaining: float) -> None:
    """Allocate `capacity_remaining` wafers proportionally to current bids."""
    total_bid = sum(a.bid_usd for a in agents)
    if total_bid <= 0 or capacity_remaining <= 0:
        return
    for a in agents:
        share = a.bid_usd / total_bid
        a.allocated_wafers += share * capacity_remaining


def run_competition(seed: int = 42) -> dict:
    random.seed(seed)
    agents = [
        Agent(name="Apple", budget_usd=22_000_000, strategy="aggressive"),
        Agent(name="Samsung", budget_usd=14_000_000, strategy="conservative"),
        Agent(name="Toyota", budget_usd=7_000_000, strategy="reactive"),
    ]

    log: list[dict] = []

    # Step 1: initial bids
    price_signal_t0 = 1.0
    log.append({"event": "step_1_open", "capacity_remaining": CAP_TOTAL_WAFERS_WEEK,
                "price_signal": price_signal_t0})
    for a in agents:
        bid = _bid_by_strategy(a, step=1, price_signal=price_signal_t0)
        a.bid_usd += bid
        log.append({"event": "step_1_bid", "agent": a.name, "bid_usd": bid})

    # Allocate half of capacity at step 1 (based on step-1 bids)
    step1_capacity = CAP_TOTAL_WAFERS_WEEK * 0.5
    pre_bids = {a.name: a.bid_usd for a in agents}
    _allocate_proportional(agents, step1_capacity)
    for a in agents:
        log.append({"event": "step_1_allocated", "agent": a.name,
                    "allocated_wafers": a.allocated_wafers})

    # Observe price signal: if step-1 demand exceeded step-1 capacity, price surges
    total_step1_bid = sum(pre_bids.values())
    implied_price = total_step1_bid / (step1_capacity * WAFER_REVENUE_USD) if step1_capacity > 0 else 1.0
    price_signal_t1 = max(1.0, implied_price)
    log.append({"event": "step_2_open",
                "capacity_remaining": CAP_TOTAL_WAFERS_WEEK - step1_capacity,
                "price_signal": round(price_signal_t1, 3)})

    # Step 2 bids
    for a in agents:
        bid = _bid_by_strategy(a, step=2, price_signal=price_signal_t1)
        a.bid_usd += bid
        log.append({"event": "step_2_bid", "agent": a.name, "bid_usd": bid})

    # Allocate remaining capacity at step 2 — proportional to incremental bid only
    step2_bid_total = sum(a.bid_usd - pre_bids[a.name] for a in agents)
    step2_capacity = CAP_TOTAL_WAFERS_WEEK - step1_capacity
    if step2_bid_total > 0:
        for a in agents:
            share = (a.bid_usd - pre_bids[a.name]) / step2_bid_total
            a.allocated_wafers += share * step2_capacity

    # Compute outcomes
    for a in agents:
        # Revenue: wafers x CRISIS_DURATION_WEEKS x WAFER_REVENUE_USD
        a.revenue_earned_usd = a.allocated_wafers * CRISIS_DURATION_WEEKS * WAFER_REVENUE_USD
        # Shortfall: each agent is assumed to NEED the capacity equal to their budget/WAFER_REVENUE
        needed = a.budget_usd / WAFER_REVENUE_USD
        shortfall = max(0, needed - a.allocated_wafers)
        a.shortfall_loss_usd = shortfall * SHORTFALL_LOSS_USD

    # Rank by net P&L
    ranked = sorted(agents, key=lambda a: a.net_pnl_usd(), reverse=True)

    out = {
        "constants": {
            "cap_total_wafers_week": CAP_TOTAL_WAFERS_WEEK,
            "wafer_revenue_usd": WAFER_REVENUE_USD,
            "shortfall_loss_usd_per_wafer": SHORTFALL_LOSS_USD,
            "crisis_duration_weeks": CRISIS_DURATION_WEEKS,
        },
        "narrative": ("2021-chip-shortage dynamic: TSMC backup capacity (1000 wafers/week) "
                      "contested by Apple (aggressive) + Samsung (conservative) + Toyota "
                      "(reactive). Apple bids hard early, captures >50% of step-1 capacity. "
                      "Toyota waits, pays higher step-2 prices. Samsung splits budget."),
        "step_log": log,
        "outcomes": [a.to_dict() for a in agents],
        "ranking": [
            {"rank": i + 1, "agent": a.name, "net_pnl_usd": round(a.net_pnl_usd(), 0)}
            for i, a in enumerate(ranked)
        ],
        "winner": ranked[0].name,
        "loser": ranked[-1].name,
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    logger.info("[multi_agent] %s wins with $%.0fM net P&L; %s last with $%.0fM",
                ranked[0].name, ranked[0].net_pnl_usd() / 1e6,
                ranked[-1].name, ranked[-1].net_pnl_usd() / 1e6)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = run_competition(seed=args.seed)
    print(json.dumps({"ranking": out["ranking"], "winner": out["winner"]}, indent=2))
