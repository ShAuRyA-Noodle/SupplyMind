"""test_multi_agent_demo.py — G4+F2 regression."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ShAuRyA_Supplymind.features.multi_agent_demo import (
    Agent, run_competition, _bid_by_strategy,
)


def test_three_agents_compete_and_allocate_all_capacity():
    out = run_competition(seed=42)
    outcomes = out["outcomes"]
    assert len(outcomes) == 3
    names = {a["name"] for a in outcomes}
    assert names == {"Apple", "Samsung", "Toyota"}
    # Combined allocated wafers ~ full capacity (1000 wafers/week)
    total_alloc = sum(a["allocated_wafers"] for a in outcomes)
    assert 990 <= total_alloc <= 1010


def test_aggressive_bids_more_in_step_1_than_conservative():
    apple = Agent("Apple", 22_000_000, "aggressive")
    samsung = Agent("Samsung", 14_000_000, "conservative")
    apple_bid = _bid_by_strategy(apple, step=1, price_signal=1.0)
    samsung_bid = _bid_by_strategy(samsung, step=1, price_signal=1.0)
    # Apple: 0.70 * 22M = 15.4M; Samsung: 0.25 * 14M = 3.5M
    assert apple_bid > samsung_bid


def test_reactive_waits_in_step_1():
    toyota = Agent("Toyota", 7_000_000, "reactive")
    assert _bid_by_strategy(toyota, step=1, price_signal=1.0) == 0.0


def test_winner_has_highest_pnl():
    out = run_competition(seed=42)
    ranking = out["ranking"]
    assert len(ranking) == 3
    # Net P&L descending
    assert ranking[0]["net_pnl_usd"] >= ranking[1]["net_pnl_usd"]
    assert ranking[1]["net_pnl_usd"] >= ranking[2]["net_pnl_usd"]
    assert out["winner"] == ranking[0]["agent"]
