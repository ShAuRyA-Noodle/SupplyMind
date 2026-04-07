"""
SupplyMind Scripted (Rule-Based) Agent

A deterministic, zero-LLM agent that follows hard-coded heuristic rules
to manage supply chain disruptions.  No API key required.

Strategy:
  1. Issue free alerts on any at-risk supplier in early steps.
  2. Activate backups for high-risk, single-source suppliers.
  3. Increase safety stock at warehouses with < 10 days cover.
  4. Reroute via operational ports when primary port is disrupted.
  5. Hedge commodities that have spiked > 1.15x.
  6. Expedite only when inventory is critical (< 3 days) and budget allows.
  7. do_nothing when nothing useful can be done.

Usage:
    python scripted_agent.py                  # default seed (deterministic)
    python scripted_agent.py --seeds 42 99 7  # average over multiple seeds
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from models import SupplyMindAction, SupplyMindObservation
from server.supply_environment import SupplyMindEnvironment

logger = logging.getLogger(__name__)

TASK_IDS = [
    "easy_typhoon_response",
    "medium_multi_front",
    "hard_cascading_crisis",
]


# ---------------------------------------------------------------------------
# Heuristic decision logic
# ---------------------------------------------------------------------------


def _has_warning_signals(obs: SupplyMindObservation) -> bool:
    """Check if any disruption is in warning phase."""
    return any(s.lifecycle_phase == "warning" for s in obs.active_signals)


def _affected_supplier_ids(obs: SupplyMindObservation) -> set[str]:
    """Get all supplier IDs affected by any active disruption."""
    ids: set[str] = set()
    for sig in obs.active_signals:
        ids.update(sig.affected_node_ids)
    return ids


def choose_action(obs: SupplyMindObservation, step: int) -> SupplyMindAction:
    """Pick the best action using deterministic proactive heuristics.

    Key strategy: act DURING warning phase (before disruptions hit) to
    maximize timeliness and proactive grader components.
    """
    budget = obs.financials.budget_remaining
    affected = _affected_supplier_ids(obs)
    has_warning = _has_warning_signals(obs)

    # ── Step 0: free alert on first at-risk supplier ──
    if step == 0:
        for n in obs.node_statuses:
            if n.node_type == "supplier" and (n.node_id in affected or n.current_risk_score > 0.1):
                return SupplyMindAction(
                    action_type="issue_supplier_alert",
                    target_node_id=n.node_id,
                )

    # ── PROACTIVE: during warning phase, activate backups immediately ──
    # This is critical for timeliness scoring — act before impact, not after
    if has_warning:
        for n in obs.node_statuses:
            if n.node_type == "supplier" and n.has_backup and n.node_id in affected:
                for backup_id in n.backup_supplier_ids:
                    return SupplyMindAction(
                        action_type="activate_backup_supplier",
                        target_node_id=n.node_id,
                        backup_supplier_id=backup_id,
                    )

    # ── PROACTIVE: increase safety stock at warehouses before disruption hits ──
    if has_warning and budget > 200_000:
        warehouses = [
            n for n in obs.node_statuses
            if n.node_type == "warehouse" and n.inventory_days_cover < 15
        ]
        if warehouses:
            warehouses.sort(key=lambda n: n.inventory_days_cover)
            target = warehouses[0]
            extra = min(10, max(5, 15 - int(target.inventory_days_cover)))
            return SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id=target.node_id,
                additional_stock_days=extra,
            )

    # ── Activate backups for disrupted/high-risk suppliers ──
    for n in obs.node_statuses:
        if n.node_type == "supplier" and n.has_backup and (
            not n.is_operational or n.current_risk_score > 0.4
        ):
            for backup_id in n.backup_supplier_ids:
                return SupplyMindAction(
                    action_type="activate_backup_supplier",
                    target_node_id=n.node_id,
                    backup_supplier_id=backup_id,
                )

    # ── Increase safety stock at low-inventory warehouses ──
    low_wh = [
        n for n in obs.node_statuses
        if n.node_type == "warehouse" and 0 < n.inventory_days_cover < 10
    ]
    if low_wh and budget > 200_000:
        low_wh.sort(key=lambda n: n.inventory_days_cover)
        target = low_wh[0]
        extra = min(15, max(5, 10 - int(target.inventory_days_cover)))
        return SupplyMindAction(
            action_type="increase_safety_stock",
            target_node_id=target.node_id,
            additional_stock_days=extra,
        )

    # ── Reroute past disrupted ports ──
    disrupted_ports = [
        n for n in obs.node_statuses
        if n.node_type == "port" and (not n.is_operational or n.current_risk_score > 0.5)
    ]
    operational_ports = [
        n for n in obs.node_statuses
        if n.node_type == "port" and n.is_operational and n.current_risk_score < 0.3
    ]
    if disrupted_ports and operational_ports and budget > 50_000:
        return SupplyMindAction(
            action_type="reroute_shipment",
            target_node_id=disrupted_ports[0].node_id,
            reroute_via=[operational_ports[0].node_id],
        )

    # ── Hedge spiking commodities ──
    spikes = {
        k: v for k, v in obs.financials.commodity_price_changes.items()
        if v > 1.10
    }
    if spikes and budget > 200_000:
        commodity = max(spikes, key=spikes.get)
        hedge_amt = min(budget * 0.05, 500_000)
        return SupplyMindAction(
            action_type="hedge_commodity",
            commodity=commodity,
            hedge_amount_usd=hedge_amt,
        )

    # ── Expedite critical shortages (expensive, last resort) ──
    critical = [
        n for n in obs.node_statuses
        if n.node_type == "warehouse" and 0 < n.inventory_days_cover < 3
    ]
    if critical and budget > 500_000:
        return SupplyMindAction(
            action_type="expedite_order",
            target_node_id=critical[0].node_id,
            expedite_mode="air",
        )

    # ── Free alert on any new signals ──
    if obs.new_signals:
        for sig in obs.new_signals:
            if sig.affected_node_ids:
                return SupplyMindAction(
                    action_type="issue_supplier_alert",
                    target_node_id=sig.affected_node_ids[0],
                )

    return SupplyMindAction(action_type="do_nothing")


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


def run_task(
    env: SupplyMindEnvironment,
    task_id: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a single task with the scripted agent."""
    start = time.time()
    obs = env.reset(task_id=task_id, seed=seed)
    step_count = 0

    while not obs.done:
        action = choose_action(obs, step_count)
        obs = env.step(action)
        step_count += 1

    result = env.grade()
    result["elapsed_seconds"] = round(time.time() - start, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SupplyMind scripted agent")
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=[None],
        help="Seeds to run per task (default: deterministic, no seed)",
    )
    args = parser.parse_args()

    env = SupplyMindEnvironment()

    print("=" * 60)
    print("SupplyMind Scripted Agent")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    all_results: dict[str, list[float]] = {t: [] for t in TASK_IDS}

    for seed in args.seeds:
        for task_id in TASK_IDS:
            result = run_task(env, task_id, seed=seed)
            score = result["score"]
            all_results[task_id].append(score)
            print(f"  {task_id} (seed={seed}): score={score:.4f}, "
                  f"steps={result['steps_taken']}, time={result['elapsed_seconds']}s")
            if "breakdown" in result:
                for k, v in result["breakdown"].items():
                    print(f"    {k}: {v['score']:.4f} (weight={v['weight']})")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0.0
    for task_id in TASK_IDS:
        scores = all_results[task_id]
        avg = sum(scores) / len(scores)
        total += avg
        print(f"  {task_id}: {avg:.4f} (n={len(scores)})")
    print(f"\n  Average: {total / len(TASK_IDS):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
