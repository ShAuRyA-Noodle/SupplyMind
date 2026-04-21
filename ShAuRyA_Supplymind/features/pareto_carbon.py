"""
pareto_carbon.py — F9. Carbon-aware multi-objective Pareto frontier.

Three objectives per action plan:
    cost_usd        — direct monetary cost
    resilience_bps  — P95 loss-avoided per dollar spent (higher = better)
    carbon_kg_co2   — emissions from transport mode choices

Emission factors (per kg cargo * km transit):
    air:  0.82 kg CO2/tonne-km       (ICAO, IATA)
    sea:  0.013 kg CO2/tonne-km      (IMO Fourth GHG Study 2020)
    rail: 0.028 kg CO2/tonne-km      (EPA)
    road: 0.096 kg CO2/tonne-km      (EPA)

We enumerate 20 candidate action plans (combinations of transport mode × backup
activation × safety-stock depth × hedge level) and extract the Pareto-optimal
subset. A weighted slider lets the user pick (cost, resilience, carbon) weights
and returns the best plan under that scalarization.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "F9_PARETO_CARBON.json"

# Emission factors — kg CO2 per tonne-km
EMISSION_FACTORS = {
    "air": 0.82,
    "express_sea": 0.020,
    "sea": 0.013,
    "rail": 0.028,
    "road": 0.096,
}

# Shipment scenarios (tonnes, km)
SHIPMENT_PROFILES = {
    "shanghai_la_base": {"tonnes": 1200, "km_sea": 10_600, "km_road": 0, "km_air": 0},
    "shanghai_la_expedite_air": {"tonnes": 120, "km_sea": 0, "km_road": 50, "km_air": 10_600},
    "shanghai_ny_rail": {"tonnes": 800, "km_sea": 2_500, "km_road": 100, "km_air": 0, "km_rail": 11_000},
    "reroute_cape": {"tonnes": 1200, "km_sea": 14_500, "km_road": 0, "km_air": 0},
}


@dataclass
class ActionPlan:
    name: str
    description: str
    cost_usd: float
    resilience_bps: float    # basis points of loss-avoided-per-dollar
    carbon_kg_co2: float
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "cost_usd": round(self.cost_usd, 0),
            "resilience_bps": round(self.resilience_bps, 1),
            "carbon_kg_co2": round(self.carbon_kg_co2, 0),
            "components": self.components,
        }


def _plan_carbon(profile: str, tonnes_moved: float, mode: str) -> float:
    """Carbon for a single shipment mode choice using the profile's km."""
    p = SHIPMENT_PROFILES[profile]
    km_key = {"air": "km_air", "sea": "km_sea", "express_sea": "km_sea",
              "rail": "km_rail", "road": "km_road"}.get(mode, "km_sea")
    km = p.get(km_key, 0)
    factor = EMISSION_FACTORS[mode]
    return tonnes_moved * km * factor / 1000.0  # -> kg


def generate_plans() -> list[ActionPlan]:
    """Enumerate ~20 candidate plans."""
    plans: list[ActionPlan] = []

    # A. 4 pure-transport-mode plans
    tonnes = 1200
    for mode in ("sea", "express_sea", "rail", "air"):
        cost_map = {"sea": 80_000, "express_sea": 220_000, "rail": 150_000, "air": 850_000}
        # Resilience: air is fastest (highest P95 save) but most expensive
        res_map = {"sea": 30, "express_sea": 45, "rail": 50, "air": 85}
        plans.append(ActionPlan(
            name=f"ship_{mode}",
            description=f"Base shipment via {mode.upper()} only",
            cost_usd=cost_map[mode],
            resilience_bps=res_map[mode],
            carbon_kg_co2=_plan_carbon("shanghai_la_base", tonnes, mode),
            components={"mode": mode, "tonnes": tonnes},
        ))

    # B. 4 reroute options
    for mode in ("sea", "rail"):
        for via in ("panama", "cape_good_hope"):
            km_extra = 3000 if via == "cape_good_hope" else 0
            base_profile = "reroute_cape" if via == "cape_good_hope" else "shanghai_la_base"
            cost_base = 80_000 if mode == "sea" else 150_000
            cost_extra = 60_000 if via == "cape_good_hope" else 30_000
            plans.append(ActionPlan(
                name=f"reroute_{mode}_{via}",
                description=f"Reroute via {via.replace('_', ' ')} using {mode}",
                cost_usd=cost_base + cost_extra,
                resilience_bps={"sea": 60, "rail": 70}[mode],
                carbon_kg_co2=_plan_carbon(base_profile, 1200, mode),
                components={"mode": mode, "via": via},
            ))

    # C. 4 backup-supplier plans (different activation depths)
    for depth_pct in (25, 50, 75, 100):
        plans.append(ActionPlan(
            name=f"backup_{depth_pct}pct",
            description=f"Activate backup supplier at {depth_pct}% of base capacity",
            cost_usd=60_000 + 8_000 * depth_pct,
            resilience_bps=55 + 0.4 * depth_pct,
            carbon_kg_co2=_plan_carbon("shanghai_la_base", 1200 * depth_pct / 100, "sea") + 900,
            components={"backup_depth_pct": depth_pct},
        ))

    # D. 4 safety-stock plans (7-30 day buffers)
    for days in (7, 14, 21, 30):
        plans.append(ActionPlan(
            name=f"safety_stock_{days}d",
            description=f"{days}-day warehouse safety stock buffer",
            cost_usd=22_000 * days,
            resilience_bps=25 + 1.5 * days,
            carbon_kg_co2=80 * days,  # storage-related emissions
            components={"days": days},
        ))

    # E. 4 combo plans
    plans.append(ActionPlan(
        name="combo_hedge_sea_backup25",
        description="Hedge oil + sea shipping + 25% backup",
        cost_usd=250_000,
        resilience_bps=72,
        carbon_kg_co2=_plan_carbon("shanghai_la_base", 1200, "sea") + 500,
        components={"hedge": True, "backup": 25},
    ))
    plans.append(ActionPlan(
        name="combo_cape_rail_backup75",
        description="Cape reroute + rail last-mile + 75% backup",
        cost_usd=410_000,
        resilience_bps=88,
        carbon_kg_co2=_plan_carbon("reroute_cape", 1200, "sea")
                     + _plan_carbon("shanghai_ny_rail", 800, "rail") + 1200,
        components={"reroute": "cape", "rail": True, "backup": 75},
    ))
    plans.append(ActionPlan(
        name="combo_air_premium_full",
        description="Air shipping + 100% backup + 14d stock (fastest + greenest-cost)",
        cost_usd=1_550_000,
        resilience_bps=95,
        carbon_kg_co2=_plan_carbon("shanghai_la_expedite_air", 120, "air")
                     + 14 * 80 + 900,
        components={"air": True, "backup": 100, "stock_days": 14},
    ))
    plans.append(ActionPlan(
        name="do_nothing",
        description="No mitigation; monitor only",
        cost_usd=0,
        resilience_bps=0,
        carbon_kg_co2=_plan_carbon("shanghai_la_base", 1200, "sea"),
        components={},
    ))

    return plans


def pareto_front(plans: list[ActionPlan]) -> list[ActionPlan]:
    """Return the Pareto-optimal subset over (cost MIN, resilience MAX, carbon MIN)."""
    frontier: list[ActionPlan] = []
    for p in plans:
        dominated = False
        for q in plans:
            if p is q:
                continue
            # q dominates p iff q is >= on all objectives and strictly > on at least one
            at_least_as_good = (q.cost_usd <= p.cost_usd and
                                q.resilience_bps >= p.resilience_bps and
                                q.carbon_kg_co2 <= p.carbon_kg_co2)
            strictly_better = (q.cost_usd < p.cost_usd or
                               q.resilience_bps > p.resilience_bps or
                               q.carbon_kg_co2 < p.carbon_kg_co2)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def best_under_weights(
    plans: list[ActionPlan],
    w_cost: float = 0.33,
    w_resilience: float = 0.34,
    w_carbon: float = 0.33,
) -> ActionPlan:
    """Linear scalarization: minimize w_cost*cost + w_carbon*carbon - w_res*resilience.

    Each objective is min-max normalized across plans for unit-free comparison.
    """
    costs = np.array([p.cost_usd for p in plans])
    res = np.array([p.resilience_bps for p in plans])
    carb = np.array([p.carbon_kg_co2 for p in plans])

    def _norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)

    c_n, r_n, k_n = _norm(costs), _norm(res), _norm(carb)
    score = w_cost * c_n + w_carbon * k_n - w_resilience * r_n  # minimize
    return plans[int(np.argmin(score))]


def run_and_save() -> dict:
    plans = generate_plans()
    frontier = pareto_front(plans)

    # Demo three weighting regimes
    conservative = best_under_weights(plans, 0.5, 0.2, 0.3)
    balanced = best_under_weights(plans, 0.33, 0.34, 0.33)
    green = best_under_weights(plans, 0.2, 0.3, 0.5)

    out = {
        "emission_factors_kg_co2_per_tonne_km": EMISSION_FACTORS,
        "shipment_profiles": SHIPMENT_PROFILES,
        "all_plans": [p.to_dict() for p in plans],
        "pareto_frontier": [p.to_dict() for p in frontier],
        "best_under_weights": {
            "conservative_cost_0.5_res_0.2_carbon_0.3": conservative.to_dict(),
            "balanced_0.33_0.34_0.33": balanced.to_dict(),
            "green_cost_0.2_res_0.3_carbon_0.5": green.to_dict(),
        },
        "meta": {
            "n_plans": len(plans),
            "n_pareto": len(frontier),
            "pareto_ratio": round(len(frontier) / max(1, len(plans)), 2),
        },
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    logger.info("[pareto] %d plans -> %d on frontier", len(plans), len(frontier))
    logger.info("[pareto] conservative: %s", conservative.name)
    logger.info("[pareto] balanced:     %s", balanced.name)
    logger.info("[pareto] green:        %s", green.name)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs=3, type=float, default=None,
                        help="Custom weights: w_cost w_resilience w_carbon")
    args = parser.parse_args()

    out = run_and_save()
    if args.weights:
        wc, wr, wk = args.weights
        plans = generate_plans()
        best = best_under_weights(plans, wc, wr, wk)
        print(f"\nbest under weights (cost={wc} resilience={wr} carbon={wk}):")
        print(json.dumps(best.to_dict(), indent=2))
    else:
        print(json.dumps(out["meta"], indent=2))
        print("\npareto frontier names:", [p.to_dict()["name"] for p in
                                             pareto_front(generate_plans())])
