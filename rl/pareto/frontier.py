"""
Pareto frontier — scripted-agent seed study for SupplyMind.

3 objectives: cost, resilience, sustainability (carbon cost).

IMPORTANT — what this actually does: it runs the SAME scripted agent under N
different environment seeds and computes a 3-objective vector per run. The
per-run weight vectors are recorded for reference but do NOT condition the
agent's behavior (the scripted agent ignores them). This is a seed study of a
single fixed policy, not a set of trained weight-conditioned policies. Do not
present it as multi-policy training.

Carbon estimates use ILLUSTRATIVE assumptions (fixed tonnage / distance per
action), not per-shipment data — every result dict carries "assumption": true.
Emission factors (kg CO2 per tonne-km) are literature values (EPA/IMO/ICAO):
air 0.82, express_sea 0.026, sea 0.013, rail 0.028, road 0.096.

Usage:
    python -m rl.pareto.frontier --run       # run the scripted-agent seed study
    python -m rl.pareto.frontier --visualize
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Real carbon emission factors (kg CO2 per tonne-km)
# Source: EPA, IMO, ICAO emission factor databases
CARBON_FACTORS = {
    "air": 0.82,
    "express_sea": 0.026,
    "sea": 0.013,
    "rail": 0.028,
    "road": 0.096,
}


class ParetoObjectives:
    """Compute 3 objectives for a policy rollout.

    Objectives (all to be MINIMIZED):
      1. Cost: total budget spent / total budget
      2. Resilience: 1 - final_score (lower = more resilient)
      3. Carbon: estimated CO2 emissions from logistics actions
    """

    def __init__(self) -> None:
        self.carbon_factors = CARBON_FACTORS

    def evaluate(
        self,
        episode_history: list[dict[str, Any]],
        financials: dict[str, float],
        total_budget: float,
    ) -> np.ndarray:
        """Compute 3-objective vector from episode data.

        Returns:
            np.array([cost, resilience_loss, carbon]) — all to minimize.
        """
        # 1. Cost objective: fraction of budget spent
        budget_spent = financials.get("cumulative_cost_incurred", 0)
        cost = budget_spent / max(total_budget, 1)

        # 2. Resilience loss: 1 - score
        score = financials.get("score", 0.5)
        resilience_loss = 1.0 - score

        # 3. Carbon emissions from logistics actions.
        # ILLUSTRATIVE ONLY: tonnage/distance per action are fixed assumptions,
        # not measured per-shipment data. Emission factors are literature values.
        carbon = 0.0
        for step in episode_history:
            action = step.get("action", "do_nothing")
            if action == "expedite_order":
                mode = step.get("expedite_mode", "air")
                # Illustrative assumption: 100 tonnes moved 5000km per expedite
                carbon += self.carbon_factors.get(mode, 0.5) * 100 * 5000
            elif action == "reroute_shipment":
                # Illustrative assumption: rerouting adds 3500nm (Red Sea scenario)
                carbon += self.carbon_factors["sea"] * 100 * 3500 * 1.852  # nm to km

        # Normalize carbon to [0, 1]
        carbon_norm = min(1.0, carbon / 500_000)

        return np.array([cost, resilience_loss, carbon_norm])


def generate_weight_grid(n_points: int = 20) -> list[np.ndarray]:
    """Generate n_points weight vectors on the 3-simplex.

    Each vector w = [w_cost, w_resilience, w_carbon] with sum(w) = 1.
    """
    weights = []
    # Uniform grid on 2-simplex
    steps = int(np.ceil(np.sqrt(n_points)))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w = np.array([i, j, k], dtype=np.float64)
            w = w / max(w.sum(), 1e-6)
            weights.append(w)
            if len(weights) >= n_points:
                return weights
    return weights[:n_points]


def scalarize(objectives: np.ndarray, weights: np.ndarray) -> float:
    """Weighted sum scalarization of objective vector."""
    return float(np.dot(objectives, weights))


def compute_pareto_front(
    all_objectives: np.ndarray,
) -> np.ndarray:
    """Find non-dominated solutions (Pareto front).

    Args:
        all_objectives: (n_solutions, 3) array — all to minimize.

    Returns:
        Boolean mask: True for Pareto-optimal solutions.
    """
    n = len(all_objectives)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i in all objectives and j < i in at least one
            if np.all(all_objectives[j] <= all_objectives[i]) and np.any(all_objectives[j] < all_objectives[i]):
                is_pareto[i] = False
                break

    return is_pareto


def run_scripted_seed_study(
    n_seeds: int = 20,
    task_id: str = "easy_typhoon_response",
    seed: int = 42,
) -> dict[str, Any]:
    """Run the scripted agent under n_seeds different env seeds and score objectives.

    The recorded weight vectors do NOT condition the agent — they are kept only
    so the same grid can later drive genuinely weight-conditioned policies. This
    is a seed study of one fixed policy, not multi-policy training.

    Returns Pareto front data for visualization.
    """
    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action

    weights_grid = generate_weight_grid(n_seeds)
    objectives_module = ParetoObjectives()

    all_objectives = []
    all_weights = []

    env = SupplyMindEnvironment()

    for w_idx, weights in enumerate(weights_grid):
        # Same scripted agent every run; only the env seed changes. Weights are
        # recorded but do not influence behavior.
        obs = env.reset(task_id=task_id, seed=seed + w_idx)
        history = []
        step = 0

        while not obs.done:
            action = choose_action(obs, step)
            history.append({
                "action": action.action_type,
                "expedite_mode": getattr(action, "expedite_mode", None),
            })
            obs = env.step(action)
            step += 1

        grade = env.grade()
        financials = {
            "cumulative_cost_incurred": obs.financials.cumulative_cost_incurred,
            "score": grade["score"],
        }

        obj = objectives_module.evaluate(history, financials, obs.financials.budget_total)
        all_objectives.append(obj)
        all_weights.append(weights)

        if (w_idx + 1) % 5 == 0:
            logger.info("  Seed run %d/%d: cost=%.3f resilience_loss=%.3f carbon=%.3f",
                        w_idx + 1, n_seeds, obj[0], obj[1], obj[2])

    all_objectives = np.array(all_objectives)
    pareto_mask = compute_pareto_front(all_objectives)

    result = {
        "study_type": "scripted_agent_seed_study",
        "policy": "scripted_agent (single fixed policy across all seeds)",
        "n_seeds": n_seeds,
        "weights_condition_behavior": False,
        "all_objectives": all_objectives.tolist(),
        "all_weights": [w.tolist() for w in all_weights],
        "pareto_mask": pareto_mask.tolist(),
        "n_pareto": int(pareto_mask.sum()),
        "objective_names": ["cost", "resilience_loss", "carbon"],
        "carbon": {
            "assumption": True,
            "note": "Carbon objective uses fixed illustrative tonnage/distance per "
                    "action, not per-shipment data.",
        },
    }

    # Save
    import json
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHECKPOINT_DIR / "pareto_results.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Pareto front: %d/%d seed runs are non-dominated", result["n_pareto"], n_seeds)

    return result
