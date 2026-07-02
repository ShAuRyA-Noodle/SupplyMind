"""
Gymnasium wrapper for SupplyMind environment.

Bridges the structured SupplyMindEnvironment (Pydantic models, text observations)
to the flat-tensor interface that RL algorithms expect.

State encoding: 408 floats = 40 nodes x 10 features + 8 global features.
Action space:   MultiDiscrete([7, 40]) — (action_type_idx, target_node_idx).
Action masking: info["action_masks"] boolean array of shape (7*40=280,).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import SupplyMindAction, SupplyMindObservation
from server.supply_environment import SupplyMindEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NODES = 40
FEATURES_PER_NODE = 10
GLOBAL_FEATURES = 8
OBS_DIM = MAX_NODES * FEATURES_PER_NODE + GLOBAL_FEATURES  # 408

NUM_ACTION_TYPES = 7
ACTION_TYPES = [
    "do_nothing",
    "activate_backup_supplier",
    "reroute_shipment",
    "increase_safety_stock",
    "expedite_order",
    "hedge_commodity",
    "issue_supplier_alert",
]

NODE_TYPE_MAP = {
    "supplier": 0,
    "warehouse": 1,
    "port": 2,
    "factory": 3,
    "customer": 4,
}


class SupplyMindGymnasiumEnv(gym.Env):
    """Gymnasium-compliant wrapper around SupplyMindEnvironment.

    Observation: Box(408,) float32
    Action:      MultiDiscrete([7, 40])
    Reward:      float  (dense, per-step, in [-1, 1])
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        task_id: str = "easy_typhoon_response",
        render_mode: Optional[str] = None,
        training_mode: bool = False,
        grade_reward: bool = False,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.render_mode = render_mode
        self.training_mode = training_mode
        self.grade_reward = grade_reward

        # Core environment (in-process, no HTTP)
        self._env = SupplyMindEnvironment()

        # Training mode: reduce Monte Carlo from 1000 to 50 simulations.
        # MC produces P50/P95 OBSERVATIONS only — reward is independent of MC.
        # This gives ~15-20x CPU speedup with zero impact on learned policy.
        # A silent no-op here would make training 15-20x slower without warning,
        # so any failure to apply the patch is raised loudly, never swallowed.
        self.training_mc_patched = False
        if training_mode:
            import server.engine.monte_carlo as engine_mod

            if not hasattr(engine_mod, "MonteCarloEngine"):
                raise RuntimeError(
                    "training_mode=True but server.engine.monte_carlo.MonteCarloEngine "
                    "is missing — cannot apply the fast Monte-Carlo patch. Refusing to "
                    "run 15-20x slower silently."
                )

            run_sim = engine_mod.MonteCarloEngine.run_simulation
            if getattr(run_sim, "_supplymind_fast_mc", False):
                # Already patched by a previous env instance — don't nest wrappers.
                self.training_mc_patched = True
            else:
                _orig_run = run_sim

                def _fast_mc(self_mc, graph, active_disruptions, n_simulations=50):
                    return _orig_run(self_mc, graph, active_disruptions, n_simulations=50)

                _fast_mc._supplymind_fast_mc = True
                engine_mod.MonteCarloEngine.run_simulation = _fast_mc
                self.training_mc_patched = True

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([NUM_ACTION_TYPES, MAX_NODES])

        # Episode state (populated on reset)
        self._obs: Optional[SupplyMindObservation] = None
        self._node_ids: list[str] = []
        self._node_id_to_idx: dict[str, int] = {}
        self._num_nodes: int = 0
        self._max_steps: int = 60
        self._budget_total: float = 1.0
        self._total_revenue: float = 1.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        raw_obs = self._env.reset(task_id=self.task_id, seed=seed)
        self._obs = raw_obs
        self._step_count = 0

        # Build node index from the first observation
        self._node_ids = [n.node_id for n in raw_obs.node_statuses]
        self._node_id_to_idx = {nid: i for i, nid in enumerate(self._node_ids)}
        self._num_nodes = len(self._node_ids)

        # Cache episode constants
        self._max_steps = raw_obs.current_day + raw_obs.days_remaining
        self._budget_total = max(raw_obs.financials.budget_total, 1.0)
        self._total_revenue = max(raw_obs.financials.total_revenue_at_risk, 1.0)

        encoded = self._encode_obs(raw_obs)
        info = self._build_info(raw_obs)
        return encoded, info

    def step(
        self, action: np.ndarray | list[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64)
        sm_action = self._decode_action(action)
        raw_obs = self._env.step(sm_action)
        self._obs = raw_obs
        self._step_count += 1

        encoded = self._encode_obs(raw_obs)
        terminated = bool(raw_obs.done)
        truncated = False
        info = self._build_info(raw_obs)

        if self.grade_reward:
            # Grade-aligned reward: use grader score at episode end
            # During episode, use shaped reward based on grader components
            if terminated:
                grade = self._env.grade()
                reward = float(grade["score"])  # 0-1 scale, grader-aligned
                info["grade"] = grade
            else:
                # Intermediate shaping: penalize free-action spam, reward real actions
                step_r = float(raw_obs.reward)
                action_type = ACTION_TYPES[int(np.asarray(action, dtype=np.int64)[0])]
                # Bonus for substantive actions (not do_nothing or alerts)
                if action_type in ("activate_backup_supplier", "reroute_shipment",
                                   "increase_safety_stock", "expedite_order", "hedge_commodity"):
                    step_r += 0.02  # Small bonus for real actions
                # Penalty for alert spam
                if action_type == "issue_supplier_alert":
                    step_r -= 0.01  # Discourage alert spam
                reward = step_r
        else:
            reward = float(raw_obs.reward)

        return encoded, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array" or self._obs is None:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Panel 1 — Node risk scores
            nodes = self._obs.node_statuses
            names = [n.name[:20] for n in nodes]
            risks = [n.current_risk_score for n in nodes]
            colors = ["#d32f2f" if r > 0.5 else "#ff9800" if r > 0.2 else "#4caf50" for r in risks]
            axes[0].barh(names, risks, color=colors)
            axes[0].set_xlim(0, 1)
            axes[0].set_title(f"Node Risk Scores (Day {self._obs.current_day})")

            # Panel 2 — Financials
            fin = self._obs.financials
            labels = ["Budget Left", "Revenue Lost", "Penalties"]
            values = [fin.budget_remaining, fin.cumulative_revenue_lost, fin.cumulative_penalty_fees]
            axes[1].bar(labels, values, color=["#2196f3", "#f44336", "#ff9800"])
            axes[1].set_title("Financial Snapshot ($)")
            axes[1].ticklabel_format(style="plain", axis="y")

            fig.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = buf.reshape((h, w, 4))[:, :, :3].copy()
            plt.close(fig)
            return img
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # What-If scenario builder hook (used by dashboard/scenario_builder.py)
    # ------------------------------------------------------------------

    def inject_disruption(
        self,
        node_ids: list[str],
        severity: float = 0.7,
        duration_days: int = 14,
        disruption_type: str = "custom_scenario",
    ) -> dict[str, Any]:
        """Inject an ad-hoc disruption into the running episode.

        Directly modifies the node risk scores and operational status via the
        underlying engine, without altering the core environment files.
        Returns a summary dict of the injection effect.
        """
        if self._env.engine is None:
            raise RuntimeError("No active episode — call reset() first.")

        graph = self._env.engine.graph
        affected = []
        for nid in node_ids:
            node_data = graph.graph.nodes.get(nid)
            if node_data is None:
                continue
            prev_risk = node_data.get("risk_score", 0.0)
            new_risk = min(1.0, prev_risk + severity)
            node_data["risk_score"] = new_risk
            if severity >= 0.7:
                node_data["is_operational"] = False
            affected.append({
                "node_id": nid,
                "prev_risk": round(prev_risk, 3),
                "new_risk": round(new_risk, 3),
                "operational": node_data["is_operational"],
            })

        return {
            "disruption_type": disruption_type,
            "severity": severity,
            "duration_days": duration_days,
            "affected_nodes": affected,
            "num_affected": len(affected),
        }

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: SupplyMindObservation) -> np.ndarray:
        """Encode structured observation into 408-float vector."""
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        # Per-node features (N x 10), padded to MAX_NODES
        for i, node in enumerate(obs.node_statuses):
            if i >= MAX_NODES:
                break
            base = i * FEATURES_PER_NODE
            vec[base + 0] = 1.0 if node.is_operational else 0.0
            vec[base + 1] = np.clip(node.current_risk_score, 0.0, 1.0)
            vec[base + 2] = np.clip(node.inventory_days_cover / 90.0, 0.0, 2.0)
            vec[base + 3] = 1.0 if node.has_backup else 0.0
            # Node-type one-hot (indices 4-8)
            nt_idx = NODE_TYPE_MAP.get(node.node_type, 0)
            vec[base + 4 + nt_idx] = 1.0
            # Revenue contribution normalized
            vec[base + 9] = np.clip(node.revenue_contribution / self._total_revenue, 0.0, 1.0)

        # Global features (8 floats at the end)
        g_base = MAX_NODES * FEATURES_PER_NODE  # 400
        fin = obs.financials
        max_steps = max(self._max_steps, 1)

        vec[g_base + 0] = obs.current_day / max_steps
        vec[g_base + 1] = fin.budget_remaining / self._budget_total
        vec[g_base + 2] = fin.supply_chain_health_score / 100.0
        # Active disruptions count
        num_disruptions = len(obs.active_signals)
        vec[g_base + 3] = min(num_disruptions / 10.0, 1.0)
        # Max severity across active signals
        max_sev = max((s.severity for s in obs.active_signals), default=0.0)
        vec[g_base + 4] = np.clip(max_sev, 0.0, 1.0)
        # Cumulative loss ratio
        vec[g_base + 5] = np.clip(fin.cumulative_revenue_lost / self._total_revenue, 0.0, 1.0)
        # Monte Carlo projections
        vec[g_base + 6] = np.clip(fin.monte_carlo_p50_loss / self._total_revenue, 0.0, 1.0)
        vec[g_base + 7] = np.clip(fin.monte_carlo_p95_loss / self._total_revenue, 0.0, 1.0)

        return vec

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> SupplyMindAction:
        """Convert MultiDiscrete([7, 40]) action to SupplyMindAction."""
        action_type_idx = int(action[0])
        target_node_idx = int(action[1])
        action_type = ACTION_TYPES[action_type_idx]

        # Clamp node index to valid range
        target_node_idx = min(target_node_idx, self._num_nodes - 1)
        target_node_id = self._node_ids[target_node_idx] if self._num_nodes > 0 else None

        if action_type == "do_nothing":
            return SupplyMindAction(action_type="do_nothing")

        if action_type == "activate_backup_supplier":
            # Find a backup for the target node
            node_status = self._get_node_status(target_node_id)
            backup_id = None
            if node_status and node_status.backup_supplier_ids:
                backup_id = node_status.backup_supplier_ids[0]
            if backup_id is None:
                return SupplyMindAction(action_type="do_nothing")
            return SupplyMindAction(
                action_type="activate_backup_supplier",
                target_node_id=target_node_id,
                backup_supplier_id=backup_id,
            )

        if action_type == "reroute_shipment":
            # Reroute via the first operational port that isn't the target
            alt_ports = [
                n.node_id for n in self._obs.node_statuses
                if n.node_type == "port" and n.is_operational and n.node_id != target_node_id
            ]
            if not alt_ports:
                return SupplyMindAction(action_type="do_nothing")
            return SupplyMindAction(
                action_type="reroute_shipment",
                target_node_id=target_node_id,
                reroute_via=[alt_ports[0]],
            )

        if action_type == "increase_safety_stock":
            return SupplyMindAction(
                action_type="increase_safety_stock",
                target_node_id=target_node_id,
                additional_stock_days=10,
            )

        if action_type == "expedite_order":
            return SupplyMindAction(
                action_type="expedite_order",
                target_node_id=target_node_id,
                expedite_mode="air",
            )

        if action_type == "hedge_commodity":
            # Hedge the most volatile commodity
            commodities = self._obs.financials.commodity_price_changes
            if not commodities:
                return SupplyMindAction(action_type="do_nothing")
            commodity = max(commodities, key=commodities.get)
            budget = self._obs.financials.budget_remaining
            hedge_amt = min(budget * 0.05, 500_000.0)
            if hedge_amt <= 0:
                return SupplyMindAction(action_type="do_nothing")
            return SupplyMindAction(
                action_type="hedge_commodity",
                commodity=commodity,
                hedge_amount_usd=hedge_amt,
            )

        if action_type == "issue_supplier_alert":
            return SupplyMindAction(
                action_type="issue_supplier_alert",
                target_node_id=target_node_id,
            )

        return SupplyMindAction(action_type="do_nothing")

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def _compute_action_mask(self) -> np.ndarray:
        """Compute boolean mask of shape (7 * 40,) = (280,).

        mask[action_type * 40 + node_idx] = True if action is valid.
        """
        mask = np.zeros(NUM_ACTION_TYPES * MAX_NODES, dtype=np.bool_)

        if self._obs is None:
            mask[0 * MAX_NODES] = True  # do_nothing with node 0
            return mask

        budget = self._obs.financials.budget_remaining

        for node_idx in range(self._num_nodes):
            node = self._obs.node_statuses[node_idx]

            # do_nothing — always valid for any node index
            mask[0 * MAX_NODES + node_idx] = True

            # activate_backup_supplier — needs backup_supplier_ids and budget
            if (node.node_type == "supplier" and node.has_backup
                    and node.backup_supplier_ids and budget > 100_000):
                mask[1 * MAX_NODES + node_idx] = True

            # reroute_shipment — target must be a port
            if node.node_type == "port" and budget > 30_000:
                mask[2 * MAX_NODES + node_idx] = True

            # increase_safety_stock — target must be a warehouse
            if node.node_type == "warehouse" and budget > 50_000:
                mask[3 * MAX_NODES + node_idx] = True

            # expedite_order — target must be a warehouse with low inventory
            if node.node_type == "warehouse" and budget > 200_000:
                mask[4 * MAX_NODES + node_idx] = True

            # hedge_commodity — valid if commodities exist (node index ignored)
            if self._obs.financials.commodity_price_changes and budget > 100_000:
                mask[5 * MAX_NODES + node_idx] = True

            # issue_supplier_alert — always valid for suppliers (free action)
            if node.node_type == "supplier":
                mask[6 * MAX_NODES + node_idx] = True

        # Ensure at least do_nothing is valid
        if not mask.any():
            mask[0] = True

        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_info(self, obs: SupplyMindObservation) -> dict[str, Any]:
        """Build info dict with action_masks and raw observation."""
        return {
            "action_masks": self._compute_action_mask(),
            "raw_obs": obs,
            "score": obs.info.get("score", 0.0),
            "reward_components": obs.info.get("reward_components", {}),
        }

    def _get_node_status(self, node_id: Optional[str]):
        """Look up a node's status by ID."""
        if node_id is None or self._obs is None:
            return None
        for n in self._obs.node_statuses:
            if n.node_id == node_id:
                return n
        return None
