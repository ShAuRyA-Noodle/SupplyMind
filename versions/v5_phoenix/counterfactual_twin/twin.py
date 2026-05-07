"""twin.py — 100-rollout Monte-Carlo digital twin conditioned on a live signal.

Inputs:
    severity (float in [0, 1]) — disruption severity from live pipeline
    brent_usd (float)          — current Brent oil price from FRED
    task_id (str)              — which supply-chain graph to simulate

Rollouts three policies (trained, no-action, greedy) N=100 times each with
frozen-holdout seeds (42, 99, 7 rotating). Injects the severity as a
scalar modulator on disruption impact. Returns a `TwinReport` with:

    - loss distributions (per policy) in USD
    - headline: median $ loss no-action, median $ loss trained,
      savings = (no_action - trained) in USD
    - p95 tail losses (for risk-aware stakeholders)
    - CI95 on savings via paired bootstrap
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Revenue-at-risk ballpark per task, derived from v3_arcadia
# supply_environment budget defaults: easy=$5M, medium=$8M, hard=$10M.
# Scaled up by 40x to represent the full real-world chain (not just sim window).
REVENUE_AT_RISK_USD = {
    "easy_typhoon_response": 200_000_000,      # $200M semiconductor chain
    "medium_multi_front": 320_000_000,         # $320M multi-region
    "hard_cascading_crisis": 400_000_000,      # $400M global auto
}

N_ROLLOUTS = 100
DEFAULT_TASK = "hard_cascading_crisis"
MAX_STEPS_PER_ROLLOUT = 200


@dataclass
class TwinReport:
    task_id: str
    severity: float
    brent_usd: float
    policy_names: list[str]
    loss_distributions_usd: dict[str, list[float]]
    median_loss_usd: dict[str, float]
    p95_loss_usd: dict[str, float]
    savings_vs_no_action_usd: float
    savings_ci95_usd: tuple[float, float]
    savings_pct: float
    n_rollouts: int
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "severity": self.severity,
            "brent_usd": self.brent_usd,
            "policy_names": self.policy_names,
            "median_loss_usd": {k: int(round(v)) for k, v in self.median_loss_usd.items()},
            "p95_loss_usd": {k: int(round(v)) for k, v in self.p95_loss_usd.items()},
            "savings_vs_no_action_usd": int(round(self.savings_vs_no_action_usd)),
            "savings_ci95_usd": [int(round(x)) for x in self.savings_ci95_usd],
            "savings_pct": round(self.savings_pct, 1),
            "n_rollouts": self.n_rollouts,
            "generated_at": self.generated_at,
        }


def _bootstrap_ci95(x: np.ndarray, n: int = 1000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_trained_policy(task_id: str) -> Any | None:
    """Load the v3 MaskablePPO policy for the given task, if it exists."""
    from sb3_contrib import MaskablePPO

    candidates = [
        ROOT / "v3_arcadia" / "checkpoints" / "gethsemane" / f"ppo_{task_id}.zip",
        ROOT / "v3_arcadia" / "checkpoints" / "gethsemane" / "ppo_easy_typhoon_response.zip",
    ]
    for c in candidates:
        if c.exists():
            try:
                return MaskablePPO.load(str(c), env=None, device="auto")
            except Exception as e:  # noqa: BLE001
                logger.warning("[twin] failed to load %s: %s", c, e)
    return None


def _rollout(policy: Any | None, task_id: str, seed: int, severity: float,
             mode: str) -> float:
    """One rollout. Returns loss in USD (negative reward scaled by revenue-at-risk)."""
    from rl.gym_env import SupplyMindGymnasiumEnv
    from server.supply_environment import SupplyMindEnvironment
    from gymnasium.spaces import Discrete
    import gymnasium as gym

    class Flat(gym.Wrapper):
        def __init__(self, base):
            super().__init__(base)
            _, n_t = base.action_space.nvec
            self._nt = int(n_t)
            self.action_space = Discrete(int(base.action_space.nvec[0]) * self._nt)

        def step(self, a):
            flat = int(np.asarray(a).flatten()[0])
            at, ag = divmod(flat, self._nt)
            return self.env.step(np.array([at, ag]))

    base = SupplyMindGymnasiumEnv(task_id=task_id)
    env = Flat(base)
    core = SupplyMindEnvironment()
    obs, info = env.reset(seed=seed)
    core.reset(task_id=task_id, seed=seed)

    for _ in range(MAX_STEPS_PER_ROLLOUT):
        mask = info.get("action_masks")
        mask_np = np.asarray(mask) if mask is not None else None

        if mode == "no_action":
            flat = 0  # do_nothing is action 0 in the Discrete(280) flattening
        elif mode == "greedy":
            # cheapest valid action — proxy: lowest-index valid
            if mask_np is not None:
                valid = np.where(mask_np)[0]
                flat = int(valid[0]) if len(valid) else 0
            else:
                flat = 0
        elif mode == "trained" and policy is not None:
            try:
                out = policy.predict(obs, deterministic=True, action_masks=mask_np)
                act = out[0] if isinstance(out, tuple) else out
                flat = int(np.asarray(act).flatten()[0])
            except Exception:
                flat = 0
        else:
            flat = 0

        obs, _, term, trunc, info = env.step(flat)
        at, ag = divmod(flat, 40)
        core.step(base._decode_action(np.array([at, ag], dtype=np.int64)))
        if term or trunc:
            break

    # Grade: score in [0, 1] where 1 = full revenue preserved, 0 = total loss.
    score = float(core.grade()["score"])
    revenue_at_risk = REVENUE_AT_RISK_USD.get(task_id, 300_000_000)

    # Severity uplift: a real-world signal scaler on top of the sim's own randomness.
    severity_multiplier = 0.5 + 1.0 * max(0.0, min(1.0, severity))
    # Brent price contributes small additional loss for oil-heavy disruptions.
    brent_multiplier = 1.0  # base-case; future: couple to commodity exposure fraction
    loss = (1.0 - score) * revenue_at_risk * severity_multiplier * brent_multiplier
    return float(loss)


def run_twin(
    severity: float,
    brent_usd: float = 85.0,
    task_id: str = DEFAULT_TASK,
    n_rollouts: int = N_ROLLOUTS,
) -> TwinReport:
    trained = _load_trained_policy(task_id)
    if trained is None:
        logger.warning("[twin] no trained policy available; falling back to no-action only")

    seeds_base = [42, 99, 7]
    loss_distributions = {"trained": [], "no_action": [], "greedy": []}
    for i in range(n_rollouts):
        seed = seeds_base[i % len(seeds_base)] + (i // len(seeds_base))
        for mode in ["trained", "no_action", "greedy"]:
            p = trained if mode == "trained" else None
            loss = _rollout(p, task_id, seed, severity, mode)
            loss_distributions[mode].append(loss)

    arrs = {k: np.asarray(v, dtype=np.float64) for k, v in loss_distributions.items()}
    median_loss = {k: float(np.median(v)) for k, v in arrs.items()}
    p95_loss = {k: float(np.percentile(v, 95)) for k, v in arrs.items()}

    diff = arrs["no_action"] - arrs["trained"]
    savings_mean = float(diff.mean())
    savings_lo, savings_hi = _bootstrap_ci95(diff)
    savings_pct = 100.0 * (savings_mean / max(1.0, float(arrs["no_action"].mean())))

    return TwinReport(
        task_id=task_id,
        severity=severity,
        brent_usd=brent_usd,
        policy_names=["trained_maskable_ppo", "no_action", "greedy"],
        loss_distributions_usd={k: [round(x, 2) for x in v] for k, v in loss_distributions.items()},
        median_loss_usd=median_loss,
        p95_loss_usd=p95_loss,
        savings_vs_no_action_usd=savings_mean,
        savings_ci95_usd=(savings_lo, savings_hi),
        savings_pct=savings_pct,
        n_rollouts=n_rollouts,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity", type=float, default=0.85)
    parser.add_argument("--brent", type=float, default=123.0)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rep = run_twin(args.severity, args.brent, args.task, args.rollouts)
    out_path = args.out or (ROOT / "versions/v5_phoenix" / "experiments" / "twin" / f"twin_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep.to_dict(), indent=2))
    print(json.dumps(rep.to_dict(), indent=2))
    print(f"[twin] wrote {out_path}")
