"""leaderboard.py — maintain the OpenEnv Arena leaderboard.

Reads every ArenaResult JSON from `ShAuRyA_Phoenix/experiments/arena/*.json`,
sorts by overall_ci95_lower (conservative ranking), writes a single
`leaderboard.json` consumed by the Gradio page and the /arena/leaderboard
endpoint.

Pre-populated baselines (injected on first call):
    - random          : random valid action sampler
    - greedy          : greedy lowest-cost action
    - MaskablePPO-v3  : our R6 Gethsemane policy
    - PPO-v3          : R6 ablation baseline without masking
    - A2C-v3          : R6 algorithm comparison
    - RecurrentPPO-v3 : R6 algorithm comparison
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ARENA_DIR = Path(__file__).resolve().parents[1] / "experiments" / "arena"
LEADERBOARD_PATH = ARENA_DIR / "leaderboard.json"


# From v3_arcadia/results/R6_EUCLIDIAN.json (10,800-episode benchmark).
# These are the pre-seeded baseline rows so the leaderboard is useful even
# before any judge uploads a policy.
BASELINES = [
    {"policy_name": "MaskablePPO-v3 (ours)", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": 2.209, "overall_ci95": [2.178, 2.239], "total_violations": 0,
     "source": "v3_arcadia/results/R6_EUCLIDIAN.json (3 tasks x 900 eps)"},
    {"policy_name": "RecurrentPPO-v3", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": 1.081, "overall_ci95": [0.98, 1.18], "total_violations": 14.9,
     "source": "v3_arcadia/results/R6_ALGO_COMPARISON.json (easy only)"},
    {"policy_name": "PPO-v3 (no masking)", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": 0.947, "overall_ci95": [0.89, 1.01], "total_violations": 13.6,
     "source": "R6 masking ablation baseline"},
    {"policy_name": "A2C-v3", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": 0.874, "overall_ci95": [0.81, 0.94], "total_violations": 13.9,
     "source": "R6 algo comparison"},
    {"policy_name": "Greedy (baseline)", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": -0.749, "overall_ci95": [-0.76, -0.74], "total_violations": 0,
     "source": "R6 Euclidian baseline"},
    {"policy_name": "Random (baseline)", "submitted_at": "2026-04-18T00:00:00Z",
     "overall_reward_mean": -0.511, "overall_ci95": [-0.55, -0.47], "total_violations": 0,
     "source": "R6 Euclidian baseline"},
]


def rebuild() -> dict:
    """Merge submitted ArenaResult files with baselines; sort + write."""
    ARENA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = list(BASELINES)
    for f in ARENA_DIR.glob("*.json"):
        if f.name == "leaderboard.json":
            continue
        try:
            blob = json.loads(f.read_text())
            if "overall_reward_mean" in blob:
                rows.append({
                    "policy_name": blob["policy_name"],
                    "submitted_at": blob["submitted_at"],
                    "overall_reward_mean": blob["overall_reward_mean"],
                    "overall_ci95": blob.get("overall_ci95", [None, None]),
                    "total_violations": blob.get("total_violations", 0),
                    "source": f"/arena submission: {f.name}",
                })
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", f, e)

    # Rank by CI95 lower (conservative)
    def _key(r):
        ci = r.get("overall_ci95") or [None, None]
        return ci[0] if ci and ci[0] is not None else r.get("overall_reward_mean", float("-inf"))
    rows.sort(key=_key, reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    board = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_submissions": len(rows) - len(BASELINES),
        "n_baselines": len(BASELINES),
        "rows": rows,
    }
    try:
        LEADERBOARD_PATH.write_text(json.dumps(board, indent=2), encoding="utf-8")
    except PermissionError as e:
        logger.warning("leaderboard rebuild computed but could not rewrite %s: %s", LEADERBOARD_PATH, e)
    return board


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    b = rebuild()
    print(f"[leaderboard] {b['n_submissions']} submissions + {b['n_baselines']} baselines = {len(b['rows'])} rows")
    for r in b["rows"][:10]:
        print(f"  {r['rank']:2d}. {r['policy_name']:40s} mean={r['overall_reward_mean']:+.3f} ci95={r['overall_ci95']}")
