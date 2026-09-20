"""leaderboard.py — maintain the OpenEnv Arena leaderboard.

Reads every judge-submitted ArenaResult JSON from
`versions/v5_phoenix/experiments/arena/*.json`, merges the pre-seeded baseline
rows, sorts by overall_ci95_lower (conservative ranking), and writes a single
`leaderboard.json` consumed by the Gradio page and the /arena/leaderboard
endpoint.

Baseline rows are DERIVED AT RUNTIME from committed R6 benchmark artifacts —
never hardcoded as constants:

    - Random / Greedy / MaskablePPO-v3 : equal-weight mean of the three per-task
      reward_means in versions/v3_arcadia/results/R6_EUCLIDIAN.json (900 eps per
      task, seed 42). Aggregate CI95 comes from standard-error propagation of
      the three per-task 95% CIs.
    - PPO-v3 (no masking) / A2C-v3 / RecurrentPPO-v3 : easy-only rows from
      versions/v3_arcadia/results/R6_ALGO_COMPARISON.json (50 eps). CI95 from the
      reported reward_std over n_episodes.

If either source artifact is missing, the leaderboard FAILS LOUD: it serves only
the baselines it could actually derive, sets `degraded=true`, and reports which
artifacts were absent — it never falls back to a hardcoded table.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
ARENA_DIR = ROOT / "versions" / "v5_phoenix" / "experiments" / "arena"
LEADERBOARD_PATH = ARENA_DIR / "leaderboard.json"

R6_EUCLIDIAN = ROOT / "versions" / "v3_arcadia" / "results" / "R6_EUCLIDIAN.json"
R6_ALGO_COMPARISON = ROOT / "versions" / "v3_arcadia" / "results" / "R6_ALGO_COMPARISON.json"

Z95 = 1.959963984540054  # standard-normal quantile for a 95% two-sided interval

# Which R6_EUCLIDIAN agent key maps to which leaderboard display name.
_EUCLIDIAN_POLICIES = {
    "ppo_v3": "MaskablePPO-v3 (ours)",
    "random": "Random (baseline)",
    "greedy": "Greedy (baseline)",
}
# Which R6_ALGO_COMPARISON algorithm maps to which display name. MaskablePPO is
# deliberately excluded here — it is already represented by the 3-task Euclidian
# row above.
_ALGO_POLICIES = {
    "PPO": "PPO-v3 (no masking)",
    "A2C": "A2C-v3",
    "RecurrentPPO": "RecurrentPPO-v3",
}


def _iso_from_mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))


def _euclidian_baselines(path: Path) -> list[dict]:
    """Derive Random/Greedy/MaskablePPO rows from the 3-task Euclidian sweep."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    tasks = blob["tasks"]
    submitted = _iso_from_mtime(path)
    rows: list[dict] = []
    for key, name in _EUCLIDIAN_POLICIES.items():
        means, ses, violations = [], [], 0.0
        for task_data in tasks.values():
            cell = task_data[key]
            means.append(float(cell["reward_mean"]))
            lo, hi = cell["reward_ci95"]
            ses.append((float(hi) - float(lo)) / (2.0 * Z95))
            violations += float(cell.get("violations_mean", 0.0))
        n = len(means)
        overall = sum(means) / n
        se_overall = math.sqrt(sum(s * s for s in ses)) / n
        half = Z95 * se_overall
        rows.append({
            "policy_name": name,
            "submitted_at": submitted,
            "overall_reward_mean": round(overall, 4),
            "overall_ci95": [round(overall - half, 4), round(overall + half, 4)],
            "total_violations": round(violations, 4),
            "source": (f"versions/v3_arcadia/results/R6_EUCLIDIAN.json "
                       f"({key}, mean of {n} tasks x "
                       f"{blob.get('config', {}).get('episodes_per_cell', '?')} eps)"),
        })
    return rows


def _algo_comparison_baselines(path: Path) -> list[dict]:
    """Derive the masking-ablation rows (easy-only) from the algo comparison."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    per_algo = blob["per_algorithm"]
    n_eps = int(blob.get("eval_episodes", 50))
    submitted = _iso_from_mtime(path)
    rows: list[dict] = []
    for key, name in _ALGO_POLICIES.items():
        cell = per_algo[key]
        mean = float(cell["reward_mean"])
        se = float(cell["reward_std"]) / math.sqrt(cell.get("n_episodes", n_eps))
        half = Z95 * se
        rows.append({
            "policy_name": name,
            "submitted_at": submitted,
            "overall_reward_mean": round(mean, 4),
            "overall_ci95": [round(mean - half, 4), round(mean + half, 4)],
            "total_violations": round(float(cell.get("invalid_action_picks_mean_per_ep", 0.0)), 4),
            "source": (f"versions/v3_arcadia/results/R6_ALGO_COMPARISON.json "
                       f"({key}, {blob.get('task', 'easy')}, {cell.get('n_episodes', n_eps)} eps)"),
        })
    return rows


def derive_baselines() -> tuple[list[dict], list[str]]:
    """Return (baseline_rows, missing_sources). Never hardcodes numbers."""
    rows: list[dict] = []
    missing: list[str] = []
    for path, fn in ((R6_EUCLIDIAN, _euclidian_baselines),
                     (R6_ALGO_COMPARISON, _algo_comparison_baselines)):
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = str(path)
        if not path.exists():
            missing.append(rel)
            logger.error("[leaderboard] baseline artifact MISSING (no fallback): %s", rel)
            continue
        try:
            rows.extend(fn(path))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            missing.append(f"{rel} ({type(e).__name__}: {e})")
            logger.error("[leaderboard] baseline artifact UNREADABLE: %s (%s)", rel, e)
    return rows, missing


def rebuild() -> dict:
    """Merge submitted ArenaResult files with derived baselines; sort + write."""
    ARENA_DIR.mkdir(parents=True, exist_ok=True)
    baselines, missing = derive_baselines()
    rows: list[dict] = list(baselines)
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
        "n_submissions": len(rows) - len(baselines),
        "n_baselines": len(baselines),
        "degraded": bool(missing),
        "degraded_reason": (
            "missing baseline artifacts: " + "; ".join(missing) if missing else None
        ),
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
    print(f"[leaderboard] {b['n_submissions']} submissions + {b['n_baselines']} baselines "
          f"= {len(b['rows'])} rows  degraded={b['degraded']}")
    if b["degraded"]:
        print(f"[leaderboard] DEGRADED: {b['degraded_reason']}")
    for r in b["rows"][:10]:
        print(f"  {r['rank']:2d}. {r['policy_name']:40s} mean={r['overall_reward_mean']:+.3f} "
              f"ci95={r['overall_ci95']}")
