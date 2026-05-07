"""compute_panel_agreement.py — Krippendorff α + majority accuracy on
the Frontier Judge Panel v2 results.

Reads every per-scenario per-model verdict cached by
scripts/run_frontier_judge_panel.py, combines with the 3 local judges from
R4_DANGEROUS_V2.json, and computes the *real* ordinal-Krippendorff α across
the expanded panel. This replaces the README's previous "α=0.750" claim —
which was actually mean_conf, not α — with a defensible number.

Output: tests/receipts/frontier_panel_alpha.json.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
R4_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json"
PANEL_JSON = ROOT / "v3_arcadia" / "results" / "R4_FRONTIER_PANEL_V2.json"
CACHE_DIR = ROOT / ".openrouter_cache"
RECEIPT = ROOT / "tests" / "receipts" / "frontier_panel_alpha.json"

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _ordinal_distance(a: str, b: str) -> float:
    """Squared-difference ordinal distance metric for Krippendorff α."""
    ia, ib = RISK_ORDER.get(a, -1), RISK_ORDER.get(b, -1)
    if ia < 0 or ib < 0:
        return 0.0
    return (ia - ib) ** 2


def _krippendorff_alpha_ordinal(table: dict[str, dict[str, str]]) -> float:
    """
    table[scenario_id][judge_id] = risk_level in {LOW,MEDIUM,HIGH,CRITICAL}.
    Implements Krippendorff's α with ordinal difference (squared-distance).

    Reference: Krippendorff 2004 "Content Analysis" — the canonical form.
    """
    # Pairs observed: for each scenario, every unordered pair of judges that
    # both answered valid-tier.
    D_o = 0.0; n_o = 0
    value_counts: dict[str, int] = {}
    for scen, judges in table.items():
        valid = [v for v in judges.values() if v in RISK_ORDER]
        for v in valid:
            value_counts[v] = value_counts.get(v, 0) + 1
        for a, b in combinations(valid, 2):
            D_o += _ordinal_distance(a, b)
            n_o += 1
    # Expected pairwise distance across the whole dataset
    values = list(value_counts.keys())
    total = sum(value_counts.values())
    D_e = 0.0; n_e = 0
    for i, v1 in enumerate(values):
        for v2 in values[i:]:
            n1 = value_counts[v1]; n2 = value_counts[v2]
            pairs = (n1 * (n1 - 1) // 2) if v1 == v2 else (n1 * n2)
            D_e += _ordinal_distance(v1, v2) * pairs
            n_e += pairs
    if n_o == 0 or n_e == 0 or D_e == 0:
        return 0.0
    obs = D_o / n_o
    exp = D_e / n_e
    return round(1.0 - (obs / exp), 4)


def _load_local_r4() -> dict[str, dict[str, str]]:
    """local_r4[scenario_id][judge_slug] = risk_level"""
    r4 = json.loads(R4_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for sid, scen in r4.get("per_scenario", {}).items():
        verdicts: dict[str, str] = {}
        for judge_id, body in (scen.get("per_judge") or {}).items():
            parsed = (body.get("parsed") if isinstance(body, dict) else {}) or {}
            v = str(parsed.get("risk_level", "")).upper()
            if v in RISK_ORDER:
                verdicts[f"local:{judge_id}"] = v
        if verdicts:
            out[sid] = verdicts
    return out


def _load_frontier_cache() -> dict[str, dict[str, str]]:
    """frontier[scenario_id][model_slug] = risk_level from cached panel calls."""
    out: dict[str, dict[str, str]] = {}
    if not CACHE_DIR.exists():
        return out
    for model_dir in CACHE_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name.replace("__", "/").replace("_free", ":free")
        for f in model_dir.glob("*.json"):
            try:
                row = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sid = row.get("scenario_id") or f.stem
            pred = str(row.get("predicted_risk", "")).upper()
            if pred in RISK_ORDER:
                out.setdefault(sid, {})[f"frontier:{model_slug}"] = pred
    return out


def _ground_truth_map() -> dict[str, str]:
    r4 = json.loads(R4_PATH.read_text(encoding="utf-8"))
    return {sid: str(s.get("ground_truth", "")).upper()
            for sid, s in r4.get("per_scenario", {}).items()}


def main() -> None:
    local = _load_local_r4()
    frontier = _load_frontier_cache()
    gt = _ground_truth_map()

    # Combined table: every judge × every scenario
    combined: dict[str, dict[str, str]] = {}
    for sid in set(local) | set(frontier):
        merged = {}
        merged.update(local.get(sid, {}))
        merged.update(frontier.get(sid, {}))
        if len(merged) >= 2:
            combined[sid] = merged

    if not combined:
        print("{}"); return

    alpha_local = _krippendorff_alpha_ordinal(local)
    alpha_frontier = _krippendorff_alpha_ordinal(frontier)
    alpha_combined = _krippendorff_alpha_ordinal(combined)

    # Majority vote accuracy vs ground truth
    def _majority_acc(table: dict[str, dict[str, str]]) -> tuple[float, int]:
        hits = 0; seen = 0
        for sid, judges in table.items():
            if not judges or sid not in gt:
                continue
            tallies: dict[str, int] = {}
            for v in judges.values():
                tallies[v] = tallies.get(v, 0) + 1
            maj = max(tallies, key=tallies.get)
            if maj == gt[sid]:
                hits += 1
            seen += 1
        return round(hits / max(1, seen), 4), seen

    acc_local, n_local = _majority_acc(local)
    acc_frontier, n_frontier = _majority_acc(frontier)
    acc_combined, n_combined = _majority_acc(combined)

    # Judge inventory
    judges_local: set[str] = set()
    judges_frontier: set[str] = set()
    for sid, judges in local.items():
        judges_local.update(judges.keys())
    for sid, judges in frontier.items():
        judges_frontier.update(judges.keys())

    receipt = {
        "summary": {
            "n_judges_local": len(judges_local),
            "n_judges_frontier": len(judges_frontier),
            "n_judges_total": len(judges_local) + len(judges_frontier),
            "n_scenarios": {"local": n_local, "frontier": n_frontier,
                             "combined": n_combined},
            "krippendorff_alpha_ordinal": {
                "local_only": alpha_local,
                "frontier_only": alpha_frontier,
                "combined_local_plus_frontier": alpha_combined,
            },
            "majority_vote_accuracy_vs_ground_truth": {
                "local_only": acc_local,
                "frontier_only": acc_frontier,
                "combined_local_plus_frontier": acc_combined,
            },
        },
        "judges_local": sorted(judges_local),
        "judges_frontier": sorted(judges_frontier),
        "reward_scale": "ordinal 4-tier: LOW=0, MEDIUM=1, HIGH=2, CRITICAL=3",
        "distance_metric": "squared-difference",
        "ground_truth_source": "versions/v3_arcadia/results/R4_DANGEROUS_V2.json per_scenario.*.ground_truth",
        "frontier_judge_source": "OpenRouter chat/completions (cached in .openrouter_cache/)",
        "inference_type": "live_http_multi_provider_panel",
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
