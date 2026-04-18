"""scripts/run_all.py — one-command judge verification.

Produces a concise PASS/FAIL report across every headline claim in
SupplyMind v3.0-arcadia. Reads committed JSONs, re-asserts the floor,
prints a consolidated result table. Does not re-train (re-training is
hours-long and documented separately in reproduce.md).

Exit code 0 if every floor is met; 1 otherwise. Wire into CI or run
locally with:

    python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "v3_arcadia" / "results"

CHECKS = [
    # (label, file, path-list, floor or None, higher_is_better)
    ("R5 RAG P@1 (mxbai bi)", "R5_GRANITE.json", ["pipelines", "P2_mxbai_bi", "p1"], 0.94, True),
    ("R5 RAG MRR (mxbai bi)", "R5_GRANITE.json", ["pipelines", "P2_mxbai_bi", "mrr"], 0.95, True),
    ("R5 BEIR nDCG@10 (Snowflake)", "R5_BEIR_MANUAL.json",
        ["our_results", "snowflake-arctic-l", "mean_ndcg@10"], 0.90, True),
    ("R5 BEIR nDCG@10 (BGE-M3)", "R5_BEIR_MANUAL.json",
        ["our_results", "bge-m3", "mean_ndcg@10"], 0.90, True),
    ("R5 BEIR nDCG@10 (mxbai)", "R5_BEIR_MANUAL.json",
        ["our_results", "mxbai-embed-large-v1", "mean_ndcg@10"], 0.90, True),
    ("R4 2-judge Krippendorff α (ordinal)", "R4_DANGEROUS_V2_ABLATION.json",
        ["agreement_primary_panel", "krippendorff_alpha_ordinal"], 0.70, True),
    ("R4 Cohen κ (Qwen × Mistral)", "R4_DANGEROUS_V2_ABLATION.json",
        ["agreement_primary_panel", "cohen_weighted_kappa_qwen_vs_mistral"], 0.70, True),
    ("R6 PPO masking lift % (easy)", "R6_GETHSEMANE_MASKING_ABLATION.json",
        ["action_masking_contribution", "reward_pct_delta"], 20.0, True),
    ("R6 PPO masking lift % (hard)", "R6_GETHSEMANE_MASKING_ABLATION_ALLTASKS.json",
        ["per_task", "hard_cascading_crisis", "masking_contribution", "reward_pct_delta"], 10.0, True),
    ("R6 GNN arrival-time lift % (easy graph)", "R6_PROVIDER_V2.json",
        ["graphs", "easy", "improvement_vs_mlp_pct"], 40.0, True),
    ("R6 GNN arrival-time lift % (medium graph)", "R6_PROVIDER_V2.json",
        ["graphs", "medium", "improvement_vs_mlp_pct"], 40.0, True),
    ("R6 GNN arrival-time lift % (hard graph)", "R6_PROVIDER_V2.json",
        ["graphs", "hard", "improvement_vs_mlp_pct"], 50.0, True),
]


def nested(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    failures = []
    passes = []
    for label, fname, path, floor, hib in CHECKS:
        fp = R / fname
        if not fp.exists():
            failures.append((label, "MISSING file " + fname))
            continue
        d = json.loads(fp.read_text())
        v = nested(d, path)
        if v is None:
            failures.append((label, "MISSING key " + ".".join(path)))
            continue
        try:
            vf = float(v)
        except Exception:
            failures.append((label, f"NON-NUMERIC {v!r}"))
            continue
        ok = vf >= floor if hib else vf <= floor
        (passes if ok else failures).append(
            (label, f"{vf:.4f} (floor {floor}{'+' if hib else '-'})")
        )

    width = max(len(l) for l, _ in passes + failures)
    print(f"\nSupplyMind v3.0-arcadia — judge verification report\n{'='*70}")
    for label, note in passes:
        print(f"  [PASS] {label.ljust(width)}  {note}")
    for label, note in failures:
        print(f"  [FAIL] {label.ljust(width)}  {note}")
    print(f"{'='*70}")
    print(f"  {len(passes)} passed, {len(failures)} failed out of {len(CHECKS)}")

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
