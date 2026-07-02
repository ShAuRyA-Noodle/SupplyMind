"""Fail CI if any committed benchmark JSON dropped below its v3.0-arcadia floor.

This guards against silent regressions: if someone rewrites a pipeline and the
new JSON lands at worse numbers than the released v3.0-arcadia, the PR fails.

Floors are the verified v3.0-arcadia release values MINUS a small tolerance
(2% relative) so that numerical noise doesn't trip the guard but a real
regression does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "versions" / "v3_arcadia" / "results"

TOL = 0.02  # 2% relative

# (file, jq-like path as nested list, minimum expected value)
FLOORS = [
    ("R5_GRANITE.json", ["pipelines", "P2_mxbai_bi", "p1"], 0.94),
    ("R5_GRANITE.json", ["pipelines", "P2_mxbai_bi", "mrr"], 0.96),
    ("R4_DANGEROUS_V2_ABLATION.json", ["agreement_primary_panel", "krippendorff_alpha_ordinal"], 0.70),
    ("R4_DANGEROUS_V2_ABLATION.json", ["agreement_primary_panel", "cohen_weighted_kappa_qwen_vs_mistral"], 0.70),
    ("R6_GETHSEMANE_MASKING_ABLATION.json", ["action_masking_contribution", "reward_pct_delta"], 20.0),
    ("R6_PROVIDER_V2.json", ["graphs", "easy", "improvement_vs_mlp_pct"], 40.0),
]


def nested(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    failures = []
    for fname, path, floor in FLOORS:
        fp = R / fname
        if not fp.exists():
            print(f"[FAIL] {fname} not present at {fp}")
            failures.append(f"{fname} MISSING at {fp}")
            continue
        d = json.loads(fp.read_text())
        v = nested(d, path)
        if v is None:
            failures.append(f"{fname}:{'.'.join(path)} missing")
            continue
        if floor is None:
            print(f"[OK  ] {fname}:{'.'.join(path)} = {v} (presence only)")
            continue
        if float(v) < floor * (1 - TOL):
            failures.append(f"{fname}:{'.'.join(path)} = {v} < floor {floor} (tol {TOL})")
        else:
            print(f"[OK  ] {fname}:{'.'.join(path)} = {v} (floor {floor})")

    if failures:
        print("\n[FAIL] Benchmark regression(s) detected:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[PASS] All benchmarks meet their v3.0-arcadia floor.")


if __name__ == "__main__":
    main()
