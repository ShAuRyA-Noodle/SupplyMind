"""final_adversarial_20suite.py — driver for the REAL adversarial gauntlet.

History: this file used to be a 20-attack Wordle-only suite whose sibling nb13
gauntlet inflated a "269/269" number with +19 phantom attacks and both-branches
counting. R7 replaces that with a genuine gauntlet: >300 DISTINCT attacks, each
executed once against the REAL system, each blocked only when a single crisp
assertion on the real return value holds.

The attack corpus lives in ``supplymind.warroom.attack_corpus`` and the executor
(with the per-attack oracle + evidence capture) in
``supplymind.warroom.adversarial_gauntlet`` — both reusable. This script runs
them, prints the honest per-category block-rate, enumerates every real breach,
and writes the receipt ``tests/receipts/adversarial_gauntlet_REAL.json``.

Categories (per R7): prompt_injection · malformed_schema · oversized_unicode ·
out_of_range · session_isolation · replay_duplicate · reward_hacking.

Run:  python scripts/final_adversarial_20suite.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supplymind.warroom.adversarial_gauntlet import run  # noqa: E402


def _write_receipt(report: dict) -> tuple[Path, str]:
    receipt = REPO / "tests" / "receipts" / "adversarial_gauntlet_REAL.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    mirror = REPO / "FINAL_SUBMIT" / "receipts" / "adversarial_gauntlet_REAL.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.with_suffix(".sha256").write_text(sha + "\n", encoding="utf-8")
    return receipt, sha


def main() -> int:
    report = run()

    print("=" * 72)
    print("SupplyMind — REAL Adversarial Gauntlet (R7)")
    print("=" * 72)
    print(f"executed : {report['n_executed']} distinct attacks against the real system")
    print(f"blocked  : {report['n_blocked']}/{report['n_executed']} "
          f"({report['block_rate_pct']}%)")
    print(f"elapsed  : {report['elapsed_s']}s   git={report['git_sha'][:10]}   "
          f"py={report['python']}")
    print("-" * 72)
    print(f"{'category':22s} {'blocked/total':>14s} {'rate':>8s}")
    for cat, d in report["per_category"].items():
        print(f"{cat:22s} {str(d['blocked']) + '/' + str(d['n']):>14s} "
              f"{str(d['block_rate_pct']) + '%':>8s}")
    print("-" * 72)

    if report["n_breaches"] == 0:
        print("BREACHES : none — every attack was verifiably blocked/sanitized.")
    else:
        print(f"BREACHES : {report['n_breaches']} REAL breach(es) found "
              f"(logged for fixing — a found breach is a real bug, not hidden):")
        for b in report["breaches"]:
            print(f"  [#{b['id']}] {b['category']}/{b['name']}")
            print(f"        reason  : {b['reason']}")
            print(f"        expect  : {b['expect']}")
            print(f"        evidence: {json.dumps(b['evidence'], default=str)[:220]}")

    ctl = report["controls"]
    print("-" * 72)
    print(f"CONTROLS : {ctl['n_accepted']}/{ctl['n']} legitimate inputs accepted "
          f"(false-positive rate {ctl['false_positive_rate_pct']}%)")
    for cr in ctl["results"]:
        if not cr["accepted"]:
            print(f"  FALSE POSITIVE: {cr['name']} -> {json.dumps(cr['evidence'], default=str)[:160]}")

    if report.get("limitations"):
        print("-" * 72)
        print("LIMITATIONS:")
        for lim in report["limitations"]:
            print(f"  - {lim}")

    receipt, sha = _write_receipt(report)
    print("-" * 72)
    print(json.dumps({
        "n_executed": report["n_executed"],
        "n_blocked": report["n_blocked"],
        "block_rate_pct": report["block_rate_pct"],
        "n_breaches": report["n_breaches"],
        "false_positive_rate_pct": ctl["false_positive_rate_pct"],
        "receipt": str(receipt),
        "sha256": sha,
    }, indent=2))

    # Exit code contract: 0 if >=300 attacks executed and no false positives
    # among controls; the presence of a real breach does NOT flip this to a fake
    # pass — breaches are reported loudly above and enumerated in the receipt.
    ok = report["n_executed"] >= 300 and ctl["false_positive_rate_pct"] == 0.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
