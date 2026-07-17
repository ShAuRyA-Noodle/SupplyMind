#!/usr/bin/env python3
"""verify_claims.py -- machine claim-checker / CI credibility gate for SupplyMind.

Every judge-facing headline number must trace to a committed receipt. This script is the
enforcement mechanism: it checks three things and EXITS NONZERO if any fail.

  1. RECEIPT-BACKED HEADLINE CLAIMS. A curated registry of the numbers cited in the
     regenerated docs (README, MODEL_CARD, DATASET_CARD, HONEST_LIMITATIONS, CLAIMS_LEDGER).
     For each: the receipt JSON must exist AND the asserted value must actually be in it.
     A claim with no backing receipt, or whose receipt disagrees, is UNBACKED -> FAIL.

  2. FABRICATION-FINGERPRINT SCAN. The 2026-07-02 audit struck a set of fabricated numbers
     (248-of-250 features, 269/269 gauntlet, 2735x variance amp, Wilcoxon p=3.9e-18 /
     Cohen d=2.73, twin $135.5M, inconsistent Wordle p-values). This scans every owned doc.
     A fingerprint is ALLOWED only inside an explicit strike/disclosure sentence (a line that
     says it was fabricated / struck / do-not-cite). A BARE fingerprint -- one presented as a
     live claim -- is FAIL.

  3. FINAL-DOC LINK EXISTENCE. Every in-repo path linked from the regenerated FINAL docs must
     exist on disk (dead links in a submission are a credibility hit).

A non-gating LEGACY-INVENTORY report is also printed (PRESENT rows in the FEATURE_INVENTORY
files whose referenced source path is missing) so the sprawl can be cleaned incrementally.

Usage:
    python scripts/verify_claims.py            # human report, exit 0/1
    python scripts/verify_claims.py --json     # machine-readable summary to stdout
    python scripts/verify_claims.py --strict-inventory   # also gate on legacy inventory

Design: pure stdlib, no network, no repo imports. Runs from a fresh clone in <1s.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# 1. RECEIPT-BACKED HEADLINE CLAIMS
# --------------------------------------------------------------------------------------
# Each claim: the number cited in the regenerated docs, plus the committed receipt it comes
# from and an assertion that must hold in that receipt. Assertions:
#   ("dotted.path", "==",  value)          exact
#   ("dotted.path", "~=",  value, tol)     abs(actual-value) <= tol
#   ("dotted.path", ">=",  value)          / "<=" / ">" / "<"
#   ("dotted.path", "is",  True/False)     identity (bools)
#   ("dotted.path", "exists")              key resolves to a non-None value
#   ("dotted.path", "eq_str", "text")      string equality
# Receipt paths are repo-relative. Receipts are committed evidence; the large data/model
# artifacts they were produced from are gitignored by design, so we verify the RECEIPT, not
# the regenerated inputs.

CLAIMS = [
    {
        "id": "R7-gauntlet",
        "desc": "Adversarial gauntlet: 318 real attacks executed, 318 blocked (100%), 0 breaches",
        "receipt": "tests/receipts/adversarial_gauntlet_REAL.json",
        "assert": [
            ("n_executed", "==", 318),
            ("n_blocked", "==", 318),
            ("block_rate_pct", "~=", 100.0, 0.001),
            ("n_breaches", "==", 0),
            ("controls.false_positive_rate_pct", "~=", 0.0, 0.001),
        ],
    },
    {
        "id": "R11-conformal",
        "desc": "Split-conformal: 90% nominal -> 90.03% empirical coverage on held-out test",
        "receipt": "tests/receipts/conformal_REAL.json",
        "assert": [
            ("status", "eq_str", "OK"),
            ("headline.nominal_coverage", "~=", 0.9, 0.001),
            ("headline.empirical_coverage_on_heldout_test", "~=", 0.90025, 0.001),
            ("headline.coverage_gap", "<=", 0.01),
            ("n_transitions_total", "==", 40000),
        ],
    },
    {
        "id": "R10-brent",
        "desc": "Brent ensemble walk-forward backtest on real FRED: mean MAPE ~7.1% over 8 events",
        "receipt": "tests/receipts/ensemble_brent_REAL.json",
        "assert": [
            ("n_events_valid", "==", 8),
            ("aggregate_metrics_vs_realized.ensemble.mean_mape_pct", "~=", 7.118, 0.05),
            ("aggregate_metrics_vs_realized.ensemble.median_mape_pct", "~=", 6.421, 0.05),
            ("brent_provenance.n_observations", "==", 9922),
        ],
    },
    {
        "id": "R12-fedavg",
        "desc": "FedAvg on DataCo regions: honest negative -- does NOT beat centralized (delta ~0)",
        "receipt": "tests/receipts/fedavg_REAL.json",
        "assert": [
            ("findings.fedavg_vs_centralized_auc_delta", "~=", 0.0, 0.001),
            ("results.fedavg.global_test.auc", "~=", 0.7218, 0.001),
            ("results.centralized.global_test.auc", "~=", 0.7218, 0.001),
            ("dataset.n_orders", "==", 180519),
        ],
    },
    {
        "id": "R9-ghost-retire",
        "desc": "Ghost models measured: embedder ensemble + reranker RETIRE (do not help); TabPFN KEEP",
        "receipt": "tests/receipts/ghost_models_eval_REAL.json",
        "assert": [
            ("retrieval.ensemble.decision", "eq_str", "RETIRE"),
            ("retrieval.rerank.decision", "eq_str", "RETIRE"),
            ("tabpfn_judge.tabpfn.decision", "eq_str", "KEEP"),
            ("retrieval.aggregate_metrics.mxbai_alone.p1", "~=", 0.9623, 0.001),
            ("retrieval.aggregate_metrics.mxbai_alone.mrr", "~=", 0.978, 0.001),
            ("tabpfn_judge.tabpfn.auc", "~=", 0.7377, 0.001),
            ("tabpfn_judge.logreg_auc", "~=", 0.7028, 0.001),
        ],
    },
    {
        "id": "R17-rag-worldbank",
        "desc": "RAG re-cook: World Bank ingestion bug fixed, 0 -> 20 WB chunks, 6602 total",
        "receipt": "tests/receipts/rag_recook_REAL.json",
        "assert": [
            ("corpus.world_bank_chunks_before", "==", 0),
            ("corpus.world_bank_chunks_after", "==", 20),
            ("corpus.n_chunks_total", "==", 6602),
            ("retrieval_wb_queries.after_wb.p1", "~=", 1.0, 0.001),
        ],
    },
    {
        "id": "R6-counterfactual",
        "desc": "4-method causal counterfactual (Tohoku 2011): 4 real methods, scopes disagree (honest)",
        "receipt": "FINAL_SUBMIT/receipts/counterfactual_4method_REAL.json",
        "assert": [
            ("result.pooled.n_methods_ok", "==", 4),
            ("result.pooled.methods_agree_within_3x", "is", False),
            ("result.pooled.pooled_range_brackets_headline", "is", True),
            ("result.methods.b.label", "eq_str", "synthetic_control_abadie"),
        ],
    },
    {
        "id": "wordle-reinforce",
        "desc": "Wordle REINFORCE: beats random (p=2.7e-18, d=4.27) but NOT info-aware greedy (p=0.93, honest)",
        "receipt": "FINAL_SUBMIT/receipts/pass27_B_real_episodic_bootstrap.json",
        "assert": [
            ("wilcoxon.reinforce_vs_random_p", "<", 1e-10),
            ("cohens_d.reinforce_vs_random", "~=", 4.275, 0.05),
            ("wilcoxon.reinforce_vs_greedy_p", ">", 0.5),
            ("paired_bootstrap_reinforce_vs_random.ci_excludes_zero", "is", True),
        ],
    },
    {
        "id": "tft-real",
        "desc": "TFT quantile forecaster on real WTI: MAE p50 ~$7.83, 90,602 params",
        "receipt": "FINAL_SUBMIT/receipts/tft_real_metrics.json",
        "assert": [
            ("mae_p50_usd", "~=", 7.827, 0.05),
            ("params", "==", 90602),
            ("target", "eq_str", "DCOILWTICO"),
        ],
    },
    {
        "id": "shap-real",
        "desc": "Real SHAP on trained BC checkpoint: NOAA feature group dominates (~60%)",
        "receipt": "FINAL_SUBMIT/receipts/shap_real.json",
        "assert": [
            ("n_explained", "==", 200),
            ("group_shares.NOAA", ">=", 0.5),
            ("checkpoint", "exists"),
        ],
    },
    {
        "id": "K1-fred-brent",
        "desc": "8 crisis events anchored to real FRED Brent daily; no synthetic substitution",
        "receipt": "FINAL_SUBMIT/receipts/pass28_K1_fred_brent_real.json",
        "assert": [
            ("n_events_with_real_data", "==", 8),
            ("no_synthetic_substitution", "is", True),
        ],
    },
    # NOTE: tests/receipts/war_room_validation.json is a REAL run but its committed JSON has a
    # malformed escape (`\scenarios`) that makes it unloadable -> cannot machine-gate it here.
    # It is logged RERUN-PENDING in CLAIMS_LEDGER (owned by the receipts wave, not docs).
    {
        "id": "test-suite",
        "desc": "Test suite: 184/184 pass, CPU-only, exit 0",
        "receipt": "tests/receipts/test_suite_grand_total.json",
        "assert": [
            ("results.passed", "==", 184),
            ("results.failed", "==", 0),
            ("results.exit_code", "==", 0),
            ("all_passed", "is", True),
        ],
    },
]

# --------------------------------------------------------------------------------------
# 2. FABRICATION FINGERPRINTS (struck by the 2026-07-02 audit)
# --------------------------------------------------------------------------------------
# A fingerprint match is OK only if the line is an explicit strike/disclosure. Otherwise it
# is a BARE fabricated claim -> FAIL. NOTE: 2.71e-18 / d=4.27 is deliberately NOT here -- it
# is the REAL Wordle-vs-random result (pass27_B), backed above.

FINGERPRINTS = [
    (r"248\s*[-–/ ]?\s*(?:of|/)\s*[-–/ ]?\s*250", "248-of-250 features (hardcoded constant, D9)"),
    (r"PROJECT_TOTAL_DEMONSTRATED", "hardcoded feature-count constant (D9)"),
    (r"248\s+features\s+demonstrated", "248 features demonstrated (D9)"),
    (r"269\s*[-–/ ]?\s*(?:of|/)\s*[-–/ ]?\s*269", "269/269 gauntlet (rigged both-branches, A3)"),
    (r"257\s*/\s*257", "257/257 adversarial (rigged, A3)"),
    (r"2735\s*[×xX]|2735[\s-]?fold", "2735x variance amplification (hardcoded demo, A4)"),
    (r"3\.9\s*[eE]\s*-?\s*18|3\.9\s*[×xX]\s*10", "Wilcoxon p=3.9e-18 (sorted-'paired', A1)"),
    (r"(?:cohen'?s?\s*d|[^a-z]d)\s*=\s*\+?\s*2\.73(?!\d)", "Cohen d=2.73 leaderboard (fabricated, A1)"),
    (r"\$?\s*135\.5\s*(?:M\b|million)", "twin $135.5M savings (exit -9, never completed, D3)"),
    (r"9\.39\s*[eE]-?35|1\.87\s*[eE]-?34|6\.6\s*[eE]-?35", "inconsistent Wordle p-values (fabrication flag, A2)"),
]

# A line carrying a fingerprint is a legitimate honest DISCLOSURE (not a live claim) if it
# also contains any of these markers.
DISCLOSURE_MARKERS = [
    "struck", "strike", "fabricat", "do not cite", "do-not-cite", "unverified",
    "removed", "retracted", "rigged", "hardcoded", "hard-coded", "audit found",
    "audit_", "false", "no longer", "superseded", "pending", "re-run", "rerun",
    "was wrong", "not a real", "not real", "deleted", "~~", "[struck]", "never ran",
    "never trained", "never completed", "exit -9", "exit_code", "synthetic",
    "invalid", "overcount", "placeholder", "stub", "caveat", "honest", "corrected",
    "replaced", "purged", "de-fake", "defake", "flagged", "constant", "by construction",
]

# --------------------------------------------------------------------------------------
# 3. OWNED-DOC SETS
# --------------------------------------------------------------------------------------
# FINAL docs -- authoritative, must be pristine (fingerprint-clean unless disclosing) and
# their in-repo links must resolve.
FINAL_DOCS = [
    "README.md",
    "FINAL_SUBMIT/README.md",
    "FINAL_SUBMIT/CLAIMS_LEDGER.md",
    "FINAL_SUBMIT/MODEL_CARD.md",
    "FINAL_SUBMIT/DATASET_CARD.md",
    "FINAL_SUBMIT/HONEST_LIMITATIONS.md",
]

# Legacy inventory files whose PRESENT rows we spot-check (non-gating by default).
INVENTORY_DOCS = [
    "FINAL_SUBMIT/FEATURE_INVENTORY.md",
    "FINAL_SUBMIT/FEATURE_INVENTORY_DI.md",
    "FINAL_SUBMIT/FEATURE_INVENTORY_JT.md",
    "FINAL_SUBMIT/FEATURE_INVENTORY_UBB.md",
]


def owned_docs():
    """Every markdown/html doc this workstream owns (for the fingerprint scan)."""
    docs = []
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        docs.append(readme)
    for base in ("docs", "FINAL_SUBMIT"):
        root = REPO_ROOT / base
        if root.exists():
            docs.extend(sorted(root.rglob("*.md")))
            docs.extend(sorted(root.rglob("*.html")))
    # de-dup, stable
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _resolve(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return (False, None)
    return (True, cur)


def _check_assertion(receipt, spec):
    path = spec[0]
    op = spec[1]
    found, actual = _resolve(receipt, path)
    if op == "exists":
        ok = found and actual is not None
        return ok, f"{path} exists={found}"
    if not found:
        return False, f"{path} MISSING in receipt"
    try:
        if op == "==":
            ok = actual == spec[2]
        elif op == "eq_str":
            ok = str(actual) == str(spec[2])
        elif op == "is":
            ok = actual is spec[2] or bool(actual) == bool(spec[2])
        elif op == "~=":
            ok = abs(float(actual) - float(spec[2])) <= float(spec[3])
        elif op == ">=":
            ok = float(actual) >= float(spec[2])
        elif op == "<=":
            ok = float(actual) <= float(spec[2])
        elif op == ">":
            ok = float(actual) > float(spec[2])
        elif op == "<":
            ok = float(actual) < float(spec[2])
        else:
            return False, f"{path} unknown op {op}"
    except (TypeError, ValueError) as exc:
        return False, f"{path} type error: {exc}"
    want = spec[2] if len(spec) > 2 else ""
    return ok, f"{path} {op} {want} (got {actual})"


def check_receipt_claims():
    results = []
    for claim in CLAIMS:
        rp = REPO_ROOT / claim["receipt"]
        if not rp.exists():
            results.append({"id": claim["id"], "ok": False, "desc": claim["desc"],
                            "receipt": claim["receipt"],
                            "detail": ["RECEIPT FILE MISSING"]})
            continue
        try:
            receipt = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            results.append({"id": claim["id"], "ok": False, "desc": claim["desc"],
                            "receipt": claim["receipt"],
                            "detail": [f"RECEIPT UNREADABLE: {exc}"]})
            continue
        details, all_ok = [], True
        for spec in claim["assert"]:
            ok, msg = _check_assertion(receipt, spec)
            all_ok = all_ok and ok
            details.append(("OK  " if ok else "FAIL") + " " + msg)
        results.append({"id": claim["id"], "ok": all_ok, "desc": claim["desc"],
                        "receipt": claim["receipt"], "detail": details})
    return results


def _is_disclosure(line):
    low = line.lower()
    return any(m in low for m in DISCLOSURE_MARKERS)


def scan_fingerprints():
    """Return (bare_hits, disclosed_hits). bare_hits are gating failures."""
    bare, disclosed = [], []
    for doc in owned_docs():
        try:
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        rel = doc.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(lines, 1):
            for pat, label in FINGERPRINTS:
                if re.search(pat, line, flags=re.IGNORECASE):
                    entry = {"file": rel, "line": i, "label": label,
                             "text": line.strip()[:160]}
                    (disclosed if _is_disclosure(line) else bare).append(entry)
    return bare, disclosed


_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def check_final_doc_links():
    """Every in-repo path linked from a FINAL doc must exist."""
    dead, checked = [], 0
    for rel in FINAL_DOCS:
        doc = REPO_ROOT / rel
        if not doc.exists():
            dead.append({"doc": rel, "target": "(FINAL DOC ITSELF MISSING)"})
            continue
        text = doc.read_text(encoding="utf-8", errors="replace")
        for m in _LINK_RE.finditer(text):
            target = m.group(1).strip()
            # skip external, anchors, mailto, images-by-data
            if target.startswith(("http://", "https://", "#", "mailto:", "data:")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0].strip()
            if not target:
                continue
            # only check things that look like repo paths (have a dir sep or a file ext)
            if "/" not in target and "." not in target:
                continue
            checked += 1
            p = (doc.parent / target).resolve() if target.startswith(".") else (REPO_ROOT / target)
            if not p.exists():
                # also try relative to the doc's own directory
                alt = (doc.parent / target)
                if not alt.exists():
                    dead.append({"doc": rel, "target": target})
    return dead, checked


_PRESENT_ROW_RE = re.compile(r"\|\s*PRESENT\s*\|", re.IGNORECASE)
_BACKTICK_PATH_RE = re.compile(r"`([^`]+?)`")


def check_inventory_present_rows():
    """Non-gating: PRESENT inventory rows whose referenced source path is missing."""
    missing, checked = [], 0
    for rel in INVENTORY_DOCS:
        doc = REPO_ROOT / rel
        if not doc.exists():
            continue
        for line in doc.read_text(encoding="utf-8", errors="replace").splitlines():
            if not _PRESENT_ROW_RE.search(line):
                continue
            for tok in _BACKTICK_PATH_RE.findall(line):
                # a path token looks like  foo/bar.ext  or  foo/bar.ext:12
                cand = tok.split(":", 1)[0].strip()
                if "/" not in cand or " " in cand:
                    continue
                if not re.search(r"\.\w{1,5}$", cand):
                    continue
                checked += 1
                if not (REPO_ROOT / cand).exists():
                    missing.append({"doc": rel, "path": cand})
    return missing, checked


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SupplyMind claim / fingerprint verifier")
    ap.add_argument("--json", action="store_true", help="emit machine-readable summary")
    ap.add_argument("--strict-inventory", action="store_true",
                    help="also fail on missing PRESENT-row inventory paths")
    args = ap.parse_args()

    # Windows consoles default to cp1252; our reports and doc snippets contain unicode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    claim_results = check_receipt_claims()
    bare_fp, disclosed_fp = scan_fingerprints()
    dead_links, n_links = check_final_doc_links()
    inv_missing, n_inv = check_inventory_present_rows()

    backed = sum(1 for c in claim_results if c["ok"])
    unbacked = sum(1 for c in claim_results if not c["ok"])

    gate_fail = (unbacked > 0) or (len(bare_fp) > 0) or (len(dead_links) > 0)
    if args.strict_inventory and inv_missing:
        gate_fail = True

    if args.json:
        print(json.dumps({
            "claims_checked": len(claim_results),
            "claims_backed": backed,
            "claims_unbacked": unbacked,
            "unbacked_ids": [c["id"] for c in claim_results if not c["ok"]],
            "fingerprints_bare": len(bare_fp),
            "fingerprints_disclosed": len(disclosed_fp),
            "final_doc_links_checked": n_links,
            "final_doc_dead_links": len(dead_links),
            "inventory_present_paths_checked": n_inv,
            "inventory_present_missing": len(inv_missing),
            "exit": 1 if gate_fail else 0,
        }, indent=2))
        return 1 if gate_fail else 0

    W = 86
    print("=" * W)
    print("SupplyMind  verify_claims.py  --  receipt-backed credibility gate")
    print("=" * W)

    print("\n[1] RECEIPT-BACKED HEADLINE CLAIMS")
    print("-" * W)
    for c in claim_results:
        tag = "PASS" if c["ok"] else "FAIL"
        print(f"  [{tag}] {c['id']:<20} {c['desc']}")
        print(f"         receipt: {c['receipt']}")
        if not c["ok"]:
            for d in c["detail"]:
                if d.startswith("FAIL") or "MISSING" in d or "UNREADABLE" in d:
                    print(f"         -> {d}")
    print(f"\n  claims checked={len(claim_results)}  backed={backed}  UNBACKED={unbacked}")

    print("\n[2] FABRICATION-FINGERPRINT SCAN (owned docs)")
    print("-" * W)
    print(f"  disclosed (struck / honest context, OK): {len(disclosed_fp)}")
    print(f"  BARE (live fabricated claim, FAIL):       {len(bare_fp)}")
    for e in bare_fp:
        print(f"    FAIL {e['file']}:{e['line']}  [{e['label']}]")
        print(f"         {e['text']}")

    print("\n[3] FINAL-DOC LINK EXISTENCE")
    print("-" * W)
    print(f"  in-repo links checked={n_links}  dead={len(dead_links)}")
    for e in dead_links:
        print(f"    FAIL {e['doc']} -> {e['target']}")

    print("\n[4] LEGACY INVENTORY PRESENT-ROW PATHS  (non-gating unless --strict-inventory)")
    print("-" * W)
    print(f"  PRESENT-row paths checked={n_inv}  missing={len(inv_missing)}")
    for e in inv_missing[:25]:
        print(f"    warn {e['doc']} -> {e['path']}")
    if len(inv_missing) > 25:
        print(f"    ... +{len(inv_missing) - 25} more")

    print("\n" + "=" * W)
    verdict = "FAIL" if gate_fail else "PASS"
    print(f"VERDICT: {verdict}   "
          f"(claims {backed}/{len(claim_results)} backed, "
          f"{len(bare_fp)} bare fingerprints, {len(dead_links)} dead links)")
    print("=" * W)
    return 1 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
