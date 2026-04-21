"""
receipts.py — F10 Reproducibility Receipt System.

For every headline number in the project, generate a pair:

    receipts/<number_id>.receipt      — JSON: {number, value, command, env_hash,
                                                 git_sha, data_hash, expected_output_hash}
    receipts/<number_id>.reproduce.sh — shell one-liner that re-derives the number

A judge can verify any claim in under 30 seconds:

    cat receipts/R5_GRANITE_mxbai_P1.receipt
    bash receipts/R5_GRANITE_mxbai_P1.reproduce.sh   # prints same number

The receipt captures:
    - The exact jq/python command
    - Git SHA at issuance
    - Hash of relevant data files
    - Expected output

No other hackathon team will have this level of third-party verifiability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = PROJECT_ROOT / "ShAuRyA_Supplymind" / "receipts"


@dataclass
class Receipt:
    number_id: str
    description: str
    value: str
    command: str
    expected_output: str
    data_files_hashes: dict = field(default_factory=dict)
    git_sha: str = ""
    generated_at: str = ""
    python_version: str = ""
    platform: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _run(cmd: str) -> str:
    """Execute command; return stripped stdout or error tag."""
    try:
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, shell=True,
                                      stderr=subprocess.STDOUT, timeout=120)
        return out.decode(errors="replace").strip()
    except subprocess.CalledProcessError as e:
        return f"[command failed rc={e.returncode}] {e.output.decode(errors='replace')[:200]}"
    except Exception as e:
        return f"[execution error] {e}"


# ---------------------------------------------------------------------------
# The full set of headline-number receipts
# ---------------------------------------------------------------------------


def _jqlike(json_path: str, jq_path: str) -> str:
    """Build a cross-platform python one-liner that emulates `jq -r 'jq_path'`.

    jq_path form is dotted (e.g. `.pipelines.P2_mxbai_bi.p1`). Each dotted
    segment drills into a dict key (quoted keys use bracket form).
    """
    # Simple conversion: replace .key -> ['key'], keeping bracket [] as-is.
    import re as _re
    segments = _re.findall(r'\[[^\]]+\]|\.[A-Za-z0-9_@]+|\."[^"]+"', jq_path)
    code_path = ""
    for seg in segments:
        if seg.startswith('.'):
            key = seg[1:]
            if key.startswith('"') and key.endswith('"'):
                key = key[1:-1]
            code_path += f"[{key!r}]"
        else:
            code_path += seg
    return (f'python -c "import json; print(json.load(open(r\'{json_path}\'))'
            f'{code_path})"')


RECEIPT_SPECS: list[dict] = [
    {
        "number_id": "R5_GRANITE_mxbai_P1",
        "description": "RAG P@1 on 6,483-chunk real corpus, mxbai bi-encoder",
        "command": _jqlike("v3_arcadia/results/R5_GRANITE.json", ".pipelines.P2_mxbai_bi.p1"),
        "data_files": ["v3_arcadia/results/R5_GRANITE.json"],
    },
    {
        "number_id": "R5_GRANITE_mxbai_MRR",
        "description": "RAG MRR on precise queries",
        "command": _jqlike("v3_arcadia/results/R5_GRANITE.json", ".pipelines.P2_mxbai_bi.mrr"),
        "data_files": ["v3_arcadia/results/R5_GRANITE.json"],
    },
    {
        "number_id": "R5_BEIR_snowflake_nDCG10",
        "description": "BEIR out-of-domain nDCG@10 (Snowflake) on 26 Wiki crisis articles",
        "command": _jqlike("v3_arcadia/results/R5_BEIR_MANUAL.json",
                           '.our_results."snowflake-arctic-l"."mean_ndcg@10"'),
        "data_files": ["v3_arcadia/results/R5_BEIR_MANUAL.json"],
    },
    {
        "number_id": "R4_2JUDGE_Krippendorff_alpha",
        "description": "2-judge panel Krippendorff ordinal alpha on 26 crisis scenarios",
        "command": _jqlike("v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json",
                           ".agreement_primary_panel.krippendorff_alpha_ordinal"),
        "data_files": ["v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json"],
    },
    {
        "number_id": "R4_Cohen_kappa_QwenMistral",
        "description": "Cohen weighted kappa Qwen-14B x Mistral-Nemo",
        "command": _jqlike("v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json",
                           ".agreement_primary_panel.cohen_weighted_kappa_qwen_vs_mistral"),
        "data_files": ["v3_arcadia/results/R4_DANGEROUS_V2_ABLATION.json"],
    },
    {
        "number_id": "R6_MaskingAblation_easy_lift",
        "description": "MaskablePPO easy-task reward lift vs plain PPO (+%)",
        "command": _jqlike("v3_arcadia/results/R6_GETHSEMANE_MASKING_ABLATION.json",
                           ".action_masking_contribution.reward_pct_delta"),
        "data_files": ["v3_arcadia/results/R6_GETHSEMANE_MASKING_ABLATION.json"],
    },
    {
        "number_id": "R6_GCN_easy_MAE_vs_MLP",
        "description": "GNN easy-graph MAE reduction vs MLP baseline (%)",
        "command": _jqlike("v3_arcadia/results/R6_PROVIDER_V2.json",
                           ".graphs.easy.improvement_vs_mlp_pct"),
        "data_files": ["v3_arcadia/results/R6_PROVIDER_V2.json"],
    },
    {
        "number_id": "R6_AquaRegia_WTI_dev95",
        "description": "Per-horizon conformal deviation at 95% nominal, WTI ARIMA",
        "command": ("python -c \"import json; d=json.load(open('v3_arcadia/results/R6_AQUA_REGIA_V2.json'));"
                    "c=d['results']['DCOILWTICO']['arima']['conf=0.95'];"
                    "print(abs(c['perhorizon_coverage_mean']-c['nominal_coverage']))\""),
        "data_files": ["v3_arcadia/results/R6_AQUA_REGIA_V2.json"],
    },
    {
        "number_id": "R3_TimesFM_CP_WTI_dev95",
        "description": "TimesFM-CP WTI deviation from 95% nominal",
        "command": ("python -c \"import json; d=json.load(open('v3_arcadia/results/R3_TIMESFM_QUANTILE.json'));"
                    "print(d['targets']['DCOILWTICO']['timesfm_conf=0.95']['dev_from_nominal'])\""),
        "data_files": ["v3_arcadia/results/R3_TIMESFM_QUANTILE.json"],
    },
    {
        "number_id": "V4_SPOF_V2_F1",
        "description": "v4 SPOF articulation-point F1 (mean across 3 graphs)",
        "command": _jqlike("ShAuRyA_Supplymind/features/R6_SPOF_V2.json",
                           ".summary.v2_mean_f1"),
        "data_files": ["ShAuRyA_Supplymind/features/R6_SPOF_V2.json"],
    },
    {
        "number_id": "V4_STACKING_V2_lift_vs_WV",
        "description": "v4 Stacking v2 AUC lift vs ensemble weighted voting",
        "command": _jqlike("ShAuRyA_Supplymind/features/R15_STACKING_V2.json",
                           ".lift_stacking_vs_wv_auc"),
        "data_files": ["ShAuRyA_Supplymind/features/R15_STACKING_V2.json"],
    },
    {
        "number_id": "V4_Live_Brent_202604",
        "description": "FRED Brent crude spot price as ingested on 2026-04-21 ($/bbl)",
        "command": ("python -c \"import sqlite3, json; c=sqlite3.connect('ShAuRyA_Supplymind/realtime/events.db');"
                    "r=c.execute('SELECT meta_json FROM events WHERE source=? ORDER BY ts_unix DESC LIMIT 1', "
                    "('fred_brent',)).fetchone();"
                    "print(json.loads(r[0])['latest_price']) if r else print('no-data')\""),
        "data_files": ["ShAuRyA_Supplymind/realtime/events.db"],
    },
    {
        "number_id": "V4_Tests_Total",
        "description": "Total test count across v3 + v4",
        "command": ("python -m pytest tests/ ShAuRyA_Supplymind/tests/ "
                    "--collect-only -q"),
        "data_files": [],
    },
    {
        "number_id": "V4_Analyst_V5_Exact_Acc",
        "description": "supplymind-analyst:v5 vs base Qwen on 10 rubric-labeled scenarios",
        "command": _jqlike("ShAuRyA_Supplymind/features/R9_ANALYST_AB_V5.json",
                           ".summary.exact_acc_lift"),
        "data_files": ["ShAuRyA_Supplymind/features/R9_ANALYST_AB_V5.json"],
    },
    {
        "number_id": "V4_Autoresearch_Best_CI95",
        "description": "Best CI95-lower accepted by autoresearch orchestrator (bootstrap 1000)",
        "command": ("python -c \"import json; d=json.load(open('ShAuRyA_Supplymind/autoresearch/state.json'));"
                    "print(d['best']['metric']['ci95_lower']) if d.get('best') else print('none')\""),
        "data_files": ["ShAuRyA_Supplymind/autoresearch/state.json"],
    },
]


def generate_all_receipts(verify: bool = True) -> list[Receipt]:
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    import platform as _p
    py = f"{_p.python_version()}"
    plat = f"{_p.system()}-{_p.release()}"

    out_receipts = []
    for spec in RECEIPT_SPECS:
        value = "(not executed)"
        expected = "(not executed)"
        if verify:
            expected = _run(spec["command"])
            value = expected.splitlines()[0][:400] if expected else ""

        r = Receipt(
            number_id=spec["number_id"],
            description=spec["description"],
            value=value,
            command=spec["command"],
            expected_output=expected,
            data_files_hashes={p: _file_hash(PROJECT_ROOT / p) for p in spec["data_files"]},
            git_sha=sha,
            generated_at=now,
            python_version=py,
            platform=plat,
        )

        # Write receipt JSON
        rec_path = RECEIPTS_DIR / f"{r.number_id}.receipt"
        rec_path.write_text(json.dumps(r.to_dict(), indent=2))

        # Write reproduce.sh
        sh_path = RECEIPTS_DIR / f"{r.number_id}.reproduce.sh"
        sh_content = (
            "#!/usr/bin/env bash\n"
            "# Auto-generated by ShAuRyA_Supplymind/features/receipts.py\n"
            f"# Verify: {r.description}\n"
            f"# Expected: {r.expected_output[:100]}\n"
            f"# Git SHA at issuance: {r.git_sha}\n"
            "set -e\n"
            f"cd \"$(dirname \"$0\")/../..\"\n"
            f"{r.command}\n"
        )
        sh_path.write_text(sh_content, encoding="utf-8")
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass

        out_receipts.append(r)
        logger.info("[receipt] %s = %s", r.number_id, (r.value[:40] if r.value else "?"))

    # Index
    index = {
        "generated_at": now,
        "git_sha": sha,
        "n_receipts": len(out_receipts),
        "receipts": [
            {"id": r.number_id, "desc": r.description,
             "value": r.value[:60], "command": r.command[:200]}
            for r in out_receipts
        ],
    }
    (RECEIPTS_DIR / "INDEX.json").write_text(json.dumps(index, indent=2))
    # Human-readable table
    lines = ["# SupplyMind Receipts — Verify Any Headline Number in 30 Seconds\n",
             f"*generated {now} from git SHA `{sha}`*\n",
             "| # | Number | Value | Verify |",
             "|---|--------|-------|--------|"]
    for r in out_receipts:
        lines.append(f"| {r.number_id} | {r.description[:60]} | `{r.value[:30]}` | `bash receipts/{r.number_id}.reproduce.sh` |")
    (RECEIPTS_DIR / "INDEX.md").write_text("\n".join(lines))
    return out_receipts


def verify_receipt(number_id: str) -> dict:
    """Re-run a receipt's command and compare to stored expected_output."""
    rec_path = RECEIPTS_DIR / f"{number_id}.receipt"
    if not rec_path.exists():
        return {"status": "missing", "number_id": number_id}
    data = json.loads(rec_path.read_text())
    now_output = _run(data["command"])
    match = now_output.strip() == data["expected_output"].strip()
    return {
        "status": "match" if match else "drift",
        "number_id": number_id,
        "stored": data["expected_output"][:200],
        "current": now_output[:200],
    }


def verify_all() -> dict:
    results: list[dict] = []
    for p in sorted(RECEIPTS_DIR.glob("*.receipt")):
        results.append(verify_receipt(p.stem))
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {"summary": by_status, "details": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate all receipts")
    parser.add_argument("--verify-all", action="store_true", help="Verify all stored receipts")
    parser.add_argument("--no-exec", action="store_true",
                        help="When generating, write receipts without executing commands")
    args = parser.parse_args()

    if args.generate:
        recs = generate_all_receipts(verify=not args.no_exec)
        print(f"generated {len(recs)} receipts in {RECEIPTS_DIR}")
    if args.verify_all:
        result = verify_all()
        print(json.dumps(result, indent=2))
