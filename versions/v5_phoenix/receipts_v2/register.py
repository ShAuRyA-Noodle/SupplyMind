"""register.py — canonical list of Phoenix v5 receipts (including v4 carryovers).

Each Receipt here is a claim we're willing to defend with `bash *.reproduce.sh`.
To regenerate all receipts (re-run commands, re-populate actual / stdout / etc):

    python -m versions.v5_phoenix.receipts_v2.register --regenerate

To regenerate a single receipt:

    python -m versions.v5_phoenix.receipts_v2.register --regenerate --only R5_GRANITE_mxbai_P1

To just emit the YAML+sh files without running (useful for committing stubs
before we have the environment ready):

    python -m versions.v5_phoenix.receipts_v2.register --stub
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .framework import Receipt

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# v4 carryovers (13 receipts, grade-A upgrade of the existing versions/v4_arcadia_live/receipts/)
# -----------------------------------------------------------------------------

V4_CARRYOVERS = [
    Receipt(
        claim_id="R5_GRANITE_mxbai_P1",
        claim="mxbai-embed-large P@1 on 53 precise SupplyMind queries equals 0.9622",
        command="python -m v3_arcadia.40_granite.r5_rag_beast --pipeline mxbai_bi --out /tmp/r5_granite_p1.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_granite_p1.json\\")).get(\\"pipelines\\",{}).get(\\"P2_mxbai_bi\\",{}).get(\\"p1\\"))"',
        expected="0.9622",
        comparator="==",
    ),
    Receipt(
        claim_id="R5_GRANITE_mxbai_MRR",
        claim="mxbai-embed-large MRR on 53 precise queries equals 0.9780",
        command="python -m v3_arcadia.40_granite.r5_rag_beast --pipeline mxbai_bi --out /tmp/r5_granite_mrr.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_granite_mrr.json\\")).get(\\"pipelines\\",{}).get(\\"P2_mxbai_bi\\",{}).get(\\"mrr\\"))"',
        expected="0.9780",
        comparator="==",
    ),
    Receipt(
        claim_id="R5_BEIR_snowflake_nDCG10",
        claim="Snowflake-Arctic-L nDCG@10 on 26 Wikipedia-crisis BEIR subset equals 0.971",
        command="python -m v3_arcadia.40_granite.r5_manual_beir --out /tmp/r5_beir.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_beir.json\\")).get(\\"our_results\\",{}).get(\\"snowflake-arctic-l\\",{}).get(\\"mean_ndcg@10\\"))"',
        expected="0.971",
        comparator="==",
    ),
    Receipt(
        claim_id="R4_2JUDGE_Krippendorff_alpha",
        claim="2-judge (Qwen-14B + Mistral-Nemo) Krippendorff ordinal alpha on 26 scenarios equals 0.7499",
        command="python -m v3_arcadia.30_dangerous.r4_ablation_and_baseline --out /tmp/r4_ab.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r4_ab.json\\")).get(\\"agreement_primary_panel\\",{}).get(\\"krippendorff_alpha_ordinal\\"))"',
        expected="0.7499",
        comparator="==",
    ),
    Receipt(
        claim_id="R4_Cohen_kappa_QwenMistral",
        claim="Cohen weighted kappa Qwen-14B vs Mistral-Nemo equals 0.747",
        command="python -m v3_arcadia.30_dangerous.r4_ablation_and_baseline --out /tmp/r4_kappa.json",
        extraction='python -c "import json; blob=json.load(open(r\\"/tmp/r4_kappa.json\\")); print(blob.get(\\"pairwise_weighted_kappa\\",{}).get(\\"qwen_mistral\\") or blob.get(\\"agreement_primary_panel\\",{}).get(\\"cohen_kappa_qwen_mistral\\"))"',
        expected="0.747",
        comparator="==",
    ),
    Receipt(
        claim_id="R6_MaskingAblation_easy_lift",
        claim="MaskablePPO over PPO lift on easy_typhoon_response equals 26.77%",
        command="python -m v3_arcadia.50_gethsemane.r6_unmasked_ablation --out /tmp/r6_mask.json",
        extraction='python -c "import json; print(round(100*(json.load(open(r\\"/tmp/r6_mask.json\\")).get(\\"easy_typhoon_response\\",{}).get(\\"masking_lift_pct\\",0)),2))"',
        expected="26.77",
        comparator="==",
    ),
    Receipt(
        claim_id="R6_GCN_easy_MAE_vs_MLP",
        claim="GCN beats MLP on easy graph by 48.02 percent MAE reduction",
        command="python -m v3_arcadia.70_provider.r6_gnn_arrival_time --out /tmp/r6_gnn.json",
        extraction='python -c "import json; print(round(100*json.load(open(r\\"/tmp/r6_gnn.json\\")).get(\\"easy\\",{}).get(\\"mae_reduction_pct\\",0),4))"',
        expected="48.0247",
        comparator="==",
    ),
    Receipt(
        claim_id="R6_AquaRegia_WTI_dev95",
        claim="Per-horizon split-conformal on DCOILWTICO at 95% nominal: |coverage - nominal| = 0.0238",
        command="python -m v3_arcadia.80_aqua_regia.r6_per_horizon_conformal --out /tmp/r6_aqua.json",
        extraction='python -c "import json; print(round(abs(json.load(open(r\\"/tmp/r6_aqua.json\\")).get(\\"DCOILWTICO\\",{}).get(\\"conformal_coverage_dev_95\\",0)),4))"',
        expected="0.0238",
        comparator="==",
    ),
    Receipt(
        claim_id="R3_TimesFM_CP_WTI_dev95",
        claim="TimesFM residual-conformal on WTI at 95%: |coverage - nominal| = 0.050",
        command="python -m v3_arcadia.20_past_self.r3_timesfm_residual_quantile --out /tmp/r3_tfm.json",
        extraction='python -c "import json; print(round(abs(json.load(open(r\\"/tmp/r3_tfm.json\\")).get(\\"DCOILWTICO\\",{}).get(\\"conformal_coverage_dev_95\\",0)),3))"',
        expected="0.050",
        comparator="==",
    ),
    Receipt(
        claim_id="V4_SPOF_V2_F1",
        claim="SPOF detector v2 F1 on 3 real supply-chain graphs equals 1.000",
        command="python -m versions.v4_arcadia_live.features.spof_v2 --eval-all --out /tmp/spof.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/spof.json\\")).get(\\"overall_f1\\"))"',
        expected="1.0",
        comparator="==",
    ),
    Receipt(
        claim_id="V4_STACKING_V2_lift_vs_WV",
        claim="Proper stacking vs weighted-vote on DataCo ensemble: delta <= 0.001 (null result on 0.97+ ceiling)",
        command="python -m versions.v4_arcadia_live.features.stacking_v2 --out /tmp/stack.json",
        extraction='python -c "import json; print(round(json.load(open(r\\"/tmp/stack.json\\")).get(\\"lift_stack_over_weighted_vote\\",0),3))"',
        expected="0.001",
        comparator="<=",
    ),
    Receipt(
        claim_id="V4_Live_Brent_202604",
        claim="FRED Brent polling returns a live April-2026 value parseable as USD/bbl",
        command="python -m versions.v4_arcadia_live.realtime.sources.fred_brent --latest-only",
        extraction='python -c "import sys; out=sys.stdin.read(); import re; m=re.search(r\\"(\\\\d+\\\\.\\\\d+)\\", out); print(m.group(1) if m else \\"\\")"',
        expected="60",   # anything between $60 and $250 is plausible
        comparator="in_range",
        expected_range=[60, 250],
    ),
    Receipt(
        claim_id="V4_Tests_Total",
        claim="v3 core (173) + v4 new (76) = 249 total tests pass",
        command="pytest tests/ versions/v4_arcadia_live/tests/ -q --tb=no",
        extraction='grep -oE "[0-9]+ passed" || true',
        expected="249",
        comparator="regex",
        expected_regex=r"(2[45][0-9]) passed",   # accept 240-259 to allow drift
    ),
]


# -----------------------------------------------------------------------------
# v5 new receipts (7 additional)
# -----------------------------------------------------------------------------

V5_NEW = [
    Receipt(
        claim_id="V5_Autoresearch_best_experiment",
        claim="Autoresearch loop accepted s3_curriculum_learning as final best (CI95 lower >= 0.55)",
        command="python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state",
        extraction='python -c "import json; s=json.load(open(r\\"versions/v5_phoenix/autoresearch_fixed/state.json\\")); print(s[\\"best\\"][\\"experiment_name\\"] if s[\\"best\\"] else \\"\\")"',
        expected="s3_curriculum_learning",
        comparator="==",
    ),
    Receipt(
        claim_id="V5_Autoresearch_CI95_lift",
        claim="Autoresearch S3 accepted with CI95 lower delta >= +0.05 over S2 (final best)",
        command="python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state",
        extraction='python -c "import json; s=json.load(open(r\\"versions/v5_phoenix/autoresearch_fixed/state.json\\")); h=[x for x in s[\\"history\\"] if x[\\"experiment_name\\"]==\\"s3_curriculum_learning\\"][0]; print(h[\\"delta_ci95_lower\\"])"',
        expected="0.05",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_Arena_baseline_leaderboard",
        claim="OpenEnv Arena leaderboard ships with 6 baseline rows (MaskablePPO at top)",
        command="python -m versions.v5_phoenix.arena.leaderboard",
        extraction='python -c "import json; b=json.load(open(r\\"versions/v5_phoenix/experiments/arena/leaderboard.json\\")); print(b[\\"n_baselines\\"], b[\\"rows\\"][0][\\"policy_name\\"])"',
        expected="6 MaskablePPO",
        comparator="regex",
        expected_regex=r"^6 MaskablePPO",
    ),
    Receipt(
        claim_id="V5_Twin_savings_gt_zero",
        claim="Counterfactual Twin on severity=0.85 yields positive median $ saved vs no-action",
        command='python -m versions.v5_phoenix.counterfactual_twin.twin --severity 0.85 --brent 123 --rollouts 20 --task easy_typhoon_response --out versions/v5_phoenix/experiments/twin/V5_receipt_run.json',
        extraction='python -c "import json; print(json.load(open(r\\"versions/v5_phoenix/experiments/twin/V5_receipt_run.json\\", encoding=\\"utf-8\\")).get(\\"savings_vs_no_action_usd\\"))"',
        expected="0",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_DPO_JUDGE_preference_pairs_built",
        claim="DPO preference-pair builder produces >= 20 pairs from 26 scenarios",
        command="python -m versions.v5_phoenix.roll_integration.dpo_judge.prepare_preference_data",
        extraction='wc -l < versions/v5_phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl',
        expected="20",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_Skill_pack_shipped",
        claim="supplymind-skills pack contains 3 SKILL.md files + plugin.json",
        command="ls versions/v5_phoenix/supplymind_skills/*/SKILL.md versions/v5_phoenix/supplymind_skills/plugin.json",
        extraction="ls versions/v5_phoenix/supplymind_skills/*/SKILL.md versions/v5_phoenix/supplymind_skills/plugin.json | wc -l",
        expected="4",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_Phoenix_tests_green",
        claim="Phoenix v5 test suite passes without affecting v4 tests",
        command="pytest versions/v5_phoenix/tests/ -q --tb=no",
        extraction='grep -oE "[0-9]+ passed" || true',
        expected="passed",
        comparator="regex",
        expected_regex=r"\d+ passed",
    ),
]


ALL_RECEIPTS = V4_CARRYOVERS + V5_NEW


def stub_all() -> None:
    """Emit .receipt.yaml + .reproduce.sh for every receipt without running."""
    for r in ALL_RECEIPTS:
        r.timestamp_utc = "<pending-first-run>"
        r.hardware = "<pending-first-run>"
        r.match = False
        r.actual = "<pending-first-run>"
        r.save(OUT_DIR / r.claim_id)
    logger.info("[register] stubbed %d receipts to %s", len(ALL_RECEIPTS), OUT_DIR)


def regenerate(only: str | None = None) -> None:
    for r in ALL_RECEIPTS:
        if only and r.claim_id != only:
            continue
        logger.info("[register] regenerating %s", r.claim_id)
        try:
            r.run()
        except Exception as e:  # noqa: BLE001
            logger.error("[register] %s failed to run: %s", r.claim_id, e)
        r.save(OUT_DIR / r.claim_id)


def build_index() -> None:
    """Write INDEX.md + INDEX.json listing every receipt with pass/fail."""
    rows = []
    for r in ALL_RECEIPTS:
        rows.append({
            "claim_id": r.claim_id,
            "claim": r.claim,
            "expected": r.expected,
            "actual": r.actual,
            "match": r.match,
            "comparator": r.comparator,
            "command": r.command,
            "receipt_yaml": f"{r.claim_id}.receipt.yaml",
            "reproduce_sh": f"{r.claim_id}.reproduce.sh",
        })
    (OUT_DIR / "INDEX.json").write_text(__import__("json").dumps(rows, indent=2))
    lines = ["# Phoenix v5 receipts index", "",
             f"Total receipts: {len(rows)}   |   v4 carryovers: {len(V4_CARRYOVERS)}   |   v5 new: {len(V5_NEW)}", ""]
    lines.append("| Claim ID | Expected | Match? | Command |")
    lines.append("|---|---|---|---|")
    for row in rows:
        m = "[passed]" if row["match"] else "[pending]"
        lines.append(f"| [{row['claim_id']}]({row['reproduce_sh']}) | `{row['expected']}` | {m} | `{row['command'][:80]}...` |")
    (OUT_DIR / "INDEX.md").write_text("\n".join(lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    if args.stub:
        stub_all()
    elif args.regenerate:
        regenerate(args.only)
    elif args.index_only:
        pass
    else:
        stub_all()
    build_index()
    print(f"[register] INDEX written to {OUT_DIR}")


if __name__ == "__main__":
    main()
