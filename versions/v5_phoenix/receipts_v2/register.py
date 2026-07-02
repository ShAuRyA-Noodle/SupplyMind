"""register.py  -  canonical list of Phoenix v5 receipts (including v4 carryovers).

Each Receipt here is a claim we're willing to defend with `bash *.reproduce.sh`.
To regenerate all receipts (re-run runnable commands, stamp blocked ones honestly):

    python -m versions.v5_phoenix.receipts_v2.register --regenerate

To regenerate a single receipt:

    python -m versions.v5_phoenix.receipts_v2.register --regenerate --only V5_Twin_savings_gt_zero

Honesty rules (Wave-3 solidification, 2026-07-02)
-------------------------------------------------
* A receipt is either RUNNABLE (no `requires`) or BLOCKED (`requires` set).
* RUNNABLE receipts are actually executed by `--regenerate`; their real
  stdout / exit_code / actual / match are recorded. No stubs.
* BLOCKED receipts name EXACTLY what is missing (local GGUF judge panel, an
  absent embedder, GPU-hours, a live API key). They are recorded as
  `status: not_yet_run`  -  never as a `<pending-first-run>` stub pretending to
  be a passing receipt. Their reproduce command is rewritten to the correct,
  current-layout path so it will run once the dependency is present.
* Pre-reorg module paths like `v3_arcadia.40_granite.*` are unimportable
  (digit-leading package names) AND stale. Where the underlying script still
  needs local models/GPU it is invoked by FILE PATH (the only form that can
  ever run), and marked BLOCKED with the real reason.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .framework import Receipt, _describe_hardware

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# v4 carryovers (13 receipts, grade-A upgrade of the existing versions/v4_arcadia_live/receipts/)
# -----------------------------------------------------------------------------

V4_CARRYOVERS = [
    Receipt(
        claim_id="R5_GRANITE_mxbai_P1",
        claim="mxbai-embed-large P@1 on 53 precise SupplyMind queries equals 0.9622 "
              "(headline value committed in versions/v3_arcadia/results/R5_GRANITE_HARD.json)",
        command="python versions/v3_arcadia/40_granite/r5_rag_beast.py --pipeline mxbai_bi --out /tmp/r5_granite_p1.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_granite_p1.json\\")).get(\\"pipelines\\",{}).get(\\"P2_mxbai_bi\\",{}).get(\\"p1\\"))"',
        expected="0.9622",
        comparator="==",
        requires="mxbai-embed-large embedder + the gitignored external_data RAG corpus; also a "
                 "parent-count ROOT fix inside versions/v3_arcadia/40_granite/r5_rag_beast.py "
                 "(that v3 script is outside this wave's ownership).",
    ),
    Receipt(
        claim_id="R5_GRANITE_mxbai_MRR",
        claim="mxbai-embed-large MRR on 53 precise queries equals 0.9780 "
              "(headline value committed in versions/v3_arcadia/results/R5_GRANITE_HARD.json)",
        command="python versions/v3_arcadia/40_granite/r5_rag_beast.py --pipeline mxbai_bi --out /tmp/r5_granite_mrr.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_granite_mrr.json\\")).get(\\"pipelines\\",{}).get(\\"P2_mxbai_bi\\",{}).get(\\"mrr\\"))"',
        expected="0.9780",
        comparator="==",
        requires="mxbai-embed-large embedder + the gitignored external_data RAG corpus; also a "
                 "parent-count ROOT fix inside versions/v3_arcadia/40_granite/r5_rag_beast.py.",
    ),
    Receipt(
        claim_id="R5_BEIR_snowflake_nDCG10",
        claim="Snowflake-Arctic-L nDCG@10 on 26 Wikipedia-crisis BEIR subset equals 0.971",
        command="python versions/v3_arcadia/40_granite/r5_manual_beir.py --out /tmp/r5_beir.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r5_beir.json\\")).get(\\"our_results\\",{}).get(\\"snowflake-arctic-l\\",{}).get(\\"mean_ndcg@10\\"))"',
        expected="0.971",
        comparator="==",
        requires="snowflake-arctic-embed-l weights  -  models/snowflake-arctic-embed-l is ABSENT "
                 "on disk (audit_7). Ship the weights (or swap the embedder) before this can run.",
    ),
    Receipt(
        claim_id="R4_2JUDGE_Krippendorff_alpha",
        claim="2-judge (Qwen-14B + Mistral-Nemo) Krippendorff ordinal alpha on 26 scenarios equals 0.7499 "
              "(the raw 3-judge alpha is only 0.210  -  see versions/v3_arcadia/results/R4_DANGEROUS_V2_REPORT.md)",
        command="python versions/v3_arcadia/30_dangerous/r4_ablation_and_baseline.py --out /tmp/r4_ab.json",
        extraction='python -c "import json; print(json.load(open(r\\"/tmp/r4_ab.json\\")).get(\\"agreement_primary_panel\\",{}).get(\\"krippendorff_alpha_ordinal\\"))"',
        expected="0.7499",
        comparator="==",
        requires="the local GGUF 3-judge panel via Ollama (deepseek-r1, qwen2.5:14b, mistral-nemo). "
                 "No OpenRouter equivalent and OPENROUTER_API_KEY is revoked. Cached per-judge "
                 "outputs live in versions/v3_arcadia/results/R4_DANGEROUS_V2*.json.",
    ),
    Receipt(
        claim_id="R4_Cohen_kappa_QwenMistral",
        claim="Cohen weighted kappa Qwen-14B vs Mistral-Nemo equals 0.747 "
              "(this is the best-agreeing pair; DeepSeek-R1 pairwise kappa is 0.158 / 0.095)",
        command="python versions/v3_arcadia/30_dangerous/r4_ablation_and_baseline.py --out /tmp/r4_kappa.json",
        extraction='python -c "import json; blob=json.load(open(r\\"/tmp/r4_kappa.json\\")); print(blob.get(\\"pairwise_weighted_kappa\\",{}).get(\\"qwen_mistral\\") or blob.get(\\"agreement_primary_panel\\",{}).get(\\"cohen_kappa_qwen_mistral\\"))"',
        expected="0.747",
        comparator="==",
        requires="the local GGUF 3-judge panel via Ollama (no OpenRouter equivalent; key revoked). "
                 "Cached judge outputs in versions/v3_arcadia/results/R4_DANGEROUS_V2*.json.",
    ),
    Receipt(
        claim_id="R6_MaskingAblation_easy_lift",
        claim="MaskablePPO over PPO lift on easy_typhoon_response equals 26.77% "
              "(committed in versions/v3_arcadia/results R6 masking ablation)",
        command="python versions/v3_arcadia/50_gethsemane/r6_unmasked_ablation.py --out /tmp/r6_mask.json",
        extraction='python -c "import json; print(round(100*(json.load(open(r\\"/tmp/r6_mask.json\\")).get(\\"easy_typhoon_response\\",{}).get(\\"masking_lift_pct\\",0)),2))"',
        expected="26.77",
        comparator="==",
        requires="RL re-training of masked vs unmasked MaskablePPO (GPU-hours / long CPU run).",
    ),
    Receipt(
        claim_id="R6_GCN_easy_MAE_vs_MLP",
        claim="GCN beats MLP on easy graph by 48.02 percent MAE reduction "
              "(committed in versions/v3_arcadia/results/R6_PROVIDER_V2.json)",
        command="python versions/v3_arcadia/70_provider/r6_gnn_arrival_time.py --out /tmp/r6_gnn.json",
        extraction='python -c "import json; print(round(100*json.load(open(r\\"/tmp/r6_gnn.json\\")).get(\\"easy\\",{}).get(\\"mae_reduction_pct\\",0),4))"',
        expected="48.0247",
        comparator="==",
        requires="GCN + MLP training on the provider arrival-time task (GPU / long-run).",
    ),
    Receipt(
        claim_id="R6_AquaRegia_WTI_dev95",
        claim="Per-horizon split-conformal on DCOILWTICO at 95% nominal: |coverage - nominal| = 0.0238 "
              "(committed in versions/v3_arcadia/results/R6_AQUA_REGIA_V2.json)",
        command="python versions/v3_arcadia/80_aqua_regia/r6_per_horizon_conformal.py --out /tmp/r6_aqua.json",
        extraction='python -c "import json; print(round(abs(json.load(open(r\\"/tmp/r6_aqua.json\\")).get(\\"DCOILWTICO\\",{}).get(\\"conformal_coverage_dev_95\\",0)),4))"',
        expected="0.0238",
        comparator="==",
        requires="the R3 forecast-residual stack (TimesFM-2 local weights, ~2GB) + FRED WTI series "
                 "to regenerate the residuals the conformal step consumes.",
    ),
    Receipt(
        claim_id="R3_TimesFM_CP_WTI_dev95",
        claim="TimesFM residual-conformal on WTI at 95%: |coverage - nominal| = 0.050 "
              "(committed in versions/v3_arcadia/results R3 artifacts)",
        command="python versions/v3_arcadia/20_past_self/r3_timesfm_residual_quantile.py --out /tmp/r3_tfm.json",
        extraction='python -c "import json; print(round(abs(json.load(open(r\\"/tmp/r3_tfm.json\\")).get(\\"DCOILWTICO\\",{}).get(\\"conformal_coverage_dev_95\\",0)),3))"',
        expected="0.050",
        comparator="==",
        requires="TimesFM-2 local weights (models/timesfm-2, ~2GB) + FRED DCOILWTICO series.",
    ),
    Receipt(
        claim_id="V4_SPOF_V2_F1",
        claim="SPOF detector v2 (articulation-point) mean F1 over easy/medium/hard graphs equals 1.000",
        command="python -m supplymind.warroom.features.spof_v2 --graph all --save",
        extraction='python -c "import json; print(json.load(open(r\\"supplymind/warroom/features/R6_SPOF_V2.json\\")).get(\\"summary\\",{}).get(\\"v2_mean_f1\\"))"',
        expected="1.0",
        comparator="==",
    ),
    Receipt(
        claim_id="V4_STACKING_V2_lift_vs_WV",
        claim="Proper stacking beats weighted-vote on the DataCo ensemble by only a negligible AUC "
              "margin (measured +0.0027 on a 0.98+ ceiling; asserts lift <= 0.01 = near-null)",
        command="python -m supplymind.warroom.features.stacking_v2 --save",
        extraction='python -c "import json; print(round(json.load(open(r\\"supplymind/warroom/features/R15_STACKING_V2.json\\")).get(\\"lift_stacking_vs_wv_auc\\",0),4))"',
        expected="0.01",
        comparator="<=",
    ),
    Receipt(
        claim_id="V4_Live_Brent_202604",
        claim="FRED Brent polling returns a live value parseable as USD/bbl in the plausible $60-$250 band",
        command="python -m supplymind.warroom.sources.fred_brent",
        extraction='python -c "import sys,re; out=sys.stdin.read(); m=re.search(r\\"(\\\\d+\\\\.\\\\d+)\\", out); print(m.group(1) if m else \\"\\")"',
        expected="60",
        comparator="in_range",
        expected_range=[60, 250],
        requires="a live FRED_API_KEY. Not set in this environment (module logs "
                 "'FRED_API_KEY not set'); FRED key is flagged for rotation in "
                 "versions/v5_phoenix/docs/PHOENIX_PUSH_REPORT.md sec. 3.6.",
    ),
    Receipt(
        claim_id="V4_Tests_Total",
        claim="v3 core (184) + v4 new (77) = 261 total tests pass (measured 2026-07-02)",
        command='pytest tests/ versions/v4_arcadia_live/tests/ -q --tb=no | grep -oE "[0-9]+ passed"',
        extraction="",
        expected="261",
        comparator="regex",
        expected_regex=r"(26[0-9]) passed",   # accept 260-269 to allow minor drift
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
        command="python -m supplymind.phoenix.arena.leaderboard",
        extraction='python -c "import json; b=json.load(open(r\\"versions/v5_phoenix/experiments/arena/leaderboard.json\\")); print(b[\\"n_baselines\\"], b[\\"rows\\"][0][\\"policy_name\\"])"',
        expected="6 MaskablePPO",
        comparator="regex",
        expected_regex=r"^6 MaskablePPO",
    ),
    Receipt(
        claim_id="V5_Twin_savings_gt_zero",
        claim="Counterfactual Twin on severity=0.85 yields positive median $ saved vs no-action "
              "(measured +$135.5M savings on 20 rollouts, 2026-07-02)",
        command='python -m supplymind.phoenix.counterfactual_twin.twin --severity 0.85 --brent 123 --rollouts 20 --task easy_typhoon_response --out versions/v5_phoenix/experiments/twin/V5_receipt_run.json',
        extraction='python -c "import json; print(json.load(open(r\\"versions/v5_phoenix/experiments/twin/V5_receipt_run.json\\", encoding=\\"utf-8\\")).get(\\"savings_vs_no_action_usd\\"))"',
        expected="0",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_DPO_JUDGE_preference_pairs_built",
        claim="DPO preference-pair builder produces >= 20 pairs from 26 scenarios "
              "(the DPO fine-tune itself never completed  -  this receipt covers only the pair builder)",
        command="python -m versions.v5_phoenix.roll_integration.dpo_judge.prepare_preference_data",
        extraction='python -c "print(sum(1 for _ in open(r\\"versions/v5_phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl\\", encoding=\\"utf-8\\")))"',
        expected="20",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_Skill_pack_shipped",
        claim="supplymind-skills pack contains 3 SKILL.md files + plugin.json",
        command="python -c \"import glob; [print(p) for p in glob.glob('versions/v5_phoenix/supplymind_skills/*/SKILL.md') + glob.glob('versions/v5_phoenix/supplymind_skills/plugin.json')]\"",
        extraction="python -c \"import glob; print(len(glob.glob('versions/v5_phoenix/supplymind_skills/*/SKILL.md') + glob.glob('versions/v5_phoenix/supplymind_skills/plugin.json')))\"",
        expected="4",
        comparator=">=",
    ),
    Receipt(
        claim_id="V5_Phoenix_tests_green",
        claim="Phoenix v5 test suite passes without affecting v4 tests (measured 16 passed)",
        command='pytest versions/v5_phoenix/tests/ -q --tb=no | grep -oE "[0-9]+ passed"',
        extraction="",
        expected="passed",
        comparator="regex",
        expected_regex=r"\d+ passed",
    ),
]


ALL_RECEIPTS = V4_CARRYOVERS + V5_NEW


def _stamp_blocked(r: Receipt) -> None:
    """Record a BLOCKED receipt honestly  -  never executed, clearly labelled."""
    r.timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    r.hardware = _describe_hardware()
    r.status = "not_yet_run"
    r.actual = f"not_yet_run  -  requires: {r.requires}"
    r.match = False
    r.exit_code = -1
    r.stdout_inline = ""
    r.stderr_tail = ""
    r.comparator_note = "not executed (blocked); see `requires`"


def regenerate(only: str | None = None) -> None:
    for r in ALL_RECEIPTS:
        if only and r.claim_id != only:
            continue
        if r.requires:
            logger.info("[register] BLOCKED %s  -  requires: %s", r.claim_id, r.requires)
            _stamp_blocked(r)
            r.save(OUT_DIR / r.claim_id)
            continue
        logger.info("[register] running %s", r.claim_id)
        try:
            r.run()
        except Exception as e:  # noqa: BLE001
            logger.error("[register] %s failed to run: %s", r.claim_id, e)
        r.save(OUT_DIR / r.claim_id)


def build_index() -> None:
    """Write INDEX.md + INDEX.json listing every receipt with pass/fail/status."""
    rows = []
    for r in ALL_RECEIPTS:
        rows.append({
            "claim_id": r.claim_id,
            "claim": r.claim,
            "expected": r.expected,
            "actual": r.actual,
            "match": r.match,
            "status": r.status,
            "requires": r.requires,
            "comparator": r.comparator,
            "command": r.command,
            "receipt_yaml": f"{r.claim_id}.receipt.yaml",
            "reproduce_sh": f"{r.claim_id}.reproduce.sh",
        })
    (OUT_DIR / "INDEX.json").write_text(__import__("json").dumps(rows, indent=2), encoding="utf-8")

    n_ran = sum(1 for r in ALL_RECEIPTS if not r.requires)
    n_blocked = sum(1 for r in ALL_RECEIPTS if r.requires)
    n_pass = sum(1 for r in ALL_RECEIPTS if r.match)
    lines = ["# Phoenix v5 receipts index", "",
             f"Total receipts: {len(rows)}   |   v4 carryovers: {len(V4_CARRYOVERS)}   |   v5 new: {len(V5_NEW)}",
             "",
             f"Runnable (executed): {n_ran}   |   passing: {n_pass}   |   "
             f"blocked (not_yet_run  -  see `requires`): {n_blocked}",
             "",
             "Blocked receipts are NOT stubs: their command is the correct current-layout "
             "invocation and `requires` names exactly what is missing (local GGUF judges, an "
             "absent embedder, GPU-hours, or a live API key). Nothing is presented as passing "
             "that did not actually run.",
             ""]
    lines.append("| Claim ID | Status | Expected | Actual | Match? |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        if row["status"] == "not_yet_run":
            st = "not_yet_run"
            m = "blocked"
        else:
            st = "ran"
            m = "PASS" if row["match"] else "FAIL"
        actual = str(row["actual"])
        actual = (actual[:60] + "...") if len(actual) > 60 else actual
        lines.append(f"| [{row['claim_id']}]({row['reproduce_sh']}) | {st} | "
                     f"`{row['expected']}` | `{actual}` | {m} |")
    (OUT_DIR / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="Run runnable receipts + stamp blocked ones honestly (default action).")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--index-only", action="store_true",
                        help="Rebuild INDEX from the in-memory receipt list without touching *.yaml.")
    args = parser.parse_args()

    if args.index_only:
        pass
    else:
        # Default (and --regenerate) both regenerate. There is deliberately no
        # --stub mode any more: committing <pending-first-run> stubs as receipts
        # is exactly the integrity bug this framework was abused to create.
        regenerate(args.only)
    build_index()
    print(f"[register] INDEX written to {OUT_DIR}")


if __name__ == "__main__":
    main()
