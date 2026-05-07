# Phoenix v5 receipts index

Total receipts: 20   |   v4 carryovers: 13   |   v5 new: 7

| Claim ID | Expected | Match? | Command |
|---|---|---|---|
| [R5_GRANITE_mxbai_P1](R5_GRANITE_mxbai_P1.reproduce.sh) | `0.9622` | [pending] | `python -m v3_arcadia.40_granite.r5_rag_beast --pipeline mxbai_bi --out /tmp/r5_g...` |
| [R5_GRANITE_mxbai_MRR](R5_GRANITE_mxbai_MRR.reproduce.sh) | `0.9780` | [pending] | `python -m v3_arcadia.40_granite.r5_rag_beast --pipeline mxbai_bi --out /tmp/r5_g...` |
| [R5_BEIR_snowflake_nDCG10](R5_BEIR_snowflake_nDCG10.reproduce.sh) | `0.971` | [pending] | `python -m v3_arcadia.40_granite.r5_manual_beir --out /tmp/r5_beir.json...` |
| [R4_2JUDGE_Krippendorff_alpha](R4_2JUDGE_Krippendorff_alpha.reproduce.sh) | `0.7499` | [pending] | `python -m v3_arcadia.30_dangerous.r4_ablation_and_baseline --out /tmp/r4_ab.json...` |
| [R4_Cohen_kappa_QwenMistral](R4_Cohen_kappa_QwenMistral.reproduce.sh) | `0.747` | [pending] | `python -m v3_arcadia.30_dangerous.r4_ablation_and_baseline --out /tmp/r4_kappa.j...` |
| [R6_MaskingAblation_easy_lift](R6_MaskingAblation_easy_lift.reproduce.sh) | `26.77` | [pending] | `python -m v3_arcadia.50_gethsemane.r6_unmasked_ablation --out /tmp/r6_mask.json...` |
| [R6_GCN_easy_MAE_vs_MLP](R6_GCN_easy_MAE_vs_MLP.reproduce.sh) | `48.0247` | [pending] | `python -m v3_arcadia.70_provider.r6_gnn_arrival_time --out /tmp/r6_gnn.json...` |
| [R6_AquaRegia_WTI_dev95](R6_AquaRegia_WTI_dev95.reproduce.sh) | `0.0238` | [pending] | `python -m v3_arcadia.80_aqua_regia.r6_per_horizon_conformal --out /tmp/r6_aqua.j...` |
| [R3_TimesFM_CP_WTI_dev95](R3_TimesFM_CP_WTI_dev95.reproduce.sh) | `0.050` | [pending] | `python -m v3_arcadia.20_past_self.r3_timesfm_residual_quantile --out /tmp/r3_tfm...` |
| [V4_SPOF_V2_F1](V4_SPOF_V2_F1.reproduce.sh) | `1.0` | [pending] | `python -m versions.v4_arcadia_live.features.spof_v2 --eval-all --out /tmp/spof.json...` |
| [V4_STACKING_V2_lift_vs_WV](V4_STACKING_V2_lift_vs_WV.reproduce.sh) | `0.001` | [pending] | `python -m versions.v4_arcadia_live.features.stacking_v2 --out /tmp/stack.json...` |
| [V4_Live_Brent_202604](V4_Live_Brent_202604.reproduce.sh) | `60` | [pending] | `python -m versions.v4_arcadia_live.realtime.sources.fred_brent --latest-only...` |
| [V4_Tests_Total](V4_Tests_Total.reproduce.sh) | `249` | [pending] | `pytest tests/ versions/v4_arcadia_live/tests/ -q --tb=no...` |
| [V5_Autoresearch_best_experiment](V5_Autoresearch_best_experiment.reproduce.sh) | `s3_curriculum_learning` | [pending] | `python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state...` |
| [V5_Autoresearch_CI95_lift](V5_Autoresearch_CI95_lift.reproduce.sh) | `0.05` | [pending] | `python -m versions.v5_phoenix.autoresearch_fixed.rebuild_state...` |
| [V5_Arena_baseline_leaderboard](V5_Arena_baseline_leaderboard.reproduce.sh) | `6 MaskablePPO` | [pending] | `python -m versions.v5_phoenix.arena.leaderboard...` |
| [V5_Twin_savings_gt_zero](V5_Twin_savings_gt_zero.reproduce.sh) | `0` | [pending] | `python -m versions.v5_phoenix.counterfactual_twin.twin --severity 0.85 --brent 123 -...` |
| [V5_DPO_JUDGE_preference_pairs_built](V5_DPO_JUDGE_preference_pairs_built.reproduce.sh) | `20` | [pending] | `python -m versions.v5_phoenix.roll_integration.dpo_judge.prepare_preference_data...` |
| [V5_Skill_pack_shipped](V5_Skill_pack_shipped.reproduce.sh) | `4` | [pending] | `ls versions/v5_phoenix/supplymind_skills/*/SKILL.md versions/v5_phoenix/supplymind_skill...` |
| [V5_Phoenix_tests_green](V5_Phoenix_tests_green.reproduce.sh) | `passed` | [pending] | `pytest versions/v5_phoenix/tests/ -q --tb=no...` |
