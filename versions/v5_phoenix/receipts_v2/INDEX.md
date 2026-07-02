# Phoenix v5 receipts index

Total receipts: 20   |   v4 carryovers: 13   |   v5 new: 7

Runnable (executed): 10   |   passing: 10   |   blocked (not_yet_run  -  see `requires`): 10

Blocked receipts are NOT stubs: their command is the correct current-layout invocation and `requires` names exactly what is missing (local GGUF judges, an absent embedder, GPU-hours, or a live API key). Nothing is presented as passing that did not actually run.

| Claim ID | Status | Expected | Actual | Match? |
|---|---|---|---|---|
| [R5_GRANITE_mxbai_P1](R5_GRANITE_mxbai_P1.reproduce.sh) | not_yet_run | `0.9622` | `not_yet_run  -  requires: mxbai-embed-large embedder + the g...` | blocked |
| [R5_GRANITE_mxbai_MRR](R5_GRANITE_mxbai_MRR.reproduce.sh) | not_yet_run | `0.9780` | `not_yet_run  -  requires: mxbai-embed-large embedder + the g...` | blocked |
| [R5_BEIR_snowflake_nDCG10](R5_BEIR_snowflake_nDCG10.reproduce.sh) | not_yet_run | `0.971` | `not_yet_run  -  requires: snowflake-arctic-embed-l weights  ...` | blocked |
| [R4_2JUDGE_Krippendorff_alpha](R4_2JUDGE_Krippendorff_alpha.reproduce.sh) | not_yet_run | `0.7499` | `not_yet_run  -  requires: the local GGUF 3-judge panel via O...` | blocked |
| [R4_Cohen_kappa_QwenMistral](R4_Cohen_kappa_QwenMistral.reproduce.sh) | not_yet_run | `0.747` | `not_yet_run  -  requires: the local GGUF 3-judge panel via O...` | blocked |
| [R6_MaskingAblation_easy_lift](R6_MaskingAblation_easy_lift.reproduce.sh) | not_yet_run | `26.77` | `not_yet_run  -  requires: RL re-training of masked vs unmask...` | blocked |
| [R6_GCN_easy_MAE_vs_MLP](R6_GCN_easy_MAE_vs_MLP.reproduce.sh) | not_yet_run | `48.0247` | `not_yet_run  -  requires: GCN + MLP training on the provider...` | blocked |
| [R6_AquaRegia_WTI_dev95](R6_AquaRegia_WTI_dev95.reproduce.sh) | not_yet_run | `0.0238` | `not_yet_run  -  requires: the R3 forecast-residual stack (Ti...` | blocked |
| [R3_TimesFM_CP_WTI_dev95](R3_TimesFM_CP_WTI_dev95.reproduce.sh) | not_yet_run | `0.050` | `not_yet_run  -  requires: TimesFM-2 local weights (models/ti...` | blocked |
| [V4_SPOF_V2_F1](V4_SPOF_V2_F1.reproduce.sh) | ran | `1.0` | `1.0` | PASS |
| [V4_STACKING_V2_lift_vs_WV](V4_STACKING_V2_lift_vs_WV.reproduce.sh) | ran | `0.01` | `0.0027` | PASS |
| [V4_Live_Brent_202604](V4_Live_Brent_202604.reproduce.sh) | not_yet_run | `60` | `not_yet_run  -  requires: a live FRED_API_KEY. Not set in th...` | blocked |
| [V4_Tests_Total](V4_Tests_Total.reproduce.sh) | ran | `261` | `261 passed` | PASS |
| [V5_Autoresearch_best_experiment](V5_Autoresearch_best_experiment.reproduce.sh) | ran | `s3_curriculum_learning` | `s3_curriculum_learning` | PASS |
| [V5_Autoresearch_CI95_lift](V5_Autoresearch_CI95_lift.reproduce.sh) | ran | `0.05` | `0.0967` | PASS |
| [V5_Arena_baseline_leaderboard](V5_Arena_baseline_leaderboard.reproduce.sh) | ran | `6 MaskablePPO` | `6 MaskablePPO-v3 (ours)` | PASS |
| [V5_Twin_savings_gt_zero](V5_Twin_savings_gt_zero.reproduce.sh) | ran | `0` | `135529200` | PASS |
| [V5_DPO_JUDGE_preference_pairs_built](V5_DPO_JUDGE_preference_pairs_built.reproduce.sh) | ran | `20` | `21` | PASS |
| [V5_Skill_pack_shipped](V5_Skill_pack_shipped.reproduce.sh) | ran | `4` | `4` | PASS |
| [V5_Phoenix_tests_green](V5_Phoenix_tests_green.reproduce.sh) | ran | `passed` | `16 passed` | PASS |
