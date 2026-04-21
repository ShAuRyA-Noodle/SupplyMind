# SupplyMind Receipts — Verify Any Headline Number in 30 Seconds

*generated 2026-04-20T20:57:26Z from git SHA `02251e94dc01`*

| # | Number | Value | Verify |
|---|--------|-------|--------|
| R5_GRANITE_mxbai_P1 | RAG P@1 on 6,483-chunk real corpus, mxbai bi-encoder | `0.9622641509433962` | `bash receipts/R5_GRANITE_mxbai_P1.reproduce.sh` |
| R5_GRANITE_mxbai_MRR | RAG MRR on precise queries | `0.9779874213836477` | `bash receipts/R5_GRANITE_mxbai_MRR.reproduce.sh` |
| R5_BEIR_snowflake_nDCG10 | BEIR out-of-domain nDCG@10 (Snowflake) on 26 Wiki crisis art | `0.9709860394574094` | `bash receipts/R5_BEIR_snowflake_nDCG10.reproduce.sh` |
| R4_2JUDGE_Krippendorff_alpha | 2-judge panel Krippendorff ordinal alpha on 26 crisis scenar | `0.7499056959637873` | `bash receipts/R4_2JUDGE_Krippendorff_alpha.reproduce.sh` |
| R4_Cohen_kappa_QwenMistral | Cohen weighted kappa Qwen-14B x Mistral-Nemo | `0.7473841554559043` | `bash receipts/R4_Cohen_kappa_QwenMistral.reproduce.sh` |
| R6_MaskingAblation_easy_lift | MaskablePPO easy-task reward lift vs plain PPO (+%) | `26.768743400211196` | `bash receipts/R6_MaskingAblation_easy_lift.reproduce.sh` |
| R6_GCN_easy_MAE_vs_MLP | GNN easy-graph MAE reduction vs MLP baseline (%) | `48.0247837147887` | `bash receipts/R6_GCN_easy_MAE_vs_MLP.reproduce.sh` |
| R6_AquaRegia_WTI_dev95 | Per-horizon conformal deviation at 95% nominal, WTI ARIMA | `0.023809523809523836` | `bash receipts/R6_AquaRegia_WTI_dev95.reproduce.sh` |
| R3_TimesFM_CP_WTI_dev95 | TimesFM-CP WTI deviation from 95% nominal | `0.04999999999999993` | `bash receipts/R3_TimesFM_CP_WTI_dev95.reproduce.sh` |
| V4_SPOF_V2_F1 | v4 SPOF articulation-point F1 (mean across 3 graphs) | `1.0` | `bash receipts/V4_SPOF_V2_F1.reproduce.sh` |
| V4_STACKING_V2_lift_vs_WV | v4 Stacking v2 AUC lift vs ensemble weighted voting | `0.001` | `bash receipts/V4_STACKING_V2_lift_vs_WV.reproduce.sh` |
| V4_Live_Brent_202604 | FRED Brent crude spot price as ingested on 2026-04-21 ($/bbl | `123.28` | `bash receipts/V4_Live_Brent_202604.reproduce.sh` |
| V4_Tests_Total | Total test count across v3 + v4 | `tests/test_engine.py::TestSupp` | `bash receipts/V4_Tests_Total.reproduce.sh` |