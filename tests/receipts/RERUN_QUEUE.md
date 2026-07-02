# RERUN QUEUE — receipts awaiting real regeneration (P1.4 follow-up)

Produced by Wave-3 evidence-refresh agent, 2026-07-02. This lists receipts that
**should exist from a real generator but could not be regenerated now** because
they are blocked. P1 picks these up once the blocker clears. Nothing here is a
fake — fakes were deleted (see agent summary / deletion list). These are honest
gaps waiting on a resource.

Blocker tags: `openrouter_key` (key in .env is REVOKED, all calls 401) ·
`ollama` (daemon + local models must be up) · `live_keyed_api` (EIA/FRED/NASA
FIRMS/GFW/NewsAPI/NOAA — do not burn during solidification) · `hf_space`
(live rollout vs the deployed Space) · `wandb` (W&B write) · `gpu` /
`long_run` (training).

## Blocked receipts

| Receipt (tests/receipts + FINAL_SUBMIT mirror) | Generator | Blocked on | Note |
|---|---|---|---|
| frontier_panel_alpha.json | scripts/run_frontier_judge_panel.py | openrouter_key | 3-judge Krippendorff alpha over OpenRouter |
| cross_corpus_alpha.json | scripts/compute_cross_corpus_alpha.py | openrouter_key | cross-corpus agreement, OpenRouter judges |
| R4_FRONTIER_PANEL_V2.json | frontier panel (OpenRouter) | openrouter_key | 427KB panel transcript; stale until re-run |
| openrouter_liveness.json | scripts/verify_openrouter_models.py | openrouter_key | current file shows 4/14 OK — regenerate after key rotation |
| ollama_v5_vs_frontier.json | scripts/ollama_v5_vs_frontier.py | openrouter_key + ollama | analyst-v5 (local) vs frontier (OpenRouter) A/B |
| pass28_A_local_scenario_extractor.json | scripts/pass28_killshot_v2.py block A | ollama | qwen2.5:14b local extraction |
| pass28_B_six_judge_panel.json | scripts/pass28_killshot_v2.py block B | ollama | 6 local judge models (qwen2.5:14b, deepseek-r1, mistral-nemo, analyst:v5, gemma, qwen-coder) |
| dual_verifier_smoke.json | wordle dual_verifier | ollama | dual-verifier smoke |
| pass28_K1_fred_brent_real.json | scripts/pass28_keys_ingest.py K1 | live_keyed_api (FRED) | 8/8 events w/ real DCOILBRENTEU |
| pass28_K2_newsapi_live_ingest.json | scripts/pass28_keys_ingest.py K2 | live_keyed_api (NewsAPI) | live headline ingest |
| pass28_K3_noaa_cdo_live.json | scripts/pass28_keys_ingest.py K3 | live_keyed_api (NOAA) | CDO endpoints |
| api_keys_live_proof.json | scripts/final_validation_bundle.py / pass27 | live_keyed_api (EIA/FIRMS/GFW/OpenRouter) | multi-key liveness proof |
| ensemble_brent_validation.json | scripts/validate_ensemble_brent.py | live_keyed_api (FRED) | audit L9: must use real FRED, not synthetic sinusoid |
| pass27_E_mirror_v2_keys.json | scripts/pass27_killshot.py block E | live_keyed_api | key mirror |
| pass27_F_gfw_honesty.json | scripts/pass27_killshot.py block F | live_keyed_api (GFW) | GFW honesty probe |
| pass26_live_supplymind_rollout.json (+ plots/supplymind_live_rollout.png) | scripts/pass26_real_evidence_expansion.py | hf_space | live rollout vs deployed Space (last run: 0/28 steps 200 OK) |
| pass27_A_fixed_hf_rollout.json | scripts/pass27_killshot.py block A | hf_space | HF-space rollout |
| pass28_C_hard_tier_rollout.json | scripts/pass28_killshot_v2.py block C | hf_space | hard-tier 60-step rollout |
| pass25_hf_space_deep_probe.json | scripts/pass25_* | hf_space | deep endpoint probe |
| plots/colab_reproduction.png | scripts/pass23_colab_local_smoke.py | hf_space / long_run | colab reproduction figure |
| (deleted) pass28_K4_wandb_smoke.json | pass28 K4 | wandb + gpu | fabricated curve deleted; only regenerate from a REAL training loop's metrics |
| wordle_real_reinforce_v2_curve.json (+ plots/real_reinforce_curve_v2.png) | scripts/final_real_reinforce_wordle_v2.py | long_run (cpu/gpu) | real 1500-ep REINFORCE; heavy |
| pass28_J_reinforce_longer.json | scripts/pass28_killshot_v2.py block J | long_run | 3000-ep REINFORCE |

## Permanently RETIRED (deleted, honest replacement already in tree — do NOT rerun)

| Deleted receipt | Honest replacement (kept) |
|---|---|
| bootstrap_leaderboard.json, wilcoxon_pairwise_leaderboard.json | pass27_B_real_episodic_bootstrap.json (real per-episode arrays, valid stats) |
| conformal_tight_v3.json, conformal_multilevel.json, pass27_G_conformal_v3_full.json, pass28_E_conformal_32k.json | conformal_calibration.json (calibrate_conformal_from_harvest.py — real harvested trajectories) |
| v2_inferential_stats.json, statistical_power_analysis.json | pass27_B_real_episodic_bootstrap.json (real paired stats) |
| tier3_generalization.json | pass27_C_tier3_degradation.json (honest OOD scaling) |
| process_supervision.json, pass26_process_supervision_concrete.json | pass28_F_process_super_plot.json + plots/process_supervision_step_credit.png (real _score_guess) |
| cross_env_transfer.json, pass28_G_cross_env_transfer.json | none — feature was fabricated (random featurizers / invented letter→SKU map); needs real env encoders (gpu/long_run) if ever revived |
| master_audit_summary_pass20 … pass28_v9_FINAL.json | none — regenerate a fresh honest inventory in P2.2 (scripts/verify_claims.py) from surviving real receipts only |

## Refreshed THIS pass (real, no blocker)
| Receipt | New sha256 |
|---|---|
| adversarial_20_attack_gauntlet.json (tests + FINAL_SUBMIT) | d447e350cf17a57cf233337b711f7f6774c4292b6fd573d5544e9e026b58cb89 |

verify_ollama_finetuning_stack.py: re-ran offline, exit 0, all checks pass (writes no receipt — stdout gate only).
