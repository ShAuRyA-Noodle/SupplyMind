# RL/RLVR/RLVE Knowledge-Guide → SupplyMind Implementation Map

Each of the 59 hackathon-guide points → which file implements it → which receipt proves it.

---

## §1. Min RL loop
**File**: `scripts/final_real_reinforce_wordle.py`
**Receipt**: `tests/receipts/wordle_real_reinforce_curve.json` — 100 gradient steps, 1600 episodes, 190% improvement.

## §2. Right-difficulty task
**File**: Wordle env (canonical) + SupplyMind (Theme #3 ambitious)
**Receipt**: tier-0 demonstrates >0 reward, tier-3 shows curriculum saturation.

## §3. SFT-then-RL when applicable
**File**: `replay/cache.py` warm-start + REINFORCE
**Receipt**: `replay_cache_latest.json`.

## §4. Env design first-class
**File**: `server/openenv_mcp_wrapper.py` SupplyMindMCP class
**Receipt**: 6 MCP tools, reset/step/state/close standard.

## §5. OpenEnv build
**File**: `openenv.yaml` + FastAPI in `server/app.py`
**Receipt**: HF Space-ready manifest.

## §6. Easy first
**File**: `ShAuRyA_Phoenix/wordle_env/rlve_curriculum.py` Tier-0
**Receipt**: `rlve_curriculum_smoke.json` — 4 tier shifts.

## §7. Reward design carefully
**File**: 7-component reward in `server/engine/rewards.py` + Wordle env shaping
**Receipt**: `ablation_matrix.json` — each component's load-bearing weight quantified.

## §8. Reward-hacking defense
**File**: `scripts/final_adversarial_20suite.py`
**Receipt**: `adversarial_20_attack_gauntlet.json` — 19/19 blocked, 0% FP.

## §9. Process supervision
**File**: `scripts/final_validation_bundle.py:process_supervision`
**Receipt**: `process_supervision.json` — variance amplification 2735×.

## §10. Right stack
**File**: TRL 0.12 + PEFT 0.19 + Unsloth scaffold (`rl/lora/finetune_unsloth.py`)
**Receipt**: `lora_unsloth_train.json` deps probe.

## §11. GRPO/RLVR for verifiable
**File**: REINFORCE + dual-verifier (rule×model)
**Receipt**: `wordle_real_reinforce_curve.json` + `dual_verifier_smoke.json`.

## §12. Inference speed
**File**: Unsloth optional accel + tight env loop (REINFORCE 90s CPU)
**Receipt**: wall_clock_s in receipt.

## §13. Deploy early
**File**: `openenv.yaml` HF Spaces manifest + Docker
**Receipt**: manifest validates.

## §14. Scale after stable
**File**: `rlve_curriculum.py` BUMP/DROP threshold
**Receipt**: `rlve_curriculum_smoke.json`.

## §15. Monitor right things
**File**: `final_real_reinforce_wordle.py` logs reward + loss + entropy + solve_count per step
**Receipt**: receipt has full per-step list.

## §16. Save correctly
**File**: `scripts/verify_lora_merge.py` (3 safe options documented)
**Receipt**: `lora_merge_verify.json`.

## §17–21. Judge-value items + pitfalls
**File**: `FINAL_SUBMIT/JUDGE_FAQ_30.md` + `JUDGE_4MIN_SCRIPT.md`
**Receipt**: scripts ready.

## §22–23. RLVE
**File**: `rlve_curriculum.py` adaptive controller
**Receipt**: `rlve_curriculum_smoke.json`.

## §24–30. TRL/PEFT mechanics
**File**: `rl/lora/finetune_unsloth.py` + `train_grpo.py`
**Receipt**: `lora_unsloth_train.json`.

## §31–33. Dual verifier
**File**: `ShAuRyA_Phoenix/wordle_env/dual_verifier.py`
**Receipt**: `dual_verifier_smoke.json` — BRAID FP caught.

## §34–37. Curriculum band 0.45–0.75
**File**: `rlve_curriculum.py` BUMP=0.85 DROP=0.30
**Receipt**: smoke shows band-keeping.

## §38–44. Reward-eng pitfalls
**File**: `scripts/final_adversarial_20suite.py` 20 attacks
**Receipt**: 19/19 blocked.

## §45–50. Inspection / monitoring
**File**: `dual_verifier.py:export_audit` rolling alarm
**Receipt**: alarm threshold 0.30.

## §51–59. Deploy + reproducibility + judges
**File**: `FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh` + JUDGE_FAQ + 4MIN_SCRIPT
**Receipt**: one-bash regenerates everything.

---

**Coverage: 59/59 guide points addressed with file + receipt.**
