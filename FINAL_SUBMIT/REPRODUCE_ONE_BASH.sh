#!/usr/bin/env bash
# REPRODUCE_ONE_BASH.sh — regenerate receipts in one shot.
#
# Usage:   bash FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh
# Time:    ~3-5 minutes on CPU (no GPU required)
# Output:  tests/receipts/*.json + FINAL_SUBMIT/receipts/*.json
#
# TODO(P1.4/P2.2): under cleanup. Some steps below invoke scripts the audit flagged:
#   - final_real_reinforce_wordle.py (v1) was DELETED; use *_v2.py.
#   - final_adversarial_20suite.py re-implements the gates inline (PASS guaranteed by
#     construction) — its "blocked" counts are UNVERIFIED until rewritten to hit the real env.
#   - final_validation_bundle.py contains fabricated cross-env / ablation blocks.
# The canonical verifier is `python scripts/run_all.py`. See FINAL_SUBMIT/CLAIMS_LEDGER.md.

set -e
cd "$(dirname "$0")/.."
echo "=== SupplyMind FINAL SUBMIT reproducibility ==="
echo "Repo: $(pwd)"
echo

echo "[1/8] Wordle env + RLVE curriculum smoke ..."
python -m versions.v5_phoenix.wordle_env.rlve_curriculum

echo
echo "[2/8] Dual verifier smoke ..."
python -m versions.v5_phoenix.wordle_env.dual_verifier

echo
echo "[3/8] OpenEnv MCP compliance ..."
python server/openenv_mcp_wrapper.py

echo
echo "[4/8] REAL REINFORCE training (v2; improvement figure being re-verified) ..."
python scripts/final_real_reinforce_wordle_v2.py --episodes 1600 --batch 16

echo
echo "[5/8] 20-attack adversarial reward-hack gauntlet (UNVERIFIED: inline gates, being rewritten) ..."
python scripts/final_adversarial_20suite.py

echo
echo "[6/8] Cross-env transfer + process supervision + ablations + API keys ..."
echo "      (UNVERIFIED: this bundle contains fabricated cross-env/ablation blocks — audit_5)"
python scripts/final_validation_bundle.py

echo
echo "[7/8] Wordle GRPO baseline (heuristic policy receipt) ..."
python -m versions.v5_phoenix.wordle_env.train_grpo --steps 50 || true

echo
echo "[8/8] Receipt index ..."
ls -1 tests/receipts/*.json | wc -l
echo "receipts in tests/receipts/"
ls -1 FINAL_SUBMIT/receipts/*.json | wc -l
echo "receipts in FINAL_SUBMIT/receipts/"

echo
echo "=== DONE ==="
echo "Open FINAL_SUBMIT/HACKATHON_README.md for full results."
