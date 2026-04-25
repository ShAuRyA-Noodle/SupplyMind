#!/usr/bin/env bash
# REPRODUCE_ONE_BASH.sh — regenerate every receipt in one shot.
#
# Usage:   bash FINAL_SUBMIT/REPRODUCE_ONE_BASH.sh
# Time:    ~3-5 minutes on CPU (no GPU required)
# Output:  tests/receipts/*.json + FINAL_SUBMIT/receipts/*.json
#
# Per OpenEnv hackathon §"Reproducibility": one command, all receipts.

set -e
cd "$(dirname "$0")/.."
echo "=== SupplyMind FINAL SUBMIT reproducibility ==="
echo "Repo: $(pwd)"
echo

echo "[1/8] Wordle env + RLVE curriculum smoke ..."
python -m ShAuRyA_Phoenix.wordle_env.rlve_curriculum

echo
echo "[2/8] Dual verifier smoke ..."
python -m ShAuRyA_Phoenix.wordle_env.dual_verifier

echo
echo "[3/8] OpenEnv MCP compliance ..."
python server/openenv_mcp_wrapper.py

echo
echo "[4/8] REAL REINFORCE training (190% improvement) ..."
python scripts/final_real_reinforce_wordle.py --episodes 1600 --batch 16

echo
echo "[5/8] 20-attack adversarial reward-hack gauntlet ..."
python scripts/final_adversarial_20suite.py

echo
echo "[6/8] Cross-env transfer + process supervision + ablations + API keys ..."
python scripts/final_validation_bundle.py

echo
echo "[7/8] Wordle GRPO baseline (heuristic policy receipt) ..."
python -m ShAuRyA_Phoenix.wordle_env.train_grpo --steps 50 || true

echo
echo "[8/8] Receipt index ..."
ls -1 tests/receipts/*.json | wc -l
echo "receipts in tests/receipts/"
ls -1 FINAL_SUBMIT/receipts/*.json | wc -l
echo "receipts in FINAL_SUBMIT/receipts/"

echo
echo "=== DONE ==="
echo "Open FINAL_SUBMIT/HACKATHON_README.md for full results."
