#!/usr/bin/env bash
set -u
VENV_PY="c:/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix/.venv-roll/Scripts/python.exe"
LOG="c:/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix/experiments/dpo_judge_v1/train.log"
echo "[dpo] $(date) starting" > "$LOG"
echo "[dpo] Qwen-2.5-1.5B + LoRA r=8, 2 epochs, 21 pairs" >> "$LOG"
cd "c:/Users/Dell/Desktop/Sleep-Token" || exit 1

"$VENV_PY" -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_trl \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 2 \
    --lora_r 8 \
    --beta 0.1 >> "$LOG" 2>&1
echo "[dpo] $(date) done exit=$?" >> "$LOG"
