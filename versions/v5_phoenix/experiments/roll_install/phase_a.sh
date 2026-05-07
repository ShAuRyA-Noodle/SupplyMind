#!/usr/bin/env bash
set -u
# Phase A — Windows-native ROLL install into .venv-roll
export PIP_DISABLE_PIP_VERSION_CHECK=1
LOG="c:/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix/experiments/roll_install/phase_a.log"
VENV_PIP="c:/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix/.venv-roll/Scripts/pip.exe"
VENV_PY="c:/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix/.venv-roll/Scripts/python.exe"
ROLL_DIR="c:/Users/Dell/Desktop/Sleep-Token/ROLL-main/ROLL-main"

echo "[phase_a] $(date) starting" > "$LOG"
echo "[phase_a] upgrading pip" >> "$LOG"
"$VENV_PY" -m pip install --upgrade pip 2>&1 | tail -20 >> "$LOG"

echo "[phase_a] installing baseline deps (trl+transformers+peft+accelerate+datasets)" >> "$LOG"
"$VENV_PIP" install "trl==0.9.6" "transformers>=4.40,<4.50" "peft>=0.11,<1.0" "accelerate>=0.28,<1.0" "datasets>=2.18,<3.0" "bitsandbytes>=0.43" "httpx>=0.25" "pyyaml" 2>&1 | tail -30 >> "$LOG"
echo "[phase_a] baseline install exit=$?" >> "$LOG"

echo "[phase_a] attempting ROLL[hf] install (editable)" >> "$LOG"
"$VENV_PIP" install -e "$ROLL_DIR" --no-deps 2>&1 | tail -40 >> "$LOG"
ROLL_EXIT=$?
echo "[phase_a] ROLL install exit=$ROLL_EXIT" >> "$LOG"

echo "[phase_a] smoke test: import roll" >> "$LOG"
"$VENV_PY" -c "import roll; print('roll imported', roll.__file__)" 2>&1 | tail -5 >> "$LOG"
echo "[phase_a] smoke test exit=$?" >> "$LOG"

echo "[phase_a] smoke test: from roll.pipeline.dpo import DPOPipeline" >> "$LOG"
"$VENV_PY" -c "from roll.pipeline.dpo import DPOPipeline; print('DPOPipeline ok')" 2>&1 | tail -10 >> "$LOG"
DPO_EXIT=$?
echo "[phase_a] DPOPipeline import exit=$DPO_EXIT" >> "$LOG"

echo "[phase_a] smoke test: trl standalone" >> "$LOG"
"$VENV_PY" -c "from trl import DPOTrainer; print('trl ok')" 2>&1 | tail -5 >> "$LOG"
TRL_EXIT=$?
echo "[phase_a] trl exit=$TRL_EXIT" >> "$LOG"

if [ $TRL_EXIT -eq 0 ] && [ $DPO_EXIT -eq 0 ]; then
  echo "[phase_a] RESULT: GREEN (ROLL + trl both work)" >> "$LOG"
elif [ $TRL_EXIT -eq 0 ]; then
  echo "[phase_a] RESULT: YELLOW (trl works, ROLL import failed -- use trl fallback path)" >> "$LOG"
else
  echo "[phase_a] RESULT: RED (trl also failed -- escalate to Phase B WSL2)" >> "$LOG"
fi
echo "[phase_a] $(date) done" >> "$LOG"
