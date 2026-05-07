# ROLL install guide — Phoenix v5

Two phases. Phase A is 30 minutes of Windows-native pip; Phase B is a full-day
WSL2 build-out (per user directive). Fall through to Phase C `trl` fallback
only if both phases fail.

**Budget ceiling**: 8 hours total across phases. After that, commit to
`trl.DPOTrainer` fallback (same scientific result, loses ROLL env PR).

---

## Phase A — Windows-native (first try, 30–60 min)

```bash
cd C:\Users\Dell\Desktop\Sleep-Token\versions/v5_phoenix
python -m venv .venv-roll
.venv-roll\Scripts\activate

# Core ROLL (skips Megatron + vLLM + sglang — the usual Windows pain points)
pip install -e ..\vendor/ROLL/[hf]

# Our DPO dependencies
pip install "trl==0.9.6" "transformers>=4.40" "peft>=0.11" "accelerate>=0.28" \
            "datasets>=2.18" "bitsandbytes>=0.43" "httpx>=0.25"

# Smoke test: can we import ROLL's DPO pipeline?
python -c "from roll.pipeline.dpo import DPOPipeline; print('roll dpo ok')"

# Smoke test: 0.5B model loads?
python -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_trl \
    --model Qwen/Qwen2.5-0.5B-Instruct --dry_run
```

**Green**: both smoke tests print OK → you're done, skip Phase B.
**Red**: note which pip install failed and move to Phase B.

Known Phase A failures and workarounds:
| Error | Workaround |
|---|---|
| `flash-attn` wheel missing for Windows | ROLL's `[hf]` extra shouldn't pull flash-attn. If it does, edit `setup.py` to gate it behind `extras_require={"linux": ["flash-attn"]}`. |
| `vllm` wheel missing | Same — ROLL's `[hf]` should skip it. We don't need vLLM for DPO. |
| `deepspeed` build errors | We don't need DeepSpeed for single-GPU LoRA DPO. Remove from any transitive req list. |
| Ray install hanging | Ray is only needed for multi-node. Skip: `pip install -e ..\vendor/ROLL/[hf] --no-deps` then install deps manually. |

---

## Phase B — WSL2 + CUDA passthrough (full day, up to 6 h)

If Phase A is unrecoverable, escalate to WSL2.

### One-time WSL setup (~30 min)

```powershell
# In PowerShell as admin
wsl --install -d Ubuntu-22.04
# Reboot if prompted.
wsl --set-default-version 2
```

After first-boot onboarding (username / password):

```bash
# Inside WSL2
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential git

# Verify CUDA passthrough — NVIDIA driver on Windows is enough; do NOT install another one inside WSL.
nvidia-smi
# Expected: see the RTX 4080 Laptop, 12GB, CUDA 12.x
```

### ROLL install (full extras) (~1–2 h compile)

```bash
cd /mnt/c/Users/Dell/Desktop/Sleep-Token/versions/v5_phoenix
python3.11 -m venv .venv-roll-wsl
source .venv-roll-wsl/bin/activate

pip install --upgrade pip
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# Core ROLL
pip install -e /mnt/c/Users/Dell/Desktop/Sleep-Token/vendor/ROLL/[hf,deepspeed]

# Linux wheels we couldn't get on Windows
pip install "vllm==0.6.3" "flash-attn" --no-build-isolation

# DPO deps
pip install "trl==0.9.6" "transformers>=4.40" "peft>=0.11" "accelerate>=0.28" "datasets>=2.18"

# Smoke tests
python -c "import vllm, flash_attn; from roll.pipeline.dpo import DPOPipeline; print('wsl roll full stack ok')"
python -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_trl \
    --model Qwen/Qwen2.5-0.5B-Instruct --dry_run
```

**Green**: green → use `.venv-roll-wsl` for all ROLL work.
**Red**: drop to Phase C.

### Known Phase B failures

| Error | Workaround |
|---|---|
| `nvidia-smi: command not found` | Install NVIDIA's Windows driver v535+ for WSL2 CUDA passthrough: https://docs.nvidia.com/cuda/wsl-user-guide/index.html |
| `flash-attn` build fails with ninja / cmake / cc1plus error | `export MAX_JOBS=2` to avoid OOM during compile. Compile takes ~30 min even on good machines. |
| vLLM ImportError about cuBLAS / cuDNN | `sudo apt install -y libcudnn8 libcudnn8-dev` |
| OOM during `flash-attn` compile | set `MAX_JOBS=1`; give WSL more RAM in `.wslconfig` (`[wsl2]\nmemory=12GB`) |

---

## Phase C — `trl` fallback (always works, ships same science)

If Phases A and B both fail, ship DPO via standalone `trl.DPOTrainer`.
Runs on Windows native with only `pip install trl transformers peft`.

```bash
cd C:\Users\Dell\Desktop\Sleep-Token\versions/v5_phoenix
python -m venv .venv-fallback
.venv-fallback\Scripts\activate

pip install "trl==0.9.6" "transformers>=4.40" "peft>=0.11" "accelerate>=0.28" \
            "datasets>=2.18" "bitsandbytes>=0.43"

python -m versions.v5_phoenix.roll_integration.dpo_judge.prepare_preference_data
python -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_trl --epochs 2
python -m versions.v5_phoenix.roll_integration.dpo_judge.evaluate_delta
```

Loses: ROLL env upstream PR (still ship draft), GiGPO agentic training (defer).
Keeps: ROLL-DPO-judge-v1 receipt, SupplyMind-as-ROLL-env code (unrun), reward bridge code.

---

## Decision flowchart

```
                  Phase A smoke pass?
                        |
                   yes  |  no
          ┌─────────────┴─────────────┐
          ▼                           ▼
    use .venv-roll              Phase B smoke pass?
                                      |
                                 yes  |  no
                          ┌───────────┴───────────┐
                          ▼                       ▼
                  use .venv-roll-wsl       use .venv-fallback (Phase C)
```

## Receipt

When any phase ends green, write a receipt at
`versions/v5_phoenix/receipts_v2/V5_ROLL_install_phase.reproduce.sh` showing
the exact commands and resulting `pip freeze` hash. Same receipt template
for all three phases; only the `phase:` field differs.
