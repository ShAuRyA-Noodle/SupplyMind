"""train_grpo_env.py — GRPO / RLVR fine-tune against the SupplyMind OpenEnv.

Why this file exists
--------------------
The Meta PyTorch x Scaler OpenEnv Hackathon 2026 self-serve guide explicitly
recommends GRPO-style RL with verifiable rewards:

    "Prefer GRPO / RLVR style training for verifiable tasks ... if the task is
     verifiable, build the verifier first, then plug that verifier into RL
     training."

SupplyMind's reward is a verifier:
    rubric_match(agent_output.risk_level, ground_truth_risk) -> {0.0, 0.5, 1.0}

This script wires the rubric as a TRL `GRPOTrainer` reward function. DPO (see
train_dpo_trl.py and notebooks/06_trl_training_colab.ipynb) is our warm-start;
GRPO is the RLVR phase that directly optimizes the verifiable reward.

Why it ships as a separate file, not a second Colab cell
--------------------------------------------------------
GRPO generates K completions per prompt per step (default K=8) and keeps a
reference model in memory. For a Qwen-2.5-0.5B policy this needs ~8 GB VRAM,
workable on Colab T4 but slow. For the 1.5B / 3B policies we actually want,
you want an HF-compute A10G or A100 — which is what the onsite HF credits are
for on 2026-04-25/26. Colab runs the DPO warm-start; HF compute runs this.

Reward design (multi-component, anti-hack)
------------------------------------------
The hackathon guide warns against single-signal rewards and reward hacking.
We use three independent signals:

    1. r_match      {0.0, 0.5, 1.0}     exact / adjacent / wrong risk level
    2. r_format     {0.0, 1.0}          parses as valid JSON with required keys
    3. r_length     {0.0, 1.0}          within [30, 400] tokens (prevents degenerate short-circuits)

Total reward: r = 0.7 * r_match + 0.2 * r_format + 0.1 * r_length

Usage
-----
    python -m ShAuRyA_Phoenix.roll_integration.dpo_judge.train_grpo_env --dry-run
    python -m ShAuRyA_Phoenix.roll_integration.dpo_judge.train_grpo_env \
        --model Qwen/Qwen2.5-1.5B-Instruct --steps 200 --gen 8

Requires: trl>=0.12, transformers>=4.46, peft>=0.12,<0.15, accelerate, datasets.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "data" / "preference_pairs.jsonl"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "grpo_env_v1"

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _load_prompts():
    """Pull (prompt, ground_truth_risk) from the same preference pairs DPO used."""
    rows = []
    for line in PAIRS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        gt = p.get("meta", {}).get("gt_risk")
        if gt:
            rows.append({"prompt": p["prompt"], "ground_truth": gt})
    return rows


def _extract_risk(text: str) -> str | None:
    """Parse a risk level out of the LLM's response (JSON first, regex fallback)."""
    # Try JSON parse
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            r = str(obj.get("risk_level", "")).upper().strip()
            if r in RISK_ORDER:
                return r
        except (json.JSONDecodeError, AttributeError):
            pass
    # Regex fallback for free-form
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{level}\b", text.upper()):
            return level
    return None


def _r_match(pred: str | None, gt: str) -> float:
    """1.0 exact / 0.5 adjacent on the 4-point ordinal / 0.0 wrong-or-missing."""
    if pred is None:
        return 0.0
    if pred == gt:
        return 1.0
    return 0.5 if abs(RISK_ORDER[pred] - RISK_ORDER[gt]) == 1 else 0.0


def _r_format(text: str) -> float:
    """JSON parses and has the required keys for escalation routing."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return 0.0
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if {"risk_level", "confidence"}.issubset(obj.keys()) else 0.0


def _r_length(text: str, lo: int = 30, hi: int = 400) -> float:
    """Anti-hack: degenerate responses like 'CRITICAL' alone would exploit r_match."""
    n = len(text.split())
    return 1.0 if lo <= n <= hi else 0.0


def reward_fn(completions, ground_truth, **_):
    """TRL GRPO reward signature: (list[str], list[str]) -> list[float]."""
    out = []
    for comp, gt in zip(completions, ground_truth):
        rm = _r_match(_extract_risk(comp), gt)
        rf = _r_format(comp)
        rl = _r_length(comp)
        out.append(0.7 * rm + 0.2 * rf + 0.1 * rl)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gen", type=int, default=4, help="completions per prompt (K)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Sanity-check the reward on the dataset itself.
    data = _load_prompts()
    logger.info("[grpo] loaded %d prompts", len(data))
    sample_pred_chosen = data[0]["ground_truth"]  # noqa: F841
    logger.info("[grpo] reward dry-check: exact=%.2f adjacent=%.2f wrong=%.2f",
                _r_match("CRITICAL", "CRITICAL"),
                _r_match("HIGH", "CRITICAL"),
                _r_match("LOW", "CRITICAL"))

    if args.dry_run:
        logger.info("[grpo] dry-run OK — dataset=%d, reward fn validated.", len(data))
        print(json.dumps({"status": "dry_run_ok",
                          "n_prompts": len(data),
                          "reward_components": ["match", "format", "length"],
                          "reward_weights": [0.7, 0.2, 0.1]}, indent=2))
        return

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = Dataset.from_list(data)

    cfg = GRPOConfig(
        output_dir=str(args.out),
        num_generations=args.gen,
        max_prompt_length=1024,
        max_completion_length=300,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        learning_rate=args.lr,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=50,
        report_to=[],
        remove_unused_columns=False,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(str(args.out / "adapter"))

    history = trainer.state.log_history
    rewards = [e.get("reward") for e in history if e.get("reward") is not None]
    metrics = {
        "base_model": args.model,
        "steps": args.steps,
        "n_prompts": len(data),
        "generations_per_prompt": args.gen,
        "reward_components": ["match", "format", "length"],
        "reward_weights": [0.7, 0.2, 0.1],
        "mean_reward_first_10": sum(rewards[:10]) / max(1, len(rewards[:10])),
        "mean_reward_last_10": sum(rewards[-10:]) / max(1, len(rewards[-10:])),
        "n_log_steps": len(rewards),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("[grpo] saved adapter to %s", args.out / "adapter")
    logger.info("[grpo] reward lift: first10=%.3f last10=%.3f",
                metrics["mean_reward_first_10"], metrics["mean_reward_last_10"])


if __name__ == "__main__":
    main()
