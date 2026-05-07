"""train_dpo_trl.py — DPO fine-tune Qwen-2.5-3B via standalone HuggingFace trl.

This is the ROLL-free fallback. Produces the same scientific result as the
ROLL pipeline (a fine-tuned Qwen-2.5-3B judge) without needing Megatron,
DeepSpeed, Ray, vLLM, or sglang. Runs on RTX 4080 Laptop 12GB with LoRA r=8.

Hardware profile:
    - Base: Qwen/Qwen2.5-3B-Instruct (~6GB bf16, ~1.5GB q4 inference)
    - Adapter: LoRA r=8 on all q/k/v/o proj + gate/up/down proj (~20MB)
    - Training VRAM: ~10GB with gradient_checkpointing + bf16 + batch_size=1
    - Wall-clock: ~3 hours for 2 epochs over 26-64 pairs on RTX 4080

Usage:
    pip install transformers>=4.40 peft trl==0.9.6 accelerate bitsandbytes datasets
    python -m versions.v5_phoenix.roll_integration.dpo_judge.train_dpo_trl

Outputs:
    versions/v5_phoenix/experiments/dpo_judge_v1/adapter/   (LoRA weights)
    versions/v5_phoenix/experiments/dpo_judge_v1/metrics.json

See train_dpo_roll.py for the ROLL-based alternative that enables multi-GPU +
async rollout when available.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "preference_pairs.jsonl"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "dpo_judge_v1"


def load_pairs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"preference pairs missing: {path}. "
                                "Run prepare_preference_data first.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--pairs", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL constraint strength)")
    parser.add_argument("--dry_run", action="store_true", help="Smoke test — load everything, train 1 step, save nothing.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred so module is importable without trl/transformers
    import torch  # noqa: F401
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    pairs = load_pairs(args.pairs)
    logger.info("[dpo] loaded %d preference pairs", len(pairs))

    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                            for p in pairs])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # No device_map="auto" — on a single 12GB GPU, that path offloads layers
    # to meta/cpu and the trainer later can't move them back (meta-tensor copy
    # NotImplementedError). Load directly onto cuda:0 in bf16.
    import torch
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    # With PEFT + trl 0.9.6, ref_model MUST be None — trl computes the
    # reference by temporarily disabling the LoRA adapter on the policy.
    ref_model = None

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs if not args.dry_run else 0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=20,
        beta=args.beta,
        max_length=2048,
        max_prompt_length=1024,
        report_to=[],
        # Skip eval-time generation — trl 0.9.6 + transformers >= 4.45 trip on
        # get_batch_samples calling model.generate where model is a generator-iter.
        generate_during_eval=False,
        eval_strategy="no",
        do_eval=False,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=lora,
    )

    if args.dry_run:
        logger.info("[dpo] dry-run OK — model + data loaded, trainer constructed.")
        return

    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    metrics = {
        "pairs": len(pairs),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "beta": args.beta,
        "lr": args.lr,
        "base_model": args.model,
        "final_train_loss": float(trainer.state.log_history[-1].get("loss", 0.0)) if trainer.state.log_history else None,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("[dpo] saved adapter to %s", args.out / "adapter")


if __name__ == "__main__":
    main()
