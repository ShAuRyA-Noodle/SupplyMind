"""finetune_unsloth.py — Unsloth-accelerated LoRA recipe (canonical hackathon stack).

Per Meta OpenEnv x Scaler guide section 10:
  TRL + Unsloth + OpenEnv = canonical hackathon stack
  Unsloth reduces memory + improves training speed 2-5x

This script mirrors `rl/lora/finetune.py` but routes through Unsloth's
FastLanguageModel for the warm-start SFT pass on 225 instruction/output pairs
in `rl/data/lora_training_data.json`.

Stack:
  - Base: Qwen-2.5-1.5B-Instruct (Unsloth provides 4-bit pre-quantized variants)
  - LoRA: r=16, alpha=16, dropout=0.05
  - SFT via TRL SFTTrainer
  - Save: adapter-only (~20MB) per guide section 16 (avoid 4-bit -> 16-bit
    upcast + naive merge: keep adapter or use proper merged-save path)

Falls back gracefully if Unsloth not installed (clear message + suggest pip).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = REPO_ROOT / "rl" / "data" / "lora_training_data.json"
OUT_DIR = REPO_ROOT / "rl" / "checkpoints" / "lora_unsloth"
RECEIPT = REPO_ROOT / "tests" / "receipts" / "lora_unsloth_train.json"


def _check_deps() -> dict:
    """Probe stack availability without crashing."""
    out = {}
    for mod, name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("unsloth", "unsloth"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
    ]:
        try:
            m = __import__(mod)
            out[name] = getattr(m, "__version__", "ok")
        except ImportError:
            out[name] = None
    return out


def _load_dataset(path: Path = TRAIN_DATA, max_samples: int | None = None):
    """Load 225 SupplyMind instruction/output pairs as HF Dataset."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if max_samples:
        raw = raw[:max_samples]
    try:
        from datasets import Dataset
        return Dataset.from_list(raw)
    except ImportError:
        return raw


def _format_prompt(example: dict) -> str:
    """SupplyMind instruction format -> chat template."""
    instr = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    if inp:
        user = f"{instr}\n\n{inp}"
    else:
        user = instr
    return (
        f"<|im_start|>system\nYou are a supply-chain risk analyst.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )


def train(
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length: int = 2048,
    n_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_samples: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Unsloth + TRL SFT entry."""
    t0 = time.time()
    deps = _check_deps()

    if deps.get("unsloth") is None or deps.get("trl") is None:
        return {
            "status": "deps_missing",
            "deps": deps,
            "install": "pip install unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git trl peft bitsandbytes",
            "note": "Recipe is wired and ready to run when Unsloth + TRL present",
            "elapsed_s": round(time.time() - t0, 2),
        }

    if dry_run:
        return {
            "status": "dry_run_OK_recipe_wired",
            "deps": deps,
            "config": {
                "model": model_name, "max_seq_length": max_seq_length,
                "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr,
                "lora_r": lora_r, "lora_alpha": lora_alpha,
            },
            "expected_output_mb": 20,
            "expected_train_time_min": "~5-8 on RTX 4080 (Unsloth 2-5x speedup)",
            "elapsed_s": round(time.time() - t0, 2),
        }

    # Real training path
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth",
    )

    raw = json.loads(TRAIN_DATA.read_text(encoding="utf-8"))
    if max_samples:
        raw = raw[:max_samples]
    formatted = [{"text": _format_prompt(r)} for r in raw]
    ds = Dataset.from_list(formatted)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_seq_length=max_seq_length,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=cfg, dataset_text_field="text",
    )
    train_out = trainer.train()
    # Save adapter-only (per guide section 16: avoid 4-bit -> 16-bit upcast merge)
    model.save_pretrained(str(OUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUT_DIR / "adapter"))

    return {
        "status": "trained_ok",
        "deps": deps,
        "config": {
            "model": model_name, "n_epochs": n_epochs, "batch_size": batch_size,
            "lr": lr, "lora_r": lora_r, "lora_alpha": lora_alpha,
            "n_samples": len(formatted),
        },
        "train_metrics": {
            "global_step": train_out.global_step,
            "training_loss": float(train_out.training_loss),
        },
        "adapter_path": str(OUT_DIR / "adapter"),
        "elapsed_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                          help="Probe deps + return config; no training")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    res = train(dry_run=args.dry_run, max_samples=args.max_samples,
                  n_epochs=args.epochs)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nReceipt: {RECEIPT}")
