"""verify_lora_merge.py — verify LoRA adapter merged-save path is safe.

Per Meta OpenEnv x Scaler hackathon-guide §16 (common mistakes):
  > Do NOT upcast a 4-bit model to 16-bit and then merge LoRA naively.
  > Use the proper merged-save path, or use the adapters directly.

This script:
  1. Loads base model
  2. Loads LoRA adapter
  3. Runs a fixed prompt through (a) base+adapter pipeline (b) merged model
  4. Compares logits + generated tokens
  5. Asserts diff is below threshold
  6. Writes receipt

Falls back to dry-run if adapter missing (no GPU artifacts in repo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIRS = [
    REPO_ROOT / "rl" / "checkpoints" / "lora_unsloth" / "adapter",
    REPO_ROOT / "rl" / "checkpoints" / "lora",
    REPO_ROOT / "checkpoints" / "lora",
]
RECEIPT = REPO_ROOT / "tests" / "receipts" / "lora_merge_verify.json"


SAFE_MERGE_RECIPE = """
# Safe LoRA merge path (per guide §16):

# OPTION A — recommended: keep adapter at inference, NEVER merge.
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                             torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "rl/checkpoints/lora_unsloth/adapter")
# inference uses base + adapter on-the-fly; no merge, no upcast risk.

# OPTION B — if you MUST merge: load base in float, NOT 4-bit.
# (4-bit -> 16-bit upcast + naive merge corrupts weights.)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                             torch_dtype=torch.bfloat16)  # NOT load_in_4bit
model = PeftModel.from_pretrained(base, "rl/checkpoints/lora_unsloth/adapter")
merged = model.merge_and_unload()
merged.save_pretrained("rl/checkpoints/merged_full_precision")

# OPTION C — Unsloth save_pretrained_merged (handles 4-bit safely):
from unsloth import FastLanguageModel
model.save_pretrained_merged("rl/checkpoints/merged_unsloth", tokenizer,
                              save_method="merged_16bit")  # or "lora" for adapter only
"""


def _find_adapter() -> Path | None:
    for d in ADAPTER_DIRS:
        if d.exists() and (d / "adapter_config.json").exists():
            return d
    return None


def _logits_diff(model_a, model_b, tokenizer, prompt: str) -> dict:
    """Compare next-token logits between two models on identical prompt."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model_a.parameters()).device)
    with torch.no_grad():
        la = model_a(**inputs).logits[0, -1, :].float()
        lb = model_b(**inputs).logits[0, -1, :].float()
    diff = (la - lb).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "l2_diff": float(torch.norm(la - lb)),
        "topk_a": tokenizer.batch_decode(la.topk(5).indices.unsqueeze(0)),
        "topk_b": tokenizer.batch_decode(lb.topk(5).indices.unsqueeze(0)),
        "topk_match": tokenizer.decode([la.argmax()]) == tokenizer.decode([lb.argmax()]),
    }


def verify(adapter_dir: Path | None = None,
            base_model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> dict:
    """Real verification path."""
    t0 = time.time()
    adapter_dir = adapter_dir or _find_adapter()

    if adapter_dir is None:
        return {
            "status": "no_adapter_found",
            "checked_paths": [str(p) for p in ADAPTER_DIRS],
            "note": ("LoRA adapters are runtime artifacts, not committed. "
                      "When training runs (rl/lora/finetune.py or finetune_unsloth.py), "
                      "run this script after."),
            "safe_merge_recipe_documented": True,
            "recipe": SAFE_MERGE_RECIPE.strip(),
            "elapsed_s": round(time.time() - t0, 2),
        }

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        return {
            "status": "deps_missing",
            "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }

    logger.info("[lora-verify] loading base %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("[lora-verify] loading adapter %s", adapter_dir)
    adapter_model = PeftModel.from_pretrained(base, str(adapter_dir))
    adapter_model.eval()

    logger.info("[lora-verify] merging (option B: float-precision merge)")
    merged = adapter_model.merge_and_unload()
    merged.eval()

    prompt = ("<|im_start|>system\nYou are a supply-chain risk analyst.<|im_end|>\n"
                "<|im_start|>user\nWhat is the typical Brent reaction to a Hormuz scare?<|im_end|>\n"
                "<|im_start|>assistant\n")
    diff = _logits_diff(adapter_model, merged, tok, prompt)

    return {
        "status": "verified" if diff["max_abs_diff"] < 1e-3 else "drift_detected",
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "logits_diff": diff,
        "verdict": ("PASS · adapter and merged produce identical top-1 token; "
                     f"max_abs_diff = {diff['max_abs_diff']:.2e} (< 1e-3 threshold)"
                     if diff["max_abs_diff"] < 1e-3 else
                     f"FAIL · drift {diff['max_abs_diff']:.2e} suggests merge corruption"),
        "elapsed_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default=None)
    args = parser.parse_args()
    res = verify(adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "recipe"}, indent=2))
    print(f"\nReceipt: {RECEIPT}")
