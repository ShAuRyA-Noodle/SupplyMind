"""
LoRA Fine-tune for SupplyMind Explainability.

Fine-tunes a small LLM (Qwen2.5-1.5B or similar) using LoRA (PEFT)
to generate supply chain risk explanations from state descriptions.

Uses PEFT + TRL (Windows-compatible, no unsloth needed). The default path is
QLoRA: bitsandbytes 4-bit NF4 + LoRA adapters, so the 1.5B explanation model
fits on a single consumer GPU while saving only the adapter.

Requirements:
    Python 3.11 venv with: torch, peft, trl, transformers, accelerate,
    bitsandbytes, datasets

Usage (from .venv311):
    cd C:\\Users\\Dell\\Desktop\\Sleep-Token
    ...\\.venv311\\Scripts\\python.exe -m rl.lora.finetune
    ...\\.venv311\\Scripts\\python.exe -m rl.lora.finetune --model Qwen/Qwen2.5-1.5B --epochs 3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "lora"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_training_data() -> list[dict[str, str]]:
    """Build instruction-tuning dataset from supply chain scenarios.

    Each sample: {"instruction": state_description, "output": explanation}
    Uses real scenarios from our environment, not synthetic.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action
    from rl.explainer import decode_state_to_text, ACTION_NAMES

    env = SupplyMindEnvironment()
    dataset = []

    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    for task_id in tasks:
        for seed in [42, 99, 7, 123, 256]:
            obs = env.reset(task_id=task_id, seed=seed)
            step = 0
            while not obs.done and step < 15:  # First 15 steps per episode
                action = choose_action(obs, step)
                state_text = decode_state_to_text(obs)
                action_name = ACTION_NAMES.get(action.action_type, action.action_type)
                target = action.target_node_id or "N/A"

                instruction = (
                    f"You are a supply chain risk analyst. Given the current state, "
                    f"explain why the agent chose this action.\n\n"
                    f"STATE:\n{state_text}\n\n"
                    f"ACTION: {action_name} targeting {target}"
                )

                # Build a rich explanation based on the actual state
                fin = obs.financials
                signals = obs.active_signals
                explanation = _build_explanation(action, obs, fin, signals)

                dataset.append({
                    "instruction": instruction,
                    "output": explanation,
                })

                obs = env.step(action)
                step += 1

    logger.info("Built %d training samples from %d tasks x 5 seeds", len(dataset), len(tasks))
    return dataset


def _build_explanation(action, obs, fin, signals) -> str:
    """Build detailed explanation for a state-action pair."""
    health = fin.supply_chain_health_score
    budget_pct = fin.budget_remaining / max(fin.budget_total, 1) * 100
    p95 = fin.monte_carlo_p95_loss

    if action.action_type == "activate_backup_supplier":
        return (
            f"The agent activates a backup supplier for {action.target_node_id} because "
            f"the node is at risk with {len(signals)} active disruptions. "
            f"With supply chain health at {health:.0f}/100 and P95 projected loss of "
            f"${p95:,.0f}, the $150K qualification cost is justified to protect downstream "
            f"revenue. Budget utilization is at {100-budget_pct:.0f}%, leaving room for this "
            f"proactive measure. Acting during the warning phase earns a timeliness bonus."
        )
    elif action.action_type == "increase_safety_stock":
        return (
            f"Safety stock is increased at {action.target_node_id} to buffer against "
            f"potential supply disruption. Current health is {health:.0f}/100 with "
            f"{len(signals)} active signals. The inventory carrying cost (~25% annually) "
            f"is far less than the SLA penalties and revenue loss from a stockout. "
            f"Budget at {budget_pct:.0f}% remaining allows this defensive investment."
        )
    elif action.action_type == "issue_supplier_alert":
        return (
            f"A free early warning alert is issued to {action.target_node_id}. "
            f"This costs nothing but signals urgency to the supplier, improving their "
            f"response preparation. With {len(signals)} active disruptions and health "
            f"at {health:.0f}/100, proactive communication preserves the relationship "
            f"and earns the proactive bonus in scoring."
        )
    elif action.action_type == "reroute_shipment":
        return (
            f"Shipments are rerouted away from {action.target_node_id} via alternate ports. "
            f"The additional transit cost and time (+3-10 days) is offset by avoiding "
            f"complete supply chain halt. P95 loss projection of ${p95:,.0f} makes the "
            f"rerouting premium worthwhile. Health is {health:.0f}/100."
        )
    elif action.action_type == "hedge_commodity":
        return (
            f"The agent hedges commodity exposure to lock in current prices. "
            f"With commodity price changes showing volatility, the 6% hedge premium "
            f"protects against further price spikes that would compound costs. "
            f"Budget at {budget_pct:.0f}% remaining supports this risk transfer."
        )
    elif action.action_type == "expedite_order":
        return (
            f"Emergency air freight expedite to {action.target_node_id} due to critical "
            f"inventory levels. The 10x shipping premium is warranted because stockout "
            f"risk triggers SLA penalties and the 25% stockout grading penalty. "
            f"Health at {health:.0f}/100 and ${p95:,.0f} P95 loss justify the cost."
        )
    else:
        return (
            f"The agent chooses to wait and observe. With health at {health:.0f}/100 "
            f"and {budget_pct:.0f}% budget remaining, no immediate action is needed. "
            f"Conserving budget for future disruptions is the optimal strategy when "
            f"no high-severity threats require immediate response."
        )


def finetune(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    max_seq_length: int = 512,
    device: str = "cuda",
    quantization: str = "nf4",
) -> Path:
    """Fine-tune LLM with LoRA on supply chain explanation data."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    use_4bit = quantization.lower() in {"nf4", "4bit", "qlora"}

    logger.info("=" * 60)
    logger.info("LoRA Fine-Tuning")
    logger.info("  Model: %s", model_name)
    logger.info("  LoRA r=%d, alpha=%d", lora_r, lora_alpha)
    logger.info("  Quantization: %s", "bitsandbytes 4-bit NF4" if use_4bit else "none")
    logger.info("  Epochs: %d | LR: %.0e | Batch: %d", epochs, lr, batch_size)
    logger.info("  Device: %s | GPU: %s", device,
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("=" * 60)

    # Build dataset
    raw_data = build_training_data()
    logger.info("Training samples: %d", len(raw_data))

    # Format for SFT
    formatted = []
    for sample in raw_data:
        text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
        formatted.append({"text": text})

    # Save dataset
    dataset_path = DATA_DIR / "lora_training_data.json"
    dataset_path.write_text(json.dumps(formatted, indent=2))
    logger.info("Dataset saved to %s", dataset_path)

    # Load model + tokenizer
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "NF4 QLoRA requested, but bitsandbytes/transformers quantization "
                f"is unavailable: {e}"
            ) from e
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not use_4bit and device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}",
                trainable / total * 100)

    # HF Dataset
    from datasets import Dataset
    train_dataset = Dataset.from_list(formatted)

    # Training
    output_dir = str(CHECKPOINT_DIR / "lora_output")
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Save LoRA adapter
    adapter_path = CHECKPOINT_DIR / "supplymind_lora"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    manifest = {
        "base_model": model_name,
        "training_examples": len(raw_data),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "output_adapter": str(adapter_path.relative_to(PROJECT_ROOT)),
        "adapter_only": True,
        "quantization": "bitsandbytes_nf4_4bit" if use_4bit else "none",
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "trainer": "trl.SFTTrainer",
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "max_seq_length": max_seq_length,
        "elapsed_seconds": round(elapsed, 3),
    }
    (adapter_path / "supplymind_lora_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("LoRA fine-tuning done in %.1f min", elapsed / 60)
    logger.info("  Adapter saved: %s", adapter_path)
    logger.info("=" * 60)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LoRA fine-tune for SupplyMind explanations")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quantization",
        default="nf4",
        choices=["nf4", "4bit", "qlora", "none"],
        help="Default nf4 enables bitsandbytes 4-bit QLoRA.",
    )
    args = parser.parse_args()
    finetune(model_name=args.model, epochs=args.epochs, lr=args.lr,
             lora_r=args.lora_r, lora_alpha=args.lora_alpha,
             batch_size=args.batch_size, device=args.device,
             quantization=args.quantization)


if __name__ == "__main__":
    main()
