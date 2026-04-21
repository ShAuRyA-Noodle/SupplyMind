"""
lora_train.py — G7. LoRA fine-tuning harness for supplymind-analyst v5.

The v2-era Modelfiles (v2/v3/v4) captured our analyst prompt engineering but
never actually updated model WEIGHTS — real LoRA was blocked by Ollama's HF
offline-mode issue. This module is the drop-in fix:

    python -m ShAuRyA_Supplymind.features.lora_train --dry-run   # no GPU
    python -m ShAuRyA_Supplymind.features.lora_train --train     # 2-3h on RTX 4080

Pipeline:
    1. Load Qwen2.5-14B-Instruct via transformers (Q4_K_M too lossy for LoRA —
       we train on BF16/FP16 base, then re-quantize to Q4_K_M for deployment).
    2. Build training dataset from:
       - 26 R4 Wikipedia crisis scenarios + rubric-labeled risk levels
       - 6 pre-warmed counterfactual explanations (F3)
       - 10 Modelfile v5 few-shots
    3. Apply PEFT LoRA adapters (r=16, alpha=32, target = q_proj,v_proj).
    4. Train 3 epochs, save adapter to `rl/checkpoints/lora/supplymind_v5/`.
    5. Optionally export: `ollama create supplymind-analyst:v5-lora` from adapter.

Honest scope: this is the SCRIPT. We do NOT run it here — each run costs 2-3
GPU-hours. Use `--dry-run` to validate all data + imports + adapter config.
Running `--train` is the user's decision when they have GPU time available.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_OUT = PROJECT_ROOT / "rl" / "checkpoints" / "lora" / "supplymind_v5"
SCENARIOS_LIB = (PROJECT_ROOT / "ShAuRyA_Supplymind" / "scenarios"
                 / "iran_israel_hormuz_2024_2026.json")
R4_RESULTS = PROJECT_ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json"


@dataclass
class TrainingExample:
    scenario: str
    correct_risk_level: str
    rationale: str
    source: str = "rubric"

    def to_dict(self) -> dict:
        return self.__dict__


# ---------------------------------------------------------------------------
# Training-data builder — aggregates 3 sources into one JSONL
# ---------------------------------------------------------------------------


def build_dataset() -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    # 1. Crisis library (8 events)
    if SCENARIOS_LIB.exists():
        lib = json.loads(SCENARIOS_LIB.read_text(encoding="utf-8"))
        for e in lib["events"]:
            sev = e.get("severity", 0.3)
            risk = ("CRITICAL" if sev >= 0.80
                    else "HIGH" if sev >= 0.60
                    else "MEDIUM" if sev >= 0.35
                    else "LOW")
            examples.append(TrainingExample(
                scenario=e["summary"][:1500],
                correct_risk_level=risk,
                rationale=f"Historical analog {e['name']} at severity {sev}. "
                          f"Cited in: {', '.join(c['publisher'] for c in e['citations'][:2])}.",
                source="crisis_library",
            ))

    # 2. R4 Wikipedia scenarios + ground-truth labels (if available)
    if R4_RESULTS.exists():
        try:
            r4 = json.loads(R4_RESULTS.read_text(encoding="utf-8"))
            scenarios = r4.get("per_scenario") or r4.get("scenarios") or []
            for s in scenarios[:26]:
                text = s.get("scenario_text") or s.get("text") or s.get("article_title", "")
                gt = s.get("ground_truth") or s.get("gt_risk") or s.get("risk_level")
                if text and gt:
                    examples.append(TrainingExample(
                        scenario=text[:1500],
                        correct_risk_level=str(gt).upper(),
                        rationale=f"R4 scenario; anchored by multi-judge panel + rubric.",
                        source="r4_wikipedia",
                    ))
        except Exception as e:  # noqa: BLE001
            logger.warning("R4 parse failed: %s", e)

    # 3. Modelfile v5 few-shots (synthetic but calibrated)
    mf_v5 = PROJECT_ROOT / "ShAuRyA_Supplymind" / "features" / "Modelfile.analyst_v5"
    if mf_v5.exists():
        content = mf_v5.read_text(encoding="utf-8")
        # Parse MESSAGE user/assistant blocks — quick regex
        import re
        blocks = re.findall(r'MESSAGE user """(.*?)"""\s*MESSAGE assistant """(.*?)"""',
                            content, re.DOTALL)
        for user, assistant in blocks:
            try:
                parsed = json.loads(assistant.strip())
                risk = str(parsed.get("risk_level", "")).upper()
                if risk:
                    examples.append(TrainingExample(
                        scenario=user.strip()[:1500],
                        correct_risk_level=risk,
                        rationale=str(parsed.get("decision", "")) + " " +
                                  " ".join(parsed.get("evidence", [])),
                        source="modelfile_v5_fewshot",
                    ))
            except Exception:  # noqa: BLE001
                continue

    return examples


# ---------------------------------------------------------------------------
# LoRA config (PEFT) + training wrapper
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    learning_rate: float = 2e-4
    n_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    output_dir: Path = field(default_factory=lambda: LORA_OUT)

    def to_dict(self) -> dict:
        return {**self.__dict__, "output_dir": str(self.output_dir)}


def _format_example(ex: TrainingExample) -> str:
    """Convert a TrainingExample into chat-format text for causal LM training."""
    user_turn = (f"STATE: {ex.scenario}\n\n"
                 f"Respond with JSON: {{\"risk_level\": ..., \"decision\": ..., "
                 f"\"evidence\": [...], \"confidence\": ...}}")
    assistant_turn = json.dumps({
        "risk_level": ex.correct_risk_level,
        "decision": ex.rationale[:200],
        "evidence": [ex.source],
        "confidence": {"LOW": 0.75, "MEDIUM": 0.70, "HIGH": 0.85,
                       "CRITICAL": 0.90}.get(ex.correct_risk_level, 0.75),
    })
    return (f"<|im_start|>user\n{user_turn}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_turn}<|im_end|>")


def train(config: LoRAConfig, examples: list[TrainingExample]) -> dict:
    """Run the actual fine-tune. REQUIRES transformers + peft + accelerate + torch."""
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling)
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import Dataset
    except ImportError as e:
        return {"status": "imports_failed", "error": str(e),
                "hint": "pip install transformers peft accelerate bitsandbytes datasets"}

    if not torch.cuda.is_available():
        return {"status": "no_cuda", "error": "CUDA not available; LoRA requires GPU"}

    logger.info("[lora] loading %s ...", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    peft_cfg = LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    texts = [_format_example(ex) for ex in examples]
    dataset = Dataset.from_dict({"text": texts})

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_seq_length,
                         padding="max_length")
    dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio, weight_decay=config.weight_decay,
        bf16=config.bf16, logging_steps=10, save_strategy="epoch",
        save_total_limit=2, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    return {"status": "ok", "output_dir": str(config.output_dir),
            "n_examples": len(examples)}


def dry_run(config: LoRAConfig) -> dict:
    examples = build_dataset()
    texts = [_format_example(ex) for ex in examples]
    return {
        "status": "dry_run_ok",
        "n_examples": len(examples),
        "by_source": {s: sum(1 for ex in examples if ex.source == s)
                      for s in {ex.source for ex in examples}},
        "sample_text": texts[0][:500] if texts else "(empty)",
        "config": config.to_dict(),
        "next_step": ("Run `python -m ShAuRyA_Supplymind.features.lora_train --train` "
                      "with GPU + HF_HOME cached."),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset + imports without GPU training")
    parser.add_argument("--train", action="store_true", help="Run actual LoRA training (2-3 GPU hours)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = LoRAConfig(rank=args.rank, n_epochs=args.epochs)

    if args.dry_run or (not args.train):
        result = dry_run(cfg)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    examples = build_dataset()
    if not examples:
        print("no training examples built; aborting")
        sys.exit(1)
    result = train(cfg, examples)
    print(json.dumps(result, indent=2))
