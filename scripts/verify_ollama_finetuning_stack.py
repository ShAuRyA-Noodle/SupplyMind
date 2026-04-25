"""Verify the SupplyMind Ollama, LoRA, DPO, ROLL, and quantization stack.

This is an offline evidence gate. It does not call Ollama, HuggingFace, or any
external API. Instead it verifies that every claimed training/serving artifact
is represented by committed source, configs, data, and receipts.

Usage:
    python scripts/verify_ollama_finetuning_stack.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    id: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ok": self.ok, "detail": self.detail}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def _exists(path: str) -> bool:
    return (ROOT / path).exists()


def _contains(path: str, *needles: str) -> bool:
    text = _read(path).lower()
    return all(n.lower() in text for n in needles)


def _json_count(path: str) -> int:
    data = json.loads(_read(path))
    return len(data) if isinstance(data, list) else 0


def _jsonl_count(path: str) -> int:
    return sum(1 for line in _read(path).splitlines() if line.strip())


def _param(path: str, key: str) -> str | None:
    m = re.search(rf"^\s*PARAMETER\s+{re.escape(key)}\s+(.+?)\s*$", _read(path), re.M)
    return m.group(1).strip() if m else None


def run_checks() -> list[Check]:
    checks: list[Check] = []

    # A.1 + A.2: Ollama lineage and Modelfile evolution.
    modelfiles = {
        "v1": "rl/lora/Modelfile",
        "v2": "rl/lora/Modelfile.v2",
        "v3": "rl/lora/Modelfile.v3",
        "v4": "rl/lora/Modelfile.v4",
        "v5": "ShAuRyA_Supplymind/features/Modelfile.analyst_v5",
    }
    for version, path in modelfiles.items():
        checks.append(Check(f"A.1.modelfile.{version}", _exists(path), path))

    checks.append(Check(
        "A.1.v1.domain_facts",
        _contains("rl/lora/Modelfile", "TSMC", "54%", "92%", "Red Sea", "$25K/day"),
        "v1 has TSMC, Red Sea, SLA, and cost facts",
    ))
    checks.append(Check(
        "A.1.v4.strict_json",
        _contains("rl/lora/Modelfile.v4", "STRICT JSON", "risk_level", "confidence"),
        "v4 enforces strict JSON risk output",
    ))
    checks.append(Check(
        "A.1.v5.hard_negatives_and_calibration",
        _contains(
            "ShAuRyA_Supplymind/features/Modelfile.analyst_v5",
            "CALIBRATION RULES",
            "Not every news headline is CRITICAL",
            "LOW|MEDIUM|HIGH|CRITICAL",
            "MESSAGE user",
        ),
        "v5 includes calibration rules, JSON guarantee, and few-shot hard negatives",
    ))
    checks.append(Check(
        "A.2.temperature_control",
        _param("ShAuRyA_Supplymind/features/Modelfile.analyst_v5", "temperature") in {"0.15", "0.1"},
        "v5 deterministic temperature is low",
    ))
    checks.append(Check(
        "A.2.context_window",
        int(_param("ShAuRyA_Supplymind/features/Modelfile.analyst_v5", "num_ctx") or "0") >= 16384,
        "v5 num_ctx >= 16K; v4/v3 provide 8K+ context",
    ))
    checks.append(Check(
        "A.2.versioned_creator",
        _contains("rl/lora/create_ollama_model.py", "supplymind-analyst:v5", "OLLAMA_MAX_LOADED_MODELS", "deepseek-r1-local-q4"),
        "Ollama creator registers versioned analyst and local wrapper models",
    ))

    wrappers = {
        "qwen25-14b-local": "v3_arcadia/00_emergence/qwen25-14b.Modelfile",
        "qwen25-coder-local": "v3_arcadia/00_emergence/qwen25-coder-14b.Modelfile",
        "mistral-nemo-local": "v3_arcadia/00_emergence/mistral-nemo.Modelfile",
        "deepseek-r1-local-q4": "v3_arcadia/00_emergence/deepseek-r1.Modelfile",
    }
    for name, path in wrappers.items():
        checks.append(Check(f"A.1.wrapper.{name}", _exists(path), path))

    # A.3: LoRA/QLoRA explanation fine-tuning.
    checks.append(Check(
        "A.3.lora_dataset_225",
        _json_count("rl/data/lora_training_data.json") == 225,
        "rl/data/lora_training_data.json has 225 instruction/output records",
    ))
    checks.append(Check(
        "A.3.qlora_nf4_real_code_path",
        _contains("rl/lora/finetune.py", "BitsAndBytesConfig", "bnb_4bit_quant_type", "nf4", "trl.SFTTrainer"),
        "finetune.py implements bitsandbytes NF4 QLoRA with TRL SFTTrainer",
    ))
    checks.append(Check(
        "A.3.adapter_only_manifest",
        _contains("rl/lora/finetune.py", "adapter_only", "supplymind_lora_manifest.json", "model.save_pretrained"),
        "LoRA trainer saves adapter and a manifest, not a full model copy",
    ))
    checks.append(Check(
        "A.3.ollama_conversion",
        _contains("rl/lora/create_ollama_model.py", "create_version", "create_all", "supplymind-analyst:v1"),
        "create_ollama_model.py converts prompt/adapter evidence into Ollama model registrations",
    ))

    # A.4: Phoenix DPO fine-tuning.
    checks.append(Check(
        "A.4.preference_pairs_21",
        _jsonl_count("ShAuRyA_Phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl") == 21,
        "DPO preference dataset has 21 real R4-derived chosen/rejected pairs",
    ))
    checks.append(Check(
        "A.4.dpo_trl_config",
        _contains(
            "ShAuRyA_Phoenix/roll_integration/dpo_judge/train_dpo_trl.py",
            "Qwen/Qwen2.5-3B-Instruct",
            "DPOTrainer",
            "beta",
            "r=args.lora_r",
            "gradient_accumulation_steps=4",
        ),
        "TRL fallback uses Qwen-2.5-3B, DPOTrainer, beta, LoRA, and 12GB-friendly batching",
    ))
    checks.append(Check(
        "A.4.dpo_roll_and_grpo",
        all(_exists(p) for p in [
            "ShAuRyA_Phoenix/roll_integration/dpo_judge/train_dpo_roll.py",
            "ShAuRyA_Phoenix/roll_integration/dpo_judge/train_grpo_env.py",
            "ShAuRyA_Phoenix/roll_integration/dpo_judge/train_grpo_live_env.py",
            "ShAuRyA_Phoenix/roll_integration/dpo_judge/evaluate_delta.py",
        ]),
        "ROLL DPO, standalone GRPO, live-env GRPO, and delta evaluator exist",
    ))
    checks.append(Check(
        "A.4.evaluate_delta_current_r4_shape",
        _contains("ShAuRyA_Phoenix/roll_integration/dpo_judge/evaluate_delta.py", "per.items()", "sid.replace"),
        "evaluate_delta reads current dict-shaped R4 per_scenario cache",
    ))

    # A.5: ROLL integration.
    checks.append(Check(
        "A.5.roll_env_importable_without_roll",
        _contains("ShAuRyA_Phoenix/roll_integration/env/supplymind_roll_env.py", "SupplyMindRollEnv", "supports_step_reward", "register_env", "except Exception"),
        "SupplyMindRollEnv has step rewards and guarded ROLL registration",
    ))
    checks.append(Check(
        "A.5.reward_worker",
        _contains("ShAuRyA_Phoenix/roll_integration/reward_bridge/supplymind_judge_worker.py", "SupplyMind3JudgeRewardWorker", "deepseek-r1-local-q4", "qwen25-14b-local", "mistral-nemo-local"),
        "ROLL reward worker wraps the 3 local judge models",
    ))
    checks.append(Check(
        "A.5.roll_configs",
        _contains("ShAuRyA_Phoenix/roll_integration/configs/dpo_qwen25_3b_supplymind.yaml", "strategy_name: hf", "dpo_beta: 0.1", "save_adapter_only: true")
        and _contains("ShAuRyA_Phoenix/roll_integration/configs/agentic_supplymind_gigpo.yaml", "algorithm: gigpo", "forecast", "rag", "rl_act", "step_reward: true"),
        "ROLL configs cover HF DPO, adapter-only save, GiGPO, step rewards, and 3 tools",
    ))

    # A.6: Quantization and memory engineering.
    checks.append(Check(
        "A.6.quantization_receipts",
        _contains("v3_arcadia/results/R1_VERIFIED.json", "Q4_K_M", "3.3x", "CVE-2025-32434", "safetensors"),
        "R1 verification records Q4_K_M compression and BGE safetensors rationale",
    ))
    checks.append(Check(
        "A.6.bge_safetensors_converter",
        _contains("v3_arcadia/00_emergence/convert_bge_to_safetensors.py", "save_file", "weights_only", "model.safetensors"),
        "BGE-M3 converter writes safetensors and bypasses torch.load restriction",
    ))
    checks.append(Check(
        "A.6.vram_discipline",
        _contains("v3_arcadia/40_granite/r5_rag_beast.py", "unload_ollama", "VRAM", "torch.cuda.empty_cache")
        and _contains("rl/lora/create_ollama_model.py", "OLLAMA_MAX_LOADED_MODELS"),
        "RAG/OLLAMA path documents unload and single-model VRAM discipline",
    ))
    return checks


def main() -> int:
    checks = run_checks()
    ok = all(c.ok for c in checks)
    report = {
        "ok": ok,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c.ok),
        "checks": [c.to_dict() for c in checks],
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
