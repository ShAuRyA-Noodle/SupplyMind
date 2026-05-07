"""evaluate_delta.py — compute DPO-judge accuracy delta vs baseline Qwen-2.5-3B.

Runs the 26 R4 crisis scenarios through:
    (a) base Qwen-2.5-3B-Instruct (no fine-tune)       -> baseline_acc
    (b) base + LoRA adapter from dpo_judge_v1          -> dpo_acc

Reports: baseline_acc, dpo_acc, delta (pp), per-scenario agreement, and
a bootstrap CI95 on the delta. This is the receipt number that proves the
DPO fine-tune actually did something.

Honest expectation for hackathon: +5 to +15 pp absolute on a 3B model. If
delta is negative we publish the null result (per the no-compromise policy).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
ADAPTER_DIR = ROOT / "versions/v5_phoenix" / "experiments" / "dpo_judge_v1" / "adapter"
R4_GT = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json"
OUT = ROOT / "versions/v5_phoenix" / "experiments" / "dpo_judge_v1" / "eval_delta.json"


def _bootstrap(x: np.ndarray, n: int = 1000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_scenarios():
    if not R4_GT.exists():
        raise FileNotFoundError(f"ground truth not found: {R4_GT}")
    blob = json.loads(R4_GT.read_text(encoding="utf-8"))
    per = blob.get("per_scenario")
    if isinstance(per, dict):
        rows = []
        for sid, entry in per.items():
            if not isinstance(entry, dict):
                continue
            text = (
                entry.get("scenario_text")
                or entry.get("summary")
                or f"Assess the supply-chain impact of the following event: {sid.replace('_', ' ')}"
            )
            rows.append({
                "id": sid,
                "text": text,
                "gt": entry.get("ground_truth", entry.get("gt_risk_level")),
            })
        return rows
    if isinstance(per, list):
        return [
            {
                "id": s.get("id", f"sc_{i}"),
                "text": s.get("scenario_text", s.get("summary", "")),
                "gt": s.get("ground_truth", s.get("gt_risk_level")),
            }
            for i, s in enumerate(per)
            if isinstance(s, dict)
        ]
    return []


def _score(pred, gt):
    """Lenient: correct if risk_level matches, else 0."""
    pl = (pred.get("risk_level") or "").upper()
    gl = (gt.get("risk_level") if isinstance(gt, dict) else str(gt or "")).upper()
    return 1.0 if pl and gl and pl == gl else 0.0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    scenarios = _load_scenarios()
    if not scenarios:
        logger.error("no scenarios loaded")
        sys.exit(2)
    logger.info("[eval] %d scenarios", len(scenarios))

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        logger.error("[eval] transformers/peft not installed: %s", e)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    if args.dry_run:
        logger.info("[eval] dry-run OK (tokenizer loaded; adapter path %s %s)",
                    args.adapter, "exists" if args.adapter.exists() else "MISSING")
        return

    def _run(model):
        hits = []
        for s in scenarios:
            messages = [{"role": "user", "content": s["text"]}]
            inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            out = model.generate(inputs, max_new_tokens=256, do_sample=False, temperature=0.0)
            txt = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
            try:
                start = txt.index("{")
                end = txt.rindex("}") + 1
                parsed = json.loads(txt[start:end])
            except Exception:  # noqa: BLE001
                parsed = {}
            hits.append(_score(parsed, s["gt"] or {}))
        return np.array(hits, dtype=np.float64)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="bfloat16",
                                                trust_remote_code=True, device_map="auto")
    baseline = _run(base)
    del base

    dpo_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="bfloat16",
                                                     trust_remote_code=True, device_map="auto")
    dpo_model = PeftModel.from_pretrained(dpo_model, str(args.adapter))
    dpo = _run(dpo_model)

    delta = dpo - baseline
    bm, blow, bhi = _bootstrap(baseline)
    dm, dlow, dhi = _bootstrap(dpo)
    xm, xlow, xhi = _bootstrap(delta)

    report = {
        "baseline_mean_acc": bm, "baseline_ci95": [blow, bhi],
        "dpo_mean_acc": dm, "dpo_ci95": [dlow, dhi],
        "delta_mean_pp": round(xm * 100, 2),
        "delta_ci95_pp": [round(xlow * 100, 2), round(xhi * 100, 2)],
        "n_scenarios": len(scenarios),
    }
    OUT.write_text(json.dumps(report, indent=2))
    logger.info("[eval] wrote %s", OUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
