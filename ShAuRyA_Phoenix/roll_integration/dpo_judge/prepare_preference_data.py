"""prepare_preference_data.py — build DPO preference pairs from crisis scenarios.

Input: the v4 real-crisis library at
    ShAuRyA_Supplymind/scenarios/iran_israel_hormuz_2024_2026.json
    v3_arcadia/results/R4_DANGEROUS_V2.json                    # hand-labeled GT
    v3_arcadia/results/R4_DANGEROUS_V2_judge_deepseek-r1.json  # weak judge (30.8% GT acc)
    v3_arcadia/results/R4_DANGEROUS_V2_judge_mistral-nemo.json # strong judge (69.2%)

Output: ShAuRyA_Phoenix/roll_integration/dpo_judge/data/preference_pairs.jsonl

Each line is a DPO training example:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Pair construction rule: `chosen` = the judge output that matches ground-truth risk
tier; `rejected` = the judge output that got it wrong. When both are right, take
the better-calibrated one (lower |confidence - correct|) as chosen.

This is the science: DPO teaches the student model (Qwen-2.5-3B) to prefer the
good judgment over the bad one without needing a separate reward model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LIVE_CRISES_PATH = ROOT / "ShAuRyA_Supplymind" / "scenarios" / "iran_israel_hormuz_2024_2026.json"
R4_GT_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2.json"
R4_DEEPSEEK_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2_judge_deepseek-r1.json"
R4_MISTRAL_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2_judge_mistral-nemo.json"
R4_QWEN_PATH = ROOT / "v3_arcadia" / "results" / "R4_DANGEROUS_V2_judge_qwen25-14b.json"

OUT_PATH = Path(__file__).resolve().parent / "data" / "preference_pairs.jsonl"


PROMPT_TEMPLATE = """You are a supply-chain risk analyst. Assess the following crisis scenario and output a JSON object with keys:
  risk_level (LOW | MEDIUM | HIGH | CRITICAL)
  confidence (float in [0,1])
  vulnerabilities (list of strings)
  mitigations (list of strings)
  escalation_tier (C_SUITE_IMMEDIATE | C_SUITE_REVIEW | OPS_DIRECTOR_4H | OPS_DIRECTOR_24H | FYI_DASHBOARD)

Scenario:
{scenario_text}

Respond with ONLY the JSON object. No preamble."""


@dataclass
class Pair:
    prompt: str
    chosen: str
    rejected: str
    meta: dict

    def to_jsonl(self) -> str:
        return json.dumps({"prompt": self.prompt, "chosen": self.chosen,
                          "rejected": self.rejected, "meta": self.meta})


def _risk_score(level: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(level.upper(), -1)


def _load_judge(path: Path) -> dict[str, dict]:
    """Return {scenario_id: judge_output}."""
    if not path.exists():
        logger.warning("missing judge file: %s", path)
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "scenarios" in data:
        return {s["id"]: s["judgment"] for s in data["scenarios"] if "id" in s}
    if isinstance(data, list):
        return {s["id"]: s.get("judgment", s) for s in data if "id" in s}
    return {}


def _assess_quality(judge_out: dict, gt: dict) -> int:
    """Return an integer score — higher is better. Tie-break: calibration."""
    if not judge_out or not gt:
        return -10
    j_level = judge_out.get("risk_level", judge_out.get("risk", ""))
    g_level = gt.get("risk_level", gt.get("gt_risk_level", ""))
    j_s, g_s = _risk_score(j_level), _risk_score(g_level)
    if j_s < 0 or g_s < 0:
        return -5
    level_distance = abs(j_s - g_s)
    base = 10 - 4 * level_distance  # perfect = 10, off-by-1 = 6, off-by-3 = -2
    conf = float(judge_out.get("confidence", 0.5))
    if level_distance == 0:
        # correctly calibrated: confidence matches correctness
        base += int(round(conf * 3))
    else:
        # overconfidence penalty for wrong answers
        base -= int(round(conf * 3))
    return base


def build_pairs(max_pairs: int = 64) -> list[Pair]:
    """Build DPO pairs from v3 R4_DANGEROUS_V2.json structure.

    Format: blob['per_scenario'] = {
      '<scenario_id>': {
        'ground_truth': 'CRITICAL' (string),
        'per_judge': {
          'deepseek-r1-local-q4': {'parsed': {...}, 'ok': bool},
          'qwen2.5:14b-instruct-q4_K_M': {...},
          'mistral-nemo:12b-instruct-q4_K_M': {...},
        },
      }
    }

    Scenario text is pulled from the live crisis library by fuzzy matching
    the key (which is a slugified event name like '2011_Tohoku_earthquake').
    If a full event body is absent, we render the committed scenario_id itself
    into readable text. That fallback is deterministic provenance text from
    the R4 cache key, not invented event content.
    """
    if not R4_GT_PATH.exists():
        raise FileNotFoundError(f"R4 ground-truth file missing: {R4_GT_PATH}")

    blob = json.loads(R4_GT_PATH.read_text(encoding="utf-8"))
    per_scenario = blob.get("per_scenario", {})
    if not per_scenario:
        raise RuntimeError("R4 file has no per_scenario block")

    live = json.loads(LIVE_CRISES_PATH.read_text(encoding="utf-8")) if LIVE_CRISES_PATH.exists() else {"events": []}

    pairs: list[Pair] = []
    for scenario_id, entry in per_scenario.items():
        if not isinstance(entry, dict):
            continue
        gt_level = entry.get("ground_truth") or entry.get("gt_risk_level")
        if not gt_level:
            continue
        ground = {"risk_level": str(gt_level).upper(), "confidence": 1.0}

        scenario_text = _scenario_text_for(scenario_id, live)
        if not scenario_text:
            continue

        per_judge = entry.get("per_judge", {})
        outputs: dict[str, dict] = {}
        for judge_name, jout in per_judge.items():
            if not isinstance(jout, dict) or not jout.get("ok") or not jout.get("parsed"):
                continue
            outputs[judge_name] = jout["parsed"]

        scored = [(name, out, _assess_quality(out, ground)) for name, out in outputs.items()]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[2], reverse=True)
        best_name, best_out, best_score = scored[0]
        worst_name, worst_out, worst_score = scored[-1]
        if best_score - worst_score < 2:
            continue

        pairs.append(Pair(
            prompt=PROMPT_TEMPLATE.format(scenario_text=scenario_text),
            chosen=json.dumps(best_out, sort_keys=True, ensure_ascii=False),
            rejected=json.dumps(worst_out, sort_keys=True, ensure_ascii=False),
            meta={
                "scenario_id": scenario_id,
                "chosen_judge": best_name,
                "rejected_judge": worst_name,
                "quality_gap": best_score - worst_score,
                "gt_risk": ground["risk_level"],
            },
        ))
        if len(pairs) >= max_pairs:
            break
    return pairs


def _scenario_text_for(scenario_id: str, live: dict) -> str:
    """Match a v3 scenario key against the live crisis library or synthesize text."""
    key_lower = scenario_id.lower()
    for ev in live.get("events", []):
        if ev.get("id", "").lower() in key_lower or ev.get("name", "").lower() in key_lower:
            return f"{ev['name']}. {ev.get('summary', '')}"
    # Fallback: use the slugified key itself as the prompt description.
    # This is provenance-preserving ID text, not synthetic event evidence.
    clean = scenario_id.replace("_", " ")
    return f"Assess the supply-chain impact of the following event: {clean}"


def _severity_to_level(sev: float) -> str:
    if sev >= 0.85: return "CRITICAL"
    if sev >= 0.65: return "HIGH"
    if sev >= 0.35: return "MEDIUM"
    return "LOW"


def _find_scenario_text(scenario_id: str, *judge_maps: dict, live_events: Path) -> str:
    for jm in judge_maps:
        entry = jm.get(scenario_id, {}) if jm else {}
        if isinstance(entry, dict):
            txt = entry.get("scenario_text") or entry.get("prompt") or entry.get("summary")
            if txt:
                return str(txt)
    # fallback: crisis library
    if live_events.exists():
        blob = json.loads(live_events.read_text())
        for e in blob.get("events", []):
            if e.get("id") == scenario_id:
                return f"{e['name']}. {e.get('summary', '')}"
    return ""


def write(pairs: list[Pair], out_path: Path = OUT_PATH) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.to_jsonl() + "\n")
    return len(pairs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pairs = build_pairs()
    n = write(pairs)
    print(f"[prepare] wrote {n} preference pairs to {OUT_PATH}")
    if pairs:
        print(f"[prepare] example quality gaps: {[p.meta['quality_gap'] for p in pairs[:5]]}")


if __name__ == "__main__":
    main()
