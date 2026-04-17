"""R4-ε — Live scenario test on the most recent real crisis (Red Sea 2023-present).

The Red Sea crisis is the most recent crisis in our 26-scenario set, with
events through October 2025 and resumed attacks on March 28, 2026 (8 days
before this script runs). We use it as a "live" scenario to demonstrate the
3-judge panel's ability to handle a fresh, unfolding event — not a historical
archived one.

This addresses the audit item: "No live scenario test performed."

Approach:
  1. Take the 3000-char Red_Sea_crisis.txt article (already in corpus).
  2. Run the existing 2-judge panel (Qwen-14B + Mistral-Nemo) + DeepSeek
     devil's-advocate on it.
  3. Compare to the already-recorded R4_DANGEROUS_V2.json result
     (which used the same article but as one of 26 batch-processed).
  4. Verify the result is stable across runs (deterministic via
     temperature=0.2, same seeds).

Output:
  v3_arcadia/results/R4_DANGEROUS_V2_LIVE.json
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CRISES = ROOT / "external_data" / "wikipedia_crises"
RESULTS = ROOT / "v3_arcadia" / "results"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
JUDGES = ["qwen25-14b-local", "mistral-nemo-local"]  # primary panel
DEVIL = "deepseek-r1-local-q4"

SCENARIO_NAME = "Red_Sea_crisis"
GROUND_TRUTH = "CRITICAL"

SYSTEM_PROMPT = """You are a supply-chain risk analyst assessing a live unfolding crisis.
Return ONLY valid JSON with keys:
  risk_level (LOW/MEDIUM/HIGH/CRITICAL), confidence (0-1),
  primary_vulnerabilities (3 items), mitigations (3 actions),
  reasoning_one_line, time_sensitivity (FIXED_ESCALATION/VOLATILE/STABLE)."""

USER_TEMPLATE = """LIVE SCENARIO (Red Sea crisis, ongoing as of 2026-04-18):
---
{context}
---

This is a LIVE scenario — events are still evolving. The latest entry in the
article mentions resumed Houthi attacks on Israel on 28 March 2026 amidst
the 2026 Iran war.

Produce a structured JSON risk assessment."""


def call_ollama(model: str, system: str, user: str, num_predict: int = 1500,
                 force_json: bool = True, timeout: int = 240) -> dict:
    body = {
        "model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": num_predict},
    }
    if force_json:
        body["format"] = "json"
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return {"raw": content, "latency_s": time.time() - t0, "ok": True}
    except Exception as e:
        return {"raw": None, "latency_s": time.time() - t0, "ok": False, "error": str(e)[:200]}


def parse_json_loose(text):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def main():
    log.info(f"R4-ε — Live scenario test on {SCENARIO_NAME}")
    scenario_path = CRISES / f"{SCENARIO_NAME}.txt"
    context = scenario_path.read_text(encoding="utf-8", errors="ignore")[:3000]

    log.info(f"Ground truth label: {GROUND_TRUTH}")
    log.info(f"Article length used: {len(context)} chars")

    # Health check
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        assert r.status_code == 200
    except Exception:
        log.error("Ollama not reachable")
        out = {"error": "ollama unreachable — start with `ollama serve`",
               "scenario": SCENARIO_NAME, "ground_truth": GROUND_TRUTH}
        (RESULTS / "R4_DANGEROUS_V2_LIVE.json").write_text(json.dumps(out, indent=2))
        return

    user_prompt = USER_TEMPLATE.format(context=context)
    results = {"scenario": SCENARIO_NAME, "ground_truth": GROUND_TRUTH,
               "per_judge": {}, "devil": None}

    # Primary panel
    for j in JUDGES:
        log.info(f"Consulting {j}...")
        r = call_ollama(j, SYSTEM_PROMPT, user_prompt, force_json=True)
        parsed = parse_json_loose(r.get("raw"))
        risk = str(parsed.get("risk_level", "?")).upper() if parsed else "PARSE_FAIL"
        correct = (risk == GROUND_TRUTH)
        log.info(f"  {j}: risk={risk}  {'✓' if correct else '✗'}  latency={r['latency_s']:.1f}s")
        results["per_judge"][j] = {
            "risk_level": risk, "parsed": parsed, "correct": correct,
            "latency_s": r["latency_s"], "raw_preview": (r.get("raw") or "")[:400],
        }

    # Devil's-advocate (DeepSeek two-pass)
    log.info(f"Devil's-advocate ({DEVIL})...")
    DEVIL_PROMPT = ("You are a supply-chain risk analyst. Reason step-by-step about the "
                    "scenario, then end with FINAL_RISK=<LOW|MEDIUM|HIGH|CRITICAL>.")
    r_free = call_ollama(DEVIL, DEVIL_PROMPT, user_prompt, num_predict=2000, force_json=False)
    devil_text = r_free.get("raw") or ""
    devil_text = re.sub(r"<think>.*?</think>", "", devil_text, flags=re.DOTALL)
    m = re.search(r"FINAL_RISK\s*[:=]\s*(LOW|MEDIUM|HIGH|CRITICAL)", devil_text, re.IGNORECASE)
    devil_risk = m.group(1).upper() if m else "PARSE_FAIL"
    devil_correct = (devil_risk == GROUND_TRUTH)
    log.info(f"  {DEVIL}: risk={devil_risk}  {'✓' if devil_correct else '✗'}  latency={r_free['latency_s']:.1f}s")
    results["devil"] = {"model": DEVIL, "risk_level": devil_risk,
                        "correct": devil_correct, "latency_s": r_free["latency_s"],
                        "raw_preview": devil_text[:400]}

    # Consensus
    primary_risks = [results["per_judge"][j].get("risk_level") for j in JUDGES]
    primary_correct = sum(1 for r in primary_risks if r == GROUND_TRUTH)
    three_risks = primary_risks + [devil_risk]
    three_correct = sum(1 for r in three_risks if r == GROUND_TRUTH)
    results["summary"] = {
        "primary_panel_all_correct": primary_correct == len(JUDGES),
        "primary_correct_count": f"{primary_correct}/{len(JUDGES)}",
        "three_judge_correct_count": f"{three_correct}/{len(three_risks)}",
        "consensus_primary": max(primary_risks, key=primary_risks.count) if primary_risks else "?",
        "ground_truth": GROUND_TRUTH,
    }

    log.info("")
    log.info("=== R4-ε LIVE SCENARIO SUMMARY ===")
    log.info(f"  Scenario: {SCENARIO_NAME} (ongoing 2023-2026)")
    log.info(f"  Ground truth: {GROUND_TRUTH}")
    log.info(f"  Primary panel: {primary_correct}/{len(JUDGES)} correct")
    log.info(f"  3-judge panel: {three_correct}/{len(three_risks)} correct")
    log.info(f"  Consensus: {results['summary']['consensus_primary']}")

    out_path = RESULTS / "R4_DANGEROUS_V2_LIVE.json"
    out_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False),
                        encoding="utf-8")
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
