"""
hypothesis_engine.py — Qwen-14B / Claude agent proposes code mutations.

Reads: program.md + current candidate_train.py + last N experiment results.
Writes: a proposed new version of candidate_train.py (full replacement) plus
        a metadata JSON {experiment_name, hypothesis, expected_metric_delta,
        justification, references}.

Two backends:
    - "ollama"   : local Qwen-14B via Ollama HTTP (no API key required)
    - "claude"   : Anthropic API (set ANTHROPIC_API_KEY or pass via env)

Guardrails (enforced post-generation):
    - Must preserve SAFE-TO-MODIFY markers.
    - Must preserve run_experiment signature.
    - Must preserve EVAL_SEEDS and EVAL_TASKS.
    - Diff size <= 150 LOC changed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

AUTORESEARCH_DIR = Path(__file__).resolve().parent
PROGRAM_MD = AUTORESEARCH_DIR / "program.md"
CANDIDATE_PATH = AUTORESEARCH_DIR / "candidate_train.py"

# Frozen markers that must survive every mutation
MARKER_BEGIN = "# --- SAFE TO MODIFY BELOW ---"
MARKER_END = "# --- SAFE TO MODIFY ABOVE ---"
FROZEN_SIGNATURE = "def run_experiment(seed: int, total_steps: int) -> dict:"
FROZEN_EVAL_SEEDS = "EVAL_SEEDS = (42, 99, 7)"
FROZEN_EVAL_TASKS = "EVAL_TASKS"

MAX_DIFF_LOC = 150


@dataclass
class Hypothesis:
    experiment_name: str
    hypothesis: str
    expected_metric_delta: str
    justification: str
    references: list[str]
    proposed_code: str  # Full new content of candidate_train.py

    def to_json(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "hypothesis": self.hypothesis,
            "expected_metric_delta": self.expected_metric_delta,
            "justification": self.justification,
            "references": self.references,
        }


SYSTEM_PROMPT = """You are an autonomous RL research agent. Your job is to
modify ONE Python file (`candidate_train.py`) to maximize a single metric
(bootstrap CI95 lower bound of grader scores across 3 tasks x 3 seeds).

You must:
1. Read `program.md` for the task spec, constraints, and fair-game changes.
2. Read the current `candidate_train.py`.
3. Read the last N experiment results (best + worst + most recent).
4. Propose exactly ONE concrete code mutation.
5. Return a JSON object with keys:
   - experiment_name (snake_case, <= 40 chars)
   - hypothesis (1-2 sentence claim)
   - expected_metric_delta (e.g., "+0.02 to +0.06 on CI95 lower")
   - justification (cite published papers or prior experiment results)
   - references (list of URLs or result-JSON paths)
   - proposed_code (FULL new content of candidate_train.py)

Rules:
- Preserve the SAFE-TO-MODIFY markers exactly as they appear.
- Preserve run_experiment signature exactly.
- Preserve EVAL_SEEDS and EVAL_TASKS constants.
- Total diff <= 150 lines of code changed.
- No external API calls during training.
- No hard-coding task-specific rules.

Respond with a SINGLE JSON object. No preamble, no explanation outside JSON.
The proposed_code field must contain the COMPLETE file content (not a diff)."""


def _format_history(history: list[dict]) -> str:
    """Take the experiments history log and format for the prompt."""
    if not history:
        return "(no prior experiments)"

    # Take best, worst, most recent 3
    sorted_by_metric = sorted(history, key=lambda h: h.get("metric_ci95_lower", 0), reverse=True)
    best = sorted_by_metric[0] if sorted_by_metric else None
    worst = sorted_by_metric[-1] if len(sorted_by_metric) > 1 else None
    recent = history[-3:]

    lines = []
    if best:
        lines.append(f"[BEST  ] {best['experiment_name']}: metric={best['metric_ci95_lower']:.4f} "
                     f"mean={best.get('metric_mean', 0):.3f} arch={best.get('architecture_summary','?')}")
    if worst and worst is not best:
        lines.append(f"[WORST ] {worst['experiment_name']}: metric={worst['metric_ci95_lower']:.4f} "
                     f"mean={worst.get('metric_mean', 0):.3f} arch={worst.get('architecture_summary','?')}")
    for r in recent:
        if r is best or r is worst:
            continue
        lines.append(f"[RECENT] {r['experiment_name']}: metric={r['metric_ci95_lower']:.4f} "
                     f"status={r.get('status','?')}")
    return "\n".join(lines) if lines else "(no prior experiments)"


def _build_prompt(history: list[dict]) -> str:
    program_md = PROGRAM_MD.read_text(encoding="utf-8")
    candidate_code = CANDIDATE_PATH.read_text(encoding="utf-8")
    history_block = _format_history(history)

    return f"""=== program.md ===
{program_md}

=== current candidate_train.py ===
```python
{candidate_code}
```

=== experiment history ===
{history_block}

=== task ===
Propose ONE code mutation to candidate_train.py that you believe will improve
the metric (bootstrap CI95 lower bound). Respond with the JSON object described
in the system prompt. Remember: full file content in proposed_code, not a diff.
"""


def _call_ollama(prompt: str, model: str = "qwen2.5:14b-instruct-q4_K_M") -> str:
    """Local Qwen-14B via Ollama. Requires ollama serve running."""
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 32768},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_claude(prompt: str, model: str = "claude-opus-4-7") -> str:
    """Anthropic Claude API. Requires ANTHROPIC_API_KEY env."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 8000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response.

    Handles both raw JSON and ```json fenced blocks.
    """
    # Try raw parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Fenced block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Fallback: greedy first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in LLM response")


def _validate_proposed_code(proposed: str, baseline: str) -> Optional[str]:
    """Return None if valid, else reason string for rejection."""
    if MARKER_BEGIN not in proposed:
        return f"missing marker `{MARKER_BEGIN}`"
    if MARKER_END not in proposed:
        return f"missing marker `{MARKER_END}`"
    if FROZEN_SIGNATURE not in proposed:
        return f"frozen signature `{FROZEN_SIGNATURE}` removed"
    if FROZEN_EVAL_SEEDS not in proposed:
        return f"frozen constant `{FROZEN_EVAL_SEEDS}` removed"
    if FROZEN_EVAL_TASKS not in proposed:
        return f"frozen constant `{FROZEN_EVAL_TASKS}` removed"

    # Diff size check
    diff_lines = list(
        unified_diff(
            baseline.splitlines(),
            proposed.splitlines(),
            lineterm="",
        )
    )
    changed = sum(1 for ln in diff_lines if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))
    if changed > MAX_DIFF_LOC:
        return f"diff too large: {changed} LOC > {MAX_DIFF_LOC} limit"

    # Quick syntax check
    try:
        compile(proposed, "<proposed>", "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    return None


def propose_hypothesis(
    history: list[dict],
    agent: str = "ollama",
    model: Optional[str] = None,
    retries: int = 3,
) -> Hypothesis:
    """Ask the agent to propose a new hypothesis + diff.

    Args:
        history: list of prior experiment summaries (from state.json).
        agent: "ollama" or "claude".
        model: override default model name.
        retries: number of retries if validation fails.
    """
    prompt = _build_prompt(history)
    baseline = CANDIDATE_PATH.read_text(encoding="utf-8")

    last_err = None
    for attempt in range(retries):
        try:
            if agent == "ollama":
                raw = _call_ollama(prompt, model or "qwen2.5:14b-instruct-q4_K_M")
            elif agent == "claude":
                raw = _call_claude(prompt, model or "claude-opus-4-7")
            else:
                raise ValueError(f"unknown agent: {agent}")

            parsed = _extract_json(raw)
            proposed_code = parsed.get("proposed_code", "")
            validation_err = _validate_proposed_code(proposed_code, baseline)
            if validation_err:
                last_err = validation_err
                logger.warning(
                    "hypothesis validation failed attempt %d/%d: %s",
                    attempt + 1, retries, validation_err,
                )
                continue

            return Hypothesis(
                experiment_name=parsed.get("experiment_name", f"exp_{attempt}")[:40],
                hypothesis=parsed.get("hypothesis", ""),
                expected_metric_delta=parsed.get("expected_metric_delta", ""),
                justification=parsed.get("justification", ""),
                references=parsed.get("references", []),
                proposed_code=proposed_code,
            )
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            logger.warning("hypothesis generation attempt %d/%d failed: %s",
                           attempt + 1, retries, e)

    raise RuntimeError(f"failed to get valid hypothesis after {retries} tries: {last_err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--history", type=str, default="state.json")
    args = parser.parse_args()

    hist_path = AUTORESEARCH_DIR / args.history
    history = []
    if hist_path.exists():
        state = json.loads(hist_path.read_text())
        history = state.get("history", [])

    try:
        hyp = propose_hypothesis(history, agent=args.agent, model=args.model)
        print(json.dumps(hyp.to_json(), indent=2))
        print(f"\n--- proposed_code is {len(hyp.proposed_code)} chars ---", file=sys.stderr)
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        sys.exit(1)
