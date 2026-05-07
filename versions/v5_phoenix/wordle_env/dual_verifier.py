"""dual_verifier.py — RLVR dual-verifier framework (rule + model).

Per RL guide §31-33 (verifier brittleness):
  > Rule-based verifiers are brittle: produce false negatives when correct
  > answer is differently formatted.
  > Model-based verifiers can be exploited: produce false positives that
  > the policy learns to game.
  > Stronger policies make verifier weaknesses more obvious.
  > Use BOTH and stress-test.

Approach:
  1. Rule verifier: exact constraint check (Wordle: word ∈ dict, valid format)
  2. Model verifier: LLM judge ("is this guess strategically sound given
     past feedback?")
  3. Composite reward: r = rule_score × (0.5 + 0.5 × model_score)
  4. Disagreement alarm: if |rule - model| > 0.3 over rolling window,
     flag for human inspection (anti-hack monitoring per §43)

For SupplyMind we already have this de-facto:
  - Rule layer: server/engine/rewards.py 7-component
  - Model layer: 3-judge Ollama panel + 12-frontier OpenRouter
  - This module formalizes it.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


@dataclass
class VerifierResult:
    rule_score: float
    rule_pass: bool
    rule_reason: str
    model_score: float | None = None
    model_pass: bool | None = None
    model_reason: str | None = None
    composite_score: float = 0.0
    disagreement: float | None = None
    alarm: bool = False
    elapsed_s: float = 0.0


@dataclass
class VerifierAuditState:
    n_calls: int = 0
    n_rule_pass: int = 0
    n_model_pass: int = 0
    n_disagreement_alarms: int = 0
    rolling_disagreement: deque = field(default_factory=lambda: deque(maxlen=50))


class DualVerifier:
    """Combines rule-based + LLM-judge verifiers with disagreement monitoring."""

    DISAGREEMENT_THRESHOLD = 0.30

    def __init__(self, model_name: str = "qwen25-14b-local:latest"):
        self.model_name = model_name
        self.audit = VerifierAuditState()

    # -------- rule layer --------
    def _rule_wordle(self, guess: str, target: str, history: list[dict]) -> tuple[float, bool, str]:
        """Wordle rule verifier: word ∈ dict, format valid, scoring exact."""
        from .env import WORD_SET, _score_guess

        if not (len(guess) == 5 and guess.isalpha()):
            return 0.0, False, "format_invalid"
        if guess.lower() not in WORD_SET:
            return 0.0, False, "non_dictionary"
        if guess.lower() == target.lower():
            return 1.0, True, "exact_match"
        feedback = _score_guess(guess.lower(), target.lower())
        n_green = sum(1 for f in feedback if f.state == "green")
        n_yellow = sum(1 for f in feedback if f.state == "yellow")
        # Partial credit: 0.05 per green, 0.02 per yellow (caps at 0.35 max)
        partial = 0.05 * n_green + 0.02 * n_yellow
        return partial, partial > 0, f"green={n_green}_yellow={n_yellow}_partial={partial:.3f}"

    # -------- model layer --------
    def _model_wordle(self, guess: str, history: list[dict],
                       remaining_guesses: int) -> tuple[float | None, bool | None, str]:
        """Ask local Ollama judge: 'is this guess strategically sound?'"""
        prompt = (
            f"You are evaluating a Wordle guess. The player has {remaining_guesses} "
            f"guesses remaining. Past guesses + feedback:\n"
            f"{self._format_history(history)}\n\n"
            f"Current guess: {guess.upper()}\n\n"
            "Score this guess 0-1 on strategic soundness. Consider:\n"
            "- Does it use information from past feedback?\n"
            "- Does it explore unused common letters?\n"
            "- Does it avoid letters already known to be absent?\n\n"
            'Respond with JSON only: {"score": 0.XX, "reasoning": "<one sentence>"}'
        )
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json", "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 4096},
                },
                timeout=30,
            )
            r.raise_for_status()
            import json as _json
            content = r.json()["message"]["content"]
            obj = _json.loads(content)
            score = float(obj.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            return score, score >= 0.5, obj.get("reasoning", "")[:200]
        except Exception as e:  # noqa: BLE001
            return None, None, f"unavailable: {str(e)[:120]}"

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "(no prior guesses)"
        lines = []
        for i, h in enumerate(history):
            fb = h.get("feedback") or []
            colors = " ".join(f"{f['letter']}={f['state'][0].upper()}"
                                for f in fb)
            lines.append(f"  {i+1}. {h['guess']} → {colors}")
        return "\n".join(lines)

    # -------- composite + alarm --------
    def verify(self, guess: str, target: str, history: list[dict],
                 remaining_guesses: int = 6,
                 use_model: bool = True) -> VerifierResult:
        t0 = time.time()
        rule_score, rule_pass, rule_reason = self._rule_wordle(guess, target, history)

        model_score: float | None = None
        model_pass: bool | None = None
        model_reason: str | None = None
        if use_model:
            model_score, model_pass, model_reason = self._model_wordle(
                guess, history, remaining_guesses)

        # Composite
        if model_score is not None:
            composite = rule_score * (0.5 + 0.5 * model_score)
        else:
            composite = rule_score

        # Disagreement (if model available)
        disagreement = None
        alarm = False
        if model_score is not None:
            disagreement = abs(rule_score - model_score)
            self.audit.rolling_disagreement.append(disagreement)
            if (len(self.audit.rolling_disagreement) >= 10 and
                    sum(self.audit.rolling_disagreement) / len(self.audit.rolling_disagreement)
                    > self.DISAGREEMENT_THRESHOLD):
                alarm = True
                self.audit.n_disagreement_alarms += 1

        self.audit.n_calls += 1
        if rule_pass:
            self.audit.n_rule_pass += 1
        if model_pass:
            self.audit.n_model_pass += 1

        return VerifierResult(
            rule_score=rule_score, rule_pass=rule_pass, rule_reason=rule_reason,
            model_score=model_score, model_pass=model_pass, model_reason=model_reason,
            composite_score=round(composite, 4),
            disagreement=round(disagreement, 4) if disagreement is not None else None,
            alarm=alarm,
            elapsed_s=round(time.time() - t0, 3),
        )

    def export_audit(self) -> dict:
        n = max(1, self.audit.n_calls)
        avg_dis = (sum(self.audit.rolling_disagreement) /
                    max(1, len(self.audit.rolling_disagreement))
                    if self.audit.rolling_disagreement else 0.0)
        return {
            "n_calls": self.audit.n_calls,
            "rule_pass_rate": round(self.audit.n_rule_pass / n, 4),
            "model_pass_rate": round(self.audit.n_model_pass / n, 4),
            "n_disagreement_alarms": self.audit.n_disagreement_alarms,
            "rolling_avg_disagreement": round(avg_dis, 4),
            "alarm_threshold": self.DISAGREEMENT_THRESHOLD,
            "framework": "RLVR dual-verifier (rule × model · §31-33 hardened)",
        }


def smoke_dual(n_trials: int = 5) -> dict:
    """Smoke test: 5 guesses against target=BRAIN, demonstrate rule + model agreement."""
    dv = DualVerifier()
    target = "brain"
    test_cases = [
        ("about", "first guess"),
        ("crane", "good explorer"),
        ("braid", "5-letter alpha, gets 4 greens"),
        ("brawn", "4 letters match positions"),
        ("brain", "exact match"),
    ]
    results = []
    for guess, note in test_cases[:n_trials]:
        r = dv.verify(guess, target, history=[], remaining_guesses=5,
                       use_model=True)
        results.append({
            "guess": guess, "note": note,
            "rule_score": r.rule_score, "rule_reason": r.rule_reason,
            "model_score": r.model_score,
            "composite": r.composite_score,
            "disagreement": r.disagreement,
            "alarm": r.alarm,
        })
    return {
        "target": target,
        "n_trials": len(results),
        "results": results,
        "audit": dv.export_audit(),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = smoke_dual()
    print(json.dumps(res, indent=2))
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[3]
    receipt = REPO / "tests" / "receipts" / "dual_verifier_smoke.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nReceipt: {receipt}")
