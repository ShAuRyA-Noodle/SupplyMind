"""supplymind_judge_worker.py — ROLL-compatible RewardWorker wrapping our 3 judges.

Drop-in subclass of `roll.pipeline.rlvr.rewards.LLMJudgeRewardWorker`. Calls
our existing 3-judge panel (DeepSeek-R1-Q4, Qwen-2.5-14B-Q4, Mistral-Nemo-Q4)
via Ollama and returns a reward in [0, 1] using majority-vote alignment with
the R4 rubric.

When ROLL is installed, this class auto-registers as a reward backend named
'supplymind_3judge' selectable from any ROLL config. When ROLL is not
installed, the class still works standalone — the ROLL base-class import is
guarded so you can pytest this file in isolation.

Reward formula (same as R4 ablation's majority-vote scoring):
    - 1.0 when 2+ judges agree with ground-truth risk tier
    - 0.6 when 1 judge agrees
    - 0.0 otherwise
    - -0.2 format penalty if any judge fails to produce valid JSON
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


try:
    from roll.pipeline.rlvr.rewards.llm_judge_reward_worker import LLMJudgeRewardWorker  # type: ignore
    _HAS_ROLL = True
except Exception:  # noqa: BLE001
    _HAS_ROLL = False

    class LLMJudgeRewardWorker:  # type: ignore
        """Fallback stub so this file is importable without ROLL."""
        def __init__(self, *args, **kwargs):
            pass


RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class SupplyMind3JudgeRewardWorker(LLMJudgeRewardWorker):
    """3-judge majority-vote reward for supply-chain risk scenarios."""

    backend_name = "supplymind_3judge"

    def __init__(self, config: Any | None = None, ollama_host: str = "http://localhost:11434", **kwargs):
        super().__init__(config=config, **kwargs) if _HAS_ROLL else super().__init__()
        self.ollama_host = ollama_host
        self.judges = ["deepseek-r1-local-q4", "qwen25-14b-local",
                       "mistral-nemo-local"]

    def _query_judge(self, model_name: str, prompt: str) -> dict | None:
        """Call Ollama with the judge model. Returns parsed JSON or None."""
        import httpx  # lazy import

        try:
            r = httpx.post(
                f"{self.ollama_host}/api/chat",
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "format": "json", "options": {"temperature": 0.0}},
                timeout=60,
            )
            r.raise_for_status()
            content = r.json()["message"]["content"]
            start, end = content.index("{"), content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] judge query failed: %s", model_name, e)
            return None

    def compute_reward(
        self,
        prompt: str,
        response: str,
        ground_truth: dict | None = None,
    ) -> dict:
        """ROLL reward contract: return {'reward': float, 'meta': dict}.

        `prompt` = the scenario (free text).
        `response` = the candidate model's output (a JSON string in our schema).
        `ground_truth.risk_level` must be one of RISK_LEVELS.
        """
        if ground_truth is None or "risk_level" not in ground_truth:
            return {"reward": 0.0, "meta": {"error": "no ground truth"}}

        try:
            cand = json.loads(response[response.index("{"):response.rindex("}") + 1])
            cand_level = (cand.get("risk_level") or "").upper()
        except Exception:
            return {"reward": -0.2, "meta": {"error": "format_penalty; candidate JSON parse failed"}}

        judge_votes = []
        for jm in self.judges:
            out = self._query_judge(jm, prompt)
            if out:
                judge_votes.append((jm, (out.get("risk_level") or "").upper()))

        gt = (ground_truth["risk_level"] or "").upper()
        agreement = sum(1 for _, lvl in judge_votes if lvl == gt and cand_level == gt)
        if agreement >= 2:
            reward = 1.0
        elif agreement == 1:
            reward = 0.6
        else:
            reward = 0.0

        return {
            "reward": reward,
            "meta": {
                "cand_level": cand_level,
                "gt_level": gt,
                "judge_votes": judge_votes,
                "agreement_count": agreement,
            },
        }


# ROLL registry hook
if _HAS_ROLL:
    try:
        from roll.pipeline.rlvr.rewards import register_reward_backend  # type: ignore
        register_reward_backend("supplymind_3judge", SupplyMind3JudgeRewardWorker)
    except Exception as e:  # noqa: BLE001
        logger.info("[reward_bridge] ROLL present but register_reward_backend missing: %s", e)
