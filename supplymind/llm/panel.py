"""supplymind.llm.panel — configurable multi-judge panel over the provider
layer (CLAUDE.md §7.2; PRODUCT_THESIS supporting-credibility R14).

An N-judge panel scores a scenario on the ordinal risk scale; the panel reports
inter-judge agreement (Krippendorff's α, ordinal) plus a median consensus. Judge
slugs are ENV-DRIVEN (``SUPPLYMIND_PANEL_MODELS``, comma-separated) or passed
explicitly — never hardcoded, per CLAUDE.md §0.

This is the scaffold: the code path is complete and exercised offline via
mock-transport tests. The LIVE 12-judge run + real α (WP7.2) is BLOCKED-ON-KEY
(OpenRouter key revoked). Unparseable / failed judges are counted SEPARATELY and
excluded from α — never faked into agreement.

Reference: Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench".
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
from dataclasses import dataclass, field
from itertools import combinations

from supplymind.llm.providers import LLMProvider, get_provider

PANEL_MODELS_ENV = "SUPPLYMIND_PANEL_MODELS"  # comma-separated judge slugs

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
_INV_RISK = {v: k for k, v in RISK_ORDER.items()}

JUDGE_SYSTEM_PROMPT = (
    "You are a senior supply-chain risk analyst. Score the scenario on the "
    "ordinal scale LOW/MEDIUM/HIGH/CRITICAL and give a calibrated confidence. "
    "Respond with ONLY a JSON object."
)

JUDGE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["risk_level", "confidence", "reason"],
    "properties": {
        "risk_level": {"type": "string", "enum": list(RISK_ORDER)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
}


def resolve_panel_models(explicit: list[str] | None = None) -> list[str]:
    """Judge slugs: explicit arg wins, else ``SUPPLYMIND_PANEL_MODELS`` env.
    Empty result -> loud (caller must fail, never invent a panel)."""
    if explicit:
        return [s.strip() for s in explicit if s.strip()]
    raw = os.environ.get(PANEL_MODELS_ENV, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def krippendorff_alpha_ordinal(values: list[str]) -> float | None:
    """Ordinal Krippendorff's α for a single scenario rated by multiple judges.

    Returns None when fewer than 2 valid ratings exist (α undefined — reported as
    such, not faked to 0 or 1). Perfect agreement returns 1.0 by convention.
    """
    valid = [v for v in values if v in RISK_ORDER]
    if len(valid) < 2:
        return None
    indices = [RISK_ORDER[v] for v in valid]
    pairs = list(combinations(indices, 2))
    if not pairs:
        return None
    d_o = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
    if d_o == 0:
        return 1.0
    counts: dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1
    keys = list(counts.keys())
    d_e_num = 0.0
    d_e_den = 0
    for i, k1 in enumerate(keys):
        for k2 in keys[i:]:
            n1, n2 = counts[k1], counts[k2]
            npairs = n1 * (n1 - 1) // 2 if k1 == k2 else n1 * n2
            d_e_num += (k1 - k2) ** 2 * npairs
            d_e_den += npairs
    if d_e_den == 0 or d_e_num == 0:
        return None
    d_e = d_e_num / d_e_den
    return round(1.0 - (d_o / d_e), 4)


def consensus_risk(values: list[str]) -> str | None:
    valid = sorted(RISK_ORDER[v] for v in values if v in RISK_ORDER)
    if not valid:
        return None
    return _INV_RISK[valid[len(valid) // 2]]


@dataclass
class PanelResult:
    scenario: str
    judges: list[str]
    per_judge: list[dict] = field(default_factory=list)
    n_ok: int = 0
    n_unparseable: int = 0
    n_failed: int = 0
    consensus_risk: str | None = None
    krippendorff_alpha_ordinal: float | None = None
    mean_confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario[:200],
            "panel_size": len(self.judges),
            "judges": self.judges,
            "n_ok": self.n_ok,
            "n_unparseable": self.n_unparseable,
            "n_failed": self.n_failed,
            "consensus_risk": self.consensus_risk,
            "krippendorff_alpha_ordinal": self.krippendorff_alpha_ordinal,
            "mean_confidence": self.mean_confidence,
            "per_judge": self.per_judge,
        }


class JudgePanel:
    """N-judge panel. Slugs env-driven; provider from the shared factory."""

    def __init__(self, models: list[str] | None = None,
                 provider: LLMProvider | None = None) -> None:
        self.models = resolve_panel_models(models)
        if not self.models:
            raise ValueError(
                f"no panel judges configured: set {PANEL_MODELS_ENV} (comma-"
                "separated slugs) or pass models=[...] — no slugs are hardcoded"
            )
        self._provider = provider

    def _prov(self) -> LLMProvider:
        return self._provider if self._provider is not None else get_provider()

    async def _one_judge(self, model: str, scenario: str) -> dict:
        prov = self._prov()
        resp = await prov.complete(
            [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
             {"role": "user", "content": scenario}],
            model=model, schema=JUDGE_SCHEMA, max_tokens=200, temperature=0.2,
        )
        if resp.ok and resp.parsed:
            return {"model": model, "status": "ok",
                    "risk_level": resp.parsed.get("risk_level"),
                    "confidence": float(resp.parsed.get("confidence") or 0.0),
                    "reason": (resp.parsed.get("reason") or "")[:300],
                    "latency_s": resp.latency_s}
        # Distinguish a transport/auth failure from an unparseable reply — both
        # are surfaced honestly and excluded from the agreement statistic.
        status = "failed" if resp.error else "unparseable"
        return {"model": model, "status": status,
                "degrade_reason": resp.degrade_reason or resp.error,
                "latency_s": resp.latency_s}

    async def run(self, scenario: str) -> PanelResult:
        judged = await asyncio.gather(
            *[self._one_judge(m, scenario) for m in self.models]
        )
        ok = [j for j in judged if j["status"] == "ok"]
        risk_levels = [j["risk_level"] for j in ok if j.get("risk_level")]
        confs = [j["confidence"] for j in ok if j.get("confidence") is not None]
        return PanelResult(
            scenario=scenario,
            judges=self.models,
            per_judge=judged,
            n_ok=len(ok),
            n_unparseable=sum(1 for j in judged if j["status"] == "unparseable"),
            n_failed=sum(1 for j in judged if j["status"] == "failed"),
            consensus_risk=consensus_risk(risk_levels),
            krippendorff_alpha_ordinal=krippendorff_alpha_ordinal(risk_levels),
            mean_confidence=round(sum(confs) / len(confs), 4) if confs else None,
        )

    def run_sync(self, scenario: str) -> PanelResult:
        """Sync wrapper safe whether or not a loop is already running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(scenario))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self.run(scenario))).result()
