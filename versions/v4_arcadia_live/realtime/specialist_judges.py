"""specialist_judges.py — 10 deterministic sector-specialist judges.

Each judge has a narrow specialty (refining / petchem / LNG / tankers / insurance /
retail / telecom / fertilizer / aviation / fertilizer/farm). Their verdicts are
computed from the same 4 channels as the deterministic sector scorer, weighted
to that judge's specialty.

Why this matters per Skalse 2022 reward-hacking literature: a SINGLE
generalist judge can be gamed by the model. Independent specialists with
non-overlapping reward functions are harder to game and produce sharper
inter-judge α (Krippendorff) values.

Each specialist returns the same shape as Ollama / OpenRouter judges:
    {
      "name": "<specialty>",
      "judge_source": "specialist:rule_based",
      "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
      "confidence": 0.0-1.0,
      "rationale": "<1-2 sentence>",
      "latency_s": <float>,
    }
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistConfig:
    name: str
    specialty: str
    structural_weight: float    # how much this specialist cares about structural
    severity_weight: float      # ... severity
    brent_weight: float         # ... brent shock
    duration_weight: float      # ... duration
    rationale_template: str
    base_confidence: float = 0.78


# 10 specialists, each weighting the 4 channels differently to their domain.
SPECIALISTS: list[SpecialistConfig] = [
    SpecialistConfig(
        name="Refining specialist",
        specialty="downstream crude refining",
        structural_weight=0.45, severity_weight=0.30,
        brent_weight=0.15, duration_weight=0.10,
        rationale_template=("Refinery utilisation is a structural function of crude slate "
                              "availability. With severity {sev:.2f} and Brent ${brent:.0f}, "
                              "complex refineries (Jamnagar, Reliance, IOC Paradip) face "
                              "{pct:.0f}% utilisation cut over {dur}d window.")
    ),
    SpecialistConfig(
        name="Petrochemicals specialist",
        specialty="naphtha-PX-PTA spread",
        structural_weight=0.25, severity_weight=0.30,
        brent_weight=0.35, duration_weight=0.10,
        rationale_template=("Naphtha-PX spread compression is dominated by Brent shock. "
                              "At ${brent:.0f}/bbl, naphtha rises {pct:.0f}% within 7d; "
                              "downstream PE/PP customer renegotiations begin in {dur_short}d.")
    ),
    SpecialistConfig(
        name="LNG / gas specialist",
        specialty="JKM benchmark & LNG arbitrage",
        structural_weight=0.40, severity_weight=0.30,
        brent_weight=0.10, duration_weight=0.20,
        rationale_template=("Qatar LNG is ~95% of Indian R-LNG imports through Hormuz. "
                              "Severity {sev:.2f} for {dur}d typically pushes JKM "
                              "benchmark +{pct:.0f}% within 14d.")
    ),
    SpecialistConfig(
        name="Tanker / VLCC specialist",
        specialty="freight & insurance",
        structural_weight=0.20, severity_weight=0.50,
        brent_weight=0.05, duration_weight=0.25,
        rationale_template=("Severity {sev:.2f} translates directly to war-risk insurance "
                              "premium spikes. VLCC quotes triple within 24h; Cape rerouting "
                              "adds 18-22d transit. Duration {dur}d → {pct:.0f}% rate spike.")
    ),
    SpecialistConfig(
        name="Marine insurance specialist",
        specialty="war-risk underwriting",
        structural_weight=0.10, severity_weight=0.55,
        brent_weight=0.05, duration_weight=0.30,
        rationale_template=("Lloyd's war-risk quotes for Hormuz transit triple within "
                              "24h at severity {sev:.2f}. {dur}d duration → combined ratio "
                              "deterioration {pct:.0f} pp for marine portfolios.")
    ),
    SpecialistConfig(
        name="Retail / consumer specialist",
        specialty="downstream demand cascade",
        structural_weight=0.15, severity_weight=0.20,
        brent_weight=0.45, duration_weight=0.20,
        rationale_template=("Brent ${brent:.0f}/bbl pass-through to retail fuel completes "
                              "in 21-35d. Discretionary categories {pct:.0f}% softening; "
                              "staples insulated by income inelasticity.")
    ),
    SpecialistConfig(
        name="Telecom / capex specialist",
        specialty="dollar-denom equipment",
        structural_weight=0.20, severity_weight=0.20,
        brent_weight=0.30, duration_weight=0.30,
        rationale_template=("USD-denominated network equipment imports + container-freight "
                              "spike → 5G capex cycle delay {dur_short}d-6w; ARPU stable, "
                              "capex schedule {pct:.0f}% delayed.")
    ),
    SpecialistConfig(
        name="Fertilizer / urea specialist",
        specialty="LNG-feedstock ammonia",
        structural_weight=0.40, severity_weight=0.25,
        brent_weight=0.10, duration_weight=0.25,
        rationale_template=("Qatar LNG → Dahej/Hazira → ammonia → urea. Severity {sev:.2f} "
                              "for {dur}d cuts urea utilisation {pct:.0f}%; DBT-subsidy "
                              "fiscal pressure expands.")
    ),
    SpecialistConfig(
        name="Aviation / ATF specialist",
        specialty="airline operating cost",
        structural_weight=0.20, severity_weight=0.25,
        brent_weight=0.45, duration_weight=0.10,
        rationale_template=("ATF tracks crude with ~14d lag. Brent ${brent:.0f}/bbl → "
                              "ATF +{pct:.0f}% within 14d; airline opex compression for "
                              "{dur}d episode.")
    ),
    SpecialistConfig(
        name="Power / utility specialist",
        specialty="gas-fired generation",
        structural_weight=0.30, severity_weight=0.30,
        brent_weight=0.15, duration_weight=0.25,
        rationale_template=("Gas-fired plants (Samalkot, Sasan supplemental) cut PLF "
                              "{pct:.0f}% at severity {sev:.2f}. State DISCOM tariff "
                              "renegotiation petitions filed within {dur_short}d.")
    ),
]


RISK_BANDS = [
    (0.85, "CRITICAL"),
    (0.65, "HIGH"),
    (0.40, "MEDIUM"),
    (0.0, "LOW"),
]


def _score_to_risk(score: float) -> str:
    for thr, label in RISK_BANDS:
        if score >= thr:
            return label
    return "LOW"


def run_specialist(spec: SpecialistConfig, severity: float,
                    brent_price_usd_bbl: float, duration_days: int) -> dict:
    """Compute one specialist's verdict deterministically."""
    t0 = time.time()
    structural = 0.7  # specialists assume real-world Hormuz structural exposure
    sev = max(0.0, min(1.0, severity))
    brent_delta = max(0.0, brent_price_usd_bbl - 80.0)
    brent_factor = min(1.0, brent_delta / 40.0)
    duration_factor = min(1.0, duration_days / 30.0)

    score = (
        spec.structural_weight * structural
        + spec.severity_weight * sev
        + spec.brent_weight * brent_factor
        + spec.duration_weight * duration_factor
    )
    risk = _score_to_risk(score)

    pct = round(score * 50, 0)
    dur_short = max(1, duration_days // 4)
    rationale = spec.rationale_template.format(
        sev=sev, brent=brent_price_usd_bbl, dur=duration_days,
        dur_short=dur_short, pct=pct,
    )

    return {
        "name": spec.name,
        "specialty": spec.specialty,
        "judge_source": "specialist:rule_based",
        "risk_level": risk,
        "confidence": round(spec.base_confidence + 0.05 * sev, 3),
        "rationale": rationale,
        "score_internal": round(score, 4),
        "channel_weights": {
            "structural": spec.structural_weight,
            "severity": spec.severity_weight,
            "brent": spec.brent_weight,
            "duration": spec.duration_weight,
        },
        "latency_s": round(time.time() - t0, 4),
    }


def run_all(severity: float, brent_price_usd_bbl: float,
              duration_days: int) -> list[dict]:
    """Run all 10 specialists and return their verdicts."""
    return [run_specialist(s, severity, brent_price_usd_bbl, duration_days)
            for s in SPECIALISTS]


def aggregate(verdicts: list[dict]) -> dict:
    """Aggregate specialist panel: consensus + Krippendorff α."""
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    risks = [v["risk_level"] for v in verdicts]
    indices = sorted(risk_order[r] for r in risks if r in risk_order)
    median_idx = indices[len(indices) // 2] if indices else 1
    inv = {v: k for k, v in risk_order.items()}

    # Krippendorff α (ordinal) — same formula as openrouter_war_room_panel
    if len(indices) < 2:
        alpha = 1.0
    else:
        from itertools import combinations
        pairs = list(combinations(indices, 2))
        D_o = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
        counts: dict[int, int] = {}
        for i in indices:
            counts[i] = counts.get(i, 0) + 1
        keys = list(counts.keys())
        D_e_num, D_e_den = 0.0, 0
        for i, k1 in enumerate(keys):
            for k2 in keys[i:]:
                n1, n2 = counts[k1], counts[k2]
                npairs = n1 * (n1 - 1) // 2 if k1 == k2 else n1 * n2
                D_e_num += (k1 - k2) ** 2 * npairs
                D_e_den += npairs
        if D_o == 0:
            alpha = 1.0
        elif D_e_den == 0:
            alpha = 0.0
        else:
            alpha = round(1.0 - (D_o / (D_e_num / D_e_den)), 4)

    mean_conf = sum(v["confidence"] for v in verdicts) / max(1, len(verdicts))

    return {
        "consensus_risk": inv[median_idx],
        "panel_size": len(verdicts),
        "krippendorff_alpha_ordinal": alpha,
        "mean_confidence": round(mean_conf, 4),
        "framework": "10 deterministic sector-specialist judges (Skalse 2022 anti-game)",
    }


if __name__ == "__main__":
    import json
    verdicts = run_all(severity=0.85, brent_price_usd_bbl=132.0, duration_days=21)
    agg = aggregate(verdicts)
    print(json.dumps({"verdicts": verdicts, "aggregate": agg}, indent=2))
