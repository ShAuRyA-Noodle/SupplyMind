"""ranker.py — severity-modulated ranker over the curated India + Gulf atlases.

Logic:

    - Atlas sectors are ranked 1..N by their canonical (peace-time) exposure.
    - At runtime, we apply a severity multiplier (input scalar in [0, 1]) plus
      live-signal boosts (e.g. if NewsAPI returned an LPG-related headline,
      boost the LPG sector by +0.05 risk-score).
    - Output: a list of dicts ranked by current_risk_score, each carrying the
      ORIGINAL exposure_facts + first_symptom + an _evidence drawer.

We do NOT invent new sectors or new percentages. The ranker is a *re-orderer*
of what the curated JSONs already declare.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from . import provenance

logger = logging.getLogger(__name__)


SECTOR_BOOST_KEYWORDS = {
    "lpg_commercial": ["lpg", "cooking gas", "cylinder", "hotpot", "restaurant"],
    "fertilizer_urea_ammonia": ["urea", "ammonia", "fertilizer", "fertiliser", "DAP"],
    "refining_diesel_petrol_atf": ["diesel", "petrol", "atf", "jet fuel", "refining", "crude"],
    "petrochemicals_naphtha_lpg_feedstock": ["naphtha", "polymer", "polyethylene", "plastic", "petchem"],
    "shipping_logistics_war_risk": ["insurance", "war risk", "tanker", "vessel", "shipping", "maersk", "msc"],
    "qatar_lng_export": ["qatar", "lng", "ras laffan", "liquefied natural gas"],
    "uae_jebel_ali_transshipment": ["jebel ali", "dp world", "container", "transshipment"],
    "fujairah_bypass_bunkering": ["fujairah", "bunker", "adcop", "bypass"],
    "gcc_food_imports": ["food", "grain", "wheat", "imports", "supermarket"],
    "gcc_desalination_power": ["desalination", "power", "electricity", "water"],
}


@dataclass
class RankedSector:
    rank_current: int
    sector_id: str
    sector_name: str
    rank_baseline: int
    severity_multiplier: float
    live_signal_boost: float
    current_risk_score: float
    exposure_facts: list[dict]
    first_symptom: str
    first_symptom_evidence: list[dict]
    sector_evidence: dict   # _evidence for the sector ranking itself

    def to_dict(self) -> dict:
        return {
            "rank_current": self.rank_current,
            "sector_id": self.sector_id,
            "sector_name": self.sector_name,
            "rank_baseline": self.rank_baseline,
            "severity_multiplier": round(self.severity_multiplier, 3),
            "live_signal_boost": round(self.live_signal_boost, 3),
            "current_risk_score": round(self.current_risk_score, 4),
            "exposure_facts": self.exposure_facts,
            "first_symptom": self.first_symptom,
            "first_symptom_evidence": self.first_symptom_evidence,
            "_evidence": self.sector_evidence,
        }


def _baseline_score(rank: int, n: int) -> float:
    """Convert a baseline rank (1 is highest exposure) to a [0, 1] score."""
    if n <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (n - 1) * 0.6   # rank 1 -> 1.0, rank N -> 0.4


def _live_signal_boost(sector_id: str, signal_text: str) -> float:
    """Boost sector score if live signals mention sector-specific keywords."""
    if not signal_text:
        return 0.0
    text = signal_text.lower()
    keywords = SECTOR_BOOST_KEYWORDS.get(sector_id, [])
    hits = sum(1 for kw in keywords if kw in text)
    if hits == 0:
        return 0.0
    return min(0.10, 0.03 * hits)   # 3% per keyword hit, cap +10%


def rank(country_atlas: dict, severity: float, signal_text: str = "") -> list[RankedSector]:
    sectors_in = country_atlas["sectors"]
    n = len(sectors_in)
    out: list[RankedSector] = []

    for sec in sectors_in:
        baseline = _baseline_score(int(sec["rank"]), n)
        sev_mult = 0.6 + 0.8 * max(0.0, min(1.0, float(severity)))   # severity=0 -> 0.6, severity=1 -> 1.4
        boost = _live_signal_boost(sec["sector_id"], signal_text)
        current = round(baseline * sev_mult + boost, 4)

        sector_evidence = provenance.model_estimate(
            derivation=(
                f"current_risk_score = baseline_score(rank={sec['rank']}, n={n}) "
                f"* severity_multiplier({severity:.2f}) + live_signal_boost(matched={int(boost > 0)})"
                f" = {baseline:.3f} * {sev_mult:.3f} + {boost:.3f}"
            )
        ).to_dict()

        out.append(RankedSector(
            rank_current=0,   # filled after sort
            sector_id=sec["sector_id"],
            sector_name=sec["sector_name"],
            rank_baseline=int(sec["rank"]),
            severity_multiplier=sev_mult,
            live_signal_boost=boost,
            current_risk_score=current,
            exposure_facts=sec.get("exposure_facts", []),
            first_symptom=sec["first_symptom_when_hormuz_hits"],
            first_symptom_evidence=sec.get("first_symptom_evidence", []),
            sector_evidence=sector_evidence,
        ))

    out.sort(key=lambda s: s.current_risk_score, reverse=True)
    for i, s in enumerate(out, start=1):
        s.rank_current = i
    return out


def recommended_actions(india_ranks: list[RankedSector],
                        gulf_ranks: list[RankedSector]) -> list[dict]:
    """Produce action recommendations LINKED to the highest-ranked sectors.

    Every action is tied to a specific sector + a specific evidence row from
    that sector. We do not invent generic actions; each recommendation cites
    why it follows from the data.
    """
    actions: list[dict] = []
    top_india = india_ranks[:3]
    top_gulf = gulf_ranks[:3]

    sector_to_action_map = {
        "lpg_commercial": (
            "Prioritise domestic LPG (14.2 kg) allocation; release strategic LPG reserve to commercial channel only after domestic stable; explore incremental LPG cargoes from US Gulf or West Africa.",
            "Government precedent: 2022 PIB advisory directed prioritised domestic allocation during commodity stress."
        ),
        "fertilizer_urea_ammonia": (
            "Protect fertiliser-sector gas allocation per Pool Pricing Mechanism; pre-stock urea ahead of next sowing window; activate emergency import contracts with non-Hormuz LNG suppliers (US, Australia).",
            "Anchored in Department of Fertilizers Annual Report 2023-24: ~80% of urea capacity is gas-feedstock."
        ),
        "refining_diesel_petrol_atf": (
            "Diversify crude basket toward non-Hormuz origins (US, West Africa, Russia, Latin America); raise OMC inventory floor 5-7 days; alert DGCA + airlines to ATF cost passthrough.",
            "MoPNG IPNG 2023-24 confirms 33-40% of crude basket is Hormuz-origin; non-Hormuz substitution is the operational lever."
        ),
        "petrochemicals_naphtha_lpg_feedstock": (
            "Polymer producers: build inventory; consumer goods companies: lock forward polymer contracts; Reliance Jamnagar / IOCL Panipat: review feedstock alternates.",
            "Polymer-feedstock 30-60 day lag is documented by S&P Platts; act inside the lag window."
        ),
        "shipping_logistics_war_risk": (
            "Pre-negotiate war-risk insurance caps with INSA carriers; route non-urgent cargo via Cape route; activate port congestion monitoring at Mundra, Kandla, Sikka, Fujairah, Jebel Ali.",
            "Lloyd's List documents 10-20× premium spikes in 2019; lock pricing while still possible."
        ),
        "qatar_lng_export": (
            "Asian importers (India, Japan, Korea, China): activate force-majeure clauses; bid for incremental US Gulf cargoes (Sabine Pass, Plaquemines).",
            "IGU 2024: Qatar = 18% of global LNG; Hormuz loss is a 15-20% global LNG supply shock."
        ),
        "uae_jebel_ali_transshipment": (
            "Reroute time-sensitive containers via Salalah (Oman, outside Hormuz) or Damietta (Egypt); accept 5-10 day delays as the cost of avoiding war-risk surcharge.",
            "Drewry analysis confirms Salalah and Damietta as the established alt-hubs."
        ),
        "fujairah_bypass_bunkering": (
            "Pre-book Fujairah berths and ADCOP capacity at current rates; bunker in Fujairah, NOT Jebel Ali, while transit risk persists.",
            "ADCOP at 1.5 mb/d nameplate is the only meaningful pipeline bypass; first-come-first-served once the rush starts."
        ),
        "gcc_food_imports": (
            "GCC governments: release Strategic Food Reserve (UAE has 3-month buffer); pre-book grain cargoes via non-Gulf routes (Salalah, Aqaba); work with UN/FAO if stress prolonged.",
            "FAO + UAE MoCCAE confirm 90% food import dependence."
        ),
        "gcc_desalination_power": (
            "Activate gas-fired plant load-shedding plans; bring solar capacity up to nameplate; prepare emergency LNG charters from Australia / US.",
            "IEA UAE country profile: 60% gas-fired power; thermal desalination is the chained risk."
        ),
    }

    for sec in top_india:
        if sec.sector_id in sector_to_action_map:
            action_text, justification = sector_to_action_map[sec.sector_id]
            actions.append({
                "action": action_text,
                "tied_to_sector": sec.sector_id,
                "country_focus": "India",
                "current_risk_score": sec.current_risk_score,
                "_evidence": provenance.Evidence(
                    source_type="model_estimate",
                    derivation=f"Action follows the curated sector_to_action_map for sector_id={sec.sector_id}. Justification: {justification}",
                ).to_dict(),
            })
    for sec in top_gulf:
        if sec.sector_id in sector_to_action_map:
            action_text, justification = sector_to_action_map[sec.sector_id]
            actions.append({
                "action": action_text,
                "tied_to_sector": sec.sector_id,
                "country_focus": "Gulf",
                "current_risk_score": sec.current_risk_score,
                "_evidence": provenance.Evidence(
                    source_type="model_estimate",
                    derivation=f"Action follows the curated sector_to_action_map for sector_id={sec.sector_id}. Justification: {justification}",
                ).to_dict(),
            })
    return actions
