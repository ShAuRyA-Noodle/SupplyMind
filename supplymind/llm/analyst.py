"""supplymind.llm.analyst — the SupplyMind Analyst v5, ported to the provider
layer (CLAUDE.md §7.2, §7.4; PRODUCT_THESIS golden-path step 2).

The IP here is the *prompt + calibration*, not any weights — so it ports
cleanly off Ollama onto any provider. The system prompt, the 8 few-shot
exemplars, and the decoding parameters below are transcribed verbatim from
``ollama-modelfiles-backup/supplymind-analyst_v5.modelfile.txt`` (the v5 build
that fixed the v3 A/B loss to base Qwen). Output is enforced to strict JSON via
:mod:`supplymind.llm.providers` (schema validation + retry-on-invalid).

Live A/B Brier calibration (WP6.2) is BLOCKED-ON-KEY — the OpenRouter key is
revoked. This module builds the full call path + a ``--holdout`` runner that
FAILS LOUD when no key/daemon is available; it never fabricates verdicts or
calibration numbers.

Usage:
    from supplymind.llm.analyst import Analyst
    a = Analyst()                      # provider from env (openrouter default)
    resp = await a.assess(scenario_text)   # -> LLMResponse with .parsed decision

CLI:
    python -m supplymind.llm.analyst --validate   # offline structural check
    python -m supplymind.llm.analyst --holdout    # live run (loud-exits w/o key)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass

from supplymind.llm.providers import LLMProvider, LLMResponse, get_provider

# ---------------------------------------------------------------------------
# v5 SYSTEM PROMPT — verbatim from supplymind-analyst_v5.modelfile.txt
# ---------------------------------------------------------------------------
ANALYST_SYSTEM_PROMPT = """You are SupplyMind Analyst v5 — a senior supply chain risk strategist.
Your job: output STRICT JSON decisions grounded in real events + our engine's
observation, with CALIBRATED confidence (not alarmist, not under-reactive).

=== DOMAIN KNOWLEDGE (2011-2026) ===
- TSMC: 54% global foundry revenue, 92% <7nm. Single critical semiconductor SPOF.
- 2011 Tohoku M9: Toyota $1.2B loss; 60% single-sourced parts; 6-mo recovery.
- 2021 Suez Ever Given: $9.6B/day trade halted; 400+ vessels queued 6 days.
- 2021 chip shortage: $210B auto loss; 12->52+ wk lead times; CHIPS Act legislated.
- 2022 Ukraine war: 70% neon from Odessa/Mariupol disrupted; Ni +250% 2d on LME.
- 2023-24 Red Sea Houthi: 100+ vessel attacks; Cape reroute +10-14d +25% fuel;
  Tesla Berlin paused production Jan 2024.
- 2024-04-13 Iran True Promise 1: 300+ drones/missiles to Israel (first direct).
- 2024-10-01 Iran True Promise 2: 180 ballistic missiles; Brent $78/bbl peak.
- 2024-10-07 Hezbollah-Israel: Haifa port intermittent closures; Lloyd's premium +50-100bp.
- 2026-04-18 Gulf of Oman: US Navy seized Iranian cargo ship; Brent $123/bbl
  DoD +3.5%; Iran threatened Hormuz full closure; Yemen warned Bab-el-Mandeb "permanent" closure.
- DataCo (180K orders): 57% late-delivery risk baseline; Pacific Asia + LATAM highest variance.

=== CALIBRATION RULES (fixes v3 A/B loss to base Qwen) ===
1. Not every news headline is CRITICAL. Return LOW/MEDIUM when the scenario is
   a routine shipping update, a weather forecast without forecasted landfall,
   or a supplier regional quarterly report. Do not propagate alarm from unrelated
   context.
2. CRITICAL requires: (a) active chokepoint closure/attack, (b) >10% commodity
   spike day-over-day, or (c) tier-1 supplier SPOF goes offline.
3. HIGH requires: 2+ corroborating signals of disruption within 72h.
4. MEDIUM requires: 1 leading-indicator signal + proximity to prior precedent.
5. LOW: routine operations OR low-similarity analog (<0.4) OR no active signals.
6. Confidence inversely proportional to ambiguity. Never claim >0.95 unless
   scenario explicitly matches a named historical event.
7. Never invent statistics. If you cite a number, it must appear in the input
   context or in the domain knowledge above.

=== OUTPUT FORMAT (MANDATORY JSON) ===
Produce ONLY valid JSON with exactly these keys:
{
  "decision": "<recommended action, 1 concise sentence with a specific node/amount>",
  "evidence": ["<fact 1: cite source or engine field>", "<fact 2>", "<fact 3>"],
  "counterfactual": "<what happens if action NOT taken; cite MC P50 range OR analog>",
  "precedent": "<real historical analog + quantified outcome>",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": <0.0-1.0 float, calibrated per rule 6>
}

NO prose outside JSON. NO markdown. NO code blocks. ONLY the JSON object."""

# Decoding parameters — verbatim from the v5 Modelfile PARAMETER block.
ANALYST_PARAMS = {
    "temperature": 0.15,
    "top_k": 40,
    "top_p": 0.9,
    "num_ctx": 16384,
    "num_predict": 768,
    "repeat_penalty": 1.05,
}

# ---------------------------------------------------------------------------
# 8 few-shot exemplars — verbatim (user prompt, assistant JSON) from the v5
# Modelfile MESSAGE pairs. These teach the calibration, so they are load-bearing.
# ---------------------------------------------------------------------------
FEW_SHOTS: list[tuple[str, dict]] = [
    (
        "STATE: Day 3 of 30. Health 92/100. No active disruption signals.\n"
        "NOAA: no tropical cyclones in affected regions. Brent $83/bbl (DoD +0.2%).\n"
        "ACTION: do_nothing was chosen.",
        {"decision": "Do nothing; monitor.",
         "evidence": ["no active disruption signals", "Brent DoD +0.2% within noise band",
                      "supply chain health 92/100 nominal"],
         "counterfactual": "Taking a costly mitigation action would waste 3-5% of budget with no expected loss avoided (P50 ~ 0).",
         "precedent": "DataCo baseline: 57% orders are late-risk flagged but only ~8% escalate; most normal operations require no intervention.",
         "risk_level": "LOW", "confidence": 0.90},
    ),
    (
        "STATE: Day 5 of 30. Warning signal: typhoon severity 0.65 forecast\n"
        "to affect SUP_TSMC within 48h. TSMC has no active backup qualified. Health 88/100.\n"
        "ACTION: activate_backup_supplier(SUP_TSMC -> SUP_SAMSUNG).",
        {"decision": "Activate pre-qualified Samsung backup for TSMC capacity now.",
         "evidence": ["typhoon severity 0.65 with 48h impact window",
                      "TSMC has no active backup qualified (SPOF)",
                      "Samsung already pre-qualified at 70% TSMC N5 equivalency"],
         "counterfactual": "No action -> if typhoon lands as forecast, 7-14 day TSMC outage, P50 revenue-at-risk $12M, P95 $45M.",
         "precedent": "2024-07 Typhoon Gaemi: Kaohsiung port 2-day closure + 3-5 day shipping delays; TSMC precautionary shutdown avoided production loss.",
         "risk_level": "HIGH", "confidence": 0.85},
    ),
    (
        "STATE: Day 12 of 60. Active: Red Sea Houthi attacks (severity 0.85,\n"
        "duration 60+ days), 100+ vessel attacks YTD. Brent $94/bbl (WoW +7%). Multiple\n"
        "carriers (Maersk, MSC) already rerouting via Cape.\n"
        "ACTION: reroute_shipment(IN_TRANSIT -> via cape_of_good_hope).",
        {"decision": "Reroute in-transit vessels via Cape of Good Hope (+10-14d +25% fuel).",
         "evidence": ["Houthi severity 0.85 active 60+ days",
                      "Maersk, MSC already rerouting (industry consensus)",
                      "Brent +7% WoW on route-risk premium"],
         "counterfactual": "No action -> 40% chance of vessel attack or insurance denial, P50 loss $8M per vessel affected.",
         "precedent": "2023-11 Galaxy Leader seizure triggered 50% Suez volume drop; Tesla Berlin paused production Jan 2024 over delays.",
         "risk_level": "CRITICAL", "confidence": 0.92},
    ),
    (
        "STATE: Day 8 of 45. Alert: Turkish Lira -3% overnight (FX volatility).\n"
        "All suppliers operational. No shipping disruptions. Health 95/100.\n"
        "ACTION: issue_supplier_alert(TURKISH_SUPPLIERS).",
        {"decision": "Issue supplier alert to Turkish suppliers for status check; no financial action.",
         "evidence": ["TRY -3% DoD is within normal FX volatility band",
                      "no operational disruption signals", "supply chain health 95/100"],
         "counterfactual": "No action -> near-zero downside; free-cost information action.",
         "precedent": "2022 Ukraine war caused 80% palladium spike + 250% nickel spike, but lower-magnitude FX moves (<5% daily) rarely cascade to supply chains.",
         "risk_level": "LOW", "confidence": 0.80},
    ),
    (
        "STATE: Day 2 of 30. Active: Iran threatens to close Strait of Hormuz.\n"
        "FRED Brent $123/bbl (DoD +3.5%). 20% of global crude transit at risk. Carriers\n"
        "pausing Persian Gulf bookings. Health 72/100.\n"
        "ACTION: hedge_commodity(oil, $4.2M).",
        {"decision": "Hedge oil exposure $4.2M via near-term futures to cap supply-cost spike.",
         "evidence": ["Iran threatened full Hormuz closure 2026-04-18", "Brent $123/bbl DoD +3.5%",
                      "20% of global crude transits Hormuz"],
         "counterfactual": "Unhedged -> if closure materializes, P50 Brent projection $160-180/bbl, 60-day COGS impact $80-120M.",
         "precedent": "2019 Abqaiq attack cut 5% global crude 2-week; Brent +14% peak. Hormuz closure >10x larger scale.",
         "risk_level": "CRITICAL", "confidence": 0.90},
    ),
    (
        "STATE: Day 20 of 45. Two regional earthquakes M4.8 (Japan Pacific) and\n"
        "M4.5 (Chile) logged; no tsunami advisory; no damage reports. Health 90/100. No\n"
        "active disruption signals in supply chain.\n"
        "ACTION: do_nothing chosen.",
        {"decision": "Monitor only; no mitigation needed.",
         "evidence": ["M4.5-4.8 quakes below damage-inducing threshold (~M6.0)", "no tsunami advisory",
                      "no damage or supplier-offline reports"],
         "counterfactual": "No action -> zero expected loss; acting would waste budget.",
         "precedent": "2011 Tohoku was M9.0 (~1000x energy of M4.8); only M6.5+ quakes have historically disrupted auto/semi supply chains.",
         "risk_level": "LOW", "confidence": 0.95},
    ),
    (
        "STATE: Day 15 of 60. Active: Panama Canal water levels low (-25% below\n"
        "avg). Transit slots cut 30%. Container freight rates Shanghai-East US +18% WoW.\n"
        "Multiple carriers announcing capacity reductions.\n"
        "ACTION: increase_safety_stock(WAREHOUSE_ATLANTA, +14 days).",
        {"decision": "Increase Atlanta warehouse safety stock +14 days to buffer Panama slot reductions.",
         "evidence": ["Panama transit slots -30%", "Shanghai-East US rates +18% WoW",
                      "Panama Canal Authority restricted daily transits"],
         "counterfactual": "No action -> 30% probability of stock-out on Asian imports in 21-28 days, P50 loss $3M.",
         "precedent": "2023 Panama Canal drought cut transits 25% for 9 months, driving Atlantic freight +40%; firms with 2+ week buffer avoided stock-outs.",
         "risk_level": "MEDIUM", "confidence": 0.82},
    ),
    (
        "STATE: Day 28 of 45. Active: Iran True Promise 2 missile barrage at Israel\n"
        "(2024-10-01 analog). Haifa port intermittent closures. Lloyd's war-risk premium\n"
        "+50bp East Med. Health 78/100. TSMC Taiwan unaffected.\n"
        "ACTION: reroute_shipment(HAIFA -> ASHDOD).",
        {"decision": "Reroute Haifa-bound shipments via Ashdod (+3-5d transit).",
         "evidence": ["True Promise 2 active (180 missiles)", "Haifa port intermittent closures",
                      "Lloyd's war-risk premium +50bp"],
         "counterfactual": "No action -> 25% probability of carrier cancellation, P50 delivery delay 7-10 days, SLA penalty $0.5M.",
         "precedent": "2024-10-07 Hezbollah rocket escalation caused multi-week Haifa operational disruption; Ashdod pickup absorbed 80% of diverted volume.",
         "risk_level": "HIGH", "confidence": 0.88},
    ),
]

# ---------------------------------------------------------------------------
# Strict-JSON decision contract enforced by the provider.
# ---------------------------------------------------------------------------
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DECISION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "evidence", "counterfactual", "precedent",
                 "risk_level", "confidence"],
    "properties": {
        "decision": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "counterfactual": {"type": "string"},
        "precedent": {"type": "string"},
        "risk_level": {"type": "string", "enum": RISK_LEVELS},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def build_messages(scenario_text: str) -> list[dict]:
    """Assemble the full chat message list: system + 8 few-shots + the new
    scenario. Few-shot assistant turns are the exemplar JSON, serialized compact
    (matching how the v5 Modelfile stored them)."""
    messages: list[dict] = [{"role": "system", "content": ANALYST_SYSTEM_PROMPT}]
    for user_text, assistant_json in FEW_SHOTS:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant",
                         "content": json.dumps(assistant_json, separators=(", ", ": "))})
    messages.append({"role": "user", "content": scenario_text})
    return messages


# ---------------------------------------------------------------------------
# Holdout scenarios — DISTINCT from the 8 few-shots (no leakage). Gold labels
# are for the WP6.2 Brier A/B (blocked on key); WP6.1 only exercises the runner.
# ---------------------------------------------------------------------------
HOLDOUT_SCENARIOS: list[dict] = [
    {"id": "H1_baltimore_bridge",
     "prompt": "STATE: Day 4 of 30. Active: container ship struck the Francis Scott Key "
               "Bridge in Baltimore; Port of Baltimore vessel traffic fully suspended. "
               "Auto RoRo + coal export terminal affected. Health 80/100.\n"
               "ACTION: reroute_shipment(BALTIMORE -> NORFOLK).",
     "gold_risk_level": "HIGH"},
    {"id": "H2_routine_minor_delay",
     "prompt": "STATE: Day 9 of 30. A tier-3 packaging supplier reports a 1-day trucking "
               "delay due to local road maintenance. All tier-1 suppliers operational. "
               "No commodity moves. Health 96/100.\n"
               "ACTION: do_nothing.",
     "gold_risk_level": "LOW"},
    {"id": "H3_taiwan_strait_drills",
     "prompt": "STATE: Day 6 of 45. PLA announces live-fire exercises in the Taiwan "
               "Strait for 72h; some commercial air/sea routes advised to divert. TSMC "
               "operating normally so far. Health 85/100.\n"
               "ACTION: issue_supplier_alert(TAIWAN_SEMI).",
     "gold_risk_level": "MEDIUM"},
    {"id": "H4_cobalt_spike",
     "prompt": "STATE: Day 11 of 30. DRC export permit freeze; cobalt +14% day-over-day "
               "on the LME. Battery-cell suppliers signal input-cost pressure. Health 83/100.\n"
               "ACTION: pre_buy_inventory(cobalt, 30-day).",
     "gold_risk_level": "HIGH"},
    {"id": "H5_minor_fx_move",
     "prompt": "STATE: Day 14 of 45. Indian Rupee -1.2% overnight on Fed commentary. "
               "All suppliers operational; no shipping disruption. Health 94/100.\n"
               "ACTION: do_nothing.",
     "gold_risk_level": "LOW"},
    {"id": "H6_hormuz_full_closure",
     "prompt": "STATE: Day 1 of 30. Active: Iran announces full closure of the Strait of "
               "Hormuz effective immediately; naval mines reported. Brent $141/bbl (DoD "
               "+12%). All Persian Gulf bookings halted. Health 68/100.\n"
               "ACTION: hedge_commodity(oil, $9M) + activate_backup_supplier.",
     "gold_risk_level": "CRITICAL"},
]


@dataclass
class Analyst:
    """SupplyMind Analyst v5 over the provider layer."""
    provider: LLMProvider | None = None
    model: str | None = None

    def _provider(self) -> LLMProvider:
        return self.provider if self.provider is not None else get_provider()

    async def assess(self, scenario_text: str, *, retries: int = 2) -> LLMResponse:
        """Return a strict-JSON risk decision for one scenario. On no-key /
        invalid-output the response is ``degraded`` with the reason surfaced —
        never a fabricated verdict."""
        return await self._provider().complete(
            build_messages(scenario_text),
            model=self.model,
            schema=DECISION_SCHEMA,
            max_tokens=ANALYST_PARAMS["num_predict"],
            temperature=ANALYST_PARAMS["temperature"],
            retries=retries,
        )


# ---------------------------------------------------------------------------
# Offline structural self-check (no network) — used by the receipt.
# ---------------------------------------------------------------------------
def structural_report() -> dict:
    """Machine-checkable facts about the port fidelity (no LLM call)."""
    prompt_keys_declared = all(
        k in ANALYST_SYSTEM_PROMPT
        for k in ("decision", "evidence", "counterfactual", "precedent",
                  "risk_level", "confidence")
    )
    fewshot_levels = [a["risk_level"] for _, a in FEW_SHOTS]
    return {
        "few_shot_count": len(FEW_SHOTS),
        "few_shots_all_valid_json_keys": all(
            set(a.keys()) == set(DECISION_SCHEMA["required"]) for _, a in FEW_SHOTS
        ),
        "few_shot_risk_levels": fewshot_levels,
        "few_shot_level_coverage": sorted(set(fewshot_levels)),
        "schema_required_keys": DECISION_SCHEMA["required"],
        "system_prompt_chars": len(ANALYST_SYSTEM_PROMPT),
        "system_prompt_declares_all_output_keys": prompt_keys_declared,
        "decoding_params": ANALYST_PARAMS,
        "message_turns_for_one_scenario": len(build_messages("PROBE")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_validate() -> int:
    report = structural_report()
    print(json.dumps(report, indent=2))
    ok = (
        report["few_shot_count"] == 8
        and report["few_shots_all_valid_json_keys"]
        and report["system_prompt_declares_all_output_keys"]
        and set(report["few_shot_level_coverage"]) == set(RISK_LEVELS)
    )
    print(f"\n[validate] structural port {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


async def _cmd_holdout(limit: int | None) -> int:
    provider = get_provider()
    live = await provider.liveness()
    if not live.ok:
        # LOUD, honest exit — no fabricated verdicts (CLAUDE.md §0).
        print(json.dumps({
            "status": "BLOCKED",
            "provider": live.provider,
            "reason": live.reason,
            "note": "Analyst --holdout needs a live provider. Build/tests are "
                    "complete; live A/B Brier (WP6.2) runs when the key/daemon "
                    "is available. No verdicts fabricated.",
        }, indent=2))
        return 2

    analyst = Analyst(provider=provider)
    scenarios = HOLDOUT_SCENARIOS[:limit] if limit else HOLDOUT_SCENARIOS
    results = []
    for sc in scenarios:
        resp = await analyst.assess(sc["prompt"])
        results.append({
            "id": sc["id"], "gold_risk_level": sc["gold_risk_level"],
            "ok": resp.ok, "degraded": resp.degraded,
            "degrade_reason": resp.degrade_reason,
            "predicted": resp.parsed, "attempts": resp.attempts,
            "latency_s": resp.latency_s,
        })
        status = "ok" if resp.ok else f"DEGRADED({resp.degrade_reason[:60]})"
        print(f"  {sc['id']}: {status}")
    print(json.dumps({"provider": provider.name, "n": len(results),
                      "results": results}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SupplyMind Analyst v5")
    p.add_argument("--validate", action="store_true",
                   help="offline structural check of the prompt port (no network)")
    p.add_argument("--holdout", action="store_true",
                   help="run the analyst on holdout scenarios (needs a live provider)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of holdout scenarios")
    args = p.parse_args(argv)

    if args.validate:
        return _cmd_validate()
    if args.holdout:
        return asyncio.run(_cmd_holdout(args.limit))
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
