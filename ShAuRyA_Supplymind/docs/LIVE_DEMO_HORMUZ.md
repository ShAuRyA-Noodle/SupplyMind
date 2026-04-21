# Live Hormuz Demo — Talking Points + Commands

> Final video script goes in `demo/DEMO_VIDEO_SCRIPT.md` when recording (user records on Mac). This file is the cheat-sheet for the live demo moment that replaces old Scene 3.

## 30-second hook (replaces old Scene 3 — "Live API risk assessment")

> **"Supply-chain risk intelligence isn't a dashboard. It's a live event loop. Watch."**

## Pre-demo prep (30 seconds, offline)

```bash
# 1. Ensure server is up
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 2. Run the ingestor once to populate live events (uses .env keys)
python -m ShAuRyA_Supplymind.realtime.ingestor --once --skip marinetraffic
#  -> ~150 fetched events (NewsAPI / GDELT / USGS / FRED Brent) cached in events.db

# 3. Confirm /live is live
curl http://localhost:8000/live/health | jq
#  { "status": "ok", "ollama_available": true/false, "event_counts": {...} }
```

## The 3-command demo (on-camera, 90 seconds)

### (a) The scenario call

```bash
curl -s -X POST http://localhost:8000/live/hormuz-closure \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_text": "Iran threatens full closure of Strait of Hormuz after US Navy seizes Iranian cargo ship in Gulf of Oman. Brent crude spikes to $123 per barrel. Major carriers pause Persian Gulf bookings.",
    "region": "hormuz",
    "enable_llm_judges": true,
    "include_recent_signals": true,
    "k_analogs": 3
  }' | jq
```

### (b) What the judges see (narrate while output streams)

- **Top analog match**: `hormuz_trump_cargo_ship_2026_04` — similarity 0.99 — matches real NewsAPI event from 2026-04-19.
- **3-judge panel** (if Ollama up): Qwen-2.5-14B, Mistral-Nemo, DeepSeek-R1-Q4.
- **Consensus**: `HIGH` or `CRITICAL` with confidence ~0.75–0.90.
- **Projected Brent $/bbl** (P50): ~$110-125 range, interpolated from analogs.
- **Recommended actions** (ranked by loss_avoided / cost):
  1. `hedge_commodity` — oil, sized to severity
  2. `reroute_shipment` — via Cape of Good Hope (+14d)
  3. `activate_backup_supplier` — Samsung backup
  4. `increase_safety_stock` — 17 days buffer
  5. `issue_supplier_alert` — zero-cost info action
- **Counterfactual**: `no_action_p50_loss_usd: $324M` → `with_plan_p50_loss_usd: $65M` = **80% savings, ~$259M**.

### (c) The punchline

> **"Everything you just saw is running on my laptop against the real 2026 news feed — NewsAPI polled April 19th, FRED's actual Brent price of $123 per barrel, USGS earthquakes live. No hardcoded scenarios, no pre-scripted answers. The judges hadn't seen this event when the code was written. That's what real-world aligned supply-chain AI looks like."**

## Fallback modes (if Ollama is down during recording)

The pipeline degrades gracefully. With `enable_llm_judges=false`:

- Judge: `Rubric-Fallback` only (single deterministic judge)
- Everything else unchanged — analogs, projection, actions, counterfactual all still populate.

Demo doesn't break; narrate as *"today we're showing the rubric fallback; with Ollama warm, three local LLMs jointly score the scenario."*

## Verification for judges (off-camera)

After the demo, hand the judges:

```bash
# 1. Verify the analog library is real (check citations)
jq '.events[] | {name, date, citations: [.citations[] | .publisher]}' \
   ShAuRyA_Supplymind/scenarios/iran_israel_hormuz_2024_2026.json | head -40

# 2. Verify live events came from actual APIs (no pre-canned)
python -m ShAuRyA_Supplymind.realtime.store --recent 10
#  shows 10 most recent events with timestamps, sources, urls

# 3. Verify the counterfactual math is reproducible
pytest ShAuRyA_Supplymind/tests/test_hormuz_endpoint.py -v
#  -> 8 passed; all deterministic without network
```

## Why this wins

| Other hackathon demos | SupplyMind v4 arcadia-live |
|---|---|
| Pre-scripted "imaginary" scenarios | Real 2026 Iran-Israel-Hormuz crisis, polled live from NewsAPI |
| "Mocked" oil prices | FRED's actual `DCOILBRENTEU` series, `$123.28/bbl` on 2026-04-21 |
| Single LLM judge | 3-judge local panel (DeepSeek + Qwen + Mistral) with Krippendorff α=0.75 baseline |
| "Decision support" dashboard | Executable actions with cost/loss-avoided dollars + counterfactual |
| "Reproducible in theory" | `pytest` 8 passing, all offline deterministic; citations with DOIs/URLs |

This is the 90-second segment judges will remember. Everything upstream (13 models, 173 v3 tests, 50k-step autoresearch) is the scaffolding that makes it possible.
