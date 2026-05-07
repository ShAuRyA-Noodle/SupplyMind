"""SupplyMind War Room — Hormuz India + Gulf Supply Chain endpoint.

Public surface:

    POST /demo/hormuz-india-war-room

Inputs:    scenario_text, severity, brent_price (optional), country_focus,
            include_gulf, horizon_days, enable_live_signals, enable_llm_judges
Outputs:   layered response (5 scenes) — what happened, why it matters,
            who is exposed, what breaks first, what to do — with a per-field
            _evidence drawer pointing at the curated atlases or live signal
            sources, and a top-level receipt with command + hash + timestamp.

Design rule: NO model-generated numbers in the user-facing fields without
the literal label `model_estimate` and a derivation note. Primary-source
facts come from `versions/v5_phoenix/scenarios/hormuz_chokepoint_atlas.json`,
`india_supply_chain_exposure.json`, and `gulf_supply_chain_exposure.json`.

Modules:
    atlas_loader.py     Load + validate the three curated JSONs.
    ranker.py           Severity-modulated ranker over the curated sectors.
    provenance.py       Build per-field _evidence drawers.
    live_signals.py     Aggregator over versions/v4_arcadia_live/realtime/* sources.
    router.py           FastAPI router mounted at /demo by phoenix_app.py.
"""
