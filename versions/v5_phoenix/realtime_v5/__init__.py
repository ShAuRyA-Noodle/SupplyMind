"""realtime_v5 — Phoenix extension of v4's live pipeline.

We do NOT edit v4's versions/v4_arcadia_live/realtime/ — that's frozen.
Instead this subpackage adds:

    replay_adapter.py     Middleware that wraps v4 Hormuz router and adds
                          ?replay=1 / FORCE_REPLAY=1 fallback to a frozen cache.
                          Enables the "live -> replay -> video" demo recovery
                          protocol from the live-demo-orchestrator skill.

    freeze_cache.py       Build a frozen replay cache from the crisis library
                          (works offline) or from a live ingestor run.
"""
