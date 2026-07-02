"""supplymind — single installable package for the SupplyMind system.

Promoted from the pre-P0.3 layout (root ``models.py``, ``scripts/openrouter_client.py``,
and the live modules under ``versions/``). See CLAUDE.md §3.1 for the target layout.

Subpackages:
  - ``supplymind.contracts``  OpenEnv Pydantic contracts (was root ``models.py``)
  - ``supplymind.llm``        OpenRouter client (was ``scripts/openrouter_client.py``)
  - ``supplymind.warroom``    live geopolitical war-room (promoted v4 realtime + scenarios + features)
  - ``supplymind.phoenix``    v5 live modules (arena, counterfactual_twin/_v2, wordle_env, action_v2, realtime_v5, forecast_v2)
"""
