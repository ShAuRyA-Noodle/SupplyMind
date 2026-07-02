# DEPRECATED COMPATIBILITY SHIM (P0.3 package reorg).
# The OpenRouter client was promoted to `supplymind/llm/client.py` (CLAUDE.md
# §0.4 "one AI gateway"). Import from the new path:
#     from supplymind.llm.client import OpenRouterClient, MODELS, ModelSpec
# This shim re-exports everything so any un-migrated importer keeps working.
from supplymind.llm.client import *  # noqa: F401,F403
from supplymind.llm.client import OpenRouterClient, MODELS, ModelSpec  # noqa: F401
