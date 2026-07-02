# DEPRECATED COMPATIBILITY SHIM (P0.3 package reorg).
# The OpenEnv contracts moved to `supplymind/contracts.py` to kill the
# `models.py`-vs-`models/` (weights dir) collision. Import from the new path:
#     from supplymind.contracts import SupplyMindAction, SupplyMindObservation
# This shim re-exports everything so any un-migrated importer keeps working.
# It will be removed once no `from models import ...` sites remain.
from supplymind.contracts import *  # noqa: F401,F403
