"""DEPRECATED shim — use the ``client/`` package instead.

Historically there were two divergent HTTP clients: this root ``client.py``
module and the ``client/`` package. They had different constructor signatures
and a different (broken) ``step()`` payload. The ``client/`` package is now the
single canonical client.

Note: because a package (``client/``) shadows a same-named module (``client.py``)
on ``sys.path``, ``import client`` already resolves to the package — this file
is unreachable via a normal import and exists only to document the move and to
keep any direct file-path import working. Prefer:

    from client import SupplyMindClient   # -> client/supplymind_client.py

This shim re-exports the canonical client so no code path can pick up a stale,
divergent implementation.
"""
from __future__ import annotations

import warnings

from client import SupplyMindClient  # canonical: client/ package

warnings.warn(
    "Importing the root client.py module is deprecated; the canonical client is "
    "the `client/` package. `from client import SupplyMindClient` already "
    "resolves to it.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SupplyMindClient"]
