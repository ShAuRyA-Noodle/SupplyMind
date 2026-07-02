"""Secure checkpoint loading — CVE-2025-32434 mitigation for SupplyMind.

Background
----------
``torch.load`` on torch < 2.6 is a remote-code-execution surface: the default
unpickler runs arbitrary ``__reduce__`` code, and CVE-2025-32434 shows the RCE
fires *even with* ``weights_only=True`` on torch <= 2.5.1. The repo pins
``torch>=2.6.0`` (see requirements-rl.txt / pyproject.toml); upgrading the
running venv is an owner action (do not pip-upgrade mid-work).

Defense in depth applied here
-----------------------------
1. Every load is attempted with ``weights_only=True`` first — the restricted
   unpickler that refuses arbitrary reduce ops.
2. A minority of the repo's committed checkpoints (the QR-DQN family and the
   TFT family) embed benign NumPy objects (``numpy.core.multiarray.scalar`` /
   ``_reconstruct``) that were pickled under NumPy-1.x module paths. On this
   machine (torch 2.5.1 + numpy 2.x) those globals cannot be allow-listed for
   ``weights_only=True`` because ``add_safe_globals`` keys off the live
   ``__module__`` (``numpy._core.*``) which no longer matches the pickled name.
   For exactly those files we fall back to ``weights_only=False`` **only after**
   the file's SHA-256 matches a trusted hash recorded below. A tampered or
   unrecognized checkpoint is refused loudly — never silently unpickled.

Recompute the manifest after (re)training a gated checkpoint::

    python -m rl.checkpoint_security --hash rl/checkpoints/qrdqn_best_easy.pt
    python -m rl.checkpoint_security --verify   # re-hash every manifest entry
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parent.parent
_CK = Path(__file__).resolve().parent / "checkpoints"

# SHA-256 of trusted, repo-committed checkpoints that FAIL ``weights_only=True``
# on torch 2.5.1 + numpy 2.x (they embed numpy scalars/arrays). Keyed by file
# basename. Regenerate with ``python -m rl.checkpoint_security --hash <file>``.
TRUSTED_SHA256: dict[str, str] = {
    "qrdqn_best_easy.pt": "b19f489f0648cc2d682c2155be9bcf00abcbb25cb0aef219bc5ce488c2dfab95",
    "qrdqn_best_hard.pt": "a65ccb6b23de1493cebc5d348d8d870610bc7e18c743d3b015ea894c8b0249a7",
    "qrdqn_best_medium.pt": "91833fec8eefac77de2a4a5d7d3311a1158b12ca6e8b3641def2cde4367f2666",
    "qrdqn_final_easy.pt": "f45b7cea843a8aa70a35fe3219239b1dbee80508f47d28fc47fc92be168d5215",
    "qrdqn_final_hard.pt": "401a6b4f8f5c39ca6ee620e42b4457ed7d752e80652b4350aa9dc22aed0e966b",
    "qrdqn_final_medium.pt": "c0ffe3f178b03412d57bdf47ec39e37fa1be6a7819b73991d4c1bd127b241a38",
    "qrdqn_v2_easy.pt": "b19f489f0648cc2d682c2155be9bcf00abcbb25cb0aef219bc5ce488c2dfab95",
    "qrdqn_v2_hard.pt": "a65ccb6b23de1493cebc5d348d8d870610bc7e18c743d3b015ea894c8b0249a7",
    "qrdqn_v2_medium.pt": "91833fec8eefac77de2a4a5d7d3311a1158b12ca6e8b3641def2cde4367f2666",
    "tft_real.pt": "e68c6b4edeb9a95b5ad779fc6efaec8836562cbbf5569110df728c7a054c611f",
    "tft_v2.pt": "9e10e8f16a55e9e96765a551497936465c99bd7f051ee3d7bfe1afcba07a4717",
}

# SHA-256 of trusted repo-committed pickles loaded outside torch (sklearn etc.).
TRUSTED_PICKLE_SHA256: dict[str, str] = {
    "financial_impact_ridge.pkl": "bbf51554fd0e53147fbc13e216b3ab1d3be1a9da0280b8c7aca720627d931a81",
}

_WEIGHTS_ONLY_MARKERS = ("Weights only load failed", "WeightsUnpickler", "Unsupported global")


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_weights_only_restriction(err: Exception) -> bool:
    msg = str(err)
    return any(m in msg for m in _WEIGHTS_ONLY_MARKERS)


def safe_load(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a torch checkpoint, refusing arbitrary code execution.

    Tries ``weights_only=True``. If (and only if) the restricted unpickler
    rejects a benign global, the file's SHA-256 is checked against
    ``TRUSTED_SHA256``; on an exact match the load is retried with
    ``weights_only=False``. An unrecognized name or a hash mismatch raises
    ``RuntimeError`` — a tampered checkpoint fails loud, never loads.
    """
    path = Path(path)
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as err:  # noqa: BLE001 - narrowed immediately below
        if not _is_weights_only_restriction(err):
            raise
        name = path.name
        expected = TRUSTED_SHA256.get(name)
        actual = sha256_file(path)
        if expected is None:
            raise RuntimeError(
                f"{name}: requires weights_only=False (embeds non-tensor globals) "
                f"but is not in the trusted SHA-256 manifest; refusing to unpickle. "
                f"If this file is trusted, add its hash ({actual}) to "
                f"rl/checkpoint_security.TRUSTED_SHA256."
            ) from err
        if actual != expected:
            raise RuntimeError(
                f"{name}: SHA-256 mismatch — checkpoint tampered or regenerated. "
                f"expected {expected}, got {actual}. Refusing weights_only=False load. "
                f"If you retrained it, update rl/checkpoint_security.TRUSTED_SHA256."
            ) from err
        # Provenance verified against recorded hash -> trusted-local unpickle.
        return torch.load(str(path), map_location=map_location, weights_only=False)


def safe_pickle_load(path: str | Path) -> Any:
    """Unpickle a repo-committed artifact only after a SHA-256 integrity check.

    ``pickle.load`` is unconditional arbitrary-code-execution on load, so this
    gate refuses any file whose hash is not on record (tampering / unknown
    provenance). Keyed by basename against ``TRUSTED_PICKLE_SHA256``.
    """
    path = Path(path)
    name = path.name
    expected = TRUSTED_PICKLE_SHA256.get(name)
    actual = sha256_file(path)
    if expected is None:
        raise RuntimeError(
            f"{name}: not in trusted pickle manifest; refusing to unpickle "
            f"(arbitrary-code-execution risk). Add its hash ({actual}) to "
            f"rl/checkpoint_security.TRUSTED_PICKLE_SHA256 if trusted."
        )
    if actual != expected:
        raise RuntimeError(
            f"{name}: SHA-256 mismatch — pickle tampered or regenerated. "
            f"expected {expected}, got {actual}. Refusing to unpickle."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Checkpoint integrity manifest tool.")
    ap.add_argument("--hash", metavar="FILE", help="print SHA-256 of a file")
    ap.add_argument("--verify", action="store_true", help="re-hash every manifest entry")
    args = ap.parse_args()

    if args.hash:
        print(sha256_file(args.hash))
        return
    if args.verify:
        ok = True
        for name, expected in {**TRUSTED_SHA256, **TRUSTED_PICKLE_SHA256}.items():
            p = _CK / name if name.endswith(".pt") else _ROOT / "rl" / "analysis" / "trained" / name
            if not p.exists():
                print(f"MISSING  {name}")
                continue
            actual = sha256_file(p)
            status = "OK    " if actual == expected else "CHANGED"
            ok = ok and actual == expected
            print(f"{status} {name} {actual}")
        raise SystemExit(0 if ok else 1)
    ap.print_help()


if __name__ == "__main__":
    _cli()
