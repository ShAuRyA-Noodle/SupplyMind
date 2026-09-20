"""trust_router.py — FastAPI APIRouter for the SupplyMind Trust Layer.

Exposes the Trust Graph (:mod:`supplymind.warroom.trust`) and the historical
replay proof (:mod:`supplymind.warroom.replay_proof`) as real HTTP endpoints the
war-room UI calls when Priya clicks a number and asks "how do you know that?".

Routes (mount under prefix "/trust"):
  GET /trust/metrics                      list every registered metric id
  GET /trust/resolve/{metric_id}          full provenance chain for one number
  GET /trust/graph                        every metric resolved into one graph
  GET /trust/replay-proof                 "what we'd have said vs what happened"
  GET /trust/replay-proof/{event_id}      one event's honest replay
  GET /trust/health                       subsystem availability

This module is NOT mounted here — server/app.py is owned by another agent. The
orchestrator should wire it with::

    from supplymind.warroom.trust_router import router as trust_router
    app.include_router(trust_router, prefix="/trust", tags=["trust"])

Everything returned is real-or-honestly-degraded. A metric with no known origin
resolves to ``status="unknown_provenance"``; the resolver never fabricates a
chain (CLAUDE.md §0).
"""
from __future__ import annotations

import logging
from pathlib import Path

try:
    from fastapi import APIRouter, HTTPException, Query
except ImportError:  # allow import without fastapi (unit tests of the logic)
    APIRouter = None  # type: ignore
    HTTPException = Exception  # type: ignore
    Query = lambda default=None, **kw: default  # type: ignore  # noqa: E731

from supplymind.warroom import replay_proof as rp
from supplymind.warroom import trust as trust_mod

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY_RECEIPT = REPO_ROOT / "tests" / "receipts" / "replay_proof_REAL.json"

router = APIRouter() if APIRouter is not None else None


if router is not None:

    @router.get("/health", tags=["trust"])
    def trust_health() -> dict:
        metrics = trust_mod.list_metrics()
        by_kind: dict[str, int] = {}
        for m in metrics:
            by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
        return {
            "status": "ok",
            "n_metrics": len(metrics),
            "by_kind": by_kind,
            "replay_receipt_present": REPLAY_RECEIPT.exists(),
            "note": ("Every rendered number resolves to its provenance chain; "
                     "unknown-provenance numbers are surfaced loudly, never "
                     "fabricated."),
        }

    @router.get("/metrics", tags=["trust"])
    def trust_metrics() -> dict:
        """List every registered metric id + label + kind (no resolution)."""
        metrics = trust_mod.list_metrics()
        return {"count": len(metrics), "metrics": metrics}

    @router.get("/resolve/{metric_id:path}", tags=["trust"])
    def trust_resolve(
        metric_id: str,
        live: bool = Query(False, description="Force a live fetch of source "
                                              "leaves (ignore cache TTL)."),
        depth: int = Query(3, ge=0, le=6,
                           description="Max derivation recursion depth."),
    ) -> dict:
        """Resolve one metric id to its full provenance chain.

        Live leaves carry a real content sha256, raw source URL, and fetch age;
        derived numbers recurse into their upstream metrics; a number with no
        known origin returns status='unknown_provenance' (never a fake chain).
        This endpoint always returns 200 — an unknown metric is a real, honest
        answer, not an error.
        """
        return trust_mod.resolve(metric_id, depth=depth, force_live=live)

    @router.get("/graph", tags=["trust"])
    def trust_graph(
        live: bool = Query(False, description="Force live fetch of source leaves."),
    ) -> dict:
        """Resolve every registered metric into one graph (nodes + edges)."""
        return trust_mod.build_graph(force_live=live)

    @router.get("/replay-proof", tags=["trust"])
    def replay_proof(
        fresh: bool = Query(False, description="Recompute a single featured "
                                               "replay instead of reading the "
                                               "committed sweep receipt."),
        featured: str = Query("iran_true_promise_2_2024_10",
                              description="Featured event id when recomputing."),
    ) -> dict:
        """Return the historical replay proof.

        By default serves the committed full-sweep receipt (fast, reproducible).
        With ``fresh=1`` recomputes one featured event live so the demo can prove
        the pipeline runs end-to-end. The honest verdict — including any MISS —
        is returned as-is; nothing is tuned.
        """
        if not fresh and REPLAY_RECEIPT.exists():
            import json
            try:
                receipt = json.loads(REPLAY_RECEIPT.read_text(encoding="utf-8"))
                receipt["_served_from"] = "committed_receipt"
                return receipt
            except Exception as e:  # noqa: BLE001 — fall through to fresh compute
                logger.warning("[trust] replay receipt unreadable: %s", e)
        try:
            r = rp.run_replay(featured)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"replay failed: {type(e).__name__}: {e}")
        return {"_served_from": "fresh_single_event", "featured_event": featured,
                "replay": r}

    @router.get("/replay-proof/{event_id}", tags=["trust"])
    def replay_proof_one(event_id: str) -> dict:
        """Honest replay of one documented crisis, computed live."""
        try:
            return rp.run_replay(event_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"replay failed: {type(e).__name__}: {e}")
