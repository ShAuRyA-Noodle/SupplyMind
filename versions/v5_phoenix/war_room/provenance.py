"""provenance.py — build _evidence drawers attached to every War Room field.

Design: every leaf-level numeric or factual claim in the response carries
an _evidence dict alongside it. This is the "click the receipt" feature.

We do NOT decorate model-generated values with primary-source citations
(that would be misleading); model-generated values get an _evidence with
source_type='model_estimate' and a derivation note instead.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Evidence:
    source_type: str   # "primary" | "secondary" | "model_estimate" | "live_api" | "internal_artifact"
    publisher: str | None = None
    title: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    derivation: str | None = None    # only for model_estimate
    artifact_path: str | None = None   # for internal_artifact (e.g. v3 result JSON)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"source_type": self.source_type}
        for k in ("publisher", "title", "url", "retrieved_at", "derivation", "artifact_path"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def primary(publisher: str, title: str, url: str) -> Evidence:
    return Evidence(source_type="primary", publisher=publisher, title=title, url=url)


def secondary(publisher: str, title: str, url: str) -> Evidence:
    return Evidence(source_type="secondary", publisher=publisher, title=title, url=url)


def model_estimate(derivation: str) -> Evidence:
    return Evidence(source_type="model_estimate", derivation=derivation,
                    retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def live_api(publisher: str, url: str) -> Evidence:
    return Evidence(source_type="live_api", publisher=publisher, url=url,
                    retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def internal_artifact(path: str, derivation: str) -> Evidence:
    return Evidence(source_type="internal_artifact", artifact_path=path,
                    derivation=derivation)


def value_with_evidence(value: Any, evidence: Evidence) -> dict:
    """Wrap a leaf value with its evidence drawer."""
    return {"value": value, "_evidence": evidence.to_dict()}


def evidence_from_atlas_fact(fact: dict) -> Evidence:
    """Extract Evidence from a row in the curated atlas JSONs."""
    st = fact.get("source_type", "secondary")
    return Evidence(
        source_type=st,
        publisher=fact.get("publisher"),
        title=fact.get("title"),
        url=fact.get("url"),
    )


def hash_payload(payload: dict) -> str:
    """Deterministic hash of the response payload for tamper-detection."""
    import json as _json
    canon = _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def build_receipt(payload: dict, command: str, runtime_s: float) -> dict:
    """Top-level receipt for the entire War Room response.

    The receipt is intentionally machine-friendly: a hash of the payload + the
    command that produced it + the timestamp + the framework version, in the
    same shape as receipts_v2/framework.py so judges can pipe both through the
    same verifier.
    """
    return {
        "framework": "phoenix_war_room_receipt_v1",
        "command": command,
        "payload_sha256": hash_payload(payload),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(runtime_s, 2),
        "verifier_note": "Recompute by hashing the response with this exact key order; should match payload_sha256.",
    }
