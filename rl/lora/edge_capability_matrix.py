"""edge_capability_matrix.py — the honest cloud/edge/local capability matrix.

This module is the machine-readable record of the Ollama -> OpenRouter migration
(CLAUDE.md P1.1, §7.1). Every LLM-touching feature now routes through the single
gateway ``supplymind.llm.providers`` — one interface, two chat backends
(``openrouter`` default, ``ollama`` edge) — plus a small set of on-device
non-LLM assets (embeddings, vision heuristic) that have no OpenRouter route and
are a *sanctioned local exception* (§7.3).

The strategic point (owner CREATIVE MANDATE): the same interface serving cloud
frontier models AND local edge models is genuinely valuable — offline-first
resilience. This matrix states, per feature and WITHOUT euphemism, what actually
works in each mode and how it degrades when nothing is available. There is no
faking: a feature that cannot run in a mode says so, and a feature that degrades
names the honest fallback.

Run:  python -m rl.lora.edge_capability_matrix

Modes:
  cloud   OpenRouter frontier/open models via the gateway. Requires
          OPENROUTER_API_KEY (currently REVOKED -> "blocked-on-key").
  edge    Local Ollama daemon via the gateway's ``ollama`` backend. Requires a
          running daemon + a pulled model. Never a hidden hard dependency.
  local   On-device, NO daemon and NO key: SentenceTransformer embeddings and
          the deterministic image heuristic. First-class edge assets.
  degrade The truthful behavior when neither cloud nor edge is available: a
          loud failure OR a clearly-labeled rule/heuristic fallback — never a
          fabricated LLM verdict (CLAUDE.md §0).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Awaitable, TypeVar

_T = TypeVar("_T")


def run_coro_sync(coro: Awaitable[_T]) -> _T:
    """Run an async gateway coroutine from synchronous code.

    Safe whether or not an event loop is already running (mirrors the pattern in
    ``supplymind.llm.panel.JudgePanel.run_sync``). Sync consumers of the async
    provider layer (explainer, dual-verifier, autoresearch, vision) use this so
    the migration does not force every caller to become async.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The matrix. Each row is one migrated feature. "cloud"/"edge"/"local" are one
# of: "yes", "blocked-on-key", "no". "degrade" is the truthful fallback string.
# ---------------------------------------------------------------------------
CAPABILITY_MATRIX: list[dict[str, Any]] = [
    {
        "feature": "analyst_decision",
        "module": "supplymind.llm.analyst",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "loud: LLMResponse(degraded=True, reason surfaced) - no verdict",
        "note": "Golden-path step 2. Owned by supplymind/llm; listed for completeness.",
    },
    {
        "feature": "judge_panel",
        "module": "supplymind.llm.panel",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "loud: failed/unparseable judges counted separately, excluded from alpha",
        "note": "Owned by supplymind/llm; listed for completeness.",
    },
    {
        "feature": "rl_action_explainer",
        "module": "rl.explainer",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "loud: raises ExplainerError with the provider's surfaced reason",
        "note": "Was mandatory-Ollama (import ollama). Now gateway-routed; Ollama is edge mode only.",
    },
    {
        "feature": "crisis_rag_embeddings",
        "module": "rl.rag.indexer",
        "kind": "embedding",
        "cloud": "no",
        "edge": "no",
        "local": "yes",
        "degrade": "loud: raises RAGError if the local mxbai model cannot load",
        "note": "SANCTIONED local exception (§7.3). Was Ollama nomic-embed-text (daemon "
                "dependency); now local mxbai SentenceTransformer (no daemon). R9 proved "
                "mxbai-alone is the best retriever. Deliberately NOT routed to OpenRouter.",
    },
    {
        "feature": "autoresearch_hypothesis_engine",
        "module": "versions.v4_arcadia_live.autoresearch.hypothesis_engine "
                  "(+ versions.v5_phoenix.autoresearch_fixed.hypothesis_engine)",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "loud: raises RuntimeError after retries; orchestrator backs off",
        "note": "Was direct Ollama HTTP + direct Anthropic HTTP. Now gateway-routed: "
                "agent='ollama' -> edge backend, agent='cloud'/'openrouter' -> cloud.",
    },
    {
        "feature": "wordle_dual_verifier_model_layer",
        "module": "supplymind.phoenix.wordle_env.dual_verifier",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "graceful: model_score=None -> composite uses rule score only, "
                   "disagreement=None (honest; the model layer is optional by design)",
        "note": "Was direct Ollama HTTP. Now gateway-routed. Rule layer is always local.",
    },
    {
        "feature": "port_imagery_vision",
        "module": "supplymind.warroom.features.qwen_vl_port_imagery",
        "kind": "vision",
        "cloud": "blocked-on-key",
        "edge": "no",
        "local": "yes",
        "degrade": "heuristic: deterministic PIL color/blob statistics, mode='heuristic', "
                   "confidence<=0.35, honestly labeled - never a fabricated VLM reading",
        "note": "Was direct Ollama qwen-vl HTTP. Now OpenRouter vision via the gateway "
                "(image_url data-URI). The gateway's ollama backend does not carry images, "
                "so edge vision is intentionally unsupported; the local heuristic covers offline.",
    },
    {
        "feature": "ollama_model_authoring",
        "module": "rl.lora.create_ollama_model",
        "kind": "tooling",
        "cloud": "n/a",
        "edge": "yes",
        "local": "n/a",
        "degrade": "n/a: edge-model build tooling, not an inference path",
        "note": "Uses the `ollama create` CLI to register edge Modelfiles — this is the "
                "explicit edge-model authoring step, not a hidden inference dependency. "
                "The post-build smoke test routes through the gateway edge backend.",
    },
    {
        "feature": "v5_vs_frontier_benchmark",
        "module": "scripts.ollama_v5_vs_frontier",
        "kind": "chat",
        "cloud": "blocked-on-key",
        "edge": "yes",
        "local": "no",
        "degrade": "honest skip: the local v5 leg is skipped with a surfaced reason when the "
                   "daemon/model is absent; a frontier-only benchmark is still produced",
        "note": "Offline benchmark (not a production path). Both legs now route through the "
                "gateway: local leg via provider='ollama', frontier via provider='openrouter'.",
    },
]


def format_capability_matrix() -> str:
    """Render the matrix as a fixed-width table for the CLI / receipts."""
    cols = ["feature", "kind", "cloud", "edge", "local"]
    widths = {c: max(len(c), *(len(str(row[c])) for row in CAPABILITY_MATRIX)) for c in cols}
    header = "  ".join(c.upper().ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for row in CAPABILITY_MATRIX:
        lines.append("  ".join(str(row[c]).ljust(widths[c]) for c in cols))
        lines.append(f"    degrade: {row['degrade']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("SupplyMind LLM capability matrix (cloud=OpenRouter | edge=Ollama | local=on-device)\n")
    print(format_capability_matrix())
    print(
        "\nKey status: OPENROUTER_API_KEY is REVOKED -> every 'cloud' cell reads "
        "'blocked-on-key'.\nThe code paths are complete and offline-tested; live cloud "
        "measurement runs the moment a valid key lands. Nothing is faked around the missing key."
    )
