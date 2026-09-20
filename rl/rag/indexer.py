"""
RAG crisis documentation retrieval — local mxbai embeddings, NO fallback.

Embeddings are a SANCTIONED local exception (CLAUDE.md §7.3): they have no
OpenRouter route and are deliberately NOT sent to the cloud. This retriever runs
fully on-device via a SentenceTransformer bi-encoder — a first-class edge asset,
not an accident. R9 measured mxbai-alone as the best retriever in this repo
(arctic ensemble -0.076 P@1 RETIRED, BGE rerank -0.038 RETIRED), so mxbai is the
canonical single embedder here.

This replaces the previous Ollama ``nomic-embed-text`` path: routing embeddings
through the Ollama daemon made the RAG a hidden hard dependency on a running
daemon. The SentenceTransformer path needs no daemon and no key. Provider chat
calls elsewhere still go through ``supplymind.llm.providers``; embeddings are the
explicit on-device exception (see ``rl.lora.edge_capability_matrix``).

Production design:
  - Embeddings: local mxbai bi-encoder (1024-d, normalized, cosine), no daemon
  - Storage: ChromaDB persistent client at rl/rag/chroma_db/
  - Retrieval: cosine similarity, min score threshold, or raise

Switching the embedder changes the vector dimension (768 nomic -> 1024 mxbai),
so the collection name is bumped; rebuild the corpus once via
``python -m rl.rag.build_corpus`` (the stale 768-d collection is left untouched).

Legacy hardcoded-precedent path preserved in:
  rl/legacy/fallbacks/rag_indexer_with_fallback.py  (tests/comparison only)

Usage:
    from rl.rag.indexer import CrisisRAG
    rag = CrisisRAG()
    rag.index_text("Supply chain report text...", source="McKinsey 2020")
    results = rag.retrieve_precedents("TSMC disruption Taiwan earthquake")
    if not results: raise RAGError("no precedent above threshold")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_DB_PATH = REPO_ROOT / "rl" / "rag" / "chroma_db"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
# Canonical on-device copy (§7.3): a complete SentenceTransformer directory. When
# present it is used directly so the retriever needs no network at all — the HF
# hub id is only a fresh-clone fallback.
LOCAL_MXBAI_DIR = REPO_ROOT / "models" / "mxbai-embed-large"
EMBED_DIM = 1024
CHUNK_SIZE_WORDS = 300
COLLECTION_NAME = "crisis_docs_v3_mxbai"
MIN_SCORE = 0.60  # cosine similarity threshold for "valid precedent"

_EMBEDDER = None  # lazily-loaded SentenceTransformer singleton (per model name)
_EMBEDDER_NAME: str | None = None


def _resolve_embedder_source(model_name: str) -> str:
    """Prefer the local on-device copy for offline-first edge operation; fall
    back to the HF hub id only on a fresh clone that lacks the local dir."""
    if model_name == EMBEDDING_MODEL and LOCAL_MXBAI_DIR.is_dir():
        return str(LOCAL_MXBAI_DIR)
    return model_name


def _get_embedder(model_name: str):
    """Load the local mxbai bi-encoder once and cache it. Raises RAGError loud if
    the model cannot be loaded (no silent fallback, per CLAUDE.md §0/§3)."""
    global _EMBEDDER, _EMBEDDER_NAME
    if _EMBEDDER is not None and _EMBEDDER_NAME == model_name:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # noqa: BLE001
        raise RAGError(
            f"sentence-transformers not installed for local embeddings: {e}"
        ) from e
    source = _resolve_embedder_source(model_name)
    try:
        _EMBEDDER = SentenceTransformer(source)
    except Exception as e:  # noqa: BLE001
        raise RAGError(
            f"failed to load local embedder '{model_name}' (source={source}): {e}"
        ) from e
    _EMBEDDER_NAME = model_name
    logger.info("Loaded local embedder '%s' from %s (%d-d)", model_name, source, EMBED_DIM)
    return _EMBEDDER


class RAGError(RuntimeError):
    """Raised when RAG cannot serve a query (no precedent above threshold, or backend down)."""


class CrisisRAG:
    """Production RAG with local mxbai embeddings. No heuristic fallback."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str = EMBEDDING_MODEL,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.db_path = db_path or RAG_DB_PATH
        self.embedding_model_name = embedding_model
        self.min_score = min_score
        self._client = None
        self._collection = None

    def _ensure_initialized(self) -> None:
        if self._client is not None:
            return
        import chromadb
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB initialized at %s (%d documents)",
                    self.db_path, self._collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with the local mxbai bi-encoder (normalized, cosine-ready).
        Raises RAGError loud on failure — never a silent fallback."""
        embedder = _get_embedder(self.embedding_model_name)
        try:
            vecs = embedder.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True,
            )
        except Exception as e:  # noqa: BLE001
            raise RAGError(f"local embedding failed: {e}") from e
        return [v.tolist() for v in vecs]

    def index_text(self, text: str, source: str = "unknown", metadata: dict | None = None) -> int:
        self._ensure_initialized()
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE_WORDS):
            chunk = " ".join(words[i:i + CHUNK_SIZE_WORDS])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        if not chunks:
            return 0

        embeddings = self._embed(chunks)
        existing = self._collection.count()
        ids = [f"{source}_{existing + i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_idx": i, **(metadata or {})} for i in range(len(chunks))]
        self._collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        logger.info("Indexed %d chunks from '%s'", len(chunks), source)
        return len(chunks)

    def retrieve_precedents(self, query: str, n: int = 3) -> list[dict[str, Any]]:
        """Retrieve top-n precedents. Returns only results with score >= min_score.

        Returns empty list if collection is empty or nothing clears the threshold.
        Caller should raise RAGError if emptiness is a hard error for their path.
        """
        self._ensure_initialized()
        if self._collection.count() == 0:
            logger.warning("RAG collection empty — call build_corpus() to populate.")
            return []

        q_emb = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(n, self._collection.count()),
        )

        precedents = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i]
            if score < self.min_score:
                continue
            precedents.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "relevance_score": round(score, 3),
            })
        return precedents

    def require_precedent(self, query: str) -> dict[str, Any]:
        """Retrieve top precedent or raise RAGError. Use in production paths."""
        ps = self.retrieve_precedents(query, n=1)
        if not ps:
            raise RAGError(
                f"No precedent above threshold {self.min_score} for query: {query[:80]}"
            )
        return ps[0]

    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()
