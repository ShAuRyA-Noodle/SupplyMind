"""
RAG crisis documentation retrieval — Ollama-powered, NO fallback.

Production design:
  - Embeddings: Ollama `nomic-embed-text` (768-d, runs locally, no HF dependency)
  - Storage: ChromaDB persistent client at rl/rag/chroma_db/
  - Retrieval: cosine similarity, min score threshold, or raise

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

RAG_DB_PATH = Path(__file__).resolve().parent.parent.parent / "rl" / "rag" / "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
EMBED_DIM = 768
CHUNK_SIZE_WORDS = 300
COLLECTION_NAME = "crisis_docs_v2"
MIN_SCORE = 0.60  # cosine similarity threshold for "valid precedent"


class RAGError(RuntimeError):
    """Raised when RAG cannot serve a query (no precedent above threshold, or backend down)."""


class CrisisRAG:
    """Production RAG with Ollama embeddings. No heuristic fallback."""

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
        """Embed with Ollama nomic-embed-text. Raises RAGError on failure."""
        import ollama
        out = []
        for t in texts:
            try:
                r = ollama.embeddings(model=self.embedding_model_name, prompt=t)
                out.append(r["embedding"])
            except Exception as e:
                raise RAGError(f"Ollama embedding failed: {e}") from e
        return out

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
