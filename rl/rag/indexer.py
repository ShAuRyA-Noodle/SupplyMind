"""Persistent local RAG with Ollama embeddings and SQLite cosine search.

The old Chroma index is read once from its SQLite queue when the new store is
empty. No Chroma package or server is needed, and its source vectors remain
available for migration on existing installations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

RAG_DB_PATH = Path(__file__).resolve().parent / "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
EMBED_DIM = 768
CHUNK_SIZE_WORDS = 300
MIN_SCORE = 0.60


class RAGError(RuntimeError):
    """Raised when RAG cannot serve a query or its local index is invalid."""


class CrisisRAG:
    """Local persistent vector retrieval with the existing public API."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str = EMBEDDING_MODEL,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else RAG_DB_PATH
        self.embedding_model_name = embedding_model
        self.min_score = min_score
        self._db_file = self.db_path / "vectors.sqlite3"
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_file) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                "id TEXT PRIMARY KEY, vector BLOB NOT NULL, "
                "document TEXT NOT NULL, source TEXT NOT NULL)"
            )
            count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            legacy = self.db_path / "chroma.sqlite3"
            if count == 0 and legacy.is_file():
                self._migrate_chroma_queue(conn, legacy)
        self._initialized = True

    @staticmethod
    def _migrate_chroma_queue(conn: sqlite3.Connection, legacy: Path) -> None:
        """Import committed Chroma vectors without importing vulnerable code."""
        with sqlite3.connect(legacy) as old:
            rows = old.execute(
                "SELECT id, vector, metadata FROM embeddings_queue "
                "WHERE operation = 0 AND vector IS NOT NULL"
            )
            migrated = 0
            for identifier, vector, raw_metadata in rows:
                if len(vector) != EMBED_DIM * 4:
                    raise RAGError(f"Legacy vector {identifier} has an unexpected dimension")
                metadata = json.loads(raw_metadata or "{}")
                document = metadata.get("chroma:document")
                if not isinstance(document, str):
                    raise RAGError(f"Legacy vector {identifier} has no document")
                conn.execute(
                    "INSERT OR IGNORE INTO vectors VALUES (?, ?, ?, ?)",
                    (identifier, vector, document, str(metadata.get("source", "unknown"))),
                )
                migrated += 1
        logger.info("Migrated %d vectors from the legacy Chroma SQLite queue", migrated)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with local Ollama; failures are explicit to callers."""
        import ollama

        output = []
        for text in texts:
            try:
                embedding = ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
            except Exception as exc:
                raise RAGError(f"Ollama embedding failed: {exc}") from exc
            if len(embedding) != EMBED_DIM:
                raise RAGError(f"Expected {EMBED_DIM} embedding dimensions, got {len(embedding)}")
            output.append(embedding)
        return output

    def index_text(self, text: str, source: str = "unknown", metadata: dict | None = None) -> int:
        self._ensure_initialized()
        words = text.split()
        chunks = []
        for index in range(0, len(words), CHUNK_SIZE_WORDS):
            chunk = " ".join(words[index:index + CHUNK_SIZE_WORDS])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        if not chunks:
            return 0
        embeddings = self._embed(chunks)
        with sqlite3.connect(self._db_file) as conn:
            conn.executemany(
                "INSERT INTO vectors VALUES (?, ?, ?, ?)",
                [
                    (str(uuid4()), np.asarray(vector, dtype="<f4").tobytes(), chunk, source)
                    for vector, chunk in zip(embeddings, chunks)
                ],
            )
        logger.info("Indexed %d chunks from '%s'", len(chunks), source)
        return len(chunks)

    def retrieve_precedents(self, query: str, n: int = 3) -> list[dict[str, Any]]:
        """Return top-n documents whose cosine similarity clears min_score."""
        self._ensure_initialized()
        with sqlite3.connect(self._db_file) as conn:
            rows = conn.execute("SELECT vector, document, source FROM vectors").fetchall()
        if not rows or n <= 0:
            return []
        query_vector = np.asarray(self._embed([query])[0], dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise RAGError("Ollama returned a zero embedding")
        vectors = np.stack([np.frombuffer(row[0], dtype="<f4") for row in rows])
        if vectors.shape[1] != EMBED_DIM:
            raise RAGError("Local index has an unexpected embedding dimension")
        norms = np.linalg.norm(vectors, axis=1) * query_norm
        scores = np.divide(vectors @ query_vector, norms, out=np.zeros(len(rows)), where=norms != 0)
        top = np.argsort(-scores)[:n]
        return [
            {
                "text": rows[index][1],
                "source": rows[index][2],
                "relevance_score": round(float(scores[index]), 3),
            }
            for index in top
            if scores[index] >= self.min_score
        ]

    def require_precedent(self, query: str) -> dict[str, Any]:
        precedents = self.retrieve_precedents(query, n=1)
        if not precedents:
            raise RAGError(f"No precedent above threshold {self.min_score} for query: {query[:80]}")
        return precedents[0]

    def count(self) -> int:
        self._ensure_initialized()
        with sqlite3.connect(self._db_file) as conn:
            return conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
