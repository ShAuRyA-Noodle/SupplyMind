"""Regression checks for the Chroma-free persistent RAG store."""

import json
import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INDEXER = ROOT / "rl/rag/indexer.py"
spec = importlib.util.spec_from_file_location("supplymind_rag_indexer", INDEXER)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CrisisRAG, EMBED_DIM = module.CrisisRAG, module.EMBED_DIM


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBED_DIM
    vector[index] = 1.0
    return vector


def test_index_retrieves_and_persists_cosine_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = CrisisRAG(db_path=tmp_path, min_score=0.6)
    monkeypatch.setattr(rag, "_embed", lambda texts: [_unit_vector(0 if "chip" in text else 1) for text in texts])
    assert rag.index_text("chip shortage " * 12, source="semiconductor-report") == 1
    assert rag.index_text("port blockade " * 12, source="port-report") == 1
    assert rag.retrieve_precedents("chip disruption", n=2) == [
        {"text": ("chip shortage " * 12).strip(), "source": "semiconductor-report", "relevance_score": 1.0}
    ]
    assert CrisisRAG(db_path=tmp_path).count() == 2


def test_migrates_legacy_chroma_queue_without_chromadb(tmp_path: Path) -> None:
    legacy = tmp_path / "chroma.sqlite3"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE embeddings_queue (id TEXT, vector BLOB, metadata TEXT, operation INTEGER)"
        )
        conn.execute(
            "INSERT INTO embeddings_queue VALUES (?, ?, ?, 0)",
            ("old-1", np.asarray(_unit_vector(0), dtype="<f4").tobytes(),
             json.dumps({"source": "old-report", "chroma:document": "historic chip shortage"})),
        )
    rag = CrisisRAG(db_path=tmp_path)
    monkeypatch_vector = lambda texts: [_unit_vector(0) for _ in texts]
    rag._embed = monkeypatch_vector
    assert rag.count() == 1
    assert rag.retrieve_precedents("chip", n=1)[0]["source"] == "old-report"
    assert CrisisRAG(db_path=tmp_path).count() == 1


def test_committed_legacy_index_migrates_all_vectors(tmp_path: Path) -> None:
    source = ROOT / "rl/rag/chroma_db/chroma.sqlite3"
    shutil.copy2(source, tmp_path / "chroma.sqlite3")
    with sqlite3.connect(source) as conn:
        expected = conn.execute(
            "SELECT COUNT(*) FROM embeddings_queue WHERE operation = 0 AND vector IS NOT NULL"
        ).fetchone()[0]
    assert expected > 0
    assert CrisisRAG(db_path=tmp_path).count() == expected
