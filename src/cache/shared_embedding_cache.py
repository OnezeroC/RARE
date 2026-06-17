"""Shared persistent embedding cache for local baseline experiments.

This cache is intentionally simple:
- backend: SQLite
- key: (model_name, normalized_flag, text_hash)
- value: float32 embedding blob

It is designed so multiple baseline runners can reuse the same query embeddings
across separate Python processes and separate experiment scripts.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


class SharedEmbeddingCache:
    """SQLite-backed embedding cache shared across baseline runs."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    normalized INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text_hash, model_name, normalized)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_model_norm
                ON embeddings(model_name, normalized)
                """
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

    def get_many(
        self,
        texts: Iterable[str],
        *,
        model_name: str,
        normalized: bool,
    ) -> dict[str, np.ndarray]:
        hits: dict[str, np.ndarray] = {}
        norm_int = 1 if normalized else 0
        with self._connect() as conn:
            for text in texts:
                row = conn.execute(
                    """
                    SELECT dim, embedding
                    FROM embeddings
                    WHERE text_hash = ? AND model_name = ? AND normalized = ?
                    """,
                    (self._hash_text(text), model_name, norm_int),
                ).fetchone()
                if row is None:
                    continue
                dim, blob = row
                vec = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
                hits[text] = vec
        return hits

    def put_many(
        self,
        texts: Iterable[str],
        embeddings: Iterable[np.ndarray],
        *,
        model_name: str,
        normalized: bool,
    ) -> None:
        norm_int = 1 if normalized else 0
        rows = []
        for text, embedding in zip(texts, embeddings):
            arr = np.asarray(embedding, dtype=np.float32)
            rows.append(
                (
                    self._hash_text(text),
                    model_name,
                    norm_int,
                    int(arr.shape[0]),
                    arr.tobytes(),
                    text,
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, model_name, normalized, dim, embedding, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def count(self, *, model_name: str | None = None, normalized: bool | None = None) -> int:
        query = "SELECT COUNT(*) FROM embeddings"
        clauses = []
        params: list[object] = []
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if normalized is not None:
            clauses.append("normalized = ?")
            params.append(1 if normalized else 0)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])
