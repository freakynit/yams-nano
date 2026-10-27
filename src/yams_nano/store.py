"""SQLite persistence, FTS5/BM25, exact cosine retrieval, and weighted RRF."""

from __future__ import annotations

import heapq
import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np

from .config import Settings, project_root
from .embeddings import Embedder, normalize_vectors
from .models import MemoryInput, MemoryPatch, Scope, SearchOptions

SCHEMA_VERSION = 1
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS memories (
        rowid INTEGER PRIMARY KEY,
        id TEXT NOT NULL UNIQUE,
        content TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        project_path TEXT,
        tags TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{}',
        inserted_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        embedding BLOB NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS memories_project ON memories(project_path, inserted_at, id)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        title, content, tags, content='memories', content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    )""",
    """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, title, content, tags)
        VALUES (new.rowid, new.title, new.content, new.tags);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
        VALUES ('delete', old.rowid, old.title, old.content, old.tags);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
        VALUES ('delete', old.rowid, old.title, old.content, old.tags);
        INSERT INTO memories_fts(rowid, title, content, tags)
        VALUES (new.rowid, new.title, new.content, new.tags);
    END""",
]
PUBLIC_COLUMNS = (
    "m.id, m.content, m.title, m.project_path, m.tags, m.metadata, "
    "m.inserted_at, m.updated_at, m.version"
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def public_memory(row: sqlite3.Row) -> dict[str, Any]:
    data = {
        key: row[key]
        for key in (
            "id",
            "content",
            "title",
            "project_path",
            "tags",
            "metadata",
            "inserted_at",
            "updated_at",
            "version",
        )
    }
    data["tags"] = json.loads(data["tags"])
    data["metadata"] = json.loads(data["metadata"])
    return data


def literal_fts(query: str) -> str:
    # Quote literal words so paths, symbols, operators, and unmatched quotes are harmless.
    words = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:64]
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words)


class MemoryStore:
    def __init__(self, settings: Settings, embedder: Embedder):
        self.settings = settings
        self.embedder = embedder
        self.path = settings.database
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        self._initialize()

    @contextmanager
    def connection(self, *, write: bool = False):
        conn = sqlite3.connect(
            self.path,
            timeout=self.settings.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout={self.settings.busy_timeout_ms}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        # Set WAL outside the schema transaction; readers and writers can overlap.
        with closing(
            sqlite3.connect(self.path, timeout=self.settings.busy_timeout_ms / 1000)
        ) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        with self.connection(write=True) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f"unsupported database schema version: {version}")
            for statement in SCHEMA:
                conn.execute(statement)
            identity = json.dumps(self.settings.embedding_identity, sort_keys=True)
            previous = conn.execute("SELECT value FROM settings WHERE key='embedding'").fetchone()
            if previous and previous[0] != identity:
                raise ValueError(
                    "embedding configuration differs from this database; restore its original "
                    "model/revision/dimensions/max_tokens or use a different YAMS_DB_PATH"
                )
            conn.execute("INSERT OR IGNORE INTO settings VALUES ('embedding', ?)", (identity,))
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _batch(self, values: list) -> None:
        if not 1 <= len(values) <= self.settings.max_batch_size:
            raise ValueError(f"provide 1 to {self.settings.max_batch_size} items")

    def _text(self, text: str) -> None:
        if not text.strip():
            raise ValueError("text must not be blank")
        if len(text) > self.settings.max_text_chars:
            raise ValueError(f"text exceeds {self.settings.max_text_chars} characters; split it")

    def _location(self, path: str | None) -> str | None:
        # Null/empty explicitly selects the global namespace.
        return project_root(path) if path else None

    def _vectors(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        return normalize_vectors(
            self.embedder.encode(texts, query=query),
            len(texts),
            self.settings.dimensions,
        )

    def create(self, memories: list[MemoryInput]) -> list[dict]:
        self._batch(memories)
        for memory in memories:
            self._text(memory.content)
            self._location(memory.project_path)
        vectors = self._vectors([memory.content for memory in memories])
        now = timestamp()
        results = []
        with self.connection(write=True) as conn:
            for memory, vector in zip(memories, vectors, strict=True):
                data = memory.model_dump()
                data["project_path"] = self._location(memory.project_path)
                data.update(id=str(uuid4()), inserted_at=now, updated_at=now, version=1)
                conn.execute(
                    """INSERT INTO memories
                    (id, content, title, project_path, tags, metadata,
                     inserted_at, updated_at, version, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data["id"],
                        data["content"],
                        data["title"],
                        data["project_path"],
                        json.dumps(data["tags"], ensure_ascii=False),
                        json.dumps(data["metadata"]),
                        now,
                        now,
                        1,
                        vector.tobytes(),
                    ),
                )
                results.append(data)
        return results

    def embed(self, texts: list[str], *, query: bool = False) -> dict:
        self._batch(texts)
        for text in texts:
            self._text(text)
        vectors = self._vectors(texts, query=query)
        return {
            "embeddings": vectors.tolist(),
            "dimensions": self.settings.dimensions,
            "task": "retrieval.query" if query else "retrieval.passage",
        }

    def get(self, ids: list[str], *, project_path: str | None) -> dict:
        self._batch(ids)
        project = self._location(project_path)
        with self.connection() as conn:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT {PUBLIC_COLUMNS} FROM memories m WHERE m.id IN ({placeholders}) "
                "AND m.project_path IS ?",
                [*ids, project],
            ).fetchall()
        found = {row["id"]: public_memory(row) for row in rows}
        return {
            "memories": [found[id_] for id_ in ids if id_ in found],
            "missing_ids": [id_ for id_ in ids if id_ not in found],
        }

    def update(
        self,
        id_: str,
        patch: MemoryPatch,
        expected_version: int | None = None,
        *,
        project_path: str | None,
    ) -> dict:
        changes = patch.model_dump(exclude_unset=True)
        if not changes:
            raise ValueError("provide at least one field to update")
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=? AND project_path IS ?",
                (id_, self._location(project_path)),
            ).fetchone()
        if row is None:
            raise ValueError(f"memory not found: {id_}")
        if expected_version is not None and expected_version != row["version"]:
            raise ValueError("version conflict; read the memory again before updating")
        before = public_memory(row)
        if "project_path" in changes:
            changes["project_path"] = self._location(changes["project_path"])
        data = {**before, **changes}
        self._text(data["content"])
        vector = row["embedding"]
        if data["content"] != before["content"]:
            vector = self._vectors([data["content"]])[0].tobytes()
        data["updated_at"] = timestamp()
        data["version"] += 1
        with self.connection(write=True) as conn:
            changed = conn.execute(
                """UPDATE memories SET content=?, title=?, project_path=?, tags=?, metadata=?,
                updated_at=?, version=?, embedding=? WHERE id=? AND version=?""",
                (
                    data["content"],
                    data["title"],
                    data["project_path"],
                    json.dumps(data["tags"], ensure_ascii=False),
                    json.dumps(data["metadata"]),
                    data["updated_at"],
                    data["version"],
                    vector,
                    id_,
                    before["version"],
                ),
            ).rowcount
            if not changed:
                raise ValueError("memory changed concurrently; read it again before updating")
        return data

    def delete(self, ids: list[str], *, project_path: str | None) -> dict:
        self._batch(ids)
        project = self._location(project_path)
        ids = list(dict.fromkeys(ids))
        placeholders = ",".join("?" for _ in ids)
        with self.connection(write=True) as conn:
            found = {
                row[0]
                for row in conn.execute(
                    f"SELECT id FROM memories WHERE id IN ({placeholders}) AND project_path IS ?",
                    [*ids, project],
                )
            }
            conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders}) AND project_path IS ?",
                [*ids, project],
            )
        return {
            "deleted_ids": [id_ for id_ in ids if id_ in found],
            "missing_ids": [id_ for id_ in ids if id_ not in found],
        }

    def _filters(
        self,
        scope: Scope,
        project_path: str | None,
        include_global: bool,
        tags: list[str] | None,
    ) -> tuple[str, list]:
        clauses, params = [], []
        if scope == "project":
            if not project_path:
                raise ValueError(
                    "project_path is required for scope='project'; "
                    "supply an absolute path or use scope='global'/'all'"
                )
            project = project_root(project_path)
            clauses.append(
                "(m.project_path=? OR m.project_path IS NULL)"
                if include_global
                else "m.project_path=?"
            )
            params.append(project)
        elif scope == "global":
            clauses.append("m.project_path IS NULL")
        elif scope != "all":
            raise ValueError("scope must be project, all, or global")
        if tags:
            tags = MemoryInput(content="validate", project_path=None, tags=tags).tags
            for tag in tags:
                clauses.append("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE value=?)")
                params.append(tag)
        return " AND ".join(clauses) or "1=1", params

    def list(
        self,
        *,
        scope: Scope = "project",
        project_path: str | None = None,
        include_global: bool = True,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be nonnegative")
        where, params = self._filters(scope, project_path, include_global, tags)
        with self.connection() as conn:
            total = conn.execute(
                f"SELECT count(*) FROM memories m WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT {PUBLIC_COLUMNS} FROM memories m WHERE {where} "
                "ORDER BY m.inserted_at DESC, m.id ASC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return {
            "memories": [public_memory(row) for row in rows],
            "total": total,
            "next_offset": offset + limit if offset + limit < total else None,
        }

    def search(
        self,
        query: str,
        *,
        scope: Scope = "project",
        project_path: str | None = None,
        include_global: bool = True,
        tags: list[str] | None = None,
        limit: int | None = None,
        candidate_limit: int | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        fts_weight: float | None = None,
        min_similarity: float | None = None,
        fts_query: str | None = None,
    ) -> dict:
        self._text(query)
        values = {
            name: value
            if value is not None
            else getattr(self.settings, "search_limit" if name == "limit" else name)
            for name, value in {
                "limit": limit,
                "candidate_limit": candidate_limit,
                "rrf_k": rrf_k,
                "vector_weight": vector_weight,
                "fts_weight": fts_weight,
            }.items()
        }
        # Raising the result limit alone should also raise the candidate pool.
        if candidate_limit is None:
            values["candidate_limit"] = max(values["candidate_limit"], values["limit"])
        options = SearchOptions(**values, min_similarity=min_similarity)
        if options.candidate_limit < options.limit:
            raise ValueError("candidate_limit must be at least limit")
        if options.vector_weight + options.fts_weight == 0:
            raise ValueError("at least one search weight must be positive")
        where, params = self._filters(scope, project_path, include_global, tags)
        expression = fts_query if fts_query is not None else literal_fts(query)
        lexical, semantic = [], []
        with self.connection() as conn:
            # Rank both branches and fetch results in one consistent SQLite snapshot.
            if options.fts_weight and expression:
                try:
                    lexical = conn.execute(
                        "SELECT m.id, bm25(memories_fts, 2.0, 1.0, 1.5) AS bm25 "
                        "FROM memories_fts JOIN memories m ON m.rowid=memories_fts.rowid "
                        f"WHERE memories_fts MATCH ? AND {where} ORDER BY bm25, m.id LIMIT ?",
                        [expression, *params, options.candidate_limit],
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    raise ValueError(f"invalid FTS5 expression: {exc}") from exc
            # Avoid loading the model for an empty namespace.
            if (
                options.vector_weight
                and conn.execute(
                    f"SELECT 1 FROM memories m WHERE {where} LIMIT 1",
                    params,
                ).fetchone()
            ):
                query_vector = self._vectors([query], query=True)[0]
                cursor = conn.execute(
                    f"SELECT m.id, m.embedding FROM memories m WHERE {where}", params
                )
                heap = []
                while rows := cursor.fetchmany(256):
                    matrix = np.vstack(
                        [np.frombuffer(row["embedding"], dtype="<f4") for row in rows]
                    )
                    similarities = np.clip(matrix @ query_vector, -1.0, 1.0)
                    for row, similarity in zip(rows, similarities, strict=True):
                        score = float(similarity)
                        if options.min_similarity is not None and score < options.min_similarity:
                            continue
                        # UUID is a stable tie breaker, independent of insertion order.
                        item = (score, row["id"])
                        if len(heap) < options.candidate_limit:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
                semantic = sorted(heap, reverse=True)
            fused: dict[str, dict] = {}
            for rank, row in enumerate(lexical, 1):
                fused[row["id"]] = {
                    "score": options.fts_weight / (options.rrf_k + rank),
                    "fts_rank": rank,
                    "bm25": row["bm25"],
                    "vector_rank": None,
                    "similarity": None,
                }
            for rank, (similarity, id_) in enumerate(semantic, 1):
                hit = fused.setdefault(
                    id_,
                    {
                        "score": 0.0,
                        "fts_rank": None,
                        "bm25": None,
                        "vector_rank": None,
                        "similarity": None,
                    },
                )
                hit.update(vector_rank=rank, similarity=similarity)
                hit["score"] += options.vector_weight / (options.rrf_k + rank)
            ranked = sorted(fused, key=lambda id_: (-fused[id_]["score"], id_))[: options.limit]
            results = []
            for id_ in ranked:
                row = conn.execute(
                    f"SELECT {PUBLIC_COLUMNS} FROM memories m WHERE id=?",
                    (id_,),
                ).fetchone()
                results.append({**public_memory(row), **fused[id_]})
        return {
            "results": results,
            "parameters": {
                **options.model_dump(),
                "scope": scope,
                "project_path": project_root(project_path) if scope == "project" else None,
                "include_global": include_global,
                "tags": tags or [],
                "fts_query": expression,
            },
        }

    def stats(self) -> dict:
        with self.connection() as conn:
            projects = conn.execute(
                "SELECT project_path, count(*) AS count FROM memories GROUP BY project_path "
                "ORDER BY project_path",
            ).fetchall()
        return {
            "database": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "total_memories": sum(row["count"] for row in projects),
            "projects": [dict(row) for row in projects],
            "embedding": self.settings.embedding_identity,
            "search_defaults": {
                name: getattr(self.settings, name)
                for name in (
                    "search_limit",
                    "candidate_limit",
                    "rrf_k",
                    "vector_weight",
                    "fts_weight",
                )
            },
            "limits": {
                name: getattr(self.settings, name)
                for name in (
                    "max_batch_size",
                    "max_text_chars",
                    "max_tokens",
                    "embedding_batch_size",
                )
            },
        }
