import concurrent.futures
import json
import sqlite3
import threading

import numpy as np
import pytest

from yams_nano.models import MemoryInput, MemoryPatch
from yams_nano.store import MemoryStore

from .fakes import PROJECT, FakeEmbedder


def add(store, content="sqlite memory", **kwargs):
    kwargs.setdefault("project_path", PROJECT)
    return store.create([MemoryInput(content=content, **kwargs)])[0]


def test_batch_crud_persistence_and_timestamps(store, settings, embedder):
    records = store.create(
        [
            MemoryInput(
                content="first",
                tags=["dev", "dev"],
                metadata={"nested": [1, True]},
                project_path=PROJECT,
            ),
            MemoryInput(content="second", project_path=""),
        ]
    )
    a, b = records
    assert a["project_path"] == PROJECT
    assert b["project_path"] is None
    assert a["tags"] == ["dev"]
    assert a["inserted_at"] == a["updated_at"]
    assert a["inserted_at"].endswith("Z")
    assert len(embedder.calls) == 1 and embedder.calls[0][0] == ["first", "second"]
    reopened = MemoryStore(settings, embedder)
    assert reopened.get([b["id"], "missing", a["id"]], project_path=PROJECT) == {
        "memories": [a],
        "missing_ids": [b["id"], "missing"],
    }
    changed = reopened.update(
        a["id"], MemoryPatch(title="New", tags=[], metadata={}), 1, project_path=PROJECT
    )
    assert changed["inserted_at"] == a["inserted_at"]
    assert changed["updated_at"] > a["updated_at"]
    assert changed["version"] == 2
    assert changed["content"] == "first" and changed["metadata"] == {}
    assert len(embedder.calls) == 1  # Metadata edits need no inference.
    reopened.update(
        a["id"], MemoryPatch(content="replacement", project_path=None), project_path=PROJECT
    )
    assert len(embedder.calls) == 2
    assert reopened.get([a["id"]], project_path=None)["memories"][0]["project_path"] is None
    assert reopened.delete([a["id"], a["id"], "missing"], project_path=None)["deleted_ids"] == [
        a["id"]
    ]
    assert reopened.delete([a["id"]], project_path=PROJECT)["missing_ids"] == [a["id"]]


def test_version_and_missing_errors(store):
    record = add(store)
    with pytest.raises(ValueError, match="version conflict"):
        store.update(record["id"], MemoryPatch(title="stale"), 99, project_path=PROJECT)
    with pytest.raises(ValueError, match="not found"):
        store.update("missing", MemoryPatch(title="no"), project_path=PROJECT)
    with pytest.raises(ValueError, match="at least one"):
        store.update(record["id"], MemoryPatch(), project_path=PROJECT)
    assert store.get([record["id"]], project_path=PROJECT)["memories"][0] == record


def test_batch_rollback_on_sql_failure(store):
    with store.connection(write=True) as conn:
        conn.execute("""CREATE TRIGGER force_failure BEFORE INSERT ON memories
                     WHEN new.title='fail' BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        store.create(
            [
                MemoryInput(content="first", project_path=PROJECT),
                MemoryInput(content="second", title="fail", project_path=PROJECT),
            ]
        )
    assert store.stats()["total_memories"] == 0
    assert store.search("first", vector_weight=0, project_path=PROJECT)["results"] == []


def test_embedding_failure_and_invalid_vectors_leave_no_records(store, embedder):
    for vector in [np.zeros((1, 32)), np.ones((1, 2)), np.full((1, 32), np.nan)]:
        embedder.encode = lambda texts, query=False, v=vector: v
        with pytest.raises(ValueError, match="vectors"):
            add(store)
    assert store.stats()["total_memories"] == 0


def test_fts_sync_and_advanced_queries(store, embedder):
    record = add(store, "connection pool retry", title="PostgreSQL", tags=["database"])
    assert (
        store.search("postgresql", vector_weight=0, project_path=PROJECT)["results"][0]["id"]
        == record["id"]
    )
    assert store.search(
        "ignored", vector_weight=0, fts_query='"connection pool"', project_path=PROJECT
    )["results"]
    assert store.search(
        "ignored", vector_weight=0, fts_query="tags:database", project_path=PROJECT
    )["results"]
    store.update(
        record["id"],
        MemoryPatch(content="redis cache", title="Redis", tags=["cache"]),
        project_path=PROJECT,
    )
    assert (
        store.search("postgresql connection database", vector_weight=0, project_path=PROJECT)[
            "results"
        ]
        == []
    )
    assert store.search("redis", vector_weight=0, project_path=PROJECT)["results"]
    store.delete([record["id"]], project_path=PROJECT)
    assert store.search("redis", vector_weight=0, project_path=PROJECT)["results"] == []
    assert len(embedder.calls) == 2


def test_project_global_tags_and_pagination(store, tmp_path):
    a = add(store, tags=["one", "two"])
    global_ = add(store, project_path="", tags=["one"])
    other = add(store, project_path=str(tmp_path / "elsewhere"), tags=["two"])
    assert store.list(project_path=PROJECT)["total"] == 2
    assert store.list(include_global=False, project_path=PROJECT)["memories"] == [a]
    assert store.list(scope="global", project_path=PROJECT)["memories"] == [global_]
    assert store.list(scope="all", project_path=PROJECT)["total"] == 3
    assert store.list(project_path=other["project_path"], include_global=False)["memories"] == [
        other
    ]
    assert store.list(tags=["one", "two"], project_path=PROJECT)["memories"] == [a]
    assert store.list(tags=["ONE"], project_path=PROJECT)["memories"] == []
    page = store.list(scope="all", limit=2, project_path=PROJECT)
    assert page["next_offset"] == 2
    final = store.list(scope="all", limit=2, offset=page["next_offset"], project_path=PROJECT)
    assert final["next_offset"] is None
    assert len({m["id"] for m in page["memories"] + final["memories"]}) == 3
    for mode in [{}, {"vector_weight": 0}, {"fts_weight": 0}]:
        results = store.search("sqlite", **mode, project_path=PROJECT)["results"]
        assert {r["id"] for r in results} <= {a["id"], global_["id"]}
        filtered = store.search(
            "sqlite", scope="all", tags=["one", "two"], **mode, project_path=PROJECT
        )["results"]
        assert [r["id"] for r in filtered] == [a["id"]]


def test_rrf_combines_independent_ranks_and_weights(store):
    store.create(
        [
            MemoryInput(content=text, project_path=PROJECT)
            for text in [
                "sqlite transaction commit",
                "sqlite sqlite sqlite",
                "database transaction rollback",
            ]
        ]
    )
    lexical = store.search("sqlite transaction", vector_weight=0, project_path=PROJECT)["results"]
    vector = store.search("sqlite transaction", fts_weight=0, project_path=PROJECT)["results"]
    hybrid = store.search(
        "sqlite transaction", rrf_k=7, vector_weight=2, fts_weight=3, project_path=PROJECT
    )["results"]
    fts_ranks = {r["id"]: i for i, r in enumerate(lexical, 1)}
    vector_ranks = {r["id"]: i for i, r in enumerate(vector, 1)}
    for hit in hybrid:
        id_ = hit["id"]
        expected = 2 / (7 + vector_ranks[id_])
        if id_ in fts_ranks:
            expected += 3 / (7 + fts_ranks[id_])
        assert hit["score"] == pytest.approx(expected)
        assert hit["vector_rank"] == vector_ranks[id_]
        assert hit["fts_rank"] == fts_ranks.get(id_)
    assert [r["score"] for r in hybrid] == sorted([r["score"] for r in hybrid], reverse=True)
    assert all(-1 <= r["similarity"] <= 1 for r in hybrid)
    assert (
        len(store.search("sqlite", candidate_limit=1, limit=1, project_path=PROJECT)["results"])
        == 1
    )


def test_min_similarity_only_filters_vector_branch(store):
    add(store, "sqlite database")
    lexical = store.search("sqlite", min_similarity=1, project_path=PROJECT)["results"]
    assert lexical and lexical[0]["vector_rank"] is None
    assert (
        store.search("sqlite", min_similarity=1, fts_weight=0, project_path=PROJECT)["results"]
        == []
    )


def test_filter_before_candidates(store, tmp_path):
    add(store, "sqlite sqlite sqlite", project_path=str(tmp_path / "other"))
    wanted = add(store, "sqlite long descriptive memory with additional words")
    for mode in [{"vector_weight": 0}, {"fts_weight": 0}, {}]:
        assert (
            store.search("sqlite", limit=1, candidate_limit=1, **mode, project_path=PROJECT)[
                "results"
            ][0]["id"]
            == wanted["id"]
        )


@pytest.mark.parametrize(
    "query", ["src/foo_bar.py:123", 'a "quote', "x OR NOT y", "***", "你好", "café"]
)
def test_literal_fts_handles_code_punctuation_and_unicode(store, query):
    add(store, query)
    result = store.search(query, vector_weight=0, project_path=PROJECT)
    assert isinstance(result["results"], list)


def test_invalid_fts_expression_is_actionable(store):
    with pytest.raises(ValueError, match="invalid FTS5 expression"):
        store.search("anything", vector_weight=0, fts_query='"unclosed', project_path=PROJECT)


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"candidate_limit": 1001},
        {"limit": 10, "candidate_limit": 2},
        {"rrf_k": 0},
        {"vector_weight": -1},
        {"fts_weight": float("nan")},
        {"vector_weight": 0, "fts_weight": 0},
        {"min_similarity": 2},
        {"scope": "invalid"},
    ],
)
def test_search_validation(store, params):
    with pytest.raises(ValueError):
        store.search("query", **params, project_path=PROJECT)


def test_empty_database_and_lexical_search_do_not_embed(store, embedder):
    assert store.search("anything", project_path=PROJECT)["results"] == []
    assert embedder.calls == []
    add(store)
    store.search("sqlite", vector_weight=0, project_path=PROJECT)
    assert len(embedder.calls) == 1


def test_batch_and_character_limits(store):
    for method in [store.create, store.get, store.delete, store.embed]:
        with pytest.raises(ValueError, match="provide 1"):
            kwargs = {"project_path": PROJECT} if method in (store.get, store.delete) else {}
            method([], **kwargs)
    with pytest.raises(ValueError):
        store.embed(["x"] * 129)
    with pytest.raises(ValueError, match="characters"):
        add(store, "a" * 50001)
    with pytest.raises(ValueError, match="blank"):
        store.search("   ", project_path=PROJECT)
    assert (
        store.search("query", limit=100, project_path=PROJECT)["parameters"]["candidate_limit"]
        == 100
    )


def test_embedding_batch_order_and_normalization(store, embedder):
    result = store.embed(["sqlite", "database"], query=True)
    assert len(result["embeddings"]) == 2
    assert np.allclose(np.linalg.norm(result["embeddings"], axis=1), 1)
    assert embedder.calls == [(["sqlite", "database"], True)]
    assert result["task"] == "retrieval.query"


def test_concurrent_writers(settings):
    def writer(index):
        independent = MemoryStore(settings, FakeEmbedder())
        for number in range(8):
            add(independent, f"writer {index} item {number}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(4)))
    assert MemoryStore(settings, FakeEmbedder()).stats()["total_memories"] == 32


def test_concurrent_updates_reject_lost_writes(store, embedder):
    record = add(store)
    barrier = threading.Barrier(2)
    original = embedder.encode

    def synchronized(texts, *, query=False):
        barrier.wait(timeout=10)
        return original(texts, query=query)

    embedder.encode = synchronized

    def update(index):
        try:
            return store.update(
                record["id"], MemoryPatch(content=f"changed {index}"), project_path=PROJECT
            )
        except ValueError as exc:
            return str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(update, [1, 2]))
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert any("concurrently" in value for value in outcomes if isinstance(value, str))
    assert store.get([record["id"]], project_path=PROJECT)["memories"][0]["version"] == 2


def test_sql_parameters_are_bound(store):
    path = "/tmp/a' OR 1=1 --"
    tag = "x' OR 1=1 --"
    record = add(store, "sqlite", project_path=path, tags=[tag], metadata={"quote": "'"})
    assert store.list(project_path=path, tags=[tag])["memories"] == [record]
    assert store.list(tags=[tag], project_path=PROJECT)["memories"] == []
    with store.connection() as conn:
        assert json.loads(conn.execute("SELECT metadata FROM memories").fetchone()[0]) == {
            "quote": "'"
        }


def test_content_preserves_code_indentation(store):
    text = "    return result\n\n"
    memory = add(store, text)
    assert memory["content"] == text
    assert store.get([memory["id"]], project_path=PROJECT)["memories"][0]["content"] == text


def test_vector_scan_crosses_chunk_boundary_and_keeps_best(store):
    # 257 records exercise multiple scan chunks and replacement in the bounded top-k heap.
    for batch in range(2):
        store.create(
            [
                MemoryInput(content=f"ordinary filler {batch} {i}", project_path=PROJECT)
                for i in range(128)
            ]
        )
    target = add(store, "unique exact target")
    result = store.search(
        "unique exact target", fts_weight=0, limit=1, candidate_limit=1, project_path=PROJECT
    )
    assert result["results"][0]["id"] == target["id"]
    assert result["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_vacuum_preserves_fts_rowids(store):
    deleted = add(store, "deleted first row")
    kept = add(store, "retained unique searchable row")
    store.delete([deleted["id"]], project_path=PROJECT)
    with sqlite3.connect(store.path) as conn:
        conn.execute("VACUUM")
    hits = store.search("retained", vector_weight=0, project_path=PROJECT)["results"]
    assert [hit["id"] for hit in hits] == [kept["id"]]
    with store.connection(write=True) as conn:
        conn.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES ('integrity-check', 1)")


def test_project_path_required_without_implicit_fallback(store, embedder):
    for method, args in [(store.list, []), (store.search, ["query"])]:
        with pytest.raises(ValueError, match="project_path is required"):
            method(*args)
    with pytest.raises(ValueError, match="absolute"):
        add(store, project_path="relative/repo")
    assert store.stats()["total_memories"] == 0
    assert "current_project" not in store.stats()
    assert embedder.calls == []
    assert store.search("query", scope="all")["results"] == []
    assert store.list(scope="global")["total"] == 0


def test_id_operations_check_supplied_project(store):
    own = add(store)
    other = add(store, project_path="/work/other-project")
    global_ = add(store, project_path=None)
    ids = [own["id"], other["id"], global_["id"]]
    assert store.get(ids, project_path=PROJECT) == {
        "memories": [own],
        "missing_ids": [other["id"], global_["id"]],
    }
    with pytest.raises(ValueError, match="not found"):
        store.update(other["id"], MemoryPatch(content="wrong project"), project_path=PROJECT)
    assert store.delete(ids, project_path=PROJECT) == {
        "deleted_ids": [own["id"]],
        "missing_ids": [other["id"], global_["id"]],
    }
    assert store.get([other["id"]], project_path="/work/other-project")["memories"] == [other]
    assert store.get([global_["id"]], project_path=None)["memories"] == [global_]
    assert store.delete([global_["id"]], project_path=None)["deleted_ids"] == [global_["id"]]


def test_move_requires_old_project_and_preserves_content(store, embedder):
    record = add(store)
    moved = store.update(
        record["id"],
        MemoryPatch(project_path="/work/destination"),
        project_path=PROJECT,
    )
    assert moved["project_path"] == "/work/destination"
    assert moved["content"] == record["content"]
    assert len(embedder.calls) == 1
    assert store.get([record["id"]], project_path=PROJECT)["missing_ids"] == [record["id"]]
    assert store.get([record["id"]], project_path="/work/destination")["memories"] == [moved]
