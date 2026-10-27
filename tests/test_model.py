import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from .fakes import PROJECT
from .test_stdio import call, session_for

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("YAMS_TEST_MODEL") != "1",
        reason="set YAMS_TEST_MODEL=1 for real inference",
    ),
]


@pytest.mark.parametrize("dimensions", [32, 768])
def test_real_jina_batch_search_update_and_stdio(tmp_path, caplog, dimensions):
    # Reuse only the model cache between runs; all memory storage remains temporary.
    model_cache = os.environ.get(
        "YAMS_TEST_MODEL_CACHE", str(Path.home() / ".cache/yams-nano-tests")
    )
    texts = [
        "Never store passwords in plaintext. Hash credentials using argon2 before saving them.",
        "Run the Python test suite with uv run pytest before submitting a patch.",
        "Use SQLite WAL mode so readers can continue while another connection writes.",
    ]

    async def run():
        async with session_for(
            tmp_path,
            fake=False,
            YAMS_DIMENSIONS=str(dimensions),
            YAMS_MODEL_CACHE=model_cache,
        ) as session:
            vectors = await call(session, "embed_texts", {"texts": texts})
            array = np.asarray(vectors["embeddings"])
            assert array.shape == (3, dimensions) and np.isfinite(array).all()
            assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5)
            assert not np.allclose(array[0], array[1])
            added = (
                await call(
                    session,
                    "add_memories",
                    {
                        "memories": [
                            {"content": text, "project_path": PROJECT, "tags": ["real"]}
                            for text in texts
                        ]
                    },
                )
            )["memories"]
            too_long = await session.call_tool(
                "add_memories",
                {
                    "memories": [{"content": "word " * 9000, "project_path": PROJECT}],
                },
            )
            assert too_long.isError and "tokens" in too_long.content[0].text
            assert (await call(session, "memory_stats"))["total_memories"] == 3
            semantic = await call(
                session,
                "search_memories",
                {
                    "project_path": PROJECT,
                    "query": "How do we securely persist user authentication secrets?",
                    "fts_weight": 0,
                },
            )
            assert semantic["results"][0]["id"] == added[0]["id"]
            hybrid = await call(
                session,
                "search_memories",
                {"project_path": PROJECT, "query": "pytest Python tests"},
            )
            assert hybrid["results"][0]["id"] == added[1]["id"]
            id_ = added[1]["id"]
            await call(
                session,
                "update_memory",
                {
                    "project_path": PROJECT,
                    "id": id_,
                    "changes": {"content": "Use ruff check to lint the Python source code."},
                },
            )
            assert (
                await call(
                    session,
                    "search_memories",
                    {
                        "project_path": PROJECT,
                        "query": "pytest",
                        "vector_weight": 0,
                    },
                )
            )["results"] == []
            assert (
                await call(
                    session,
                    "search_memories",
                    {
                        "project_path": PROJECT,
                        "query": "ruff",
                        "vector_weight": 0,
                    },
                )
            )["results"][0]["id"] == id_
            await call(
                session, "delete_memories", {"project_path": PROJECT, "ids": [added[2]["id"]]}
            )
        async with session_for(
            tmp_path,
            fake=False,
            YAMS_DIMENSIONS=str(dimensions),
            YAMS_MODEL_CACHE=model_cache,
            HF_HUB_OFFLINE="1",
        ) as reopened:
            assert (await call(reopened, "memory_stats"))["total_memories"] == 2
            assert (await call(reopened, "get_memories", {"project_path": PROJECT, "ids": [id_]}))[
                "memories"
            ][0]["version"] == 2
            offline = await call(reopened, "embed_texts", {"texts": ["offline cached model"]})
            assert len(offline["embeddings"][0]) == dimensions

    with caplog.at_level(logging.ERROR):
        asyncio.run(run())
    assert not any("parse" in record.message.lower() for record in caplog.records)
