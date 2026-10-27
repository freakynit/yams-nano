import asyncio
import json
import os
import select
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .fakes import PROJECT

FAKE_SERVER = (
    "from yams_nano.config import Settings; "
    "from yams_nano.server import create_server; "
    "from tests.fakes import FakeEmbedder; "
    "s=Settings.from_env(); "
    "create_server(s, FakeEmbedder(s.dimensions)).run(transport='stdio')"
)


def server_env(tmp_path, **overrides):
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith("YAMS_")},
        "YAMS_HOME": str(tmp_path / "data"),
        "YAMS_DIMENSIONS": "32",
        **overrides,
    }


@asynccontextmanager
async def session_for(tmp_path, *, fake=True, **overrides):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", FAKE_SERVER] if fake else ["-m", "yams_nano"],
        env=server_env(tmp_path, **overrides),
    )
    with (tmp_path / "stderr.log").open("a") as stderr:
        async with stdio_client(params, errlog=stderr) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=600),
            ) as session:
                await session.initialize()
                yield session


async def call(session, name, arguments=None):
    result = await session.call_tool(name, arguments or {})
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


def test_full_stdio_lifecycle(tmp_path):
    async def run():
        async with session_for(tmp_path) as session:
            discovered = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(discovered) == {
                "add_memories",
                "get_memories",
                "update_memory",
                "delete_memories",
                "list_memories",
                "search_memories",
                "embed_texts",
                "memory_stats",
            }
            assert discovered["search_memories"].annotations.readOnlyHint
            assert discovered["delete_memories"].annotations.destructiveHint
            for name in ("get_memories", "update_memory", "delete_memories"):
                assert "project_path" in discovered[name].inputSchema["required"]
            stats = await call(session, "memory_stats")
            assert stats["total_memories"] == 0
            assert "current_project" not in stats
            for name, args in [
                ("add_memories", {"memories": [{"content": "missing project"}]}),
                ("get_memories", {"ids": ["id"]}),
                ("update_memory", {"id": "id", "changes": {"title": "missing project"}}),
                ("delete_memories", {"ids": ["id"]}),
                ("search_memories", {"query": "missing project"}),
                ("list_memories", {}),
            ]:
                assert (await session.call_tool(name, args)).isError
            added = await call(
                session,
                "add_memories",
                {
                    "memories": [
                        {"content": "sqlite transaction", "project_path": PROJECT, "tags": ["db"]},
                        {"content": "pytest workflow", "project_path": ""},
                    ]
                },
            )
            id_ = added["memories"][0]["id"]
            found = await call(session, "get_memories", {"project_path": PROJECT, "ids": [id_]})
            assert found["memories"][0]["content"] == "sqlite transaction"
            search = await call(
                session,
                "search_memories",
                {"project_path": PROJECT, "query": "sqlite", "tags": ["db"]},
            )
            assert search["results"][0]["id"] == id_
            changed = await call(
                session,
                "update_memory",
                {
                    "project_path": PROJECT,
                    "id": id_,
                    "changes": {"content": "redis pipeline"},
                    "expected_version": 1,
                },
            )
            assert changed["version"] == 2
            stale = await session.call_tool(
                "update_memory",
                {
                    "project_path": PROJECT,
                    "id": id_,
                    "changes": {"title": "stale"},
                    "expected_version": 1,
                },
            )
            assert stale.isError
            invalid = await session.call_tool(
                "add_memories", {"memories": [{"content": " ", "project_path": PROJECT}]}
            )
            assert invalid.isError
            listed = await call(session, "list_memories", {"scope": "global"})
            assert len(listed["memories"]) == 1
            embedded = await call(session, "embed_texts", {"texts": ["one", "two"], "query": True})
            assert len(embedded["embeddings"]) == 2
            await call(session, "delete_memories", {"project_path": PROJECT, "ids": [id_]})
            assert (await call(session, "memory_stats"))["total_memories"] == 1
        async with session_for(tmp_path) as reopened:
            assert (await call(reopened, "memory_stats"))["total_memories"] == 1

    asyncio.run(run())


def test_stdout_is_only_jsonrpc_and_real_startup_is_lazy(tmp_path):
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory_stats",
                "arguments": {},
            },
        },
    ]
    process = subprocess.Popen(
        [sys.executable, "-m", "yams_nano"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=server_env(tmp_path),
    )
    parsed = []
    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            if "id" in message:
                assert select.select([process.stdout], [], [], 30)[0], "no MCP response"
                parsed.append(json.loads(process.stdout.readline()))
        process.stdin.close()
        assert process.wait(timeout=30) == 0, process.stderr.read()
        parsed.extend(json.loads(line) for line in process.stdout)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    assert {message.get("id") for message in parsed} >= {1, 2}
    assert all(message["jsonrpc"] == "2.0" for message in parsed)
    assert not (tmp_path / "data" / "models").exists()


def test_two_server_processes_share_database(tmp_path):
    async def run():
        async with session_for(tmp_path) as first, session_for(tmp_path) as second:
            batches = await asyncio.gather(
                *[
                    call(
                        session,
                        "add_memories",
                        {"memories": [{"content": f"item {i}", "project_path": PROJECT}]},
                    )
                    for i, session in enumerate([first, second, first, second])
                ]
            )
            assert (await call(second, "memory_stats"))["total_memories"] == 4
            id_ = batches[0]["memories"][0]["id"]
            await call(
                first,
                "update_memory",
                {"project_path": PROJECT, "id": id_, "changes": {"title": "shared"}},
            )
            record = (await call(second, "get_memories", {"project_path": PROJECT, "ids": [id_]}))[
                "memories"
            ][0]
            assert record["title"] == "shared" and record["version"] == 2

    asyncio.run(run())
