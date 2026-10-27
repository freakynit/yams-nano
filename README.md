# yams-nano

**Yet Another Memory System - Nano** version

1. Persistent local memory for AI coding agents, exposed through a stdio MCP server.
2. Stores past work, decisions, and known issues in one shared global **SQLite**, with project and global scopes.
3. Combines semantic search using local Jina embeddings with SQLite full-text search.
4. Runs without an embedding API, API key, or separate database service.
5. **No feature-bloat, messy setup, slowness, or heavy token usage.**

## Install

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and SQLite with FTS5 and JSON support.
From a local clone of this repository:

```sh
uv sync
uv run yams-nano
```

**The first embedding operation downloads the model; subsequent inference runs locally on CPU.**

## Connect your agent

Replace `/absolute/path/to/yams-nano` with your clone's location after running `uv sync`.
These commands configure the server for your user across projects:

```sh
# Codex
codex mcp add yams-nano -- /absolute/path/to/yams-nano/.venv/bin/yams-nano

# Claude Code
claude mcp add --scope user --transport stdio yams-nano -- /absolute/path/to/yams-nano/.venv/bin/yams-nano

# Gemini CLI
gemini mcp add --scope user --transport stdio yams-nano /absolute/path/to/yams-nano/.venv/bin/yams-nano
```

For Cursor, merge this into `~/.cursor/mcp.json` (user) or `.cursor/mcp.json` (project):

```json
{
  "mcpServers": {
    "yams-nano": {
      "command": "/absolute/path/to/yams-nano/.venv/bin/yams-nano"
    }
  }
}
```

## Add this to your AGENTS.md or CLAUDE.md (locally or globally)

When the `yams-nano` memory MCP server is available, search for relevant past work, decisions, and known issues before starting a task. Memory may be incomplete or outdated; verify relevant details against the current code and documentation. After making changes, save a concise summary of what changed, why, and any important findings or follow-up work to help future agents.

## Configuration and tools

Set `YAMS_HOME` to change the storage/cache directory, `YAMS_DB_PATH` for a separate database,
or `YAMS_DEVICE=mps` / `cuda` for a compatible accelerator. Read `memory_stats()` for active settings.

## Tools support
1. Batch creation
2. Reading and deletion
3. Partial updates
4. Project/tag filtering
5. Hybrid search, and standalone embeddings.

See the [reference](docs/reference.md) for tool usage,
search options, all environment variables, and backups.

## Development

```sh
uv sync --dev
uv run ruff check .
uv run pytest
uv build
```

Run `YAMS_TEST_MODEL=1 uv run pytest -m model -v` to test with the actual embedding model.

## License

The yams-nano source code is available under the [MIT License](LICENSE).

The default embedding model, [Jina `jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano/blob/8a7f00aac812071b69403df470f1038ec85f8925/README.md),
is licensed separately under **CC BY-NC 4.0**, which permits **non-commercial use only** under its terms.
Commercial use of the model requires a separate license from Jina. The MIT license for this
project does not grant rights to the model. Dependencies retain their respective licenses.
