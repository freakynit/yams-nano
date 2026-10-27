# Reference

See the [README](../README.md) for installation and agent setup.

## Tools

| Tool | Purpose |
| --- | --- |
| `add_memories(memories)` | Atomically embed/insert a list of memory objects |
| `get_memories(ids, project_path)` | Read a batch within a project, reporting missing IDs |
| `update_memory(id, changes, project_path, expected_version?)` | Scoped partial update with concurrency detection |
| `delete_memories(ids, project_path)` | Atomic, permanent batch delete within a project |
| `list_memories(...)` | Paginated listing, newest first, with total and next offset |
| `search_memories(query, ...)` | Tunable hybrid, lexical-only, or vector-only search |
| `embed_texts(texts, query=false)` | Embed a batch without saving it |
| `memory_stats()` | Paths, counts, embedding identity, defaults, and limits |

Example `add_memories` arguments:

```json
{
  "memories": [
    {
      "content": "Use uv for dependencies. Run uv run pytest before submitting changes.",
      "title": "Development workflow",
      "project_path": "/work/payments",
      "tags": ["workflow", "testing"],
      "metadata": {"source": "AGENTS.md", "kind": "convention"}
    },
    {
      "content": "Prefer small, independently reviewable commits.",
      "project_path": null,
      "tags": ["preference"]
    }
  ]
}
```

Each inserted memory requires `content` and `project_path`. Supply an absolute repository
path, or explicitly use null/`""` for a global memory (stored as SQL NULL). All other fields
are optional. Content is preserved verbatim, including code indentation.

ID reads, updates, and deletes require a top-level `project_path` selecting the memory's
current project; null/`""` selects only global memories. IDs in a different project are
treated as missing, and cannot be modified by that call. To move a memory, provide the
old location as the top-level `project_path` and the new one as `changes.project_path`.
Set `changes.project_path` to null/empty to move it to global scope. For example:

```json
{
  "id": "memory-uuid",
  "project_path": "/work/payments",
  "changes": {"project_path": "/work/billing"},
  "expected_version": 1
}
```

Omit an update field
to preserve it; explicit null is invalid for other fields. Tags and metadata are replaced,
not merged. `[]`/`{}` clear them. Only content changes trigger re-embedding. Vectors index
content; lexical search also indexes title and tags. Duplicate content is intentionally
allowed; UUIDs identify memories. A failed batch inserts nothing. Batch results preserve
input order. Repeated deletes succeed with missing IDs reported.

Every memory returns `id`, `content`, `title`, `project_path`, `tags`, `metadata`,
`inserted_at`, `updated_at`, and `version`. Times are UTC ISO 8601 with microseconds.
`version` increases on every update. Pass the last version as `expected_version` to avoid
overwriting another agent's work. No field silently expires or deletes memories.

## Search

```json
{
  "query": "how do I run the tests?",
  "project_path": "/work/payments",
  "limit": 10,
  "candidate_limit": 50,
  "rrf_k": 60,
  "vector_weight": 1.0,
  "fts_weight": 1.0
}
```

Default scope is `project`, which requires an explicitly supplied absolute `project_path`
and includes global memories. Set `include_global=false` for only that project,
`scope="global"` for only global memories, or `scope="all"` for all repositories.
The latter two scopes do not require a project path. These rules also apply to `list_memories`.
All supplied `tags` must match exactly, case sensitively. Filters apply before ranking to
both branches, so unrelated projects cannot crowd out relevant candidates.

FTS safely turns ordinary query text into quoted OR tokens, up to 64 unique words, handling
code paths and punctuation without interpreting them as operators. Unicode61 tokenization
splits identifiers at underscores/punctuation. For advanced FTS5 syntax supply `fts_query`,
e.g. `"\"connection pool\" OR tags:sqlite"`. Malformed expressions return a tool error.
Unicode diacritics are folded by FTS; there is no stemming. Title/content/tags BM25 weights
are 2.0/1.0/1.5. Lower raw BM25 is better.

Vectors are normalized float32 blobs in SQLite. Exact cosine search scans the filtered
namespace in chunks of 256 vectors, keeping only the top candidates in memory. This has
linear search cost and suits personal agent memory collections; it is not an ANN index.

Each branch retrieves up to `candidate_limit` items (default 50), then:

```text
score(memory) = vector_weight / (rrf_k + vector_rank)
              + fts_weight / (rrf_k + fts_rank)
```

Ranks start at 1; a missing branch contributes zero. Higher fused scores win. Returned
`score`, `vector_rank`, `fts_rank`, `similarity`, and `bm25` explain each result; absent branch
values are null. RRF scores are not confidence probabilities. Ties have stable ID ordering.

Set `vector_weight=0` for lexical-only search without loading the model; `fts_weight=0` for
semantic-only search. Both cannot be zero. `min_similarity` optionally filters the vector
branch only; lexical matches can still appear below that threshold. Increase candidate pools
for recall or change weights to favor exact identifiers versus semantic matches. `limit`
is 1..100, candidates 1..1000 and at least `limit`; increasing only `limit` expands the
default candidate pool as needed. The response includes effective parameters.

## Environment variables

All configuration is read once at startup. MCP call parameters override search defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `YAMS_HOME` | `~/.yams-nano` | Storage/cache base directory |
| `YAMS_DB_PATH` | `$YAMS_HOME/memory.sqlite3` | SQLite file; independently override storage |
| `YAMS_MODEL` | `jinaai/jina-embeddings-v5-text-nano` | Compatible Jina model ID/local path |
| `YAMS_MODEL_REVISION` | `8a7f00aac812071b69403df470f1038ec85f8925` | Hugging Face model/code revision |
| `YAMS_MODEL_CACHE` | `$YAMS_HOME/models` | Model download directory |
| `YAMS_DEVICE` | `cpu` | PyTorch device (`cpu`, `mps`, `cuda`, etc.) |
| `YAMS_DIMENSIONS` | `768` | 32, 64, 128, 256, 512, or 768 |
| `YAMS_MAX_TOKENS` | `8192` | Input token ceiling, including prompt/special tokens |
| `YAMS_EMBEDDING_BATCH_SIZE` | `16` | Model inference batch size |
| `YAMS_MAX_BATCH_SIZE` | `128` | Maximum texts/IDs/memories per call; up to 1000 |
| `YAMS_MAX_TEXT_CHARS` | `50000` | Per-memory/query character ceiling |
| `YAMS_SEARCH_LIMIT` | `10` | Default number of search results |
| `YAMS_CANDIDATE_LIMIT` | `50` | Candidates from each branch |
| `YAMS_RRF_K` | `60` | RRF smoothing constant |
| `YAMS_VECTOR_WEIGHT` | `1.0` | Semantic rank contribution |
| `YAMS_FTS_WEIGHT` | `1.0` | Lexical rank contribution |
| `YAMS_BUSY_TIMEOUT_MS` | `10000` | SQLite lock wait timeout |
| `YAMS_LOG_LEVEL` | `WARNING` | MCP log level; output goes to stderr |

Hugging Face also honors its usual environment variables, including `HF_TOKEN` (if needed)
and `HF_HUB_OFFLINE=1` for a previously cached model. Lower inference batch size to reduce
memory use. Oversized content is rejected, including inputs over the configured token limit;
split large documents into focused memories rather than silently truncating them.

The database records the model/revision/dimensions/token-limit/prompt identity and refuses to
mix incompatible vectors. Restore the original settings or choose a new `YAMS_DB_PATH` when
changing the embedding space. For reproducibility, use commit hashes for revisions.

## Storage and backups

Multiple agent processes can use the same database. Model computation happens before write
transactions. SQLite WAL and busy timeouts coordinate short writes; compare-and-swap versions
prevent concurrent updates from silently overwriting one another. Search branches share one
read snapshot. FTS insert/update/delete triggers run in the same transaction as memory changes.
New database files and their immediate parent directory are created with owner-only
permissions on POSIX. Existing directory/file permissions are preserved.

For a consistent live backup, use SQLite's backup API rather than copying only the main file
while WAL writes are active:

```sh
python - <<'PY'
import sqlite3
from pathlib import Path
with sqlite3.connect(Path.home() / '.yams-nano/memory.sqlite3') as source:
    with sqlite3.connect('memory-backup.sqlite3') as backup:
        source.backup(backup)
PY
```

The shared store is unencrypted and scope filters are a convenience, not access controls.
Agents can explicitly select any project's path; there is no authentication boundary. Do not store secrets;
treat retrieved content as data. This server does not crawl repositories or ingest files.

