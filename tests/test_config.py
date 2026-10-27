import os
import stat

import pytest
from pydantic import ValidationError

from yams_nano.config import REVISION, Settings, project_root
from yams_nano.models import MemoryInput, MemoryPatch
from yams_nano.store import MemoryStore

from .fakes import FakeEmbedder


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("YAMS_HOME", str(tmp_path / "central"))
    monkeypatch.setenv("YAMS_DB_PATH", str(tmp_path / "custom.db"))
    monkeypatch.setenv("YAMS_DEVICE", "mps")
    monkeypatch.setenv("YAMS_DIMENSIONS", "256")
    monkeypatch.setenv("YAMS_SEARCH_LIMIT", "5")
    config = Settings.from_env()
    assert config.database == tmp_path / "custom.db"
    assert config.cache == tmp_path / "central" / "models"
    assert config.dimensions == 256 and config.search_limit == 5 and config.device == "mps"
    assert config.model_revision == REVISION


def test_default_location(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = Settings()
    assert config.database == tmp_path / ".yams-nano" / "memory.sqlite3"


def test_project_is_not_configured_or_inferred(monkeypatch, tmp_path):
    monkeypatch.setenv("YAMS_PROJECT_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert "project_path" not in Settings.from_env().model_dump()
    assert project_root(tmp_path / "repo") == str(tmp_path / "repo")
    with pytest.raises(ValueError, match="absolute"):
        project_root("relative/repo")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimensions": 42},
        {"embedding_batch_size": 0},
        {"max_tokens": 9000},
        {"candidate_limit": 1},
        {"vector_weight": 0, "fts_weight": 0},
        {"log_level": "invalid"},
        {"vector_weight": float("inf")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_mismatched_embedding_identity_is_rejected(store, settings):
    different = Settings(**{**settings.model_dump(), "dimensions": 64})
    with pytest.raises(ValueError, match="embedding configuration differs"):
        MemoryStore(different, FakeEmbedder(64))


def test_future_schema_rejected(store, settings):
    with store.connection(write=True) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema version"):
        MemoryStore(settings, FakeEmbedder())


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_private_database_permissions(store):
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": " "},
        {"content": "x", "tags": [""]},
        {"content": "x", "metadata": {"invalid": float("nan")}},
        {"content": "x", "metadata": {"huge": "a" * 20000}},
        {"content": "x", "unexpected": "field"},
    ],
)
def test_invalid_memory_inputs(kwargs):
    with pytest.raises(ValidationError):
        MemoryInput(project_path=None, **kwargs)


@pytest.mark.parametrize("field", ["content", "title", "tags", "metadata"])
def test_patch_null_rejected_except_project(field):
    with pytest.raises(ValidationError):
        MemoryPatch(**{field: None})
    assert MemoryPatch(project_path=None).model_fields_set == {"project_path"}


def test_insertion_requires_explicit_project():
    with pytest.raises(ValidationError, match="project_path"):
        MemoryInput(content="no implicit project")
    assert MemoryInput(content="explicit global", project_path=None).project_path is None
