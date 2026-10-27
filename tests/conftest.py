import pytest

from yams_nano.config import Settings
from yams_nano.store import MemoryStore

from .fakes import FakeEmbedder


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path / "data", dimensions=32)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(settings, embedder):
    return MemoryStore(settings, embedder)
