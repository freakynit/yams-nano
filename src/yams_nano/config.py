"""Environment configuration, captured once at process startup."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODEL = "jinaai/jina-embeddings-v5-text-nano"
REVISION = "8a7f00aac812071b69403df470f1038ec85f8925"


def project_root(path: str | Path) -> str:
    """Canonicalize an explicitly supplied absolute project location."""
    location = Path(path).expanduser()
    if not location.is_absolute():
        raise ValueError("project_path must be an absolute path supplied by the agent")
    return str(location.resolve())


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    home: Path = Path("~/.yams-nano")
    db_path: Path | None = None
    model: str = MODEL
    model_revision: str = REVISION
    model_cache: Path | None = None
    device: str = "cpu"
    dimensions: int = 768
    max_tokens: int = Field(default=8192, ge=16, le=8192)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    max_batch_size: int = Field(default=128, ge=1, le=1000)
    max_text_chars: int = Field(default=50000, ge=1)
    search_limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=100000)
    vector_weight: float = Field(default=1.0, ge=0, le=100, allow_inf_nan=False)
    fts_weight: float = Field(default=1.0, ge=0, le=100, allow_inf_nan=False)
    busy_timeout_ms: int = Field(default=10000, ge=1, le=300000)
    log_level: str = "WARNING"

    @field_validator("dimensions")
    @classmethod
    def supported_dimensions(cls, value: int) -> int:
        if value not in (32, 64, 128, 256, 512, 768):
            raise ValueError("dimensions must be one of 32, 64, 128, 256, 512, 768")
        return value

    @model_validator(mode="after")
    def validate_search(self) -> "Settings":
        if self.candidate_limit < self.search_limit:
            raise ValueError("candidate_limit must be at least search_limit")
        if self.vector_weight + self.fts_weight == 0:
            raise ValueError("at least one search weight must be positive")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("invalid log_level")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.model_validate(
            {
                name: os.environ[f"YAMS_{name.upper()}"]
                for name in cls.model_fields
                if f"YAMS_{name.upper()}" in os.environ
            }
        )

    @property
    def database(self) -> Path:
        return (self.db_path or self.home / "memory.sqlite3").expanduser().resolve()

    @property
    def cache(self) -> Path:
        return (self.model_cache or self.home / "models").expanduser().resolve()

    @property
    def embedding_identity(self) -> dict:
        # Device and batch size do not change the embedding space.
        return {
            "model": self.model,
            "revision": self.model_revision,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "task": "retrieval",
            "document_prompt": "document",
            "query_prompt": "query",
        }
