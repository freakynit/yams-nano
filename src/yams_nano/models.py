"""Validated input types shared by the MCP interface and storage layer."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Scope = Literal["project", "all", "global"]


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    title: str = Field(default="", max_length=500)
    project_path: str | None = Field(
        description="Absolute repository path; explicitly null or empty for a global memory."
    )
    tags: list[str] = Field(default_factory=list, max_length=64)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value.strip()) > 100 for value in values):
            raise ValueError("tags must contain 1 to 100 characters")
        return list(dict.fromkeys(value.strip() for value in values))

    @field_validator("metadata")
    @classmethod
    def json_metadata(cls, value: dict) -> dict:
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded) > 16384:
            raise ValueError("metadata must be at most 16384 JSON characters")
        return value


class MemoryPatch(MemoryInput):
    project_path: str | None = Field(
        default=None,
        description="New project path; omit to preserve it, null/empty moves to global scope.",
    )
    content: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = Field(default=None, max_length=64)
    metadata: dict[str, JsonValue] | None = None

    @field_validator("content", "title", "tags", "metadata", mode="before")
    @classmethod
    def disallow_null(cls, value):
        if value is None:
            raise ValueError(
                "omit fields to keep them unchanged; null is only allowed for project_path"
            )
        return value


class SearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(ge=1, le=100)
    candidate_limit: int = Field(ge=1, le=1000)
    rrf_k: int = Field(ge=1, le=100000)
    vector_weight: float = Field(ge=0, le=100, allow_inf_nan=False)
    fts_weight: float = Field(ge=0, le=100, allow_inf_nan=False)
    min_similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
