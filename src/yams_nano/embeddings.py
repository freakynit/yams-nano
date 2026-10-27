"""Lazy, serialized Jina inference. No memory text leaves this process."""

import contextlib
import sys
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


def normalize_vectors(vectors, count: int, dimensions: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (count, dimensions) or not np.isfinite(array).all():
        raise ValueError("embedding model returned invalid shape or non-finite vectors")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise ValueError("embedding model returned zero or invalid vectors")
    return np.asarray(array / norms, dtype="<f4")


class JinaEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.settings.dimensions), dtype=np.float32)
        # The upstream loader and optional libraries may print. Keep stdout exclusively MCP.
        with self._lock, contextlib.redirect_stdout(sys.stderr):
            if self._model is None:
                from huggingface_hub import snapshot_download
                from sentence_transformers import SentenceTransformer

                # Upstream nested model/adapter loaders do not forward revision. Load a
                # complete pinned snapshot by local path so every file uses that revision.
                source = self.settings.model
                if not Path(source).is_dir():
                    source = snapshot_download(
                        repo_id=source,
                        revision=self.settings.model_revision,
                        cache_dir=str(self.settings.cache),
                    )
                self._model = SentenceTransformer(
                    source,
                    trust_remote_code=True,
                    device=self.settings.device,
                    cache_folder=str(self.settings.cache),
                    config_kwargs={"attn_implementation": "sdpa"},
                )
                self._model.max_seq_length = self.settings.max_tokens
            prompt = "query" if query else "document"
            # Refuse silent truncation: long memories should be split by the caller.
            prefixed = [self._model.prompts[prompt] + text for text in texts]
            tokens = self._model.tokenizer(prefixed, truncation=False, padding=False)["input_ids"]
            if any(len(ids) > self.settings.max_tokens for ids in tokens):
                raise ValueError(
                    f"text exceeds {self.settings.max_tokens} tokens including the prompt; "
                    "split it into smaller memories"
                )
            vectors = self._model.encode(
                texts,
                task="retrieval",
                prompt_name=prompt,
                batch_size=self.settings.embedding_batch_size,
                truncate_dim=self.settings.dimensions,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return normalize_vectors(vectors, len(texts), self.settings.dimensions)
