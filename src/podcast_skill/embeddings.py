"""Embedding providers for the index step.

An embedder turns texts into L2-normalized float32 vectors, so a dot product is the cosine
similarity. Only OpenAI is implemented; another provider only has to follow `Embedder`.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from podcast_skill.config import section

DEFAULT_MODEL = "text-embedding-3-large"   # text-embedding-3-small is cheaper, good for testing
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class Embedder(Protocol):
    model: str
    dimensions: Optional[int]   # None = the model's native size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalized."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        batch_size: int = 96,
        client: Any = None,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._api_key_env = api_key_env
        self._client = client

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self._api_key_env} is not set. "
                    f"Export your OpenAI API key first: export {self._api_key_env}=sk-..."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, max_retries=5)
        return self._client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self.model}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = client.embeddings.create(input=list(texts[start:start + self.batch_size]), **kwargs)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return normalize(np.array(vectors, dtype=np.float32))


def build_embedder(config: dict[str, Any], model: Optional[str] = None) -> Embedder:
    cfg = section(config, "index")
    return OpenAIEmbedder(
        model=model or cfg.get("embedding_model") or DEFAULT_MODEL,
        dimensions=cfg.get("dimensions") or None,
        api_key_env=cfg.get("api_key_env") or DEFAULT_API_KEY_ENV,
        batch_size=int(cfg.get("batch_size") or 96),
    )
