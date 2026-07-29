from __future__ import annotations

import hashlib
import math
from typing import Iterable, Protocol


class EmbeddingProvider(Protocol):
    """Interface for replacing deterministic local embeddings with real LLM embeddings."""

    def embed(self, text: str) -> list[float]:
        """Return a dense vector for text."""


class HashEmbeddingProvider:
    """Deterministic offline text embedding fallback.

    This is not a real LLM embedding model. It provides a stable vectorization
    interface so the recommender can be tested without network or API keys.
    Replace this provider with a real embedding client for production or paper
    experiments that claim LLM semantic embeddings.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text_features(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


def text_features(text: str) -> Iterable[str]:
    normalized = text.lower()
    for token in normalized.replace("；", " ").replace("，", " ").replace(",", " ").split():
        yield token
    chars = [char for char in normalized if not char.isspace()]
    for size in (1, 2, 3):
        for index in range(0, max(len(chars) - size + 1, 0)):
            yield "".join(chars[index : index + size])
