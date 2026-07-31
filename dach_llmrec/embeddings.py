from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Protocol


DEFAULT_REAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_CACHE_DIR = Path("artifacts/embedding_cache")


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
        return _normalize(vec)


class SentenceTransformerEmbeddingProvider:
    """Local real embedding provider backed by sentence-transformers."""

    provider_name = "sentence_transformer"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_REAL_EMBEDDING_MODEL,
        device: str = "auto",
        cache_dir: str | Path | None = DEFAULT_EMBEDDING_CACHE_DIR,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = _select_embedding_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normalize_embeddings = normalize_embeddings
        self._memory_cache: dict[str, list[float]] = {}
        self._model = model or self._load_model()

    def embed(self, text: str) -> list[float]:
        cache_key = self._cache_key(text)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        disk_cached = self._read_disk_cache(cache_key)
        if disk_cached is not None:
            self._memory_cache[cache_key] = disk_cached
            return disk_cached

        vector = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        values = [float(value) for value in vector.tolist()]
        if self.normalize_embeddings:
            values = _normalize(values)
        self._memory_cache[cache_key] = values
        self._write_disk_cache(cache_key, values)
        return values

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Could not import sentence-transformers for embedding_provider='real': "
                f"{exc}. Install or repair the optional embedding dependency, "
                "or use embedding_provider='hash'."
            ) from exc
        return SentenceTransformer(self.model_name_or_path, device=self.device)

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = "|".join(
            [
                self.provider_name,
                self.model_name_or_path,
                str(self.normalize_embeddings),
                text_hash,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def _read_disk_cache(self, cache_key: str) -> list[float] | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [float(value) for value in payload["embedding"]]

    def _write_disk_cache(self, cache_key: str, values: list[float]) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.provider_name,
            "model": self.model_name_or_path,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding": values,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_embedding_provider(
    embedding_provider: str = "hash",
    embedding_model: str = DEFAULT_REAL_EMBEDDING_MODEL,
    embedding_device: str = "auto",
    embedding_cache_dir: str | Path | None = DEFAULT_EMBEDDING_CACHE_DIR,
) -> EmbeddingProvider:
    if embedding_provider == "hash":
        return HashEmbeddingProvider()
    if embedding_provider == "real":
        return SentenceTransformerEmbeddingProvider(
            model_name_or_path=embedding_model,
            device=embedding_device,
            cache_dir=embedding_cache_dir,
        )
    raise ValueError("embedding_provider must be 'hash' or 'real'")


def text_features(text: str) -> Iterable[str]:
    normalized = text.lower()
    for token in normalized.replace(";", " ").replace(",", " ").split():
        yield token
    chars = [char for char in normalized if not char.isspace()]
    for size in (1, 2, 3):
        for index in range(0, max(len(chars) - size + 1, 0)):
            yield "".join(chars[index : index + size])


def _select_embedding_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("embedding_device must be 'auto', 'cpu', or 'cuda'")
    return device


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
