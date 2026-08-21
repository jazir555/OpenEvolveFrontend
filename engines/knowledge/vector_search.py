"""
Vector Search Module

Provides a real, dependency-light vector search interface for OpenEvolve built
on top of :mod:`vector_store`. Supports metadata filtering, thresholding and
pluggable embedding providers so it can be used for semantic retrieval without
any external service.

Author: OpenEvolve Team
Date: 2026-02-06 (reimplemented 2026-08)
"""
from __future__ import annotations


import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from vector_store import VectorStore, VectorStoreConfig, create_vector_store

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchConfig:
    """Configuration for vector search."""
    threshold: float = 0.0
    max_results: int = 100
    store: Optional[VectorStore] = None
    embedding_provider: Optional[Any] = None


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class HashEmbeddingProvider:
    """
    Deterministic, dependency-free embedding provider.

    Hashes tokens into a fixed-dimensional bag-of-words vector. Good enough for
    lexical/semantic-ish retrieval in tests and offline scenarios. Replace with a
    real model (see ``TransformerEmbeddingProvider``) for production quality.
    """
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        if not text:
            return vec
        for tok in _TOKEN_RE.findall(text.lower()):
            h = hash(tok) % self.dimension
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec


class TransformerEmbeddingProvider:
    """
    Optional transformer-backed embedding provider (clearly-marked optional
    import). Used only when ``transformers``/``sentence-transformers`` is
    installed; otherwise constructing one raises and callers fall back to
    :class:`HashEmbeddingProvider`.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        except Exception:
            try:  # pragma: no cover - optional dependency
                from transformers import AutoModel, AutoTokenizer  # type: ignore
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                self.dimension = self._model.config.hidden_size
            except Exception as exc:
                raise RuntimeError(
                    "transformers/sentence-transformers not available"
                ) from exc

    def embed(self, text: str) -> List[float]:  # pragma: no cover - optional dep
        return self._model.encode(text, normalize_embeddings=True).tolist()


class VectorSearch:
    """Vector Search class wrapping a :class:`VectorStore` with filtering."""
    def __init__(self, config: Optional[VectorSearchConfig] = None):
        self.config = config or VectorSearchConfig()
        if self.config.store is None:
            self.config.store = create_vector_store(VectorStoreConfig())
        if self.config.embedding_provider is None:
            self.config.embedding_provider = HashEmbeddingProvider()
        logger.info("Vector Search initialized (threshold=%.2f)", self.config.threshold)

    def _embed(self, text: str) -> List[float]:
        if isinstance(text, list):
            return text  # already a vector
        return self.config.embedding_provider.embed(text)

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any],
                         filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, val in filters.items():
            if metadata.get(key) != val:
                return False
        return True

    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        vec = self._embed(text)
        metadata = dict(metadata)
        metadata.setdefault("text", text)
        return self.config.store.store(vec, metadata)

    def search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search by text/vector, applying filters and threshold."""
        qvec = self._embed(query)
        raw = self.config.store.search(
            qvec, top_k=self.config.max_results, threshold=self.config.threshold
        )
        return [r for r in raw if self._matches_filters(r["metadata"], filters)]

    def find_similar(self, vector: List[float], threshold: float = None) -> List[Dict[str, Any]]:
        """Find similar vectors to an explicit query vector."""
        thr = self.config.threshold if threshold is None else threshold
        return self.config.store.search(vector, top_k=self.config.max_results, threshold=thr)


def create_vector_search(config: Optional[VectorSearchConfig] = None) -> VectorSearch:
    """Factory function to create a vector search instance."""
    return VectorSearch(config)
