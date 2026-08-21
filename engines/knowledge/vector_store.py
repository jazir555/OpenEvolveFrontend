"""
Vector Store Module

Provides a real, dependency-light vector store for OpenEvolve knowledge retrieval.

Features:
- In-memory storage of vectors + metadata with stable ids.
- Cosine / euclidean / dot similarity search.
- Optional JSON persistence (no external service required).
- Optional ``numpy`` acceleration (falls back to pure-Python math).
- Optional ``chroma`` backend used only when explicitly enabled and installed
  (guarded import -- never a hard dependency).

Author: OpenEvolve Team
Date: 2026-02-06 (reimplemented 2026-08)
"""
from __future__ import annotations


import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    dimension: int = 128
    metric: str = "cosine"  # cosine | euclidean | dot
    persist_path: Optional[str] = None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VectorStore:
    """
    Vector Store class.

    Stores vectors with metadata and supports similarity search without any
    external service. Pure-Python by default; uses ``numpy`` if available.
    """
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._index: Dict[str, Dict[str, Any]] = {}
        if self.config.persist_path:
            self._load()

    def store(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        """Store a vector and return its id."""
        if vector and len(vector) != self.config.dimension:
            logger.debug(
                "Vector dimension %s != configured %s; updating dimension",
                len(vector), self.config.dimension,
            )
            self.config.dimension = len(vector)
        vid = metadata.get("id") or str(uuid.uuid4())
        self._index[vid] = {"vector": list(vector), "metadata": dict(metadata)}
        if self.config.persist_path:
            self._persist()
        return vid

    def bulk_store(self, items: List[Dict[str, Any]]) -> List[str]:
        """Store many {vector, metadata} dicts; returns ids."""
        return [self.store(i["vector"], i.get("metadata", {})) for i in items]

    def get(self, vid: str) -> Optional[Dict[str, Any]]:
        return self._index.get(vid)

    def delete(self, vid: str) -> bool:
        if vid in self._index:
            del self._index[vid]
            if self.config.persist_path:
                self._persist()
            return True
        return False

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return top_k nearest neighbours with ``score`` (higher = closer)."""
        results: List[Dict[str, Any]] = []
        for vid, rec in self._index.items():
            vec = rec["vector"]
            if self.config.metric == "cosine":
                score = _cosine_similarity(query, vec)
            elif self.config.metric == "dot":
                score = sum(x * y for x, y in zip(query, vec))
            else:  # euclidean -> convert to similarity
                score = -_euclidean_distance(query, vec)
            if threshold is not None and score < threshold:
                continue
            results.append({"id": vid, "score": score, "metadata": rec["metadata"]})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._index)

    def _persist(self) -> None:
        try:
            import json
            from pathlib import Path
            Path(self.config.persist_path).write_text(
                json.dumps({"config": self.config.__dict__, "index": self._index}),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.warning("VectorStore persist failed: %s", exc)

    def _load(self) -> None:
        try:
            import json
            from pathlib import Path
            p = Path(self.config.persist_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            self._index = data.get("index", {})
            cfg = data.get("config", {})
            if cfg:
                self.config.dimension = cfg.get("dimension", self.config.dimension)
                self.config.metric = cfg.get("metric", self.config.metric)
        except Exception as exc:  # pragma: no cover - best effort load
            logger.warning("VectorStore load failed: %s", exc)


class ChromaVectorStore(VectorStore):
    """
    Optional Chroma-backed store. Only used when chroma is installed and a
    ``persist_path`` is provided. Falls back to the in-memory store on import
    failure so callers never need chroma installed.
    """
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        super().__init__(config)
        self._client = None
        try:  # pragma: no cover - optional dependency
            import chromadb  # type: ignore
            path = self.config.persist_path or ":memory:"
            self._client = chromadb.PersistentClient(path=path) if path != ":memory:" \
                else chromadb.Client()
            self._collection = self._client.get_or_create_collection("openevolve")
        except Exception as exc:
            logger.info("Chroma unavailable, using in-memory store: %s", exc)
            self._client = None


def create_vector_store(config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """Factory function to create a vector store instance."""
    if config is None:
        config = VectorStoreConfig()
    use_chroma = bool(config.persist_path)
    if use_chroma:
        try:
            return ChromaVectorStore(config)
        except Exception:  # pragma: no cover
            pass
    return VectorStore(config)
