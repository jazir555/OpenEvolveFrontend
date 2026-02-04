"""Evolution memory system for LoongFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterable
import uuid

from .base_memory import BaseMemory


@dataclass
class Solution:
    """Lightweight solution record."""
    content: Any
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class InMemory(BaseMemory):
    """In-memory storage for solutions."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def all(self) -> Iterable[Any]:
        return self._store.values()


class RedisMemory(InMemory):
    """Redis-backed memory placeholder (uses in-memory fallback)."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        super().__init__()
        self.redis_url = redis_url or "redis://localhost:6379/0"


class EvolveMemory:
    """High-level memory interface for PES runs."""

    def __init__(self, backend: Optional[BaseMemory] = None) -> None:
        self.backend = backend or InMemory()

    def store_solution(self, solution: Solution) -> None:
        self.backend.add(solution.id, solution)

    def get_solution(self, solution_id: str) -> Optional[Solution]:
        return self.backend.get(solution_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solutions": [s.__dict__ for s in self.backend.all()],
            "count": len(list(self.backend.all())),
        }


class MemoryFactory:
    """Factory for building memory backends."""

    def create(self, backend: str = "in_memory") -> BaseMemory:
        if backend == "redis":
            return RedisMemory()
        return InMemory()


__all__ = [
    "EvolveMemory",
    "Solution",
    "InMemory",
    "MemoryFactory",
    "RedisMemory",
]
