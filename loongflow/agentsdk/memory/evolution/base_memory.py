"""Base memory interfaces for LoongFlow."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class BaseMemory:
    """Base memory interface."""

    def add(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def all(self) -> Iterable[Any]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": list(self.all())}
