"""Graded memory for LoongFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GradedMemory:
    """Simple graded memory store."""

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, content: Any, score: float, metadata: Dict[str, Any] | None = None) -> None:
        self.entries.append({
            "content": content,
            "score": score,
            "metadata": metadata or {},
        })

    def top(self, limit: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.entries, key=lambda e: e.get("score", 0), reverse=True)[:limit]
