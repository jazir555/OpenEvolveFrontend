"""Context objects for PES execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PESContext:
    problem: str
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
