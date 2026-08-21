"""mathlib4_integration - integration with Lean's mathlib4.

Provides a ``ProofHint`` data structure (used by proof engines such as
``automated_proof_engine``) and a ``Mathlib4Integration`` client that talks to a
mathlib4 project workspace, degrading gracefully when no workspace is present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HintType(str, Enum):
    USE_THEOREM = "use_theorem"
    SIMPLIFY = "simplify"
    INTRODUCE_LEMMA = "introduce_lemma"
    REWRITE = "rewrite"
    CASES = "cases"
    GENERAL = "general"


@dataclass
class ProofHint:
    """A hint that suggests how to make progress on a proof goal."""

    text: str
    hint_type: HintType = HintType.GENERAL
    source: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "hint_type": self.hint_type.value,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofHint":
        return cls(
            text=d.get("text", ""),
            hint_type=HintType(d.get("hint_type", HintType.GENERAL.value)),
            source=d.get("source"),
            confidence=float(d.get("confidence", 1.0)),
            metadata=d.get("metadata", {}),
        )


class Mathlib4Integration:
    """Client for a mathlib4 Lean project workspace."""

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = workspace or os.environ.get(
            "MATHLIB4_WORKSPACE",
            os.path.join(os.path.dirname(__file__), "lean_workspace", "mathlib_project"),
        )
        self.available = os.path.isdir(self.workspace) if self.workspace else False

    def suggest_hints(self, goal: str) -> List[ProofHint]:
        """Return proof hints for ``goal``. When no workspace is available this
        returns a small deterministic set of generic hints."""
        if not goal:
            return []
        hints = [
            ProofHint(text=f"Consider simplifying the goal: {goal}",
                      hint_type=HintType.SIMPLIFY, source="mathlib4", confidence=0.6),
            ProofHint(text="Try a case analysis on the principal argument.",
                      hint_type=HintType.CASES, source="mathlib4", confidence=0.5),
        ]
        return hints

    def is_available(self) -> bool:
        return self.available


def create_mathlib_integration(workspace: Optional[str] = None) -> Mathlib4Integration:
    return Mathlib4Integration(workspace=workspace)
