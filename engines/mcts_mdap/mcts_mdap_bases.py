"""mcts_mdap_bases - canonical proof-state / tactic primitives for mcts_mdap.

This module is the single source of truth for the shared ``ProofState``,
``Tactic`` and ``ProofHint`` symbols used across ``engines/mcts_mdap``.

It prefers the project-wide shared module ``proof_state`` (provided by another
agent in ``engines/other``). Because the mcts_mdap proof-search code was written
against a richer "legacy" interface (``apply``, ``is_complete``, ``context``,
``tactics_sequence``, ``tactic``, ``children`` and a ``Tactic.arguments``
attribute), this module exposes **compatibility adapters** that subclass the
shared definitions and add exactly those members. When ``proof_state`` is not
(yet) present, fully-functional standalone definitions are used instead, so the
mcts_mdap modules remain importable and runnable regardless of parallel timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical source resolution
# ---------------------------------------------------------------------------
try:  # Prefer the project-wide shared module.
    from proof_state import (  # type: ignore
        ProofState as _RealProofState,
        Tactic as _RealTactic,
        ProofHint as _RealProofHint,
    )

    _USING_SHARED = True
except ImportError:  # pragma: no cover - depends on parallel agent timing
    _RealProofState = None
    _RealTactic = None
    _RealProofHint = None
    _USING_SHARED = False


# ---------------------------------------------------------------------------
# Tactic
# ---------------------------------------------------------------------------
if _USING_SHARED:

    class Tactic(_RealTactic):
        """Compatibility adapter over the shared ``Tactic``.

        Adds a legacy ``arguments`` attribute (the shared type uses ``args``)
        and accepts ``arguments=`` at construction time.
        """

        def __init__(
            self,
            name: str = "skip",
            arguments: Any = None,
            args: Any = None,
            confidence: float = 1.0,
            **kwargs: Any,
        ) -> None:
            if args is None:
                args = arguments
            if isinstance(args, str):
                args = [a for a in args.split() if a]
            elif args is None:
                args = []
            else:
                args = [str(a) for a in args]
            super().__init__(name=str(name), args=args, confidence=float(confidence))

        @property
        def arguments(self) -> List[str]:
            return list(self.args)

else:

    @dataclass
    class Tactic:
        """A single Lean-style tactic application (standalone fallback)."""

        name: str
        arguments: Any = ""
        confidence: float = 1.0

        def to_string(self) -> str:
            if self.arguments:
                if isinstance(self.arguments, (list, tuple)):
                    return f"{self.name} {' '.join(str(a) for a in self.arguments)}"
                return f"{self.name} {self.arguments}"
            return self.name

        def to_dict(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "arguments": self.arguments,
                "confidence": self.confidence,
            }

        @classmethod
        def parse(cls, text: str) -> "Tactic":
            parts = str(text).strip().split(None, 1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            return cls(name=name, arguments=args)


# ---------------------------------------------------------------------------
# ProofState
# ---------------------------------------------------------------------------
if _USING_SHARED:

    @dataclass
    class ProofState(_RealProofState):
        """Compatibility adapter over the shared ``ProofState``.

        Adds the legacy members the mcts_mdap search code expects:
        ``apply``, ``is_complete``, ``context``, ``tactics_sequence``,
        ``tactic`` and ``children``.
        """

        # New legacy fields appended after the shared base fields.
        context: List[str] = field(default_factory=list)
        tactics_sequence: List["Tactic"] = field(default_factory=list)
        tactic: Optional["Tactic"] = None
        children: List["ProofState"] = field(default_factory=list)
        is_complete: bool = False

        def apply(
            self, tactic: "Tactic", new_goals: Optional[List[str]] = None
        ) -> "ProofState":
            child = self.__class__(
                goals=list(new_goals if new_goals is not None else self.goals),
                context=list(self.context),
                tactics_sequence=list(self.tactics_sequence) + [tactic],
                depth=self.depth + 1,
                theorem=self.theorem,
            )
            child.tactic = tactic
            self.children.append(child)
            child.is_complete = len(child.goals) == 0
            return child

        def to_dict(self) -> Dict[str, Any]:  # type: ignore[override]
            base = super().to_dict()
            base.update(
                {
                    "context": list(self.context),
                    "tactics_sequence": [t.to_string() for t in self.tactics_sequence],
                    "tactic": self.tactic.to_string() if self.tactic else None,
                    "children": len(self.children),
                    "is_complete": self.is_complete,
                }
            )
            return base

else:

    @dataclass
    class ProofState:
        """A node in a proof search: goals plus bookkeeping (standalone fallback)."""

        goals: List[str] = field(default_factory=list)
        context: List[str] = field(default_factory=list)
        tactics_sequence: List["Tactic"] = field(default_factory=list)
        depth: int = 0
        parent: Optional["ProofState"] = None
        tactic: Optional["Tactic"] = None
        children: List["ProofState"] = field(default_factory=list)
        solved: bool = False
        is_complete: bool = False

        def is_solved(self) -> bool:
            return self.is_complete or self.solved or len(self.goals) == 0

        def apply(
            self, tactic: "Tactic", new_goals: Optional[List[str]] = None
        ) -> "ProofState":
            child = ProofState(
                goals=list(new_goals if new_goals is not None else self.goals),
                context=list(self.context),
                tactics_sequence=list(self.tactics_sequence) + [tactic],
                depth=self.depth + 1,
                parent=self,
                tactic=tactic,
            )
            self.children.append(child)
            child.is_complete = len(child.goals) == 0
            return child

        def to_dict(self) -> Dict[str, Any]:
            return {
                "goals": list(self.goals),
                "context": list(self.context),
                "depth": self.depth,
                "tactic": self.tactic.to_string() if self.tactic else None,
                "solved": self.is_solved(),
            }


# ---------------------------------------------------------------------------
# ProofHint
# ---------------------------------------------------------------------------
if _USING_SHARED:
    ProofHint = _RealProofHint  # Shared ProofHint is already compatible.
else:

    class ProofHint:
        """A lightweight advisory hint attached to a proof state (fallback)."""

        def __init__(self, hint: str = "", **metadata: Any):
            self.hint = hint
            self.metadata = metadata

        def to_dict(self) -> Dict[str, Any]:
            return {"hint": self.hint, "metadata": dict(self.metadata)}


__all__ = ["ProofState", "Tactic", "ProofHint", "_USING_SHARED"]
