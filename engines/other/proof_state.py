"""proof_state - canonical proof-search data structures.

Flat-script module providing the shared ``ProofState`` / ``Tactic`` / ``ProofHint``
definitions that the ``engines/`` proof, MCTS and decomposition scripts expect via
``from proof_state import ProofState, Tactic, ProofHint``.

Everything here is pure-Python with no external dependencies, so the module
imports cleanly with no Lean4 server, no Z3 and no LLM backend present.

Shapes:
- ``Tactic``     -- name + args, plus the pre/post proof states it bridges.
- ``ProofState`` -- the open goals, the tactics assigned to it and its history.
- ``ProofHint``  -- a natural-language hint plus the target it applies to.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProofStatus(str, Enum):
    """Lifecycle of a proof state."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"
    STUCK = "stuck"


class HintKind(str, Enum):
    """Category of a :class:`ProofHint`."""

    GENERAL = "general"
    USE_THEOREM = "use_theorem"
    SIMPLIFY = "simplify"
    REWRITE = "rewrite"
    INDUCTION = "induction"
    CASES = "cases"
    COUNTEREXAMPLE = "counterexample"


# ---------------------------------------------------------------------------
# Tactic
# ---------------------------------------------------------------------------
@dataclass
class Tactic:
    """A single tactic application.

    ``pre_state`` / ``post_state`` link the tactic to the states it bridges.
    They are excluded from ``repr``/``eq`` because ``ProofState`` holds its
    tactics, which would otherwise recurse infinitely.
    """

    name: str
    args: List[str] = field(default_factory=list)
    pre_state: Optional["ProofState"] = field(default=None, repr=False, compare=False)
    post_state: Optional["ProofState"] = field(default=None, repr=False, compare=False)
    confidence: float = 1.0
    succeeded: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept a bare string of arguments for ergonomics: Tactic("rw", "h1")
        if isinstance(self.args, str):
            self.args = [a for a in self.args.split() if a]
        else:
            self.args = [str(a) for a in self.args]
        self.name = str(self.name).strip()

    # -- rendering -------------------------------------------------------
    def to_string(self) -> str:
        """Render as Lean-style tactic text, e.g. ``rw [h1] at h2``."""
        if self.args:
            return f"{self.name} {' '.join(self.args)}"
        return self.name

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_string()

    @classmethod
    def parse(cls, text: str) -> "Tactic":
        """Parse ``"simp [foo, bar]"`` into a :class:`Tactic`."""
        parts = str(text).strip().split()
        if not parts:
            return cls(name="skip")
        return cls(name=parts[0], args=parts[1:])

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": list(self.args),
            "pre_state": self.pre_state.signature() if self.pre_state else None,
            "post_state": self.post_state.signature() if self.post_state else None,
            "confidence": self.confidence,
            "succeeded": self.succeeded,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tactic":
        return cls(
            name=data.get("name", "skip"),
            args=list(data.get("args") or []),
            confidence=float(data.get("confidence", 1.0)),
            succeeded=data.get("succeeded"),
            metadata=dict(data.get("metadata") or {}),
        )

    def is_closing(self) -> bool:
        """True for tactics that typically close a goal outright."""
        return self.name in {
            "rfl", "trivial", "simp", "decide", "omega", "linarith",
            "norm_num", "assumption", "exact", "tauto", "aesop",
        }


# ---------------------------------------------------------------------------
# ProofHint
# ---------------------------------------------------------------------------
@dataclass
class ProofHint:
    """A hint suggesting how to progress on ``target``."""

    text: str
    target: Optional[str] = None
    kind: HintKind = HintKind.GENERAL
    confidence: float = 1.0
    source: Optional[str] = None
    suggested_tactics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = str(self.text).strip()
        if isinstance(self.kind, str):
            try:
                self.kind = HintKind(self.kind)
            except ValueError:
                self.kind = HintKind.GENERAL
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def applies_to(self, goal: str) -> bool:
        """True when this hint targets ``goal`` (or is untargeted/global)."""
        if not self.target:
            return True
        return self.target.strip() in str(goal)

    def as_tactics(self) -> List[Tactic]:
        """Materialize the suggested tactic strings as :class:`Tactic` objects."""
        return [Tactic.parse(t) for t in self.suggested_tactics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "target": self.target,
            "kind": self.kind.value,
            "confidence": self.confidence,
            "source": self.source,
            "suggested_tactics": list(self.suggested_tactics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofHint":
        return cls(
            text=data.get("text", ""),
            target=data.get("target"),
            kind=data.get("kind", HintKind.GENERAL),
            confidence=float(data.get("confidence", 1.0)),
            source=data.get("source"),
            suggested_tactics=list(data.get("suggested_tactics") or []),
            metadata=dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# ProofState
# ---------------------------------------------------------------------------
@dataclass
class ProofState:
    """The open goals of a proof, the tactics assigned to it and its history."""

    goals: List[str] = field(default_factory=list)
    assigned_tactics: List[Tactic] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    hints: List[ProofHint] = field(default_factory=list)
    depth: int = 0
    solved: bool = False
    status: ProofStatus = ProofStatus.OPEN
    theorem: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Accept a single goal string for ergonomics: ProofState("1 + 1 = 2")
        if isinstance(self.goals, str):
            self.goals = [self.goals]
        else:
            self.goals = [str(g) for g in self.goals]
        # Accept raw tactic strings in assigned_tactics.
        self.assigned_tactics = [
            t if isinstance(t, Tactic) else Tactic.parse(str(t))
            for t in self.assigned_tactics
        ]
        if isinstance(self.status, str):
            try:
                self.status = ProofStatus(self.status)
            except ValueError:
                self.status = ProofStatus.OPEN
        self._refresh_status()

    # -- queries ---------------------------------------------------------
    def is_solved(self) -> bool:
        """A state is solved when explicitly marked or when no goals remain."""
        return self.solved or not self.goals

    @property
    def open_goals(self) -> int:
        return len(self.goals)

    def signature(self) -> str:
        """Stable, hashable identity of the goal set (used for dedup/memoization)."""
        return " | ".join(self.goals) if self.goals else "<solved>"

    def tactic_sequence(self) -> List[str]:
        """The rendered tactics assigned to this state, in order."""
        return [t.to_string() for t in self.assigned_tactics]

    def _refresh_status(self) -> None:
        if self.is_solved():
            self.solved = True
            self.status = ProofStatus.SOLVED
        elif self.status is ProofStatus.SOLVED:
            # Goals were reopened; downgrade the status.
            self.solved = False
            self.status = ProofStatus.IN_PROGRESS

    # -- mutation --------------------------------------------------------
    def add_goal(self, goal: str) -> None:
        if goal:
            self.goals.append(str(goal))
            self._refresh_status()

    def close_goal(self, goal: str) -> bool:
        """Discharge ``goal``. Returns True when it was present."""
        if goal in self.goals:
            self.goals.remove(goal)
            self.record(f"closed:{goal}")
            self._refresh_status()
            return True
        return False

    def add_hint(self, hint: "ProofHint | str", target: Optional[str] = None) -> ProofHint:
        """Attach a hint, accepting either a ``ProofHint`` or plain text."""
        obj = hint if isinstance(hint, ProofHint) else ProofHint(text=str(hint), target=target)
        self.hints.append(obj)
        return obj

    def hints_for(self, goal: str) -> List[ProofHint]:
        return [h for h in self.hints if h.applies_to(goal)]

    def record(self, event: str) -> None:
        """Append an event to the history log."""
        self.history.append(str(event))

    def assign(self, tactic: "Tactic | str") -> Tactic:
        """Assign a tactic to this state without advancing it."""
        obj = tactic if isinstance(tactic, Tactic) else Tactic.parse(str(tactic))
        self.assigned_tactics.append(obj)
        self.record(f"assigned:{obj.to_string()}")
        return obj

    def apply_tactic(
        self,
        tactic: "Tactic | str",
        new_goals: Optional[Iterable[str]] = None,
        succeeded: bool = True,
    ) -> "ProofState":
        """Apply ``tactic`` and return the resulting child state.

        ``new_goals`` replaces the goal set (``[]`` closes the proof). When it is
        omitted the first goal is discharged, which models the common
        "one tactic closes one goal" case. The returned state is linked to this
        one through ``tactic.pre_state`` / ``tactic.post_state``.
        """
        obj = tactic if isinstance(tactic, Tactic) else Tactic.parse(str(tactic))
        obj.succeeded = succeeded

        if new_goals is not None:
            goals = [str(g) for g in new_goals]
        elif succeeded and self.goals:
            goals = list(self.goals[1:])
        else:
            goals = list(self.goals)

        child = ProofState(
            goals=goals,
            history=list(self.history) + [f"apply:{obj.to_string()}"],
            hints=list(self.hints),
            depth=self.depth + 1,
            theorem=self.theorem,
            status=ProofStatus.IN_PROGRESS if goals else ProofStatus.SOLVED,
            metadata=dict(self.metadata),
        )
        if not succeeded:
            child.status = ProofStatus.STUCK

        obj.pre_state = self
        obj.post_state = child
        self.assigned_tactics.append(obj)
        child._refresh_status()
        return child

    def clone(self) -> "ProofState":
        """Deep-enough copy: goal/history/tactic lists are independent."""
        return ProofState(
            goals=list(self.goals),
            assigned_tactics=[Tactic.from_dict(t.to_dict()) for t in self.assigned_tactics],
            history=list(self.history),
            hints=list(self.hints),
            depth=self.depth,
            solved=self.solved,
            status=self.status,
            theorem=self.theorem,
            metadata=dict(self.metadata),
        )

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goals": list(self.goals),
            "assigned_tactics": [t.to_dict() for t in self.assigned_tactics],
            "history": list(self.history),
            "hints": [h.to_dict() for h in self.hints],
            "depth": self.depth,
            "solved": self.is_solved(),
            "status": self.status.value,
            "theorem": self.theorem,
            "signature": self.signature(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofState":
        return cls(
            goals=list(data.get("goals") or []),
            assigned_tactics=[
                Tactic.from_dict(t) if isinstance(t, dict) else Tactic.parse(str(t))
                for t in (data.get("assigned_tactics") or [])
            ],
            history=list(data.get("history") or []),
            hints=[
                ProofHint.from_dict(h) if isinstance(h, dict) else ProofHint(text=str(h))
                for h in (data.get("hints") or [])
            ],
            depth=int(data.get("depth", 0)),
            solved=bool(data.get("solved", False)),
            status=data.get("status", ProofStatus.OPEN),
            theorem=data.get("theorem"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_theorem(cls, theorem: str) -> "ProofState":
        """Build the initial state for ``theorem`` (its statement is the goal)."""
        return cls(goals=[str(theorem)], theorem=str(theorem))


def replay(state: ProofState) -> Tuple[List[str], bool]:
    """Walk ``state``'s assigned tactics, returning ``(tactic_strings, solved)``."""
    return state.tactic_sequence(), state.is_solved()


__all__ = [
    "ProofState",
    "Tactic",
    "ProofHint",
    "ProofStatus",
    "HintKind",
    "replay",
]
