"""
Re-export of ``SubProblem`` (and ``ProblemDefinition``) for flat scripts.

Flat engines can simply do ``from subproblem import SubProblem``. The canonical
definition lives in ``core-projects/openevolve/openevolve/kernel/schema.py``; if
that module is not importable (e.g. running outside the monorepo layout) a
minimal dependency-free fallback is provided so the symbol always resolves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Try the canonical schema location first.
# --------------------------------------------------------------------------- #
try:
    from core_projects.openevolve.openevolve.kernel.schema import (  # type: ignore
        ProblemDefinition,
        SubProblem,
    )
except Exception:  # pragma: no cover - fallback path
    try:
        from openevolve.kernel.schema import (  # type: ignore
            ProblemDefinition,
            SubProblem,
        )
    except Exception:
        from dataclasses import dataclass, field  # type: ignore

        @dataclass
        class SubProblem:
            id: str = ""
            title: str = ""
            description: str = ""
            parent_id: Optional[str] = None
            status: str = "pending"
            strategy: Optional[str] = None
            order: int = 0
            metadata: Dict[str, Any] = field(default_factory=dict)

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "id": self.id,
                    "title": self.title,
                    "description": self.description,
                    "parent_id": self.parent_id,
                    "status": self.status,
                    "strategy": self.strategy,
                    "order": self.order,
                    "metadata": dict(self.metadata),
                }

        @dataclass
        class ProblemDefinition:
            id: str = ""
            title: str = ""
            description: str = ""
            problem_type: str = "analysis"
            constraints: List[Any] = field(default_factory=list)
            success_criteria: List[Any] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "id": self.id,
                    "title": self.title,
                    "description": self.description,
                    "problem_type": self.problem_type,
                    "metadata": dict(self.metadata),
                }


__all__ = ["SubProblem", "ProblemDefinition"]
