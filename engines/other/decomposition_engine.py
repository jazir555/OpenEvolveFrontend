"""
Decomposition Engine for the BubbleLab -> OpenEvolve integration.

This module provides a real, dependency-light :class:`DecompositionEngine` that
decomposes a problem definition into a structured plan of sub-problems. It works
without an external LLM by using the heuristic analysis already produced by
:class:`problem_analyzer.ProblemAnalyzer`.

The original stub in this file only defined ``RecursiveSolver``; the real engine
was referenced widely across the codebase but never lived here. This
implementation restores that contract so the ``/api/decomposition/plan`` endpoint
returns genuine analysis instead of failing.

The sub-problem / plan models are defined locally (rather than imported from
``openevolve.kernel.schema``) so the engine is robust against the schema
definitions living in multiple openevolve locations in this monorepo.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RecursiveSolver:
    """Stub class for RecursiveSolver (retained for backward compatibility)."""

    pass


class SubProblemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


_STRATEGY_ALIASES = {
    "hierarchical": "hierarchical",
    "tree": "hierarchical",
    "semantic": "semantic",
    "flow": "flow_based",
    "flow_based": "flow_based",
    "adaptive": "adaptive",
    "auto": "hierarchical",
}


@dataclass
class SubProblem:
    id: str = ""
    title: str = ""
    description: str = ""
    parent_id: Optional[str] = None
    status: SubProblemStatus = SubProblemStatus.PENDING
    strategy: Optional[str] = None
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class DecompositionPlan:
    """Result of a decomposition: an ordered set of sub-problems."""

    problem_id: str = ""
    strategy: str = "hierarchical"
    sub_problems: List[SubProblem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "strategy": self.strategy,
            "sub_problems": [sp.to_dict() for sp in self.sub_problems],
            "sub_problem_count": len(self.sub_problems),
            "metadata": dict(self.metadata),
        }


class DecompositionEngine:
    """Decomposes a problem definition into a plan of sub-problems."""

    def __init__(
        self,
        problem_analyzer: Any = None,
        enable_adaptive_selection: bool = False,
        maker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.problem_analyzer = problem_analyzer
        self.enable_adaptive_selection = bool(enable_adaptive_selection)
        self.maker_config = maker_config or {}
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Strategy selection
    # ------------------------------------------------------------------ #
    def select_strategy(self, problem: Any, strategy: str) -> str:
        normalized = _STRATEGY_ALIASES.get((strategy or "").lower(), "hierarchical")
        if normalized == "adaptive":
            normalized = self._select_adaptive_strategy(problem)
        return normalized

    def _select_adaptive_strategy(self, problem: Any) -> str:
        # Heuristic: many constraints or high complexity => hierarchical,
        # otherwise semantic decomposition gives finer granularity.
        complexity = 0.0
        try:
            complexity = float(
                getattr(getattr(problem, "complexity_score", None), "overall_complexity", 0.0) or 0.0
            )
        except Exception:
            complexity = 0.0
        if complexity >= 7.0 or len(getattr(problem, "constraints", []) or []) >= 3:
            return "hierarchical"
        return "semantic"

    # ------------------------------------------------------------------ #
    # Decomposition
    # ------------------------------------------------------------------ #
    def decompose(self, problem: Any, strategy: str = "hierarchical") -> DecompositionPlan:
        if problem is None:
            raise ValueError("decompose() requires a ProblemDefinition")

        resolved = self.select_strategy(problem, strategy)
        sub_problems: List[SubProblem] = []

        if resolved == "semantic":
            sub_problems = self._semantic_decompose(problem)
        elif resolved == "flow_based":
            sub_problems = self._flow_decompose(problem)
        else:  # hierarchical (default)
            sub_problems = self._hierarchical_decompose(problem)

        plan = DecompositionPlan(
            problem_id=getattr(problem, "id", "") or generate_id("problem"),
            strategy=resolved,
            sub_problems=sub_problems,
            metadata={
                "engine": "DecompositionEngine",
                "strategy_selected": resolved,
                "adaptive_selection": self.enable_adaptive_selection,
            },
        )
        return plan

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _hierarchical_decompose(self, problem: Any) -> List[SubProblem]:
        subs: List[SubProblem] = []

        # Root analysis node.
        subs.append(
            SubProblem(
                id=generate_id("sub"),
                title=f"Analyze: {getattr(problem, 'title', '') or 'problem'}",
                description=(
                    "Understand the problem, its domain context and success criteria "
                    "before producing solution components."
                ),
                parent_id=None,
                status=SubProblemStatus.PENDING,
                strategy="analysis",
                order=0,
            )
        )

        # One sub-problem per success criterion (the measurable objectives).
        criteria = list(getattr(problem, "success_criteria", []) or [])
        for idx, criterion in enumerate(criteria, start=1):
            desc = getattr(criterion, "description", "") or "Unnamed objective"
            subs.append(
                SubProblem(
                    id=getattr(criterion, "id", "") or generate_id("sub"),
                    title=f"Objective {idx}: {desc}",
                    description=(
                        f"Satisfy the success criterion: {desc} "
                        f"(metric={getattr(criterion, 'metric', 'n/a')}, "
                        f"threshold={getattr(criterion, 'threshold', 'n/a')})."
                    ),
                    parent_id=subs[0].id,
                    status=SubProblemStatus.PENDING,
                    strategy="implementation",
                    order=idx,
                )
            )

        # A synthesis node if there is more than one objective.
        if len(subs) > 2:
            subs.append(
                SubProblem(
                    id=generate_id("sub"),
                    title="Synthesize and validate integrated solution",
                    description=(
                        "Integrate the objective sub-problems, verify against the "
                        "original constraints and success criteria, and resolve conflicts."
                    ),
                    parent_id=subs[0].id,
                    status=SubProblemStatus.PENDING,
                    strategy="synthesis",
                    order=len(subs),
                )
            )

        return subs

    def _semantic_decompose(self, problem: Any) -> List[SubProblem]:
        # Group by domain concept / keyword clusters for finer-grained semantic units.
        subs: List[SubProblem] = []
        concepts: List[str] = []
        dc = getattr(problem, "domain_context", None)
        if dc is not None:
            dk = getattr(dc, "domain_knowledge", None) or {}
            concepts = list(dk.get("key_concepts", []) or [])

        if not concepts:
            concepts = ["core"]

        for idx, concept in enumerate(concepts, start=0):
            subs.append(
                SubProblem(
                    id=generate_id("sub"),
                    title=f"Semantic unit: {concept}",
                    description=(
                        f"Address the semantic cluster '{concept}' within the "
                        f"{getattr(dc, 'domain', 'general')} domain."
                    ),
                    parent_id=None,
                    status=SubProblemStatus.PENDING,
                    strategy="semantic",
                    order=idx,
                )
            )
        return subs

    def _flow_decompose(self, problem: Any) -> List[SubProblem]:
        # Linear pipeline: ingest -> transform -> evaluate -> deliver.
        stages = [
            ("Ingest & scope", "Collect requirements, constraints and context."),
            ("Transform", "Produce the core solution transformation."),
            ("Evaluate", "Validate against success criteria and constraints."),
            ("Deliver", "Package and hand off the verified solution."),
        ]
        subs: List[SubProblem] = []
        prev = None
        for idx, (title, desc) in enumerate(stages):
            sp = SubProblem(
                id=generate_id("sub"),
                title=title,
                description=desc,
                parent_id=prev,
                status=SubProblemStatus.PENDING,
                strategy="flow",
                order=idx,
            )
            subs.append(sp)
            prev = sp.id
        return subs


# Convenience aliases used by some callers in the wider codebase.
HierarchicalDecomposition = DecompositionEngine
SemanticDecomposition = DecompositionEngine
FlowBasedDecomposition = DecompositionEngine


__all__ = [
    "DecompositionEngine",
    "DecompositionPlan",
    "SubProblem",
    "SubProblemStatus",
    "RecursiveSolver",
    "HierarchicalDecomposition",
    "SemanticDecomposition",
    "FlowBasedDecomposition",
]
