"""
Compatibility schema for the BubbleLab integration engines.

The ``engines/other`` decomposition stack historically imported shared problem
models from ``openevolve.kernel.schema`` (a kernel subpackage that is not part
of the openevolve distribution shipped here). This module provides a faithful,
dependency-free reimplementation of the symbols those engines expect so the
BubbleLab -> OpenEvolve integration can run them locally (using the engines'
built-in heuristic fallbacks rather than an LLM).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


def generate_id(prefix: str = "id") -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ProblemType(str, Enum):
    """Category of problem, used to select decomposition strategy."""

    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    DESIGN = "design"


class SubProblemStatus(str, Enum):
    """Lifecycle status of a decomposed sub-problem."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DomainContext:
    domain: str = "general"
    subdomain: Optional[str] = None
    related_domains: List[str] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplexityScore:
    cognitive_complexity: float = 5.0
    computational_complexity: float = 5.0
    domain_complexity: float = 5.0
    integration_complexity: float = 5.0
    overall_complexity: float = 5.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Constraint:
    id: str = ""
    description: str = ""
    type: str = "general"
    severity: str = "soft"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessCriterion:
    id: str = ""
    description: str = ""
    metric: str = ""
    threshold: float = 0.0
    validation_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProblemDefinition:
    id: str = ""
    title: str = ""
    description: str = ""
    problem_type: ProblemType = ProblemType.ANALYSIS
    domain_context: DomainContext = field(default_factory=DomainContext)
    complexity_score: ComplexityScore = field(default_factory=ComplexityScore)
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    resources_available: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "problem_type": self.problem_type.value
            if isinstance(self.problem_type, ProblemType)
            else self.problem_type,
            "domain_context": self.domain_context.to_dict(),
            "complexity_score": self.complexity_score.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "success_criteria": [s.to_dict() for s in self.success_criteria],
            "stakeholders": list(self.stakeholders),
            "resources_available": dict(self.resources_available),
            "metadata": dict(self.metadata),
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
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "strategy": self.strategy,
            "order": self.order,
            "metadata": dict(self.metadata),
        }


@dataclass
class ModelConfig:
    model: str = "default"
    provider: str = "default"
    temperature: float = 0.3
    max_tokens: int = 512
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowState(str, Enum):
    """High level workflow state used by some integration bridges."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


__all__ = [
    "generate_id",
    "ProblemType",
    "SubProblemStatus",
    "DomainContext",
    "ComplexityScore",
    "Constraint",
    "SuccessCriterion",
    "ProblemDefinition",
    "SubProblem",
    "ModelConfig",
    "WorkflowState",
]
