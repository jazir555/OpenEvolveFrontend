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
    VALIDATION = "validation"


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


class MathematicalDomain(str, Enum):
    """Enumeration of mathematical domains used by decomposition / proof engines."""

    GENERAL = "general"
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    TOPOLOGY = "topology"
    LOGIC = "logic"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    COMPUTER_SCIENCE = "computer_science"
    OPTIMIZATION = "optimization"

    @classmethod
    def from_string(cls, value: str) -> "MathematicalDomain":
        try:
            return cls(value.lower())
        except ValueError:
            return cls.GENERAL

    @classmethod
    def all_domains(cls) -> List["MathematicalDomain"]:
        return [d for d in cls]


# --------------------------------------------------------------------------- #
# Workflow structures (re-exported by engines/workflow/workflow_structures.py
# via ``from openevolve.kernel.schema import *``). These mirror the dataclasses
# the flat engines expect from ``workflow_structures`` so guarded imports there
# resolve without an external workflow service.
# --------------------------------------------------------------------------- #
@dataclass
class Team:
    id: str = ""
    name: str = ""
    members: List[str] = field(default_factory=list)
    lead: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolutionAttempt:
    id: str = ""
    sub_problem_id: Optional[str] = None
    solution: str = ""
    status: str = "pending"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CritiqueReport:
    id: str = ""
    target_id: Optional[str] = None
    critique: str = ""
    score: float = 0.0
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    id: str = ""
    target_id: Optional[str] = None
    verified: bool = False
    method: str = ""
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecompositionPlan:
    id: str = ""
    problem_id: Optional[str] = None
    strategy: str = "hierarchical"
    sub_problems: List[SubProblem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem_id": self.problem_id,
            "strategy": self.strategy,
            "sub_problem_count": len(self.sub_problems),
            "metadata": dict(self.metadata),
        }


@dataclass
class GauntletRoundRule:
    id: str = ""
    name: str = ""
    description: str = ""
    threshold: float = 0.0
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GauntletDefinition:
    id: str = ""
    name: str = ""
    description: str = ""
    rounds: List[GauntletRoundRule] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeArtifact:
    id: str = ""
    title: str = ""
    content: str = ""
    artifact_type: str = "generic"
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeArtifactModel:
    id: str = ""
    name: str = ""
    artifacts: List[KnowledgeArtifact] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeArtifactManager:
    """Minimal in-memory registry of knowledge artifacts."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, KnowledgeArtifact] = {}

    def add(self, artifact: KnowledgeArtifact) -> None:
        self._artifacts[artifact.id or generate_id("ka")] = artifact

    def get(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        return self._artifacts.get(artifact_id)

    def all(self) -> List[KnowledgeArtifact]:
        return list(self._artifacts.values())


@dataclass
class TeamPerformanceArtifact:
    team_id: str = ""
    metric: str = ""
    value: float = 0.0
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GauntletEffectivenessArtifact:
    gauntlet_id: str = ""
    round_id: str = ""
    passed: bool = False
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CritiqueInsightArtifact:
    id: str = ""
    insight: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    latency_ms: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LeanProofStatus(str, Enum):
    UNKNOWN = "unknown"
    PROVEN = "proven"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class LeanVerificationResult:
    proved: bool = False
    status: LeanProofStatus = LeanProofStatus.UNKNOWN
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"proved": self.proved, "status": self.status.value, "details": self.details}


class VerificationMethod(str, Enum):
    NONE = "none"
    UNIT_TEST = "unit_test"
    FORMAL = "formal"
    MANUAL = "manual"


class Needed:
    """Marker used by some engines to denote a required (needed) capability."""

    def __init__(self, name: str = ""):
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {"needed": self.name}


# Backwards-compatible alias used by some decomposition engines.
WorkflowMathDomain = MathematicalDomain


# --------------------------------------------------------------------------- #
# Sovereign data-model symbols (re-exported by engines/other/sovereign_data_models
# via ``from openevolve.kernel.schema import *``). These are referenced by the
# strategy / team-assignment engines.
# --------------------------------------------------------------------------- #
class ProblemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class SubProblemType(str, Enum):
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    GENERAL = "general"


@dataclass
class QualityMetrics:
    correctness: float = 0.0
    completeness: float = 0.0
    efficiency: float = 0.0
    robustness: float = 0.0
    overall: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubProblemTeamAssignment:
    sub_problem_id: str = ""
    team_id: str = ""
    assigned_members: List[str] = field(default_factory=list)
    priority: int = 0
    status: ProblemStatus = ProblemStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_problem_id": self.sub_problem_id,
            "team_id": self.team_id,
            "assigned_members": list(self.assigned_members),
            "priority": self.priority,
            "status": self.status.value,
        }


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
    "MathematicalDomain",
    "Team",
    "SolutionAttempt",
    "CritiqueReport",
    "VerificationReport",
    "DecompositionPlan",
    "GauntletRoundRule",
    "GauntletDefinition",
    "KnowledgeArtifact",
    "KnowledgeArtifactModel",
    "KnowledgeArtifactManager",
    "TeamPerformanceArtifact",
    "GauntletEffectivenessArtifact",
    "CritiqueInsightArtifact",
    "PerformanceMetrics",
    "LeanProofStatus",
    "LeanVerificationResult",
    "VerificationMethod",
    "Needed",
    "WorkflowMathDomain",
    "ProblemStatus",
    "SubProblemType",
    "QualityMetrics",
    "SubProblemTeamAssignment",
]
