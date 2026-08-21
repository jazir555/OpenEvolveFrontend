"""leanaide_decomposition_integration - Lean-aware decomposition helpers.

Flat-script module providing the Lean mathematical-decomposition types referenced
by ``decomposition_engine_lean_enhanced`` (e.g.
``from leanaide_decomposition_integration import MathematicalDomain, LeanDecomposer``).
Self-contained and importable without external services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MathematicalDomain(str, Enum):
    """Mathematical domain taxonomy for decomposition routing."""

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
            return cls(str(value).lower())
        except ValueError:
            return cls.GENERAL


class ComponentType(str, Enum):
    AXIOM = "axiom"
    DEFINITION = "definition"
    LEMMA = "lemma"
    THEOREM = "theorem"
    PROOF = "proof"
    COMPUTATION = "computation"


class DecompositionStrategy(str, Enum):
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    FLOW_BASED = "flow_based"
    LEAN_AWARE = "lean_aware"


@dataclass
class MathematicalComponent:
    id: str = ""
    name: str = ""
    component_type: ComponentType = ComponentType.DEFINITION
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    expression: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "component_type": self.component_type.value,
            "domain": self.domain.value,
            "expression": self.expression,
            "dependencies": list(self.dependencies),
        }


@dataclass
class LeanSubProblem:
    id: str = ""
    title: str = ""
    statement: str = ""
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "statement": self.statement,
            "domain": self.domain.value,
            "dependencies": list(self.dependencies),
            "status": self.status,
        }


@dataclass
class LeanDecompositionPlan:
    problem_id: str = ""
    strategy: DecompositionStrategy = DecompositionStrategy.LEAN_AWARE
    components: List[MathematicalComponent] = field(default_factory=list)
    sub_problems: List[LeanSubProblem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "strategy": self.strategy.value,
            "components": [c.to_dict() for c in self.components],
            "sub_problems": [s.to_dict() for s in self.sub_problems],
        }


class LeanDecompositionStrategy:
    """Selects a decomposition strategy based on the problem's math domain."""

    @staticmethod
    def select(domain: MathematicalDomain) -> DecompositionStrategy:
        if domain in (MathematicalDomain.OPTIMIZATION, MathematicalDomain.COMPUTER_SCIENCE):
            return DecompositionStrategy.FLOW_BASED
        return DecompositionStrategy.LEAN_AWARE


class LeanDecomposer:
    """Decomposes a mathematical problem into Lean components / sub-problems."""

    def __init__(self, strategy: Optional[DecompositionStrategy] = None):
        self.strategy = strategy or DecompositionStrategy.LEAN_AWARE

    def decompose(self, problem_statement: str,
                   domain: MathematicalDomain = MathematicalDomain.GENERAL
                   ) -> LeanDecompositionPlan:
        strat = LeanDecompositionStrategy.select(domain)
        component = MathematicalComponent(
            name="root", domain=domain, expression=problem_statement,
            component_type=ComponentType.THEOREM,
        )
        sub = LeanSubProblem(
            title="Prove / solve", statement=problem_statement, domain=domain,
        )
        return LeanDecompositionPlan(
            problem_id="problem", strategy=strat,
            components=[component], sub_problems=[sub],
        )


__all__ = [
    "MathematicalDomain",
    "ComponentType",
    "DecompositionStrategy",
    "MathematicalComponent",
    "LeanSubProblem",
    "LeanDecompositionPlan",
    "LeanDecompositionStrategy",
    "LeanDecomposer",
]
