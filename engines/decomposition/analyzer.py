"""
analyzer.py - Lightweight, dependency-free problem analyzer.

Produces structured features (complexity, structure flags, recommended
decomposition strategy) from a problem definition. It can be handed to the
shared :class:`DecompositionEngine` as its ``problem_analyzer`` and is also used
directly by the local strategy selection.

No external services or LLM calls are required.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from strategies import StrategyKind, normalize_problem  # type: ignore


@dataclass
class ComplexityBreakdown:
    cognitive: float = 0.0
    computational: float = 0.0
    integration: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cognitive": round(self.cognitive, 3),
            "computational": round(self.computational, 3),
            "integration": round(self.integration, 3),
            "overall": round(self.overall, 3),
        }


@dataclass
class AnalysisResult:
    problem_id: str = ""
    title: str = ""
    domain: str = ""
    complexity: ComplexityBreakdown = field(default_factory=ComplexityBreakdown)
    num_constraints: int = 0
    num_requirements: int = 0
    num_objectives: int = 0
    structure_flags: Dict[str, bool] = field(default_factory=dict)
    recommended_strategy: str = "hierarchical"
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "title": self.title,
            "domain": self.domain,
            "complexity": self.complexity.to_dict(),
            "num_constraints": self.num_constraints,
            "num_requirements": self.num_requirements,
            "num_objectives": self.num_objectives,
            "structure_flags": dict(self.structure_flags),
            "recommended_strategy": self.recommended_strategy,
            "features": dict(self.features),
        }


_PHASE_KEYWORDS = ["plan", "design", "implement", "develop", "test", "verify",
                   "validate", "deploy", "release", "deliver", "maintain"]
_COMPONENT_PATTERNS = [r"(\w+)\s+(?:module|component|service|system|interface)"]


class ProblemAnalyzer:
    """Heuristic analyzer used for strategy selection and feature extraction."""

    def analyze(self, problem: Any) -> AnalysisResult:
        p = normalize_problem(problem)
        desc = p["description"] or ""
        desc_lower = desc.lower()
        reqs = p["requirements"]
        constraints = p["constraints"]
        objectives = p["success_criteria"]

        cognitive = min(10.0, len(desc) / 200.0 + 0.5 * len(reqs))
        multiplier = {
            "software_engineering": 1.2, "data_science": 1.3, "research": 1.1,
            "operations": 0.8, "business": 0.7,
        }.get((p["domain"] or "general").lower(), 1.0)
        computational = min(10.0, cognitive * multiplier)
        integration = min(10.0, 2.0 + 0.5 * len(constraints))
        overall = (cognitive + computational + integration) / 3.0

        structure = {
            "has_phases": any(k in desc_lower for k in _PHASE_KEYWORDS),
            "has_components": any(re.search(pat, desc_lower) for pat in _COMPONENT_PATTERNS),
            "has_constraints": len(constraints) > 0,
            "has_objectives": len(objectives) > 0,
        }

        recommended = self.recommend_strategy(
            overall_complexity=overall,
            num_constraints=len(constraints),
            num_objectives=len(objectives),
            desc_len=len(desc),
            num_requirements=len(reqs),
        )

        return AnalysisResult(
            problem_id=p["id"],
            title=p["title"],
            domain=p["domain"],
            complexity=ComplexityBreakdown(cognitive, computational, integration, overall),
            num_constraints=len(constraints),
            num_requirements=len(reqs),
            num_objectives=len(objectives),
            structure_flags=structure,
            recommended_strategy=recommended,
            features={
                "description_length": len(desc),
                "avg_word_len": (sum(len(w) for w in re.findall(r"\w+", desc)) /
                                max(1, len(re.findall(r"\w+", desc)))),
            },
        )

    def recommend_strategy(
        self,
        overall_complexity: float = 0.0,
        num_constraints: int = 0,
        num_objectives: int = 0,
        desc_len: int = 0,
        num_requirements: int = 0,
    ) -> str:
        if num_constraints >= 3:
            return StrategyKind.DEPENDENCY.value
        if overall_complexity >= 7.0 or num_objectives >= 4:
            return StrategyKind.HIERARCHICAL.value
        if desc_len < 200:
            return StrategyKind.SEMANTIC.value
        if desc_len > 800 or num_requirements >= 4:
            return StrategyKind.HIERARCHICAL.value
        return StrategyKind.FLOW.value


__all__ = [
    "ProblemAnalyzer",
    "AnalysisResult",
    "ComplexityBreakdown",
]
