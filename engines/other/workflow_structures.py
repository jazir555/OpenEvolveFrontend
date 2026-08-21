"""
Facade re-exporting workflow data models for the flat ``engines/`` scripts
(sys.path-style imports, no ``__init__.py``).

A thin mirror of ``openevolve_structures`` exposing the names historically
imported from ``workflow_structures``. The canonical definitions live in
``openevolve.kernel.schema``. Falls back to stub dataclasses if the schema is
unavailable.
"""
from __future__ import annotations


try:
    from openevolve.kernel.schema import (  # type: ignore
        CritiqueReport,
        DecompositionPlan,
        GauntletDefinition,
        GauntletRoundRule,
        KnowledgeArtifact,
        ModelConfig,
        PerformanceMetrics,
        SolutionAttempt,
        SubProblem,
        Team,
        VerificationMethod,
        VerificationReport,
        WorkflowState,
        MathematicalDomain as WorkflowMathDomain,
    )
except Exception:  # pragma: no cover - fallback path
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    @dataclass
    class CritiqueReport:
        pass

    @dataclass
    class DecompositionPlan:
        pass

    @dataclass
    class GauntletDefinition:
        pass

    @dataclass
    class GauntletRoundRule:
        pass

    @dataclass
    class KnowledgeArtifact:
        pass

    @dataclass
    class ModelConfig:
        model: str = ""
        temperature: float = 0.0
        max_tokens: int = 0

    @dataclass
    class PerformanceMetrics:
        pass

    @dataclass
    class SolutionAttempt:
        pass

    @dataclass
    class SubProblem:
        pass

    @dataclass
    class Team:
        id: str = ""
        name: str = ""
        members: List[Any] = field(default_factory=list)

    @dataclass
    class VerificationMethod:
        pass

    @dataclass
    class VerificationReport:
        pass

    @dataclass
    class WorkflowState:
        pass

    @dataclass
    class WorkflowMathDomain:
        pass


__all__ = [
    "CritiqueReport",
    "DecompositionPlan",
    "GauntletDefinition",
    "GauntletRoundRule",
    "KnowledgeArtifact",
    "ModelConfig",
    "PerformanceMetrics",
    "SolutionAttempt",
    "SubProblem",
    "Team",
    "VerificationMethod",
    "VerificationReport",
    "WorkflowState",
    "WorkflowMathDomain",
]
