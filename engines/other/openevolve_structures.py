"""
Facade re-exporting workflow/gauntlet data models for the flat ``engines/``
scripts (sys.path-style imports, no ``__init__.py``).

The canonical definitions live in ``openevolve.kernel.schema``. This thin
module lets the flat scripts resolve ``openevolve_structures`` unchanged. If
the canonical schema is unavailable, minimal stub dataclasses are provided so
the symbols still resolve.
"""
from __future__ import annotations


try:
    from openevolve.kernel.schema import (  # type: ignore
        CritiqueReport,
        DecompositionPlan,
        Feedback,
        GauntletDefinition,
        GauntletExecution,
        GauntletRoundRule,
        GauntletRoundResult,
        GauntletRoundStatus,
        ModelConfig,
        SolutionAttempt,
        SubProblem,
        Team,
        ValidationCheckpoint,
        ValidationResult,
        VerificationReport,
        WorkflowState,
        generate_id,
    )
except Exception:  # pragma: no cover - fallback path
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    @dataclass
    class GauntletDefinition:
        pass

    @dataclass
    class GauntletRoundRule:
        pass

    @dataclass
    class GauntletRoundStatus:
        pass

    @dataclass
    class GauntletRoundResult:
        pass

    @dataclass
    class GauntletExecution:
        pass

    @dataclass
    class ValidationCheckpoint:
        pass

    @dataclass
    class ValidationResult:
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
        capabilities: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ModelConfig:
        model: str = ""
        temperature: float = 0.0
        max_tokens: int = 0

    @dataclass
    class CritiqueReport:
        pass

    @dataclass
    class DecompositionPlan:
        pass

    @dataclass
    class Feedback:
        pass

    @dataclass
    class WorkflowState:
        workflow_id: str = ""
        workflow_type: str = ""
        problem_statement: str = ""
        current_stage: Any = None
        status: str = ""
        mdap_enabled: bool = False
        mdap_config: Dict[str, Any] = field(default_factory=dict)
        maker_enabled: bool = False
        maker_config: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class VerificationReport:
        solution_attempt_id: Any = None
        gauntlet_name: str = ""
        is_approved: bool = False
        reports_by_judge: List[Any] = field(default_factory=list)
        summary: str = ""

    def generate_id(prefix: str = "") -> str:
        import uuid
        return f"{prefix}{uuid.uuid4().hex}"


__all__ = [
    "CritiqueReport",
    "DecompositionPlan",
    "Feedback",
    "GauntletDefinition",
    "GauntletExecution",
    "GauntletRoundRule",
    "GauntletRoundResult",
    "GauntletRoundStatus",
    "ModelConfig",
    "SolutionAttempt",
    "SubProblem",
    "Team",
    "ValidationCheckpoint",
    "ValidationResult",
    "VerificationReport",
    "WorkflowState",
    "generate_id",
]
