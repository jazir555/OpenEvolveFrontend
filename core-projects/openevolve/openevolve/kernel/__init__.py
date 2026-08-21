"""Kernel subpackage: shared problem/decomposition schema for integrations.

This subpackage mirrors the ``openevolve.kernel`` namespace expected by the
flat ``engines/`` scripts (which do ``from openevolve.kernel.schema import ...``).
The installed ``openevolve 0.3.2`` distribution does not expose this path, so
when ``core-projects/openevolve`` is placed ahead of site-packages on
``sys.path`` (see the project's loader / conftest), this module provides the
real package.

Resolver order (no ``.pth``, no site-packages edits): the project adds
``core-projects/openevolve`` to ``sys.path`` before importing ``openevolve``, so
``import openevolve.kernel`` resolves here rather than to the installed package.
"""

from __future__ import annotations

from . import schema
from .schema import (
    generate_id,
    ProblemType,
    SubProblemStatus,
    DomainContext,
    ComplexityScore,
    Constraint,
    SuccessCriterion,
    ProblemDefinition,
    SubProblem,
    ModelConfig,
    WorkflowState,
    MathematicalDomain,
)

__all__ = [
    "schema",
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
]
