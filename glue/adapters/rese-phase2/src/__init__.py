"""
RESE Phase II: Isomorphic Mapping Adapter

This package implements Phase II of the RESE (Recursive Epistemic Solvability Engine).
"""

__version__ = "1.0.0"
__author__ = "RESE Team"
__created__ = "2026-02-04"

from .phase2_executor import (
    IsomorphicMappingExecutor,
    StructureIdentifier,
    DependencyGraphBuilder,
    CrossDomainMapper,
    ConstraintInverter,
    ConstraintHardener,
    create_executor,
    is_available,
)

from .phase2_adapter import (
    Phase2Adapter,
    DeadLetterQueue,
)

__all__ = [
    # Executor
    "IsomorphicMappingExecutor",
    "StructureIdentifier",
    "DependencyGraphBuilder",
    "CrossDomainMapper",
    "ConstraintInverter",
    "ConstraintHardener",
    "create_executor",
    "is_available",

    # Adapter
    "Phase2Adapter",
    "DeadLetterQueue",
]
