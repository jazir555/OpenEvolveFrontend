"""
RESE-SCE Adapter Source Package

This package provides the SCE (Symbolic Constraint Engine) bridge
and DITO optimizer for RESE integration.
"""

from .sce_bridge import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType,
    ConstraintCategory,
    ConstraintNode,
    NodeStatus,
    ContradictionReport,
    Z3SCEBridge,
    create_sce,
    Z3_AVAILABLE,
)

__all__ = [
    "SymbolicConstraintEngine",
    "Constraint",
    "ConstraintType",
    "ConstraintCategory",
    "ConstraintNode",
    "NodeStatus",
    "ContradictionReport",
    "Z3SCEBridge",
    "create_sce",
    "Z3_AVAILABLE",
]
