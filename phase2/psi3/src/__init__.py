"""
Ψ₃ Constraint Inversion System

Top-level package for Ψ₃ implementation.

Author: Agent G1 (Ψ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

__version__ = "0.1.0-alpha"
__author__ = "Agent G1"

# Core imports
from .core import (
    Constraint,
    ConstraintType,
    Metadata,
    Expr,
    BoolExpr,
    ArithExpr,
    QuantExpr,
    BoolOp,
    ArithOp,
    Quantifier,
    PSI3Config,
    PSI3Result,
    ConstraintInverter,
)

# Algorithms
from .algorithms import (
    syntactic_preprocessing,
    estimate_redundancy,
    build_dependency_graph,
    DependencyGraph,
)

# Solvers
from .solvers import SATInterface, SatResult

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Core
    "Constraint",
    "ConstraintType",
    "Metadata",
    "Expr",
    "BoolExpr",
    "ArithExpr",
    "QuantExpr",
    "BoolOp",
    "ArithOp",
    "Quantifier",

    # Main API
    "PSI3Config",
    "PSI3Result",
    "ConstraintInverter",

    # Algorithms
    "syntactic_preprocessing",
    "estimate_redundancy",
    "build_dependency_graph",
    "DependencyGraph",

    # Solvers
    "SATInterface",
    "SatResult",
]
