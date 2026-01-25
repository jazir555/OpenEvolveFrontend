"""
Ψ₃ Constraint Inversion System - Core Module

Implements 10x complexity reduction through functional dependency analysis.

Author: Agent G1 (Ψ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

from .constraint import Constraint, ConstraintType, Metadata
from .expression import Expr, BoolExpr, ArithExpr, QuantExpr, BoolOp, ArithOp, Quantifier
from .constraint_inverter import PSI3Config, PSI3Result, ConstraintInverter

__all__ = [
    # Core structures
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
]
