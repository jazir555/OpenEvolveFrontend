"""
Compatibility shim for legacy imports.

This module re-exports the Symbolic Constraint Engine from rese.core so
imports like `from symbolic_constraint_engine import ...` work.
"""
from __future__ import annotations


from rese.core.symbolic_constraint_engine import Constraint, ConstraintType, SymbolicConstraintEngine

__all__ = ["Constraint", "ConstraintType", "SymbolicConstraintEngine"]
