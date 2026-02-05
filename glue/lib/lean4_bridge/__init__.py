"""
Lean 4 Bridge for RESE Formal Verification

This module provides the Python interface to Lean 4 for formal verification
of RESE constraints, theorems, and Functional Dependency Graphs (FDGs).

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Runtime Truth: Verify Lean 4 before use
- Circuit Breaker: Stop hammering if Lean 4 is down
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC

Usage:
    >>> from glue.lib.lean4_bridge import Lean4Interface
    >>> interface = Lean4Interface()
    >>> result = interface.formalize_constraint("forall x, P(x) -> Q(x)")
"""

__version__ = "1.0.0"
__author__ = "RESE Project"

from .lean4_interface import Lean4Interface, Lean4Error, Lean4TimeoutError, Lean4VerificationError

__all__ = [
    "Lean4Interface",
    "Lean4Error",
    "Lean4TimeoutError",
    "Lean4VerificationError",
]
