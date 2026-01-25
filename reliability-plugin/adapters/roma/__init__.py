"""
ROMA Reliability Adapter
========================

Wrapper that adds LMQL constraints and Guardrails validation to ROMA
without modifying ROMA core code (AIR GAP PRINCIPLE).

Architecture:
    ROMA Core (READ ONLY)
        ↓
    MCP Tools Interface (public API)
        ↓
    Reliability Adapter (LMQL + Guardrails)
        ↓
    Unified Bridge

Author: OpenEvolve Team
Version: 1.0.0
"""

from .roma_reliability_adapter import (
    RomaReliabilityAdapter,
    create_roma_adapter,
    solve_with_constraints,
    analyze_with_constraints
)

__all__ = [
    "RomaReliabilityAdapter",
    "create_roma_adapter",
    "solve_with_constraints",
    "analyze_with_constraints"
]
