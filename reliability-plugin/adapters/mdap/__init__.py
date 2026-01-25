"""
MDAP Reliability Adapter Package
=================================

A production-ready adapter that adds Guardrails validation to MDAP voting
WITHOUT modifying MDAP core code.

This package provides:
- MDAPReliabilityAdapter: Main adapter class
- Factory functions for easy instantiation
- Convenience functions for one-off operations
- Vote-level validation
- Comprehensive error handling
- Graceful degradation

Air Gap Principle:
- NO imports from MDAP core source files
- NO modifications to MDAP core files
- All Guardrails logic lives in the ADAPTER
- Uses MDAP MCP tools as read-only interface

Example Usage:
    from reliability_plugin.adapters.mdap import MDAPReliabilityAdapter

    adapter = MDAPReliabilityAdapter()
    result = adapter.solve_with_validation(
        task="Solve this problem",
        mdap_k_ahead=5,
        validators=["vote_format", "json_structure"]
    )

    if result.success:
        print(f"Solution: {result.result}")
    else:
        print(f"Error: {result.error}")

Author: OpenEvolve Team
Version: 1.0.0
License: MIT
"""

from .mdap_reliability_adapter import (
    MDAPReliabilityAdapter,
    VoteValidationResult,
    MDAPSolveResult,
    RemediationStrategy,
    create_mdap_adapter,
    solve_with_guardrails
)

__all__ = [
    "MDAPReliabilityAdapter",
    "VoteValidationResult",
    "MDAPSolveResult",
    "RemediationStrategy",
    "create_mdap_adapter",
    "solve_with_guardrails"
]

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"
