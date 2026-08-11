"""
OpenEvolve Unified Math Service - Compatibility Shim
=====================================================

This module provides backward compatibility for imports of
openevolve_unified_math_service by re-exporting from the
actual location in openevolve.unified_math_service.

Author: OpenEvolve Team
"""

# Re-export all public names from the actual module
from openevolve.unified_math_service import (
    UnifiedMathService,
    FormalizationResult,
    ProofResult,
    ElaborationResult,
    DocumentationResult,
    CAV_NLP_AVAILABLE,
    LEAN4_AVAILABLE,
    LEANAIDE_CLIENT_AVAILABLE,
)

__all__ = [
    "UnifiedMathService",
    "FormalizationResult",
    "ProofResult",
    "ElaborationResult",
    "DocumentationResult",
    "CAV_NLP_AVAILABLE",
    "LEAN4_AVAILABLE",
    "LEANAIDE_CLIENT_AVAILABLE",
]
