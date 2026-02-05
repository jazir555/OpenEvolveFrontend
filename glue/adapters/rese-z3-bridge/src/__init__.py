"""
RESE-Z3 Bridge Adapter

Provides unified interface for all RESE phases to access Z3 capabilities.

Following CLAUDE.md principles:
- Law of the "Air Gap": No imports from core-projects
- Law of Runtime Truth: Verify Z3 integration via probes
- Law of Idempotency: All operations safe to run 100x
- Circuit Breaker Pattern: Detect Z3 failures
- Structured Logging: JSON with correlation_id
- Law of Configuration Explicitness: All config via environment
- Law of UTC: All timestamps in UTC ISO-8601

Author: RESE Team
Created: 2026-02-04
"""

__version__ = "1.0.0"

from .rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig
from .rese_z3_client import Z3Client, Z3ClientError
from .rese_z3_schema import (
    CanonicalSolverRequest,
    CanonicalSolverResponse,
    CanonicalTheoremRequest,
    CanonicalTheoremResponse,
    canonical_to_z3_request,
    z3_to_canonical_response,
)

__all__ = [
    "RESEZ3Bridge",
    "RESEZ3BridgeConfig",
    "Z3Client",
    "Z3ClientError",
    "CanonicalSolverRequest",
    "CanonicalSolverResponse",
    "CanonicalTheoremRequest",
    "CanonicalTheoremResponse",
    "canonical_to_z3_request",
    "z3_to_canonical_response",
]
