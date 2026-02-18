"""
Adaptive MDAP/MAKER Adapter - Glue Layer

Federation Constitution Compliant Adapter for Adaptive Multi-Dimensional
Adaptive Processing (MDAP) and MAKER Engine integration.

This package provides the Anti-Corruption Layer (ACL) that transforms
between external data formats and canonical schemas, ensuring isolation
from changes in core systems.

Components:
- AdaptiveMDAPAdapter: Main adapter for Adaptive MDAP operations
- MakerAdapter: Adapter for MAKER Engine operations
- Canonical schemas: Data models for ACL transformation

Usage:
    from glue.adapters.adaptive_mdap_adapter import (
        get_adapter,
        AdaptiveMDAPAdapterConfig,
        CanonicalSubProblem,
        CanonicalRequest,
        CanonicalResponse
    )

    # Create adapter (loads config from env vars)
    adapter = get_adapter()

    # Analyze complexity
    subproblem = CanonicalSubProblem(
        id="task-001",
        description="Implement secure authentication",
        domain="security",
        depth=3
    )
    response = adapter.analyze_complexity(subproblem)

    # Check result
    if response.status == TaskStatus.COMPLETED:
        print(f"Complexity: {response.complexity_score.overall_score}")
"""

from .adaptive_mdap_adapter import (
    AdaptiveMDAPAdapter,
    AdaptiveMDAPAdapterConfig,
    get_adapter,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    CanonicalStrategy,
    CanonicalRequest,
    CanonicalResponse,
    ProcessingDomain,
    AdaptationMode,
    TaskStatus
)

from .maker_adapter import (
    MakerAdapter,
    get_maker_adapter,
    CanonicalMakerConfig,
    CanonicalMakerStep,
    CanonicalAgentVote,
    CanonicalMakerResult,
    VotingMode,
    RedFlagSeverity
)

__all__ = [
    # Adaptive MDAP Adapter
    "AdaptiveMDAPAdapter",
    "AdaptiveMDAPAdapterConfig",
    "get_adapter",
    "CanonicalSubProblem",
    "CanonicalComplexityScore",
    "CanonicalStrategy",
    "CanonicalRequest",
    "CanonicalResponse",
    "ProcessingDomain",
    "AdaptationMode",
    "TaskStatus",
    # MAKER Adapter
    "MakerAdapter",
    "get_maker_adapter",
    "CanonicalMakerConfig",
    "CanonicalMakerStep",
    "CanonicalAgentVote",
    "CanonicalMakerResult",
    "VotingMode",
    "RedFlagSeverity"
]

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"
__description__ = "Adaptive MDAP/MAKER Adapter - Anti-Corruption Layer"
