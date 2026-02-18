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

from .bubblelab_api_client import (
    BubbleLabAPIClient,
    BubbleLabAPIClientConfig,
    get_bubblelab_client,
    BubbleLabAPIClientError,
    BubbleLabAPIConnectionError,
    BubbleLabAPIResponseError
)

# Integration components
from .openevolve_integration import (
    OpenEvolveWorkflowType,
    OpenEvolveStage,
    OpenEvolveIntegrationConfig,
    WorkflowComplexityAnalysis,
    MAKERWorkflowDecision,
    OpenEvolveMDAPIntegration,
    get_openevolve_integration
)

from .bubblelab_ui_integration import (
    UIComponent,
    UIState,
    ComplexityAnalysisResult,
    MAKERVotingDisplay,
    BubbleLabUIIntegration,
    get_bubblelab_ui_integration
)

from .integration_manager import (
    IntegrationStatus,
    IntegrationHealth,
    ComprehensiveIntegrationManager,
    get_integration_manager
)

# Monitoring components (optional, import separately)
# from .monitoring_dashboard import AdapterMonitor, get_monitor
# from .prometheus_exporter import PrometheusMetricsExporter, get_prometheus_exporter

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
    "RedFlagSeverity",
    # BubbleLab API Client
    "BubbleLabAPIClient",
    "BubbleLabAPIClientConfig",
    "get_bubblelab_client",
    "BubbleLabAPIClientError",
    "BubbleLabAPIConnectionError",
    "BubbleLabAPIResponseError",
    # OpenEvolve Integration
    "OpenEvolveWorkflowType",
    "OpenEvolveStage",
    "OpenEvolveIntegrationConfig",
    "WorkflowComplexityAnalysis",
    "MAKERWorkflowDecision",
    "OpenEvolveMDAPIntegration",
    "get_openevolve_integration",
    # BubbleLab UI Integration
    "UIComponent",
    "UIState",
    "ComplexityAnalysisResult",
    "MAKERVotingDisplay",
    "BubbleLabUIIntegration",
    "get_bubblelab_ui_integration",
    # Comprehensive Integration Manager
    "IntegrationStatus",
    "IntegrationHealth",
    "ComprehensiveIntegrationManager",
    "get_integration_manager"
]

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"
__description__ = "Adaptive MDAP/MAKER Adapter - Anti-Corruption Layer"
