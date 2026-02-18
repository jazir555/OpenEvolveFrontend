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

# Advanced Integration Components
from .openevolve_advanced import (
    WorkflowStage,
    TeamRole,
    SubProblemDecomposition,
    TeamSelectionResult,
    ResourceOptimization,
    WorkflowCheckpoint,
    AdvancedOpenEvolveIntegration,
    get_advanced_openevolve_integration
)

from .bubblelab_ui_advanced import (
    ChartType,
    AlertSeverity,
    ChartData,
    Alert,
    TimelineEvent,
    AdvancedBubbleLabUI,
    get_advanced_bubblelab_ui
)

from .gauntlet_advanced import (
    GauntletType,
    GauntletSeverity,
    GauntletConfig,
    GauntletExecution,
    GauntletPipeline,
    AggregatedGauntletResult,
    AdvancedGauntletIntegration,
    get_advanced_gauntlet_integration
)

from .icr_advanced import (
    PatternCluster,
    PatternSimilarityResult,
    AdaptiveThresholdResult,
    AdvancedICRIntegration,
    get_advanced_icr_integration
)

from .performance_optimization import (
    CachePolicy,
    CacheEntry,
    ResponseCache,
    ConnectionPool,
    AsyncMDAPAdapter,
    cached,
    batch_processor,
    PerformanceMonitor,
    get_async_adapter,
    get_performance_monitor
)

from .additional_systems_integration import (
    SystemStatus,
    SystemHealth,
    CrewAIIntegration,
    MCPToolsIntegration,
    KnowledgeEngineIntegration,
    LeanAideIntegration,
    Z3ProverIntegration,
    UnifiedSystemMonitor,
    get_unified_system_monitor
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
    "get_integration_manager",
    # Advanced OpenEvolve Integration
    "WorkflowStage",
    "TeamRole",
    "SubProblemDecomposition",
    "TeamSelectionResult",
    "ResourceOptimization",
    "WorkflowCheckpoint",
    "AdvancedOpenEvolveIntegration",
    "get_advanced_openevolve_integration",
    # Advanced BubbleLab UI
    "ChartType",
    "AlertSeverity",
    "ChartData",
    "Alert",
    "TimelineEvent",
    "AdvancedBubbleLabUI",
    "get_advanced_bubblelab_ui",
    # Advanced Gauntlet Integration
    "GauntletType",
    "GauntletSeverity",
    "GauntletConfig",
    "GauntletExecution",
    "GauntletPipeline",
    "AggregatedGauntletResult",
    "AdvancedGauntletIntegration",
    "get_advanced_gauntlet_integration",
    # Advanced ICR Integration
    "PatternCluster",
    "PatternSimilarityResult",
    "AdaptiveThresholdResult",
    "AdvancedICRIntegration",
    "get_advanced_icr_integration",
    # Performance Optimization
    "CachePolicy",
    "CacheEntry",
    "ResponseCache",
    "ConnectionPool",
    "AsyncMDAPAdapter",
    "cached",
    "batch_processor",
    "PerformanceMonitor",
    "get_async_adapter",
    "get_performance_monitor",
    # Additional Systems Integration
    "SystemStatus",
    "SystemHealth",
    "CrewAIIntegration",
    "MCPToolsIntegration",
    "KnowledgeEngineIntegration",
    "LeanAideIntegration",
    "Z3ProverIntegration",
    "UnifiedSystemMonitor",
    "get_unified_system_monitor"
]

__version__ = "2.0.0"
__author__ = "OpenEvolve Team"
__description__ = "Adaptive MDAP/MAKER Adapter - Anti-Corruption Layer"
