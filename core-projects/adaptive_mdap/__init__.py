"""
Adaptive MDAP - Massively Decomposed Agentic Processes with Adaptive Resource Allocation

This package implements the Adaptive-MAKER integration, combining the MAKER framework
(MDAP - Massively Decomposed Agentic Processes) with adaptive resource allocation
to achieve 30-50% cost reduction while maintaining quality within ±1% of baseline.

Based on: "Solving a Million-Step LLM Task with Zero Errors" by Meyerson et al.

Example:
    >>> from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    >>> from adaptive_mdap.core.types import SubProblem
    >>> 
    >>> # Create components
    >>> classifier = TaskComplexityClassifier()
    >>> allocator = AdaptiveMDAPAllocator()
    >>> 
    >>> # Create a sub-problem
    >>> sp = SubProblem(
    ...     id="example-001",
    ...     description="Solve this complex mathematical optimization problem",
    ...     domain="mathematics",
    ...     depth=3,
    ...     dependencies=[],
    ...     metadata={},
    ... )
    >>> 
    >>> # Compute complexity and allocate resources
    >>> complexity = classifier.compute_complexity(sp)
    >>> config = allocator.allocate_resources(complexity.overall_score)
    >>> print(f"Strategy: {config.strategy.value}, Agents: {config.n_agents}")
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve Integration Team"
__license__ = "MIT"

# Core components - import first as they're dependencies for others
from adaptive_mdap.core.types import (
    SubProblem,
    ComplexityScore,
    SolveConfig,
    SolveStrategy,
    AllocationDecision,
    ExecutionResult,
)

from adaptive_mdap.core.errors import (
    AdaptiveMDAPError,
    ClassificationError,
    AllocationError,
    ExecutionError,
    ConfigurationError,
    CacheError,
)

# Utilities - needed by other modules
from adaptive_mdap.utils.logger import get_logger, setup_logging
from adaptive_mdap.utils.metrics import get_metrics, MetricsCollector
from adaptive_mdap.utils.cache import (
    EmbeddingCache,
    FeatureCache,
    get_cache_stats,
)

# Classifiers
from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)

# Allocators
from adaptive_mdap.allocators.resource_allocator import (
    AdaptiveMDAPAllocator,
    AllocationContext,
    AllocationStats,
)

# Controllers
from adaptive_mdap.controllers.execution_controller import (
    AdaptiveExecutionController,
    SolutionAttempt,
    SolutionStatus,
    ExecutionMetrics,
)

# Configuration
from adaptive_mdap.config.profiles import (
    ConfigProfile,
    get_profile_config,
    load_profile,
)

# Monitoring - with error handling
try:
    from adaptive_mdap.monitoring.health import (
        HealthChecker,
        HealthCheckResult,
        ComponentStatus,
        get_health_checker,
        check_health,
    )
except ImportError as _e:
    HealthChecker = None
    HealthCheckResult = None
    ComponentStatus = None
    get_health_checker = None
    check_health = None

try:
    from adaptive_mdap.monitoring.dashboard import (
        DashboardGenerator,
        DashboardPanel,
        DashboardConfig,
        get_dashboard,
        get_summary,
        get_full_dashboard,
        get_prometheus_metrics,
    )
except ImportError as _e:
    DashboardGenerator = None
    DashboardPanel = None
    DashboardConfig = None
    get_dashboard = None
    get_summary = None
    get_full_dashboard = None
    get_prometheus_metrics = None

try:
    from adaptive_mdap.monitoring.alerts import (
        AlertingEngine,
        Alert,
        AlertRule,
        AlertSeverity,
        AlertStatus,
        get_alerting_engine,
        check_and_alert,
        get_active_alerts,
    )
except ImportError as _e:
    AlertingEngine = None
    Alert = None
    AlertRule = None
    AlertSeverity = None
    AlertStatus = None
    get_alerting_engine = None
    check_and_alert = None
    get_active_alerts = None

# Integrations - with error handling
try:
    from adaptive_mdap.integrations.crewai_integration import (
        CrewAIIntegration,
        AdaptiveCrewConfig,
    )
except ImportError as _e:
    CrewAIIntegration = None
    AdaptiveCrewConfig = None

try:
    from adaptive_mdap.integrations.subproblem_solver_integration import (
        AdaptiveSubProblemSolver,
        AdaptiveSolverConfig,
        create_adaptive_solver,
    )
except ImportError as _e:
    AdaptiveSubProblemSolver = None
    AdaptiveSolverConfig = None
    create_adaptive_solver = None

try:
    from adaptive_mdap.integrations.workflow_engine_integration import (
        AdaptiveWorkflowIntegration,
        AdaptiveWorkflowConfig,
        get_adaptive_workflow,
        configure_adaptive_workflow,
        adaptive_solve_subproblem,
    )
    WORKFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    WORKFLOW_INTEGRATION_AVAILABLE = False
    AdaptiveWorkflowIntegration = None
    AdaptiveWorkflowConfig = None
    get_adaptive_workflow = None
    configure_adaptive_workflow = None
    adaptive_solve_subproblem = None

# Tools - with error handling
try:
    from adaptive_mdap.tools.cost_calculator import (
        CostCalculator,
        APIPricing,
        TokenUsage,
        WorkloadDistribution,
        StrategyCost,
    )
except ImportError as _e:
    CostCalculator = None
    APIPricing = None
    TokenUsage = None
    WorkloadDistribution = None
    StrategyCost = None


__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    
    # Core Types
    "SubProblem",
    "ComplexityScore",
    "SolveConfig",
    "SolveStrategy",
    "AllocationDecision",
    "ExecutionResult",
    
    # Core Errors
    "AdaptiveMDAPError",
    "ClassificationError",
    "AllocationError",
    "ExecutionError",
    "ConfigurationError",
    "CacheError",
    
    # Classifiers
    "TaskComplexityClassifier",
    "ClassifierConfig",
    
    # Allocators
    "AdaptiveMDAPAllocator",
    "AllocationContext",
    "AllocationStats",
    
    # Controllers
    "AdaptiveExecutionController",
    "SolutionAttempt",
    "SolutionStatus",
    "ExecutionMetrics",
    
    # Integrations
    "CrewAIIntegration",
    "AdaptiveCrewConfig",
    "AdaptiveSubProblemSolver",
    "AdaptiveSolverConfig",
    "create_adaptive_solver",
    "AdaptiveWorkflowIntegration",
    "AdaptiveWorkflowConfig",
    "get_adaptive_workflow",
    "configure_adaptive_workflow",
    "adaptive_solve_subproblem",
    "WORKFLOW_INTEGRATION_AVAILABLE",
    
    # Tools
    "CostCalculator",
    "APIPricing",
    "TokenUsage",
    "WorkloadDistribution",
    "StrategyCost",
    
    # Configuration
    "ConfigProfile",
    "get_profile_config",
    "load_profile",
    
    # Monitoring - Health
    "HealthChecker",
    "HealthCheckResult",
    "ComponentStatus",
    "get_health_checker",
    "check_health",
    
    # Monitoring - Dashboard
    "DashboardGenerator",
    "DashboardPanel",
    "DashboardConfig",
    "get_dashboard",
    "get_summary",
    "get_full_dashboard",
    "get_prometheus_metrics",
    
    # Monitoring - Alerts
    "AlertingEngine",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "get_alerting_engine",
    "check_and_alert",
    "get_active_alerts",
    
    # Utilities
    "get_logger",
    "setup_logging",
    "get_metrics",
    "MetricsCollector",
    "EmbeddingCache",
    "FeatureCache",
    "get_cache_stats",
]
