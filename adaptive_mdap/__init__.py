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

# Core components
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

# Main components
from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)

from adaptive_mdap.allocators.resource_allocator import (
    AdaptiveMDAPAllocator,
    AllocationContext,
    AllocationStats,
)

from adaptive_mdap.controllers.execution_controller import (
    AdaptiveExecutionController,
    SolutionAttempt,
    SolutionStatus,
    ExecutionMetrics,
)

# Integrations
from adaptive_mdap.integrations.crewai_integration import (
    CrewAIIntegration,
    AdaptiveCrewConfig,
)

from adaptive_mdap.integrations.subproblem_solver_integration import (
    AdaptiveSubProblemSolver,
    AdaptiveSolverConfig,
    create_adaptive_solver,
)

# Tools
from adaptive_mdap.tools.cost_calculator import (
    CostCalculator,
    APIPricing,
    TokenUsage,
    WorkloadDistribution,
    StrategyCost,
)

# Configuration
from adaptive_mdap.config.profiles import (
    ConfigProfile,
    get_profile_config,
    load_profile,
)

# Monitoring
from adaptive_mdap.monitoring.health import (
    HealthChecker,
    HealthCheckResult,
    ComponentStatus,
    get_health_checker,
    check_health,
)

from adaptive_mdap.monitoring.dashboard import (
    DashboardGenerator,
    DashboardPanel,
    DashboardConfig,
    get_dashboard,
    get_summary,
    get_full_dashboard,
    get_prometheus_metrics,
)

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

# Utilities
from adaptive_mdap.utils.metrics import get_metrics
from adaptive_mdap.utils.logger import get_logger
from adaptive_mdap.utils.cache import (
    EmbeddingCache,
    FeatureCache,
    get_cache_stats,
)

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
    "get_metrics",
    "get_logger",
    "EmbeddingCache",
    "FeatureCache",
    "get_cache_stats",
]
