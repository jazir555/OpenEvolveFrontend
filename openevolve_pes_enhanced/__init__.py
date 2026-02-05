"""
OpenEvolve PES Enhanced - Pure Enhancement Layer
================================================

This module provides a **non-invasive enhancement layer** for OpenEvolve's
existing PES integration. It wraps around the current implementation without
modifying any existing code.

All existing features are preserved:
- openevolve_agnostic_pes.py - Language-agnostic evolution (unchanged)
- openevolve_pes_integration.py - Current integration (unchanged)
- leanaide_pes_handler.py - Lean 4 theorem proving (unchanged)
- Z3 integration - Formal verification (unchanged)
- MAP-Elites, NSGA-II - Core evolution (unchanged)

New enhancements added:
- Cost-aware planning before evolution
- Dynamic execution monitoring
- Early stopping with convergence detection
- Budget tracking and optimization
- Summarization and learning extraction
- Evolution callbacks for iteration-level monitoring and control
- REST API for external integration
- Workflow Engine integration with cost-aware execution

Usage:
    # Existing API still works exactly the same
    from openevolve_pes_integration import enhance_code
    
    # New enhanced API with cost optimization
    from openevolve_pes_enhanced import PESCostOptimizer
    optimizer = PESCostOptimizer()
    result = await optimizer.enhance_with_planning(code, problem, tests)
    
    # REST API (see API_INTEGRATION.md)
    from openevolve_pes_enhanced.api_routes import router as pes_enhanced_router
    app.include_router(pes_enhanced_router)
    
    # Workflow Engine integration with PES cost tracking
    from openevolve_pes_enhanced import run_sovereign_workflow_with_pes
    result = await run_sovereign_workflow_with_pes(
        workflow_state,
        content_analyzer_team=...,
        planner_team=...,
        max_cost_usd=10.0  # Enable cost tracking
    )
"""

__version__ = "1.0.0"

# Core enhancement components (new)
from .cost_optimizer import (
    CostOptimizer,
    BudgetTracker,
    EfficiencyMetrics,
    CostAwarePlanner
)

from .execution_monitor import (
    ExecutionMonitor,
    ConvergenceDetector,
    EarlyStoppingController
)

from .strategy_enhancer import (
    StrategyEnhancer,
    AdaptiveParameterTuner,
    CostAwareStrategySelector
)

from .summarization_engine import (
    SummarizationEngine,
    InsightExtractor,
    LearningCapture
)

from .budget_enforcer import (
    BudgetEnforcer,
    BudgetCheckResult,
    BudgetEnforcedResult
)

# Evolution callbacks for iteration-level monitoring
try:
    from .evolution_callbacks import (
        EvolutionCallback,
        BudgetAwareCallback,
        MonitoringCallback,
        LoggingCallback,
        CompositeCallback,
        IterationMetrics,
        EvolutionContext,
        EvolutionState,
        create_budget_callback,
        create_monitoring_callback,
        create_logging_callback,
        create_standard_callbacks,
    )
    from .monitored_engine import (
        MonitoredAgnosticPES,
        MonitoredEvolutionResult,
        CallbackEnabledEngine,
        create_monitored_engine,
    )
    _CALLBACKS_AVAILABLE = True
except ImportError:
    _CALLBACKS_AVAILABLE = False

from .integration_wrapper import (
    PESIntegrationWrapper,
    EnhancedAgnosticPES,
    EnhancedLeanHandler,
    create_cost_aware_enhancer,
    create_fully_enhanced
)

# Workflow Adapter (new)
from .workflow_adapter import (
    WorkflowPESAdapter,
    run_sovereign_workflow_with_pes,
    CostAwareWorkflowTracker,
    WorkflowCostMetrics,
    BudgetExceededError,
    AllocationDecision,
    SubProblemAllocation,
    create_cost_aware_workflow_config,
    WorkflowStatePESExtension,
)

# Configuration
from .config import PESEnhancedConfig

# API Routes (optional - requires FastAPI)
try:
    from .api_routes import router as pes_enhanced_router
    from .api_routes import get_pes_enhanced_router
    API_ROUTES_AVAILABLE = True
except ImportError:
    API_ROUTES_AVAILABLE = False
    pes_enhanced_router = None
    get_pes_enhanced_router = None

__all__ = [
    # Cost Optimization (new)
    "CostOptimizer",
    "BudgetTracker", 
    "EfficiencyMetrics",
    "CostAwarePlanner",
    
    # Execution Monitoring (new)
    "ExecutionMonitor",
    "ConvergenceDetector",
    "EarlyStoppingController",
    
    # Strategy Enhancement (new)
    "StrategyEnhancer",
    "AdaptiveParameterTuner",
    "CostAwareStrategySelector",
    
    # Summarization (new)
    "SummarizationEngine",
    "InsightExtractor",
    "LearningCapture",
    
    # Budget Enforcement (new)
    "BudgetEnforcer",
    "BudgetCheckResult",
    "BudgetEnforcedResult",
    
    # Evolution Callbacks (new)
    "EvolutionCallback",
    "BudgetAwareCallback",
    "MonitoringCallback",
    "LoggingCallback",
    "CompositeCallback",
    "IterationMetrics",
    "EvolutionContext",
    "EvolutionState",
    "create_budget_callback",
    "create_monitoring_callback",
    "create_logging_callback",
    "create_standard_callbacks",
    
    # Monitored Engine (new)
    "MonitoredAgnosticPES",
    "MonitoredEvolutionResult",
    "CallbackEnabledEngine",
    "create_monitored_engine",
    
    # Integration Wrappers (enhance existing)
    "PESIntegrationWrapper",
    "EnhancedAgnosticPES",
    "EnhancedLeanHandler",
    
    # Convenience functions
    "create_cost_aware_enhancer",
    "create_fully_enhanced",
    
    # Workflow Adapter (new)
    "WorkflowPESAdapter",
    "run_sovereign_workflow_with_pes",
    "CostAwareWorkflowTracker",
    "WorkflowCostMetrics",
    "BudgetExceededError",
    "AllocationDecision",
    "SubProblemAllocation",
    "create_cost_aware_workflow_config",
    "WorkflowStatePESExtension",
    
    # Config
    "PESEnhancedConfig",
    
    # API Routes (optional)
    "pes_enhanced_router",
    "get_pes_enhanced_router",
    "API_ROUTES_AVAILABLE",
]
