"""
Integrations for Adaptive MDAP.

Provides integration with:
- CrewAI for orchestration
- SubProblemSolver for existing solver integration
- WorkflowEngine for workflow orchestration
- Cloud APIs for cost calculation
"""

from adaptive_mdap.integrations.crewai_integration import (
    CrewAIIntegration,
    AdaptiveCrewConfig,
)

from adaptive_mdap.integrations.subproblem_solver_integration import (
    AdaptiveSubProblemSolver,
    AdaptiveSolverConfig,
    create_adaptive_solver,
)

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

try:
    from adaptive_mdap.integrations.cloud_api_client import (
        CloudAPIClient,
        Provider,
        ModelPricing,
    )
    CLOUD_API_AVAILABLE = True
except ImportError:
    CLOUD_API_AVAILABLE = False
    CloudAPIClient = None
    Provider = None
    ModelPricing = None

__all__ = [
    # CrewAI Integration
    "CrewAIIntegration",
    "AdaptiveCrewConfig",
    # SubProblemSolver Integration
    "AdaptiveSubProblemSolver",
    "AdaptiveSolverConfig",
    "create_adaptive_solver",
    # Workflow Engine Integration
    "AdaptiveWorkflowIntegration",
    "AdaptiveWorkflowConfig",
    "get_adaptive_workflow",
    "configure_adaptive_workflow",
    "adaptive_solve_subproblem",
]

if CLOUD_API_AVAILABLE:
    __all__.extend([
        "CloudAPIClient",
        "Provider",
        "ModelPricing",
    ])
