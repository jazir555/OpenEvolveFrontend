"""
Integrations for Adaptive MDAP.

Provides integration with:
- CrewAI for orchestration
- SubProblemSolver for existing solver integration
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
]

if CLOUD_API_AVAILABLE:
    __all__.extend([
        "CloudAPIClient",
        "Provider",
        "ModelPricing",
    ])
