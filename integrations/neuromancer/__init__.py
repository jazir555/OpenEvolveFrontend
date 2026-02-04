"""NeuroMANCER Integration for OpenEvolve

This package provides physics-informed optimization, system identification,
and differential equation solving using NeuroMANCER as a backend.

Key components:
- NeuroMANCERAdapter: Interface to NeuroMANCER optimization engine
- HybridSolver: Combines LeanAide (symbolic) + NeuroMANCER (numerical)
- LeanAideNeuroMANCERBridge: High-level interface for common operations
- Neural Operators: DeepONet, FNO, PINNs for physics-informed ML
- Physics Constraints: Conservation laws, thermodynamic constraints
- Scientific Domains: Pre-configured models for climate, fluids, structures, etc.
- KGPhysicsBridge: Bridge between Knowledge Graph and physics simulations

Usage:
    from integrations.neuromancer import NeuroMANCERAdapter, HybridSolver
    from integrations.neuromancer import NeuromancerAdapter, NeuralOperatorConfig
    from integrations.neuromancer import DomainLibrary, KGPhysicsBridge

    # Initialize neural operator adapter
    adapter = NeuromancerAdapter(device="cuda")
    await adapter.initialize(config)

    # Solve ODE/PDE
    result = await adapter.solve_ode(system, initial_conditions, t_span)
    
    # Use scientific domain
    fluid_domain = DomainLibrary.fluid_dynamics()
    result = fluid_domain.solve(problem)
    
    # Bridge KG and physics
    bridge = KGPhysicsBridge()
    physics_problem = bridge.kg_to_physics_problem(kg_subgraph)
    kg_updates = bridge.physics_solution_to_kg(solution)
"""

# Core adapter components
from .adapter import NeuroMANCERAdapter
from .bridge import HybridSolver, LeanAideNeuroMANCERBridge

# Neural operators
from .neural_operators import (
    NeuralOperatorType,
    NeuralOperatorConfig,
    SolutionResult,
    DynamicsModel,
    TrajectoryResult,
    CalibratedModel,
    NeuralOperatorBase,
    FNOOperator,
    DeepONetOperator,
    PINNOperator,
    NeuromancerAdapter,
    create_operator,
    MODEL_REGISTRY
)

# Physics constraints
from .physics_constraints import (
    ConstraintType,
    ConservationQuantity,
    ConstraintViolation,
    ConstraintConfig,
    PhysicsConstraint,
    ConservationLawConstraint,
    ThermodynamicConstraint,
    MechanicalConstraint,
    ElectromagneticConstraint,
    ChemicalConstraint,
    ConstraintLibrary,
    create_physics_loss
)

# Scientific domains
from .scientific_domains import (
    DomainType,
    DomainConfig,
    SimulationResult,
    ScientificDomain,
    ClimateModeling,
    FluidDynamics,
    StructuralMechanics,
    ChemicalKinetics,
    BiologicalSystems,
    DomainLibrary
)

# KG Physics bridge
from .kg_physics_bridge import (
    EntityPhysicsType,
    RelationshipPhysicsType,
    PhysicsProblem,
    KGUpdates,
    ConsistencyReport,
    InferredProperties,
    SimulationResultKG,
    KGPhysicsBridge
)

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"

__all__ = [
    # Core adapters
    "NeuroMANCERAdapter",
    "HybridSolver",
    "LeanAideNeuroMANCERBridge",
    "create_neuromancer_adapter",
    "create_hybrid_solver",
    
    # Neural operators
    "NeuralOperatorType",
    "NeuralOperatorConfig",
    "SolutionResult",
    "DynamicsModel",
    "TrajectoryResult",
    "CalibratedModel",
    "NeuralOperatorBase",
    "FNOOperator",
    "DeepONetOperator",
    "PINNOperator",
    "NeuromancerAdapter",
    "create_operator",
    "MODEL_REGISTRY",
    
    # Physics constraints
    "ConstraintType",
    "ConservationQuantity",
    "ConstraintViolation",
    "ConstraintConfig",
    "PhysicsConstraint",
    "ConservationLawConstraint",
    "ThermodynamicConstraint",
    "MechanicalConstraint",
    "ElectromagneticConstraint",
    "ChemicalConstraint",
    "ConstraintLibrary",
    "create_physics_loss",
    
    # Scientific domains
    "DomainType",
    "DomainConfig",
    "SimulationResult",
    "ScientificDomain",
    "ClimateModeling",
    "FluidDynamics",
    "StructuralMechanics",
    "ChemicalKinetics",
    "BiologicalSystems",
    "DomainLibrary",
    
    # KG Physics bridge
    "EntityPhysicsType",
    "RelationshipPhysicsType",
    "PhysicsProblem",
    "KGUpdates",
    "ConsistencyReport",
    "InferredProperties",
    "SimulationResultKG",
    "KGPhysicsBridge",
    
    # Configuration
    "DEFAULT_CONFIG",
    "get_default_config",
    
    # Convenience functions
    "quick_optimize",
    "quick_ode_solve",
    "quick_system_identification"
]


def create_neuromancer_adapter(config: dict = None) -> NeuroMANCERAdapter:
    """
    Factory function to create a NeuroMANCER adapter.

    Args:
        config: Optional configuration dictionary

    Returns:
        Initialized NeuroMANCERAdapter instance

    Example:
        >>> adapter = create_neuromancer_adapter({
        ...     "pytorch_env": "neuromancer_env",
        ...     "device": "cpu"
        ... })
    """
    adapter = NeuroMANCERAdapter()
    if config:
        # Note: Initialize is async, caller must await it
        adapter.config = config
    return adapter


def create_hybrid_solver(
    leanaide_client=None,
    config: dict = None
) -> HybridSolver:
    """
    Factory function to create a hybrid solver.

    Args:
        leanaide_client: Optional LeanAide client instance
        config: Optional configuration dictionary

    Returns:
        Initialized HybridSolver instance

    Example:
        >>> from integrations.neuromancer import create_hybrid_solver
        >>> hybrid = create_hybrid_solver(config={
        ...     "hybrid_mode": "sequential",
        ...     "max_iterations": 3
        ... })
    """
    solver = HybridSolver(leanaide_client=leanaide_client)
    if config:
        solver.hybrid_mode = config.get("hybrid_mode", "sequential")
        solver.max_iterations = config.get("max_iterations", 3)
        solver.convergence_tolerance = config.get("convergence_tolerance", 1e-6)
    return solver


# Default configuration
DEFAULT_CONFIG = {
    "pytorch_env": "neuromancer_env",
    "device": "cpu",
    "max_workers": 4,
    "timeout": 30,
    "cache_enabled": True,
    "cache_ttl": 3600,
    "hybrid_mode": "sequential",
    "max_iterations": 3,
    "convergence_tolerance": 1e-6,
    # Neural operator defaults
    "neural_operator": {
        "operator_type": "fno",
        "hidden_dim": 64,
        "num_layers": 4,
        "modes": 12,
        "activation": "gelu",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 1000,
        "use_physics_loss": True,
        "physics_weight": 1.0
    }
}


def get_default_config() -> dict:
    """
    Get the default configuration for NeuroMANCER integration.

    Returns:
        Dictionary with default configuration values
    """
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


# Convenience functions for common operations

async def quick_optimize(
    objective: str,
    constraints: list,
    variables: dict,
    config: dict = None
) -> dict:
    """
    Quick optimization interface for simple problems.

    Args:
        objective: Objective function description
        constraints: List of constraint descriptions
        variables: Variable definitions with bounds
        config: Optional configuration

    Returns:
        Optimization results as dictionary

    Example:
        >>> result = await quick_optimize(
        ...     objective="minimize x^2 + y^2",
        ...     constraints=["x + y >= 1"],
        ...     variables={"x": (0, 10), "y": (0, 10)}
        ... )
    """
    config = config or get_default_config()
    bridge = LeanAideNeuroMANCERBridge()
    await bridge.initialize(config)
    return await bridge.optimize(objective, constraints, variables)


async def quick_ode_solve(
    equation: str,
    initial_conditions: dict,
    time_span: tuple,
    config: dict = None
) -> dict:
    """
    Quick ODE solver interface.

    Args:
        equation: ODE equation string
        initial_conditions: Initial values for variables
        time_span: (t_start, t_end) tuple
        config: Optional configuration

    Returns:
        Solution as dictionary

    Example:
        >>> result = await quick_ode_solve(
        ...     equation="dy/dt = -k*y",
        ...     initial_conditions={"y": 1.0, "k": 0.5},
        ...     time_span=(0, 10)
        ... )
    """
    config = config or get_default_config()
    bridge = LeanAideNeuroMANCERBridge()
    await bridge.initialize(config)
    return await bridge.solve_differential_equation(
        equation=equation,
        equation_type="ode",
        conditions={"initial": initial_conditions}
    )


async def quick_system_identification(
    input_data: list,
    output_data: list,
    config: dict = None
) -> dict:
    """
    Quick system identification interface.

    Args:
        input_data: System input trajectories
        output_data: System output trajectories
        config: Optional configuration

    Returns:
        Identified system model as dictionary

    Example:
        >>> result = await quick_system_identification(
        ...     input_data=[[1, 2, 3], [2, 3, 4]],
        ...     output_data=[[2, 4, 6], [3, 5, 7]]
        ... )
    """
    config = config or get_default_config()
    bridge = LeanAideNeuroMANCERBridge()
    await bridge.initialize(config)
    return await bridge.identify_system(input_data, output_data)


async def quick_neural_operator_solve(
    problem_type: str,
    equation: str,
    domain: dict,
    config: dict = None
) -> SolutionResult:
    """
    Quick neural operator solver interface.

    Args:
        problem_type: 'ode' or 'pde'
        equation: Equation description
        domain: Domain definition
        config: Optional configuration

    Returns:
        SolutionResult with solution data

    Example:
        >>> result = await quick_neural_operator_solve(
        ...     problem_type="pde",
        ...     equation="laplace",
        ...     domain={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
        ... )
    """
    config = config or get_default_config()
    adapter = NeuromancerAdapter(device=config.get("device", "cpu"))
    adapter.initialize(config)
    
    if problem_type == "ode":
        return adapter.solve_ode(
            system=equation,
            initial_conditions=domain.get("initial_conditions", {}),
            t_span=domain.get("time_span", (0, 1))
        )
    else:
        return adapter.solve_pde(
            equation=equation,
            domain=domain,
            boundary_conditions=domain.get("boundary_conditions", {})
        )
