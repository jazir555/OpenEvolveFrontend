"""
NeuroMANCER Integration for OpenEvolve

This package provides physics-informed optimization, system identification,
and differential equation solving using NeuroMANCER as a backend.

Key components:
- NeuroMANCERAdapter: Interface to NeuroMANCER optimization engine
- HybridSolver: Combines LeanAide (symbolic) + NeuroMANCER (numerical)
- LeanAideNeuroMANCERBridge: High-level interface for common operations

Usage:
    from integrations.neuromancer import NeuroMANCERAdapter, HybridSolver

    # Initialize adapter
    adapter = NeuroMANCERAdapter()
    await adapter.initialize(config)

    # Solve optimization problem
    result = await adapter.solve(problem)

    # Or use hybrid solver with LeanAide
    hybrid = HybridSolver()
    await hybrid.initialize(config)
    result = await hybrid.solve_optimization_problem(problem)
"""

from .adapter import NeuroMANCERAdapter
from .bridge import HybridSolver, LeanAideNeuroMANCERBridge

__version__ = "0.1.0"
__author__ = "OpenEvolve Team"

__all__ = [
    "NeuroMANCERAdapter",
    "HybridSolver",
    "LeanAideNeuroMANCERBridge",
    "create_neuromancer_adapter",
    "create_hybrid_solver"
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
    "convergence_tolerance": 1e-6
}


def get_default_config() -> dict:
    """
    Get the default configuration for NeuroMANCER integration.

    Returns:
        Dictionary with default configuration values
    """
    return DEFAULT_CONFIG.copy()


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
