"""
Base Optimization Interface for OpenEvolve

This module defines the abstract interface that all optimization implementations must follow.
It provides a consistent API for numerical optimization across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import numpy as np


class OptimizationType(Enum):
    """Types of optimization problems."""
    UNCONSTRAINED = "unconstrained"
    CONSTRAINED = "constrained"
    PHYSICS_INFORMED = "physics_informed"
    SYSTEM_IDENTIFICATION = "system_identification"
    DIFFERENTIABLE_PROGRAMMING = "differentiable_programming"


class ProblemType(Enum):
    """Specific problem categories."""
    ODE = "ordinary_differential_equation"
    PDE = "partial_differential_equation"
    OPTIMIZATION = "optimization"
    CONTROL = "control"
    ESTIMATION = "estimation"


class OptimizationResult:
    """
    Standard result structure for optimization operations.
    """

    def __init__(
        self,
        success: bool,
        optimal_value: float,
        optimal_variables: Union[np.ndarray, Dict[str, Any]],
        iterations: int,
        convergence_history: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        self.success = success
        self.optimal_value = optimal_value
        self.optimal_variables = optimal_variables
        self.iterations = iterations
        self.convergence_history = convergence_history or []
        self.metadata = metadata or {}
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "optimal_value": self.optimal_value,
            "optimal_variables": self.optimal_variables.tolist() if isinstance(
                self.optimal_variables, np.ndarray
            ) else self.optimal_variables,
            "iterations": self.iterations,
            "convergence_history": self.convergence_history,
            "metadata": self.metadata,
            "error_message": self.error_message
        }


class OptimizationProblem:
    """
    Standard problem definition structure.
    """

    def __init__(
        self,
        problem_type: ProblemType,
        objective_function: Optional[callable] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        variables: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        physics_constraints: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.problem_type = problem_type
        self.objective_function = objective_function
        self.constraints = constraints or []
        self.variables = variables or {}
        self.parameters = parameters or {}
        self.physics_constraints = physics_constraints or {}
        self.bounds = bounds or {}

    def validate(self) -> bool:
        """Validate problem definition."""
        if self.problem_type == ProblemType.OPTIMIZATION and self.objective_function is None:
            return False
        return True


class OptimizationInterface(ABC):
    """
    Abstract base class for optimization implementations.

    This interface defines the contract that all optimization adapters must implement,
    ensuring consistency across different backend technologies (NeuroMANCER, etc.).
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the optimization engine with the given configuration.

        Args:
            config: Configuration dictionary containing:
                - pytorch_env: Name of conda environment for PyTorch
                - device: Device to use ('cuda' or 'cpu')
                - max_workers: Number of parallel workers
                - timeout: Operation timeout in seconds
                - cache_enabled: Whether to enable caching
                - cache_ttl: Cache time-to-live in seconds

        Returns:
            True if initialization was successful, False otherwise.

        Raises:
            ConfigurationError: If configuration is invalid
            ConnectionError: If connection to backend fails
        """
        pass

    @abstractmethod
    async def solve(
        self,
        problem: OptimizationProblem,
        optimization_type: OptimizationType = OptimizationType.UNCONSTRAINED,
        solver_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Solve an optimization problem.

        Args:
            problem: OptimizationProblem instance defining the problem
            optimization_type: Type of optimization to perform
            solver_params: Optional solver-specific parameters

        Returns:
            OptimizationResult containing solution details

        Raises:
            ValidationError: If problem definition is invalid
            SolverError: If solver fails to find a solution
            TimeoutError: If solver exceeds timeout
        """
        pass

    @abstractmethod
    async def identify_system(
        self,
        data: Dict[str, Any],
        model_structure: Optional[Dict[str, Any]] = None,
        physics_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform physics-informed system identification.

        Args:
            data: Dictionary containing:
                - inputs: Input data
                - outputs: Output data
                - timestamps: Optional time stamps
            model_structure: Optional model structure specification
            physics_constraints: Optional physics-based constraints

        Returns:
            Dictionary containing:
                - model: Identified model parameters
                - metrics: Model fit metrics
                - predictions: Model predictions on training data
                - residual: Residuals between predictions and data

        Raises:
            ValidationError: If data is invalid
            IdentificationError: If identification fails
        """
        pass

    @abstractmethod
    async def solve_ode(
        self,
        ode_definition: Dict[str, Any],
        initial_conditions: Dict[str, Any],
        time_span: Tuple[float, float],
        method: str = "automatic"
    ) -> Dict[str, Any]:
        """
        Solve an ordinary differential equation.

        Args:
            ode_definition: Dictionary defining the ODE system
                - equations: List of equation strings or functions
                - variables: Variable names
                - parameters: Parameter values
            initial_conditions: Initial values for variables
            time_span: (t_start, t_end) tuple
            method: Solver method to use

        Returns:
            Dictionary containing:
                - solution: Time series solution
                - time_points: Time points for solution
                - success: Whether solve was successful
                - metrics: Solver performance metrics
        """
        pass

    @abstractmethod
    async def solve_pde(
        self,
        pde_definition: Dict[str, Any],
        boundary_conditions: Dict[str, Any],
        initial_conditions: Optional[Dict[str, Any]] = None,
        domain: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a partial differential equation.

        Args:
            pde_definition: Dictionary defining the PDE
                - equation: PDE string or function
                - variables: Spatial and temporal variables
                - parameters: Parameter values
            boundary_conditions: Boundary condition specifications
            initial_conditions: Initial condition specifications
            domain: Spatial domain definition

        Returns:
            Dictionary containing:
                - solution: Field solution
                - grid: Spatial/temporal grid
                - success: Whether solve was successful
                - metrics: Solver performance metrics
        """
        pass

    @abstractmethod
    async def constrained_optimization(
        self,
        objective: callable,
        constraints: List[Dict[str, Any]],
        variables: Dict[str, Any],
        method: str = "interior_point"
    ) -> OptimizationResult:
        """
        Solve a constrained optimization problem.

        Args:
            objective: Objective function to minimize
            constraints: List of constraint dictionaries
                - type: 'eq' for equality, 'ineq' for inequality
                - function: Constraint function
                - bounds: Optional bounds
            variables: Variable definitions
                - names: Variable names
                - initial_values: Starting points
                - bounds: Optional (lower, upper) bounds
            method: Optimization method

        Returns:
            OptimizationResult with solution

        Raises:
            ValidationError: If problem is ill-defined
            SolverError: If solver fails
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the optimization engine state and connections.

        Returns:
            Dictionary containing:
                - is_valid: Overall validation status
                - checks: Individual check results
                - issues: List of any issues found
                - metrics: Performance and health metrics
                - device_info: Available devices
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the optimization engine.

        Performs cleanup and releases resources.

        Returns:
            True if shutdown was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get a problem template by name.

        Args:
            template_name: Name of the template (e.g., 'ode', 'pde', 'optimization')

        Returns:
            Dictionary containing template definition

        Raises:
            TemplateNotFoundError: If template doesn't exist
        """
        pass

    @abstractmethod
    async def list_templates(self) -> List[str]:
        """
        List available problem templates.

        Returns:
            List of template names
        """
        pass


class OptimizationError(Exception):
    """Base exception for optimization operations."""
    pass


class ConfigurationError(OptimizationError):
    """Raised when configuration is invalid."""
    pass


class ConnectionError(OptimizationError):
    """Raised when connection to backend fails."""
    pass


class ValidationError(OptimizationError):
    """Raised when validation fails."""
    pass


class SolverError(OptimizationError):
    """Raised when solver fails."""
    pass


class TimeoutError(OptimizationError):
    """Raised when operation times out."""
    pass


class IdentificationError(OptimizationError):
    """Raised when system identification fails."""
    pass


class TemplateNotFoundError(OptimizationError):
    """Raised when template is not found."""
    pass
