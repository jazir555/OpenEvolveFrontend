"""
Base Uncertainty Quantification Interface for OpenEvolve

This module defines the abstract interface that all UQ test function implementations must follow.
It provides a consistent API for uncertainty quantification and validation across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np


class SamplingMethod(Enum):
    """Sampling methods for UQ analysis."""
    MONTE_CARLO = "monte_carlo"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    HALTON = "halton"
    GRID = "grid"


class SensitivityMethod(Enum):
    """Sensitivity analysis methods."""
    SOBOL = "sobol"
    MORRIS = "morris"
    FAST = "fast"
    DELTA = "delta"


@dataclass
class ProbabilisticInput:
    """
    Defines a probabilistic input specification.

    Attributes:
        name: Input parameter name
        distribution: Distribution type (e.g., 'uniform', 'normal', 'beta')
        parameters: Distribution parameters (e.g., [loc, scale] for normal)
        bounds: Optional bounds for the input
    """
    name: str
    distribution: str
    parameters: List[float]
    bounds: Optional[tuple] = None


@dataclass
class UQResult:
    """
    Results from uncertainty quantification analysis.

    Attributes:
        function_name: Name of the test function
        input_samples: Sampled input points
        output_samples: Corresponding output values
        statistics: Statistical summaries (mean, variance, etc.)
        sensitivity: Sensitivity analysis results (if computed)
        metadata: Additional metadata about the analysis
    """
    function_name: str
    input_samples: np.ndarray
    output_samples: np.ndarray
    statistics: Dict[str, Any]
    sensitivity: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class UncertaintyQuantificationInterface(ABC):
    """
    Abstract base class for uncertainty quantification implementations.

    This interface defines the contract that all UQ adapters must implement,
    ensuring consistency across different UQ backends (uqtestfuns, SALib, etc.).
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the UQ system with the given configuration.

        Args:
            config: Configuration dictionary containing:
                - enabled: Whether UQ is enabled
                - cache_enabled: Whether to cache results
                - max_workers: Maximum parallel workers
                - timeout: Operation timeout in seconds

        Returns:
            True if initialization was successful, False otherwise.

        Raises:
            ConfigurationError: If configuration is invalid
            ImportError: If required dependencies are not available
        """
        pass

    @abstractmethod
    async def list_available_functions(self) -> List[str]:
        """
        List all available test functions.

        Returns:
            List of function names available for UQ analysis.

        Raises:
            RetrievalError: If function listing fails
        """
        pass

    @abstractmethod
    async def get_function_info(self, function_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific test function.

        Args:
            function_name: Name of the test function

        Returns:
            Dictionary containing:
                - name: Function name
                - description: Function description
                - input_dimension: Number of input dimensions
                - input_bounds: Bounds for each input
                - default_distributions: Default input distributions
                - references: Academic references (if available)

        Raises:
            ValidationError: If function name is invalid
            RetrievalError: If information retrieval fails
        """
        pass

    @abstractmethod
    async def define_probabilistic_inputs(
        self,
        inputs: List[ProbabilisticInput]
    ) -> Dict[str, Any]:
        """
        Define probabilistic input specifications for UQ analysis.

        Args:
            inputs: List of probabilistic input definitions

        Returns:
            Dictionary containing:
                - input_spec: Validated input specifications
                - correlations: Input correlation matrix (if applicable)
                - metadata: Additional metadata

        Raises:
            ValidationError: If input specifications are invalid
        """
        pass

    @abstractmethod
    async def sample_inputs(
        self,
        inputs: List[ProbabilisticInput],
        n_samples: int,
        method: SamplingMethod = SamplingMethod.MONTE_CARLO,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample input points from probabilistic specifications.

        Args:
            inputs: List of probabilistic input definitions
            n_samples: Number of samples to generate
            method: Sampling method to use
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, n_inputs) containing sampled inputs

        Raises:
            ValidationError: If input specifications are invalid
            SamplingError: If sampling fails
        """
        pass

    @abstractmethod
    async def evaluate_test_function(
        self,
        function_name: str,
        input_samples: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate test function on sampled inputs.

        Args:
            function_name: Name of the test function
            input_samples: Input points to evaluate (array of shape (n_samples, n_inputs))

        Returns:
            Array of output values (shape: (n_samples,))

        Raises:
            ValidationError: If function name or inputs are invalid
            EvaluationError: If function evaluation fails
        """
        pass

    @abstractmethod
    async def compute_statistics(
        self,
        output_samples: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute statistical summaries of output samples.

        Args:
            output_samples: Output values from function evaluation

        Returns:
            Dictionary containing:
                - mean: Mean of outputs
                - variance: Variance of outputs
                - std: Standard deviation
                - percentiles: Various percentiles (5th, 25th, 50th, 75th, 95th)
                - min/max: Minimum and maximum values
                - histogram: Histogram data (bins, counts)

        Raises:
            AnalysisError: If statistical computation fails
        """
        pass

    @abstractmethod
    async def compute_sensitivity(
        self,
        function_name: str,
        inputs: List[ProbabilisticInput],
        n_samples: int,
        method: SensitivityMethod = SensitivityMethod.SOBOL,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on test function.

        Args:
            function_name: Name of the test function
            inputs: List of probabilistic input definitions
            n_samples: Number of samples for analysis
            method: Sensitivity analysis method
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - method: Method used
                - first_order: First-order sensitivity indices
                - total_order: Total-order sensitivity indices
                - second_order: Second-order indices (if available)
                - confidence_intervals: Confidence intervals for indices

        Raises:
            ValidationError: If parameters are invalid
            AnalysisError: If sensitivity analysis fails
        """
        pass

    @abstractmethod
    async def run_uq_pipeline(
        self,
        function_name: str,
        inputs: List[ProbabilisticInput],
        n_samples: int,
        sampling_method: SamplingMethod = SamplingMethod.MONTE_CARLO,
        compute_sensitivity: bool = True,
        sensitivity_method: SensitivityMethod = SensitivityMethod.SOBOL,
        seed: Optional[int] = None
    ) -> UQResult:
        """
        Run complete UQ validation pipeline.

        This is a convenience method that orchestrates the full UQ workflow:
        1. Define inputs
        2. Sample input points
        3. Evaluate test function
        4. Compute statistics
        5. Perform sensitivity analysis (optional)

        Args:
            function_name: Name of the test function
            inputs: List of probabilistic input definitions
            n_samples: Number of samples for analysis
            sampling_method: Method for sampling inputs
            compute_sensitivity: Whether to compute sensitivity indices
            sensitivity_method: Method for sensitivity analysis
            seed: Random seed for reproducibility

        Returns:
            UQResult object containing all analysis results

        Raises:
            ValidationError: If parameters are invalid
            PipelineError: If pipeline execution fails
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the UQ system state and dependencies.

        Returns:
            Dictionary containing:
                - is_valid: Overall validation status
                - checks: Individual check results
                - dependencies: Status of required dependencies
                - issues: List of any issues found

        Raises:
            ValidationError: If validation itself fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the UQ system.

        Performs cleanup and releases resources.

        Returns:
            True if shutdown was successful, False otherwise

        Raises:
            ShutdownError: If shutdown fails
        """
        pass


class UQError(Exception):
    """Base exception for UQ operations."""
    pass


class ConfigurationError(UQError):
    """Raised when configuration is invalid."""
    pass


class ImportError(UQError):
    """Raised when required dependencies are not available."""
    pass


class ValidationError(UQError):
    """Raised when input validation fails."""
    pass


class SamplingError(UQError):
    """Raised when sampling operations fail."""
    pass


class EvaluationError(UQError):
    """Raised when function evaluation fails."""
    pass


class AnalysisError(UQError):
    """Raised when analysis operations fail."""
    pass


class PipelineError(UQError):
    """Raised when pipeline execution fails."""
    pass


class ShutdownError(UQError):
    """Raised when shutdown operations fail."""
    pass


class RetrievalError(UQError):
    """Raised when information retrieval fails."""
    pass
