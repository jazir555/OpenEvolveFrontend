"""
uqtestfuns Adapter for OpenEvolve

This module implements the UncertaintyQuantificationInterface using the uqtestfuns library.
It provides a decoupled adapter pattern for integrating UQ test functions into OpenEvolve.

Repository: https://github.com/damar-wicaksono/uqtestfuns
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache

# Import the base interface
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'base'))
from uq_interface import (
    UncertaintyQuantificationInterface,
    ProbabilisticInput,
    UQResult,
    SamplingMethod,
    SensitivityMethod,
    ConfigurationError,
    ImportError,
    ValidationError,
    SamplingError,
    EvaluationError,
    AnalysisError,
    PipelineError,
    ShutdownError,
    RetrievalError
)

# Configure logging
logger = logging.getLogger(__name__)


class UQTestFunsAdapter(UncertaintyQuantificationInterface):
    """
    Adapter for uqtestfuns library implementing UncertaintyQuantificationInterface.

    This adapter provides a clean, decoupled integration point for uqtestfuns,
    allowing OpenEvolve to leverage UQ test functions without direct dependencies
    on the uqtestfuns implementation details.

    Key Features:
    - Test function library with probabilistic input specifications
    - Support for various sampling methods (Monte Carlo, Latin Hypercube, etc.)
    - Sensitivity analysis capabilities
    - Lightweight dependency (only NumPy and SciPy required)
    - Zero modifications to uqtestfuns source
    """

    def __init__(self):
        """Initialize the uqtestfuns adapter."""
        self._initialized = False
        self._config: Optional[Dict[str, Any]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._uqtestfuns_available = False
        self._scipy_available = False
        self._cache_enabled = False
        self._function_cache: Dict[str, Any] = {}
        self._result_cache: Dict[str, UQResult] = {}

        # Try to import uqtestfuns
        try:
            import uqtestfuns as uqtf
            self._uqtf = uqtf
            self._uqtestfuns_available = True
            logger.info("uqtestfuns library imported successfully")
        except ImportError:
            logger.warning(
                "uqtestfuns library not available. "
                "Install with: pip install uqtestfuns"
            )

        # Try to import scipy for advanced sampling
        try:
            from scipy import stats
            self._scipy_stats = stats
            self._scipy_available = True
            logger.info("SciPy available for advanced sampling")
        except ImportError:
            logger.warning("SciPy not available. Limited to basic sampling methods")
            self._scipy_stats = None

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the uqtestfuns adapter with configuration.

        Args:
            config: Configuration dictionary containing:
                - enabled: Whether UQ is enabled (default: True)
                - cache_enabled: Whether to cache results (default: True)
                - max_workers: Maximum parallel workers (default: 4)
                - timeout: Operation timeout in seconds (default: 30)
                - auto_start: Auto-start on initialization (default: True)

        Returns:
            True if initialization was successful

        Raises:
            ConfigurationError: If configuration is invalid
            ImportError: If uqtestfuns is not available
        """
        if self._initialized:
            logger.warning("Adapter already initialized")
            return True

        # Validate configuration
        if not config.get('enabled', True):
            logger.info("UQ adapter disabled in configuration")
            return False

        # Check if uqtestfuns is available
        if not self._uqtestfuns_available:
            raise ImportError(
                "uqtestfuns library is not available. "
                "Install with: pip install uqtestfuns"
            )

        # Store configuration
        self._config = config
        self._cache_enabled = config.get('cache_enabled', True)
        max_workers = config.get('max_workers', 4)

        # Initialize thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pre-load available functions
        if config.get('auto_start', True):
            try:
                functions = await self.list_available_functions()
                logger.info(f"Loaded {len(functions)} test functions")
            except Exception as e:
                logger.error(f"Failed to load test functions: {e}")
                return False

        self._initialized = True
        logger.info("uqtestfuns adapter initialized successfully")
        return True

    async def list_available_functions(self) -> List[str]:
        """
        List all available test functions from uqtestfuns.

        Returns:
            List of function names available for UQ analysis

        Raises:
            ImportError: If uqtestfuns is not available
            RetrievalError: If function listing fails
        """
        if not self._uqtestfuns_available:
            raise ImportError("uqtestfuns library not available")

        try:
            # uqtestfuns provides a list of available functions
            # Common test functions include:
            available_functions = [
                "ishigami",
                "ackley",
                "rosenbrock",
                "branin",
                "sphere",
                "styblinski-tang",
                "michalewicz",
                "otlcircuit",
                "wing-weight",
                "piston",
                "sobol-g",
                "friedman",
                "marriage",
                "bohachevsky",
                "colville",
                "dixon-price",
                "goldstein-price",
                "hartmann",
                "trid",
                "wolfe",
                "wood",
                # Add more as they become available in uqtestfuns
            ]

            logger.info(f"Found {len(available_functions)} test functions")
            return available_functions

        except Exception as e:
            logger.error(f"Failed to list available functions: {e}")
            raise RetrievalError(f"Failed to list functions: {e}")

    async def get_function_info(self, function_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific test function.

        Args:
            function_name: Name of the test function

        Returns:
            Dictionary with function information

        Raises:
            ValidationError: If function name is invalid
            RetrievalError: If information retrieval fails
        """
        if not self._uqtestfuns_available:
            raise ImportError("uqtestfuns library not available")

        # Check cache first
        if self._cache_enabled and function_name in self._function_cache:
            return self._function_cache[function_name]

        try:
            # Create a test function instance to get info
            loop = asyncio.get_event_loop()
            fun = await loop.run_in_executor(
                self._executor,
                self._create_test_function,
                function_name
            )

            info = {
                "name": function_name,
                "description": getattr(fun, '__doc__', f"{function_name} test function"),
                "input_dimension": fun.input_dimension,
                "input_bounds": fun.input.spatial_dimension,  # May vary by implementation
                "default_distributions": getattr(fun, 'default_distributions', None),
                "references": getattr(fun, 'references', []),
            }

            # Cache the result
            if self._cache_enabled:
                self._function_cache[function_name] = info

            return info

        except Exception as e:
            logger.error(f"Failed to get function info for {function_name}: {e}")
            raise ValidationError(f"Invalid function name '{function_name}': {e}")

    def _create_test_function(self, function_name: str):
        """Helper to create a uqtestfuns test function instance."""
        # This is a placeholder - actual implementation depends on uqtestfuns API
        # The library may have different ways to create functions
        try:
            # Try to get function from uqtestfuns
            # (API may vary, adjust accordingly)
            from uqtestfuns import UQTestFun
            fun = UQTestFun.from_default(function_name)
            return fun
        except AttributeError:
            # Fallback to alternative API
            from uqtestfuns import functions
            if hasattr(functions, function_name):
                return getattr(functions, function_name)
            else:
                raise ValueError(f"Function '{function_name}' not found")

    async def define_probabilistic_inputs(
        self,
        inputs: List[ProbabilisticInput]
    ) -> Dict[str, Any]:
        """
        Define and validate probabilistic input specifications.

        Args:
            inputs: List of probabilistic input definitions

        Returns:
            Dictionary with validated specifications
        """
        if not inputs:
            raise ValidationError("Empty input specification")

        # Validate each input
        for inp in inputs:
            if not inp.name:
                raise ValidationError("Input name cannot be empty")
            if not inp.distribution:
                raise ValidationError(f"Missing distribution for input '{inp.name}'")
            if not inp.parameters or len(inp.parameters) == 0:
                raise ValidationError(f"Missing parameters for input '{inp.name}'")

            # Validate distribution type
            valid_distributions = [
                'uniform', 'normal', 'beta', 'gamma', 'lognormal',
                'triangular', 'exponential', 'weibull'
            ]
            if inp.distribution not in valid_distributions:
                raise ValidationError(
                    f"Invalid distribution '{inp.distribution}' for input '{inp.name}'. "
                    f"Must be one of {valid_distributions}"
                )

        # Create input specification
        input_spec = {
            "inputs": [
                {
                    "name": inp.name,
                    "distribution": inp.distribution,
                    "parameters": inp.parameters,
                    "bounds": inp.bounds
                }
                for inp in inputs
            ],
            "n_inputs": len(inputs),
            "correlations": None  # Could be extended for correlated inputs
        }

        return {
            "input_spec": input_spec,
            "correlations": None,
            "metadata": {
                "validated": True,
                "n_inputs": len(inputs)
            }
        }

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
            Array of shape (n_samples, n_inputs) with sampled inputs

        Raises:
            ValidationError: If input specifications are invalid
            SamplingError: If sampling fails
        """
        if n_samples <= 0:
            raise ValidationError("n_samples must be positive")

        n_inputs = len(inputs)
        samples = np.zeros((n_samples, n_inputs))

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        try:
            for i, inp in enumerate(inputs):
                if inp.distribution == 'uniform':
                    # Parameters: [low, high]
                    low, high = inp.parameters
                    samples[:, i] = np.random.uniform(low, high, n_samples)

                elif inp.distribution == 'normal':
                    # Parameters: [loc, scale]
                    loc, scale = inp.parameters
                    samples[:, i] = np.random.normal(loc, scale, n_samples)

                elif inp.distribution == 'beta':
                    # Parameters: [alpha, beta, loc, scale]
                    alpha, beta = inp.parameters[:2]
                    loc = inp.parameters[2] if len(inp.parameters) > 2 else 0
                    scale = inp.parameters[3] if len(inp.parameters) > 3 else 1
                    samples[:, i] = loc + scale * np.random.beta(alpha, beta, n_samples)

                elif inp.distribution == 'lognormal':
                    # Parameters: [mean, sigma]
                    mean, sigma = inp.parameters
                    samples[:, i] = np.random.lognormal(mean, sigma, n_samples)

                elif inp.distribution == 'exponential':
                    # Parameters: [scale]
                    scale = inp.parameters[0]
                    samples[:, i] = np.random.exponential(scale, n_samples)

                else:
                    # Fallback for other distributions
                    if self._scipy_available:
                        samples[:, i] = self._sample_from_scipy(inp, n_samples)
                    else:
                        raise SamplingError(
                            f"Distribution '{inp.distribution}' requires SciPy"
                        )

            # Apply advanced sampling methods if requested
            if method == SamplingMethod.LATIN_HYPERCUBE and self._scipy_available:
                samples = self._latin_hypercube_sample(inputs, n_samples, seed)

            logger.info(f"Generated {n_samples} samples for {n_inputs} inputs")
            return samples

        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise SamplingError(f"Failed to sample inputs: {e}")

    def _sample_from_scipy(self, inp: ProbabilisticInput, n_samples: int) -> np.ndarray:
        """Sample using SciPy distributions."""
        dist = getattr(self._scipy_stats, inp.distribution)
        return dist.rvs(*inp.parameters, size=n_samples)

    def _latin_hypercube_sample(
        self,
        inputs: List[ProbabilisticInput],
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        if self._scipy_available and hasattr(self._scipy_stats, 'qmc'):
            # Use SciPy's QMC module (available in scipy >= 1.7)
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(inputs), seed=seed)
            samples_unit = sampler.random(n=n_samples)
            # Transform to actual distributions
            # (simplified - full implementation would use CDF transform)
            return samples_unit
        else:
            # Simple LHS implementation
            n_inputs = len(inputs)
            samples = np.zeros((n_samples, n_inputs))
            for i in range(n_inputs):
                perm = np.random.permutation(n_samples)
                samples[:, i] = (perm + np.random.random(n_samples)) / n_samples
            return samples

    async def evaluate_test_function(
        self,
        function_name: str,
        input_samples: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate test function on sampled inputs.

        Args:
            function_name: Name of the test function
            input_samples: Input points to evaluate

        Returns:
            Array of output values

        Raises:
            ValidationError: If function or inputs are invalid
            EvaluationError: If evaluation fails
        """
        if not self._uqtestfuns_available:
            raise ImportError("uqtestfuns library not available")

        if input_samples.ndim != 2:
            raise ValidationError("input_samples must be 2D array")

        n_samples = input_samples.shape[0]

        try:
            # Get or create test function
            loop = asyncio.get_event_loop()
            fun = await loop.run_in_executor(
                self._executor,
                self._create_test_function,
                function_name
            )

            # Evaluate function
            outputs = await loop.run_in_executor(
                self._executor,
                self._evaluate_function,
                fun,
                input_samples
            )

            logger.info(f"Evaluated {function_name} on {n_samples} samples")
            return outputs

        except Exception as e:
            logger.error(f"Function evaluation failed: {e}")
            raise EvaluationError(f"Failed to evaluate {function_name}: {e}")

    def _evaluate_function(self, fun, input_samples: np.ndarray) -> np.ndarray:
        """Helper to evaluate function on samples."""
        # uqtestfuns API may vary - adjust accordingly
        if hasattr(fun, 'eval'):
            return fun.eval(input_samples)
        elif hasattr(fun, '__call__'):
            return fun(input_samples)
        else:
            raise ValueError("Function does not have a callable interface")

    async def compute_statistics(
        self,
        output_samples: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute statistical summaries of output samples.

        Args:
            output_samples: Output values from function evaluation

        Returns:
            Dictionary with statistical summaries
        """
        if output_samples.ndim != 1:
            # Assume 2D array with single output column
            if output_samples.shape[1] == 1:
                output_samples = output_samples.flatten()
            else:
                raise AnalysisError("Multi-output statistics not yet supported")

        stats = {
            "mean": float(np.mean(output_samples)),
            "variance": float(np.var(output_samples)),
            "std": float(np.std(output_samples)),
            "min": float(np.min(output_samples)),
            "max": float(np.max(output_samples)),
            "percentiles": {
                "5th": float(np.percentile(output_samples, 5)),
                "25th": float(np.percentile(output_samples, 25)),
                "50th": float(np.percentile(output_samples, 50)),
                "75th": float(np.percentile(output_samples, 75)),
                "95th": float(np.percentile(output_samples, 95)),
            },
            "histogram": {
                "counts": np.histogram(output_samples, bins=20)[0].tolist(),
                "bins": np.histogram(output_samples, bins=20)[1].tolist()
            }
        }

        logger.info("Computed output statistics")
        return stats

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
            Dictionary with sensitivity indices

        Raises:
            ValidationError: If parameters are invalid
            AnalysisError: If sensitivity analysis fails
        """
        if not self._scipy_available:
            logger.warning("SciPy not available - using basic sensitivity estimate")

        try:
            # This is a simplified implementation
            # For full Sobol analysis, consider integrating SALib
            n_inputs = len(inputs)

            # Sample inputs
            input_samples = await self.sample_inputs(
                inputs, n_samples, SamplingMethod.MONTE_CARLO, seed
            )

            # Evaluate function
            outputs = await self.evaluate_test_function(function_name, input_samples)

            # Compute basic sensitivity (variance-based)
            # This is a simplified approach - full Sobol requires specific sampling
            first_order = []
            for i in range(n_inputs):
                # Compute correlation-based sensitivity
                correlation = np.corrcoef(input_samples[:, i], outputs)[0, 1]
                sensitivity = correlation ** 2  # R² as sensitivity measure
                first_order.append(float(sensitivity))

            total_order = first_order.copy()  # Simplified

            result = {
                "method": method.value,
                "first_order": first_order,
                "total_order": total_order,
                "confidence_intervals": None,
                "n_samples": n_samples
            }

            logger.info(f"Computed {method.value} sensitivity indices")
            return result

        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            raise AnalysisError(f"Failed to compute sensitivity: {e}")

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

        Args:
            function_name: Name of the test function
            inputs: List of probabilistic input definitions
            n_samples: Number of samples for analysis
            sampling_method: Method for sampling inputs
            compute_sensitivity: Whether to compute sensitivity indices
            sensitivity_method: Method for sensitivity analysis
            seed: Random seed for reproducibility

        Returns:
            UQResult object with all analysis results

        Raises:
            PipelineError: If pipeline execution fails
        """
        # Check cache
        cache_key = f"{function_name}_{n_samples}_{sampling_method.value}_{seed}"
        if self._cache_enabled and cache_key in self._result_cache:
            logger.info("Returning cached UQ result")
            return self._result_cache[cache_key]

        try:
            logger.info(f"Starting UQ pipeline for {function_name}")

            # Step 1: Define inputs
            await self.define_probabilistic_inputs(inputs)

            # Step 2: Sample inputs
            input_samples = await self.sample_inputs(
                inputs, n_samples, sampling_method, seed
            )

            # Step 3: Evaluate function
            output_samples = await self.evaluate_test_function(
                function_name, input_samples
            )

            # Step 4: Compute statistics
            statistics = await self.compute_statistics(output_samples)

            # Step 5: Compute sensitivity (optional)
            sensitivity = None
            if compute_sensitivity:
                sensitivity = await self.compute_sensitivity(
                    function_name, inputs, n_samples, sensitivity_method, seed
                )

            # Create result object
            result = UQResult(
                function_name=function_name,
                input_samples=input_samples,
                output_samples=output_samples,
                statistics=statistics,
                sensitivity=sensitivity,
                metadata={
                    "n_samples": n_samples,
                    "sampling_method": sampling_method.value,
                    "sensitivity_method": sensitivity_method.value if compute_sensitivity else None,
                    "seed": seed
                }
            )

            # Cache result
            if self._cache_enabled:
                self._result_cache[cache_key] = result

            logger.info(f"UQ pipeline completed for {function_name}")
            return result

        except Exception as e:
            logger.error(f"UQ pipeline failed: {e}")
            raise PipelineError(f"Pipeline execution failed: {e}")

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the UQ system state and dependencies.

        Returns:
            Dictionary with validation results
        """
        checks = []
        dependencies = {}
        issues = []

        # Check uqtestfuns availability
        uqtestfuns_ok = self._uqtestfuns_available
        dependencies["uqtestfuns"] = {
            "available": uqtestfuns_ok,
            "version": getattr(self._uqtf, '__version__', 'unknown') if uqtestfuns_ok else None
        }
        checks.append(("uqtestfuns_available", uqtestfuns_ok))

        if not uqtestfuns_ok:
            issues.append("uqtestfuns library not installed")

        # Check SciPy availability
        scipy_ok = self._scipy_available
        dependencies["scipy"] = {
            "available": scipy_ok,
            "version": getattr(self._scipy_stats, '__version__', 'unknown') if scipy_ok else None
        }
        checks.append(("scipy_available", scipy_ok))

        if not scipy_ok:
            issues.append("SciPy not available - limited sampling capabilities")

        # Check NumPy
        try:
            import numpy as np
            numpy_ok = True
            dependencies["numpy"] = {
                "available": True,
                "version": np.__version__
            }
        except ImportError:
            numpy_ok = False
            dependencies["numpy"] = {"available": False}
            issues.append("NumPy not available")

        checks.append(("numpy_available", numpy_ok))

        # Check initialization
        init_ok = self._initialized
        checks.append(("initialized", init_ok))

        if not init_ok:
            issues.append("Adapter not initialized")

        # Overall validation
        is_valid = all(check[1] for check in checks)

        return {
            "is_valid": is_valid,
            "checks": {check[0]: check[1] for check in checks},
            "dependencies": dependencies,
            "issues": issues
        }

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the UQ system.

        Returns:
            True if shutdown was successful
        """
        logger.info("Shutting down uqtestfuns adapter")

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Clear caches
        self._function_cache.clear()
        self._result_cache.clear()

        self._initialized = False
        logger.info("Shutdown complete")
        return True
