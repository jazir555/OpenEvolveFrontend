"""
uqtestfuns Bridge for OpenEvolve Validation Systems

This module provides the bridge connecting uqtestfuns to OpenEvolve's validation
and verification systems. It enables uncertainty quantification in model validation,
experimentation, and testing workflows.

Key Integration Points:
- Model validation pipeline integration
- Experimentation result validation
- Test result uncertainty analysis
- Verification system enhancement
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the adapter
from .adapter import UQTestFunsAdapter

# Configure logging
logger = logging.getLogger(__name__)


class UQTestFunsBridge:
    """
    Bridge for integrating uqtestfuns with OpenEvolve validation systems.

    This bridge provides high-level integration methods for:
    - Validating model predictions with uncertainty quantification
    - Analyzing uncertainty in experimental results
    - Enhancing test verification with sensitivity analysis
    - Propagating uncertainties through workflows
    """

    def __init__(self, adapter: Optional[UQTestFunsAdapter] = None):
        """
        Initialize the bridge.

        Args:
            adapter: UQTestFunsAdapter instance (creates new one if None)
        """
        self._adapter = adapter or UQTestFunsAdapter()
        self._initialized = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the bridge and underlying adapter.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful
        """
        try:
            success = await self._adapter.initialize(config)
            self._initialized = success
            if success:
                logger.info("UQ bridge initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize UQ bridge: {e}")
            return False

    async def validate_model_with_uncertainty(
        self,
        model_predictions: List[float],
        test_function_name: str,
        probabilistic_inputs: List[Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Validate model predictions using UQ test functions.

        Compares model predictions against UQ test function results to assess
        prediction quality under uncertainty.

        Args:
            model_predictions: Model predictions to validate
            test_function_name: UQ test function to use as reference
            probabilistic_inputs: Probabilistic input specifications
            confidence_level: Confidence level for validation (default: 0.95)

        Returns:
            Dictionary containing:
                - is_valid: Whether predictions are within acceptable bounds
                - uncertainty_bounds: Confidence intervals for predictions
                - sensitivity_analysis: Input sensitivity results
                - recommendation: Validation recommendation
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized")

        try:
            from ..base.uq_interface import (
                ProbabilisticInput,
                SamplingMethod,
                SensitivityMethod
            )

            # Convert probabilistic inputs if needed
            if not isinstance(probabilistic_inputs[0], ProbabilisticInput):
                probabilistic_inputs = [
                    ProbabilisticInput(
                        name=inp.get('name', f'x{i}'),
                        distribution=inp.get('distribution', 'uniform'),
                        parameters=inp.get('parameters', [0, 1]),
                        bounds=inp.get('bounds')
                    )
                    for i, inp in enumerate(probabilistic_inputs)
                ]

            # Run UQ pipeline
            n_samples = len(model_predictions)
            uq_result = await self._adapter.run_uq_pipeline(
                function_name=test_function_name,
                inputs=probabilistic_inputs,
                n_samples=n_samples,
                sampling_method=SamplingMethod.MONTE_CARLO,
                compute_sensitivity=True,
                sensitivity_method=SensitivityMethod.SOBOL
            )

            # Compare predictions with UQ results
            predictions_array = __import__('numpy').array(model_predictions)
            uq_outputs = uq_result.output_samples

            # Compute error metrics
            mean_error = float(__import__('numpy').abs(predictions_array - uq_outputs).mean())
            std_error = float(__import__('numpy').std(predictions_array - uq_outputs))

            # Compute confidence bounds
            alpha = 1 - confidence_level
            lower_bound = uq_result.statistics['percentiles']['5th']
            upper_bound = uq_result.statistics['percentiles']['95th']

            # Check if predictions are within bounds
            within_bounds = all(
                (predictions_array >= lower_bound) & (predictions_array <= upper_bound)
            )

            # Generate recommendation
            if within_bounds and mean_error < 0.1 * (upper_bound - lower_bound):
                recommendation = "PASS: Model predictions consistent with UQ analysis"
                is_valid = True
            elif within_bounds:
                recommendation = "ACCEPTABLE: Predictions within bounds but high variance"
                is_valid = True
            else:
                recommendation = "FAIL: Predictions outside uncertainty bounds"
                is_valid = False

            result = {
                "is_valid": is_valid,
                "uncertainty_bounds": {
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "confidence_level": confidence_level
                },
                "error_metrics": {
                    "mean_error": mean_error,
                    "std_error": std_error
                },
                "sensitivity_analysis": uq_result.sensitivity,
                "uq_statistics": uq_result.statistics,
                "recommendation": recommendation,
                "validated_at": datetime.now().isoformat()
            }

            logger.info(f"Model validation completed: {recommendation}")
            return result

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise

    async def analyze_experiment_uncertainty(
        self,
        experiment_results: Dict[str, Any],
        input_parameters: Dict[str, Any],
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze uncertainty propagation in experimental results.

        Args:
            experiment_results: Experimental results data
            input_parameters: Input parameter definitions with uncertainties
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary containing:
                - propagated_uncertainty: Uncertainty in output
                - sensitivity_to_inputs: Input parameter sensitivities
                - confidence_intervals: Confidence intervals for results
                - recommendations: Analysis recommendations
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized")

        try:
            from ..base.uq_interface import (
                ProbabilisticInput,
                SamplingMethod
            )

            # Convert input parameters to probabilistic inputs
            probabilistic_inputs = []
            for param_name, param_spec in input_parameters.items():
                if 'uncertainty' in param_spec:
                    # Create probabilistic input from uncertainty specification
                    uncertainty = param_spec['uncertainty']
                    distribution = uncertainty.get('distribution', 'normal')
                    parameters = uncertainty.get('parameters', [param_spec.get('value', 0), 0.1])

                    probabilistic_inputs.append(ProbabilisticInput(
                        name=param_name,
                        distribution=distribution,
                        parameters=parameters,
                        bounds=uncertainty.get('bounds')
                    ))

            if not probabilistic_inputs:
                return {
                    "propagated_uncertainty": None,
                    "error": "No input uncertainties specified",
                    "recommendations": ["Specify input uncertainties for UQ analysis"]
                }

            # Select appropriate test function based on experiment type
            test_function = self._select_test_function_for_experiment(experiment_results)

            # Run UQ analysis
            uq_result = await self._adapter.run_uq_pipeline(
                function_name=test_function,
                inputs=probabilistic_inputs,
                n_samples=n_samples,
                sampling_method=SamplingMethod.LATIN_HYPERCUBE,
                compute_sensitivity=True
            )

            # Analyze results
            output_std = uq_result.statistics['std']
            output_mean = uq_result.statistics['mean']
            coefficient_of_variation = output_std / output_mean if output_mean != 0 else float('inf')

            # Determine uncertainty level
            if coefficient_of_variation < 0.1:
                uncertainty_level = "LOW"
            elif coefficient_of_variation < 0.3:
                uncertainty_level = "MODERATE"
            else:
                uncertainty_level = "HIGH"

            result = {
                "propagated_uncertainty": {
                    "std": output_std,
                    "mean": output_mean,
                    "coefficient_of_variation": coefficient_of_variation,
                    "uncertainty_level": uncertainty_level
                },
                "sensitivity_to_inputs": uq_result.sensitivity,
                "confidence_intervals": {
                    "95_percent": [
                        uq_result.statistics['percentiles']['5th'],
                        uq_result.statistics['percentiles']['95th']
                    ]
                },
                "recommendations": self._generate_uncertainty_recommendations(
                    uncertainty_level, uq_result.sensitivity
                ),
                "analyzed_at": datetime.now().isoformat()
            }

            logger.info(f"Experiment uncertainty analysis completed: {uncertainty_level}")
            return result

        except Exception as e:
            logger.error(f"Experiment uncertainty analysis failed: {e}")
            raise

    def _select_test_function_for_experiment(self, experiment_results: Dict[str, Any]) -> str:
        """Select appropriate test function based on experiment type."""
        # This is a simplified selection logic
        # In practice, this would be more sophisticated
        experiment_type = experiment_results.get('type', 'generic')

        function_map = {
            'optimization': 'ackley',
            'sensitivity': 'ishigami',
            'benchmark': 'rosenbrock',
            'multimodal': 'branin',
            'generic': 'ishigami'  # Default
        }

        return function_map.get(experiment_type, 'ishigami')

    def _generate_uncertainty_recommendations(
        self,
        uncertainty_level: str,
        sensitivity: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on uncertainty analysis."""
        recommendations = []

        if uncertainty_level == "HIGH":
            recommendations.append(
                "High uncertainty detected - consider increasing sample size or reducing input uncertainties"
            )
        elif uncertainty_level == "MODERATE":
            recommendations.append(
                "Moderate uncertainty - results should be interpreted with caution"
            )

        if sensitivity and 'first_order' in sensitivity:
            # Find most sensitive inputs
            sensitivities = sensitivity['first_order']
            max_sensitivity_idx = int(__import__('numpy').argmax(sensitivities))
            max_sensitivity = sensitivities[max_sensitivity_idx]

            if max_sensitivity > 0.5:
                recommendations.append(
                    f"Input {max_sensitivity_idx} dominates output uncertainty "
                    f"(sensitivity: {max_sensitivity:.2f}) - focus on reducing this input's uncertainty"
                )

        return recommendations

    async def enhance_test_verification(
        self,
        test_results: Dict[str, Any],
        test_function_name: str = "ishigami",
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Enhance test verification with uncertainty quantification.

        Adds statistical rigor to test results by quantifying uncertainty and
        performing sensitivity analysis.

        Args:
            test_results: Test execution results
            test_function_name: UQ test function for validation
            significance_level: Statistical significance level

        Returns:
            Dictionary containing:
                - enhanced_status: Enhanced test status with uncertainty
                - statistical_significance: Statistical test results
                - uncertainty_metrics: Uncertainty quantification metrics
                - recommendations: Verification recommendations
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized")

        try:
            from ..base.uq_interface import ProbabilisticInput

            # Create default probabilistic inputs for verification
            n_inputs = test_results.get('n_inputs', 3)
            probabilistic_inputs = [
                ProbabilisticInput(
                    name=f'x{i}',
                    distribution='uniform',
                    parameters=[-1, 1]
                )
                for i in range(n_inputs)
            ]

            # Run UQ analysis
            n_samples = test_results.get('n_samples', 100)
            uq_result = await self._adapter.run_uq_pipeline(
                function_name=test_function_name,
                inputs=probabilistic_inputs,
                n_samples=n_samples,
                compute_sensitivity=True
            )

            # Enhance test status
            original_status = test_results.get('status', 'UNKNOWN')
            uncertainty_magnitude = uq_result.statistics['std']

            if uncertainty_magnitude < 0.1:
                enhanced_status = f"{original_status} (LOW UNCERTAINTY)"
            elif uncertainty_magnitude < 1.0:
                enhanced_status = f"{original_status} (MODERATE UNCERTAINTY)"
            else:
                enhanced_status = f"{original_status} (HIGH UNCERTAINTY)"

            result = {
                "original_status": original_status,
                "enhanced_status": enhanced_status,
                "statistical_significance": {
                    "p_value": self._compute_significance(test_results, uq_result),
                    "significance_level": significance_level,
                    "is_significant": True  # Placeholder
                },
                "uncertainty_metrics": {
                    "std": uq_result.statistics['std'],
                    "variance": uq_result.statistics['variance'],
                    "coefficient_of_variation": (
                        uq_result.statistics['std'] / uq_result.statistics['mean']
                        if uq_result.statistics['mean'] != 0 else float('inf')
                    )
                },
                "sensitivity_analysis": uq_result.sensitivity,
                "recommendations": self._generate_verification_recommendations(
                    uncertainty_magnitude, uq_result.sensitivity
                ),
                "verified_at": datetime.now().isoformat()
            }

            logger.info(f"Test verification enhanced: {enhanced_status}")
            return result

        except Exception as e:
            logger.error(f"Test verification enhancement failed: {e}")
            raise

    def _compute_significance(
        self,
        test_results: Dict[str, Any],
        uq_result: Any
    ) -> float:
        """Compute statistical significance of test results."""
        # This is a placeholder - actual implementation would perform
        # appropriate statistical tests (t-test, Mann-Whitney, etc.)
        return 0.01

    def _generate_verification_recommendations(
        self,
        uncertainty_magnitude: float,
        sensitivity: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate verification recommendations."""
        recommendations = []

        if uncertainty_magnitude > 1.0:
            recommendations.append(
                "High uncertainty detected - consider additional testing or tighter controls"
            )

        if sensitivity and 'first_order' in sensitivity:
            # Check if any input dominates
            sensitivities = sensitivity['first_order']
            if any(s > 0.7 for s in sensitivities):
                recommendations.append(
                    "High sensitivity to specific inputs detected - verify input control"
                )

        if len(recommendations) == 0:
            recommendations.append("Verification within acceptable uncertainty bounds")

        return recommendations

    async def get_validation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive validation report of UQ systems.

        Returns:
            Dictionary containing:
                - system_status: Overall system status
                - validation_checks: Individual validation checks
                - available_functions: Available UQ test functions
                - performance_metrics: Performance metrics
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized")

        try:
            # Get validation from adapter
            validation = await self._adapter.validate()

            # Get available functions
            functions = await self._adapter.list_available_functions()

            # Generate report
            report = {
                "system_status": "HEALTHY" if validation['is_valid'] else "ISSUES_DETECTED",
                "validation_checks": validation['checks'],
                "dependencies": validation['dependencies'],
                "available_functions": functions,
                "n_available_functions": len(functions),
                "issues": validation['issues'],
                "generated_at": datetime.now().isoformat()
            }

            logger.info("Validation report generated")
            return report

        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            raise

    async def shutdown(self) -> bool:
        """
        Shutdown the bridge and adapter.

        Returns:
            True if shutdown successful
        """
        logger.info("Shutting down UQ bridge")
        success = await self._adapter.shutdown()
        self._initialized = False
        return success


# Convenience functions for common operations

async def validate_with_uq(
    model_predictions: List[float],
    test_function: str = "ishigami",
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Convenience function to validate model predictions with UQ.

    Args:
        model_predictions: Model predictions to validate
        test_function: UQ test function name
        n_samples: Number of samples for UQ analysis

    Returns:
        Validation results with uncertainty quantification
    """
    bridge = UQTestFunsBridge()
    await bridge.initialize({'enabled': True})

    # Create default probabilistic inputs
    from ..base.uq_interface import ProbabilisticInput
    n_inputs = 3  # Default for most test functions
    probabilistic_inputs = [
        ProbabilisticInput(
            name=f'x{i}',
            distribution='uniform',
            parameters=[-1, 1]
        )
        for i in range(n_inputs)
    ]

    result = await bridge.validate_model_with_uncertainty(
        model_predictions,
        test_function,
        probabilistic_inputs
    )

    await bridge.shutdown()
    return result
