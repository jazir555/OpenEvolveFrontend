"""
Uncertainty Propagation Module for End-to-End Invention Planner

Implements Monte Carlo simulation for error propagation analysis in invention planning.
This module provides tools to:
- Enumerate actual error sources from equipment, materials, measurements, environment
- Calculate real probabilities based on specifications and tolerances
- Propagate uncertainties through mathematical models
- Identify critical error sources via sensitivity analysis
- Perform Monte Carlo simulation for error propagation

Author: Agent 2 - Error Analysis and Adversarial Testing
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import json

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of error sources in invention planning"""
    EQUIPMENT_SPECIFICATION = "equipment_specification"
    MATERIAL_PROPERTIES = "material_properties"
    MEASUREMENT_UNCERTAINTY = "measurement_uncertainty"
    ENVIRONMENTAL_FACTORS = "environmental_factors"
    HUMAN_FACTORS = "human_factors"
    SYSTEMATIC_ERRORS = "systematic_errors"
    PROCESS_VARIATION = "process_variation"
    TIMING_ERRORS = "timing_errors"


class ProbabilityDistribution(Enum):
    """Types of probability distributions for error modeling"""
    NORMAL = "normal"  # Gaussian distribution
    UNIFORM = "uniform"  # Uniform distribution
    TRIANGULAR = "triangular"  # Triangular distribution
    EXPONENTIAL = "exponential"  # Exponential distribution
    LOGNORMAL = "lognormal"  # Log-normal distribution
    WEIBULL = "weibull"  # Weibull distribution


@dataclass
class ErrorSource:
    """
    Represents an actual error source with quantified uncertainty

    Attributes:
        name: Name of the error source
        category: Category of error (equipment, material, measurement, etc.)
        description: Detailed description of the error source
        distribution: Type of probability distribution
        distribution_params: Parameters for the distribution (e.g., mean, std for normal)
        nominal_value: Expected/nominal value
        tolerance: Tolerance range (± value)
        probability_of_occurrence: Actual probability of error occurring (0-1)
        impact_severity: Impact on overall success (critical/high/medium/low)
        mitigation_strategy: How to mitigate this error
        verification_method: How to verify this error source
        acceptance_criteria: Criteria for accepting this risk
    """
    name: str
    category: ErrorCategory
    description: str
    distribution: ProbabilityDistribution
    distribution_params: Dict[str, float]
    nominal_value: float
    tolerance: float
    probability_of_occurrence: float
    impact_severity: str  # "critical", "high", "medium", "low"
    mitigation_strategy: str
    verification_method: str
    acceptance_criteria: str
    sensitivity_score: float = 0.0  # Will be calculated via sensitivity analysis

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from this error source's distribution"""
        if self.distribution == ProbabilityDistribution.NORMAL:
            return np.random.normal(
                self.distribution_params.get('mean', self.nominal_value),
                self.distribution_params.get('std', self.tolerance / 3),  # 3-sigma
                n_samples
            )
        elif self.distribution == ProbabilityDistribution.UNIFORM:
            return np.random.uniform(
                self.nominal_value - self.tolerance,
                self.nominal_value + self.tolerance,
                n_samples
            )
        elif self.distribution == ProbabilityDistribution.TRIANGULAR:
            return np.random.triangular(
                self.nominal_value - self.tolerance,
                self.nominal_value,
                self.nominal_value + self.tolerance,
                n_samples
            )
        elif self.distribution == ProbabilityDistribution.LOGNORMAL:
            return np.random.lognormal(
                self.distribution_params.get('mean', 0),
                self.distribution_params.get('sigma', 0.1),
                n_samples
            )
        else:
            # Default to normal
            return np.random.normal(self.nominal_value, self.tolerance / 3, n_samples)


@dataclass
class UncertaintyPropagationResult:
    """
    Results from Monte Carlo uncertainty propagation

    Attributes:
        mean: Mean of the output distribution
        std: Standard deviation of the output distribution
        percentile_5: 5th percentile
        percentile_95: 95th percentile
        confidence_interval_95: 95% confidence interval
        probability_of_success: Probability that result meets criteria
        critical_error_sources: List of most critical error sources by sensitivity
        samples: All samples from the simulation (for further analysis)
    """
    mean: float
    std: float
    percentile_5: float
    percentile_95: float
    confidence_interval_95: Tuple[float, float]
    probability_of_success: float
    critical_error_sources: List[Tuple[str, float]]  # (error_name, sensitivity_score)
    samples: np.ndarray = field(default_factory=lambda: np.array([]))


class UncertaintyPropagator:
    """
    Performs Monte Carlo uncertainty propagation analysis for invention planning

    This class provides comprehensive error analysis by:
    1. Enumerating actual error sources from specifications
    2. Calculating real probabilities based on tolerances
    3. Propagating uncertainties through mathematical models
    4. Identifying critical error sources via sensitivity analysis
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the uncertainty propagator

        Args:
            random_seed: Optional seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.logger = logging.getLogger(__name__)

    def enumerate_equipment_errors(
        self,
        equipment_specs: List[Dict[str, Any]]
    ) -> List[ErrorSource]:
        """
        Enumerate actual error sources from equipment specifications

        Args:
            equipment_specs: List of equipment specifications with tolerances

        Returns:
            List of error sources from equipment specifications
        """
        error_sources = []

        for spec in equipment_specs:
            name = spec.get('name', 'Unknown Equipment')
            accuracy = spec.get('accuracy', None)
            precision = spec.get('precision', None)
            tolerance = spec.get('tolerance', None)
            failure_rate = spec.get('failure_rate', 0.0)

            # Accuracy error (systematic)
            if accuracy is not None:
                error_sources.append(ErrorSource(
                    name=f"{name}_accuracy_error",
                    category=ErrorCategory.EQUIPMENT_SPECIFICATION,
                    description=f"Accuracy specification error for {name}: ±{accuracy}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': 0.0, 'std': accuracy / 3},
                    nominal_value=0.0,
                    tolerance=accuracy,
                    probability_of_occurrence=1.0,  # Always present
                    impact_severity="high" if accuracy > 0.01 else "medium",
                    mitigation_strategy=f"Calibrate equipment regularly, use higher precision equipment",
                    verification_method="Calibration against reference standard",
                    acceptance_criteria=f"Accuracy within ±{accuracy}"
                ))

            # Precision error (repeatability)
            if precision is not None:
                error_sources.append(ErrorSource(
                    name=f"{name}_precision_error",
                    category=ErrorCategory.EQUIPMENT_SPECIFICATION,
                    description=f"Precision/repeatability error for {name}: ±{precision}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': 0.0, 'std': precision / 3},
                    nominal_value=0.0,
                    tolerance=precision,
                    probability_of_occurrence=1.0,
                    impact_severity="medium",
                    mitigation_strategy="Multiple measurements, statistical averaging",
                    verification_method="Repeated measurements of reference standard",
                    acceptance_criteria=f"Precision within ±{precision}"
                ))

            # Equipment failure
            if failure_rate > 0:
                error_sources.append(ErrorSource(
                    name=f"{name}_failure",
                    category=ErrorCategory.EQUIPMENT_SPECIFICATION,
                    description=f"Equipment failure for {name} (rate: {failure_rate})",
                    distribution=ProbabilityDistribution.EXPONENTIAL,
                    distribution_params={'scale': 1.0 / failure_rate},
                    nominal_value=0.0,
                    tolerance=1.0,
                    probability_of_occurrence=failure_rate,
                    impact_severity="critical",
                    mitigation_strategy="Redundant systems, preventive maintenance",
                    verification_method="Mean time between failures (MTBF) analysis",
                    acceptance_criteria=f"Failure rate < {failure_rate * 10}"
                ))

        return error_sources

    def enumerate_material_errors(
        self,
        material_specs: List[Dict[str, Any]]
    ) -> List[ErrorSource]:
        """
        Enumerate actual error sources from material properties

        Args:
            material_specs: List of material specifications with variations

        Returns:
            List of error sources from material properties
        """
        error_sources = []

        for spec in material_specs:
            name = spec.get('name', 'Unknown Material')
            property_variations = spec.get('property_variations', {})
            impurity_level = spec.get('impurity_level', 0.0)
            batch_variation = spec.get('batch_variation', 0.0)

            # Property variations
            for prop_name, variation in property_variations.items():
                error_sources.append(ErrorSource(
                    name=f"{name}_{prop_name}_variation",
                    category=ErrorCategory.MATERIAL_PROPERTIES,
                    description=f"{prop_name} variation in {name}: ±{variation}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': 0.0, 'std': variation / 3},
                    nominal_value=0.0,
                    tolerance=variation,
                    probability_of_occurrence=1.0,
                    impact_severity="high" if variation > 0.05 else "medium",
                    mitigation_strategy=f"Material testing, quality control, supplier qualification",
                    verification_method="Batch-to-batch testing, certificate of analysis",
                    acceptance_criteria=f"{prop_name} within specification"
                ))

            # Impurities
            if impurity_level > 0:
                error_sources.append(ErrorSource(
                    name=f"{name}_impurities",
                    category=ErrorCategory.MATERIAL_PROPERTIES,
                    description=f"Impurity level in {name}: {impurity_level}",
                    distribution=ProbabilityDistribution.LOGNORMAL,
                    distribution_params={'mean': impurity_level, 'sigma': impurity_level * 0.5},
                    nominal_value=0.0,
                    tolerance=impurity_level * 2,
                    probability_of_occurrence=0.5,
                    impact_severity="high" if impurity_level > 0.01 else "low",
                    mitigation_strategy="Purification, higher grade materials",
                    verification_method="Chemical analysis, spectroscopy",
                    acceptance_criteria=f"Impurity level < {impurity_level}"
                ))

            # Batch-to-batch variation
            if batch_variation > 0:
                error_sources.append(ErrorSource(
                    name=f"{name}_batch_variation",
                    category=ErrorCategory.MATERIAL_PROPERTIES,
                    description=f"Batch-to-batch variation in {name}: ±{batch_variation}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': 0.0, 'std': batch_variation / 3},
                    nominal_value=0.0,
                    tolerance=batch_variation,
                    probability_of_occurrence=1.0,
                    impact_severity="medium",
                    mitigation_strategy="Vendor qualification, incoming inspection",
                    verification_method="Batch testing, statistical process control",
                    acceptance_criteria=f"Variation within ±{batch_variation}"
                ))

        return error_sources

    def enumerate_measurement_errors(
        self,
        measurement_specs: List[Dict[str, Any]]
    ) -> List[ErrorSource]:
        """
        Enumerate actual error sources from measurements

        Args:
            measurement_specs: List of measurement specifications

        Returns:
            List of error sources from measurements
        """
        error_sources = []

        for spec in measurement_specs:
            name = spec.get('name', 'Unknown Measurement')
            resolution = spec.get('resolution', None)
            uncertainty = spec.get('uncertainty', None)
            bias = spec.get('bias', 0.0)

            # Resolution error (quantization)
            if resolution is not None:
                error_sources.append(ErrorSource(
                    name=f"{name}_resolution_error",
                    category=ErrorCategory.MEASUREMENT_UNCERTAINTY,
                    description=f"Resolution/quantization error for {name}: {resolution}",
                    distribution=ProbabilityDistribution.UNIFORM,
                    distribution_params={},
                    nominal_value=0.0,
                    tolerance=resolution / 2,
                    probability_of_occurrence=1.0,
                    impact_severity="low",
                    mitigation_strategy="Use higher resolution equipment",
                    verification_method="Calibration with known standards",
                    acceptance_criteria=f"Resolution error < {resolution}"
                ))

            # Measurement uncertainty
            if uncertainty is not None:
                error_sources.append(ErrorSource(
                    name=f"{name}_uncertainty",
                    category=ErrorCategory.MEASUREMENT_UNCERTAINTY,
                    description=f"Measurement uncertainty for {name}: ±{uncertainty}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': 0.0, 'std': uncertainty / 2},
                    nominal_value=0.0,
                    tolerance=uncertainty,
                    probability_of_occurrence=1.0,
                    impact_severity="high" if uncertainty > 0.05 else "medium",
                    mitigation_strategy="Multiple measurements, error averaging",
                    verification_method="Gage R&R studies",
                    acceptance_criteria=f"Uncertainty within ±{uncertainty}"
                ))

            # Bias error (systematic)
            if abs(bias) > 0:
                error_sources.append(ErrorSource(
                    name=f"{name}_bias_error",
                    category=ErrorCategory.MEASUREMENT_UNCERTAINTY,
                    description=f"Bias error for {name}: {bias}",
                    distribution=ProbabilityDistribution.NORMAL,
                    distribution_params={'mean': bias, 'std': abs(bias) / 3},
                    nominal_value=0.0,
                    tolerance=abs(bias) * 2,
                    probability_of_occurrence=1.0,
                    impact_severity="high" if abs(bias) > 0.01 else "medium",
                    mitigation_strategy="Calibration, correction factors",
                    verification_method="Comparison with reference standard",
                    acceptance_criteria=f"Bias < {abs(bias) * 0.5}"
                ))

        return error_sources

    def monte_carlo_propagation(
        self,
        error_sources: List[ErrorSource],
        model_function: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 10000,
        success_criteria: Optional[Callable[[float], bool]] = None,
        target_value: Optional[float] = None,
        tolerance: Optional[float] = None
    ) -> UncertaintyPropagationResult:
        """
        Perform Monte Carlo uncertainty propagation

        Args:
            error_sources: List of error sources
            model_function: Function that maps error sources to output
                           Takes array of shape (n_samples, n_errors) and returns array of shape (n_samples,)
            n_samples: Number of Monte Carlo samples
            success_criteria: Optional function to determine if result is successful
            target_value: Optional target value for success criteria
            tolerance: Optional tolerance for success criteria

        Returns:
            UncertaintyPropagationResult with analysis
        """
        self.logger.info(f"Running Monte Carlo propagation with {n_samples} samples")

        # Sample from each error source
        n_errors = len(error_sources)
        samples = np.zeros((n_samples, n_errors))

        for i, error_source in enumerate(error_sources):
            samples[:, i] = error_source.sample(n_samples)

        # Apply model function
        # Each row of samples represents one Monte Carlo trial
        outputs = np.array([model_function(sample_row) for sample_row in samples])

        # Calculate statistics
        mean = np.mean(outputs)
        std = np.std(outputs)
        percentile_5 = np.percentile(outputs, 5)
        percentile_95 = np.percentile(outputs, 95)
        ci_95 = (percentile_5, percentile_95)

        # Calculate probability of success
        if success_criteria:
            probability_of_success = np.mean([success_criteria(output) for output in outputs])
        elif target_value is not None and tolerance is not None:
            probability_of_success = np.mean(np.abs(outputs - target_value) <= tolerance)
        else:
            probability_of_success = 0.5  # Unknown

        # Sensitivity analysis to find critical error sources
        critical_error_sources = self._sensitivity_analysis(
            error_sources, samples, outputs, model_function
        )

        self.logger.info(f"Monte Carlo complete: mean={mean:.4f}, std={std:.4f}, "
                        f"success_prob={probability_of_success:.2%}")

        return UncertaintyPropagationResult(
            mean=mean,
            std=std,
            percentile_5=percentile_5,
            percentile_95=percentile_95,
            confidence_interval_95=ci_95,
            probability_of_success=probability_of_success,
            critical_error_sources=critical_error_sources,
            samples=outputs
        )

    def _sensitivity_analysis(
        self,
        error_sources: List[ErrorSource],
        samples: np.ndarray,
        outputs: np.ndarray,
        model_function: Callable[[np.ndarray], float]
    ) -> List[Tuple[str, float]]:
        """
        Perform sensitivity analysis to identify critical error sources

        Uses correlation-based sensitivity analysis

        Args:
            error_sources: List of error sources
            samples: Sampled error values (n_samples, n_errors)
            outputs: Model outputs (n_samples,)
            model_function: Model function

        Returns:
            List of (error_name, sensitivity_score) sorted by sensitivity
        """
        sensitivities = []

        for i, error_source in enumerate(error_sources):
            # Calculate correlation between this error source and output
            correlation = np.corrcoef(samples[:, i], outputs)[0, 1]
            sensitivity_score = abs(correlation) if not np.isnan(correlation) else 0.0

            # Store sensitivity score in error source
            error_source.sensitivity_score = sensitivity_score

            sensitivities.append((error_source.name, sensitivity_score))

        # Sort by sensitivity score (descending)
        sensitivities.sort(key=lambda x: x[1], reverse=True)

        return sensitivities

    def calculate_failure_probability(
        self,
        error_sources: List[ErrorSource],
        model_function: Callable[[np.ndarray], np.ndarray],
        failure_threshold: float,
        n_samples: int = 10000
    ) -> float:
        """
        Calculate probability of failure

        Args:
            error_sources: List of error sources
            model_function: Model function
            failure_threshold: Value above/below which is failure
            n_samples: Number of Monte Carlo samples

        Returns:
            Probability of failure (0-1)
        """
        result = self.monte_carlo_propagation(
            error_sources, model_function, n_samples
        )

        # Count samples outside acceptable range
        failures = np.sum(np.abs(result.samples) > failure_threshold)
        return failures / len(result.samples)

    def to_dict(self, result: UncertaintyPropagationResult) -> Dict[str, Any]:
        """Convert propagation result to dictionary"""
        return {
            'mean': result.mean,
            'std': result.std,
            'percentile_5': result.percentile_5,
            'percentile_95': result.percentile_95,
            'confidence_interval_95': result.confidence_interval_95,
            'probability_of_success': result.probability_of_success,
            'critical_error_sources': [
                {'name': name, 'sensitivity': score}
                for name, score in result.critical_error_sources
            ]
        }


def enumerate_all_errors(
    equipment_specs: Optional[List[Dict[str, Any]]] = None,
    material_specs: Optional[List[Dict[str, Any]]] = None,
    measurement_specs: Optional[List[Dict[str, Any]]] = None
) -> List[ErrorSource]:
    """
    Convenience function to enumerate all error sources

    Args:
        equipment_specs: Equipment specifications
        material_specs: Material specifications
        measurement_specs: Measurement specifications

    Returns:
        Combined list of all error sources
    """
    propagator = UncertaintyPropagator()
    all_errors = []

    if equipment_specs:
        all_errors.extend(propagator.enumerate_equipment_errors(equipment_specs))

    if material_specs:
        all_errors.extend(propagator.enumerate_material_errors(material_specs))

    if measurement_specs:
        all_errors.extend(propagator.enumerate_measurement_errors(measurement_specs))

    return all_errors


def propagate_uncertainties(
    error_sources: List[ErrorSource],
    model_function: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 10000
) -> UncertaintyPropagationResult:
    """
    Convenience function for uncertainty propagation

    Args:
        error_sources: List of error sources
        model_function: Model function
        n_samples: Number of Monte Carlo samples

    Returns:
        UncertaintyPropagationResult
    """
    propagator = UncertaintyPropagator()
    return propagator.monte_carlo_propagation(
        error_sources, model_function, n_samples
    )
