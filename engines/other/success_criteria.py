"""
Binary Success Criteria System for End-to-End Invention Planning

Implements truly binary success/fail criteria for invention validation.
Each criterion has:
- Exact measurement method
- Exact threshold
- Exact verification procedure
- Exact acceptance criteria
- Fallback criteria
- Error bounds

Author: Agent 5 - Success Criteria and Validation
Version: 1.0.0
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols and Types
# ============================================================================

class CriterionType(Enum):
    """Types of success criteria"""
    QUANTITATIVE = "quantitative"  # Numerical measurement
    QUALITATIVE = "qualitative"    # Binary yes/no observation
    STATISTICAL = "statistical"    # Statistical test
    PHYSICAL = "physical"          # Physical law verification
    LOGICAL = "logical"            # Logical consistency
    FUNCTIONAL = "functional"      # Functional requirement


class MeasurementMethod(Enum):
    """Measurement method types"""
    DIRECT = "direct"              # Direct measurement
    INDIRECT = "indirect"          # Calculated from other measurements
    OBSERVATION = "observation"    # Visual/observable check
    EXPERIMENTAL = "experimental"  # Experimental test
    COMPUTATIONAL = "computational"  # Computation/simulation
    ANALYTICAL = "analytical"      # Analytical calculation


@dataclass
class ErrorBounds:
    """Error bounds for a measurement"""
    lower_bound: float
    upper_bound: float
    confidence_interval: float  # e.g., 0.95 for 95% confidence
    measurement_uncertainty: float
    systematic_error: float = 0.0


@dataclass
class VerificationProcedure:
    """Exact verification procedure"""
    procedure_id: str
    steps: List[str]
    equipment_required: List[str]
    expertise_level: str
    estimated_duration: str
    safety_considerations: List[str] = field(default_factory=list)
    independent_verification: bool = True


@dataclass
class FallbackCriterion:
    """Fallback criterion when primary measurement fails"""
    criterion: str
    trigger_condition: str
    alternative_method: str
    reliability_factor: float  # 0-1, relative to primary


class BinarySuccessCriterion(Protocol):
    """Protocol for binary success criteria

    All criteria must implement these methods to be truly binary.
    """

    def measure(self, experiment_result: Dict[str, Any]) -> float:
        """Get actual measurement from experiment result

        Args:
            experiment_result: Raw experiment/observation data

        Returns:
            Measured value

        Raises:
            MeasurementError: If measurement cannot be obtained
        """
        ...

    def passes(self, experiment_result: Dict[str, Any]) -> bool:
        """Binary pass/fail determination

        Args:
            experiment_result: Raw experiment/observation data

        Returns:
            True if passes, False if fails (no ambiguity)
        """
        ...

    def verify(self, experiment_result: Dict[str, Any]) -> bool:
        """Independent verification of result

        Args:
            experiment_result: Raw experiment/observation data

        Returns:
            True if verification passes, False otherwise
        """
        ...

    def get_error_bounds(self) -> ErrorBounds:
        """Get error bounds for this criterion

        Returns:
            Error bounds information
        """
        ...

    def get_verification_procedure(self) -> VerificationProcedure:
        """Get detailed verification procedure

        Returns:
            Verification procedure
        """
        ...


# ============================================================================
# Implementations
# ============================================================================

@dataclass
class QuantitativeSuccessCriterion:
    """Quantitative success criterion with numerical threshold"""

    name: str
    description: str
    criterion_type: CriterionType
    measurement_method: MeasurementMethod
    measurement_procedure: str  # How to measure
    threshold: float  # Exact threshold
    threshold_type: str  # "min", "max", "exact", "range"
    units: str
    error_tolerance: float  # Acceptable error margin
    verification_procedure: VerificationProcedure
    fallback_criteria: List[FallbackCriterion] = field(default_factory=list)
    error_bounds: Optional[ErrorBounds] = None

    def measure(self, experiment_result: Dict[str, Any]) -> float:
        """Extract measurement from experiment result"""
        # Look for measurement in result
        if 'value' in experiment_result:
            return float(experiment_result['value'])
        elif 'measurement' in experiment_result:
            return float(experiment_result['measurement'])
        elif self.name.lower().replace(' ', '_') in experiment_result:
            return float(experiment_result[self.name.lower().replace(' ', '_')])
        else:
            # Try to extract from description
            match = re.search(r'(\d+\.?\d*)', str(experiment_result))
            if match:
                return float(match.group(1))
            raise ValueError(f"Cannot extract measurement for {self.name} from result")

    def passes(self, experiment_result: Dict[str, Any]) -> bool:
        """Binary pass/fail"""
        try:
            value = self.measure(experiment_result)
        except (ValueError, KeyError) as e:
            logger.error(f"Measurement failed for {self.name}: {e}")
            return False

        if self.threshold_type == "min":
            return value >= self.threshold - self.error_tolerance
        elif self.threshold_type == "max":
            return value <= self.threshold + self.error_tolerance
        elif self.threshold_type == "exact":
            return abs(value - self.threshold) <= self.error_tolerance
        elif self.threshold_type == "range":
            min_val, max_val = self.threshold
            return min_val - self.error_tolerance <= value <= max_val + self.error_tolerance
        else:
            logger.error(f"Unknown threshold type: {self.threshold_type}")
            return False

    def verify(self, experiment_result: Dict[str, Any]) -> bool:
        """Independent verification"""
        # Check if verification was performed
        if 'verification' in experiment_result:
            return bool(experiment_result['verification'])
        elif 'verified' in experiment_result:
            return bool(experiment_result['verified'])
        elif self.verification_procedure.independent_verification:
            # If independent verification required but not provided, fail
            logger.warning(f"Independent verification required for {self.name} but not provided")
            return False
        return True

    def get_error_bounds(self) -> ErrorBounds:
        """Get error bounds"""
        if self.error_bounds:
            return self.error_bounds
        # Calculate default error bounds
        return ErrorBounds(
            lower_bound=self.threshold - self.error_tolerance * 3,
            upper_bound=self.threshold + self.error_tolerance * 3,
            confidence_interval=0.95,
            measurement_uncertainty=self.error_tolerance
        )

    def get_verification_procedure(self) -> VerificationProcedure:
        """Get verification procedure"""
        return self.verification_procedure


@dataclass
class QualitativeSuccessCriterion:
    """Qualitative success criterion (binary yes/no observation)"""

    name: str
    description: str
    criterion_type: CriterionType
    expected_observation: str
    verification_procedure: str
    negative_indicator: str  # What indicates failure
    ambiguous_indicators: List[str] = field(default_factory=list)
    fallback_criteria: List[FallbackCriterion] = field(default_factory=list)

    def measure(self, experiment_result: Dict[str, Any]) -> float:
        """Convert observation to binary value"""
        observation = str(experiment_result.get('observation', ''))
        if self.expected_observation.lower() in observation.lower():
            return 1.0
        elif any(ind.lower() in observation.lower() for ind in self.negative_indicator.split('|')):
            return 0.0
        else:
            # Check for ambiguous indicators
            for amb in self.ambiguous_indicators:
                if amb.lower() in observation.lower():
                    raise ValueError(f"Ambiguous observation: {amb}")
            # Default to fail if ambiguous
            return 0.0

    def passes(self, experiment_result: Dict[str, Any]) -> bool:
        """Binary pass/fail"""
        try:
            value = self.measure(experiment_result)
            return value >= 0.5
        except ValueError as e:
            logger.error(f"Ambiguous observation for {self.name}: {e}")
            return False

    def verify(self, experiment_result: Dict[str, Any]) -> bool:
        """Verify observation is clear and unambiguous"""
        observation = str(experiment_result.get('observation', ''))
        # Check for ambiguous indicators
        for amb in self.ambiguous_indicators:
            if amb.lower() in observation.lower():
                return False
        # Check if verification was performed
        if 'verified' in experiment_result:
            return bool(experiment_result['verified'])
        return True

    def get_error_bounds(self) -> ErrorBounds:
        """Qualitative criteria have no error bounds"""
        return ErrorBounds(
            lower_bound=0.0,
            upper_bound=1.0,
            confidence_interval=1.0,  # Binary is certain
            measurement_uncertainty=0.0
        )

    def get_verification_procedure(self) -> VerificationProcedure:
        """Get verification procedure"""
        return VerificationProcedure(
            procedure_id=f"{self.name}_verification",
            steps=[self.verification_procedure],
            equipment_required=["visual_inspection"],
            expertise_level="trained_observer",
            estimated_duration="immediate",
            independent_verification=True
        )


@dataclass
class StatisticalSuccessCriterion:
    """Statistical success criterion (e.g., p-value, confidence)"""

    name: str
    description: str
    criterion_type: CriterionType
    test_type: str  # "t_test", "chi_square", "anova", etc.
    threshold: float  # e.g., p < 0.05
    comparison: str  # "less_than", "greater_than"
    sample_size: int
    confidence_level: float
    verification_procedure: VerificationProcedure
    fallback_criteria: List[FallbackCriterion] = field(default_factory=list)

    def measure(self, experiment_result: Dict[str, Any]) -> float:
        """Extract test statistic from result"""
        if 'p_value' in experiment_result:
            return float(experiment_result['p_value'])
        elif 'statistic' in experiment_result:
            return float(experiment_result['statistic'])
        elif 'test_result' in experiment_result:
            return float(experiment_result['test_result'])
        else:
            raise ValueError(f"Cannot extract test result for {self.name}")

    def passes(self, experiment_result: Dict[str, Any]) -> bool:
        """Binary pass/fail based on statistical test"""
        try:
            value = self.measure(experiment_result)
            if self.comparison == "less_than":
                return value < self.threshold
            elif self.comparison == "greater_than":
                return value > self.threshold
            else:
                logger.error(f"Unknown comparison: {self.comparison}")
                return False
        except (ValueError, KeyError) as e:
            logger.error(f"Statistical test failed for {self.name}: {e}")
            return False

    def verify(self, experiment_result: Dict[str, Any]) -> bool:
        """Verify statistical test validity"""
        # Check sample size
        if 'sample_size' in experiment_result:
            if int(experiment_result['sample_size']) < self.sample_size:
                logger.warning(f"Sample size below minimum for {self.name}")
                return False
        # Check confidence level
        if 'confidence_level' in experiment_result:
            if float(experiment_result['confidence_level']) < self.confidence_level:
                logger.warning(f"Confidence level below minimum for {self.name}")
                return False
        return True

    def get_error_bounds(self) -> ErrorBounds:
        """Statistical error bounds"""
        return ErrorBounds(
            lower_bound=0.0,
            upper_bound=1.0,
            confidence_interval=self.confidence_level,
            measurement_uncertainty=1.0 / self.sample_size
        )

    def get_verification_procedure(self) -> VerificationProcedure:
        """Get verification procedure"""
        return self.verification_procedure


# ============================================================================
# Criterion Derivation Functions
# ============================================================================

def derive_criteria_from_goal(
    goal: Dict[str, Any],
    domain: str
) -> List[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion, StatisticalSuccessCriterion]]:
    """Derive binary success criteria from invention goal

    Args:
        goal: Invention goal with requirements
        domain: Scientific domain

    Returns:
        List of binary success criteria
    """
    criteria = []

    # Extract requirements from goal
    requirements = goal.get('key_requirements', [])
    success_definition = goal.get('success_definition', '')
    constraints = goal.get('constraints', [])

    # Parse success definition for quantitative criteria
    quantitative_patterns = [
        r'(?:efficiency|yield|purity|conversion) (>?:\s*<?\s*\d+\.?\d*)\s*%?',
        r'(?:temperature|pressure|pH|voltage|current|speed|power) (?:of\s*)?[:=]?\s*\d+\.?\d*\s*[A-Za-z]*',
        r'(?:greater than|less than|at least|no more than)\s+\d+\.?\d*',
    ]

    for pattern in quantitative_patterns:
        matches = re.finditer(pattern, success_definition, re.IGNORECASE)
        for match in matches:
            # Extract criterion from match
            criterion_text = match.group(0)
            criterion = _parse_quantitative_criterion(criterion_text, domain)
            if criterion:
                criteria.append(criterion)

    # Parse for qualitative criteria
    qualitative_indicators = [
        'visible', 'detectable', 'measurable', 'observable', 'functional', 'stable'
    ]

    for indicator in qualitative_indicators:
        if indicator in success_definition.lower():
            criterion = QualitativeSuccessCriterion(
                name=f"{indicator.replace('ble', 'bility').capitalize()} Check",
                description=f"Verify that the invention is {indicator}",
                criterion_type=CriterionType.QUALITATIVE,
                expected_observation=f"{indicator} outcome observed",
                verification_procedure=f"Visually inspect and confirm {indicator} behavior",
                negative_indicator=f"not {indicator}|failed|degraded",
                ambiguous_indicators=["partially", "somewhat", "unclear"]
            )
            criteria.append(criterion)

    # Parse constraints as criteria
    for constraint in constraints:
        if any(x in constraint.lower() for x in ['must', 'shall', 'required to']):
            criterion = _parse_constraint_criterion(constraint, domain)
            if criterion:
                criteria.append(criterion)

    # Add domain-specific criteria
    domain_criteria = _get_domain_specific_criteria(domain, goal)
    criteria.extend(domain_criteria)

    return criteria


def derive_criteria_from_math(
    math_models: List[Dict[str, Any]],
    error_analysis: List[Dict[str, Any]]
) -> List[Union[QuantitativeSuccessCriterion, StatisticalSuccessCriterion]]:
    """Derive criteria from mathematical models and error analysis

    Args:
        math_models: Formalized mathematical relationships
        error_analysis: Error source analysis

    Returns:
        List of success criteria derived from math
    """
    criteria = []

    for math_model in math_models:
        # Extract variables and their ranges
        variables = math_model.get('variables', {})
        theorem = math_model.get('theorem', '')

        # For each variable, create a criterion
        for var_name, var_def in variables.items():
            # Check if variable has a constraint
            if 'range' in str(var_def).lower():
                # Parse range
                match = re.search(r'\[([\d.]+),\s*([\d.]+)\]', str(var_def))
                if match:
                    min_val = float(match.group(1))
                    max_val = float(match.group(2))

                    criterion = QuantitativeSuccessCriterion(
                        name=f"{var_name} Range Constraint",
                        description=f"Verify {var_name} is within valid range",
                        criterion_type=CriterionType.MATHEMATICAL,
                        measurement_method=MeasurementMethod.DIRECT,
                        measurement_procedure=f"Measure {var_name} using appropriate instrument",
                        threshold=(min_val, max_val),
                        threshold_type="range",
                        units="",
                        error_tolerance=0.01,
                        verification_procedure=VerificationProcedure(
                            procedure_id=f"{var_name}_verification",
                            steps=[f"Measure {var_name}", "Compare to range", "Confirm within bounds"],
                            equipment_required=["calibrated_instrument"],
                            expertise_level="qualified_technician",
                            estimated_duration="5_minutes"
                        )
                    )
                    criteria.append(criterion)

    # Add statistical criteria from error analysis
    for error in error_analysis:
        if error.get('impact') == 'critical':
            # Add criterion to verify critical error is mitigated
            criterion = QualitativeSuccessCriterion(
                name=f"{error.get('error_type', 'Error')} Mitigation",
                description=f"Verify that {error.get('description', 'critical error')} is mitigated",
                criterion_type=CriterionType.FUNCTIONAL,
                expected_observation=f"No {error.get('error_type')} observed",
                verification_procedure=error.get('verification_method', 'Inspect'),
                negative_indicator=f"{error.get('error_type')} detected|failure|error",
                ambiguous_indicators=["unclear", "uncertain"]
            )
            criteria.append(criterion)

    return criteria


def derive_criteria_from_physics(
    physics_validation: Dict[str, bool],
    constraints: List[str]
) -> List[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion]]:
    """Derive criteria from physical constraints

    Args:
        physics_validation: Physics validation results
        constraints: Physical constraints

    Returns:
        List of physics-based criteria
    """
    criteria = []

    # Energy conservation
    if 'energy_conservation' in physics_validation:
        criterion = QuantitativeSuccessCriterion(
            name="Energy Conservation",
            description="Verify energy is conserved throughout the process",
            criterion_type=CriterionType.PHYSICAL,
            measurement_method=MeasurementMethod.CALCULATED,
            measurement_procedure="Calculate total energy input and output",
            threshold=1.0,
            threshold_type="exact",
            units="Joules",
            error_tolerance=0.05,  # 5% tolerance
            verification_procedure=VerificationProcedure(
                procedure_id="energy_conservation_verification",
                steps=["Measure all energy inputs", "Measure all energy outputs", "Calculate ratio", "Verify ratio ≈ 1.0"],
                equipment_required=["energy_meter", "calorimeter"],
                expertise_level="physicist",
                estimated_duration="30_minutes"
            )
        )
        criteria.append(criterion)

    # Mass conservation
    if 'mass_conservation' in physics_validation:
        criterion = QuantitativeSuccessCriterion(
            name="Mass Conservation",
            description="Verify mass is conserved throughout the process",
            criterion_type=CriterionType.PHYSICAL,
            measurement_method=MeasurementMethod.DIRECT,
            measurement_procedure="Weigh all inputs and outputs",
            threshold=1.0,
            threshold_type="exact",
            units="grams",
            error_tolerance=0.02,  # 2% tolerance
            verification_procedure=VerificationProcedure(
                procedure_id="mass_conservation_verification",
                steps=["Weigh all inputs", "Weigh all outputs", "Calculate ratio", "Verify ratio ≈ 1.0"],
                equipment_required=["precision_balance"],
                expertise_level="chemist",
                estimated_duration="15_minutes"
            )
        )
        criteria.append(criterion)

    # Thermodynamic constraints
    for constraint in constraints:
        if 'entropy' in constraint.lower() or 'second law' in constraint.lower():
            criterion = QualitativeSuccessCriterion(
                name="Second Law Compliance",
                description="Verify process complies with second law of thermodynamics",
                criterion_type=CriterionType.PHYSICAL,
                expected_observation="Entropy increases or remains constant",
                verification_procedure="Calculate entropy change for all steps",
                negative_indicator="entropy decreases|violates second law",
                ambiguous_indicators=["unclear entropy", "insufficient data"]
            )
            criteria.append(criterion)

    return criteria


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_quantitative_criterion(
    text: str,
    domain: str
) -> Optional[QuantitativeSuccessCriterion]:
    """Parse quantitative criterion from text"""
    # Extract numeric value
    match = re.search(r'(\d+\.?\d*)', text)
    if not match:
        return None

    value = float(match.group(1))

    # Determine threshold type
    if '>' in text or 'greater than' in text.lower() or 'at least' in text.lower():
        threshold_type = "min"
    elif '<' in text or 'less than' in text.lower() or 'no more than' in text.lower():
        threshold_type = "max"
    else:
        threshold_type = "exact"

    # Extract units
    units_match = re.search(r'%|(?:degrees?|°C|°F|K|Pa|atm|V|A|W|Hz|nm|μm|mm|cm|m|g|kg|mol|M)', text, re.IGNORECASE)
    units = units_match.group(0) if units_match else ""

    # Extract measurement type
    measurement_type = re.search(r'(efficiency|yield|purity|temperature|pressure|pH|voltage|current)', text, re.IGNORECASE)
    name = measurement_type.group(0).capitalize() if measurement_type else "Performance Metric"

    return QuantitativeSuccessCriterion(
        name=name,
        description=text,
        criterion_type=CriterionType.QUANTITATIVE,
        measurement_method=MeasurementMethod.DIRECT,
        measurement_procedure=f"Measure {name} using standard method for {domain}",
        threshold=value,
        threshold_type=threshold_type,
        units=units,
        error_tolerance=value * 0.05,  # 5% tolerance
        verification_procedure=VerificationProcedure(
            procedure_id=f"{name}_verification",
            steps=[f"Measure {name}", "Compare to threshold", "Document result"],
            equipment_required=["standard_equipment"],
            expertise_level="qualified_technician",
            estimated_duration="10_minutes"
        )
    )


def _parse_constraint_criterion(
    constraint: str,
    domain: str
) -> Optional[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion]]:
    """Parse constraint as success criterion"""
    # Check if constraint has numeric value
    numeric_match = re.search(r'(\d+\.?\d*)', constraint)
    if numeric_match:
        return _parse_quantitative_criterion(constraint, domain)
    else:
        # Create qualitative criterion
        return QualitativeSuccessCriterion(
            name="Constraint Satisfaction",
            description=constraint,
            criterion_type=CriterionType.FUNCTIONAL,
            expected_observation=constraint.replace('must', 'is').replace('shall', 'is'),
            verification_procedure=f"Verify compliance: {constraint}",
            negative_indicator="violates|exceeds|fails|breaches",
            ambiguous_indicators=["unclear", "uncertain", "borderline"]
        )


def _get_domain_specific_criteria(
    domain: str,
    goal: Dict[str, Any]
) -> List[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion]]:
    """Get domain-specific success criteria"""
    criteria = []

    if domain.lower() in ['chemistry', 'chemical engineering', 'materials science']:
        # Chemistry-specific criteria
        criteria.append(QualitativeSuccessCriterion(
            name="Chemical Stability",
            description="Verify chemical stability of final product",
            criterion_type=CriterionType.PHYSICAL,
            expected_observation="No decomposition or adverse reactions",
            verification_procedure="Perform stability test under standard conditions",
            negative_indicator="decomposition|degradation|reaction|precipitation",
            ambiguous_indicators=["slight change", "minor discoloration"]
        ))

    elif domain.lower() in ['physics', 'engineering']:
        # Physics/engineering criteria
        criteria.append(QuantitativeSuccessCriterion(
            name="Efficiency Threshold",
            description="Verify system meets minimum efficiency requirement",
            criterion_type=CriterionType.QUANTITATIVE,
            measurement_method=MeasurementMethod.CALCULATED,
            measurement_procedure="Calculate efficiency from input/output measurements",
            threshold=0.80,  # 80% minimum
            threshold_type="min",
            units="fraction",
            error_tolerance=0.02,
            verification_procedure=VerificationProcedure(
                procedure_id="efficiency_verification",
                steps=["Measure power input", "Measure useful output", "Calculate ratio", "Verify threshold"],
                equipment_required=["power_meter", "load_meter"],
                expertise_level="engineer",
                estimated_duration="20_minutes"
            )
        ))

    elif domain.lower() in ['biology', 'biotechnology']:
        # Biology-specific criteria
        criteria.append(QualitativeSuccessCriterion(
            name="Biological Activity",
            description="Verify biological activity is maintained",
            criterion_type=CriterionType.FUNCTIONAL,
            expected_observation="Expected biological activity observed",
            verification_procedure="Perform bioassay or activity test",
            negative_indicator="no activity|inactive|denatured|degraded",
            ambiguous_indicators=["reduced activity", "partial activity"]
        ))

    return criteria


# ============================================================================
# Criterion Evaluation
# ============================================================================

def evaluate_all_criteria(
    criteria: List[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion, StatisticalSuccessCriterion]],
    experiment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate all success criteria against experiment results

    Args:
        criteria: List of success criteria
        experiment_results: List of experiment results

    Returns:
        Evaluation results with pass/fail for each criterion
    """
    results = {
        'total_criteria': len(criteria),
        'passed_criteria': 0,
        'failed_criteria': 0,
        'ambiguous_results': 0,
        'criterion_results': [],
        'overall_pass': False
    }

    for criterion in criteria:
        criterion_result = {
            'name': getattr(criterion, 'name', 'Unknown'),
            'type': str(criterion.criterion_type),
            'passed': False,
            'verified': False,
            'measurement': None,
            'error': None
        }

        # Try to evaluate against each experiment result
        for exp_result in experiment_results:
            try:
                # Check if passes
                passed = criterion.passes(exp_result)
                criterion_result['passed'] = passed

                # Get measurement
                try:
                    measurement = criterion.measure(exp_result)
                    criterion_result['measurement'] = measurement
                except:
                    pass

                # Verify
                verified = criterion.verify(exp_result)
                criterion_result['verified'] = verified

                # If found a matching result, break
                if passed is not None:
                    break

            except Exception as e:
                criterion_result['error'] = str(e)
                logger.error(f"Error evaluating criterion {criterion_result['name']}: {e}")

        # Tally results
        if criterion_result['error']:
            results['ambiguous_results'] += 1
        elif criterion_result['passed'] and criterion_result['verified']:
            results['passed_criteria'] += 1
        else:
            results['failed_criteria'] += 1

        results['criterion_results'].append(criterion_result)

    # Overall pass if all criteria pass
    results['overall_pass'] = (
        results['passed_criteria'] == results['total_criteria'] and
        results['ambiguous_results'] == 0
    )

    return results


# ============================================================================
# Main Interface
# ============================================================================

def create_binary_success_criteria(
    goal: Dict[str, Any],
    math_models: List[Dict[str, Any]],
    error_analysis: List[Dict[str, Any]],
    physics_validation: Dict[str, bool],
    domain: str
) -> List[Union[QuantitativeSuccessCriterion, QualitativeSuccessCriterion, StatisticalSuccessCriterion]]:
    """Create complete set of binary success criteria

    Args:
        goal: Invention goal
        math_models: Formalized mathematical models
        error_analysis: Error source analysis
        physics_validation: Physics validation results
        domain: Scientific domain

    Returns:
        Complete list of binary success criteria
    """
    criteria = []

    # Derive from goal requirements
    goal_criteria = derive_criteria_from_goal(goal, domain)
    criteria.extend(goal_criteria)

    # Derive from math models
    math_criteria = derive_criteria_from_math(math_models, error_analysis)
    criteria.extend(math_criteria)

    # Derive from physics
    physics_criteria = derive_criteria_from_physics(physics_validation, goal.get('constraints', []))
    criteria.extend(physics_criteria)

    # Remove duplicates
    seen = set()
    unique_criteria = []
    for criterion in criteria:
        name = getattr(criterion, 'name', '')
        if name and name not in seen:
            seen.add(name)
            unique_criteria.append(criterion)

    logger.info(f"Created {len(unique_criteria)} binary success criteria for {domain}")
    return unique_criteria
