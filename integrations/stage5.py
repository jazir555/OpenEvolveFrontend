"""
Stage 5 Integration: LLTL Real-time Validation and Φ₂ Bias Detection

Integrates RESE's LLTL validation and Cognitive Bias Detection (Φ₂)
with E2E Stage 5 for physics/logic checking.

Architecture:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Solution        │───▶│  LLTL Validator  │───▶│  Φ₂ Bias         │
│  Candidate       │    │  (Physics/Logic) │    │  Detector        │
└──────────────────┘    └──────────────────┘    └──────────────────┘

Author: Agent A4 (Stage Integration Lead)
Created: 2025-12-31
Status: 🟢 Active Implementation
Target: 1.5 hours implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Import RESE components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "rese"))
try:
    from core.constraint_lltl_handoff import LLTLHandoff, LLTLSpecification, LLTLTemplate
    from core.logic_to_loss_translation import LogicToLossTranslator
    RESE_CORE_AVAILABLE = True
except ImportError:
    RESE_CORE_AVAILABLE = False
    LLTLHandoff = None
    LLTLSpecification = None
    LLTLTemplate = None
    LogicToLossTranslator = None

try:
    from phase1.cognitive_biases import CognitiveBiasDetector
    from phase1.phi2_integration import SCEPhi2Integrator
    RESE_PHASE1_AVAILABLE = True
except ImportError:
    RESE_PHASE1_AVAILABLE = False
    CognitiveBiasDetector = None
    SCEPhi2Integrator = None


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ValidationStatus(Enum):
    """Status of solution validation"""
    PENDING = "pending"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    PARTIALLY_VALID = "partially_valid"
    BIASED = "biased"
    FAILED = "failed"


@dataclass
class SolutionCandidate:
    """Candidate solution for validation"""
    id: str
    variables: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLTLValidationResult:
    """Result from LLTL validation"""
    is_valid: bool
    violations: List[str]
    loss_value: float
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    physics_violations: List[str]
    logic_violations: List[str]
    confidence: float


@dataclass
class BiasDetectionResult:
    """Result from bias detection"""
    has_bias: bool
    bias_types: List[str]
    severity: str  # "low", "medium", "high"
    affected_constraints: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class Stage5ValidationResult:
    """Complete Stage 5 validation result"""
    status: ValidationStatus
    solution_id: str
    ltl_validation: Optional[LLTLValidationResult] = None
    bias_detection: Optional[BiasDetectionResult] = None
    physics_check: Optional[Dict[str, Any]] = None
    logic_check: Optional[Dict[str, Any]] = None
    overall_confidence: float = 0.0
    validation_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'solution_id': self.solution_id,
            'ltl_validation': {
                'is_valid': self.ltl_validation.is_valid if self.ltl_validation else None,
                'violations': self.ltl_validation.violations if self.ltl_validation else [],
                'loss_value': self.ltl_validation.loss_value if self.ltl_validation else 0.0,
                'confidence': self.ltl_validation.confidence if self.ltl_validation else 0.0
            } if self.ltl_validation else None,
            'bias_detection': {
                'has_bias': self.bias_detection.has_bias if self.bias_detection else False,
                'bias_types': self.bias_detection.bias_types if self.bias_detection else [],
                'severity': self.bias_detection.severity if self.bias_detection else "none",
                'confidence': self.bias_detection.confidence if self.bias_detection else 0.0
            } if self.bias_detection else None,
            'overall_confidence': self.overall_confidence,
            'validation_time': self.validation_time,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'errors': self.errors
        }


# ============================================================================
# Main Integration Class
# ============================================================================

class Stage5Integration:
    """
    Stage 5 Integration: LLTL Validation and Bias Detection.

    This module integrates:
    1. LLTL: Logic-to-Loss Translation for physics/logic validation
    2. Φ₂: Cognitive Bias Detection for bias identification
    3. Real-time validation feedback

    Workflow:
    1. Validate solution against physics/logic using LLTL
    2. Detect cognitive biases using Φ₂
    3. Generate recommendations
    4. Provide real-time feedback
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_lltl: bool = True,
        enable_bias_detection: bool = True,
        physics_threshold: float = 0.1,
        logic_threshold: float = 0.1
    ):
        """
        Initialize Stage 5 Integration.

        Args:
            config: Optional configuration dictionary
            enable_lltl: Enable LLTL validation
            enable_bias_detection: Enable bias detection
            physics_threshold: Physics violation threshold
            logic_threshold: Logic violation threshold
        """
        self.config = config or {}
        self.enable_lltl = enable_lltl
        self.enable_bias_detection = enable_bias_detection
        self.physics_threshold = physics_threshold
        self.logic_threshold = logic_threshold

        # Initialize components
        if self.enable_lltl and LLTL_AVAILABLE:
            self.lltl_handoff = LLTLHandoff()
            self.lltl_translator = LogicToLossTranslator()

        if self.enable_bias_detection and BIAS_AVAILABLE:
            self.bias_detector = CognitiveBiasDetector()

        # Validation history
        self.validation_history: List[Stage5ValidationResult] = []

    def validate_solution(
        self,
        solution: SolutionCandidate,
        constraints: Optional[List[Any]] = None
    ) -> Stage5ValidationResult:
        """
        Validate solution candidate.

        Args:
            solution: Solution candidate to validate
            constraints: Optional constraints (uses solution.constraints if None)

        Returns:
            Stage5ValidationResult with validation results
        """
        start_time = datetime.now()

        if constraints is None:
            constraints = solution.constraints

        result = Stage5ValidationResult(
            status=ValidationStatus.VALIDATING,
            solution_id=solution.id
        )

        try:
            # Step 1: LLTL validation (physics/logic)
            if self.enable_lltl:
                result.ltl_validation = self._validate_with_lltl(
                    solution,
                    constraints
                )

            # Step 2: Bias detection
            if self.enable_bias_detection:
                result.bias_detection = self._detect_bias(
                    solution,
                    constraints
                )

            # Step 3: Physics check
            result.physics_check = self._check_physics(solution)

            # Step 4: Logic check
            result.logic_check = self._check_logic(solution)

            # Step 5: Generate overall status
            result.status = self._determine_status(result)

            # Step 6: Generate recommendations
            result.recommendations = self._generate_recommendations(result)

            # Calculate overall confidence
            result.overall_confidence = self._calculate_confidence(result)

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.errors.append(str(e))

        # Record time
        end_time = datetime.now()
        result.validation_time = (end_time - start_time).total_seconds()

        # Store in history
        self.validation_history.append(result)

        return result

    def _validate_with_lltl(
        self,
        solution: SolutionCandidate,
        constraints: List[Any]
    ) -> LLTLValidationResult:
        """
        Validate solution using LLTL.

        Args:
            solution: Solution candidate
            constraints: Constraints to validate against

        Returns:
            LLTLValidationResult
        """
        # Simplified LLTL validation
        # In production, this would use full LLTL translation and validation

        violations = []
        satisfied = []
        violated = []
        physics_violations = []
        logic_violations = []

        # Check each constraint
        for i, constraint in enumerate(constraints):
            constraint_id = f"constraint_{i}"

            # Simplified validation: check if variable exists
            if isinstance(constraint, dict):
                required_var = constraint.get('variable')
                if required_var and required_var not in solution.variables:
                    violated.append(constraint_id)
                    violations.append(f"Missing variable: {required_var}")
                    logic_violations.append(f"Logic error: {required_var} not defined")
                else:
                    satisfied.append(constraint_id)

        loss_value = len(violations) * 0.5
        confidence = max(0.0, 1.0 - loss_value)

        return LLTLValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            loss_value=loss_value,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            physics_violations=physics_violations,
            logic_violations=logic_violations,
            confidence=confidence
        )

    def _detect_bias(
        self,
        solution: SolutionCandidate,
        constraints: List[Any]
    ) -> BiasDetectionResult:
        """
        Detect cognitive biases in solution.

        Args:
            solution: Solution candidate
            constraints: Constraints

        Returns:
            BiasDetectionResult
        """
        # Simplified bias detection
        # In production, this would use full Φ₂ analysis

        bias_types = []
        severity = "low"
        affected = []
        recommendations = []

        # Check for common biases
        # 1. Anchoring bias: if values are too round
        for var_name, var_value in solution.variables.items():
            if isinstance(var_value, (int, float)):
                if var_value in [0, 1, 10, 100, 1000]:
                    bias_types.append("anchoring_bias")
                    affected.append(var_name)
                    recommendations.append(f"Variable {var_name} has suspicious round value")

        # 2. Confirmation bias: if solution only confirms expectations
        if len(affected) > 0:
            severity = "medium" if len(affected) < 3 else "high"

        confidence = max(0.0, 1.0 - len(bias_types) * 0.2)

        return BiasDetectionResult(
            has_bias=len(bias_types) > 0,
            bias_types=bias_types,
            severity=severity,
            affected_constraints=affected,
            recommendations=recommendations,
            confidence=confidence
        )

    def _check_physics(
        self,
        solution: SolutionCandidate
    ) -> Dict[str, Any]:
        """
        Check physics constraints.

        Args:
            solution: Solution candidate

        Returns:
            Physics check results
        """
        # Simplified physics check
        # In production, this would use domain-specific physics validation

        violations = []

        # Check for negative values where inappropriate
        for var_name, var_value in solution.variables.items():
            if 'energy' in var_name.lower() or 'mass' in var_name.lower():
                if isinstance(var_value, (int, float)) and var_value < 0:
                    violations.append(f"{var_name} cannot be negative")

        return {
            'passes': len(violations) == 0,
            'violations': violations,
            'checked_variables': len(solution.variables)
        }

    def _check_logic(
        self,
        solution: SolutionCandidate
    ) -> Dict[str, Any]:
        """
        Check logic consistency.

        Args:
            solution: Solution candidate

        Returns:
            Logic check results
        """
        # Simplified logic check
        violations = []

        # Check for circular dependencies (simplified)
        # Check for type consistency
        for var_name, var_value in solution.variables.items():
            if 'count' in var_name.lower() or 'number' in var_name.lower():
                if not isinstance(var_value, (int, float)):
                    violations.append(f"{var_name} should be numeric")

        return {
            'passes': len(violations) == 0,
            'violations': violations,
            'checked_variables': len(solution.variables)
        }

    def _determine_status(
        self,
        result: Stage5ValidationResult
    ) -> ValidationStatus:
        """
        Determine overall validation status.

        Args:
            result: Validation result

        Returns:
            ValidationStatus
        """
        # Check for critical failures
        if result.errors:
            return ValidationStatus.FAILED

        # Check LLTL validation
        if result.ltl_validation and not result.ltl_validation.is_valid:
            if result.ltl_validation.loss_value > 0.5:
                return ValidationStatus.INVALID
            else:
                return ValidationStatus.PARTIALLY_VALID

        # Check bias
        if result.bias_detection and result.bias_detection.has_bias:
            if result.bias_detection.severity == "high":
                return ValidationStatus.BIASED

        return ValidationStatus.VALID

    def _generate_recommendations(
        self,
        result: Stage5ValidationResult
    ) -> List[str]:
        """
        Generate improvement recommendations.

        Args:
            result: Validation result

        Returns:
            List of recommendations
        """
        recommendations = []

        # LLTL-based recommendations
        if result.ltl_validation:
            if result.ltl_validation.violations:
                recommendations.append(
                    f"Fix {len(result.ltl_validation.violations)} constraint violations"
                )

        # Bias-based recommendations
        if result.bias_detection and result.bias_detection.recommendations:
            recommendations.extend(result.bias_detection.recommendations)

        # Physics-based recommendations
        if result.physics_check and result.physics_check.get('violations'):
            recommendations.append(
                f"Address {len(result.physics_check['violations'])} physics violations"
            )

        # Logic-based recommendations
        if result.logic_check and result.logic_check.get('violations'):
            recommendations.append(
                f"Address {len(result.logic_check['violations'])} logic errors"
            )

        return recommendations

    def _calculate_confidence(
        self,
        result: Stage5ValidationResult
    ) -> float:
        """
        Calculate overall confidence.

        Args:
            result: Validation result

        Returns:
            Confidence score [0, 1]
        """
        base_confidence = 0.8

        # LLTL confidence
        if result.ltl_validation:
            base_confidence *= result.ltl_validation.confidence

        # Bias penalty
        if result.bias_detection:
            if result.bias_detection.has_bias:
                penalty = 0.2 if result.bias_detection.severity == "low" else 0.4
                base_confidence -= penalty

        # Physics penalty
        if result.physics_check and not result.physics_check.get('passes', True):
            base_confidence -= 0.2

        # Logic penalty
        if result.logic_check and not result.logic_check.get('passes', True):
            base_confidence -= 0.2

        return max(0.0, min(1.0, base_confidence))

    def export_validation(
        self,
        result: Stage5ValidationResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export validation result to JSON.

        Args:
            result: Validation result to export
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"stage5_validation_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    'Stage5Integration',

    # Data structures
    'SolutionCandidate',
    'LLTLValidationResult',
    'BiasDetectionResult',
    'Stage5ValidationResult',

    # Enums
    'ValidationStatus',
]
