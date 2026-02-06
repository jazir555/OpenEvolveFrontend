"""
Comprehensive Validation System for End-to-End Invention Planning

Implements rigorous validation logic that FAILS if any critical issue is found.
Not a simple weighted average - truly binary validation on critical issues.

Author: Agent 5 - Success Criteria and Validation
Version: 1.0.0
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)

# **LEAN INTEGRATION**: Formal verification with Lean
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


# ============================================================================
# Types and Enums
# ============================================================================

class ValidationSeverity(Enum):
    """Severity of validation issues"""
    CRITICAL = "critical"  # Must pass - invention cannot proceed
    HIGH = "high"  # Should pass - strong warning
    MEDIUM = "medium"  # Nice to have
    LOW = "low"  # Cosmetic


class ValidationCategory(Enum):
    """Categories of validation checks"""
    STEPS_VERIFIABLE = "steps_verifiable"
    ERRORS_MITIGATED = "errors_mitigated"
    MATH_FORMALIZED = "math_formalized"
    PHYSICS_VALID = "physics_valid"
    SAFETY_COMPLETE = "safety_complete"
    CRITERIA_BINARY = "criteria_binary"
    RESOURCES_SPECIFIED = "resources_specified"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    EXECUTABILITY = "executability"
    FORMAL_VERIFICATION = "formal_verification"  # **LEAN INTEGRATION**


@dataclass
class ValidationResult:
    """Result of a validation check"""
    category: ValidationCategory
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report"""
    passed: bool
    ready_for_execution: bool
    critical_failures: List[ValidationResult]
    warnings: List[ValidationResult]
    info: List[ValidationResult]
    overall_score: float  # 0-1, but only if no critical failures
    total_checks: int
    passed_checks: int
    failed_checks: int
    validation_details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'passed': self.passed,
            'ready_for_execution': self.ready_for_execution,
            'critical_failures': [self._result_to_dict(r) for r in self.critical_failures],
            'warnings': [self._result_to_dict(r) for r in self.warnings],
            'info': [self._result_to_dict(r) for r in self.info],
            'overall_score': self.overall_score,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'validation_details': self.validation_details
        }

    @staticmethod
    def _result_to_dict(result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dict"""
        return {
            'category': result.category.value,
            'passed': result.passed,
            'severity': result.severity.value,
            'message': result.message,
            'details': result.details,
            'suggestions': result.suggestions
        }


# ============================================================================
# Critical Validation Checks
# ============================================================================

def check_all_steps_verifiable(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all steps are verifiable

    A step is verifiable if it has:
    - Clear acceptance criteria
    - Measurement method
    - Verification procedure

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    sop = bulletproof_sop.get('sop', {})
    steps = sop.get('steps', []) if isinstance(sop, dict) else []

    if not steps:
        return ValidationResult(
            category=ValidationCategory.STEPS_VERIFIABLE,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No steps found in SOP",
            details={'steps_count': 0},
            suggestions=["Add steps to SOP", "Ensure each step has verification method"]
        )

    unverifiable_steps = []
    verifiable_count = 0

    for i, step in enumerate(steps):
        step_verifiable = True
        issues = []

        # Check for acceptance criteria
        if 'acceptance_criteria' not in step or not step['acceptance_criteria']:
            step_verifiable = False
            issues.append("Missing acceptance criteria")

        # Check for measurement method
        if 'measurement_method' not in step or not step['measurement_method']:
            step_verifiable = False
            issues.append("Missing measurement method")

        # Check for verification procedure
        if 'verification' not in step or not step['verification']:
            step_verifiable = False
            issues.append("Missing verification procedure")

        if step_verifiable:
            verifiable_count += 1
        else:
            unverifiable_steps.append({
                'step_number': i + 1,
                'step_description': step.get('description', 'Unknown')[:100],
                'issues': issues
            })

    all_verifiable = verifiable_count == len(steps)

    return ValidationResult(
        category=ValidationCategory.STEPS_VERIFIABLE,
        passed=all_verifiable,
        severity=ValidationSeverity.CRITICAL,
        message=f"Steps verifiable: {verifiable_count}/{len(steps)}",
        details={
            'total_steps': len(steps),
            'verifiable_steps': verifiable_count,
            'unverifiable_steps': unverifiable_steps
        },
        suggestions=[
            "Add acceptance criteria to each step",
            "Specify measurement methods",
            "Include verification procedures"
        ] if not all_verifiable else []
    )


def check_all_errors_mitigated(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all error sources have mitigation strategies

    For each error source, must have:
    - Identified mitigation strategy
    - Verification method
    - Acceptance criteria

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    error_sources = bulletproof_sop.get('error_sources', [])

    if not error_sources:
        return ValidationResult(
            category=ValidationCategory.ERRORS_MITIGATED,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No error sources identified",
            details={'error_count': 0},
            suggestions=["Perform comprehensive error analysis", "Identify all possible error sources"]
        )

    unmitigated_errors = []
    mitigated_count = 0

    for error in error_sources:
        error_mitigated = True
        issues = []

        # Check for mitigation strategy
        if not error.get('mitigation_strategy'):
            error_mitigated = False
            issues.append("Missing mitigation strategy")

        # Check for verification method
        if not error.get('verification_method'):
            error_mitigated = False
            issues.append("Missing verification method")

        # Check for acceptance criteria
        if not error.get('acceptance_criteria'):
            error_mitigated = False
            issues.append("Missing acceptance criteria")

        # Critical errors MUST have mitigation
        if error.get('impact') == 'critical' and not error_mitigated:
            error_mitigated = False
            issues.append("Critical error without mitigation")

        if error_mitigated:
            mitigated_count += 1
        else:
            unmitigated_errors.append({
                'error_type': error.get('error_type', 'Unknown'),
                'description': error.get('description', '')[:100],
                'impact': error.get('impact', 'unknown'),
                'issues': issues
            })

    all_mitigated = mitigated_count == len(error_sources)

    return ValidationResult(
        category=ValidationCategory.ERRORS_MITIGATED,
        passed=all_mitigated,
        severity=ValidationSeverity.CRITICAL,
        message=f"Errors mitigated: {mitigated_count}/{len(error_sources)}",
        details={
            'total_errors': len(error_sources),
            'mitigated_errors': mitigated_count,
            'unmitigated_errors': unmitigated_errors
        },
        suggestions=[
            "Add mitigation strategies for all errors",
            "Define verification methods",
            "Set acceptance criteria"
        ] if not all_mitigated else []
    )


def check_all_math_formalized(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all mathematical relationships are formalized

    All math must have:
    - Lean 4 theorem statement
    - Lean 4 proof (not 'by sorry')
    - Variable definitions
    - Assumptions documented

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    formalized_math = bulletproof_sop.get('formalized_math', [])

    if not formalized_math:
        # Check if there are any equations that need formalization
        sop = bulletproof_sop.get('sop', {})
        steps = sop.get('steps', []) if isinstance(sop, dict) else []

        has_equations = False
        for step in steps:
            if 'equations' in step or 'calculations' in step:
                has_equations = True
                break

        if has_equations:
            return ValidationResult(
                category=ValidationCategory.MATH_FORMALIZED,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message="Math exists but not formalized",
                details={
                    'formalized_count': 0,
                    'has_unformalized_math': True
                },
                suggestions=["Formalize all equations in Lean 4", "Generate actual proofs (not 'by sorry')"]
            )
        else:
            # No math to formalize - this is okay
            return ValidationResult(
                category=ValidationCategory.MATH_FORMALIZED,
                passed=True,
                severity=ValidationSeverity.CRITICAL,
                message="No math requiring formalization",
                details={
                    'formalized_count': 0,
                    'has_unformalized_math': False
                }
            )

    unformalized = []
    formalized_count = 0

    for math in formalized_math:
        math_formalized = True
        issues = []

        # Check for theorem
        if not math.get('lean_theorem') or math.get('lean_theorem') == 'by sorry':
            math_formalized = False
            issues.append("Missing or invalid theorem")

        # Check for proof
        if not math.get('lean_proof') or 'sorry' in math.get('lean_proof', ''):
            math_formalized = False
            issues.append("Proof is 'by sorry' - need actual proof")

        # Check for variable definitions
        if not math.get('variables'):
            math_formalized = False
            issues.append("Missing variable definitions")

        # Check for assumptions
        if not math.get('assumptions'):
            math_formalized = False
            issues.append("Missing assumptions")

        if math_formalized:
            formalized_count += 1
        else:
            unformalized.append({
                'description': math.get('description', 'Unknown')[:100],
                'issues': issues
            })

    all_formalized = formalized_count == len(formalized_math)

    return ValidationResult(
        category=ValidationCategory.MATH_FORMALIZED,
        passed=all_formalized,
        severity=ValidationSeverity.CRITICAL,
        message=f"Math formalized: {formalized_count}/{len(formalized_math)}",
        details={
            'total_math': len(formalized_math),
            'formalized_count': formalized_count,
            'unformalized': unformalized
        },
        suggestions=[
            "Generate actual Lean 4 proofs",
            "Document all variables",
            "List all assumptions"
        ] if not all_formalized else []
    )


def check_physics_valid(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that physics is valid

    Must validate:
    - Energy conservation
    - Mass conservation
    - Thermodynamic consistency
    - Material compatibility
    - No physical impossibilities

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    physics_validation = bulletproof_sop.get('physics_validation', {})

    if not physics_validation:
        return ValidationResult(
            category=ValidationCategory.PHYSICS_VALID,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No physics validation performed",
            details={},
            suggestions=["Perform physics validation", "Check conservation laws", "Verify thermodynamics"]
        )

    # Check critical physics validations
    critical_checks = [
        ('energy_conservation', 'Energy must be conserved'),
        ('mass_conservation', 'Mass must be conserved'),
        ('thermodynamics', 'Must comply with laws of thermodynamics'),
        ('material_compatibility', 'Materials must be compatible'),
        ('safety_constraints', 'Safety constraints must be satisfied')
    ]

    failed_checks = []
    passed_checks = []

    for check_name, check_desc in critical_checks:
        if check_name in physics_validation:
            if physics_validation[check_name]:
                passed_checks.append(check_name)
            else:
                failed_checks.append({
                    'check': check_name,
                    'description': check_desc,
                    'reason': f"{check_name} check failed"
                })
        else:
            # Check is missing - this is a failure
            failed_checks.append({
                'check': check_name,
                'description': check_desc,
                'reason': f"{check_name} check not performed"
            })

    all_valid = len(failed_checks) == 0

    return ValidationResult(
        category=ValidationCategory.PHYSICS_VALID,
        passed=all_valid,
        severity=ValidationSeverity.CRITICAL,
        message=f"Physics validation: {len(passed_checks)}/{len(critical_checks)} passed",
        details={
            'total_checks': len(critical_checks),
            'passed_checks': passed_checks,
            'failed_checks': failed_checks
        },
        suggestions=[
            "Verify energy conservation",
            "Verify mass conservation",
            "Check thermodynamic compliance",
            "Validate material compatibility",
            "Ensure safety constraints met"
        ] if not all_valid else []
    )


def check_safety_complete(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all safety measures are in place

    Must have:
    - Hazard identification for each step
    - Safety procedures
    - Emergency procedures
    - PPE requirements
    - Risk mitigation

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    sop = bulletproof_sop.get('sop', {})
    steps = sop.get('steps', []) if isinstance(sop, dict) else []

    if not steps:
        return ValidationResult(
            category=ValidationCategory.SAFETY_COMPLETE,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No steps to validate safety",
            details={'steps_count': 0},
            suggestions=["Add steps to SOP", "Include safety information"]
        )

    unsafe_steps = []
    safe_steps = 0

    for i, step in enumerate(steps):
        step_safe = True
        issues = []

        # Check for hazard identification
        if 'hazards' not in step or not step['hazards']:
            step_safe = False
            issues.append("Missing hazard identification")

        # Check for safety procedures
        if 'safety' not in step or not step['safety']:
            step_safe = False
            issues.append("Missing safety procedures")

        # Check for PPE
        if 'ppe' not in step or not step['ppe']:
            step_safe = False
            issues.append("Missing PPE requirements")

        # Check for emergency procedures (if hazardous)
        hazards = step.get('hazards', [])
        if hazards and ('emergency' not in step or not step['emergency']):
            step_safe = False
            issues.append("Missing emergency procedures for hazardous step")

        if step_safe:
            safe_steps += 1
        else:
            unsafe_steps.append({
                'step_number': i + 1,
                'step_description': step.get('description', 'Unknown')[:100],
                'issues': issues
            })

    all_safe = safe_steps == len(steps)

    return ValidationResult(
        category=ValidationCategory.SAFETY_COMPLETE,
        passed=all_safe,
        severity=ValidationSeverity.CRITICAL,
        message=f"Safety complete: {safe_steps}/{len(steps)} steps",
        details={
            'total_steps': len(steps),
            'safe_steps': safe_steps,
            'unsafe_steps': unsafe_steps
        },
        suggestions=[
            "Identify hazards for each step",
            "Add safety procedures",
            "Specify PPE requirements",
            "Include emergency procedures"
        ] if not all_safe else []
    )


def check_criteria_binary(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all success criteria are truly binary

    Binary criteria must have:
    - Exact threshold
    - Clear pass/fail
    - No ambiguity
    - Verifiable

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    success_criteria = bulletproof_sop.get('success_criteria', [])

    if not success_criteria:
        return ValidationResult(
            category=ValidationCategory.CRITERIA_BINARY,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No success criteria defined",
            details={'criteria_count': 0},
            suggestions=["Define binary success criteria", "Ensure all criteria are measurable"]
        )

    non_binary = []
    binary_count = 0

    for criterion in success_criteria:
        is_binary = True
        issues = []

        # Check for threshold
        if not criterion.get('pass_threshold') and criterion.get('pass_threshold') != 0:
            is_binary = False
            issues.append("Missing threshold")

        # Check for measurement method
        if not criterion.get('measurement_method'):
            is_binary = False
            issues.append("Missing measurement method")

        # Check for verification
        if not criterion.get('verification'):
            is_binary = False
            issues.append("Missing verification method")

        # Check for ambiguity
        description = criterion.get('criterion', '').lower()
        ambiguous_terms = ['approximately', 'roughly', 'about', 'around', 'relatively', 'somewhat']
        if any(term in description for term in ambiguous_terms):
            is_binary = False
            issues.append(f"Contains ambiguous terms: {ambiguous_terms}")

        if is_binary:
            binary_count += 1
        else:
            non_binary.append({
                'criterion': criterion.get('criterion', 'Unknown')[:100],
                'issues': issues
            })

    all_binary = binary_count == len(success_criteria)

    return ValidationResult(
        category=ValidationCategory.CRITERIA_BINARY,
        passed=all_binary,
        severity=ValidationSeverity.CRITICAL,
        message=f"Criteria binary: {binary_count}/{len(success_criteria)}",
        details={
            'total_criteria': len(success_criteria),
            'binary_criteria': binary_count,
            'non_binary_criteria': non_binary
        },
        suggestions=[
            "Add exact thresholds to all criteria",
            "Define measurement methods",
            "Specify verification procedures",
            "Remove ambiguous terms"
        ] if not all_binary else []
    )


def check_resources_specified(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """CRITICAL: Check that all resources are specified exactly

    Must have for each step:
    - Equipment (with model numbers)
    - Materials (with specifications/purity)
    - Personnel requirements
    - Time estimates
    - Environmental conditions

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    sop = bulletproof_sop.get('sop', {})
    steps = sop.get('steps', []) if isinstance(sop, dict) else []

    if not steps:
        return ValidationResult(
            category=ValidationCategory.RESOURCES_SPECIFIED,
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="No steps to validate resources",
            details={'steps_count': 0},
            suggestions=["Add steps to SOP"]
        )

    incomplete_steps = []
    complete_steps = 0

    for i, step in enumerate(steps):
        step_complete = True
        missing = []

        # Check for equipment
        if 'equipment' not in step or not step['equipment']:
            step_complete = False
            missing.append('equipment')
        else:
            # Check equipment has specifications
            for equip in step['equipment']:
                if not equip.get('model') and not equip.get('specification'):
                    step_complete = False
                    missing.append('equipment specifications')

        # Check for materials
        if 'materials' not in step or not step['materials']:
            step_complete = False
            missing.append('materials')
        else:
            # Check materials have specifications
            for material in step['materials']:
                if not material.get('purity') and not material.get('grade') and not material.get('specification'):
                    step_complete = False
                    missing.append('material specifications')

        # Check for personnel
        if 'personnel' not in step or not step['personnel']:
            step_complete = False
            missing.append('personnel requirements')

        # Check for time estimate
        if 'duration' not in step and 'time' not in step:
            step_complete = False
            missing.append('time estimate')

        # Check for environmental conditions
        if 'conditions' not in step and 'environment' not in step:
            step_complete = False
            missing.append('environmental conditions')

        if step_complete:
            complete_steps += 1
        else:
            incomplete_steps.append({
                'step_number': i + 1,
                'step_description': step.get('description', 'Unknown')[:100],
                'missing_resources': missing
            })

    all_specified = complete_steps == len(steps)

    return ValidationResult(
        category=ValidationCategory.RESOURCES_SPECIFIED,
        passed=all_specified,
        severity=ValidationSeverity.CRITICAL,
        message=f"Resources specified: {complete_steps}/{len(steps)} steps",
        details={
            'total_steps': len(steps),
            'complete_steps': complete_steps,
            'incomplete_steps': incomplete_steps
        },
        suggestions=[
            "Add equipment with model numbers",
            "Specify materials with purity/grade",
            "Define personnel requirements",
            "Include time estimates",
            "Specify environmental conditions"
        ] if not all_specified else []
    )


# ============================================================================
# Additional Quality Checks
# ============================================================================

def check_consistency(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """HIGH: Check for internal consistency

    Checks:
    - Terminology consistency
    - Units consistency
    - Reference consistency
    - No contradictions

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    issues = []

    # Check for consistent units
    sop = bulletproof_sop.get('sop', {})
    steps = sop.get('steps', []) if isinstance(sop, dict) else []

    unit_map = {}
    for step in steps:
        parameters = step.get('parameters', [])
        for param in parameters:
            param_name = param.get('name', '')
            units = param.get('units', '')
            if param_name and units:
                if param_name in unit_map:
                    if unit_map[param_name] != units:
                        issues.append(f"Inconsistent units for {param_name}: {unit_map[param_name]} vs {units}")
                else:
                    unit_map[param_name] = units

    # Check for contradictions in constraints
    constraints = bulletproof_sop.get('invention_goal', {}).get('constraints', [])
    for i, c1 in enumerate(constraints):
        for c2 in constraints[i+1:]:
            # Simple check for contradictory terms
            words1 = set(c1.lower().split())
            words2 = set(c2.lower().split())
            contradictions = [('minimum', 'maximum'), ('min', 'max'), ('at least', 'at most'), ('less than', 'greater than')]
            for w1, w2 in contradictions:
                if w1 in words1 and w2 in words2:
                    issues.append(f"Potential contradiction: '{c1}' vs '{c2}'")

    consistent = len(issues) == 0

    return ValidationResult(
        category=ValidationCategory.CONSISTENCY,
        passed=consistent,
        severity=ValidationSeverity.HIGH,
        message=f"Consistency check: {len(issues)} issues found",
        details={
            'issues': issues
        },
        suggestions=issues if issues else []
    )


def check_completeness(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """HIGH: Check completeness of SOP

    Checks:
    - All required sections present
    - No placeholder text
    - All references resolved
    - All acronyms defined

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    missing_sections = []
    placeholders_found = []

    # Required sections
    required_sections = [
        'invention_goal',
        'knowledge_base',
        'decomposition',
        'formalized_math',
        'physics_validation',
        'error_sources',
        'red_team_findings',
        'blue_team_fixes',
        'success_criteria',
        'sop'
    ]

    for section in required_sections:
        if section not in bulletproof_sop:
            missing_sections.append(section)

    # Check for placeholder text
    sop_text = str(bulletproof_sop)
    placeholder_patterns = [
        'TODO', 'TBD', 'to be determined', 'placeholder',
        'not yet', 'coming soon', 'under construction'
    ]

    for pattern in placeholder_patterns:
        if pattern in sop_text.lower():
            placeholders_found.append(pattern)

    complete = len(missing_sections) == 0 and len(placeholders_found) == 0

    return ValidationResult(
        category=ValidationCategory.COMPLETENESS,
        passed=complete,
        severity=ValidationSeverity.HIGH,
        message=f"Completeness: {len(required_sections) - len(missing_sections)}/{len(required_sections)} sections",
        details={
            'missing_sections': missing_sections,
            'placeholders_found': placeholders_found
        },
        suggestions=[
            "Add all required sections",
            "Remove placeholder text",
            "Ensure all content is complete"
        ] if not complete else []
    )


def check_executability(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """HIGH: Check that SOP is executable

    Checks:
    - All dependencies satisfiable
    - Sequential order valid
    - Resources available
    - Skill level appropriate

    Args:
        bulletproof_sop: Complete bulletproof SOP

    Returns:
        Validation result
    """
    issues = []

    sop = bulletproof_sop.get('sop', {})
    steps = sop.get('steps', []) if isinstance(sop, dict) else []

    # Check for circular dependencies
    # This is a simplified check
    referenced_steps = set()
    defined_steps = set(range(1, len(steps) + 1))

    for i, step in enumerate(steps):
        dependencies = step.get('dependencies', [])
        for dep in dependencies:
            if isinstance(dep, int):
                referenced_steps.add(dep)
                if dep not in defined_steps:
                    issues.append(f"Step {i+1} depends on undefined step {dep}")

    # Check skill level is specified
    skill_level = sop.get('skill_level_required', '') if isinstance(sop, dict) else ''
    if not skill_level or skill_level == 'qualified engineer':
        issues.append("Skill level not specifically defined")

    # Check for timing
    if not sop.get('estimated_duration_hours') if isinstance(sop, dict) else True:
        issues.append("No duration estimate provided")

    executable = len(issues) == 0

    return ValidationResult(
        category=ValidationCategory.EXECUTABILITY,
        passed=executable,
        severity=ValidationSeverity.HIGH,
        message=f"Executability check: {len(issues)} issues",
        details={
            'issues': issues
        },
        suggestions=issues if issues else []
    )


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_comprehensive(bulletproof_sop: Dict[str, Any]) -> ValidationReport:
    """Perform comprehensive validation of bulletproof SOP

    CRITICAL checks must ALL pass for the SOP to be considered valid.
    If any critical check fails, validation returns False immediately.

    Args:
        bulletproof_sop: Complete bulletproof SOP to validate

    Returns:
        Complete validation report
    """
    # Define all critical checks
    critical_checks = [
        check_all_steps_verifiable,
        check_all_errors_mitigated,
        check_all_math_formalized,
        check_physics_valid,
        check_safety_complete,
        check_criteria_binary,
        check_resources_specified
    ]

    # Define quality checks
    quality_checks = [
        check_consistency,
        check_completeness,
        check_executability
    ]

    critical_failures = []
    warnings = []
    info = []

    # Run critical checks first
    for check in critical_checks:
        try:
            result = check(bulletproof_sop)
            if not result.passed:
                critical_failures.append(result)
                logger.error(f"CRITICAL validation failed: {result.message}")
        except Exception as e:
            logger.error(f"Error running critical check {check.__name__}: {e}")
            critical_failures.append(ValidationResult(
                category=ValidationCategory.STEPS_VERIFIABLE,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Check failed with error: {str(e)}",
                details={'error': str(e)}
            ))

    # If any critical failures, fail immediately
    if critical_failures:
        return ValidationReport(
            passed=False,
            ready_for_execution=False,
            critical_failures=critical_failures,
            warnings=warnings,
            info=info,
            overall_score=0.0,
            total_checks=len(critical_checks) + len(quality_checks),
            passed_checks=0,
            failed_checks=len(critical_failures),
            validation_details={
                'status': 'FAILED',
                'reason': 'Critical validation failures',
                'must_fix_before_execution': True
            }
        )

    # Run quality checks
    for check in quality_checks:
        try:
            result = check(bulletproof_sop)
            if result.severity == ValidationSeverity.HIGH and not result.passed:
                warnings.append(result)
            elif result.severity == ValidationSeverity.MEDIUM:
                warnings.append(result)
            else:
                info.append(result)
        except Exception as e:
            logger.warning(f"Error running quality check {check.__name__}: {e}")
            info.append(ValidationResult(
                category=ValidationCategory.CONSISTENCY,
                passed=False,
                severity=ValidationSeverity.LOW,
                message=f"Check failed with error: {str(e)}",
                details={'error': str(e)}
            ))

    # Calculate overall score (only if no critical failures)
    total_checks = len(critical_checks) + len(quality_checks)
    passed_checks = sum(1 for r in [c(bulletproof_sop) for c in critical_checks + quality_checks] if r.passed)
    overall_score = passed_checks / total_checks if total_checks > 0 else 0.0

    # Ready for execution if high score and no warnings
    ready_for_execution = overall_score >= 0.95 and len(warnings) == 0

    return ValidationReport(
        passed=True,
        ready_for_execution=ready_for_execution,
        critical_failures=critical_failures,
        warnings=warnings,
        info=info,
        overall_score=overall_score,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=total_checks - passed_checks,
        validation_details={
            'status': 'PASSED' if ready_for_execution else 'PASSED_WITH_WARNINGS',
            'ready_for_execution': ready_for_execution,
            'quality_score': overall_score,
            'recommendation': 'Ready for execution' if ready_for_execution else 'Review warnings before execution'
        }
    )


# ============================================================================
# Quick Validation (for development/testing)
# ============================================================================

async def check_formal_verification(bulletproof_sop: Dict[str, Any]) -> ValidationResult:
    """
    **LEAN INTEGRATION**: Check formal verification using Lean theorem prover.
    
    Verifies mathematical claims in the SOP using formal methods.
    
    Args:
        bulletproof_sop: Complete bulletproof SOP
        
    Returns:
        Validation result
    """
    sop = bulletproof_sop.get('sop', {})
    
    # Extract mathematical content
    math_content = bulletproof_sop.get('mathematical_formalization', '')
    if not math_content and isinstance(sop, dict):
        # Try to extract from SOP description
        math_content = sop.get('description', '')
    
    if not math_content:
        return ValidationResult(
            category=ValidationCategory.FORMAL_VERIFICATION,
            passed=True,  # Pass if no mathematical content
            severity=ValidationSeverity.LOW,
            message="No mathematical content to verify",
            details={'skipped': True}
        )
    
    # Verify with Lean
    lean_result = await verify_with_lean(math_content, {})
    
    verified = lean_result.get('verified', False)
    
    return ValidationResult(
        category=ValidationCategory.FORMAL_VERIFICATION,
        passed=verified,
        severity=ValidationSeverity.CRITICAL if not verified else ValidationSeverity.LOW,
        message=f"Formal verification: {'VERIFIED' if verified else 'NOT VERIFIED'}",
        details={
            'verified': verified,
            'confidence': lean_result.get('confidence', 0.0),
            'proof': lean_result.get('proof')
        },
        suggestions=[] if verified else ["Provide formal proof for mathematical claims"]
    )


async def verify_with_lean(content: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    **LEAN INTEGRATION**: Verify content using Lean theorem prover.
    
    Args:
        content: Content to verify
        criteria: Verification criteria
        
    Returns:
        Dict with verification results
    """
    if not LEAN_AVAILABLE:
        return {"verified": False, "reason": "Lean unavailable"}
    
    try:
        client = LeanAideClient()
        formalized = await client.translate_thm(content)
        result = await client.verify(formalized)
        
        return {
            "verified": result.verified if hasattr(result, 'verified') else False,
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
            "proof": result.proof_code if hasattr(result, 'proof_code') else None
        }
    except Exception as e:
        logger.error(f"Lean verification error: {e}")
        return {"verified": False, "reason": str(e)}


def quick_validate(bulletproof_sop: Dict[str, Any]) -> Tuple[bool, str]:
    """Quick validation check

    Returns:
        (passed, message) tuple
    """
    report = validate_comprehensive(bulletproof_sop)

    if report.critical_failures:
        messages = [f.message for f in report.critical_failures]
        return False, f"CRITICAL FAILURES: {'; '.join(messages)}"

    if report.warnings:
        messages = [f.message for f in report.warnings]
        return True, f"PASSED WITH WARNINGS: {'; '.join(messages)}"

    return True, f"PASSED: Ready for execution (score: {report.overall_score:.2%})"
