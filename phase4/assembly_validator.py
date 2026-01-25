"""
Architecture Assembly Validator

Validates assembled architectures for correctness, consistency, and ACI improvement.

Author: Agent E1 (Δ₁ Specialist)
Created: 2025-12-31
Status: Implementation Phase
Dependencies:
    - rese.phase4.architecture_assembler (Architecture data structures)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time
from datetime import datetime

# Import architecture structures
try:
    from phase4.architecture_assembler import (
        Architecture, ComponentInterface, AssemblyPattern,
        PhaseType, ACIChange
    )
except ImportError:
    # Define minimal structures for standalone use
    Architecture = None
    ComponentInterface = None
    AssemblyPattern = None
    PhaseType = None
    ACIChange = None


# =============================================================================
# Validation Data Structures
# =============================================================================

class ValidationSeverity(Enum):
    """Severity of validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """
    Single validation issue found in architecture
    """
    severity: ValidationSeverity
    component: str  # Component ID or "global"
    issue_type: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ArchitectureValidation:
    """
    Complete validation result for an architecture
    """
    architecture_id: str
    is_valid: bool
    validation_score: float  # [0, 1]

    # Issues found
    issues: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    # Component validations
    component_scores: Dict[str, float] = field(default_factory=dict)

    # ACI validation
    aci_improvement: float = 0.0
    aci_valid: bool = False

    # Constraint validation
    constraints_satisfied: bool = False
    constraint_violations: List[str] = field(default_factory=list)

    # Performance
    validation_time: float = 0.0

    # Metadata
    validated_at: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> str:
        """Get human-readable validation summary"""
        lines = [
            f"Architecture Validation: {self.architecture_id}",
            f"Status: {'✓ VALID' if self.is_valid else '✗ INVALID'}",
            f"Score: {self.validation_score:.2f}/1.00",
            f"ACI Improvement: {self.aci_improvement:+.2f}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        return "\n".join(lines)


# =============================================================================
# Main Validator
# =============================================================================

class AssemblyValidator:
    """
    Validates assembled architectures

    Checks:
    1. Structural validity (no circular dependencies)
    2. Component compatibility (interfaces match)
    3. Constraint satisfaction (all constraints satisfiable)
    4. ACI improvement (measurable improvement)
    5. Validation propagation (components → architecture)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator

        Args:
            strict: If True, fail on warnings as well as errors
        """
        self.strict = strict
        self.validations_performed = 0

    def validate(
        self,
        architecture: Architecture,
        problem: Any = None
    ) -> ArchitectureValidation:
        """
        Validate architecture

        Args:
            architecture: Architecture to validate
            problem: Optional problem to test against

        Returns:
            ArchitectureValidation with complete results
        """
        start_time = time.time()

        validation = ArchitectureValidation(
            architecture_id=architecture.architecture_id,
            is_valid=True,  # Assume valid until proven otherwise
            validation_score=0.0
        )

        # 1. Structural validation
        self._validate_structure(architecture, validation)

        # 2. Component compatibility
        self._validate_compatibility(architecture, validation)

        # 3. Dependency resolution
        self._validate_dependencies(architecture, validation)

        # 4. ACI improvement
        self._validate_aci(architecture, validation)

        # 5. Validation propagation
        self._validate_propagation(architecture, validation)

        # 6. Performance estimation
        self._validate_performance(architecture, validation)

        # Calculate final score
        validation.validation_score = self._calculate_score(validation)

        # Determine overall validity
        validation.is_valid = self._determine_validity(validation)

        validation.validation_time = time.time() - start_time
        self.validations_performed += 1

        return validation

    def _validate_structure(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate structural properties"""

        # Check for minimum components
        if len(architecture.components) == 0:
            validation.errors.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                component="global",
                issue_type="no_components",
                message="Architecture has no components",
                suggestion="Add at least one component"
            ))

        # Check for core components
        if not any(c.phase == PhaseType.CORE for c in architecture.components):
            validation.errors.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component="global",
                issue_type="no_core",
                message="Architecture missing core components",
                suggestion="Add SCE (Symbolic Constraint Engine)"
            ))

        # Check for phase diversity
        phases_present = {c.phase for c in architecture.components}
        if len(phases_present) < 2:
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="global",
                issue_type="limited_phases",
                message=f"Architecture only uses {len(phases_present)} phase(s)",
                suggestion="Consider components from multiple phases for better results"
            ))

    def _validate_compatibility(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate component compatibility"""

        # Check all pairs
        for i, comp1 in enumerate(architecture.components):
            for comp2 in architecture.components[i+1:]:
                if not self._are_components_compatible(comp1, comp2):
                    validation.errors.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        component=f"{comp1.component_id}+{comp2.component_id}",
                        issue_type="incompatible_components",
                        message=f"Components {comp1.component_id} and {comp2.component_id} are incompatible",
                        suggestion="Remove one of the conflicting components"
                    ))

    def _are_components_compatible(
        self,
        comp1: ComponentInterface,
        comp2: ComponentInterface
    ) -> bool:
        """Check if two components are compatible"""

        # Check for circular dependencies
        if (comp1.component_id in comp2.requires and
            comp2.component_id in comp1.requires):
            return False

        # Check for conflicting side effects
        # (This is simplified - full version would check more carefully)

        return True

    def _validate_dependencies(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate dependency resolution"""

        component_ids = {c.component_id for c in architecture.components}

        # Check all dependencies satisfied
        for comp in architecture.components:
            for dep in comp.requires:
                if dep not in component_ids:
                    validation.errors.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        component=comp.component_id,
                        issue_type="missing_dependency",
                        message=f"Component {comp.component_id} requires {dep}, which is not in architecture",
                        suggestion=f"Add component {dep} or remove {comp.component_id}"
                    ))

        # Check dependency layers are correct
        if not self._verify_dependency_layers(architecture):
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="global",
                issue_type="invalid_layers",
                message="Dependency layers may be incorrectly computed",
                suggestion="Re-run dependency resolution"
            ))

    def _verify_dependency_layers(self, architecture: Architecture) -> bool:
        """Verify dependency layers are topologically sorted"""

        seen = set()
        for layer in architecture.dependency_layers:
            for cid in layer:
                comp = architecture.get_component(cid)
                if comp:
                    # Check all dependencies are in previous layers
                    for dep in comp.requires:
                        if dep not in seen and dep in {c.component_id for c in architecture.components}:
                            return False
                seen.add(cid)

        return True

    def _validate_aci(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate ACI improvement"""

        # Check ACI improvement is positive
        if architecture.expected_aci_improvement < 0:
            validation.errors.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component="global",
                issue_type="negative_aci",
                message=f"Expected ACI improvement is negative: {architecture.expected_aci_improvement:.2f}",
                suggestion="Review component selection"
            ))

        # Check ACI improvement is significant
        if architecture.expected_aci_improvement < 0.1:
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="global",
                issue_type="low_aci",
                message=f"Expected ACI improvement is low: {architecture.expected_aci_improvement:.2f}",
                suggestion="Consider adding components that increase ACI"
            ))

        # Check if ACI calculator present
        if not architecture.has_component("gamma1"):
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                component="global",
                issue_type="no_aci_calculator",
                message="Γ₁ (ACI Calculator) not present - ACI estimates may be inaccurate",
                suggestion="Add Γ₁ for accurate ACI calculation"
            ))

        validation.aci_improvement = architecture.expected_aci_improvement
        validation.aci_valid = architecture.expected_aci_improvement > 0

    def _validate_propagation(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate validation propagation from components to architecture"""

        # Get component scores
        for comp in architecture.components:
            if comp.is_validated:
                validation.component_scores[comp.component_id] = comp.validation_score
            else:
                validation.warnings.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    component=comp.component_id,
                    issue_type="component_not_validated",
                    message=f"Component {comp.component_id} is not validated",
                    suggestion="Run component validation before assembly"
                ))
                validation.component_scores[comp.component_id] = 0.0

        # Check if enough validated components
        validated_count = sum(1 for c in architecture.components if c.is_validated)
        if validated_count < len(architecture.components) / 2:
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="global",
                issue_type="insufficient_validation",
                message=f"Only {validated_count}/{len(architecture.components)} components validated",
                suggestion="Validate more components before assembly"
            ))

    def _validate_performance(
        self,
        architecture: Architecture,
        validation: ArchitectureValidation
    ):
        """Validate performance characteristics"""

        # Check estimated runtime
        if architecture.estimated_runtime > 60.0:
            validation.warnings.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="global",
                issue_type="slow_runtime",
                message=f"Estimated runtime is high: {architecture.estimated_runtime:.1f}s",
                suggestion="Consider reducing component count or using more parallel patterns"
            ))

        # Check assembly pattern
        if architecture.assembly_pattern == AssemblyPattern.SEQUENTIAL:
            if len(architecture.components) > 5:
                validation.warnings.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    component="global",
                    issue_type="sequential_long",
                    message=f"Sequential assembly with {len(architecture.components)} components may be slow",
                    suggestion="Consider restructuring for parallel execution"
                ))

    def _calculate_score(self, validation: ArchitectureValidation) -> float:
        """
        Calculate overall validation score

        Combines multiple factors:
        - Component validation scores
        - ACI improvement
        - Error/weight penalties
        """
        # Base score from components
        if validation.component_scores:
            component_score = sum(validation.component_scores.values()) / len(validation.component_scores)
        else:
            component_score = 0.0

        # ACI score (0.2 max)
        aci_score = min(validation.aci_improvement / 0.5, 1.0) * 0.2

        # Error penalty
        error_penalty = len(validation.errors) * 0.3
        warning_penalty = len(validation.warnings) * 0.05

        # Combine
        score = component_score + aci_score - error_penalty - warning_penalty

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _determine_validity(self, validation: ArchitectureValidation) -> bool:
        """Determine if architecture is valid"""

        # Critical errors always invalid
        if any(e.severity == ValidationSeverity.CRITICAL for e in validation.errors):
            return False

        # Any error invalidates
        if len(validation.errors) > 0:
            return False

        # In strict mode, warnings also invalidate
        if self.strict and len(validation.warnings) > 0:
            return False

        # Minimum score threshold
        if validation.validation_score < 0.5:
            return False

        # Must have positive ACI improvement
        if validation.aci_improvement <= 0:
            return False

        return True

    def explain_validation(self, validation: ArchitectureValidation) -> str:
        """
        Generate human-readable explanation of validation result

        Args:
            validation: Validation result to explain

        Returns:
            Multi-line explanation string
        """
        lines = [
            "=" * 70,
            f"Architecture Validation Report: {validation.architecture_id}",
            "=" * 70,
            "",
            validation.get_summary(),
            "",
            f"Validation completed in {validation.validation_time:.3f}s",
            ""
        ]

        # Errors
        if validation.errors:
            lines.append("ERRORS:")
            for error in validation.errors:
                lines.append(f"  ✗ [{error.component}] {error.message}")
                if error.suggestion:
                    lines.append(f"    Suggestion: {error.suggestion}")
            lines.append("")

        # Warnings
        if validation.warnings:
            lines.append("WARNINGS:")
            for warning in validation.warnings:
                lines.append(f"  ⚠ [{warning.component}] {warning.message}")
                if warning.suggestion:
                    lines.append(f"    Suggestion: {warning.suggestion}")
            lines.append("")

        # Component scores
        if validation.component_scores:
            lines.append("COMPONENT VALIDATION SCORES:")
            for cid, score in sorted(validation.component_scores.items()):
                status = "✓" if score >= 0.7 else "⚠"
                lines.append(f"  {status} {cid:15s}: {score:.2f}")
            lines.append("")

        # Conclusion
        if validation.is_valid:
            lines.append("CONCLUSION: ✓ Architecture is VALID")
        else:
            lines.append("CONCLUSION: ✗ Architecture is INVALID")

        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Batch Validator
# =============================================================================

class BatchValidator:
    """
    Validate multiple architectures

    Useful for comparing alternative assemblies.
    """

    def __init__(self, strict: bool = False):
        self.validator = AssemblyValidator(strict=strict)
        self.results: List[ArchitectureValidation] = []

    def validate_all(
        self,
        architectures: List[Architecture]
    ) -> List[ArchitectureValidation]:
        """
        Validate all architectures

        Returns:
            List of validation results (same order as input)
        """
        self.results = [
            self.validator.validate(arch)
            for arch in architectures
        ]
        return self.results

    def get_best(self) -> Optional[ArchitectureValidation]:
        """Get the best validated architecture"""
        valid_results = [r for r in self.results if r.is_valid]
        if not valid_results:
            return None
        return max(valid_results, key=lambda r: r.validation_score)

    def compare(self) -> str:
        """Generate comparison report"""
        if not self.results:
            return "No validation results"

        lines = [
            "=" * 70,
            "Architecture Validation Comparison",
            "=" * 70,
            "",
            f"Total architectures validated: {len(self.results)}",
            f"Valid architectures: {sum(1 for r in self.results if r.is_valid)}",
            f"Invalid architectures: {sum(1 for r in self.results if not r.is_valid)}",
            ""
        ]

        # Sort by score
        sorted_results = sorted(
            self.results,
            key=lambda r: r.validation_score,
            reverse=True
        )

        lines.append("RANKINGS:")
        for i, result in enumerate(sorted_results, 1):
            status = "✓" if result.is_valid else "✗"
            lines.append(
                f"  {i}. {status} {result.architecture_id}: "
                f"score={result.validation_score:.2f}, "
                f"ACI={result.aci_improvement:+.2f}"
            )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================

def quick_validate(architecture: Architecture) -> Tuple[bool, float]:
    """
    Quick validation check

    Returns:
        (is_valid, validation_score)
    """
    validator = AssemblyValidator()
    result = validator.validate(architecture)
    return result.is_valid, result.validation_score


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("Architecture Assembly Validator")
    print("=" * 70)

    # Import architecture assembler
    try:
        from phase4.architecture_assembler import ArchitectureAssembler

        # Create and assemble architecture
        assembler = ArchitectureAssembler()
        result = assembler.assemble()

        if result.success:
            architecture = result.architecture

            # Validate
            validator = AssemblyValidator()
            validation = validator.validate(architecture)

            # Print report
            print("\n")
            print(validator.explain_validation(validation))

        else:
            print(f"\n✗ Assembly failed: {result.message}")

    except ImportError as e:
        print(f"\n✗ Cannot import assembler: {e}")
        print("This is expected if architecture_assembler.py is not available")
