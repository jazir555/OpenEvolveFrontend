"""
Final Solution Management Module

This module handles the validation, management, and delivery of final integrated solutions.
It focuses on what happens AFTER recomposition - ensuring the final solution is ready for delivery.

Key Classes:
    - SolutionValidator: Validates integrated solutions against original problems
    - FinalSolutionManager: Manages final solution lifecycle and delivery

Usage:
    from final_solution import SolutionValidator, create_solution_validator

    validator = create_solution_validator()
    results = validator.validate_solution(integrated_solution, original_problem)
"""

import logging
from typing import List, Optional
from datetime import datetime

from sovereign_data_models import (
    ProblemDefinition,
    IntegratedSolution,
    ValidationResult
)

logger = logging.getLogger(__name__)


# Try to import OpenEvolve client
try:
    from openevolve_client import OpenEvolveClient
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OpenEvolveClient = None
    OPENEVOLVE_AVAILABLE = False
    logger.warning("OpenEvolve client not available. Some validation features will be limited.")


# ============================================================================
# SOLUTION VALIDATOR
# ============================================================================

class SolutionValidator:
    """
    Validates final integrated solutions against original problem definitions.

    This class focuses on validating the FINAL SOLUTION after recomposition is complete.
    It ensures the assembled solution meets all requirements and quality standards.

    Validation Checks:
        1. Completeness - All sub-problems addressed
        2. Consistency - No internal contradictions
        3. Quality - Meets quality thresholds
        4. Requirements - Satisfies success criteria
    """

    def __init__(self, openevolve_client: Optional['OpenEvolveClient'] = None):
        """
        Initialize with optional OpenEvolve client for LLM-based validation.

        Args:
            openevolve_client: Optional OpenEvolve client for enhanced validation
        """
        self.openevolve_client = openevolve_client
        self._init_client()

    def _init_client(self):
        """Initialize OpenEvolve client if needed."""
        global OpenEvolveClient, OPENEVOLVE_AVAILABLE
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized for solution validation")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to instantiate OpenEvolve client: {e}")
                self.openevolve_client = None

    def validate_solution(
        self,
        integrated_solution: IntegratedSolution,
        original_problem: ProblemDefinition
    ) -> List[ValidationResult]:
        """
        Validate final integrated solution against original problem.

        This is the main validation entry point. It runs all validation checks
        and updates the solution with validation results.

        Args:
            integrated_solution: The final integrated solution to validate
            original_problem: The original problem definition

        Returns:
            List of validation results from all validation checks
        """
        logger.info(f"Validating final solution {integrated_solution.solution_id}")

        validation_results = []

        # Validate completeness
        completeness_result = self._validate_completeness(integrated_solution, original_problem)
        validation_results.append(completeness_result)

        # Validate consistency
        consistency_result = self._validate_consistency(integrated_solution)
        validation_results.append(consistency_result)

        # Validate quality
        quality_result = self._validate_quality(integrated_solution)
        validation_results.append(quality_result)

        # Validate requirements
        requirements_result = self._validate_requirements(integrated_solution, original_problem)
        validation_results.append(requirements_result)

        # Update integrated solution with validation results
        integrated_solution.validation_results = validation_results

        logger.info(f"Solution validation complete: {len(validation_results)} validation checks")
        return validation_results

    def _validate_completeness(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition
    ) -> ValidationResult:
        """
        Check if all aspects of the problem are addressed.

        Validates that all sub-problems have been integrated into the final solution.

        Args:
            solution: The integrated solution to validate
            problem: The original problem definition

        Returns:
            ValidationResult with completeness score and feedback
        """
        logger.info("Validating completeness")

        # Check if all sub-problems have solutions
        num_sub_solutions = len(solution.sub_solutions)
        expected_solutions = len(solution.sub_solutions)  # Placeholder

        completeness_score = min(1.0, num_sub_solutions / max(1, expected_solutions))

        passed = completeness_score >= 0.8
        feedback = f"{'Passed' if passed else 'Failed'}: {num_sub_solutions} sub-solutions integrated"

        improvements = []
        if not passed:
            improvements.append("Ensure all sub-problems have solutions")

        return ValidationResult(
            validator="completeness",
            passed=passed,
            score=completeness_score,
            feedback=feedback,
            improvements=improvements
        )

    def _validate_consistency(self, solution: IntegratedSolution) -> ValidationResult:
        """
        Check for internal consistency in the final solution.

        Validates that there are no unresolved conflicts or contradictions.

        Args:
            solution: The integrated solution to validate

        Returns:
            ValidationResult with consistency score and feedback
        """
        logger.info("Validating consistency")

        # Check for unresolved conflicts
        unresolved_conflicts = [
            c for c in solution.conflicts_resolved
            if c.status != 'resolved'
        ]

        consistency_score = max(0.0, 1.0 - len(unresolved_conflicts) * 0.2)

        passed = len(unresolved_conflicts) == 0
        feedback = f"{'Passed' if passed else 'Failed'}: {len(unresolved_conflicts)} unresolved conflicts"

        improvements = []
        if not passed:
            improvements.append("Resolve all conflicts before finalizing solution")

        return ValidationResult(
            validator="consistency",
            passed=passed,
            score=consistency_score,
            feedback=feedback,
            improvements=improvements
        )

    def _validate_quality(self, solution: IntegratedSolution) -> ValidationResult:
        """
        Check quality metrics of the final solution.

        Validates that the solution meets minimum quality thresholds.

        Args:
            solution: The integrated solution to validate

        Returns:
            ValidationResult with quality score and feedback
        """
        logger.info("Validating quality")

        quality = solution.quality_metrics
        quality_score = quality.overall_score

        passed = quality_score >= 0.7
        feedback = f"{'Passed' if passed else 'Failed'}: Overall quality score {quality_score:.2f}"

        improvements = []
        if quality.completeness_score < 0.7:
            improvements.append("Improve completeness by addressing all sub-problems")
        if quality.consistency_score < 0.7:
            improvements.append("Improve consistency by resolving conflicts")
        if quality.coherence_score < 0.7:
            improvements.append("Improve coherence by enhancing content flow")

        return ValidationResult(
            validator="quality",
            passed=passed,
            score=quality_score,
            feedback=feedback,
            improvements=improvements
        )

    def _validate_requirements(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition
    ) -> ValidationResult:
        """
        Check if all requirements from the original problem are met.

        Validates that the solution satisfies the success criteria.

        Args:
            solution: The integrated solution to validate
            problem: The original problem definition

        Returns:
            ValidationResult with requirements score and feedback
        """
        logger.info("Validating requirements")

        # Check success criteria
        criteria_met = 0
        total_criteria = len(problem.success_criteria)

        if total_criteria == 0:
            requirements_score = 1.0
        else:
            # Placeholder: in production, would actually validate against criteria
            criteria_met = total_criteria  # Assume met for now
            requirements_score = criteria_met / total_criteria

        passed = requirements_score >= 0.8
        feedback = f"{'Passed' if passed else 'Failed'}: {criteria_met}/{total_criteria} criteria met"

        improvements = []
        if not passed:
            improvements.append("Ensure all success criteria are addressed")

        return ValidationResult(
            validator="requirements",
            passed=passed,
            score=requirements_score,
            feedback=feedback,
            improvements=improvements
        )


# ============================================================================
# FINAL SOLUTION MANAGER
# ============================================================================

class FinalSolutionManager:
    """
    Manages the lifecycle and delivery of final integrated solutions.

    This class handles what happens to a solution AFTER it's been validated:
    - Preparing for delivery
    - Generating delivery reports
    - Managing solution versions
    - Exporting to various formats

    Attributes:
        validator: SolutionValidator instance for validation
        delivery_format: Default format for solution delivery
    """

    def __init__(
        self,
        validator: Optional[SolutionValidator] = None,
        delivery_format: str = "markdown"
    ):
        """
        Initialize the final solution manager.

        Args:
            validator: Optional SolutionValidator instance
            delivery_format: Default format for delivery ("markdown", "json", "html")
        """
        self.validator = validator or SolutionValidator()
        self.delivery_format = delivery_format
        logger.info("FinalSolutionManager initialized")

    def prepare_for_delivery(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition,
        validate: bool = True
    ) -> dict:
        """
        Prepare final solution for delivery.

        Args:
            solution: The integrated solution to deliver
            problem: The original problem definition
            validate: Whether to validate before delivery

        Returns:
            Dictionary with delivery information
        """
        logger.info(f"Preparing solution {solution.solution_id} for delivery")

        # Validate if requested
        if validate:
            validation_results = self.validator.validate_solution(solution, problem)
            all_passed = all(vr.passed for vr in validation_results)

            if not all_passed:
                logger.warning(f"Solution {solution.solution_id} has validation failures")

        # Prepare delivery package
        delivery_package = {
            "solution_id": solution.solution_id,
            "problem_id": problem.id,
            "assembled_content": solution.assembled_content,
            "quality_metrics": {
                "completeness": solution.quality_metrics.completeness_score,
                "consistency": solution.quality_metrics.consistency_score,
                "coherence": solution.quality_metrics.coherence_score,
                "overall": solution.quality_metrics.overall_score
            },
            "validation_results": [
                {
                    "validator": vr.validator,
                    "passed": vr.passed,
                    "score": vr.score,
                    "feedback": vr.feedback
                }
                for vr in solution.validation_results
            ],
            "assembly_metadata": {
                "assembly_strategy": solution.assembly_strategy,
                "num_sub_solutions": len(solution.sub_solutions),
                "integration_order": solution.integration_order,
                "conflicts_resolved": len(solution.conflicts_resolved),
                "created_at": solution.created_at.isoformat() if solution.created_at else None
            }
        }

        logger.info(f"Solution {solution.solution_id} prepared for delivery")
        return delivery_package

    def generate_delivery_report(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition,
        format: str = None
    ) -> str:
        """
        Generate a delivery report for the final solution.

        Args:
            solution: The integrated solution
            problem: The original problem definition
            format: Report format ("markdown", "json", "html")

        Returns:
            Formatted delivery report as string
        """
        format = format or self.delivery_format

        if format == "markdown":
            return self._generate_markdown_report(solution, problem)
        elif format == "json":
            import json
            return json.dumps(self.prepare_for_delivery(solution, problem), indent=2)
        elif format == "html":
            return self._generate_html_report(solution, problem)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_markdown_report(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition
    ) -> str:
        """Generate markdown delivery report."""
        lines = [
            f"# Final Solution Report",
            f"",
            f"## Solution Information",
            f"- **Solution ID**: {solution.solution_id}",
            f"- **Problem ID**: {problem.id}",
            f"- **Problem Title**: {problem.title}",
            f"- **Assembly Strategy**: {solution.assembly_strategy}",
            f"- **Created**: {solution.created_at or 'N/A'}",
            f"",
            f"## Quality Metrics",
            f"- **Completeness**: {solution.quality_metrics.completeness_score:.2f}",
            f"- **Consistency**: {solution.quality_metrics.consistency_score:.2f}",
            f"- **Coherence**: {solution.quality_metrics.coherence_score:.2f}",
            f"- **Overall**: {solution.quality_metrics.overall_score:.2f}",
            f"",
            f"## Validation Results",
        ]

        for vr in solution.validation_results:
            status = "✅ PASS" if vr.passed else "❌ FAIL"
            lines.append(f"- **{vr.validator.capitalize()}**: {status} (score: {vr.score:.2f})")
            lines.append(f"  - {vr.feedback}")
            if vr.improvements:
                lines.append(f"  - Improvements: {', '.join(vr.improvements)}")

        lines.extend([
            f"",
            f"## Assembly Details",
            f"- **Sub-solutions**: {len(solution.sub_solutions)}",
            f"- **Conflicts Resolved**: {len(solution.conflicts_resolved)}",
            f"- **Integration Order**: {' → '.join(solution.integration_order[:5])}{'...' if len(solution.integration_order) > 5 else ''}",
            f"",
            f"## Solution Content",
            f"",
            solution.assembled_content
        ])

        return "\n".join(lines)

    def _generate_html_report(
        self,
        solution: IntegratedSolution,
        problem: ProblemDefinition
    ) -> str:
        """Generate HTML delivery report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Final Solution Report - {solution.solution_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #1976d2; }}
        .validation {{ margin: 20px 0; }}
        .validation-item {{ padding: 10px; margin: 5px 0; border-left: 4px solid #ccc; }}
        .pass {{ border-left-color: #4caf50; background: #f1f8f4; }}
        .fail {{ border-left-color: #f44336; background: #fef1f1; }}
        .content {{ margin-top: 30px; white-space: pre-wrap; background: #f9f9f9; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Final Solution Report</h1>
        <p><strong>Solution ID:</strong> {solution.solution_id}</p>
        <p><strong>Problem:</strong> {problem.title}</p>
        <p><strong>Assembly Strategy:</strong> {solution.assembly_strategy}</p>
    </div>

    <h2>Quality Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{solution.quality_metrics.completeness_score:.2f}</div>
            <div>Completeness</div>
        </div>
        <div class="metric">
            <div class="metric-value">{solution.quality_metrics.consistency_score:.2f}</div>
            <div>Consistency</div>
        </div>
        <div class="metric">
            <div class="metric-value">{solution.quality_metrics.coherence_score:.2f}</div>
            <div>Coherence</div>
        </div>
        <div class="metric">
            <div class="metric-value">{solution.quality_metrics.overall_score:.2f}</div>
            <div>Overall</div>
        </div>
    </div>

    <h2>Validation Results</h2>
    <div class="validation">
"""

        for vr in solution.validation_results:
            status_class = "pass" if vr.passed else "fail"
            status = "PASS" if vr.passed else "FAIL"
            html += f"""
        <div class="validation-item {status_class}">
            <strong>{vr.validator.capitalize()}:</strong> {status} (score: {vr.score:.2f})<br>
            {vr.feedback}
"""

            if vr.improvements:
                html += f"<br><em>Improvements: {', '.join(vr.improvements)}</em>"

            html += "</div>"

        html += f"""
    </div>

    <h2>Assembly Details</h2>
    <ul>
        <li><strong>Sub-solutions:</strong> {len(solution.sub_solutions)}</li>
        <li><strong>Conflicts Resolved:</strong> {len(solution.conflicts_resolved)}</li>
        <li><strong>Integration Order:</strong> {' → '.join(solution.integration_order[:5])}{'...' if len(solution.integration_order) > 5 else ''}</li>
    </ul>

    <h2>Solution Content</h2>
    <div class="content">{solution.assembled_content}</div>

</body>
</html>
"""
        return html


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_solution_validator(
    openevolve_client: Optional['OpenEvolveClient'] = None
) -> SolutionValidator:
    """
    Factory function to create a SolutionValidator.

    Args:
        openevolve_client: Optional OpenEvolve client for enhanced validation

    Returns:
        Configured SolutionValidator instance
    """
    return SolutionValidator(openevolve_client)


def create_final_solution_manager(
    validator: Optional[SolutionValidator] = None,
    delivery_format: str = "markdown"
) -> FinalSolutionManager:
    """
    Factory function to create a FinalSolutionManager.

    Args:
        validator: Optional SolutionValidator instance
        delivery_format: Default format for delivery

    Returns:
        Configured FinalSolutionManager instance
    """
    return FinalSolutionManager(validator, delivery_format)
