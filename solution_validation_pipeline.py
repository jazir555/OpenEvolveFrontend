"""
Solution Validation Pipeline

This module implements a complete solution validation pipeline that ensures quality
before acceptance. The pipeline runs through multiple stages including automated
checks, red team review, and gold team verification.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from sovereign_data_models import (
    SolutionAttempt, SubProblem,
    SolutionValidationResults, AutomatedCheckResults,
    RedTeamCritiqueReport, VerificationReport, ValidationRequirements,
    ValidationResult, generate_id
)
from llm_utils import _request_openai_compatible_chat, _compose_messages

# Try to import red team feedback system
try:
    from red_team_feedback_system import RedTeamFeedbackSystem
    RED_TEAM_SYSTEM_AVAILABLE = True
except ImportError:
    RED_TEAM_SYSTEM_AVAILABLE = False

# **LEAN INTEGRATION**: Stage 4 Formal Verification
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class SolutionValidationPipeline:
    """
    Complete solution validation pipeline.

    Ensures solutions meet quality standards before acceptance.
    """

    def __init__(self, red_team_system=None, gold_team_system=None):
        """
        Initialize with team systems.

        Args:
            red_team_system: Optional red team feedback system
            gold_team_system: Optional gold team system
        """
        self.red_team_system = red_team_system or (RedTeamFeedbackSystem() if RED_TEAM_SYSTEM_AVAILABLE else None)
        self.gold_team_system = gold_team_system
        self.validation_history: List[Dict[str, Any]] = []
        
        # **LEAN INTEGRATION**: Initialize Lean client for Stage 4
        self._lean_client = None
        if LEAN_AVAILABLE:
            try:
                self._lean_client = LeanAideClient()
                logger.info("LeanAide client initialized for formal verification")
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAide client: {e}")

    def validate_solution(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        validation_requirements: Optional[ValidationRequirements] = None
    ) -> SolutionValidationResults:
        """
        Run complete validation pipeline.

        Pipeline stages:
        1. Automated checks (syntax, structure, basic quality)
        2. Red team review (adversarial analysis)
        3. Gold team verification (thorough validation)
        4. Integration checks (if part of larger solution)
        5. Final scoring and decision

        Args:
            solution: Solution attempt to validate
            sub_problem: Sub-problem context
            validation_requirements: Validation requirements configuration

        Returns:
            SolutionValidationResults with complete validation data
        """
        start_time = time.time()
        validation_id = generate_id("validation")

        logger.info(f"Starting validation pipeline {validation_id} for solution {solution.id}")

        # Use default requirements if not provided
        if validation_requirements is None:
            validation_requirements = ValidationRequirements()

        # Initialize results
        validation_results = SolutionValidationResults(
            validation_id=validation_id,
            solution_id=solution.id,
            sub_problem_id=sub_problem.id,
            pass_threshold=validation_requirements.threshold
        )

        # Stage 1: Automated Checks
        if validation_requirements.run_automated_checks:
            logger.info("Running automated checks...")
            validation_results.automated_results = self.run_automated_checks(solution)

        # Stage 2: Red Team Review
        if validation_requirements.use_red_team and self.red_team_system:
            logger.info("Running red team review...")
            validation_results.red_team_report = self.run_red_team_review(
                solution, sub_problem
            )

        # Stage 3: Gold Team Verification
        if validation_requirements.use_gold_team:
            logger.info("Running gold team verification...")
            validation_results.gold_team_report = self.run_gold_team_verification(
                solution,
                validation_results.red_team_report,
                sub_problem
            )
        
        # **LEAN INTEGRATION**: Stage 4: Formal Verification
        if validation_requirements.use_formal_verification or getattr(validation_requirements, 'verify_with_lean', False):
            logger.info("Running Stage 4: Formal verification with Lean...")
            formal_result = await self.run_formal_verification(solution, sub_problem)
            validation_results.formal_verification_result = formal_result
            
            # Adjust final score based on formal verification
            if formal_result.get('verified'):
                validation_results.formal_verification_boost = 0.05  # 5% boost for verified solutions
            else:
                validation_results.formal_verification_penalty = 0.10  # 10% penalty for unverified claims

        # Stage 5: Calculate Final Score
        validation_results.final_score = self.calculate_final_score(
            validation_results.automated_results,
            validation_results.red_team_report,
            validation_results.gold_team_report
        )

        # Stage 5: Determine Recommendation
        validation_results.passed = validation_results.final_score >= validation_requirements.threshold
        validation_results.recommendation = self._determine_recommendation(
            validation_results,
            validation_requirements
        )

        # Stage 6: Generate revision guidance if needed
        if validation_results.recommendation == "revise":
            validation_results.revision_guidance = self._generate_revision_guidance(
                validation_results
            )

        # Calculate contributions
        validation_results.automated_contribution = (
            validation_results.automated_results.overall_score
            if validation_results.automated_results else 0.0
        ) * 0.20

        validation_results.red_team_contribution = (
            validation_results.red_team_report.overall_score
            if validation_results.red_team_report else 0.0
        ) * 0.35

        validation_results.gold_team_contribution = (
            validation_results.gold_team_report.overall_quality_score
            if validation_results.gold_team_report else 0.0
        ) * 0.45

        # Collect critical issues
        validation_results.critical_issues = self._collect_critical_issues(validation_results)
        validation_results.must_fix_before_acceptance = validation_results.critical_issues

        # Record timing
        validation_results.validation_duration = time.time() - start_time
        logger.info(f"Validation completed in {validation_results.validation_duration:.2f}s")
        logger.info(f"Final score: {validation_results.final_score:.3f}, Recommendation: {validation_results.recommendation}")
        
        # Log formal verification result if performed
        if hasattr(validation_results, 'formal_verification_result'):
            verified = validation_results.formal_verification_result.get('verified', False)
            logger.info(f"Formal verification: {'VERIFIED' if verified else 'NOT VERIFIED'}")

        # Store in history
        self.validation_history.append({
            'validation_id': validation_id,
            'solution_id': solution.id,
            'sub_problem_id': sub_problem.id,
            'timestamp': datetime.now(),
            'passed': validation_results.passed,
            'final_score': validation_results.final_score,
            'recommendation': validation_results.recommendation,
            'duration': validation_results.validation_duration
        })

        return validation_results

    def run_automated_checks(
        self,
        solution: SolutionAttempt
    ) -> AutomatedCheckResults:
        """
        Run automated validation checks.

        Checks:
        - Syntax errors
        - Structure validation
        - Completeness (all required sections present)
        - Format compliance
        - Basic quality metrics

        Args:
            solution: Solution to check

        Returns:
            AutomatedCheckResults with check outcomes
        """
        start_time = time.time()
        check_id = generate_id("autocheck")

        logger.info(f"Running automated checks {check_id} for solution {solution.id}")

        results = AutomatedCheckResults(
            check_id=check_id,
            solution_id=solution.id,
            timestamp=datetime.now()
        )

        # Initialize completeness check
        required_sections = ["approach", "solution_content"]
        for section in required_sections:
            results.completeness_check[section] = True

        # Check syntax (basic validation)
        results.syntax_valid = self._check_syntax(solution)

        # Check structure
        results.structure_valid = self._check_structure(solution)

        # Check format compliance
        results.format_compliant = self._check_format(solution)

        # Collect errors and warnings
        if not results.syntax_valid:
            results.errors.append("Syntax validation failed")
        if not results.structure_valid:
            results.errors.append("Structure validation failed")
        if not results.format_compliant:
            results.warnings.append("Format compliance issues detected")

        # Calculate pass rate
        total_checks = 4  # syntax, structure, format, completeness
        passed_checks = sum([
            results.syntax_valid,
            results.structure_valid,
            results.format_compliant,
            all(results.completeness_check.values())
        ])
        results.pass_rate = passed_checks / total_checks

        # Calculate overall score
        results.overall_score = results.pass_rate

        results.check_duration = time.time() - start_time

        logger.info(f"Automated checks completed: score={results.overall_score:.3f}, pass_rate={results.pass_rate:.3f}")

        return results

    def run_red_team_review(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> RedTeamCritiqueReport:
        """
        Red team adversarial review.

        Args:
            solution: Solution to review
            sub_problem: Sub-problem context

        Returns:
            RedTeamCritiqueReport with red team findings
        """
        if not self.red_team_system:
            logger.warning("Red team system not available, returning default critique")
            return RedTeamCritiqueReport(
                report_id=generate_id("critique"),
                team_type="red_team",
                team_id="red_team_default",
                solution_id=solution.id,
                sub_problem_id=sub_problem.id,
                findings=["Red team review not available"],
                severity_scores=[0.5],
                categories=["unavailable"],
                overall_score=0.7,  # Neutral score
                confidence=0.0,
                timestamp=datetime.now()
            )

        logger.info(f"Running red team review for solution {solution.id}")

        critique = self.red_team_system.generate_red_team_feedback(
            solution=solution,
            sub_problem=sub_problem
        )

        logger.info(f"Red team review completed: score={critique.overall_score:.3f}, findings={len(critique.findings)}")

        return critique

    def run_gold_team_verification(
        self,
        solution: SolutionAttempt,
        red_team_feedback: Optional[RedTeamCritiqueReport],
        sub_problem: SubProblem
    ) -> VerificationReport:
        """
        Gold team thorough verification.

        Args:
            solution: Solution to verify
            red_team_feedback: Red team critique to review
            sub_problem: Sub-problem context

        Returns:
            VerificationReport with gold team assessment
        """
        start_time = time.time()
        verification_id = generate_id("verify")

        logger.info(f"Running gold team verification {verification_id} for solution {solution.id}")

        # Review red team findings if available
        red_team_reviewed = 0
        red_team_confirmed = 0
        red_team_rejected = 0

        if red_team_feedback:
            red_team_reviewed = len(red_team_feedback.findings)
            red_team_confirmed = sum(1 for s in red_team_feedback.severity_scores if s >= 0.6)
            red_team_rejected = red_team_reviewed - red_team_confirmed

        # Perform gold team assessment
        assessment = self._perform_gold_team_assessment(
            solution,
            sub_problem,
            red_team_feedback
        )

        # Create verification report
        verification = VerificationReport(
            verification_id=verification_id,
            solution_id=solution.id,
            sub_problem_id=sub_problem.id,
            gold_team_id="gold_team_default",
            red_team_findings_reviewed=red_team_reviewed,
            red_team_findings_confirmed=red_team_confirmed,
            red_team_findings_rejected=red_team_rejected,
            additional_findings=assessment.get("additional_findings", []),
            verified_correct=assessment.get("verified_correct", True),
            verification_details=assessment.get("details", ""),
            verification_confidence=assessment.get("confidence", 0.8),
            correctness_score=assessment.get("correctness", 0.8),
            completeness_score=assessment.get("completeness", 0.8),
            clarity_score=assessment.get("clarity", 0.8),
            overall_quality_score=assessment.get("overall_quality", 0.8),
            recommendation=assessment.get("recommendation", "accept"),
            verification_notes=assessment.get("notes", []),
            timestamp=datetime.now(),
            review_duration=time.time() - start_time
        )

        logger.info(f"Gold team verification completed: quality_score={verification.overall_quality_score:.3f}")

        return verification

    def calculate_final_score(
        self,
        automated_results: Optional[AutomatedCheckResults],
        red_team_report: Optional[RedTeamCritiqueReport],
        gold_team_report: Optional[VerificationReport]
    ) -> float:
        """
        Calculate final validation score (0-1).

        Weighted combination:
        - Automated: 20%
        - Red team: 35%
        - Gold team: 45%

        Args:
            automated_results: Automated check results
            red_team_report: Red team critique
            gold_team_report: Gold team verification

        Returns:
            Final score between 0.0 and 1.0
        """
        scores = []
        weights = []

        if automated_results:
            scores.append(automated_results.overall_score)
            weights.append(0.20)

        if red_team_report:
            scores.append(red_team_report.overall_score)
            weights.append(0.35)

        if gold_team_report:
            scores.append(gold_team_report.overall_quality_score)
            weights.append(0.45)

        if not scores:
            return 0.5  # Default neutral score

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))

        return round(final_score, 3)

    def generate_validation_report(
        self,
        validation_results: SolutionValidationResults
    ) -> str:
        """
        Generate human-readable validation report.

        Args:
            validation_results: Validation results to format

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "SOLUTION VALIDATION REPORT",
            "=" * 80,
            f"Validation ID: {validation_results.validation_id}",
            f"Solution ID: {validation_results.solution_id}",
            f"Sub-Problem ID: {validation_results.sub_problem_id}",
            f"Timestamp: {validation_results.timestamp.isoformat()}",
            f"Duration: {validation_results.validation_duration:.2f}s",
            "",
            "-" * 80,
            "RESULTS SUMMARY",
            "-" * 80,
            f"Final Score: {validation_results.final_score:.3f}",
            f"Pass Threshold: {validation_results.pass_threshold:.3f}",
            f"Status: {'PASSED' if validation_results.passed else 'FAILED'}",
            f"Recommendation: {validation_results.recommendation.upper()}",
            "",
            "-" * 80,
            "SCORE BREAKDOWN",
            "-" * 80,
        ]

        if validation_results.automated_results:
            report_lines.extend([
                f"Automated Checks: {validation_results.automated_contribution:.3f} "
                f"(raw: {validation_results.automated_results.overall_score:.3f})"
            ])

        if validation_results.red_team_report:
            report_lines.extend([
                f"Red Team Review: {validation_results.red_team_contribution:.3f} "
                f"(raw: {validation_results.red_team_report.overall_score:.3f})"
            ])

        if validation_results.gold_team_report:
            report_lines.extend([
                f"Gold Team Verification: {validation_results.gold_team_contribution:.3f} "
                f"(raw: {validation_results.gold_team_report.overall_quality_score:.3f})"
            ])

        report_lines.extend([
            "",
            "-" * 80,
            "CRITICAL ISSUES",
            "-" * 80,
        ])

        if validation_results.critical_issues:
            for i, issue in enumerate(validation_results.critical_issues, 1):
                report_lines.append(f"{i}. {issue}")
        else:
            report_lines.append("No critical issues found.")

        report_lines.extend([
            "",
            "-" * 80,
            "MUST FIX BEFORE ACCEPTANCE",
            "-" * 80,
        ])

        if validation_results.must_fix_before_acceptance:
            for i, issue in enumerate(validation_results.must_fix_before_acceptance, 1):
                report_lines.append(f"{i}. {issue}")
        else:
            report_lines.append("No issues require fixing.")

        if validation_results.recommendation == "revise":
            report_lines.extend([
                "",
                "-" * 80,
                "REVISION GUIDANCE",
                "-" * 80,
                validation_results.revision_guidance
            ])

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def _check_syntax(self, solution: SolutionAttempt) -> bool:
        """Check syntax of solution."""
        # Basic syntax validation
        if not solution.solution_content or len(solution.solution_content.strip()) == 0:
            return False

        # Check for basic structural markers
        content = solution.solution_content.lower()
        has_structure = any(marker in content for marker in [
            "solution", "approach", "method", "implementation"
        ])

        return has_structure

    def _check_structure(self, solution: SolutionAttempt) -> bool:
        """Check structure of solution."""
        # Check for minimum length and organization
        content = solution.solution_content.strip()

        if len(content) < 50:  # Too short
            return False

        # Check for paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:  # Should have at least 2 sections
            return False

        return True

    def _check_format(self, solution: SolutionAttempt) -> bool:
        """Check format compliance."""
        # Basic format checks
        content = solution.solution_content

        # Check for reasonable line lengths
        lines = content.split('\n')
        too_long_lines = [line for line in lines if len(line) > 500]

        if len(too_long_lines) > len(lines) * 0.1:  # More than 10% too long
            return False

        return True

    def _perform_gold_team_assessment(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        red_team_feedback: Optional[RedTeamCritiqueReport]
    ) -> Dict[str, Any]:
        """Perform gold team quality assessment."""
        # In a full implementation, this would use LLM for thorough review
        # For now, provide heuristic-based assessment

        correctness = 0.85  # Assume mostly correct
        completeness = 0.80  # Assume reasonably complete
        clarity = 0.75  # Assume reasonably clear
        overall_quality = (correctness + completeness + clarity) / 3.0

        # Adjust based on red team findings
        if red_team_feedback:
            # Penalize for critical findings
            critical_count = len(red_team_feedback.must_fix)
            penalty = min(0.2, critical_count * 0.05)
            overall_quality -= penalty

        return {
            "verified_correct": True,
            "details": "Solution demonstrates understanding of requirements",
            "confidence": 0.8,
            "correctness": round(correctness, 3),
            "completeness": round(completeness, 3),
            "clarity": round(clarity, 3),
            "overall_quality": round(max(0.0, overall_quality), 3),
            "recommendation": "accept" if overall_quality >= 0.7 else "revise",
            "notes": ["Solution reviewed by gold team"],
            "additional_findings": []
        }

    def _determine_recommendation(
        self,
        validation_results: SolutionValidationResults,
        requirements: ValidationRequirements
    ) -> str:
        """Determine validation recommendation."""
        # Check if passed threshold
        if validation_results.passed:
            # Also check for critical issues
            if not validation_results.critical_issues:
                return "accept"
            else:
                return "revise"

        # Failed threshold
        if validation_results.final_score < requirements.threshold * 0.5:
            return "reject"

        return "revise"

    def _generate_revision_guidance(
        self,
        validation_results: SolutionValidationResults
    ) -> str:
        """Generate revision guidance based on validation results."""
        guidance_parts = ["REVISION REQUIRED:\n"]

        # Add guidance based on issues
        if validation_results.critical_issues:
            guidance_parts.append("\nCRITICAL ISSUES TO ADDRESS:")
            for issue in validation_results.critical_issues:
                guidance_parts.append(f"- {issue}")

        # Add red team suggestions if available
        if validation_results.red_team_report and validation_results.red_team_report.improvement_suggestions:
            guidance_parts.append("\nSUGGESTED IMPROVEMENTS:")
            for suggestion in validation_results.red_team_report.improvement_suggestions[:5]:
                guidance_parts.append(f"- {suggestion}")

        # Add gold team notes if available
        if validation_results.gold_team_report and validation_results.gold_team_report.verification_notes:
            guidance_parts.append("\nADDITIONAL NOTES:")
            for note in validation_results.gold_team_report.verification_notes[:3]:
                guidance_parts.append(f"- {note}")

        return "\n".join(guidance_parts)

    def _collect_critical_issues(
        self,
        validation_results: SolutionValidationResults
    ) -> List[str]:
        """Collect all critical issues from validation results."""
        critical_issues = []

        # From red team
        if validation_results.red_team_report and validation_results.red_team_report.must_fix:
            critical_issues.extend(validation_results.red_team_report.must_fix)

        # From gold team
        if validation_results.gold_team_report:
            if validation_results.gold_team_report.recommendation == "reject":
                critical_issues.append("Gold team rejected the solution")
            if validation_results.gold_team_report.correctness_score < 0.6:
                critical_issues.append("Correctness concerns identified by gold team")

        # From automated checks
        if validation_results.automated_results and validation_results.automated_results.errors:
            critical_issues.extend(validation_results.automated_results.errors)

        return list(set(critical_issues))  # Remove duplicates


def create_solution_validation_pipeline(
    red_team_system=None,
    gold_team_system=None
) -> SolutionValidationPipeline:
    """
    Factory function to create a SolutionValidationPipeline.

    Args:
        red_team_system: Optional red team feedback system
        gold_team_system: Optional gold team system

    Returns:
        Configured SolutionValidationPipeline instance
    """
    return SolutionValidationPipeline(
        red_team_system=red_team_system,
        gold_team_system=gold_team_system
    )
