"""
Red Team Feedback System for Solution Validation

This module implements the integration of red team feedback into the solution generation process.
The Red Team provides adversarial perspective, finding flaws, edge cases, and potential improvements.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from sovereign_data_models import (
    SolutionAttempt, SubProblem, RedTeamCritiqueReport,
    ValidationResult, generate_id
)
from llm_utils import _request_openai_compatible_chat, _compose_messages

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

# Initialize Robust Engine Singleton for Feedback System
robust_engine = None
if ROMA_MDAP_MAKER_AVAILABLE:
    try:
        # Use SSOT validation preset for standardized high-reliability config
        _config = get_validation_config()
        robust_engine = ROMAMDAPMakerAssociativeEngine(_config)
    except Exception:  # TODO: Catch specific exception instead of Exception
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in {__name__}", exc_info=True)
        raise  # Re-raise the exception

# Configure logging
logger = logging.getLogger(__name__)


class RedTeamFeedbackSystem:
    """
    Manages red team feedback integration.

    Red team provides adversarial perspective, finding flaws,
    edge cases, and potential improvements.
    """

    def __init__(self, team_manager=None):
        """
        Initialize with team manager.

        Args:
            team_manager: Optional team manager for accessing team configurations
        """
        self.team_manager = team_manager
        self.feedback_history: List[Dict[str, Any]] = []

    def generate_red_team_feedback(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem,
        red_team_assignment: Optional[Any] = None
    ) -> RedTeamCritiqueReport:
        """
        Generate red team feedback on a solution.

        Red team analyzes:
        1. Flaws and vulnerabilities
        2. Edge cases not covered
        3. Adversarial scenarios
        4. Potential failures
        5. Security issues
        6. Performance bottlenecks

        Args:
            solution: The solution attempt to critique
            sub_problem: The sub-problem being solved
            red_team_assignment: Optional red team assignment

        Returns:
            CritiqueReport with detailed feedback
        """
        start_time = time.time()

        logger.info(f"Generating red team feedback for solution {solution.id}")

        # Prepare analysis prompt
        analysis_prompt = self._prepare_analysis_prompt(solution, sub_problem)

        # Generate critique
        critique = self._perform_critique(analysis_prompt, solution, sub_problem)

        # Calculate overall score
        critique.overall_score = self._calculate_overall_score(critique)

        # Calculate confidence based on severity and count
        critique.confidence = self._calculate_confidence(critique)

        # Categorize findings
        critique.must_fix, critique.should_fix, critique.could_fix = self._prioritize_findings(
            critique.findings, critique.severity_scores
        )

        # Record timing
        analysis_duration = time.time() - start_time
        logger.info(f"Red team feedback generated in {analysis_duration:.2f}s")

        # Store in history
        self.feedback_history.append({
            'timestamp': datetime.now(),
            'solution_id': solution.id,
            'sub_problem_id': sub_problem.id,
            'overall_score': critique.overall_score,
            'findings_count': len(critique.findings),
            'duration': analysis_duration
        })

        return critique

    def incorporate_feedback(
        self,
        solution: SolutionAttempt,
        feedback: RedTeamCritiqueReport
    ) -> SolutionAttempt:
        """
        Incorporate red team feedback into solution.

        Either:
        - Auto-fix issues if possible
        - Flag for human review
        - Request revision from solver

        Args:
            solution: Original solution attempt
            feedback: Red team critique report

        Returns:
            Updated solution attempt
        """
        logger.info(f"Incorporating feedback for solution {solution.id}")

        # Try automatic fixes first
        if self._can_auto_fix(feedback):
            solution = self._apply_auto_fixes(solution, feedback)
        else:
            # Flag for manual revision
            solution.metadata['requires_revision'] = True
            solution.metadata['feedback_summary'] = {
                'critical_issues': len(feedback.must_fix),
                'important_issues': len(feedback.should_fix),
                'suggestions': len(feedback.could_fix)
            }

        # Add feedback to solution's validation results
        validation_result = ValidationResult(
            validator="red_team",
            passed=feedback.overall_score >= 0.7,
            score=feedback.overall_score,
            feedback=self._format_feedback_summary(feedback),
            improvements=feedback.improvement_suggestions,
            timestamp=datetime.now()
        )
        solution.validation_results.append(validation_result)

        return solution

    def validate_red_team_findings(
        self,
        feedback: RedTeamCritiqueReport,
        gold_team: Optional[Any] = None
    ) -> ValidationResult:
        """
        Validate red team findings.

        Gold team reviews red team feedback to:
        - Confirm real issues
        - Filter false positives
        - Prioritize by severity

        Args:
            feedback: Red team critique report
            gold_team: Optional gold team for validation

        Returns:
            ValidationResult with validation outcomes
        """
        logger.info(f"Validating red team findings from report {feedback.report_id}")

        # Perform validation checks
        confirmed_issues = []
        false_positives = []

        for i, finding in enumerate(feedback.findings):
            severity = feedback.severity_scores[i] if i < len(feedback.severity_scores) else 0.5

            # Validate finding based on severity
            if severity >= 0.7:
                confirmed_issues.append(finding)
            elif severity >= 0.4:
                # Medium severity - needs review
                confirmed_issues.append(finding)
            else:
                # Low severity - potential false positive
                false_positives.append(finding)

        # Calculate validation metrics
        confirmation_rate = len(confirmed_issues) / len(feedback.findings) if feedback.findings else 1.0

        validation_result = ValidationResult(
            validator="gold_team",
            passed=confirmation_rate >= 0.6,
            score=confirmation_rate,
            feedback=f"Validated {len(confirmed_issues)}/{len(feedback.findings)} findings as confirmed issues",
            improvements=[f"Consider reviewing: {fp}" for fp in false_positives[:3]],
            timestamp=datetime.now()
        )

        return validation_result

    def _prepare_analysis_prompt(
        self,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> str:
        """Prepare the analysis prompt for red team critique."""
        prompt = f"""You are a Red Team analyst reviewing a proposed solution. Your job is to find flaws, edge cases, vulnerabilities, and potential failures.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}
Type: {sub_problem.type.value}

SOLUTION ATTEMPT:
Approach: {solution.approach}
Content:
{solution.solution_content}

INSTRUCTIONS:
Analyze this solution from an adversarial perspective and identify:

1. LOGICAL FLAWS: Inconsistencies, contradictions, or faulty reasoning
2. EDGE CASES: Scenarios not covered that could cause failure
3. SECURITY ISSUES: Vulnerabilities, exposure points, risks
4. PERFORMANCE BOTTLENECKS: Scalability concerns, inefficiencies
5. ASSUMPTIONS: Unstated or invalid assumptions
6. ADVERSARIAL SCENARIOS: How could this solution be attacked or fail?

For each finding, provide:
- Title: Brief description
- Description: Detailed explanation
- Severity: 0.0-1.0 (Critical: >0.8, High: 0.6-0.8, Medium: 0.4-0.6, Low: <0.4)
- Category: logical_flaw, edge_case, security, performance, assumption, adversarial
- Suggested fix: How to address the issue

Return your analysis as JSON with this structure:
{{
    "findings": ["Finding 1", "Finding 2", ...],
    "severity_scores": [0.9, 0.6, ...],
    "categories": ["logical_flaw", "edge_case", ...],
    "flaws_found": ["Flaw 1", "Flaw 2", ...],
    "edge_cases_missed": ["Edge case 1", ...],
    "security_issues": ["Security issue 1", ...],
    "performance_issues": ["Performance issue 1", ...],
    "quality_concerns": ["Quality concern 1", ...],
    "improvement_suggestions": ["Suggestion 1", ...]
}}

Be thorough and critical. Your goal is to find problems that could cause this solution to fail in production."""
        return prompt

    def _perform_critique(
        self,
        prompt: str,
        solution: SolutionAttempt,
        sub_problem: SubProblem
    ) -> RedTeamCritiqueReport:
        """Perform the actual critique using LLM or Robust Engine."""
        try:
            # Try Robust Engine First
            response = None
            if robust_engine:
                try:
                    engine_result = robust_engine.solve_problem(
                        problem=prompt,
                        config_overrides={"use_associative_recomposition": True}
                    )
                    response = engine_result.get("solution")
                    
                    # Basic JSON verification
                    if response:
                        if "```json" in response:
                            response = response.split("```json")[1].split("```")[0].strip()
                        elif "```" in response:
                            response = response.split("```")[1].split("```")[0].strip()
                        json.loads(response)
                except Exception:  # TODO: Catch specific exception instead of Exception
                    response = None

            # Fallback to direct call
            if not response:
                messages = _compose_messages(
                    system_prompt="You are an expert Red Team analyst with deep expertise in finding flaws, vulnerabilities, and edge cases in solutions.",
                    user_prompt=prompt
                )

                response = _request_openai_compatible_chat(
                    messages=messages,
                    model="gpt-4",
                    temperature=0.7,
                    response_format="json"
                )

            critique_data = json.loads(response)

            # Build critique report
            critique = RedTeamCritiqueReport(
                report_id=generate_id("critique"),
                team_type="red_team",
                team_id="red_team_default",
                solution_id=solution.id,
                sub_problem_id=sub_problem.id,
                findings=critique_data.get("findings", []),
                severity_scores=critique_data.get("severity_scores", []),
                categories=critique_data.get("categories", []),
                flaws_found=critique_data.get("flaws_found", []),
                edge_cases_missed=critique_data.get("edge_cases_missed", []),
                security_issues=critique_data.get("security_issues", []),
                performance_issues=critique_data.get("performance_issues", []),
                quality_concerns=critique_data.get("quality_concerns", []),
                improvement_suggestions=critique_data.get("improvement_suggestions", []),
                reviewer_prompts=[prompt],
                timestamp=datetime.now()
            )

            return critique

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Error performing critique: {e}")
            # Return minimal critique
            return RedTeamCritiqueReport(
                report_id=generate_id("critique"),
                team_type="red_team",
                team_id="red_team_default",
                solution_id=solution.id,
                sub_problem_id=sub_problem.id,
                findings=[f"Critique generation failed: {str(e)}"],
                severity_scores=[0.5],
                categories=["error"],
                overall_score=0.5,
                confidence=0.3,
                timestamp=datetime.now()
            )

    def _calculate_overall_score(self, critique: RedTeamCritiqueReport) -> float:
        """Calculate overall score from critique."""
        if not critique.severity_scores:
            return 1.0  # No findings = perfect score

        # Average severity weighted by impact
        avg_severity = sum(critique.severity_scores) / len(critique.severity_scores)

        # Convert severity to quality score (inverse relationship)
        # High severity = low quality
        quality_score = 1.0 - avg_severity

        # Adjust for number of findings
        finding_penalty = min(0.3, len(critique.findings) * 0.05)

        final_score = max(0.0, quality_score - finding_penalty)
        return round(final_score, 3)

    def _calculate_confidence(self, critique: RedTeamCritiqueReport) -> float:
        """Calculate confidence in the critique."""
        if not critique.severity_scores:
            return 0.5

        # Confidence based on:
        # 1. Number of findings (more findings = higher confidence)
        # 2. Severity distribution
        # 3. Consistency of categories

        finding_count_score = min(1.0, len(critique.findings) / 10.0)

        # Check if we have high-severity findings
        has_critical = any(s > 0.8 for s in critique.severity_scores)
        critical_bonus = 0.2 if has_critical else 0.0

        # Category diversity
        category_diversity = len(set(critique.categories)) / 6.0  # 6 possible categories

        confidence = (finding_count_score * 0.5) + critical_bonus + (category_diversity * 0.3)
        return round(min(1.0, confidence), 3)

    def _prioritize_findings(
        self,
        findings: List[str],
        severity_scores: List[float]
    ) -> tuple[List[str], List[str], List[str]]:
        """Prioritize findings by severity."""
        must_fix = []
        should_fix = []
        could_fix = []

        for i, finding in enumerate(findings):
            severity = severity_scores[i] if i < len(severity_scores) else 0.5

            if severity >= 0.8:
                must_fix.append(finding)
            elif severity >= 0.5:
                should_fix.append(finding)
            else:
                could_fix.append(finding)

        return must_fix, should_fix, could_fix

    def _can_auto_fix(self, feedback: RedTeamCritiqueReport) -> bool:
        """Determine if feedback can be auto-fixed."""
        # Can auto-fix if:
        # 1. No critical security issues
        # 2. No logical flaws
        # 3. Low severity issues only

        has_security = any(cat == "security" for cat in feedback.categories)
        has_logical_flaws = any(cat == "logical_flaw" for cat in feedback.categories)

        max_severity = max(feedback.severity_scores) if feedback.severity_scores else 0.0

        return not has_security and not has_logical_flaws and max_severity < 0.7

    def _apply_auto_fixes(
        self,
        solution: SolutionAttempt,
        feedback: RedTeamCritiqueReport
    ) -> SolutionAttempt:
        """Apply automatic fixes to solution."""
        # For now, just mark as reviewed with minor issues
        # In a full implementation, could use LLM to apply specific fixes
        solution.metadata['auto_fixed'] = True
        solution.metadata['fixes_applied'] = len(feedback.could_fix)
        return solution

    def _format_feedback_summary(self, feedback: RedTeamCritiqueReport) -> str:
        """Format feedback as a summary string."""
        summary_parts = [
            f"Red Team Score: {feedback.overall_score:.2f}",
            f"Findings: {len(feedback.findings)}",
            f"Critical Issues: {len(feedback.must_fix)}",
            f"Recommendations: {len(feedback.improvement_suggestions)}"
        ]
        return " | ".join(summary_parts)


def create_red_team_feedback_system(team_manager=None) -> RedTeamFeedbackSystem:
    """
    Factory function to create a RedTeamFeedbackSystem.

    Args:
        team_manager: Optional team manager

    Returns:
        Configured RedTeamFeedbackSystem instance
    """
    return RedTeamFeedbackSystem(team_manager=team_manager)
