"""
Critique Aggregator - Usage Examples

This file demonstrates practical usage of the critique_aggregator module
in various scenarios, including integration with the SGD workflow orchestrator.

Author: OpenEvolve Frontend Team
Created: 2026-01-22
"""

from critique_aggregator import (
    CritiqueAggregator,
    JudgeReport,
    CritiqueReport,
    JudgeType,
    CritiqueSeverity,
    AggregationConfig,
    export_critique_report,
    import_critique_report,
    create_sample_judge_reports
)
from typing import List, Dict, Any
import json


# =============================================================================
# EXAMPLE 1: Basic Red Team Gauntlet Integration
# =============================================================================

def example_red_team_gauntlet():
    """
    Example: Using CritiqueAggregator for Red Team gauntlet evaluation.

    This demonstrates how a Red Team gauntlet might use multiple judges
    to critique a solution attempt.
    """
    print("=" * 70)
    print("EXAMPLE 1: Red Team Gauntlet")
    print("=" * 70)

    aggregator = CritiqueAggregator()

    # Simulate multiple Red Team judges evaluating a solution
    judge_reports = [
        JudgeReport(
            judge_name="adversarial_ai_agent",
            judge_type=JudgeType.AI_MODEL,
            is_approved=False,
            score=0.4,
            feedback="Solution fails to handle edge cases with malformed input",
            improvements=[
                "Add input validation for all user-provided data",
                "Implement comprehensive error handling",
                "Add fuzzing tests to find edge cases"
            ],
            severity=CritiqueSeverity.HIGH,
            confidence=0.85,
            metrics={"edge_cases_found": 7, "attack_vectors": 3}
        ),
        JudgeReport(
            judge_name="security_analyzer",
            judge_type=JudgeType.SECURITY_SCANNER,
            is_approved=False,
            score=0.5,
            feedback="Potential SQL injection vulnerability in user query handling",
            improvements=[
                "Use parameterized queries instead of string concatenation",
                "Implement prepared statements",
                "Add ORM layer for database access"
            ],
            severity=CritiqueSeverity.CRITICAL,
            confidence=0.95,
            metrics={"vulnerabilities_found": 3, "cves_matched": 1}
        ),
        JudgeReport(
            judge_name="performance_profiler",
            judge_type=JudgeType.PERFORMANCE_ANALYZER,
            is_approved=True,
            score=0.75,
            feedback="Performance is acceptable but can be optimized",
            improvements=[
                "Add database indexing for frequently queried columns",
                "Implement caching for repeated queries",
                "Optimize nested loops"
            ],
            severity=CritiqueSeverity.MEDIUM,
            confidence=0.8,
            metrics={"avg_response_time_ms": 250, "memory_mb": 512}
        )
    ]

    # Create the critique report
    critique_report = aggregator.create_critique_report(
        solution_id="sub_problem_1_solution_v1",
        gauntlet_name="red_team_security_gauntlet",
        critiques=judge_reports
    )

    # Display results
    print(f"\nSolution ID: {critique_report.solution_attempt_id}")
    print(f"Gauntlet: {critique_report.gauntlet_name}")
    print(f"Approved: {critique_report.is_approved}")
    print(f"Aggregate Score: {critique_report.aggregate_score:.2f}")
    print(f"Consensus: {critique_report.consensus_score:.2f}")

    print(f"\n--- Summary ---")
    print(critique_report.summary[:500] + "...")

    print(f"\n--- Critical Improvements Needed ---")
    for improvement in critique_report.improvements_needed[:5]:
        print(f"  - {improvement}")

    return critique_report


# =============================================================================
# EXAMPLE 2: Gold Team Verification
# =============================================================================

def example_gold_team_verification():
    """
    Example: Using CritiqueAggregator for Gold Team verification.

    The Gold Team provides constructive feedback and verification
    after issues have been addressed.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Gold Team Verification")
    print("=" * 70)

    # Configure aggregator for Gold Team (higher standards)
    config = AggregationConfig(
        default_approval_threshold=0.8,  # Higher bar for Gold Team
        default_weights={
            JudgeType.HUMAN: 1.0,
            JudgeType.AI_MODEL: 0.85,
            JudgeType.AUTOMATED_TEST: 0.9
        },
        min_judges_required=2,
        enable_outlier_detection=False
    )

    aggregator = CritiqueAggregator(config)

    # Gold Team judges evaluate the improved solution
    judge_reports = [
        JudgeReport(
            judge_name="senior_developer_review",
            judge_type=JudgeType.HUMAN,
            is_approved=True,
            score=0.85,
            feedback="Code quality is much improved. Good error handling and security practices.",
            improvements=[
                "Consider adding more comprehensive integration tests",
                "Documentation could be more detailed for complex algorithms"
            ],
            severity=CritiqueSeverity.LOW,
            confidence=0.9
        ),
        JudgeReport(
            judge_name="test_suite_automated",
            judge_type=JudgeType.AUTOMATED_TEST,
            is_approved=True,
            score=0.95,
            feedback="All tests passing. Coverage: 92%",
            improvements=[],
            severity=CritiqueSeverity.INFO,
            confidence=1.0,
            metrics={
                "tests_total": 145,
                "tests_passed": 145,
                "tests_failed": 0,
                "coverage_percent": 92.3
            }
        ),
        JudgeReport(
            judge_name="code_review_ai",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.88,
            feedback="Solution follows best practices. Clean, maintainable code.",
            improvements=["Minor style suggestions: PEP8 compliance"],
            severity=CritiqueSeverity.INFO,
            confidence=0.82
        )
    ]

    critique_report = aggregator.create_critique_report(
        solution_id="sub_problem_1_solution_v2",
        gauntlet_name="gold_team_quality_gauntlet",
        critiques=judge_reports
    )

    print(f"\nSolution ID: {critique_report.solution_attempt_id}")
    print(f"Approved: {critique_report.is_approved}")
    print(f"Score: {critique_report.aggregate_score:.2f}")
    print(f"Consensus: {critique_report.consensus_score:.2f}")

    return critique_report


# =============================================================================
# EXAMPLE 3: Integration with SGD Workflow Orchestrator
# =============================================================================

def example_sgd_workflow_integration():
    """
    Example: Integration with SGD Workflow Orchestrator.

    This shows how the CritiqueAggregator would be used within
    the context of the sgd_workflow_orchestrator.py workflow.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: SGD Workflow Integration")
    print("=" * 70)

    from typing import Optional
    from datetime import datetime

    # Simulate the workflow context
    class SimulatedSGDWorkflow:
        """Simulated SGD workflow for demonstration."""

        def __init__(self):
            self.aggregator = CritiqueAggregator()
            self.sub_problems = {
                "sp_1": "Implement user authentication",
                "sp_2": "Design database schema",
                "sp_3": "Create API endpoints"
            }

        def evaluate_solution_attempt(
            self,
            sub_problem_id: str,
            solution_content: str
        ) -> CritiqueReport:
            """
            Evaluate a solution attempt using configured gauntlets.

            This simulates what happens in the SGD workflow when a
            solution is submitted to Red Team and Gold Team gauntlets.
            """
            print(f"\nEvaluating solution for: {self.sub_problems[sub_problem_id]}")

            # Simulate Red Team critique (would use actual gauntlet in production)
            red_team_reports = self._run_red_team_gauntlet(
                sub_problem_id,
                solution_content
            )

            red_team_critique = self.aggregator.create_critique_report(
                solution_id=f"{sub_problem_id}_attempt_1",
                gauntlet_name=f"red_team_{sub_problem_id}",
                critiques=red_team_reports
            )

            print(f"  Red Team - Approved: {red_team_critique.is_approved}, "
                  f"Score: {red_team_critique.aggregate_score:.2f}")

            # If Red Team rejects, stop here
            if not red_team_critique.is_approved:
                print("  Solution failed Red Team evaluation")
                return red_team_critique

            # Run Gold Team verification
            gold_team_reports = self._run_gold_team_gauntlet(
                sub_problem_id,
                solution_content
            )

            gold_team_critique = self.aggregator.create_critique_report(
                solution_id=f"{sub_problem_id}_attempt_1",
                gauntlet_name=f"gold_team_{sub_problem_id}",
                critiques=gold_team_reports
            )

            print(f"  Gold Team - Approved: {gold_team_critique.is_approved}, "
                  f"Score: {gold_team_critique.aggregate_score:.2f}")

            return gold_team_critique

        def _run_red_team_gauntlet(
            self,
            sub_problem_id: str,
            solution_content: str
        ) -> List[JudgeReport]:
            """Simulate Red Team gauntlet execution."""
            # In production, this would call actual gauntlet systems
            return [
                JudgeReport(
                    judge_name="security_scanner",
                    judge_type=JudgeType.SECURITY_SCANNER,
                    is_approved=True,
                    score=0.85,
                    feedback="No critical security issues found",
                    improvements=[],
                    severity=CritiqueSeverity.INFO
                ),
                JudgeReport(
                    judge_name="adversarial_tester",
                    judge_type=JudgeType.AI_MODEL,
                    is_approved=True,
                    score=0.78,
                    feedback="Solution handles most edge cases",
                    improvements=["Consider additional input validation"],
                    severity=CritiqueSeverity.MEDIUM
                )
            ]

        def _run_gold_team_gauntlet(
            self,
            sub_problem_id: str,
            solution_content: str
        ) -> List[JudgeReport]:
            """Simulate Gold Team gauntlet execution."""
            return [
                JudgeReport(
                    judge_name="quality_assurance",
                    judge_type=JudgeType.AUTOMATED_TEST,
                    is_approved=True,
                    score=0.92,
                    feedback="All tests passed",
                    metrics={"tests_passed": 45, "coverage": 0.89}
                ),
                JudgeReport(
                    judge_name="code_reviewer",
                    judge_type=JudgeType.HUMAN,
                    is_approved=True,
                    score=0.88,
                    feedback="Code quality is good",
                    improvements=["Minor style improvements"]
                )
            ]

    # Run the simulated workflow
    workflow = SimulatedSGDWorkflow()

    # Evaluate a solution for sub_problem_1
    solution_content = """
    def authenticate_user(username, password):
        # Implementation here
        pass
    """

    critique_report = workflow.evaluate_solution_attempt(
        sub_problem_id="sp_1",
        solution_content=solution_content
    )

    return critique_report


# =============================================================================
# EXAMPLE 4: Multi-Round Iteration with Improvement Tracking
# =============================================================================

def example_multi_round_iteration():
    """
    Example: Tracking improvements across multiple iterations.

    This demonstrates how to use CritiqueAggregator to track
    progress as a solution is refined through multiple iterations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Multi-Round Iteration")
    print("=" * 70)

    aggregator = CritiqueAggregator()

    # Track improvements across iterations
    iterations = []

    for iteration in range(1, 4):
        print(f"\n--- Iteration {iteration} ---")

        # Simulate that each iteration addresses previous feedback
        if iteration == 1:
            feedback = "Initial implementation has several issues"
            score = 0.55
            improvements = [
                "Add error handling",
                "Implement input validation",
                "Add unit tests"
            ]
        elif iteration == 2:
            feedback = "Improvement seen, but more work needed"
            score = 0.72
            improvements = [
                "Optimize performance",
                "Improve code documentation"
            ]
        else:
            feedback = "Solution is now production-ready"
            score = 0.91
            improvements = []

        judge_report = JudgeReport(
            judge_name=f"iteration_{iteration}_reviewer",
            judge_type=JudgeType.HUMAN,
            is_approved=score >= 0.8,
            score=score,
            feedback=feedback,
            improvements=improvements,
            severity=CritiqueSeverity.MEDIUM if score < 0.8 else CritiqueSeverity.INFO
        )

        critique_report = aggregator.create_critique_report(
            solution_id=f"solution_iteration_{iteration}",
            gauntlet_name="quality_gauntlet",
            critiques=[judge_report]
        )

        iterations.append({
            "iteration": iteration,
            "score": critique_report.aggregate_score,
            "approved": critique_report.is_approved,
            "improvements_count": len(critique_report.improvements_needed)
        })

        print(f"  Score: {score:.2f}")
        print(f"  Approved: {critique_report.is_approved}")
        print(f"  Improvements needed: {len(improvements)}")

    # Show progress
    print("\n--- Progress Summary ---")
    for it in iterations:
        print(f"Iteration {it['iteration']}: "
              f"Score {it['score']:.2f}, "
              f"Approved: {it['approved']}, "
              f"Improvements: {it['improvements_count']}")

    return iterations


# =============================================================================
# EXAMPLE 5: Advanced Configuration and Custom Weights
# =============================================================================

def example_advanced_configuration():
    """
    Example: Advanced configuration with custom judge weights and
    specialized aggregation settings.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Advanced Configuration")
    print("=" * 70)

    # Create custom configuration for a security-focused evaluation
    config = AggregationConfig(
        default_approval_threshold=0.85,  # High bar for security-critical code
        default_weights={
            JudgeType.SECURITY_SCANNER: 1.5,  # Extra weight on security
            JudgeType.HUMAN: 1.2,  # Human reviewers are important
            JudgeType.AI_MODEL: 0.8,
            JudgeType.AUTOMATED_TEST: 0.9,
            JudgeType.PERFORMANCE_ANALYZER: 0.6,
            JudgeType.LINTING_TOOL: 0.3
        },
        min_judges_required=3,
        enable_outlier_detection=True,
        outlier_std_dev_threshold=1.5,
        consensus_algorithm="pairwise_agreement",
        summary_max_length=3000,
        extract_improvements=True
    )

    aggregator = CritiqueAggregator(config)

    # Simulate a security-focused evaluation
    judge_reports = [
        JudgeReport(
            judge_name="owasp_zap",
            judge_type=JudgeType.SECURITY_SCANNER,
            is_approved=True,
            score=0.92,
            feedback="No OWASP Top 10 vulnerabilities detected",
            improvements=[],
            severity=CritiqueSeverity.INFO,
            confidence=1.0
        ),
        JudgeReport(
            judge_name="security_expert_human",
            judge_type=JudgeType.HUMAN,
            is_approved=True,
            score=0.88,
            feedback="Security implementation is solid",
            improvements=["Consider adding rate limiting"],
            severity=CritiqueSeverity.LOW,
            confidence=0.95
        ),
        JudgeReport(
            judge_name="gpt4_security_review",
            judge_type=JudgeType.AI_MODEL,
            is_approved=True,
            score=0.85,
            feedback="Good security practices followed",
            improvements=["Add more detailed security documentation"],
            severity=CritiqueSeverity.LOW,
            confidence=0.82
        )
    ]

    critique_report = aggregator.create_critique_report(
        solution_id="security_critical_component",
        gauntlet_name="security_focused_gauntlet",
        critiques=judge_reports
    )

    print(f"\nSecurity Evaluation Results:")
    print(f"  Approved: {critique_report.is_approved}")
    print(f"  Score: {critique_report.aggregate_score:.2f} "
          f"(threshold: {config.default_approval_threshold:.2f})")
    print(f"  Consensus: {critique_report.consensus_score:.2f} "
          f"(algorithm: {config.consensus_algorithm})")

    return critique_report


# =============================================================================
# EXAMPLE 6: Export and Import for Audit Trail
# =============================================================================

def example_audit_trail():
    """
    Example: Creating an audit trail by exporting and importing reports.

    This demonstrates how to maintain a permanent record of all evaluations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Audit Trail")
    print("=" * 70)

    aggregator = CritiqueAggregator()

    # Create a critique report
    judge_reports = create_sample_judge_reports(num_reports=4)
    critique_report = aggregator.create_critique_report(
        solution_id="audited_solution_001",
        gauntlet_name="compliance_gauntlet",
        critiques=judge_reports
    )

    # Export for audit trail
    json_path = "/tmp/audit_report_001.json"
    txt_path = "/tmp/audit_report_001.txt"

    export_critique_report(critique_report, json_path, format="json")
    print(f"\nExported JSON audit trail to: {json_path}")

    export_critique_report(critique_report, txt_path, format="txt")
    print(f"Exported TXT audit trail to: {txt_path}")

    # Demonstrate importing for verification
    imported_report = import_critique_report(json_path)
    print(f"\nVerified imported report:")
    print(f"  Solution ID: {imported_report.solution_attempt_id}")
    print(f"  Approved: {imported_report.is_approved}")
    print(f"  Score: {imported_report.aggregate_score:.2f}")

    return critique_report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print(" CRITIQUE AGGREGATOR - USAGE EXAMPLES")
    print("*" * 70)

    # Run all examples
    example_red_team_gauntlet()
    example_gold_team_verification()
    example_sgd_workflow_integration()
    example_multi_round_iteration()
    example_advanced_configuration()
    example_audit_trail()

    print("\n" + "*" * 70)
    print(" ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
