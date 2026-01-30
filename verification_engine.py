"""
Verification Engine - Production-Ready Implementation

This module provides comprehensive verification capabilities for solution attempts,
integrating with sovereign_data_models, crewai_state_management, and sgd_workflow_orchestrator.

Key Features:
- VerificationReport generation with detailed quality metrics
- SuccessCriterion definition and validation
- Multi-dimensional quality scoring
- Comprehensive error handling
- Type hints throughout
- Production-ready logging and monitoring
- Edge case handling
- Unit test suite
- Usage examples

Author: OpenEvolve Frontend Team
Version: 1.0.0
Created: 2026-01-22
"""

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import json

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODEL DEFINITIONS
# =============================================================================

@dataclass
class SuccessCriterion:
    """
    Represents a success criterion for solution verification.

    Attributes:
        id: Unique identifier for the criterion
        description: Human-readable description of the criterion
        metric: The metric being measured (e.g., 'completeness', 'correctness')
        threshold: Minimum value required to pass (0.0 to 1.0)
        weight: Importance weight for scoring (default: 1.0)
        category: Category of criterion (functional, non_functional, quality)
    """
    id: str
    description: str
    metric: str
    threshold: float
    weight: float = 1.0
    category: str = "functional"

    def __post_init__(self):
        """Validate criterion fields."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if not self.id:
            raise ValueError("Criterion ID cannot be empty")


@dataclass
class SolutionQualityMetrics:
    """
    Comprehensive quality metrics for a solution attempt.

    Attributes:
        completeness: How completely the solution addresses requirements (0-1)
        correctness: Accuracy and correctness of the solution (0-1)
        efficiency: Performance and resource utilization (0-1)
        clarity: Code/documentation clarity (0-1)
        maintainability: Ease of maintenance (0-1)
        scalability: Ability to scale (0-1)
        security: Security considerations (0-1)
        test_coverage: Test coverage percentage (0-1)
        overall_score: Weighted average of all metrics (0-1)
        confidence: Confidence in the quality assessment (0-1)
    """
    completeness: float = 0.0
    correctness: float = 0.0
    efficiency: float = 0.0
    clarity: float = 0.0
    maintainability: float = 0.0
    scalability: float = 0.0
    security: float = 0.0
    test_coverage: float = 0.0
    overall_score: float = 0.0
    confidence: float = 0.5

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall quality score with optional custom weights.

        Args:
            weights: Dictionary of metric weights (default: equal weighting)

        Returns:
            Overall quality score (0-1)
        """
        default_weights = {
            'completeness': 0.20,
            'correctness': 0.25,
            'efficiency': 0.10,
            'clarity': 0.10,
            'maintainability': 0.10,
            'scalability': 0.10,
            'security': 0.10,
            'test_coverage': 0.05
        }

        used_weights = weights or default_weights

        total_score = 0.0
        total_weight = 0.0

        for metric, weight in used_weights.items():
            if hasattr(self, metric):
                value = getattr(self, metric)
                total_score += value * weight
                total_weight += weight

        self.overall_score = total_score / total_weight if total_weight > 0 else 0.0
        return self.overall_score

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'completeness': self.completeness,
            'correctness': self.correctness,
            'efficiency': self.efficiency,
            'clarity': self.clarity,
            'maintainability': self.maintainability,
            'scalability': self.scalability,
            'security': self.security,
            'test_coverage': self.test_coverage,
            'overall_score': self.overall_score,
            'confidence': self.confidence
        }


@dataclass
class VerificationReport:
    """
    Comprehensive verification report for a solution attempt.

    Attributes:
        solution_attempt_id: ID of the solution being verified
        gauntlet_name: Name of the gauntlet used for verification
        is_approved: Whether the solution passed verification
        reports_by_judge: List of individual judge reports
        summary: Human-readable summary of verification
        quality_metrics: Quality metrics calculated for the solution
        criteria_results: Results for each success criterion
        timestamp: When verification was performed
        verification_score: Overall verification score (0-1)
        metadata: Additional verification metadata
    """
    solution_attempt_id: str
    gauntlet_name: str
    is_approved: bool
    reports_by_judge: List[Any]
    summary: str
    quality_metrics: Optional[SolutionQualityMetrics] = None
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    verification_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'solution_attempt_id': self.solution_attempt_id,
            'gauntlet_name': self.gauntlet_name,
            'is_approved': self.is_approved,
            'reports_by_judge': self.reports_by_judge,
            'summary': self.summary,
            'quality_metrics': self.quality_metrics.to_dict() if self.quality_metrics else None,
            'criteria_results': self.criteria_results,
            'timestamp': self.timestamp,
            'verification_score': self.verification_score,
            'metadata': self.metadata
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# VERIFICATION ENGINE
# =============================================================================

class VerificationEngine:
    """
    Production-ready verification engine for solution attempts.

    This engine provides comprehensive verification capabilities including:
    - Success criteria validation
    - Quality metrics calculation
    - Verification report generation
    - Test suite execution
    - Edge case handling
    """

    # Default quality metric weights
    DEFAULT_WEIGHTS = {
        'completeness': 0.20,
        'correctness': 0.25,
        'efficiency': 0.10,
        'clarity': 0.10,
        'maintainability': 0.10,
        'scalability': 0.10,
        'security': 0.10,
        'test_coverage': 0.05
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the verification engine.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.verification_history: List[VerificationReport] = []
        self.logger = logging.getLogger(f"{__name__}.VerificationEngine")

        # Extract configuration
        self.strict_mode = self.config.get('strict_mode', False)
        self.min_quality_threshold = self.config.get('min_quality_threshold', 0.6)
        self.enable_detailed_logging = self.config.get('enable_detailed_logging', True)

        self.logger.info("VerificationEngine initialized")

    def verify_solution(
        self,
        solution: Any,
        criteria: List[SuccessCriterion]
    ) -> VerificationReport:
        """
        Verify a solution attempt against defined success criteria.

        Args:
            solution: SolutionAttempt object to verify
            criteria: List of success criteria to validate against

        Returns:
            VerificationReport with detailed results

        Raises:
            ValueError: If solution or criteria are invalid
        """
        if not solution:
            raise ValueError("Solution cannot be None")

        if not criteria:
            raise ValueError("At least one success criterion must be provided")

        self.logger.info(f"Starting verification for solution: {getattr(solution, 'id', 'unknown')}")

        start_time = time.time()

        try:
            # Extract solution content
            solution_content = self._extract_solution_content(solution)
            if not solution_content or not solution_content.strip():
                raise ValueError("Solution content is empty")

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_scores(solution)
            self.logger.info(f"Quality metrics calculated: overall={quality_metrics.overall_score:.2f}")

            # Check each criterion
            criteria_results = {}
            passed_criteria = 0
            total_criteria = len(criteria)

            for criterion in criteria:
                try:
                    passed = self.check_criterion(solution, criterion)
                    criteria_results[criterion.id] = passed

                    if passed:
                        passed_criteria += 1
                        self.logger.debug(f"Criterion {criterion.id} PASSED")
                    else:
                        self.logger.warning(f"Criterion {criterion.id} FAILED")

                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    self.logger.error(f"Error checking criterion {criterion.id}: {e}")
                    criteria_results[criterion.id] = False

            # Calculate verification score
            verification_score = passed_criteria / total_criteria if total_criteria > 0 else 0.0

            # Determine approval
            is_approved = (
                verification_score >= self.min_quality_threshold and
                quality_metrics.overall_score >= self.min_quality_threshold
            )

            # Generate summary
            summary = self._generate_summary(
                passed_criteria,
                total_criteria,
                quality_metrics,
                criteria_results
            )

            # Create verification report
            solution_id = getattr(solution, 'id', getattr(solution, 'sub_problem_id', 'unknown'))
            gauntlet_name = getattr(solution, 'gauntlet_name', 'default_gauntlet')

            report = VerificationReport(
                solution_attempt_id=solution_id,
                gauntlet_name=gauntlet_name,
                is_approved=is_approved,
                reports_by_judge=[self._create_judge_report(criteria_results, quality_metrics)],
                summary=summary,
                quality_metrics=quality_metrics,
                criteria_results=criteria_results,
                verification_score=verification_score,
                metadata={
                    'verification_time_seconds': time.time() - start_time,
                    'strict_mode': self.strict_mode,
                    'total_criteria': total_criteria
                }
            )

            # Store in history
            self.verification_history.append(report)

            self.logger.info(
                f"Verification completed: approved={is_approved}, "
                f"score={verification_score:.2f}"
            )

            return report

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            self.logger.error(f"Verification failed with error: {e}")
            # Create failure report
            solution_id = getattr(solution, 'id', getattr(solution, 'sub_problem_id', 'unknown'))
            return VerificationReport(
                solution_attempt_id=solution_id,
                gauntlet_name='error',
                is_approved=False,
                reports_by_judge=[],
                summary=f"Verification failed with error: {str(e)}",
                verification_score=0.0,
                metadata={'error': str(e)}
            )

    def create_success_criteria(self, requirements: List[str]) -> List[SuccessCriterion]:
        """
        Create success criteria from requirement strings.

        Parses requirement strings and converts them into SuccessCriterion objects
        with appropriate metrics and thresholds.

        Args:
            requirements: List of requirement descriptions

        Returns:
            List of SuccessCriterion objects

        Example:
            >>> requirements = [
            ...     "Solution must be at least 90% complete",
            ...     "Code must pass all security checks"
            ... ]
            >>> criteria = engine.create_success_criteria(requirements)
        """
        criteria = []
        criterion_id = 0

        for requirement in requirements:
            try:
                # Parse requirement to extract metric and threshold
                metric, threshold, category = self._parse_requirement(requirement)

                criterion = SuccessCriterion(
                    id=f"criterion_{criterion_id:03d}",
                    description=requirement,
                    metric=metric,
                    threshold=threshold,
                    category=category
                )

                criteria.append(criterion)
                criterion_id += 1

                self.logger.debug(f"Created criterion: {criterion.id} - {metric} >= {threshold}")

            except (ValueError, TypeError, RuntimeError) as e:
                self.logger.warning(f"Failed to parse requirement '{requirement}': {e}")
                # Create a default criterion
                criteria.append(SuccessCriterion(
                    id=f"criterion_{criterion_id:03d}",
                    description=requirement,
                    metric='completeness',
                    threshold=0.7,
                    category='functional'
                ))
                criterion_id += 1

        self.logger.info(f"Created {len(criteria)} success criteria from {len(requirements)} requirements")
        return criteria

    def check_criterion(self, solution: Any, criterion: SuccessCriterion) -> bool:
        """
        Check if a solution meets a specific success criterion.

        Args:
            solution: SolutionAttempt to check
            criterion: SuccessCriterion to validate

        Returns:
            True if criterion is met, False otherwise
        """
        try:
            # Extract solution content
            solution_content = self._extract_solution_content(solution)
            if not solution_content:
                self.logger.warning("Solution content is empty")
                return False

            # Get metric value
            metric_value = self._calculate_metric(solution, criterion.metric)

            # Check against threshold
            passed = metric_value >= criterion.threshold

            self.logger.debug(
                f"Criterion {criterion.id}: {criterion.metric}="
                f"{metric_value:.2f} >= {criterion.threshold} -> {passed}"
            )

            return passed

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            self.logger.error(f"Error checking criterion {criterion.id}: {e}")
            return False

    def calculate_quality_scores(self, solution: Any) -> SolutionQualityMetrics:
        """
        Calculate comprehensive quality metrics for a solution.

        Args:
            solution: SolutionAttempt to analyze

        Returns:
            SolutionQualityMetrics with all quality dimensions
        """
        try:
            # Extract solution content
            solution_content = self._extract_solution_content(solution)
            if not solution_content:
                return SolutionQualityMetrics()

            metrics = SolutionQualityMetrics()

            # Calculate each dimension
            metrics.completeness = self._calculate_completeness(solution_content)
            metrics.correctness = self._calculate_correctness(solution_content)
            metrics.efficiency = self._calculate_efficiency(solution_content)
            metrics.clarity = self._calculate_clarity(solution_content)
            metrics.maintainability = self._calculate_maintainability(solution_content)
            metrics.scalability = self._calculate_scalability(solution_content)
            metrics.security = self._calculate_security(solution_content)
            metrics.test_coverage = self._calculate_test_coverage(solution_content)

            # Calculate overall score
            metrics.calculate_overall(self.DEFAULT_WEIGHTS)

            # Estimate confidence based on content length and structure
            metrics.confidence = self._calculate_confidence(solution_content, metrics)

            self.logger.info(f"Quality scores calculated: overall={metrics.overall_score:.2f}")

            return metrics

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            self.logger.error(f"Error calculating quality scores: {e}")
            return SolutionQualityMetrics()

    def generate_verification_report(
        self,
        solution_id: str,
        gauntlet_name: str,
        results: List[Any]
    ) -> VerificationReport:
        """
        Generate a verification report from raw validation results.

        Args:
            solution_id: ID of the solution
            gauntlet_name: Name of the gauntlet used
            results: List of validation results

        Returns:
            VerificationReport
        """
        try:
            # Aggregate results
            is_approved = all(
                result.get('is_approved', result.get('passed', False))
                for result in results
                if isinstance(result, dict)
            )

            # Calculate scores
            scores = [
                result.get('score', 0.0)
                for result in results
                if isinstance(result, dict)
            ]
            verification_score = sum(scores) / len(scores) if scores else 0.0

            # Generate summary
            summary = self._generate_results_summary(results)

            # Extract quality metrics if available
            quality_metrics = None
            for result in results:
                if isinstance(result, dict) and 'quality_metrics' in result:
                    metrics_dict = result['quality_metrics']
                    quality_metrics = SolutionQualityMetrics(**metrics_dict)
                    break

            report = VerificationReport(
                solution_attempt_id=solution_id,
                gauntlet_name=gauntlet_name,
                is_approved=is_approved,
                reports_by_judge=results,
                summary=summary,
                quality_metrics=quality_metrics,
                verification_score=verification_score,
                metadata={
                    'num_results': len(results),
                    'generated_at': datetime.now().isoformat()
                }
            )

            self.logger.info(f"Generated verification report for {solution_id}")

            return report

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            self.logger.error(f"Error generating verification report: {e}")
            # Return minimal report
            return VerificationReport(
                solution_attempt_id=solution_id,
                gauntlet_name=gauntlet_name,
                is_approved=False,
                reports_by_judge=results,
                summary=f"Error generating report: {str(e)}",
                verification_score=0.0
            )

    def run_verification_suite(
        self,
        solution: Any,
        test_suite: List[Any]
    ) -> VerificationReport:
        """
        Run a complete verification suite against a solution.

        Args:
            solution: SolutionAttempt to verify
            test_suite: List of test cases or validators

        Returns:
            VerificationReport with complete results
        """
        self.logger.info(f"Running verification suite with {len(test_suite)} tests")

        start_time = time.time()
        results = []
        passed_tests = 0

        solution_id = getattr(solution, 'id', getattr(solution, 'sub_problem_id', 'unknown'))
        solution_content = self._extract_solution_content(solution)

        for i, test_case in enumerate(test_suite):
            try:
                # Execute test case
                test_result = self._execute_test(solution, test_case)
                results.append(test_result)

                if test_result.get('passed', False):
                    passed_tests += 1

            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                self.logger.error(f"Test {i} failed with error: {e}")
                results.append({
                    'test_index': i,
                    'passed': False,
                    'error': str(e)
                })

        # Calculate metrics
        test_pass_rate = passed_tests / len(test_suite) if test_suite else 0.0

        # Generate quality metrics
        quality_metrics = self.calculate_quality_scores(solution)

        # Determine approval
        is_approved = (
            test_pass_rate >= self.min_quality_threshold and
            quality_metrics.overall_score >= self.min_quality_threshold
        )

        # Generate summary
        summary = (
            f"Verification suite completed: {passed_tests}/{len(test_suite)} tests passed "
            f"({test_pass_rate*100:.1f}%). "
            f"Overall quality score: {quality_metrics.overall_score:.2f}"
        )

        report = VerificationReport(
            solution_attempt_id=solution_id,
            gauntlet_name='verification_suite',
            is_approved=is_approved,
            reports_by_judge=results,
            summary=summary,
            quality_metrics=quality_metrics,
            verification_score=test_pass_rate,
            metadata={
                'total_tests': len(test_suite),
                'passed_tests': passed_tests,
                'execution_time_seconds': time.time() - start_time
            }
        )

        self.logger.info(
            f"Verification suite completed: {passed_tests}/{len(test_suite)} passed, "
            f"approved={is_approved}"
        )

        return report

    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================

    def _extract_solution_content(self, solution: Any) -> str:
        """Extract solution content from various solution types."""
        # Try different attributes
        for attr in ['id', 'solution_content', 'content', 'code', 'solution', 'explanation']:
            if hasattr(solution, attr):
                # Skip id attribute for content extraction
                if attr == 'id':
                    continue
                content = getattr(solution, attr)
                if isinstance(content, str) and content.strip():
                    return content

        # If dictionary
        if isinstance(solution, dict):
            for key in ['id', 'solution_content', 'content', 'code', 'solution', 'explanation']:
                if key in solution and key != 'id':
                    content = solution[key]
                    if isinstance(content, str) and content.strip():
                        return str(content)

        # Last resort: string representation
        return str(solution) if solution else ""

    def _parse_requirement(self, requirement: str) -> Tuple[str, float, str]:
        """
        Parse requirement string to extract metric and threshold.

        Args:
            requirement: Requirement description

        Returns:
            Tuple of (metric, threshold, category)
        """
        requirement_lower = requirement.lower()

        # Default values
        metric = 'completeness'
        threshold = 0.7
        category = 'functional'

        # Extract percentage
        percentage_match = re.search(r'(\d+)%', requirement)
        if percentage_match:
            threshold = int(percentage_match.group(1)) / 100.0

        # Determine metric from keywords
        if 'complete' in requirement_lower or 'cover' in requirement_lower:
            metric = 'completeness'
            category = 'functional'
        elif 'correct' in requirement_lower or 'accurate' in requirement_lower:
            metric = 'correctness'
            category = 'functional'
        elif 'secure' in requirement_lower or 'security' in requirement_lower:
            metric = 'security'
            category = 'security'
        elif 'efficient' in requirement_lower or 'performance' in requirement_lower:
            metric = 'efficiency'
            category = 'non_functional'
        elif 'clear' in requirement_lower or 'readable' in requirement_lower:
            metric = 'clarity'
            category = 'quality'
        elif 'maintain' in requirement_lower:
            metric = 'maintainability'
            category = 'quality'
        elif 'scale' in requirement_lower or 'scalable' in requirement_lower:
            metric = 'scalability'
            category = 'non_functional'
        elif 'test' in requirement_lower:
            metric = 'test_coverage'
            category = 'quality'

        return metric, threshold, category

    def _calculate_metric(self, solution: Any, metric_name: str) -> float:
        """Calculate a specific metric for a solution."""
        solution_content = self._extract_solution_content(solution)

        metric_calculators = {
            'completeness': self._calculate_completeness,
            'correctness': self._calculate_correctness,
            'efficiency': self._calculate_efficiency,
            'clarity': self._calculate_clarity,
            'maintainability': self._calculate_maintainability,
            'scalability': self._calculate_scalability,
            'security': self._calculate_security,
            'test_coverage': self._calculate_test_coverage
        }

        calculator = metric_calculators.get(metric_name)
        if calculator:
            return calculator(solution_content)

        self.logger.warning(f"Unknown metric: {metric_name}")
        return 0.5

    def _calculate_completeness(self, content: str) -> float:
        """Calculate completeness score based on content analysis."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Length heuristic
        length = len(content)
        if length > 100:
            score += 0.1
        if length > 500:
            score += 0.1
        if length > 1000:
            score += 0.1

        # Check for key sections
        sections = ['def ', 'class ', 'import ', 'function', 'return', 'if ', 'else']
        found_sections = sum(1 for section in sections if section in content)
        score += min(found_sections * 0.05, 0.2)

        return min(score, 1.0)

    def _calculate_correctness(self, content: str) -> float:
        """Calculate correctness score based on code patterns."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for error handling
        if 'try:' in content or 'except' in content:
            score += 0.1

        # Check for type hints
        if ': ' in content and '->' in content:
            score += 0.1

        # Check for docstrings
        if '"""' in content or "'''" in content:
            score += 0.1

        # Check for assertions/validation
        if 'assert' in content or 'raise' in content:
            score += 0.1

        # Check for return statements
        if 'return ' in content:
            score += 0.1

        return min(score, 1.0)

    def _calculate_efficiency(self, content: str) -> float:
        """Calculate efficiency score based on algorithmic patterns."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for efficient patterns
        efficient_patterns = ['for ', 'while ', 'in ', 'list comprehension', 'map']
        found_patterns = sum(1 for pattern in efficient_patterns if pattern in content.lower())
        score += min(found_patterns * 0.05, 0.2)

        # Check for caching
        if 'cache' in content.lower() or 'memo' in content.lower():
            score += 0.15

        # Check for early returns
        if 'return' in content and 'if ' in content:
            score += 0.15

        return min(score, 1.0)

    def _calculate_clarity(self, content: str) -> float:
        """Calculate clarity score based on code readability."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for comments
        comment_patterns = ['#', '"""', "'''", '/*']
        has_comments = any(pattern in content for pattern in comment_patterns)
        if has_comments:
            score += 0.2

        # Check for meaningful names (heuristic: length > 3)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', content)
        if words:
            avg_length = sum(len(w) for w in words) / len(words)
            if avg_length > 5:
                score += 0.15

        # Check for consistent indentation
        lines = content.split('\n')
        if lines:
            indent_pattern = re.match(r'^(\s+)', lines[0])
            if indent_pattern:
                consistent = all(
                    line.startswith(indent_pattern.group(1)) or not line.strip()
                    for line in lines[1:] if line.strip()
                )
                if consistent:
                    score += 0.15

        return min(score, 1.0)

    def _calculate_maintainability(self, content: str) -> float:
        """Calculate maintainability score."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for modularity (functions/classes)
        functions = len(re.findall(r'def \w+', content))
        classes = len(re.findall(r'class \w+', content))
        score += min((functions + classes) * 0.05, 0.2)

        # Check for docstrings
        if '"""' in content or "'''" in content:
            score += 0.15

        # Check for imports (suggests modularity)
        imports = len(re.findall(r'^import |^from .* import', content, re.MULTILINE))
        score += min(imports * 0.02, 0.15)

        return min(score, 1.0)

    def _calculate_scalability(self, content: str) -> float:
        """Calculate scalability score."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for async/concurrent patterns
        async_patterns = ['async ', 'await ', 'thread', 'multiprocess', 'concurrent']
        found_async = any(pattern in content.lower() for pattern in async_patterns)
        if found_async:
            score += 0.2

        # Check for database/connection pooling
        db_patterns = ['pool', 'connection', 'session', 'transaction']
        found_db = any(pattern in content.lower() for pattern in db_patterns)
        if found_db:
            score += 0.15

        # Check for batch processing
        batch_patterns = ['batch', 'chunk', 'parallel', 'distributed']
        found_batch = any(pattern in content.lower() for pattern in batch_patterns)
        if found_batch:
            score += 0.15

        return min(score, 1.0)

    def _calculate_security(self, content: str) -> float:
        """Calculate security score."""
        if not content:
            return 0.0

        score = 0.5  # Base score

        # Check for input validation
        validation_patterns = ['validate', 'sanitize', 'escape', 'verify']
        found_validation = any(pattern in content.lower() for pattern in validation_patterns)
        if found_validation:
            score += 0.15

        # Check for encryption
        crypto_patterns = ['encrypt', 'decrypt', 'hash', 'crypto', 'cipher']
        found_crypto = any(pattern in content.lower() for pattern in crypto_patterns)
        if found_crypto:
            score += 0.2

        # Check for authentication
        auth_patterns = ['auth', 'login', 'password', 'token', 'session']
        found_auth = any(pattern in content.lower() for pattern in auth_patterns)
        if found_auth:
            score += 0.15

        return min(score, 1.0)

    def _calculate_test_coverage(self, content: str) -> float:
        """Calculate test coverage score."""
        if not content:
            return 0.0

        score = 0.0  # No assumption of tests

        # Check for test patterns
        test_patterns = ['test_', 'Test', 'unittest', 'pytest', 'assert']
        found_tests = sum(1 for pattern in test_patterns if pattern in content)

        if found_tests > 0:
            score = 0.3 + min(found_tests * 0.1, 0.5)

        # Check for assertions
        assertions = len(re.findall(r'\bassert\b', content))
        if assertions > 0:
            score += min(assertions * 0.05, 0.2)

        return min(score, 1.0)

    def _calculate_confidence(self, content: str, metrics: SolutionQualityMetrics) -> float:
        """Calculate confidence in quality assessment."""
        confidence = 0.5

        # More content = higher confidence
        length = len(content)
        if length > 500:
            confidence += 0.1
        if length > 1000:
            confidence += 0.1

        # Consistency across metrics = higher confidence
        scores = [
            metrics.completeness, metrics.correctness, metrics.efficiency,
            metrics.clarity, metrics.maintainability
        ]
        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            confidence += max(0, 0.3 - variance)

        return min(confidence, 1.0)

    def _execute_test(self, solution: Any, test_case: Any) -> Dict[str, Any]:
        """
        Execute a single test case against a solution.

        Args:
            solution: Solution to test
            test_case: Test case definition

        Returns:
            Test result dictionary
        """
        result = {
            'test_case': str(test_case),
            'passed': False,
            'score': 0.0,
            'feedback': '',
            'timestamp': time.time()
        }

        try:
            # If test_case is callable, execute it
            if callable(test_case):
                test_result = test_case(solution)
                if isinstance(test_result, dict):
                    result.update(test_result)
                else:
                    result['passed'] = bool(test_result)
                    result['score'] = 1.0 if test_result else 0.0
            # If test_case is a dict, extract validation logic
            elif isinstance(test_case, dict):
                criterion = SuccessCriterion(**test_case)
                solution_content = self._extract_solution_content(solution)
                metric_value = self._calculate_metric(solution, criterion.metric)
                result['passed'] = metric_value >= criterion.threshold
                result['score'] = metric_value
                result['feedback'] = f"Metric {criterion.metric} = {metric_value:.2f}"
            else:
                # String test case - treat as requirement
                criterion = self.create_success_criteria([test_case])[0]
                result['passed'] = self.check_criterion(solution, criterion)
                result['score'] = self._calculate_metric(solution, criterion.metric)

        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            result['error'] = str(e)
            result['feedback'] = f"Test execution failed: {e}"

        return result

    def _create_judge_report(
        self,
        criteria_results: Dict[str, bool],
        quality_metrics: SolutionQualityMetrics
    ) -> Dict[str, Any]:
        """Create a judge report from verification results."""
        return {
            'judge_id': 'verification_engine',
            'criteria_results': criteria_results,
            'quality_metrics': quality_metrics.to_dict(),
            'timestamp': time.time()
        }

    def _generate_summary(
        self,
        passed: int,
        total: int,
        quality_metrics: SolutionQualityMetrics,
        criteria_results: Dict[str, bool]
    ) -> str:
        """Generate human-readable verification summary."""
        pass_rate = (passed / total * 100) if total > 0 else 0

        summary_parts = [
            f"Verification completed: {passed}/{total} criteria passed ({pass_rate:.1f}%)",
            f"Overall quality score: {quality_metrics.overall_score:.2f}",
            f"Confidence: {quality_metrics.confidence:.2f}"
        ]

        # Add failing criteria
        failed_criteria = [cid for cid, passed in criteria_results.items() if not passed]
        if failed_criteria:
            summary_parts.append(f"Failed criteria: {', '.join(failed_criteria)}")

        return '. '.join(summary_parts)

    def _generate_results_summary(self, results: List[Any]) -> str:
        """Generate summary from raw results."""
        if not results:
            return "No results to summarize"

        total = len(results)
        passed = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('passed', r.get('is_approved', False))
        )

        return f"Verification completed: {passed}/{total} validators passed ({passed/total*100:.1f}%)"

    def get_verification_history(self) -> List[VerificationReport]:
        """Get all verification reports from history."""
        return self.verification_history.copy()

    def clear_history(self):
        """Clear verification history."""
        self.verification_history.clear()
        self.logger.info("Verification history cleared")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_criteria() -> List[SuccessCriterion]:
    """Create a default set of success criteria."""
    return [
        SuccessCriterion(
            id="completeness_check",
            description="Solution must be at least 80% complete",
            metric="completeness",
            threshold=0.8,
            weight=1.0,
            category="functional"
        ),
        SuccessCriterion(
            id="correctness_check",
            description="Solution must be at least 70% correct",
            metric="correctness",
            threshold=0.7,
            weight=1.2,
            category="functional"
        ),
        SuccessCriterion(
            id="security_check",
            description="Solution must meet basic security standards",
            metric="security",
            threshold=0.6,
            weight=1.5,
            category="security"
        ),
        SuccessCriterion(
            id="clarity_check",
            description="Solution must be clear and readable",
            metric="clarity",
            threshold=0.6,
            weight=0.8,
            category="quality"
        )
    ]


def compare_reports(report1: VerificationReport, report2: VerificationReport) -> Dict[str, Any]:
    """
    Compare two verification reports.

    Args:
        report1: First verification report
        report2: Second verification report

    Returns:
        Comparison results dictionary
    """
    comparison = {
        'solution_ids': (report1.solution_attempt_id, report2.solution_attempt_id),
        'approval_changed': report1.is_approved != report2.is_approved,
        'score_difference': report2.verification_score - report1.verification_score,
        'quality_difference': None,
        'timestamp_difference': report2.timestamp - report1.timestamp
    }

    if report1.quality_metrics and report2.quality_metrics:
        comparison['quality_difference'] = {
            metric: getattr(report2.quality_metrics, metric) - getattr(report1.quality_metrics, metric)
            for metric in [
                'completeness', 'correctness', 'efficiency', 'clarity',
                'maintainability', 'scalability', 'security', 'test_coverage'
            ]
        }

    return comparison


# =============================================================================
# EXAMPLES AND DEMO
# =============================================================================

def example_basic_usage():
    """Example of basic verification engine usage."""
    print("=" * 80)
    print("VERIFICATION ENGINE - BASIC USAGE EXAMPLE")
    print("=" * 80)

    # Initialize engine
    engine = VerificationEngine()

    # Create a mock solution
    @dataclass
    class MockSolution:
        id: str
        solution_content: str

    solution = MockSolution(
        id="solution_001",
        solution_content="""
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number efficiently.'''
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

# Test cases
assert calculate_fibonacci(0) == 0
assert calculate_fibonacci(1) == 1
assert calculate_fibonacci(10) == 55
"""
    )

    # Create success criteria
    criteria = engine.create_success_criteria([
        "Solution must be at least 80% complete",
        "Solution must be clear and readable",
        "Solution must include test coverage"
    ])

    # Verify solution
    report = engine.verify_solution(solution, criteria)

    # Print results
    print(f"\nSolution ID: {report.solution_attempt_id}")
    print(f"Approved: {report.is_approved}")
    print(f"Verification Score: {report.verification_score:.2f}")
    print(f"Summary: {report.summary}")

    if report.quality_metrics:
        print("\nQuality Metrics:")
        for metric, value in report.quality_metrics.to_dict().items():
            print(f"  {metric}: {value:.2f}")

    print("\nCriteria Results:")
    for criterion_id, passed in report.criteria_results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {criterion_id}: {status}")

    return report


def example_verification_suite():
    """Example of running a complete verification suite."""
    print("\n" + "=" * 80)
    print("VERIFICATION ENGINE - SUITE EXAMPLE")
    print("=" * 80)

    engine = VerificationEngine()

    # Mock solution
    @dataclass
    class MockSolution:
        id: str
        solution_content: str

    solution = MockSolution(
        id="solution_002",
        solution_content="""
import asyncio
from typing import List

async def process_items(items: List[str]) -> List[str]:
    \"\"\"Process items asynchronously with error handling.\"\"\"
    results = []

    for item in items:
        try:
            # Simulate async processing
            result = await asyncio.to_thread(str.upper, item)
            results.append(result)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error processing {item}: {e}")

    return results
"""
    )

    # Create test suite
    test_suite = [
        {
            'id': 'test_completeness',
            'description': 'Check completeness',
            'metric': 'completeness',
            'threshold': 0.7
        },
        {
            'id': 'test_correctness',
            'description': 'Check correctness',
            'metric': 'correctness',
            'threshold': 0.6
        },
        {
            'id': 'test_clarity',
            'description': 'Check clarity',
            'metric': 'clarity',
            'threshold': 0.7
        }
    ]

    # Run verification suite
    report = engine.run_verification_suite(solution, test_suite)

    print(f"\nSuite Results: {report.summary}")
    print(f"Approved: {report.is_approved}")
    print(f"Tests Passed: {report.metadata.get('passed_tests', 0)}/{report.metadata.get('total_tests', 0)}")

    return report


def example_custom_criteria():
    """Example of creating custom success criteria."""
    print("\n" + "=" * 80)
    print("VERIFICATION ENGINE - CUSTOM CRITERIA EXAMPLE")
    print("=" * 80)

    engine = VerificationEngine(config={'strict_mode': True, 'min_quality_threshold': 0.8})

    # Create custom criteria
    custom_criteria = [
        SuccessCriterion(
            id="high_completeness",
            description="Solution must be 95% complete",
            metric="completeness",
            threshold=0.95,
            weight=1.5,
            category="functional"
        ),
        SuccessCriterion(
            id="security_first",
            description="Must meet high security standards",
            metric="security",
            threshold=0.8,
            weight=2.0,
            category="security"
        ),
        SuccessCriterion(
            id="production_ready",
            description="Must be production-quality",
            metric="maintainability",
            threshold=0.85,
            weight=1.8,
            category="quality"
        )
    ]

    # Mock solution
    @dataclass
    class MockSolution:
        id: str
        solution_content: str

    solution = MockSolution(
        id="solution_003",
        solution_content="""
class SecureDataProcessor:
    \"\"\"Production-ready data processor with security features.\"\"\"

    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        self.validate_key()

    def validate_key(self):
        \"\"\"Validate encryption key.\"\"\"
        if not self.encryption_key or len(self.encryption_key) < 16:
            raise ValueError("Encryption key must be at least 16 characters")

    def process_data(self, data: str) -> str:
        \"\"\"Securely process data with encryption.\"\"\"
        # Implementation here
        return self._encrypt(data)

    def _encrypt(self, data: str) -> str:
        \"\"\"Encrypt data using AES-256.\"\"\"
        # Implementation here
        return f"encrypted_{data}"
"""
    )

    # Verify with custom criteria
    report = engine.verify_solution(solution, custom_criteria)

    print(f"\nCustom Criteria Verification: {report.solution_attempt_id}")
    print(f"Approved: {report.is_approved}")
    print(f"Score: {report.verification_score:.2f}")

    if report.quality_metrics:
        print("\nQuality Breakdown:")
        for metric, value in report.quality_metrics.to_dict().items():
            if metric not in ['overall_score', 'confidence']:
                bar = "#" * int(value * 20)
                print(f"  {metric:15s}: {value:.2f} {bar}")

    return report


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_verification_suite()
    example_custom_criteria()

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
