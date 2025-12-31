"""
Enhanced Gauntlet Validator

Integrates RAGBits evaluation framework with the existing gauntlet system,
providing multi-dimensional scoring and validation.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    EvaluationMetricsCollector,
    MetricSet,
    MetricCategory,
    MetricType,
    MetricValue
)

logger = logging.getLogger(__name__)


class GauntletTestType(Enum):
    """Types of gauntlet tests"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    STRESS = "stress"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class MultiDimensionalScore:
    """
    Multi-dimensional score for gauntlet validation.

    Provides scoring across multiple dimensions rather than
    a single pass/fail result.
    """
    # Dimension scores (0-10 scale)
    functionality: float = 0.0
    performance: float = 0.0
    security: float = 0.0
    reliability: float = 0.0
    completeness: float = 0.0
    efficiency: float = 0.0
    maintainability: float = 0.0
    scalability: float = 0.0

    # Overall weighted score
    overall_score: float = 0.0

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0

    # Dimensions with issues
    critical_dimensions: List[str] = field(default_factory=list)

    # Detailed scores
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Timestamp
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def __post_init__(self):
        """Calculate overall score after initialization"""
        self.dimension_scores = {
            "functionality": self.functionality,
            "performance": self.performance,
            "security": self.security,
            "reliability": self.reliability,
            "completeness": self.completeness,
            "efficiency": self.efficiency,
            "maintainability": self.maintainability,
            "scalability": self.scalability
        }

        # Calculate weighted average (security and functionality weighted higher)
        weights = {
            "functionality": 1.2,
            "performance": 0.9,
            "security": 1.3,
            "reliability": 1.1,
            "completeness": 1.0,
            "efficiency": 0.9,
            "maintainability": 0.8,
            "scalability": 0.8
        }

        weighted_sum = sum(score * weights[dim] for dim, score in self.dimension_scores.items())
        total_weight = sum(weights.values())

        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Identify critical dimensions (below 5/10)
        self.critical_dimensions = [
            dim for dim, score in self.dimension_scores.items()
            if score < 5.0
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dimension_scores": self.dimension_scores,
            "overall_score": self.overall_score,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_total": self.tests_total,
            "critical_dimensions": self.critical_dimensions,
            "timestamp": self.timestamp,
            "pass_rate": self.tests_passed / self.tests_total if self.tests_total > 0 else 0.0
        }

    def get_verdict(self) -> str:
        """Get verdict based on scores"""
        if self.overall_score >= 8.0 and not self.critical_dimensions:
            return "EXCELLENT"
        elif self.overall_score >= 6.5 and len(self.critical_dimensions) <= 1:
            return "GOOD"
        elif self.overall_score >= 5.0:
            return "ACCEPTABLE"
        else:
            return "POOR"


@dataclass
class GauntletTestResult:
    """Result of a single gauntlet test"""
    test_name: str
    test_type: GauntletTestType
    result: TestResult
    score: float  # 0-10 scale
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "result": self.result.value,
            "score": self.score,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class GauntletValidationResult:
    """Complete gauntlet validation result"""
    artifact_id: str
    artifact_type: str
    multi_dimensional_score: MultiDimensionalScore
    test_results: List[GauntletTestResult]
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "score": self.multi_dimensional_score.to_dict(),
            "test_results": [tr.to_dict() for tr in self.test_results],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "verdict": self.multi_dimensional_score.get_verdict()
        }


class EnhancedGauntletValidator:
    """
    Enhanced gauntlet validator with multi-dimensional scoring.

    Integrates with RAGBits metrics system to provide comprehensive
    validation beyond simple pass/fail.

    Usage:
        validator = EnhancedGauntletValidator(metrics_collector)

        # Run validation
        result = await validator.validate_solution(
            artifact_id="art_123",
            solution_text="...",
            test_types=[GauntletTestType.FUNCTIONAL, GauntletTestType.SECURITY]
        )

        # Check verdict
        verdict = result.score.get_verdict()
        print(f"Validation verdict: {verdict}")
    """

    def __init__(
        self,
        metrics_collector: EvaluationMetricsCollector,
        test_registry: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize enhanced gauntlet validator.

        Args:
            metrics_collector: Metrics collector for integration
            test_registry: Optional custom test functions
        """
        self.metrics_collector = metrics_collector
        self.test_registry = test_registry or {}

        # Built-in test functions
        self._register_builtin_tests()

        logger.info("EnhancedGauntletValidator initialized")

    def _register_builtin_tests(self):
        """Register built-in test functions"""
        # Functional tests
        self.test_registry["functional_requirements_coverage"] = self._test_requirements_coverage
        self.test_registry["functional_edge_cases"] = self._test_edge_case_handling

        # Performance tests
        self.test_registry["performance_time_complexity"] = self._test_time_complexity
        self.test_registry["performance_resource_usage"] = self._test_resource_usage

        # Security tests
        self.test_registry["security_vulnerabilities"] = self._test_vulnerabilities
        self.test_registry["security_input_validation"] = self._test_input_validation

        # Reliability tests
        self.test_registry["reliability_error_handling"] = self._test_error_handling
        self.test_registry["reliability_fault_tolerance"] = self._test_fault_tolerance

    async def validate_solution(
        self,
        artifact_id: str,
        solution_text: str,
        test_types: Optional[List[GauntletTestType]] = None,
        requirements: Optional[List[str]] = None,
        custom_tests: Optional[List[str]] = None
    ) -> GauntletValidationResult:
        """
        Run gauntlet validation on a solution.

        Args:
            artifact_id: Artifact to validate
            solution_text: Solution content
            test_types: Types of tests to run (default: all)
            requirements: Optional requirements list
            custom_tests: Optional custom test names

        Returns:
            Validation result
        """
        import time
        start_time = time.time()

        # Default to all test types
        if test_types is None:
            test_types = list(GauntletTestType)

        logger.info(
            f"Starting gauntlet validation for {artifact_id} "
            f"with {len(test_types)} test types"
        )

        # Run tests
        test_results = []

        for test_type in test_types:
            type_results = await self._run_tests_for_type(
                test_type,
                artifact_id,
                solution_text,
                requirements or []
            )
            test_results.extend(type_results)

        # Run custom tests if provided
        if custom_tests:
            for test_name in custom_tests:
                if test_name in self.test_registry:
                    result = await self._run_single_test(
                        test_name,
                        self.test_registry[test_name],
                        artifact_id,
                        solution_text,
                        requirements or []
                    )
                    test_results.append(result)

        # Calculate multi-dimensional score
        multi_score = self._calculate_multi_dimensional_score(test_results)

        # Create result
        validation_result = GauntletValidationResult(
            artifact_id=artifact_id,
            artifact_type="solution",
            multi_dimensional_score=multi_score,
            test_results=test_results,
            metadata={
                "duration_ms": (time.time() - start_time) * 1000,
                "test_types_run": [tt.value for tt in test_types],
                "requirements_count": len(requirements or [])
            }
        )

        # Store metrics
        await self._store_validation_metrics(validation_result)

        logger.info(
            f"Validation complete for {artifact_id}: "
            f"verdict={multi_score.get_verdict()}, "
            f"score={multi_score.overall_score:.2f}"
        )

        return validation_result

    async def _run_tests_for_type(
        self,
        test_type: GauntletTestType,
        artifact_id: str,
        solution_text: str,
        requirements: List[str]
    ) -> List[GauntletTestResult]:
        """Run all tests for a specific type"""
        results = []

        # Get test functions for this type
        test_functions = {
            GauntletTestType.FUNCTIONAL: [
                "functional_requirements_coverage",
                "functional_edge_cases"
            ],
            GauntletTestType.PERFORMANCE: [
                "performance_time_complexity",
                "performance_resource_usage"
            ],
            GauntletTestType.SECURITY: [
                "security_vulnerabilities",
                "security_input_validation"
            ],
            GauntletTestType.RELIABILITY: [
                "reliability_error_handling",
                "reliability_fault_tolerance"
            ]
        }

        test_names = test_functions.get(test_type, [])

        for test_name in test_names:
            if test_name in self.test_registry:
                result = await self._run_single_test(
                    test_name,
                    self.test_registry[test_name],
                    artifact_id,
                    solution_text,
                    requirements
                )
                results.append(result)

        return results

    async def _run_single_test(
        self,
        test_name: str,
        test_func: Callable,
        artifact_id: str,
        solution_text: str,
        requirements: List[str]
    ) -> GauntletTestResult:
        """Run a single test function"""
        import time

        start_time = time.time()

        try:
            result_data = await test_func(solution_text, requirements)
            duration = (time.time() - start_time) * 1000

            return GauntletTestResult(
                test_name=test_name,
                test_type=self._get_test_type_for_name(test_name),
                result=TestResult(result_data.get("result", "passed")),
                score=result_data.get("score", 0.0),
                duration_ms=duration,
                message=result_data.get("message", ""),
                details=result_data.get("details", {})
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Test {test_name} failed: {e}")

            return GauntletTestResult(
                test_name=test_name,
                test_type=self._get_test_type_for_name(test_name),
                result=TestResult.ERROR,
                score=0.0,
                duration_ms=duration,
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def _calculate_multi_dimensional_score(
        self,
        test_results: List[GauntletTestResult]
    ) -> MultiDimensionalScore:
        """Calculate multi-dimensional score from test results"""
        # Group test results by dimension
        dimension_scores = {
            "functionality": [],
            "performance": [],
            "security": [],
            "reliability": [],
            "completeness": [],
            "efficiency": [],
            "maintainability": [],
            "scalability": []
        }

        tests_passed = 0
        tests_failed = 0

        for result in test_results:
            # Map test type to dimension
            dimension = self._map_test_type_to_dimension(result.test_type)

            if dimension:
                dimension_scores[dimension].append(result.score)

            if result.result == TestResult.PASSED:
                tests_passed += 1
            else:
                tests_failed += 1

        # Calculate average score per dimension
        final_scores = {}
        for dim, scores in dimension_scores.items():
            final_scores[dim] = sum(scores) / len(scores) if scores else 0.0

        return MultiDimensionalScore(
            functionality=final_scores.get("functionality", 0.0),
            performance=final_scores.get("performance", 0.0),
            security=final_scores.get("security", 0.0),
            reliability=final_scores.get("reliability", 0.0),
            completeness=final_scores.get("completeness", 0.0),
            efficiency=final_scores.get("efficiency", 0.0),
            maintainability=final_scores.get("maintainability", 0.0),
            scalability=final_scores.get("scalability", 0.0),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_total=len(test_results)
        )

    def _map_test_type_to_dimension(
        self,
        test_type: GauntletTestType
    ) -> Optional[str]:
        """Map test type to score dimension"""
        mapping = {
            GauntletTestType.FUNCTIONAL: "functionality",
            GauntletTestType.PERFORMANCE: "performance",
            GauntletTestType.SECURITY: "security",
            GauntletTestType.RELIABILITY: "reliability",
            GauntletTestType.USABILITY: "maintainability",
            GauntletTestType.COMPATIBILITY: "maintainability",
            GauntletTestType.STRESS: "scalability"
        }
        return mapping.get(test_type)

    def _get_test_type_for_name(self, test_name: str) -> GauntletTestType:
        """Get test type from test name"""
        if "functional" in test_name:
            return GauntletTestType.FUNCTIONAL
        elif "performance" in test_name:
            return GauntletTestType.PERFORMANCE
        elif "security" in test_name:
            return GauntletTestType.SECURITY
        elif "reliability" in test_name:
            return GauntletTestType.RELIABILITY
        else:
            return GauntletTestType.FUNCTIONAL

    async def _store_validation_metrics(
        self,
        validation_result: GauntletValidationResult
    ):
        """Store validation metrics in metrics collector"""
        metric_set = MetricSet(
            artifact_id=validation_result.artifact_id,
            artifact_type=validation_result.artifact_type,
            sub_problem_id=validation_result.metadata.get("sub_problem_id"),
            workflow_stage="gauntlet_validation"
        )

        # Add scores as metrics
        score = validation_result.multi_dimensional_score

        metric_set.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=score.functionality,
            category=MetricCategory.QUALITY
        ))

        metric_set.add_metric(MetricValue(
            metric_type=MetricType.RESOURCE_USAGE,
            value=score.performance,
            category=MetricCategory.PERFORMANCE
        ))

        metric_set.add_metric(MetricValue(
            metric_type=MetricType.SECURITY_SCORE,
            value=score.security,
            category=MetricCategory.SECURITY
        ))

        metric_set.add_metric(MetricValue(
            metric_type=MetricType.FAULT_TOLERANCE,
            value=score.reliability,
            category=MetricCategory.RELIABILITY
        ))

        metric_set.add_metric(MetricValue(
            metric_type=MetricType.OPTIMIZATION_SCORE,
            value=score.efficiency,
            category=MetricCategory.EFFICIENCY
        ))

        await self.metrics_collector.collect_metrics(metric_set)

    # Built-in test implementations
    async def _test_requirements_coverage(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test requirements coverage"""
        # Simple keyword-based coverage check
        # In production, this would use more sophisticated analysis
        covered = 0

        for req in requirements:
            # Check if requirement keywords appear in solution
            keywords = req.lower().split()
            if any(kw in solution_text.lower() for kw in keywords if len(kw) > 3):
                covered += 1

        coverage = covered / len(requirements) if requirements else 1.0
        score = coverage * 10

        return {
            "result": "passed" if coverage >= 0.8 else "warning",
            "score": score,
            "message": f"Requirements coverage: {coverage:.1%}",
            "details": {"covered": covered, "total": len(requirements)}
        }

    async def _test_edge_case_handling(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test edge case handling"""
        # Check for common edge case handling patterns
        edge_case_keywords = [
            "null", "none", "empty", "zero", "invalid",
            "error", "exception", "timeout", "boundary"
        ]

        mentions = sum(
            1 for kw in edge_case_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / len(edge_case_keywords)) * 10)

        return {
            "result": "passed" if mentions >= 4 else "warning",
            "score": score,
            "message": f"Edge cases mentioned: {mentions}",
            "details": {"mentions": mentions}
        }

    async def _test_time_complexity(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test time complexity considerations"""
        # Look for complexity discussions
        complexity_keywords = [
            "o(n)", "o(log n)", "o(1)", "complexity",
            "efficient", "optimize", "scalable"
        ]

        mentions = sum(
            1 for kw in complexity_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / 3) * 10)

        return {
            "result": "passed" if mentions >= 2 else "warning",
            "score": score,
            "message": f"Complexity analysis: {mentions} mentions",
            "details": {"mentions": mentions}
        }

    async def _test_resource_usage(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test resource usage considerations"""
        resource_keywords = [
            "memory", "cpu", "storage", "cache",
            "resource", "pool", "limit"
        ]

        mentions = sum(
            1 for kw in resource_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / 3) * 10)

        return {
            "result": "passed" if mentions >= 2 else "warning",
            "score": score,
            "message": f"Resource usage: {mentions} mentions",
            "details": {"mentions": mentions}
        }

    async def _test_vulnerabilities(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test for security vulnerabilities"""
        # Check for common vulnerability patterns
        vulnerability_patterns = [
            "sql injection", "xss", "csrf", "buffer overflow",
            "insecure", "plaintext", "hardcoded"
        ]

        # These should NOT appear (unless discussing mitigation)
        suspicious = [
            pattern for pattern in vulnerability_patterns
            if pattern in solution_text.lower()
        ]

        # If they appear, check if it's about mitigation
        issues = []
        for pattern in suspicious:
            context_lower = solution_text.lower()
            pattern_idx = context_lower.find(pattern)

            # Check 100 chars before and after
            start = max(0, pattern_idx - 100)
            end = min(len(solution_text), pattern_idx + 100)
            context = solution_text[start:end]

            if not any(kw in context.lower() for kw in ["prevent", "mitigate", "avoid", "protect"]):
                issues.append(pattern)

        score = max(0, 10 - len(issues) * 5)

        return {
            "result": "failed" if issues else "passed",
            "score": score,
            "message": f"Potential vulnerabilities: {len(issues)}",
            "details": {"issues": issues}
        }

    async def _test_input_validation(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test input validation"""
        validation_keywords = [
            "validate", "sanitize", "check", "verify",
            "input", "parameter", "filter"
        ]

        mentions = sum(
            1 for kw in validation_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / 3) * 10)

        return {
            "result": "passed" if mentions >= 2 else "warning",
            "score": score,
            "message": f"Input validation: {mentions} mentions",
            "details": {"mentions": mentions}
        }

    async def _test_error_handling(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test error handling"""
        error_keywords = [
            "try", "catch", "except", "error", "exception",
            "handle", "recover", "retry"
        ]

        mentions = sum(
            1 for kw in error_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / 4) * 10)

        return {
            "result": "passed" if mentions >= 3 else "warning",
            "score": score,
            "message": f"Error handling: {mentions} mentions",
            "details": {"mentions": mentions}
        }

    async def _test_fault_tolerance(
        self,
        solution_text: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Test fault tolerance"""
        ft_keywords = [
            "fallback", "redundant", "backup", "recover",
            "tolerance", "graceful", "degrade"
        ]

        mentions = sum(
            1 for kw in ft_keywords
            if kw in solution_text.lower()
        )

        score = min(10.0, (mentions / 2) * 10)

        return {
            "result": "passed" if mentions >= 1 else "warning",
            "score": score,
            "message": f"Fault tolerance: {mentions} mentions",
            "details": {"mentions": mentions}
        }
