"""
Comprehensive Testing and Validation Framework for OpenEvolve Gauntlet

Provides testing utilities, validation helpers, and quality assurance
tools for the Gauntlet system.

Key Features:
- Unit test utilities
- Integration test helpers
- Validation helpers
- Test data generators
- Performance testing tools
- Contract testing
- Property-based testing
"""

from typing import Dict, List, Any, Optional, Callable, Type, TypeVar, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import json
import random
import string
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TestCase:
    """A test case"""
    name: str
    description: str
    test_func: Callable
    expected_result: Any = None
    should_fail: bool = False
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a test case"""
    test_case: TestCase
    success: bool
    actual_result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestSuite:
    """A collection of test cases"""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Report from validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestDataGenerator:
    """
    Generates test data for Gauntlet components.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_problem(
        self,
        complexity: str = "medium",
        domain: str = "general"
    ) -> Dict[str, Any]:
        """Generate a test problem"""
        complexity_levels = {
            "simple": {"requirements": 1, "subproblems": 0},
            "easy": {"requirements": 2, "subproblems": 1},
            "medium": {"requirements": 4, "subproblems": 2},
            "hard": {"requirements": 6, "subproblems": 3},
            "complex": {"requirements": 8, "subproblems": 4},
        }

        level = complexity_levels.get(complexity, complexity_levels["medium"])

        problem = {
            "id": f"test_problem_{self._random_string(8)}",
            "statement": f"Test {complexity} problem in {domain} domain",
            "requirements": [f"Requirement {i+1}" for i in range(level["requirements"])],
            "domain": domain,
            "complexity": complexity,
            "estimated_effort_hours": level["requirements"] * 2,
        }

        # Add subproblems if needed
        if level["subproblems"] > 0:
            problem["subproblems"] = [
                self.generate_problem("simple", domain)
                for _ in range(level["subproblems"])
            ]

        return problem

    def generate_solution(
        self,
        problem: Dict[str, Any],
        success: bool = True,
        score: float = 0.8
    ) -> Dict[str, Any]:
        """Generate a test solution"""
        return {
            "problem_id": problem.get("id", "unknown"),
            "success": success,
            "score": score if success else random.uniform(0.0, 0.5),
            "solution": f"Solution for {problem.get('id', 'unknown')}",
            "confidence": score if success else random.uniform(0.0, 0.5),
            "team_id": f"team_{self._random_string(4)}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def generate_team_performance(
        self,
        team_count: int = 5,
        problem_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate team performance data"""
        domains = ["web", "ml", "data", "security", "general"]
        performances = []

        for team_id in range(team_count):
            for problem_id in range(problem_count):
                performances.append({
                    "team_id": f"team_{team_id}",
                    "problem_id": f"problem_{problem_id}",
                    "domain": random.choice(domains),
                    "difficulty": random.randint(1, 5),
                    "success": random.random() > 0.3,
                    "score": random.uniform(0.5, 1.0),
                    "execution_time": random.uniform(50, 500),
                })

        return performances

    def generate_decomposition_tree(
        self,
        depth: int = 3,
        branching_factor: int = 2
    ) -> Dict[str, Any]:
        """Generate a test decomposition tree"""
        def create_node(current_depth: int) -> Dict[str, Any]:
            if current_depth >= depth:
                return {
                    "id": f"node_{self._random_string(8)}",
                    "statement": "Atomic problem",
                    "atomic": True,
                }

            return {
                "id": f"node_{self._random_string(8)}",
                "statement": f"Problem at depth {current_depth}",
                "atomic": False,
                "subproblems": [
                    create_node(current_depth + 1)
                    for _ in range(branching_factor)
                ],
            }

        return create_node(0)

    def _random_string(self, length: int) -> str:
        """Generate random string"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class TestRunner:
    """
    Runs test cases and generates reports.
    """

    def __init__(self):
        self.results: List[TestResult] = []

    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        logger.info(f"Running test: {test_case.name}")
        start_time = datetime.utcnow()

        try:
            # Run test with timeout
            result = await asyncio.wait_for(
                test_case.test_func(),
                timeout=test_case.timeout_seconds
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Check if result matches expectation
            if test_case.should_fail:
                success = False  # Expected to fail but succeeded
                error = "Test should have failed but succeeded"
            elif test_case.expected_result is not None:
                success = result == test_case.expected_result
                error = None if success else f"Expected {test_case.expected_result}, got {result}"
            else:
                success = True
                error = None

            return TestResult(
                test_case=test_case,
                success=success,
                actual_result=result,
                error=error,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return TestResult(
                test_case=test_case,
                success=False,
                error=f"Test timed out after {test_case.timeout_seconds}s",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            success = test_case.should_fail

            return TestResult(
                test_case=test_case,
                success=success,
                error=str(e) if not test_case.should_fail else None,
                execution_time=execution_time
            )

    async def run_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """Run all tests in a suite"""
        logger.info(f"Running test suite: {test_suite.name}")
        results = []

        # Run setup
        if test_suite.setup_func:
            try:
                await test_suite.setup_func()
            except Exception as e:
                logger.error(f"Setup failed: {e}")
                return results

        # Run all test cases
        for test_case in test_suite.test_cases:
            result = await self.run_test_case(test_case)
            results.append(result)
            self.results.append(result)

        # Run teardown
        if test_suite.teardown_func:
            try:
                await test_suite.teardown_func()
            except Exception as e:
                logger.error(f"Teardown failed: {e}")

        return results

    def generate_report(self, results: List[TestResult] = None) -> Dict[str, Any]:
        """Generate test report"""
        if results is None:
            results = self.results

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests

        total_time = sum(r.execution_time for r in results)

        failed_test_names = [r.test_case.name for r in results if not r.success]

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / total_tests if total_tests > 0 else 0,
            "failed_tests": failed_test_names,
            "timestamp": datetime.utcnow().isoformat(),
        }


class ValidationHelper:
    """
    Helper methods for validating Gauntlet components.
    """

    @staticmethod
    def validate_problem(problem: Dict[str, Any]) -> ValidationReport:
        """Validate problem structure"""
        report = ValidationReport(is_valid=True)
        required_fields = ["id", "statement"]

        # Check required fields
        for field in required_fields:
            if field not in problem:
                report.errors.append(f"Missing required field: {field}")
                report.is_valid = False

        # Validate types
        if "id" in problem and not isinstance(problem["id"], str):
            report.errors.append("Field 'id' must be a string")
            report.is_valid = False

        if "statement" in problem and not isinstance(problem["statement"], str):
            report.errors.append("Field 'statement' must be a string")
            report.is_valid = False

        # Validate subproblems if present
        if "subproblems" in problem:
            if not isinstance(problem["subproblems"], list):
                report.errors.append("Field 'subproblems' must be a list")
                report.is_valid = False
            else:
                for i, subproblem in enumerate(problem["subproblems"]):
                    sub_report = ValidationHelper.validate_problem(subproblem)
                    if not sub_report.is_valid:
                        report.errors.extend([
                            f"subproblems[{i}]: {error}"
                            for error in sub_report.errors
                        ])
                        report.is_valid = False

        return report

    @staticmethod
    def validate_solution(solution: Dict[str, Any]) -> ValidationReport:
        """Validate solution structure"""
        report = ValidationReport(is_valid=True)
        required_fields = ["problem_id", "success"]

        for field in required_fields:
            if field not in solution:
                report.errors.append(f"Missing required field: {field}")
                report.is_valid = False

        # Validate types
        if "problem_id" in solution and not isinstance(solution["problem_id"], str):
            report.errors.append("Field 'problem_id' must be a string")
            report.is_valid = False

        if "success" in solution and not isinstance(solution["success"], bool):
            report.errors.append("Field 'success' must be a boolean")
            report.is_valid = False

        if "score" in solution:
            score = solution["score"]
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                report.errors.append("Field 'score' must be a number between 0 and 1")
                report.is_valid = False

        return report

    @staticmethod
    def validate_decomposition_tree(tree: Dict[str, Any]) -> ValidationReport:
        """Validate decomposition tree structure"""
        report = ValidationReport(is_valid=True)

        # Check required fields
        if "id" not in tree:
            report.errors.append("Missing required field: id")
            report.is_valid = False

        if "statement" not in tree:
            report.errors.append("Missing required field: statement")
            report.is_valid = False

        # Validate atomic flag
        if "atomic" in tree and not isinstance(tree["atomic"], bool):
            report.errors.append("Field 'atomic' must be a boolean")
            report.is_valid = False

        # Validate subproblems
        if "subproblems" in tree:
            if not isinstance(tree["subproblems"], list):
                report.errors.append("Field 'subproblems' must be a list")
                report.is_valid = False
            else:
                for i, subproblem in enumerate(tree["subproblems"]):
                    sub_report = ValidationHelper.validate_decomposition_tree(subproblem)
                    if not sub_report.is_valid:
                        report.errors.extend([
                            f"subproblems[{i}]: {error}"
                            for error in sub_report.errors
                        ])
                        report.is_valid = False

        return report

    @staticmethod
    def validate_checkpoint_state(state: Any) -> ValidationReport:
        """Validate checkpoint state"""
        report = ValidationReport(is_valid=True)

        if state is None:
            report.errors.append("Checkpoint state is None")
            report.is_valid = False
            return report

        # Check if state has required attributes
        required_attrs = ["problem", "context"]
        for attr in required_attrs:
            if not hasattr(state, attr):
                report.errors.append(f"Missing required attribute: {attr}")
                report.is_valid = False

        return report


class PerformanceTestHelper:
    """
    Helper methods for performance testing.
    """

    @staticmethod
    async def benchmark_function(
        func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark a function"""
        import time

        # Warmup
        for _ in range(warmup_iterations):
            await func()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        sorted_times = sorted(times)

        return {
            "iterations": iterations,
            "total_time_ms": sum(times),
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": sorted_times[0],
            "max_time_ms": sorted_times[-1],
            "p50_time_ms": sorted_times[int(len(times) * 0.5)],
            "p90_time_ms": sorted_times[int(len(times) * 0.9)],
            "p95_time_ms": sorted_times[int(len(times) * 0.95)],
            "p99_time_ms": sorted_times[int(len(times) * 0.99)],
        }

    @staticmethod
    async def load_test(
        func: Callable,
        concurrent_requests: int = 10,
        total_requests: int = 100
    ) -> Dict[str, Any]:
        """Perform load testing"""
        import time

        start_time = time.time()

        # Create batches of concurrent requests
        batch_size = concurrent_requests
        batches = (total_requests + batch_size - 1) // batch_size

        all_times = []

        for batch in range(batches):
            batch_start = time.time()

            # Run concurrent requests
            tasks = [func() for _ in range(batch_size)]
            await asyncio.gather(*tasks, return_exceptions=True)

            batch_time = time.time() - batch_start
            all_times.append(batch_time)

        total_time = time.time() - start_time

        return {
            "total_requests": total_requests,
            "concurrent_requests": concurrent_requests,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time,
            "avg_batch_time_seconds": sum(all_times) / len(all_times),
        }


class ContractTestHelper:
    """
    Helper methods for contract testing.
    """

    @staticmethod
    def verify_api_contract(
        actual_response: Dict[str, Any],
        expected_fields: List[str]
    ) -> ValidationReport:
        """Verify API response matches expected contract"""
        report = ValidationReport(is_valid=True)

        for field in expected_fields:
            if field not in actual_response:
                report.errors.append(f"Missing expected field: {field}")
                report.is_valid = False

        return report

    @staticmethod
    def verify_type_contract(
        data: Dict[str, Any],
        type_contract: Dict[str, Type]
    ) -> ValidationReport:
        """Verify data matches type contract"""
        report = ValidationReport(is_valid=True)

        for field, expected_type in type_contract.items():
            if field not in data:
                report.errors.append(f"Missing field: {field}")
                report.is_valid = False
            elif not isinstance(data[field], expected_type):
                report.errors.append(
                    f"Field '{field}' has wrong type. "
                    f"Expected {expected_type}, got {type(data[field])}"
                )
                report.is_valid = False

        return report


# Convenience functions
def create_test_generator(seed: Optional[int] = None) -> TestDataGenerator:
    """Create a test data generator"""
    return TestDataGenerator(seed)


def create_test_runner() -> TestRunner:
    """Create a test runner"""
    return TestRunner()


# Example usage
async def demo_testing():
    """Demonstration of testing framework"""

    print("\n" + "=" * 60)
    print("Gauntlet Testing Framework Demo")
    print("=" * 60)

    # Example 1: Test data generation
    print("\n1. Test Data Generation:")
    generator = TestDataGenerator(seed=42)

    problem = generator.generate_problem("medium", "web")
    print(f"   Generated problem: {problem['id']}")
    print(f"   Requirements: {len(problem['requirements'])}")

    solution = generator.generate_solution(problem, success=True, score=0.85)
    print(f"   Generated solution: {solution['problem_id']}")
    print(f"   Score: {solution['score']}")

    # Example 2: Test case execution
    print("\n2. Test Case Execution:")

    async def test_addition():
        return 2 + 2

    async def test_failing():
        raise ValueError("This test fails")

    test_case1 = TestCase(
        name="test_addition",
        description="Test that 2 + 2 = 4",
        test_func=test_addition,
        expected_result=4
    )

    test_case2 = TestCase(
        name="test_failing",
        description="Test that should fail",
        test_func=test_failing,
        should_fail=True
    )

    runner = TestRunner()
    result1 = await runner.run_test_case(test_case1)
    result2 = await runner.run_test_case(test_case2)

    print(f"   Test 1: {result1.test_case.name} - {'PASS' if result1.success else 'FAIL'}")
    print(f"   Test 2: {result2.test_case.name} - {'PASS' if result2.success else 'FAIL'}")

    # Example 3: Validation
    print("\n3. Validation:")
    valid_problem = {
        "id": "problem_123",
        "statement": "Test problem",
    }
    invalid_problem = {
        "id": "problem_456",
        # Missing statement
    }

    report1 = ValidationHelper.validate_problem(valid_problem)
    report2 = ValidationHelper.validate_problem(invalid_problem)

    print(f"   Valid problem: {report1.is_valid}")
    print(f"   Invalid problem: {report2.is_valid}")
    if not report2.is_valid:
        print(f"   Errors: {report2.errors}")

    # Example 4: Test report
    print("\n4. Test Report:")
    report = runner.generate_report()
    print(f"   Total tests: {report['total_tests']}")
    print(f"   Passed: {report['passed_tests']}")
    print(f"   Failed: {report['failed_tests']}")
    print(f"   Pass rate: {report['pass_rate']:.1%}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_testing())
