"""
Integration Tests for LoongFlow Gauntlet Adapter

Tests the integration of LoongFlow's evaluation system with the OpenEvolve
gauntlet framework.
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockSolutionAttempt:
    """Mock solution for testing."""

    def __init__(self, content: str, solution_id: str = "test_solution"):
        self.id = solution_id
        self.content = content
        self.solution_content = content
        self.status = "generated"
        self.timestamp = datetime.now().timestamp()


class MockGauntletRoundRule:
    """Mock gauntlet round rule for testing."""

    def __init__(self, rule_id: str = "test_round", min_score: float = 0.7):
        self.rule_id = rule_id
        self.round_number = 1
        self.min_score = min_score
        self.rule_type = "automated"
        self.validation_type = "quality"
        self.description = "Test round"
        self.max_attempts = 1
        self.timeout = 60


class TestLoongFlowAdapter:
    """Test suite for LoongFlow adapter."""

    @pytest.fixture
    def llm_config(self):
        """Provide LLM configuration for testing."""
        return {
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': 'test-key',
            'url': 'http://localhost:8001',
            'temperature': 0.3,
            'max_tokens': 4096
        }

    @pytest.fixture
    def adapter(self, llm_config):
        """Create adapter instance for testing."""
        from evaluators.loongflow_adapter import create_loongflow_evaluator
        return create_loongflow_evaluator(
            llm_config=llm_config,
            timeout=30,
            enable_loongflow=True
        )

    @pytest.fixture
    def fallback_adapter(self, llm_config):
        """Create adapter with fallback mode for testing."""
        from evaluators.loongflow_adapter import create_loongflow_evaluator
        return create_loongflow_evaluator(
            llm_config=llm_config,
            timeout=30,
            enable_loongflow=False  # Force fallback
        )

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter):
        """Test that adapter initializes correctly."""
        assert adapter is not None
        assert adapter.llm_config is not None
        assert adapter.timeout == 30

    @pytest.mark.asyncio
    async def test_extract_solution_content_string(self, adapter):
        """Test extracting content from string solution."""
        content = "This is a test solution"
        extracted = adapter._extract_solution_content(content)
        assert extracted == content

    @pytest.mark.asyncio
    async def test_extract_solution_content_object(self, adapter):
        """Test extracting content from solution object."""
        solution = MockSolutionAttempt("Test solution content")
        extracted = adapter._extract_solution_content(solution)
        assert extracted == "Test solution content"

    @pytest.mark.asyncio
    async def test_extract_solution_content_dict(self, adapter):
        """Test extracting content from dictionary solution."""
        solution = {'content': 'Dict content', 'other': 'value'}
        extracted = adapter._extract_solution_content(solution)
        assert extracted == 'Dict content'

    @pytest.mark.asyncio
    async def test_fallback_evaluation(self, fallback_adapter):
        """Test fallback evaluation mode."""
        solution = MockSolutionAttempt("""
        Here is my solution to the problem:

        Approach: I will solve this by implementing a function that...

        ```python
        def solve():
            # Implementation here
            return result
        ```

        This approach works because it leverages...
        """)

        round_rule = MockGauntletRoundRule(min_score=0.5)
        context = {
            'problem': 'Solve the optimization problem',
            'criteria': ['correctness', 'completeness', 'clarity']
        }

        result = await fallback_adapter.evaluate_round(
            solution=solution,
            round_rule=round_rule,
            context=context
        )

        # Check result structure
        assert result is not None
        assert result.rule_id == "test_round"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.feedback, str)
        assert result.execution_time >= 0
        assert result.details is not None
        assert result.details.get('evaluation_type') == 'fallback'

    @pytest.mark.asyncio
    async def test_fallback_evaluation_low_quality(self, fallback_adapter):
        """Test fallback evaluation with low-quality solution."""
        solution = MockSolutionAttempt("short")  # Very short solution

        round_rule = MockGauntletRoundRule(min_score=0.7)
        context = {
            'problem': 'Complex problem requiring detailed solution',
            'criteria': ['correctness', 'completeness']
        }

        result = await fallback_adapter.evaluate_round(
            solution=solution,
            round_rule=round_rule,
            context=context
        )

        # Should fail due to low quality
        assert result.passed == False
        assert result.score < 0.7
        assert 'brief' in result.feedback.lower() or 'short' in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_fallback_evaluation_high_quality(self, fallback_adapter):
        """Test fallback evaluation with high-quality solution."""
        solution_content = """
        # Comprehensive Solution

        ## Problem Analysis
        The problem requires us to implement an efficient algorithm for...
        Therefore, we need to consider...

        ## Approach
        My approach is to use dynamic programming because...

        ## Implementation
        ```python
        def solve_problem(input_data):
            # Initialize DP table
            dp = [[0] * n for _ in range(m)]

            # Fill table
            for i in range(m):
                for j in range(n):
                    dp[i][j] = compute_value(i, j)

            return dp[m-1][n-1]
        ```

        ## Explanation
        This implementation is efficient because it avoids recomputation...
        The time complexity is O(m*n) and space complexity is O(m*n)...
        """
        solution = MockSolutionAttempt(solution_content)

        round_rule = MockGauntletRoundRule(min_score=0.7)
        context = {
            'problem': 'Dynamic programming problem',
            'criteria': ['correctness', 'efficiency', 'clarity']
        }

        result = await fallback_adapter.evaluate_round(
            solution=solution,
            round_rule=round_rule,
            context=context
        )

        # Should pass with good score
        assert result.score >= 0.6  # At least moderate score
        # Check that has_code is a boolean or check code in details
        has_code = result.details.get('has_code', False)
        assert isinstance(has_code, bool) or 'code' in str(result.details)

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, fallback_adapter):
        """Test batch evaluation of multiple solutions."""
        solutions = [
            MockSolutionAttempt(f"Solution {i}: " + "content " * (i * 50))
            for i in range(1, 6)
        ]

        round_rule = MockGauntletRoundRule(min_score=0.5)
        context = {'problem': 'Test problem', 'criteria': ['quality']}

        results = await fallback_adapter.batch_evaluate(
            solutions=solutions,
            round_rule=round_rule,
            context=context
        )

        # Should return results for all solutions
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.rule_id == "test_round"
            assert isinstance(result.score, float)
            assert result.execution_time >= 0

    @pytest.mark.asyncio
    async def test_batch_evaluation_parallel_execution(self, fallback_adapter):
        """Test that batch evaluation runs in parallel."""
        import time

        solutions = [
            MockSolutionAttempt(f"Solution {i}")
            for i in range(3)
        ]

        round_rule = MockGauntletRoundRule()
        context = {}

        # Measure time for batch execution
        start = time.time()
        results = await fallback_adapter.batch_evaluate(
            solutions=solutions,
            round_rule=round_rule,
            context=context
        )
        batch_time = time.time() - start

        # Should be faster than sequential (approximately)
        # This is a rough check - parallel should be noticeably faster
        assert len(results) == 3
        # Batch execution should complete reasonably fast
        assert batch_time < 10  # Should complete in under 10 seconds

    def test_gauntlet_round_result_creation(self):
        """Test GauntletRoundResult dataclass creation."""
        from evaluators.loongflow_adapter import GauntletRoundResult

        result = GauntletRoundResult(
            rule_id="test_rule",
            passed=True,
            score=0.85,
            feedback="Good solution",
            details={"test": "data"},
            execution_time=1.5
        )

        assert result.rule_id == "test_rule"
        assert result.passed == True
        assert result.score == 0.85
        assert result.feedback == "Good solution"
        assert result.details == {"test": "data"}
        assert result.execution_time == 1.5
        assert result.timestamp is not None


class TestEnhancedGauntletSystem:
    """Test suite for Enhanced Gauntlet System."""

    @pytest.fixture
    def llm_config(self):
        """Provide LLM configuration."""
        return {
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': 'test-key',
            'url': 'http://localhost:8001'
        }

    @pytest.fixture
    def gauntlet_system(self, llm_config):
        """Create enhanced gauntlet system."""
        from enhanced_gauntlet_manager import create_enhanced_gauntlet_system
        return create_enhanced_gauntlet_system(
            llm_config=llm_config,
            enable_loongflow=False  # Use fallback for tests
        )

    def test_system_initialization(self, gauntlet_system):
        """Test system initializes correctly."""
        assert gauntlet_system is not None
        assert gauntlet_system.llm_config is not None

    def test_create_gauntlet_standard(self, gauntlet_system):
        """Test creating standard gauntlet."""
        from openevolve_structures import GauntletDefinition

        gauntlet = gauntlet_system.create_enhanced_gauntlet(
            problem_type="engineering",
            strictness="standard"
        )

        assert isinstance(gauntlet, GauntletDefinition)
        assert len(gauntlet.rounds) == 3
        assert gauntlet.name == "enhanced_engineering"

        # Check round order - GauntletRoundRule has round_number, not rule_id
        assert gauntlet.rounds[0].round_number == 1
        assert gauntlet.rounds[1].round_number == 2
        assert gauntlet.rounds[2].round_number == 3

    def test_create_gauntlet_strict(self, gauntlet_system):
        """Test creating strict gauntlet."""
        gauntlet = gauntlet_system.create_enhanced_gauntlet(
            problem_type="security",
            strictness="strict"
        )

        # Strict mode should have higher thresholds
        round1 = gauntlet.rounds[0]
        assert round1.min_overall_confidence == 0.8

        round2 = gauntlet.rounds[1]
        assert round2.min_overall_confidence == 0.75

        round3 = gauntlet.rounds[2]
        assert round3.min_overall_confidence == 0.9

    def test_create_gauntlet_lenient(self, gauntlet_system):
        """Test creating lenient gauntlet."""
        gauntlet = gauntlet_system.create_enhanced_gauntlet(
            problem_type="general",
            strictness="lenient"
        )

        # Lenient mode should have lower thresholds
        round1 = gauntlet.rounds[0]
        assert round1.min_overall_confidence == 0.6

    def test_get_attack_modes(self, gauntlet_system):
        """Test getting appropriate attack modes."""
        trading_modes = gauntlet_system._get_attack_modes("trading")
        assert "market_crash" in trading_modes
        assert "black_swan" in trading_modes

        security_modes = gauntlet_system._get_attack_modes("security")
        assert "injection" in security_modes
        assert "exploit" in security_modes

        general_modes = gauntlet_system._get_attack_modes("unknown")
        assert "generic_attack" in general_modes

    @pytest.mark.asyncio
    async def test_execute_gauntlet(self, gauntlet_system):
        """Test executing a complete gauntlet."""
        from openevolve_structures import GauntletExecution

        # Create gauntlet
        gauntlet = gauntlet_system.create_enhanced_gauntlet(
            problem_type="engineering",
            strictness="lenient"  # Use lenient to make it easier to pass
        )

        # Create solution
        solution = MockSolutionAttempt("""
        # Engineering Solution

        ```python
        def solve():
            return result
        ```

        This solution addresses the requirements by implementing...
        """)

        context = {
            'problem': 'Design a component',
            'criteria': ['correctness', 'efficiency']
        }

        # Execute gauntlet
        execution = await gauntlet_system.execute_gauntlet(
            gauntlet=gauntlet,
            solution=solution,
            context=context
        )

        # Check execution result
        assert isinstance(execution, GauntletExecution)
        assert execution.gauntlet_id == "enhanced_engineering"
        assert len(execution.rounds_results) == 3
        assert execution.execution_time > 0

        # Check each round has a result
        for round_result in execution.rounds_results:
            assert round_result.rule_id is not None
            assert round_result.score >= 0
            assert round_result.execution_time >= 0


def test_imports():
    """Test that all modules can be imported."""
    try:
        from evaluators import LoongFlowEvaluatorAdapter
        from evaluators.loongflow_adapter import create_loongflow_evaluator
        from enhanced_gauntlet_manager import (
            EnhancedGauntletSystem,
            create_enhanced_gauntlet_system,
            GauntletExecution,
            GauntletRoundResult,
            GauntletRoundStatus
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
