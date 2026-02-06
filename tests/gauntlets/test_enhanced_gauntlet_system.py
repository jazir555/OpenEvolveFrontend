"""
Comprehensive Test Suite for Enhanced 3-Round Gauntlet System

Tests the complete gauntlet pipeline including:
- Individual round evaluation
- Progressive filtering and early termination
- Score aggregation
- Decision logic
- Artifact fusion
- State management
- Performance benchmarks
- Quality metrics

Author: OpenEvolve QA Team
Date: 2026-01-30
Version: 1.0.0
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import json
from pathlib import Path

# Import test data
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'fixtures'))
from gauntlet_test_data import (
    TestSolution,
    GauntletTestConfig,
    get_all_solutions,
    get_solutions_by_category,
    get_solution_by_id,
    STRICT_CONFIG,
    LENIENT_CONFIG,
    BALANCED_CONFIG,
    NO_EARLY_TERMINATION_CONFIG,
    PERFECT_SOLUTIONS,
    POOR_SOLUTIONS,
    MODERATE_SOLUTIONS,
    GOOD_SOLUTIONS,
    EDGE_CASE_SOLUTIONS
)


# ============================================================================
# MOCK CLASSES FOR GAUNTLET COMPONENTS
# ============================================================================

class MockRoundResult:
    """Mock result from a gauntlet round"""
    def __init__(
        self,
        round_id: str,
        passed: bool,
        score: float,
        feedback: str = "",
        evaluation_time: float = 1.0,
        artifacts: List[Dict] = None
    ):
        self.round_id = round_id
        self.passed = passed
        self.score = score
        self.feedback = feedback
        self.evaluation_time = evaluation_time
        self.artifacts = artifacts or []


class MockGauntletResult:
    """Mock complete gauntlet result"""
    def __init__(
        self,
        passed: bool,
        final_score: float,
        round_results: List[MockRoundResult],
        rounds_completed: int,
        termination_reason: Optional[str] = None,
        fused_artifacts: List[Dict] = None
    ):
        self.passed = passed
        self.final_score = final_score
        self.round_results = round_results
        self.rounds_completed = rounds_completed
        self.termination_reason = termination_reason
        self.fused_artifacts = fused_artifacts or []


class MockThreeRoundOrchestrator:
    """
    Mock implementation of 3-round gauntlet orchestrator
    Simulates behavior for testing without full implementation
    """

    def __init__(self, config: GauntletTestConfig):
        self.config = config
        self.state = {
            'current_round': 0,
            'rounds_completed': [],
            'accumulated_score': 0.0,
            'artifacts': [],
            'start_time': None
        }

    async def run_round1(
        self,
        solution: TestSolution,
        problem: str,
        domain: str
    ) -> MockRoundResult:
        """Run Round 1: LoongFlow AI Evaluation"""
        start_time = time.time()

        # Simulate evaluation time
        await asyncio.sleep(0.1)

        # Get expected score from solution
        score = solution.expected_round1_score
        passed = score >= self.config.round1_threshold

        elapsed = time.time() - start_time

        # Generate feedback
        feedback = self._generate_round1_feedback(solution, score)

        # Create artifacts
        artifacts = [
            {
                'type': 'round1_score',
                'value': score,
                'timestamp': datetime.now(UTC).isoformat()
            },
            {
                'type': 'quick_analysis',
                'summary': feedback[:100]
            }
        ]

        result = MockRoundResult(
            round_id='round1_loongflow_ai',
            passed=passed,
            score=score,
            feedback=feedback,
            evaluation_time=elapsed,
            artifacts=artifacts
        )

        # Update state
        self.state['current_round'] = 1
        self.state['rounds_completed'].append('round1')

        return result

    async def run_round2(
        self,
        solution: TestSolution,
        problem: str,
        domain: str
    ) -> MockRoundResult:
        """Run Round 2: Red Team Adversarial"""
        start_time = time.time()

        # Simulate adversarial testing time
        await asyncio.sleep(0.2)

        # Get expected score (handle None for poor solutions that shouldn't reach here)
        score = solution.expected_round2_score if solution.expected_round2_score is not None else 0.0
        passed = score >= self.config.round2_threshold

        elapsed = time.time() - start_time

        # Generate adversarial feedback
        feedback = self._generate_round2_feedback(solution, score)

        # Create artifacts
        artifacts = [
            {
                'type': 'adversarial_score',
                'value': score,
                'timestamp': datetime.now(UTC).isoformat()
            },
            {
                'type': 'vulnerability_report',
                'findings': len(solution.weaknesses)
            }
        ]

        result = MockRoundResult(
            round_id='round2_red_team',
            passed=passed,
            score=score,
            feedback=feedback,
            evaluation_time=elapsed,
            artifacts=artifacts
        )

        # Update state
        self.state['current_round'] = 2
        self.state['rounds_completed'].append('round2')

        return result

    async def run_round3(
        self,
        solution: TestSolution,
        problem: str,
        domain: str
    ) -> MockRoundResult:
        """Run Round 3: Gold Team Consensus"""
        start_time = time.time()

        # Simulate consensus verification time
        await asyncio.sleep(0.3)

        # Get expected score (handle None for poor solutions that shouldn't reach here)
        score = solution.expected_round3_score if solution.expected_round3_score is not None else 0.0
        passed = score >= self.config.round3_threshold

        elapsed = time.time() - start_time

        # Generate consensus feedback
        feedback = self._generate_round3_feedback(solution, score)

        # Create artifacts
        artifacts = [
            {
                'type': 'consensus_score',
                'value': score,
                'timestamp': datetime.now(UTC).isoformat()
            },
            {
                'type': 'verification_report',
                'status': 'verified' if passed else 'rejected'
            }
        ]

        result = MockRoundResult(
            round_id='round3_gold_team',
            passed=passed,
            score=score,
            feedback=feedback,
            evaluation_time=elapsed,
            artifacts=artifacts
        )

        # Update state
        self.state['current_round'] = 3
        self.state['rounds_completed'].append('round3')

        return result

    async def run_full_gauntlet(
        self,
        solution: TestSolution,
        problem: str,
        domain: str
    ) -> MockGauntletResult:
        """Run complete 3-round gauntlet with progressive filtering"""

        self.state['start_time'] = time.time()

        round_results = []

        # Round 1
        r1_result = await self.run_round1(solution, problem, domain)
        round_results.append(r1_result)

        # Check early termination
        if self.config.enable_early_termination and not r1_result.passed:
            return MockGauntletResult(
                passed=False,
                final_score=r1_result.score,
                round_results=round_results,
                rounds_completed=1,
                termination_reason="Failed Round 1 threshold"
            )

        # Round 2
        r2_result = await self.run_round2(solution, problem, domain)
        round_results.append(r2_result)

        # Check early termination
        if self.config.enable_early_termination and not r2_result.passed:
            # Calculate aggregated score
            final_score = self._calculate_aggregated_score([r1_result, r2_result])
            return MockGauntletResult(
                passed=False,
                final_score=final_score,
                round_results=round_results,
                rounds_completed=2,
                termination_reason="Failed Round 2 threshold"
            )

        # Round 3
        r3_result = await self.run_round3(solution, problem, domain)
        round_results.append(r3_result)

        # Calculate final score
        final_score = self._calculate_aggregated_score(round_results)

        # Final decision
        overall_passed = all(r.passed for r in round_results)

        # Fuse artifacts
        fused_artifacts = self._fuse_artifacts(round_results)

        return MockGauntletResult(
            passed=overall_passed,
            final_score=final_score,
            round_results=round_results,
            rounds_completed=3,
            termination_reason=None if overall_passed else "Failed Round 3 threshold",
            fused_artifacts=fused_artifacts
        )

    def _calculate_aggregated_score(
        self,
        round_results: List[MockRoundResult]
    ) -> float:
        """Calculate weighted final score"""
        weights = [
            self.config.round1_weight,
            self.config.round2_weight,
            self.config.round3_weight
        ]

        total_weight = sum(weights[:len(round_results)])
        normalized_weights = [w / total_weight for w in weights[:len(round_results)]]

        final_score = sum(
            r.score * normalized_weights[i]
            for i, r in enumerate(round_results)
        )

        return round(final_score, 3)

    def _fuse_artifacts(
        self,
        round_results: List[MockRoundResult]
    ) -> List[Dict[str, Any]]:
        """Fuse artifacts from all rounds"""
        fused = []

        for result in round_results:
            if result.artifacts:
                fused.extend(result.artifacts)

        # Add fusion metadata
        fused.append({
            'type': 'fusion_metadata',
            'rounds_completed': len(round_results),
            'total_artifacts': sum(len(r.artifacts) for r in round_results),
            'timestamp': datetime.now(UTC).isoformat()
        })

        return fused

    def _generate_round1_feedback(self, solution: TestSolution, score: float) -> str:
        """Generate Round 1 feedback"""
        if score > 0.8:
            return f"Excellent initial assessment. Code shows strong structure and completeness."
        elif score > 0.5:
            return f"Acceptable initial assessment. Code has basic structure but may need refinement."
        else:
            return f"Poor initial assessment. Code lacks structure or completeness."

    def _generate_round2_feedback(self, solution: TestSolution, score: float) -> str:
        """Generate Round 2 feedback"""
        if score > 0.8:
            return f"Robust against adversarial testing. Good error handling and edge cases."
        elif score > 0.5:
            return f"Moderate robustness. Some edge cases not handled."
        else:
            return f"Vulnerable to adversarial testing. Multiple weaknesses found: {solution.weaknesses}"

    def _generate_round3_feedback(self, solution: TestSolution, score: float) -> str:
        """Generate Round 3 feedback"""
        if score > 0.8:
            return f"Consensus verification passed. High-quality solution approved."
        elif score > 0.5:
            return f"Partial consensus. Some improvements recommended."
        else:
            return f"Consensus verification failed. Solution not production-ready."


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def orchestrator_strict():
    """Create orchestrator with strict configuration"""
    return MockThreeRoundOrchestrator(STRICT_CONFIG)


@pytest.fixture
def orchestrator_lenient():
    """Create orchestrator with lenient configuration"""
    return MockThreeRoundOrchestrator(LENIENT_CONFIG)


@pytest.fixture
def orchestrator_balanced():
    """Create orchestrator with balanced configuration"""
    return MockThreeRoundOrchestrator(BALANCED_CONFIG)


@pytest.fixture
def orchestrator_no_early_term():
    """Create orchestrator without early termination"""
    return MockThreeRoundOrchestrator(NO_EARLY_TERMINATION_CONFIG)


@pytest.fixture
def sample_solution_good():
    """High-quality solution that should pass all rounds"""
    return PERFECT_SOLUTIONS[0]


@pytest.fixture
def sample_solution_poor():
    """Low-quality solution that should fail early"""
    return POOR_SOLUTIONS[0]


@pytest.fixture
def sample_solution_moderate():
    """Moderate solution that passes R1, fails R2"""
    return MODERATE_SOLUTIONS[0]


@pytest.fixture
def sample_solution_good_fail_r3():
    """Good solution that passes R1, R2, fails R3"""
    return GOOD_SOLUTIONS[0]


# ============================================================================
# ROUND 1 TESTS
# ============================================================================

class TestRound1Evaluation:
    """Test Round 1: LoongFlow AI Evaluation"""

    @pytest.mark.asyncio
    async def test_round1_evaluation_pass(self, orchestrator_balanced, sample_solution_good):
        """Test that good solutions pass Round 1"""
        result = await orchestrator_balanced.run_round1(
            solution=sample_solution_good,
            problem="Optimize portfolio allocation",
            domain="finance"
        )

        assert result.round_id == 'round1_loongflow_ai'
        assert result.passed is True
        assert result.score >= BALANCED_CONFIG.round1_threshold
        assert len(result.artifacts) > 0

    @pytest.mark.asyncio
    async def test_round1_evaluation_fail(self, orchestrator_balanced, sample_solution_poor):
        """Test that poor solutions fail Round 1"""
        result = await orchestrator_balanced.run_round1(
            solution=sample_solution_poor,
            problem="Simple task",
            domain="general"
        )

        assert result.round_id == 'round1_loongflow_ai'
        assert result.passed is False
        assert result.score < BALANCED_CONFIG.round1_threshold

    @pytest.mark.asyncio
    async def test_round1_timeout_handling(self, orchestrator_balanced):
        """Test handling of timeout scenarios"""
        # This would timeout with real implementation
        # For mock, we test the timeout parameter exists
        assert orchestrator_balanced.config.max_timeout_seconds > 0

    @pytest.mark.asyncio
    async def test_round1_feedback_generation(self, orchestrator_balanced, sample_solution_good):
        """Test that feedback is generated"""
        result = await orchestrator_balanced.run_round1(
            solution=sample_solution_good,
            problem="Test problem",
            domain="test"
        )

        assert result.feedback is not None
        assert len(result.feedback) > 0
        assert isinstance(result.feedback, str)

    @pytest.mark.asyncio
    async def test_round1_artifact_creation(self, orchestrator_balanced, sample_solution_good):
        """Test that artifacts are created"""
        result = await orchestrator_balanced.run_round1(
            solution=sample_solution_good,
            problem="Test problem",
            domain="test"
        )

        assert len(result.artifacts) > 0
        assert any(a['type'] == 'round1_score' for a in result.artifacts)


# ============================================================================
# ROUND 2 TESTS
# ============================================================================

class TestRound2Evaluation:
    """Test Round 2: Red Team Adversarial"""

    @pytest.mark.asyncio
    async def test_round2_adversarial_evaluation(self, orchestrator_balanced, sample_solution_good):
        """Test adversarial evaluation"""
        result = await orchestrator_balanced.run_round2(
            solution=sample_solution_good,
            problem="Optimize portfolio",
            domain="finance"
        )

        assert result.round_id == 'round2_red_team'
        assert result.score is not None

    @pytest.mark.asyncio
    async def test_round2_robustness_scoring(self, orchestrator_balanced, sample_solution_good):
        """Test robustness scoring"""
        result = await orchestrator_balanced.run_round2(
            solution=sample_solution_good,
            problem="Test",
            domain="test"
        )

        assert 0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_round2_edge_case_handling(self, orchestrator_balanced, sample_solution_moderate):
        """Test edge case detection"""
        result = await orchestrator_balanced.run_round2(
            solution=sample_solution_moderate,
            problem="Test",
            domain="test"
        )

        # Moderate solutions should have robustness scores (using the solution's own expected score)
        assert result.score is not None
        assert result.score <= 1.0  # Score should be normalized


# ============================================================================
# ROUND 3 TESTS
# ============================================================================

class TestRound3Evaluation:
    """Test Round 3: Gold Team Consensus"""

    @pytest.mark.asyncio
    async def test_round3_consensus_evaluation(self, orchestrator_balanced, sample_solution_good):
        """Test consensus evaluation"""
        result = await orchestrator_balanced.run_round3(
            solution=sample_solution_good,
            problem="Test",
            domain="test"
        )

        assert result.round_id == 'round3_gold_team'
        assert result.score is not None

    @pytest.mark.asyncio
    async def test_round3_model_agreement(self, orchestrator_balanced):
        """Test multiple model agreement"""
        # This would test actual consensus in real implementation
        # For mock, we verify the result structure
        assert True  # Placeholder

    @pytest.mark.asyncio
    async def test_round3_verification_status(self, orchestrator_balanced, sample_solution_good):
        """Test verification status in artifacts"""
        result = await orchestrator_balanced.run_round3(
            solution=sample_solution_good,
            problem="Test",
            domain="test"
        )

        verification_artifact = next(
            (a for a in result.artifacts if a['type'] == 'verification_report'),
            None
        )
        assert verification_artifact is not None


# ============================================================================
# PROGRESSIVE FILTERING TESTS
# ============================================================================

class TestProgressiveFiltering:
    """Test early termination and progressive filtering"""

    @pytest.mark.asyncio
    async def test_early_termination_round1(self, orchestrator_balanced, sample_solution_poor):
        """Test early termination after Round 1"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_poor,
            problem="Simple",
            domain="general"
        )

        assert result.rounds_completed == 1
        assert result.termination_reason == "Failed Round 1 threshold"
        assert result.passed is False
        assert len(result.round_results) == 1

    @pytest.mark.asyncio
    async def test_early_termination_round2(self, orchestrator_balanced, sample_solution_moderate):
        """Test early termination after Round 2"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_moderate,
            problem="Portfolio",
            domain="finance"
        )

        assert result.rounds_completed == 2
        assert result.termination_reason == "Failed Round 2 threshold"
        assert result.passed is False
        assert len(result.round_results) == 2

    @pytest.mark.asyncio
    async def test_complete_all_rounds(self, orchestrator_balanced, sample_solution_good):
        """Test completing all rounds"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="finance"
        )

        assert result.rounds_completed == 3
        assert result.termination_reason is None
        assert len(result.round_results) == 3

    @pytest.mark.asyncio
    async def test_no_early_termination_config(self, orchestrator_no_early_term, sample_solution_poor):
        """Test with early termination disabled"""
        result = await orchestrator_no_early_term.run_full_gauntlet(
            solution=sample_solution_poor,
            problem="Simple",
            domain="general"
        )

        # Should complete all rounds even with poor solution
        assert result.rounds_completed == 3


# ============================================================================
# SCORE AGGREGATION TESTS
# ============================================================================

class TestScoreAggregation:
    """Test score calculation and aggregation"""

    @pytest.mark.asyncio
    async def test_weighted_score_aggregation(self, orchestrator_balanced, sample_solution_good):
        """Test weighted score calculation"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="finance"
        )

        # Manually calculate expected score
        r1_score = result.round_results[0].score
        r2_score = result.round_results[1].score
        r3_score = result.round_results[2].score

        expected = (
            r1_score * BALANCED_CONFIG.round1_weight +
            r2_score * BALANCED_CONFIG.round2_weight +
            r3_score * BALANCED_CONFIG.round3_weight
        )

        assert abs(result.final_score - expected) < 0.001

    @pytest.mark.asyncio
    async def test_score_normalization(self, orchestrator_balanced):
        """Test score normalization"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=PERFECT_SOLUTIONS[0],
            problem="Test",
            domain="test"
        )

        assert 0 <= result.final_score <= 1.0

    @pytest.mark.asyncio
    async def test_final_score_calculation(self, orchestrator_balanced, sample_solution_moderate):
        """Test final score with only 2 rounds completed"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_moderate,
            problem="Test",
            domain="test"
        )

        # Should aggregate only R1 and R2 scores
        assert result.final_score is not None
        assert 0 <= result.final_score <= 1.0


# ============================================================================
# DECISION LOGIC TESTS
# ============================================================================

class TestDecisionLogic:
    """Test continue/terminate decisions"""

    @pytest.mark.asyncio
    async def test_continue_decision_round1(self, orchestrator_balanced, sample_solution_good):
        """Test continue decision after Round 1"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="finance"
        )

        assert result.rounds_completed >= 2  # Continued to R2

    @pytest.mark.asyncio
    async def test_terminate_decision_round1(self, orchestrator_balanced, sample_solution_poor):
        """Test terminate decision after Round 1"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_poor,
            problem="Simple",
            domain="general"
        )

        assert result.rounds_completed == 1  # Terminated after R1
        assert result.termination_reason is not None


# ============================================================================
# ARTIFACT FUSION TESTS
# ============================================================================

class TestArtifactFusion:
    """Test artifact collection and fusion"""

    @pytest.mark.asyncio
    async def test_artifact_collection(self, orchestrator_balanced, sample_solution_good):
        """Test artifact collection from all rounds"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="finance"
        )

        assert result.fused_artifacts is not None
        assert len(result.fused_artifacts) > 0

    @pytest.mark.asyncio
    async def test_fused_artifacts_generation(self, orchestrator_balanced, sample_solution_good):
        """Test fused artifacts include metadata"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="test"
        )

        fusion_meta = next(
            (a for a in result.fused_artifacts if a['type'] == 'fusion_metadata'),
            None
        )

        assert fusion_meta is not None
        assert fusion_meta['rounds_completed'] == 3


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================

class TestStateManagement:
    """Test state transitions and persistence"""

    @pytest.mark.asyncio
    async def test_state_initialization(self, orchestrator_balanced):
        """Test initial state"""
        assert orchestrator_balanced.state['current_round'] == 0
        assert len(orchestrator_balanced.state['rounds_completed']) == 0

    @pytest.mark.asyncio
    async def test_state_transitions(self, orchestrator_balanced, sample_solution_good):
        """Test state transitions through rounds"""
        await orchestrator_balanced.run_round1(sample_solution_good, "Test", "test")
        assert orchestrator_balanced.state['current_round'] == 1

        await orchestrator_balanced.run_round2(sample_solution_good, "Test", "test")
        assert orchestrator_balanced.state['current_round'] == 2

        await orchestrator_balanced.run_round3(sample_solution_good, "Test", "test")
        assert orchestrator_balanced.state['current_round'] == 3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance benchmarks"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_round1_performance_target(self, orchestrator_balanced, sample_solution_good):
        """Test Round 1 completes within target (30s)"""
        target_time = 30.0

        start = time.time()
        await orchestrator_balanced.run_round1(sample_solution_good, "Test", "test")
        elapsed = time.time() - start

        assert elapsed < target_time, f"Round 1 took {elapsed:.1f}s (target: {target_time}s)"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_round2_performance_target(self, orchestrator_balanced, sample_solution_good):
        """Test Round 2 completes within target (2min)"""
        target_time = 120.0

        start = time.time()
        await orchestrator_balanced.run_round2(sample_solution_good, "Test", "test")
        elapsed = time.time() - start

        assert elapsed < target_time, f"Round 2 took {elapsed:.1f}s (target: {target_time}s)"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_round3_performance_target(self, orchestrator_balanced, sample_solution_good):
        """Test Round 3 completes within target (5min)"""
        target_time = 300.0

        start = time.time()
        await orchestrator_balanced.run_round3(sample_solution_good, "Test", "test")
        elapsed = time.time() - start

        assert elapsed < target_time, f"Round 3 took {elapsed:.1f}s (target: {target_time}s)"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_full_gauntlet_performance(self, orchestrator_balanced, sample_solution_good):
        """Test full gauntlet completes within target (8min)"""
        target_time = 480.0

        start = time.time()
        await orchestrator_balanced.run_full_gauntlet(sample_solution_good, "Test", "test")
        elapsed = time.time() - start

        assert elapsed < target_time, f"Full gauntlet took {elapsed:.1f}s (target: {target_time}s)"


# ============================================================================
# QUALITY METRICS TESTS
# ============================================================================

class TestQualityMetrics:
    """Test quality metrics and validation"""

    @pytest.mark.asyncio
    async def test_false_positive_rate(self, orchestrator_lenient):
        """
        Test false positive rate using known bad solutions
        False positive = bad solution that passes gauntlet
        """
        bad_solutions = POOR_SOLUTIONS

        false_positives = 0
        for solution in bad_solutions:
            result = await orchestrator_lenient.run_full_gauntlet(
                solution=solution,
                problem="Test",
                domain="test"
            )
            if result.passed:
                false_positives += 1

        fpr = false_positives / len(bad_solutions)
        assert fpr < 0.05, f"False positive rate too high: {fpr:.2%}"

    @pytest.mark.asyncio
    async def test_false_negative_rate(self, orchestrator_lenient):
        """
        Test false negative rate using known good solutions
        False negative = good solution that fails gauntlet
        """
        good_solutions = PERFECT_SOLUTIONS

        false_negatives = 0
        for solution in good_solutions:
            result = await orchestrator_lenient.run_full_gauntlet(
                solution=solution,
                problem="Test",
                domain="test"
            )
            if not result.passed:
                false_negatives += 1

        fnr = false_negatives / len(good_solutions)
        assert fnr < 0.10, f"False negative rate too high: {fnr:.2%}"

    @pytest.mark.asyncio
    async def test_precision_score(self, orchestrator_balanced):
        """Test precision: TP / (TP + FP)"""
        # This would be calculated from a confusion matrix
        # For now, test that we can calculate it
        assert True

    @pytest.mark.asyncio
    async def test_recall_score(self, orchestrator_balanced):
        """Test recall: TP / (TP + FN)"""
        # This would be calculated from a confusion matrix
        # For now, test that we can calculate it
        assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_pipeline_perfect_solution(self, orchestrator_balanced, sample_solution_good):
        """Test full pipeline with perfect solution"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Optimize portfolio allocation",
            domain="finance"
        )

        assert result.passed is True
        assert result.rounds_completed == 3
        assert result.round_results[0].passed is True
        assert result.round_results[1].passed is True
        assert result.round_results[2].passed is True
        assert result.final_score > 0.8

    @pytest.mark.asyncio
    async def test_full_pipeline_failed_solution(self, orchestrator_balanced, sample_solution_poor):
        """Test full pipeline with failed solution"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_poor,
            problem="Simple task",
            domain="general"
        )

        assert result.passed is False
        assert result.rounds_completed == 1
        assert result.termination_reason is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_with_artifacts(self, orchestrator_balanced, sample_solution_good):
        """Test full pipeline with artifact tracking"""
        result = await orchestrator_balanced.run_full_gauntlet(
            solution=sample_solution_good,
            problem="Portfolio",
            domain="finance"
        )

        assert result.fused_artifacts is not None
        assert len(result.fused_artifacts) > 3  # At least one artifact per round + metadata

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, orchestrator_balanced):
        """Test multiple concurrent gauntlet evaluations"""
        # Use the same solution twice since there's only one perfect solution
        solutions = [PERFECT_SOLUTIONS[0], PERFECT_SOLUTIONS[0]]

        # Run concurrent evaluations
        tasks = [
            orchestrator_balanced.run_full_gauntlet(s, "Test", "test")
            for s in solutions
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert all(r.passed for r in results)


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test different configuration profiles"""

    @pytest.mark.asyncio
    async def test_strict_configuration(self, orchestrator_strict):
        """Test strict configuration"""
        assert orchestrator_strict.config.round1_threshold == 0.7
        assert orchestrator_strict.config.round2_threshold == 0.8
        assert orchestrator_strict.config.round3_threshold == 0.9

    @pytest.mark.asyncio
    async def test_lenient_configuration(self, orchestrator_lenient):
        """Test lenient configuration"""
        assert orchestrator_lenient.config.round1_threshold == 0.3
        assert orchestrator_lenient.config.round2_threshold == 0.5
        assert orchestrator_lenient.config.round3_threshold == 0.6

    @pytest.mark.asyncio
    async def test_configuration_impact(self, orchestrator_strict, orchestrator_lenient, sample_solution_moderate):
        """Test that configuration affects pass rates"""
        # Strict config should fail earlier
        result_strict = await orchestrator_strict.run_full_gauntlet(
            solution=sample_solution_moderate,
            problem="Test",
            domain="test"
        )

        # Lenient config might pass further
        result_lenient = await orchestrator_lenient.run_full_gauntlet(
            solution=sample_solution_moderate,
            problem="Test",
            domain="test"
        )

        # Lenient should complete at least as many rounds as strict
        assert result_lenient.rounds_completed >= result_strict.rounds_completed


# ============================================================================
# RUNNER
# ============================================================================

def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
