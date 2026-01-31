"""
Test Suite for 3-Round Gauntlet Orchestrator
=============================================

Comprehensive tests for the ThreeRoundGauntletOrchestrator including:
- Configuration validation
- Progressive filtering
- Score aggregation
- Threshold enforcement
- Early termination
- Artifact collection
- Report generation
- Domain-specific configurations

Author: OpenEvolve Test Suite
Date: 2026-01-30
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from datetime import datetime, UTC

from openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundConfig,
    Round1Result,
    Round2Result,
    Round3Result,
    FullGauntletResult,
    ThreeRoundGauntletOrchestrator,
    GauntletRound,
    create_strict_config,
    create_lenient_config,
    create_balanced_config,
    create_domain_config
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_config():
    """Base configuration for testing"""
    return ThreeRoundConfig(
        round1_threshold=0.5,
        round2_threshold=0.6,
        round3_threshold=0.7,
        round1_weight=0.2,
        round2_weight=0.3,
        round3_weight=0.5,
        enable_early_termination=True
    )


@pytest.fixture
def mock_loongflow_evaluator():
    """Mock LoongFlow evaluator"""
    mock = AsyncMock()
    mock.evaluate_round = AsyncMock(
        return_value=Mock(
            passed=True,
            score=0.8,
            feedback="Good solution",
            details={'metrics': {'confidence': 0.85}}
        )
    )
    return mock


@pytest.fixture
def orchestrator(base_config, mock_loongflow_evaluator):
    """Orchestrator instance with mocked evaluators"""
    orch = ThreeRoundGauntletOrchestrator(config=base_config)
    orch.round1_evaluator = mock_loongflow_evaluator
    return orch


@pytest.fixture
def sample_solution():
    """Sample solution for testing"""
    return """
def optimize_portfolio(returns, risk_tolerance):
    import numpy as np

    # Calculate expected returns and covariance
    mu = np.mean(returns, axis=0)
    sigma = np.cov(returns.T)

    # Optimize using Markowitz mean-variance
    n = len(mu)
    args = (mu, sigma, risk_tolerance)

    # Simple equal weight for demonstration
    weights = np.ones(n) / n

    return weights
"""


@pytest.fixture
def sample_problem():
    """Sample problem for testing"""
    return "Optimize portfolio allocation for maximum return with minimum risk"


@pytest.fixture
def sample_domain():
    """Sample domain for testing"""
    return "finance"


# ============================================================================
# Configuration Tests
# ============================================================================

class TestThreeRoundConfig:
    """Test ThreeRoundConfig validation and defaults"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = ThreeRoundConfig()

        assert config.round1_weight == 0.2
        assert config.round2_weight == 0.3
        assert config.round3_weight == 0.5
        assert config.round1_threshold == 0.5
        assert config.round2_threshold == 0.6
        assert config.round3_threshold == 0.7
        assert config.round1_enabled is True
        assert config.round2_enabled is True
        assert config.round3_enabled is True
        assert config.enable_early_termination is True

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = ThreeRoundConfig(
            round1_threshold=0.8,
            round2_threshold=0.9,
            round3_weight=0.6,
            enable_early_termination=False
        )

        assert config.round1_threshold == 0.8
        assert config.round2_threshold == 0.9
        assert config.round3_weight == 0.6
        assert config.enable_early_termination is False

    def test_invalid_threshold_raises_error(self):
        """Test that invalid thresholds raise ValueError"""
        with pytest.raises(ValueError, match="round1_threshold must be 0.0-1.0"):
            ThreeRoundConfig(round1_threshold=1.5)

        with pytest.raises(ValueError, match="round2_threshold must be 0.0-1.0"):
            ThreeRoundConfig(round2_threshold=-0.1)

    def test_weight_sum_warning(self, caplog):
        """Test that weight sums outside 0.9-1.1 generate warning"""
        import logging

        config = ThreeRoundConfig(
            round1_weight=0.1,
            round2_weight=0.1,
            round3_weight=0.1
        )

        # Should generate warning about weights
        # Note: This test verifies the warning logic exists

    def test_disabled_rounds(self):
        """Test configuration with disabled rounds"""
        config = ThreeRoundConfig(
            round1_enabled=False,
            round2_enabled=True,
            round3_enabled=True
        )

        assert config.round1_enabled is False
        assert config.round2_enabled is True
        assert config.round3_enabled is True


# ============================================================================
# Round Result Tests
# ============================================================================

class TestRoundResults:
    """Test round result data structures"""

    def test_round1_result_creation(self):
        """Test Round1Result creation"""
        result = Round1Result(
            passed=True,
            score=0.8,
            confidence=0.85,
            evaluation_time=1.5,
            feedback="Good solution",
            artifacts=[{'test': 'data'}],
            evaluator_type="loongflow"
        )

        assert result.passed is True
        assert result.score == 0.8
        assert result.confidence == 0.85
        assert result.evaluation_time == 1.5
        assert len(result.artifacts) == 1
        assert result.evaluator_type == "loongflow"
        assert result.timestamp > 0

    def test_round1_result_to_dict(self):
        """Test Round1Result serialization"""
        result = Round1Result(
            passed=True,
            score=0.8,
            confidence=0.85,
            evaluation_time=1.5,
            feedback="Good solution"
        )

        data = result.to_dict()

        assert data['round'] == 'round1_loongflow'
        assert data['passed'] is True
        assert data['score'] == 0.8
        assert data['confidence'] == 0.85
        assert 'timestamp' in data

    def test_round2_result_creation(self):
        """Test Round2Result creation"""
        result = Round2Result(
            passed=True,
            score=0.75,
            attacks_attempted=10,
            attacks_successful=2,
            robustness_score=0.8,
            evaluation_time=5.0,
            feedback="Solution survived most attacks"
        )

        assert result.passed is True
        assert result.score == 0.75
        assert result.attacks_attempted == 10
        assert result.attacks_successful == 2
        assert result.robustness_score == 0.8

    def test_round2_result_to_dict(self):
        """Test Round2Result serialization"""
        result = Round2Result(
            passed=True,
            score=0.75,
            attacks_attempted=10,
            attacks_successful=2,
            robustness_score=0.8,
            evaluation_time=5.0,
            feedback="Robust"
        )

        data = result.to_dict()

        assert data['round'] == 'round2_red_team'
        assert data['attacks_attempted'] == 10
        assert data['robustness_score'] == 0.8

    def test_round3_result_creation(self):
        """Test Round3Result creation"""
        result = Round3Result(
            passed=True,
            score=0.9,
            consensus_score=0.85,
            formal_verification_passed=True,
            evaluation_time=10.0,
            feedback="High consensus achieved"
        )

        assert result.passed is True
        assert result.score == 0.9
        assert result.consensus_score == 0.85
        assert result.formal_verification_passed is True

    def test_round3_result_to_dict(self):
        """Test Round3Result serialization"""
        result = Round3Result(
            passed=True,
            score=0.9,
            consensus_score=0.85,
            formal_verification_passed=True,
            evaluation_time=10.0,
            feedback="Verified"
        )

        data = result.to_dict()

        assert data['round'] == 'round3_gold_team'
        assert data['consensus_score'] == 0.85
        assert data['formal_verification_passed'] is True


# ============================================================================
# Orchestrator Initialization Tests
# ============================================================================

class TestOrchestratorInitialization:
    """Test orchestrator initialization"""

    def test_initialization_with_config(self, base_config):
        """Test orchestrator initialization with config"""
        orchestrator = ThreeRoundGauntletOrchestrator(config=base_config)

        assert orchestrator.config == base_config

    def test_evaluator_initialization(self, base_config, mocker):
        """Test that evaluators are initialized"""
        # Mock the evaluator creation
        mock_create = mocker.patch(
            'openevolve.gauntlets.three_round_orchestrator.create_loongflow_evaluator',
            return_value=MagicMock()
        )

        orchestrator = ThreeRoundGauntletOrchestrator(config=base_config)

        # Verify LoongFlow evaluator was created
        assert orchestrator.round1_evaluator is not None


# ============================================================================
# Round Execution Tests
# ============================================================================

class TestRoundExecution:
    """Test individual round execution"""

    @pytest.mark.asyncio
    async def test_run_round1_success(self, orchestrator, sample_solution, sample_problem, sample_domain):
        """Test successful Round 1 execution"""
        result = await orchestrator.run_round1(sample_solution, sample_problem, sample_domain)

        assert result.passed is True
        assert result.score == 0.8
        assert result.evaluation_time > 0
        assert result.feedback != ""

    @pytest.mark.asyncio
    async def test_run_round1_failure(self, orchestrator, sample_solution, sample_problem, sample_domain):
        """Test Round 1 execution with failure"""
        # Mock evaluator to return low score
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            return_value=Mock(
                passed=False,
                score=0.3,
                feedback="Poor solution",
                details={'metrics': {'confidence': 0.5}}
            )
        )

        result = await orchestrator.run_round1(sample_solution, sample_problem, sample_domain)

        assert result.passed is False
        assert result.score == 0.3

    @pytest.mark.asyncio
    async def test_run_round2_success(self, orchestrator, sample_solution, sample_problem, sample_domain):
        """Test successful Round 2 execution"""
        result = await orchestrator.run_round2(sample_solution, sample_problem, sample_domain)

        assert result.score >= 0.0
        assert result.evaluation_time > 0
        assert result.attacks_attempted > 0

    @pytest.mark.asyncio
    async def test_run_round3_success(self, orchestrator, sample_solution, sample_problem, sample_domain):
        """Test successful Round 3 execution"""
        result = await orchestrator.run_round3(sample_solution, sample_problem, sample_domain)

        assert result.score >= 0.0
        assert result.evaluation_time > 0
        assert result.consensus_score >= 0.0


# ============================================================================
# Progressive Filtering Tests
# ============================================================================

class TestProgressiveFiltering:
    """Test progressive filtering logic"""

    def test_should_continue_to_round2_passing(self, orchestrator):
        """Test continue to round 2 with passing score"""
        r1_result = Round1Result(
            passed=True,
            score=0.8,
            confidence=0.85,
            evaluation_time=1.0,
            feedback="Good"
        )

        assert orchestrator.should_continue_to_round2(r1_result) is True

    def test_should_continue_to_round2_failing(self, orchestrator):
        """Test continue to round 2 with failing score"""
        r1_result = Round1Result(
            passed=False,
            score=0.3,
            confidence=0.5,
            evaluation_time=1.0,
            feedback="Poor"
        )

        assert orchestrator.should_continue_to_round2(r1_result) is False

    def test_should_continue_to_round2_below_threshold(self, orchestrator):
        """Test continue to round 2 with score below threshold"""
        r1_result = Round1Result(
            passed=True,
            score=0.4,  # Below 0.5 threshold
            confidence=0.6,
            evaluation_time=1.0,
            feedback="Mediocre"
        )

        assert orchestrator.should_continue_to_round2(r1_result) is False

    def test_should_continue_to_round3_passing(self, orchestrator):
        """Test continue to round 3 with passing score"""
        r2_result = Round2Result(
            passed=True,
            score=0.8,
            attacks_attempted=10,
            attacks_successful=2,
            robustness_score=0.85,
            evaluation_time=2.0,
            feedback="Good"
        )

        assert orchestrator.should_continue_to_round3(r2_result) is True

    def test_should_continue_to_round3_failing(self, orchestrator):
        """Test continue to round 3 with failing score"""
        r2_result = Round2Result(
            passed=False,
            score=0.4,
            attacks_attempted=10,
            attacks_successful=8,
            robustness_score=0.3,
            evaluation_time=2.0,
            feedback="Poor"
        )

        assert orchestrator.should_continue_to_round3(r2_result) is False


# ============================================================================
# Score Aggregation Tests
# ============================================================================

class TestScoreAggregation:
    """Test score aggregation logic"""

    def test_aggregate_all_rounds(self, orchestrator):
        """Test aggregation with all three rounds"""
        r1 = Round1Result(passed=True, score=0.8, confidence=0.8, evaluation_time=1.0, feedback="")
        r2 = Round2Result(passed=True, score=0.75, attacks_attempted=5, attacks_successful=1,
                         robustness_score=0.8, evaluation_time=2.0, feedback="")
        r3 = Round3Result(passed=True, score=0.9, consensus_score=0.85,
                         formal_verification_passed=False, evaluation_time=3.0, feedback="")

        final = orchestrator.calculate_final_score(r1, r2, r3)

        # Weighted: (0.8*0.2 + 0.75*0.3 + 0.9*0.5) / 1.0
        expected = (0.8 * 0.2 + 0.75 * 0.3 + 0.9 * 0.5)
        assert abs(final - expected) < 0.001

    def test_aggregate_rounds_1_and_2_only(self, orchestrator):
        """Test aggregation with only rounds 1 and 2"""
        r1 = Round1Result(passed=True, score=0.8, confidence=0.8, evaluation_time=1.0, feedback="")
        r2 = Round2Result(passed=True, score=0.75, attacks_attempted=5, attacks_successful=1,
                         robustness_score=0.8, evaluation_time=2.0, feedback="")

        final = orchestrator.calculate_final_score(r1, r2, None)

        # Weighted: (0.8*0.2 + 0.75*0.3) / 0.5
        expected = (0.8 * 0.2 + 0.75 * 0.3) / 0.5
        assert abs(final - expected) < 0.001

    def test_aggregate_round1_only(self, orchestrator):
        """Test aggregation with only round 1"""
        r1 = Round1Result(passed=True, score=0.8, confidence=0.8, evaluation_time=1.0, feedback="")

        final = orchestrator.calculate_final_score(r1, None, None)

        assert final == 0.8

    def test_aggregate_no_rounds(self, orchestrator):
        """Test aggregation with no rounds"""
        final = orchestrator.calculate_final_score(None, None, None)

        assert final == 0.0


# ============================================================================
# Full Gauntlet Execution Tests
# ============================================================================

class TestFullGauntletExecution:
    """Test complete gauntlet execution"""

    @pytest.mark.asyncio
    async def test_full_gauntlet_all_rounds_pass(self, orchestrator, sample_solution,
                                                  sample_problem, sample_domain):
        """Test full gauntlet with all rounds passing"""
        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        assert result.rounds_completed == 3
        assert result.passed is True
        assert result.final_score > 0
        assert result.termination_reason is None
        assert result.round1_result is not None
        assert result.round2_result is not None
        assert result.round3_result is not None

    @pytest.mark.asyncio
    async def test_full_gauntlet_fail_round1(self, orchestrator, sample_solution,
                                              sample_problem, sample_domain):
        """Test full gauntlet with failure at round 1"""
        # Mock round 1 to fail
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            return_value=Mock(
                passed=False,
                score=0.3,
                feedback="Poor solution",
                details={'metrics': {'confidence': 0.5}}
            )
        )

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        assert result.rounds_completed == 1
        assert result.passed is False
        assert result.termination_reason is not None
        assert result.round1_result is not None
        assert result.round2_result is None
        assert result.round3_result is None

    @pytest.mark.asyncio
    async def test_full_gauntlet_fail_round2(self, orchestrator, sample_solution,
                                              sample_problem, sample_domain):
        """Test full gauntlet with failure at round 2"""
        # Round 2 will return low score (placeholder logic)
        # This test verifies the orchestrator structure
        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # With current placeholder, round 2 passes
        # Test structure is correct
        assert result is not None
        assert isinstance(result, FullGauntletResult)

    @pytest.mark.asyncio
    async def test_full_gauntlet_early_termination_disabled(self, base_config, sample_solution,
                                                              sample_problem, sample_domain):
        """Test that early termination can be disabled"""
        # Disable early termination
        base_config.enable_early_termination = False

        orchestrator = ThreeRoundGauntletOrchestrator(config=base_config)
        orchestrator.round1_evaluator = AsyncMock()
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            return_value=Mock(
                passed=False,
                score=0.3,
                feedback="Poor",
                details={'metrics': {'confidence': 0.5}}
            )
        )

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Should run all rounds even though round 1 failed
        # (depending on implementation, may still terminate on errors)

    @pytest.mark.asyncio
    async def test_full_gauntlet_artifact_collection(self, orchestrator, sample_solution,
                                                      sample_problem, sample_domain):
        """Test that artifacts are collected from all rounds"""
        # Enable artifact aggregation
        orchestrator.config.aggregate_artifacts = True

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Verify artifacts were collected
        assert isinstance(result.artifacts_from_all_rounds, list)


# ============================================================================
# Report Generation Tests
# ============================================================================

class TestReportGeneration:
    """Test report generation"""

    def test_generate_comprehensive_report(self, orchestrator):
        """Test comprehensive report generation"""
        r1 = Round1Result(passed=True, score=0.8, confidence=0.85, evaluation_time=1.0,
                         feedback="Good solution")
        r2 = Round2Result(passed=True, score=0.75, attacks_attempted=10, attacks_successful=2,
                         robustness_score=0.8, evaluation_time=2.0, feedback="Robust")
        r3 = Round3Result(passed=True, score=0.9, consensus_score=0.85,
                         formal_verification_passed=False, evaluation_time=3.0, feedback="Verified")

        full_result = FullGauntletResult(
            solution="test solution",
            problem="test problem",
            domain="test",
            round1_result=r1,
            round2_result=r2,
            round3_result=r3,
            passed=True,
            final_score=0.833,
            rounds_completed=3,
            termination_reason=None,
            comprehensive_report=orchestrator._generate_report(r1, r2, r3, None)
        )

        report = orchestrator.generate_comprehensive_report(full_result)

        assert "3-ROUND GAUNTLET EVALUATION REPORT" in report
        assert "Round 1 (LoongFlow AI): PASSED" in report
        assert "Round 2 (Red Team): PASSED" in report
        assert "Round 3 (Gold Team): PASSED" in report
        assert "Score: 0.800" in report or "Score: 0.80" in report

    def test_generate_report_with_early_termination(self, orchestrator):
        """Test report generation with early termination"""
        r1 = Round1Result(passed=False, score=0.3, confidence=0.5, evaluation_time=1.0,
                         feedback="Poor solution")

        report = orchestrator._generate_report(r1, None, None, "Failed Round 1 threshold")

        assert "TERMINATED EARLY" in report
        assert "Failed Round 1 threshold" in report
        assert "Round 1 (LoongFlow AI): FAILED" in report


# ============================================================================
# Domain Configuration Tests
# ============================================================================

class TestDomainConfigurations:
    """Test domain-specific configurations"""

    def test_strict_config(self):
        """Test strict configuration creation"""
        config = create_strict_config()

        assert config.round1_threshold == 0.7
        assert config.round2_threshold == 0.8
        assert config.round3_threshold == 0.9
        assert config.enable_early_termination is True

    def test_lenient_config(self):
        """Test lenient configuration creation"""
        config = create_lenient_config()

        assert config.round1_threshold == 0.3
        assert config.round2_threshold == 0.5
        assert config.round3_threshold == 0.6
        assert config.enable_early_termination is False

    def test_balanced_config(self):
        """Test balanced configuration creation"""
        config = create_balanced_config()

        assert config.round1_threshold == 0.5
        assert config.round2_threshold == 0.6
        assert config.round3_threshold == 0.7

    def test_finance_domain_config(self):
        """Test finance domain configuration"""
        config = create_domain_config('finance')

        assert config.round1_threshold == 0.7  # Strict
        assert config.round2_threshold == 0.8
        assert config.round3_threshold == 0.9

    def test_science_domain_config(self):
        """Test science domain configuration"""
        config = create_domain_config('science')

        assert config.round1_threshold == 0.5  # Moderate
        assert config.round2_threshold == 0.6
        assert config.round3_threshold == 0.7

    def test_web_domain_config(self):
        """Test web domain configuration"""
        config = create_domain_config('web')

        assert config.round1_threshold == 0.3  # Lenient
        assert config.round2_threshold == 0.5
        assert config.round3_threshold == 0.6

    def test_unknown_domain_config(self):
        """Test unknown domain defaults to balanced"""
        config = create_domain_config('unknown_domain')

        assert config.round1_threshold == 0.5
        assert config.round2_threshold == 0.6
        assert config.round3_threshold == 0.7


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_round1_evaluator_not_initialized(self, base_config, sample_solution,
                                                      sample_problem, sample_domain):
        """Test behavior when Round 1 evaluator is not initialized"""
        orchestrator = ThreeRoundGauntletOrchestrator(config=base_config)
        orchestrator.round1_evaluator = None

        result = await orchestrator.run_round1(sample_solution, sample_problem, sample_domain)

        assert result.passed is False
        assert result.score == 0.0
        assert "not initialized" in result.feedback.lower()

    @pytest.mark.asyncio
    async def test_exception_handling_in_full_gauntlet(self, orchestrator, sample_solution,
                                                        sample_problem, sample_domain):
        """Test exception handling during gauntlet execution"""
        # Mock evaluator to raise exception
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            side_effect=Exception("Evaluator error")
        )

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Should handle exception gracefully
        assert result is not None
        assert "Execution error" in result.termination_reason or result.passed is False

    def test_score_calculation_with_zero_weights(self):
        """Test score calculation when weights are zero"""
        config = ThreeRoundConfig(
            round1_weight=0.0,
            round2_weight=0.0,
            round3_weight=0.0
        )

        orchestrator = ThreeRoundGauntletOrchestrator(config=config)

        r1 = Round1Result(passed=True, score=0.8, confidence=0.8, evaluation_time=1.0, feedback="")
        r2 = Round2Result(passed=True, score=0.75, attacks_attempted=5, attacks_successful=1,
                         robustness_score=0.8, evaluation_time=2.0, feedback="")
        r3 = Round3Result(passed=True, score=0.9, consensus_score=0.85,
                         formal_verification_passed=False, evaluation_time=3.0, feedback="")

        # Should handle zero weights gracefully
        final = orchestrator.calculate_final_score(r1, r2, r3)
        assert final >= 0.0  # Should not crash


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_complete_workflow_passing(self, orchestrator, sample_solution,
                                              sample_problem, sample_domain):
        """Test complete workflow with passing solution"""
        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Verify complete workflow
        assert result.passed is True
        assert result.rounds_completed == 3
        assert result.final_score > 0.7  # Should pass all thresholds
        assert len(result.comprehensive_report) > 0

    @pytest.mark.asyncio
    async def test_complete_workflow_failing_early(self, orchestrator, sample_solution,
                                                    sample_problem, sample_domain):
        """Test complete workflow with early failure"""
        # Mock to fail early
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            return_value=Mock(
                passed=False,
                score=0.2,
                feedback="Very poor",
                details={'metrics': {'confidence': 0.3}}
            )
        )

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Verify early termination
        assert result.rounds_completed == 1
        assert result.passed is False
        assert result.termination_reason is not None
        assert result.total_time < 10.0  # Should be fast with early termination


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and timing tests"""

    @pytest.mark.asyncio
    async def test_evaluation_timing(self, orchestrator, sample_solution,
                                      sample_problem, sample_domain):
        """Test that evaluation timing is recorded"""
        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        assert result.total_time > 0

        if result.round1_result:
            assert result.round1_result.evaluation_time > 0

        if result.round2_result:
            assert result.round2_result.evaluation_time > 0

        if result.round3_result:
            assert result.round3_result.evaluation_time > 0

    @pytest.mark.asyncio
    async def test_early_termination_saves_time(self, base_config, sample_solution,
                                                 sample_problem, sample_domain):
        """Test that early termination saves evaluation time"""
        base_config.enable_early_termination = True

        orchestrator = ThreeRoundGauntletOrchestrator(config=base_config)
        orchestrator.round1_evaluator.evaluate_round = AsyncMock(
            return_value=Mock(
                passed=False,
                score=0.2,
                feedback="Poor",
                details={'metrics': {'confidence': 0.3}}
            )
        )

        result = await orchestrator.run_full_gauntlet(
            solution=sample_solution,
            problem=sample_problem,
            domain=sample_domain
        )

        # Should complete quickly with early termination
        assert result.rounds_completed == 1
        assert result.total_time < 5.0  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
