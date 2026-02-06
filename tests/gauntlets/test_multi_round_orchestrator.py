"""
Unit Tests for Multi-Round Gauntlet Orchestrator

Comprehensive test suite covering:
1. State management and initialization
2. Decision logic for all rounds
3. Score normalization
4. Artifact fusion and consensus detection
5. Progress reporting
6. Performance metrics
7. Parallel execution
8. Edge cases and error handling
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import time

# Import the module to test
from openevolve.gauntlets.multi_round_orchestrator import (
    MultiRoundGauntletOrchestrator,
    GauntletState,
    FusedArtifacts,
    PerformanceMetrics,
    MultiRoundConfig,
    Round1Result,
    Round2Result,
    Round3Result,
    RoundStatus,
    create_multi_round_orchestrator
)


class TestMultiRoundConfig:
    """Test MultiRoundConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = MultiRoundConfig()

        assert config.round1_threshold == 0.7
        assert config.round2_threshold == 0.6
        assert config.round3_threshold == 0.85
        assert config.min_confidence == 0.6
        assert config.max_weaknesses == 5
        assert config.max_vulnerabilities == 3
        assert config.min_robustness == 0.5
        assert config.min_consensus == 0.75
        assert config.require_formal_verification is False

    def test_custom_config(self):
        """Test custom configuration"""
        config = MultiRoundConfig(
            round1_threshold=0.8,
            round2_threshold=0.7,
            round3_threshold=0.9,
            min_confidence=0.75,
            max_weaknesses=3
        )

        assert config.round1_threshold == 0.8
        assert config.round2_threshold == 0.7
        assert config.round3_threshold == 0.9
        assert config.min_confidence == 0.75
        assert config.max_weaknesses == 3

    def test_config_to_dict(self):
        """Test config serialization"""
        config = MultiRoundConfig()
        config_dict = config.to_dict()

        assert 'round1_threshold' in config_dict
        assert 'round3_threshold' in config_dict
        assert 'round_weights' in config_dict
        assert config_dict['round_weights']['round1'] == 0.2
        assert config_dict['round_weights']['round2'] == 0.3
        assert config_dict['round_weights']['round3'] == 0.5


class TestGauntletState:
    """Test GauntletState dataclass"""

    def test_state_initialization(self):
        """Test state initialization with basic parameters"""
        state = GauntletState(
            solution="def solution(): return 42",
            problem="Find the answer to life",
            domain="mathematics"
        )

        assert state.solution == "def solution(): return 42"
        assert state.problem == "Find the answer to life"
        assert state.domain == "mathematics"
        assert state.current_round == 0
        assert state.rounds_completed == []
        assert state.status == "not_started"
        assert isinstance(state.started_at, datetime)

    def test_state_with_round1_result(self):
        """Test state after Round 1 completion"""
        state = GauntletState(
            solution="test solution",
            problem="test problem",
            domain="test"
        )

        state.round1_result = Round1Result(
            score=0.8,
            confidence=0.9,
            feedback="Good solution"
        )
        state.round1_normalized_score = 0.8
        state.round1_decision = "continue"
        state.rounds_completed = [1]
        state.current_round = 1

        assert state.round1_normalized_score == 0.8
        assert state.round1_decision == "continue"
        assert 1 in state.rounds_completed

    def test_state_serialization(self):
        """Test state to_dict conversion"""
        state = GauntletState(
            solution="x" * 200,  # Long solution
            problem="test",
            domain="test"
        )

        state.round1_normalized_score = 0.8
        state.round1_decision = "continue"
        state.status = "completed"

        state_dict = state.to_dict()

        assert state_dict['solution'].endswith('...')
        assert len(state_dict['solution']) <= 103  # 100 + '...'
        assert state_dict['round1_score'] == 0.8
        assert state_dict['status'] == 'completed'


class TestRoundResults:
    """Test Round result dataclasses"""

    def test_round1_result(self):
        """Test Round1Result creation"""
        result = Round1Result(
            score=0.85,
            confidence=0.9,
            feedback="Excellent solution",
            strengths=["Clear logic", "Well-documented"],
            weaknesses=["Minor optimization needed"],
            robustness_score=0.8
        )

        assert result.score == 0.85
        assert result.confidence == 0.9
        assert len(result.strengths) == 2
        assert len(result.weaknesses) == 1
        assert result.robustness_score == 0.8

    def test_round2_result(self):
        """Test Round2Result creation"""
        result = Round2Result(
            score=75.0,  # 0-100 scale
            attacks_attempted=10,
            attacks_successful=2,
            vulnerabilities_found=["Buffer overflow", "Null pointer"],
            robustness_score=0.7
        )

        assert result.score == 75.0
        assert result.attacks_attempted == 10
        assert result.attacks_successful == 2
        assert len(result.vulnerabilities_found) == 2
        assert result.robustness_score == 0.7

    def test_round3_result(self):
        """Test Round3Result creation"""
        result = Round3Result(
            score=8.5,  # 0-10 scale
            consensus_score=0.85,
            formal_verification_passed=True,
            judge_scores=[8.0, 9.0, 8.5],
            judge_feedback=["Excellent", "Very good", "Good"],
            robustness_score=0.9
        )

        assert result.score == 8.5
        assert result.consensus_score == 0.85
        assert result.formal_verification_passed is True
        assert len(result.judge_scores) == 3
        assert len(result.judge_feedback) == 3


class TestMultiRoundOrchestratorInit:
    """Test MultiRoundGauntletOrchestrator initialization"""

    def test_initialization_with_default_config(self):
        """Test orchestrator initialization"""
        config = MultiRoundConfig()
        orchestrator = MultiRoundGauntletOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.state is None

    def test_initialization_with_custom_config(self):
        """Test orchestrator with custom configuration"""
        config = MultiRoundConfig(
            round1_threshold=0.9,
            enable_early_termination=False
        )
        orchestrator = MultiRoundGauntletOrchestrator(config)

        assert orchestrator.config.round1_threshold == 0.9
        assert orchestrator.config.enable_early_termination is False


class TestGauntletInitialization:
    """Test gauntlet initialization"""

    @pytest.mark.asyncio
    async def test_initialize_gauntlet(self):
        """Test basic gauntlet initialization"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        state = await orchestrator.initialize_gauntlet(
            solution="def solution(): pass",
            problem="Test problem",
            domain="test"
        )

        assert state.solution == "def solution(): pass"
        assert state.problem == "Test problem"
        assert state.domain == "test"
        assert state.status == "in_progress"
        assert state.current_round == 0
        assert isinstance(state.started_at, datetime)

    @pytest.mark.asyncio
    async def test_initialize_with_context(self):
        """Test initialization with additional context"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        context = {
            'criteria': ['speed', 'accuracy'],
            'constraints': ['memory_limit']
        }

        state = await orchestrator.initialize_gauntlet(
            solution="test",
            problem="test",
            domain="test",
            context=context
        )

        assert state.context == context
        assert 'criteria' in state.context


class TestScoreNormalization:
    """Test score normalization across rounds"""

    def test_normalize_round1_score(self):
        """Test Round 1 score normalization (already 0-1)"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_result = Round1Result(
            score=0.75,
            confidence=0.8,
            feedback="test"
        )

        normalized_state = orchestrator.normalize_scores(state)

        assert normalized_state.round1_normalized_score == 0.75

    def test_normalize_round2_score(self):
        """Test Round 2 score normalization (0-100 to 0-1)"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round2_result = Round2Result(
            score=85.0,  # 85/100
            attacks_attempted=10,
            attacks_successful=0
        )

        normalized_state = orchestrator.normalize_scores(state)

        assert normalized_state.round2_normalized_score == 0.85

    def test_normalize_round3_score(self):
        """Test Round 3 score normalization (0-10 to 0-1)"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round3_result = Round3Result(
            score=9.0,  # 9/10
            consensus_score=0.9
        )

        normalized_state = orchestrator.normalize_scores(state)

        assert normalized_state.round3_normalized_score == 0.9

    def test_normalize_all_rounds(self):
        """Test normalization of all rounds together"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_result = Round1Result(score=0.8, confidence=0.8, feedback="test")
        state.round2_result = Round2Result(score=70.0, attacks_attempted=10, attacks_successful=2)
        state.round3_result = Round3Result(score=8.0, consensus_score=0.8)

        normalized_state = orchestrator.normalize_scores(state)

        assert normalized_state.round1_normalized_score == 0.8
        assert normalized_state.round2_normalized_score == 0.7
        assert normalized_state.round3_normalized_score == 0.8


class TestDecisionLogic:
    """Test decision logic for each round"""

    @pytest.mark.asyncio
    async def test_round1_decision_continue(self):
        """Test Round 1 continue decision"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # High score, high confidence, few weaknesses
        state.round1_result = Round1Result(
            score=0.85,
            confidence=0.9,
            feedback="Excellent",
            weaknesses=[]
        )
        state.round1_normalized_score = 0.85

        decision = await orchestrator.make_decision(1, state)

        assert decision == "continue"

    @pytest.mark.asyncio
    async def test_round1_decision_terminate_low_score(self):
        """Test Round 1 terminate due to low score"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig(
            round1_threshold=0.7
        ))
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Low score
        state.round1_result = Round1Result(
            score=0.5,
            confidence=0.8,
            feedback="Poor"
        )
        state.round1_normalized_score = 0.5

        decision = await orchestrator.make_decision(1, state)

        assert decision == "terminate"

    @pytest.mark.asyncio
    async def test_round1_decision_terminate_many_weaknesses(self):
        """Test Round 1 terminate due to too many weaknesses"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig(
            max_weaknesses=3
        ))
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Many weaknesses
        state.round1_result = Round1Result(
            score=0.8,
            confidence=0.9,
            feedback="Has issues",
            weaknesses=["weak1", "weak2", "weak3", "weak4", "weak5"]
        )
        state.round1_normalized_score = 0.8

        decision = await orchestrator.make_decision(1, state)

        assert decision == "terminate"

    @pytest.mark.asyncio
    async def test_round2_decision_continue(self):
        """Test Round 2 continue decision"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round2_result = Round2Result(
            score=75.0,  # 0.75 normalized
            attacks_attempted=10,
            attacks_successful=2,
            robustness_score=0.7
        )
        state.round2_normalized_score = 0.75

        decision = await orchestrator.make_decision(2, state)

        assert decision == "continue"

    @pytest.mark.asyncio
    async def test_round2_decision_terminate_too_many_attacks(self):
        """Test Round 2 terminate due to too many successful attacks"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig(
            max_vulnerabilities=2
        ))
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round2_result = Round2Result(
            score=60.0,
            attacks_attempted=10,
            attacks_successful=5,  # Too many
            robustness_score=0.5
        )
        state.round2_normalized_score = 0.6

        decision = await orchestrator.make_decision(2, state)

        assert decision == "terminate"

    @pytest.mark.asyncio
    async def test_round3_decision_continue(self):
        """Test Round 3 continue (final approval)"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round3_result = Round3Result(
            score=9.0,  # 0.9 normalized
            consensus_score=0.85,
            formal_verification_passed=True
        )
        state.round3_normalized_score = 0.9

        decision = await orchestrator.make_decision(3, state)

        assert decision == "continue"

    @pytest.mark.asyncio
    async def test_round3_decision_terminate_low_consensus(self):
        """Test Round 3 terminate due to low consensus"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig(
            min_consensus=0.75
        ))
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round3_result = Round3Result(
            score=7.0,
            consensus_score=0.6,  # Too low
            formal_verification_passed=True
        )
        state.round3_normalized_score = 0.7

        decision = await orchestrator.make_decision(3, state)

        assert decision == "terminate"


class TestArtifactFusion:
    """Test artifact fusion functionality"""

    def test_fuse_artifacts_all_rounds(self):
        """Test fusion from all three rounds"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Populate all rounds
        state.round1_result = Round1Result(
            score=0.8,
            confidence=0.8,
            feedback="Good",
            strengths=["Clear logic", "Well-documented"],
            weaknesses=["Performance issue"],
            robustness_score=0.8
        )
        state.round1_normalized_score = 0.8

        state.round2_result = Round2Result(
            score=80.0,
            attacks_attempted=10,
            attacks_successful=1,
            vulnerabilities_found=["Memory leak"],
            robustness_score=0.7
        )
        state.round2_normalized_score = 0.8

        state.round3_result = Round3Result(
            score=8.5,
            consensus_score=0.85,
            judge_feedback=["Approved"],
            robustness_score=0.9
        )
        state.round3_normalized_score = 0.85

        fused = orchestrator.fuse_artifacts(state)

        assert len(fused.all_scores) == 3
        assert fused.all_scores['round1'] == 0.8
        assert fused.all_scores['round2'] == 0.8
        assert fused.all_scores['round3'] == 0.85
        assert len(fused.all_strengths) == 2
        assert len(fused.all_weaknesses) >= 2
        assert len(fused.robustness_trend) == 3

    def test_find_consensus(self):
        """Test consensus detection"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        items = [
            "Clear code structure",
            "Good documentation",
            "Clear code structure",  # Duplicate
            "Well-tested",
            "Good documentation"  # Duplicate
        ]

        consensus = orchestrator._find_consensus(items, min_mentions=2)

        assert "clear code structure" in consensus
        assert "good documentation" in consensus
        assert "well-tested" not in consensus  # Only mentioned once

    def test_detect_conflicts(self):
        """Test conflict detection"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        strengths = ["Good performance", "Clear code", "Secure design"]
        weaknesses = ["Poor performance", "Confusing code", "Security issues"]

        conflicts = orchestrator._detect_conflicts(strengths, weaknesses)

        # Should detect "performance" conflict
        assert any("performance" in str(c).lower() for c in conflicts)

    def test_generate_recommendation_approved(self):
        """Test recommendation generation for approved solution"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.status = "completed"
        state.round3_result = Round3Result(
            score=9.0,
            consensus_score=0.9
        )
        state.round3_decision = "continue"

        recommendation = orchestrator._generate_recommendation(state)

        assert "APPROVED" in recommendation

    def test_generate_recommendation_terminated(self):
        """Test recommendation generation for terminated solution"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.status = "terminated"

        recommendation = orchestrator._generate_recommendation(state)

        assert "not recommended" in recommendation.lower()

    def test_prioritize_improvements(self):
        """Test improvement prioritization"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        weaknesses = [
            "Security vulnerability in authentication",
            "Code style issues",
            "Performance optimization needed",
            "Buffer overflow risk",  # Security
            "Memory leak",  # Mentioned twice (consensus)
            "Memory leak"
        ]

        priorities = orchestrator._prioritize_improvements(weaknesses, state)

        # Security items should be first
        assert any("HIGH PRIORITY" in p for p in priorities)
        # Consensus items should be included
        assert any("MEDIUM PRIORITY" in p for p in priorities)


class TestFinalScoreCalculation:
    """Test final score calculation"""

    def test_calculate_final_score_all_rounds(self):
        """Test final score with all three rounds"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_normalized_score = 0.8
        state.round2_normalized_score = 0.7
        state.round3_normalized_score = 0.9

        final_score = orchestrator.calculate_final_score(state)

        # Weighted average: (0.8*0.2 + 0.7*0.3 + 0.9*0.5) = 0.16 + 0.21 + 0.45 = 0.82
        expected = 0.8 * 0.2 + 0.7 * 0.3 + 0.9 * 0.5
        assert abs(final_score - expected) < 0.01

    def test_calculate_final_score_partial_rounds(self):
        """Test final score with only some rounds completed"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Only Round 1 and 2
        state.round1_normalized_score = 0.8
        state.round2_normalized_score = 0.7

        final_score = orchestrator.calculate_final_score(state)

        # Should only weight the completed rounds
        expected = (0.8 * 0.2 + 0.7 * 0.3) / (0.2 + 0.3)
        assert abs(final_score - expected) < 0.01

    def test_calculate_final_score_no_rounds(self):
        """Test final score with no rounds"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        final_score = orchestrator.calculate_final_score(state)

        assert final_score == 0.0


class TestProgressReporting:
    """Test progress report generation"""

    def test_generate_progress_report_initial(self):
        """Test progress report for initial state"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="def solution(): return 42",
            problem="Test problem",
            domain="test"
        )

        report = orchestrator.generate_progress_report(state)

        assert "GAUNTLET PROGRESS REPORT" in report
        assert "Test problem" in report
        assert "TEST" in report.upper()  # Domain is uppercased in report
        assert "0/3" in report

    def test_generate_progress_report_round1_complete(self):
        """Test progress report after Round 1"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_result = Round1Result(
            score=0.85,
            confidence=0.9,
            feedback="Excellent",
            strengths=["Clear", "Documented"],
            weaknesses=["Minor optimization"]
        )
        state.round1_normalized_score = 0.85
        state.round1_decision = "continue"
        state.rounds_completed = [1]
        state.round_times = {1: 10.0}
        state.total_evaluation_time = 10.0

        report = orchestrator.generate_progress_report(state)

        assert "ROUND 1: LoongFlow" in report
        # The score is formatted as percentage with 2 decimals
        assert "85.00%" in report or "85%" in report
        assert "CONTINUE" in report.upper()
        assert "Clear" in report

    def test_generate_progress_report_completed(self):
        """Test progress report for completed gauntlet"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Populate all rounds
        state.round1_result = Round1Result(
            score=0.8, confidence=0.8, feedback="Good",
            strengths=["Clear"], weaknesses=[]
        )
        state.round1_normalized_score = 0.8
        state.round1_decision = "continue"

        state.round2_result = Round2Result(
            score=80.0, attacks_attempted=10, attacks_successful=1
        )
        state.round2_normalized_score = 0.8
        state.round2_decision = "continue"

        state.round3_result = Round3Result(
            score=9.0, consensus_score=0.9,
            judge_scores=[9.0, 9.0, 9.0]
        )
        state.round3_normalized_score = 0.9
        state.round3_decision = "continue"

        state.rounds_completed = [1, 2, 3]
        state.round_times = {1: 10.0, 2: 20.0, 3: 30.0}
        state.total_evaluation_time = 60.0
        state.status = "completed"
        state.completed_at = datetime.utcnow()

        report = orchestrator.generate_progress_report(state)

        assert "FINAL RESULT" in report
        assert "PASSED" in report
        assert "9.0" in report
        assert "60.0s" in report


class TestPerformanceMetrics:
    """Test performance metrics calculation"""

    def test_performance_metrics_completed(self):
        """Test metrics for completed gauntlet"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_normalized_score = 0.7
        state.round2_normalized_score = 0.8
        state.round3_normalized_score = 0.9
        state.rounds_completed = [1, 2, 3]
        state.round_times = {1: 10.0, 2: 20.0, 3: 30.0}
        state.total_evaluation_time = 60.0
        state.status = "completed"
        state.round3_decision = "continue"

        metrics = orchestrator.get_performance_metrics(state)

        assert metrics.total_time == 60.0
        assert metrics.round_times == {1: 10.0, 2: 20.0, 3: 30.0}
        assert metrics.average_score == pytest.approx(0.8)  # (0.7+0.8+0.9)/3
        assert metrics.trend == "improving"
        assert metrics.total_evaluations > 0

    def test_performance_metrics_terminated(self):
        """Test metrics for terminated gauntlet"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        state.round1_normalized_score = 0.5
        state.rounds_completed = [1]
        state.round_times = {1: 10.0}
        state.total_evaluation_time = 10.0
        state.status = "terminated"
        state.current_round = 1
        state.round1_decision = "terminate"

        metrics = orchestrator.get_performance_metrics(state)

        assert metrics.termination_round == 1
        assert metrics.termination_reason is not None
        assert metrics.false_positive_risk < 0.1
        assert metrics.false_negative_risk > 0.1

    def test_performance_metrics_trend(self):
        """Test trend calculation"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        # Improving trend
        state1 = GauntletState(solution="test", problem="test", domain="test")
        state1.round1_normalized_score = 0.6
        state1.round2_normalized_score = 0.7
        state1.round3_normalized_score = 0.8
        metrics1 = orchestrator.get_performance_metrics(state1)
        assert metrics1.trend == "improving"

        # Declining trend
        state2 = GauntletState(solution="test", problem="test", domain="test")
        state2.round1_normalized_score = 0.8
        state2.round2_normalized_score = 0.7
        state2.round3_normalized_score = 0.6
        metrics2 = orchestrator.get_performance_metrics(state2)
        assert metrics2.trend == "declining"

        # Stable trend
        state3 = GauntletState(solution="test", problem="test", domain="test")
        state3.round1_normalized_score = 0.7
        state3.round2_normalized_score = 0.75
        state3.round3_normalized_score = 0.7
        metrics3 = orchestrator.get_performance_metrics(state3)
        assert metrics3.trend == "stable"


class TestFactoryFunction:
    """Test factory function"""

    def test_create_with_defaults(self):
        """Test factory with default parameters"""
        orchestrator = create_multi_round_orchestrator()

        assert orchestrator.config.round1_threshold == 0.7
        assert orchestrator.config.round2_threshold == 0.6
        assert orchestrator.config.round3_threshold == 0.85
        assert orchestrator.config.enable_early_termination is True

    def test_create_with_custom_params(self):
        """Test factory with custom parameters"""
        orchestrator = create_multi_round_orchestrator(
            round1_threshold=0.9,
            round2_threshold=0.8,
            round3_threshold=0.95,
            enable_early_termination=False
        )

        assert orchestrator.config.round1_threshold == 0.9
        assert orchestrator.config.round2_threshold == 0.8
        assert orchestrator.config.round3_threshold == 0.95
        assert orchestrator.config.enable_early_termination is False


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_invalid_round_number(self):
        """Test handling of invalid round number"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        with pytest.raises(ValueError, match="Invalid round number"):
            await orchestrator.execute_round(5, state)

    def test_score_normalization_with_missing_results(self):
        """Test normalization with missing round results"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        # Only Round 1 result
        state.round1_result = Round1Result(
            score=0.8,
            confidence=0.8,
            feedback="test"
        )

        normalized_state = orchestrator.normalize_scores(state)

        assert normalized_state.round1_normalized_score == 0.8
        assert normalized_state.round2_normalized_score is None
        assert normalized_state.round3_normalized_score is None

    def test_artifact_fusion_with_no_results(self):
        """Test artifact fusion with no results"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())
        state = GauntletState(
            solution="test",
            problem="test",
            domain="test"
        )

        fused = orchestrator.fuse_artifacts(state)

        assert len(fused.all_scores) == 0
        assert len(fused.all_strengths) == 0
        assert len(fused.consensus_strengths) == 0
        assert fused.overall_recommendation == ""

    def test_empty_list_handling(self):
        """Test handling of empty lists in artifact fusion"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        consensus = orchestrator._find_consensus([], min_mentions=2)

        assert consensus == []

    def test_items_similar_identical(self):
        """Test similarity detection with identical items"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        similar = orchestrator._items_similar(
            "good performance",
            "good performance"
        )

        assert similar is True

    def test_items_similar_different(self):
        """Test similarity detection with different items"""
        orchestrator = MultiRoundGauntletOrchestrator(MultiRoundConfig())

        similar = orchestrator._items_similar(
            "good performance",
            "bad security"
        )

        assert similar is False


class TestFusedArtifacts:
    """Test FusedArtifacts dataclass"""

    def test_fused_artifacts_to_dict(self):
        """Test FusedArtifacts serialization"""
        fused = FusedArtifacts()
        fused.all_scores = {'round1': 0.8, 'round2': 0.7}
        fused.consensus_strengths = ["Clear code"]
        fused.overall_recommendation = "APPROVED"

        fused_dict = fused.to_dict()

        assert 'all_scores' in fused_dict
        assert fused_dict['all_scores']['round1'] == 0.8
        assert 'consensus_strengths' in fused_dict
        assert fused_dict['overall_recommendation'] == "APPROVED"


class TestPerformanceMetricsDataclass:
    """Test PerformanceMetrics dataclass"""

    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics serialization"""
        metrics = PerformanceMetrics()
        metrics.total_time = 100.0
        metrics.average_score = 0.8
        metrics.trend = "improving"
        metrics.termination_round = 2

        metrics_dict = metrics.to_dict()

        assert metrics_dict['total_time'] == 100.0
        assert metrics_dict['average_score'] == 0.8
        assert metrics_dict['trend'] == "improving"
        assert metrics_dict['termination_round'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
