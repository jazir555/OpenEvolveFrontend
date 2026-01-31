"""
Comprehensive Test Suite for Strategy Recommender

Tests all aspects of AI-powered strategy recommendation including:
- Problem characteristic extraction
- Historical performance queries
- Strategy ranking and scoring
- Recommendation generation
- Learning from runs
- Confidence calibration
- Domain-specific scenarios

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, Any

from knowledge_engine.core.strategy_recommender import (
    StrategyRecommender,
    StrategyRecommendation,
    ProblemCharacteristics,
    HistoricalRun,
    RankedStrategy,
    EvolutionSystem,
    EvolutionMode,
    DomainType,
    EvaluationCost,
    ComplexityLevel,
    recommend_evolutionary_strategy
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_historical_runs():
    """Sample historical run data"""
    now = datetime.now(UTC)
    return [
        HistoricalRun(
            run_id="run_001",
            domain="finance",
            strategy_used="pes",
            mode_used="pes",
            problem_complexity="high",
            final_score=0.85,
            convergence_speed=30,
            evaluation_count=30,
            diversity_score=0.7,
            timestamp=now - timedelta(days=10),
            sample_efficiency=0.028
        ),
        HistoricalRun(
            run_id="run_002",
            domain="finance",
            strategy_used="qd",
            mode_used="qd",
            problem_complexity="high",
            final_score=0.75,
            convergence_speed=100,
            evaluation_count=100,
            diversity_score=0.9,
            timestamp=now - timedelta(days=8),
            sample_efficiency=0.0075
        ),
        HistoricalRun(
            run_id="run_003",
            domain="science",
            strategy_used="pes",
            mode_used="pes",
            problem_complexity="high",
            final_score=0.90,
            convergence_speed=20,
            evaluation_count=20,
            diversity_score=0.6,
            timestamp=now - timedelta(days=5),
            sample_efficiency=0.045
        ),
        HistoricalRun(
            run_id="run_004",
            domain="trading",
            strategy_used="adversarial",
            mode_used="adversarial",
            problem_complexity="medium",
            final_score=0.70,
            convergence_speed=80,
            evaluation_count=80,
            diversity_score=0.5,
            timestamp=now - timedelta(days=3),
            sample_efficiency=0.00875
        ),
    ]


@pytest.fixture
def recommender(sample_historical_runs):
    """Initialize recommender with sample data"""
    recommender = StrategyRecommender(use_ai_analysis=False)
    for run in sample_historical_runs:
        recommender.historical_runs[run.run_id] = run
    return recommender


# ============================================================================
# PROBLEM ANALYSIS TESTS
# ============================================================================

class TestProblemAnalysis:
    """Test problem characteristic extraction"""

    @pytest.mark.asyncio
    async def test_analyze_finance_problem(self, recommender):
        """Test analysis of finance domain problem"""
        problem = """
        Optimize portfolio allocation for maximum return with minimum risk.
        Need to balance return, volatility, and liquidity constraints.
        Run backtests on 5 years of historical data.
        """

        constraints = {
            "objectives": ["return", "risk", "liquidity"],
            "constraints": ["no_short_selling", "max_allocation_0.4"],
            "time_limit_seconds": 300  # 5 minutes per backtest
        }

        chars = await recommender.analyze_problem_characteristics(
            problem, "finance", constraints
        )

        assert chars.domain == "finance"
        assert chars.complexity in ["high", "medium"]
        assert chars.evaluation_cost in ["expensive", "very_expensive"]
        assert chars.has_multiple_objectives is True
        assert chars.requires_diversity is True
        assert chars.requires_robustness is True
        assert len(chars.keywords) > 0

    @pytest.mark.asyncio
    async def test_analyze_science_problem(self, recommender):
        """Test analysis of science domain problem"""
        problem = """
        Design experiment to optimize chemical reaction yield.
        Need to explore temperature, pressure, and catalyst concentration.
        Each experiment requires running simulation.
        """

        constraints = {
            "objectives": ["yield", "cost"],
            "constraints": ["max_temperature_500"],
            "time_limit_seconds": 600  # 10 minutes per simulation
        }

        chars = await recommender.analyze_problem_characteristics(
            problem, "science", constraints
        )

        assert chars.domain == "science"
        assert chars.evaluation_cost == "very_expensive"
        assert chars.has_multiple_objectives is True
        assert chars.requires_diversity is True

    @pytest.mark.asyncio
    async def test_analyze_web_problem(self, recommender):
        """Test analysis of web design problem"""
        problem = """
        Optimize landing page layout for conversion.
        Test different button colors and placements.
        Use Lighthouse for performance testing.
        """

        constraints = {
            "objectives": ["conversion", "performance"],
            "time_limit_seconds": 5  # Very fast evaluation
        }

        chars = await recommender.analyze_problem_characteristics(
            problem, "web", constraints
        )

        assert chars.domain == "web"
        assert chars.evaluation_cost == "cheap"
        assert chars.estimated_iterations > 100

    @pytest.mark.asyncio
    async def test_complexity_assessment(self, recommender):
        """Test complexity level assessment"""
        simple_problem = "Sort a list of numbers"
        complex_problem = """
        Optimize multi-objective portfolio allocation under
        complex regulatory constraints with risk management
        """

        simple_chars = await recommender.analyze_problem_characteristics(
            simple_problem, "general", {}
        )
        complex_chars = await recommender.analyze_problem_characteristics(
            complex_problem, "finance", {"constraints": ["c1", "c2", "c3", "c4"]}
        )

        assert simple_chars.complexity == "low"
        assert complex_chars.complexity == "high"

    @pytest.mark.asyncio
    async def test_evaluation_cost_assessment(self, recommender):
        """Test evaluation cost assessment"""
        expensive_problem = "Run Monte Carlo simulation with backtest"
        cheap_problem = "Use Lighthouse to test web page"

        expensive_chars = await recommender.analyze_problem_characteristics(
            expensive_problem, "finance", {}
        )
        cheap_chars = await recommender.analyze_problem_characteristics(
            cheap_problem, "web", {}
        )

        assert expensive_chars.evaluation_cost in ["expensive", "very_expensive"]
        assert cheap_chars.evaluation_cost == "cheap"


# ============================================================================
# HISTORICAL PERFORMANCE TESTS
# ============================================================================

class TestHistoricalPerformance:
    """Test historical performance querying"""

    @pytest.mark.asyncio
    async def test_query_by_domain(self, recommender):
        """Test querying historical runs by domain"""
        finance_history = await recommender.query_historical_performance(
            "finance", "high"
        )

        assert len(finance_history) >= 2
        assert all(run.domain == "finance" for run in finance_history)

    @pytest.mark.asyncio
    async def test_query_empty_domain(self, recommender):
        """Test querying domain with no history"""
        pharma_history = await recommender.query_historical_performance(
            "pharma", "high"
        )

        assert len(pharma_history) == 0

    @pytest.mark.asyncio
    async def test_historical_data_parsing(self, recommender):
        """Test parsing historical run data"""
        raw_data = {
            "run_id": "test_run",
            "domain": "trading",
            "strategy_used": "qd",
            "mode_used": "qd",
            "problem_complexity": "medium",
            "final_score": 0.75,
            "convergence_speed": 50,
            "evaluation_count": 50,
            "diversity_score": 0.8,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {}
        }

        parsed = recommender._parse_historical_run(raw_data)

        assert parsed.run_id == "test_run"
        assert parsed.domain == "trading"
        assert parsed.final_score == 0.75


# ============================================================================
# STRATEGY RANKING TESTS
# ============================================================================

class TestStrategyRanking:
    """Test strategy ranking and scoring"""

    @pytest.mark.asyncio
    async def test_rank_strategies_expensive_eval(self, recommender):
        """Test ranking with expensive evaluations"""
        problem_chars = ProblemCharacteristics(
            domain="finance",
            complexity="high",
            evaluation_cost="very_expensive",
            has_multiple_objectives=True,
            requires_diversity=True,
            requires_robustness=True,
            constraint_count=3,
            estimated_iterations=50
        )

        history = await recommender.query_historical_performance("finance", "high")
        ranked = await recommender.rank_strategies(problem_chars, history)

        assert len(ranked) > 0
        # PES should rank high for expensive evaluations
        pes_rank = next((i for i, s in enumerate(ranked)
                         if s.mode == "pes"), None)
        assert pes_rank is not None
        assert pes_rank < 3  # Top 3

    @pytest.mark.asyncio
    async def test_rank_strategies_multi_objective(self, recommender):
        """Test ranking with multiple objectives"""
        problem_chars = ProblemCharacteristics(
            domain="engineering",
            complexity="medium",
            evaluation_cost="moderate",
            has_multiple_objectives=True,
            requires_diversity=False,
            requires_robustness=False,
            constraint_count=2,
            estimated_iterations=100
        )

        ranked = await recommender.rank_strategies(problem_chars, [])

        # MO mode should rank high for multiple objectives
        mo_strategy = next((s for s in ranked if s.mode == "mo"), None)
        assert mo_strategy is not None
        assert any("multi-objective" in p.lower() for p in mo_strategy.pros)

    @pytest.mark.asyncio
    async def test_rank_strategies_diversity_needed(self, recommender):
        """Test ranking when diversity is required"""
        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity="high",
            evaluation_cost="expensive",
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=False,
            constraint_count=1,
            estimated_iterations=50
        )

        ranked = await recommender.rank_strategies(problem_chars, [])

        # QD should rank high for diversity
        qd_strategy = next((s for s in ranked if s.mode == "qd"), None)
        assert qd_strategy is not None
        assert any("diverse" in p.lower() or "diversity" in p.lower()
                  for p in qd_strategy.pros)

    @pytest.mark.asyncio
    async def test_rank_strategies_robustness_needed(self, recommender):
        """Test ranking when robustness is required"""
        problem_chars = ProblemCharacteristics(
            domain="engineering",
            complexity="high",
            evaluation_cost="expensive",
            has_multiple_objectives=False,
            requires_diversity=False,
            requires_robustness=True,
            constraint_count=2,
            estimated_iterations=50
        )

        ranked = await recommender.rank_strategies(problem_chars, [])

        # Adversarial should rank high for robustness
        adv_strategy = next((s for s in ranked if s.mode == "adversarial"), None)
        assert adv_strategy is not None
        assert any("robust" in p.lower() for p in adv_strategy.pros)

    @pytest.mark.asyncio
    async def test_score_calculation(self, recommender):
        """Test strategy score calculation"""
        problem_chars = ProblemCharacteristics(
            domain="science",
            complexity="high",
            evaluation_cost="very_expensive",
            has_multiple_objectives=False,
            requires_diversity=True,
            requires_robustness=False,
            constraint_count=1,
            estimated_iterations=30
        )

        strategy = await recommender._score_strategy(
            EvolutionSystem.LOONGFLOW,
            EvolutionMode.PES,
            problem_chars,
            []
        )

        assert 0 <= strategy.score <= 100
        assert len(strategy.pros) > 0
        assert strategy.system == EvolutionSystem.LOONGFLOW
        assert strategy.mode == EvolutionMode.PES


# ============================================================================
# RECOMMENDATION TESTS
# ============================================================================

class TestRecommendationGeneration:
    """Test complete recommendation generation"""

    @pytest.mark.asyncio
    async def test_recommend_finance_problem(self, recommender):
        """Test recommendation for finance domain"""
        problem = "Optimize portfolio allocation for max Sharpe ratio"
        constraints = {
            "objectives": ["return", "risk"],
            "time_limit_seconds": 300
        }

        recommendation = await recommender.recommend_strategy(
            problem, "finance", constraints
        )

        assert isinstance(recommendation, StrategyRecommendation)
        assert recommendation.recommended_system in [s.value for s in EvolutionSystem]
        assert recommendation.recommended_mode in [m.value for m in EvolutionMode]
        assert 0.0 <= recommendation.confidence <= 1.0
        assert recommendation.reasoning is not None
        assert len(recommendation.alternatives) >= 2

    @pytest.mark.asyncio
    async def test_recommend_science_problem(self, recommender):
        """Test recommendation for science domain"""
        problem = "Optimize experimental design for maximum yield"
        constraints = {
            "time_limit_seconds": 600
        }

        recommendation = await recommender.recommend_strategy(
            problem, "science", constraints
        )

        # Should recommend PES for expensive evaluations
        assert recommendation.recommended_system == EvolutionSystem.LOONGFLOW
        assert recommendation.recommended_mode == EvolutionMode.PES

    @pytest.mark.asyncio
    async def test_recommend_trading_problem(self, recommender):
        """Test recommendation for trading domain"""
        problem = "Develop trading strategy with robustness testing"

        recommendation = await recommender.recommend_strategy(
            problem, "trading", {"safety_critical": True}
        )

        # Should consider adversarial for robustness
        assert any(alt.mode == "adversarial" for alt in recommendation.alternatives)

    @pytest.mark.asyncio
    async def test_config_overrides_generation(self, recommender):
        """Test configuration override generation"""
        problem = "Optimize with expensive evaluations"
        recommendation = await recommender.recommend_strategy(
            problem, "science", {"time_limit_seconds": 600}
        )

        assert isinstance(recommendation.config_overrides, dict)
        assert "max_iterations" in recommendation.config_overrides

    @pytest.mark.asyncio
    async def test_explanation_generation(self, recommender):
        """Test explanation text generation"""
        problem = "Optimize portfolio allocation"
        recommendation = await recommender.recommend_strategy(
            problem, "finance", {}
        )

        explanation = recommender.explain_recommendation(recommendation)

        assert "Strategy Recommendation" in explanation
        assert "Recommended System:" in explanation
        assert "Recommended Mode:" in explanation
        assert "Confidence:" in explanation
        assert "Primary Reason:" in explanation
        assert "Alternatives" in explanation

    @pytest.mark.asyncio
    async def test_performance_prediction(self, recommender):
        """Test performance prediction"""
        problem = "Simple optimization problem"
        recommendation = await recommender.recommend_strategy(
            problem, "web", {"time_limit_seconds": 5}
        )

        perf = recommendation.expected_performance
        assert perf.expected_iterations > 0
        assert perf.expected_time_seconds >= 0
        assert 0.0 <= perf.expected_score <= 1.0
        assert perf.success_probability >= 0.0


# ============================================================================
# LEARNING TESTS
# ============================================================================

class TestLearning:
    """Test learning from completed runs"""

    @pytest.mark.asyncio
    async def test_learn_from_run(self, recommender):
        """Test learning from completed evolutionary run"""
        run_result = {
            "run_id": "test_learn_001",
            "domain": "finance",
            "strategy_used": "pes",
            "mode_used": "pes",
            "complexity": "high",
            "final_score": 0.88,
            "iterations": 25,
            "evaluations": 25,
            "diversity_score": 0.65,
            "evaluation_cost": "expensive",
            "predicted_score": 0.85
        }

        await recommender.learn_from_run(run_result)

        # Verify stored in memory
        assert "test_learn_001" in recommender.historical_runs
        stored = recommender.historical_runs["test_learn_001"]
        assert stored.final_score == 0.88
        assert stored.domain == "finance"

        # Verify accuracy tracking
        assert len(recommender.recommendation_accuracy) > 0

    @pytest.mark.asyncio
    async def test_learning_affects_recommendations(self, recommender):
        """Test that learning affects future recommendations"""
        # Get initial recommendation
        problem = "Optimize trading strategy"
        rec1 = await recommender.recommend_strategy(
            problem, "trading", {}
        )

        # Learn from successful PES run
        await recommender.learn_from_run({
            "run_id": "pes_success",
            "domain": "trading",
            "strategy_used": "pes",
            "mode_used": "pes",
            "complexity": "medium",
            "final_score": 0.95,
            "iterations": 20,
            "evaluations": 20,
            "diversity_score": 0.7
        })

        # Get new recommendation
        rec2 = await recommender.recommend_strategy(
            problem, "trading", {}
        )

        # PES should now rank higher
        assert rec2.recommended_mode == "pes"


# ============================================================================
# CONFIDENCE TESTS
# ============================================================================

class TestConfidence:
    """Test confidence scoring and calibration"""

    @pytest.mark.asyncio
    async def test_confidence_with_no_history(self, recommender):
        """Test confidence when no historical data available"""
        problem = "Brand new problem type"
        recommendation = await recommender.recommend_strategy(
            problem, "general", {}
        )

        # Confidence should be lower without history
        assert 0.0 < recommendation.confidence < 1.0

    @pytest.mark.asyncio
    async def test_confidence_with_history(self, recommender):
        """Test confidence with historical data"""
        problem = "Optimize portfolio allocation"
        recommendation = await recommender.recommend_strategy(
            problem, "finance", {}
        )

        # Should have moderate confidence from historical runs
        assert recommendation.confidence > 0.3

    @pytest.mark.asyncio
    async def test_confidence_adjustment(self, recommender):
        """Test confidence adjustment based on accuracy"""
        # Simulate some accurate predictions
        for i in range(5):
            await recommender.learn_from_run({
                "run_id": f"accuracy_test_{i}",
                "domain": "finance",
                "strategy_used": "pes",
                "mode_used": "pes",
                "complexity": "high",
                "final_score": 0.8,
                "predicted_score": 0.8,
                "iterations": 30,
                "evaluations": 30,
                "diversity_score": 0.7
            })

        problem = "Optimize portfolio"
        recommendation = await recommender.recommend_strategy(
            problem, "finance", {}
        )

        # Get adjusted confidence
        adjusted = recommender.get_recommendation_confidence(recommendation)

        # Should account for accuracy
        assert 0.0 <= adjusted <= 1.0


# ============================================================================
# DOMAIN-SPECIFIC TESTS
# ============================================================================

class TestDomainScenarios:
    """Test domain-specific recommendation scenarios"""

    @pytest.mark.asyncio
    async def test_finance_domain_scenario(self, recommender):
        """Test finance domain recommendation scenario"""
        problem = """
        Optimize portfolio allocation across 5 assets.
        Maximize return while minimizing risk (volatility).
        Ensure diversification and liquidity constraints.
        Use backtest on 3 years of daily data.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "finance", {"time_limit_seconds": 300}
        )

        # PES recommended for expensive backtests
        assert recommendation.recommended_mode == "pes"
        assert "evaluation" in str(recommendation.config_overrides).lower()

    @pytest.mark.asyncio
    async def test_science_domain_scenario(self, recommender):
        """Test science domain recommendation scenario"""
        problem = """
        Optimize chemical reaction conditions.
        Variables: temperature, pressure, catalyst amount.
        Each experiment requires molecular dynamics simulation.
        Goal: maximize yield while minimizing cost.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "science", {"time_limit_seconds": 900}
        )

        # Should recommend PES or QD for exploration
        assert recommendation.recommended_mode in ["pes", "qd"]

    @pytest.mark.asyncio
    async def test_engineering_domain_scenario(self, recommender):
        """Test engineering domain recommendation scenario"""
        problem = """
        Optimize truss bridge design.
        Minimize weight while supporting 1000kg load.
        Requires FEA simulation for each design.
        Must satisfy safety constraints.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "engineering", {
                "time_limit_seconds": 600,
                "safety_critical": True
            }
        )

        # Should include adversarial for safety
        modes = [recommendation.recommended_mode] + [alt.mode for alt in recommendation.alternatives]
        assert "adversarial" in modes

    @pytest.mark.asyncio
    async def test_trading_domain_scenario(self, recommender):
        """Test trading domain recommendation scenario"""
        problem = """
        Develop algorithmic trading strategy.
        Test on historical price data across market regimes.
        Need robustness to adverse market conditions.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "trading", {"safety_critical": True}
        )

        # Adversarial or QD for robustness
        assert recommendation.recommended_mode in ["adversarial", "qd", "pes"]

    @pytest.mark.asyncio
    async def test_web_domain_scenario(self, recommender):
        """Test web design domain scenario"""
        problem = """
        Optimize landing page design.
        Test button colors, placement, and copy.
        Use Lighthouse for performance scoring.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "web", {"time_limit_seconds": 10}
        )

        # Cheap evaluations, standard mode should work
        assert recommendation.recommended_mode in ["standard", "qd"]

    @pytest.mark.asyncio
    async def test_pharma_domain_scenario(self, recommender):
        """Test pharma domain recommendation scenario"""
        problem = """
        Optimize molecular structure for drug target.
        Maximize binding affinity, minimize toxicity.
        Each evaluation requires molecular docking simulation.
        """

        recommendation = await recommender.recommend_strategy(
            problem, "pharma", {"time_limit_seconds": 1200}
        )

        # QD for exploring chemical space
        assert recommendation.recommended_mode in ["qd", "pes"]


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_recommend_evolutionary_strategy(self):
        """Test convenience function"""
        recommendation = await recommend_evolutionary_strategy(
            problem_description="Optimize portfolio allocation",
            domain="finance"
        )

        assert isinstance(recommendation, StrategyRecommendation)
        assert recommendation.recommended_system is not None


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
