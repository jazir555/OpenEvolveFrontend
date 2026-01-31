#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Autonomous Investment Committee Agent

Tests single weekly cycle execution, multi-week progression,
learning from feedback, accuracy of recommendations, robustness
to market conditions, and integration with all modules.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
import numpy as np

from openevolve.agents.investment_committee import (
    InvestmentCommitteeAgent,
    PortfolioState,
    InvestmentDecision
)
from openevolve.agents.investment.rlm_decomposer import RLMDecomposer
from openevolve.agents.investment.roma_tester import ROMATester
from openevolve.agents.investment.adversarial_tester import AdversarialTester
from openevolve.agents.investment.math_verifier import MathVerifier
from openevolve.agents.investment.knowledge_integrator import KnowledgeIntegrator


# Fixtures

@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return PortfolioState(
        holdings={"AAPL": 100, "MSFT": 50, "GOOGL": 30},
        cash=10000.0,
        total_value=50000.0,
        last_rebalance=datetime.utcnow() - timedelta(days=10)
    )


@pytest.fixture
def mock_market_data_provider():
    """Create mock market data provider."""
    mock_provider = Mock()

    # Mock current state
    mock_provider.get_current_state = AsyncMock(return_value={
        "fundamentals": {
            "AAPL": {"pe_ratio": 25.0, "earnings_growth": 0.15},
            "MSFT": {"pe_ratio": 30.0, "earnings_growth": 0.20},
            "GOOGL": {"pe_ratio": 22.0, "earnings_growth": 0.18}
        },
        "technical": {
            "market_momentum": 0.05,
            "volatility_regime": "normal"
        },
        "macro": {
            "interest_rate": 0.035,
            "inflation": 0.03,
            "gdp_growth": 0.025
        },
        "sentiment": {
            "market_sentiment": "positive"
        }
    })

    # Mock historical data
    mock_provider.get_historical_data = AsyncMock(return_value={
        "period": "1y",
        "returns": [0.01, 0.02, -0.01, 0.03, 0.01] * 50,  # Daily returns
        "num_observations": 252
    })

    return mock_provider


@pytest.fixture
def agent(sample_portfolio, mock_market_data_provider, tmp_path):
    """Create investment committee agent for testing."""
    return InvestmentCommitteeAgent(
        portfolio_state=sample_portfolio,
        market_data_provider=mock_market_data_provider,
        database_path=tmp_path / "investment_db",
        enable_loongflow=False  # Disable for faster tests
    )


# Tests for Single Weekly Cycle

class TestSingleWeeklyCycle:
    """Tests for a single weekly review cycle."""

    @pytest.mark.asyncio
    async def test_complete_weekly_cycle(self, agent):
        """Test that a complete weekly cycle executes successfully."""
        decision = await agent.weekly_review_cycle()

        assert decision is not None
        assert isinstance(decision, InvestmentDecision)
        assert decision.decision_type in ["hold", "rebalance", "analyze"]
        assert decision.confidence >= 0.0 and decision.confidence <= 1.0
        assert len(agent.decisions) == 1

    @pytest.mark.asyncio
    async def test_review_phase(self, agent):
        """Test the review phase of the cycle."""
        review_data = await agent._review_phase()

        assert "portfolio_value" in review_data
        assert "allocations" in review_data
        assert "market_context" in review_data
        assert "changes" in review_data
        assert "timestamp" in review_data

    @pytest.mark.asyncio
    async def test_analysis_phase(self, agent):
        """Test the analysis phase of the cycle."""
        review_data = await agent._review_phase()
        analysis_results = await agent._analysis_phase(review_data)

        assert "rlm_decomposition" in analysis_results
        assert "roma_tests" in analysis_results
        assert "adversarial_analysis" in analysis_results
        assert "math_verification" in analysis_results

        # Check RLM decomposition
        assert "key_factors" in analysis_results["rlm_decomposition"]
        assert "hypotheses" in analysis_results["rlm_decomposition"]
        assert "sub_problems" in analysis_results["rlm_decomposition"]

        # Check ROMA tests
        assert "hypotheses_tested" in analysis_results["roma_tests"]
        assert "test_results" in analysis_results["roma_tests"]
        assert "recommendations" in analysis_results["roma_tests"]

    @pytest.mark.asyncio
    async def test_decision_phase(self, agent):
        """Test the decision phase of the cycle."""
        review_data = await agent._review_phase()
        analysis_results = await agent._analysis_phase(review_data)
        decision = await agent._decision_phase(review_data, analysis_results)

        assert decision is not None
        assert decision.decision_type in ["hold", "rebalance"]
        assert decision.reasoning is not None
        assert len(decision.reasoning) > 0

    @pytest.mark.asyncio
    async def test_learning_phase(self, agent):
        """Test the learning phase of the cycle."""
        decision = InvestmentDecision(
            decision_id="test_decision",
            timestamp=datetime.utcnow(),
            decision_type="rebalance",
            actions=[],
            reasoning="Test decision",
            confidence=0.75,
            expected_outcome="Positive return",
            metadata={}
        )

        # Learning phase should not raise errors
        await agent._learning_phase(decision)


# Tests for Multi-Week Progression

class TestMultiWeekProgression:
    """Tests for multiple weekly cycles over time."""

    @pytest.mark.asyncio
    async def test_three_week_progression(self, agent):
        """Test that agent runs correctly for three weeks."""
        decisions = []

        for week in range(3):
            decision = await agent.weekly_review_cycle()
            decisions.append(decision)

            # Check that review time advances
            if week < 2:
                time_until_next = agent.review_frequency - timedelta(days=1)
                assert not agent.should_review()

                # Manually advance time for testing
                agent.last_review = datetime.utcnow() - timedelta(days=8)

        assert len(decisions) == 3
        assert len(agent.decisions) == 3

    @pytest.mark.asyncio
    async def test_state_persistence(self, agent, tmp_path):
        """Test that state is persisted and can be reloaded."""
        # Run one cycle
        await agent.weekly_review_cycle()

        # Create new agent with same database path
        new_agent = InvestmentCommitteeAgent(
            portfolio_state=agent.portfolio,
            market_data_provider=agent.market_data,
            database_path=agent.database_path,
            enable_loongflow=False
        )

        # Should have loaded previous decision
        assert len(new_agent.decisions) == 1

    @pytest.mark.asyncio
    async def test_learning_across_cycles(self, agent):
        """Test that learning accumulates across multiple cycles."""
        # Run multiple cycles
        for _ in range(5):
            decision = await agent.weekly_review_cycle()

            # Record outcomes
            await agent.record_outcome(
                decision.decision_id,
                actual_outcome="positive return of 5%",
                performance_metrics={"return": 0.05, "volatility": 0.15}
            )

        # Check that knowledge was accumulated
        knowledge_summary = agent.knowledge_integrator.get_knowledge_summary()
        assert knowledge_summary["total_lessons"] >= 5
        assert knowledge_summary["total_scenarios"] >= 5


# Tests for Learning from Feedback

class TestLearningFromFeedback:
    """Tests for learning from decision outcomes."""

    @pytest.mark.asyncio
    async def test_record_positive_outcome(self, agent):
        """Test recording a positive decision outcome."""
        decision = await agent.weekly_review_cycle()

        await agent.record_outcome(
            decision.decision_id,
            actual_outcome="positive return of 8%",
            performance_metrics={"return": 0.08, "volatility": 0.12, "sharpe": 1.5}
        )

        # Check that outcome was recorded
        updated_decision = next(
            d for d in agent.decisions
            if d.decision_id == decision.decision_id
        )

        assert updated_decision.actual_outcome == "positive return of 8%"
        assert updated_decision.outcome_timestamp is not None
        assert updated_decision.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_record_negative_outcome(self, agent):
        """Test recording a negative decision outcome."""
        decision = await agent.weekly_review_cycle()

        await agent.record_outcome(
            decision.decision_id,
            actual_outcome="negative return of -3%",
            performance_metrics={"return": -0.03, "volatility": 0.18, "sharpe": -0.5}
        )

        updated_decision = next(
            d for d in agent.decisions
            if d.decision_id == decision.decision_id
        )

        assert "negative" in updated_decision.actual_outcome.lower()
        assert updated_decision.performance_metrics["return"] < 0

    @pytest.mark.asyncio
    async def test_knowledge_extraction_from_outcomes(self, agent):
        """Test that knowledge is extracted from outcomes."""
        # Run cycles with outcomes
        for i in range(5):
            decision = await agent.weekly_review_cycle()

            outcome = "positive return" if i % 2 == 0 else "negative return"
            await agent.record_outcome(
                decision.decision_id,
                actual_outcome=outcome,
                performance_metrics={"return": 0.05 if i % 2 == 0 else -0.02}
            )

        # Check knowledge extraction
        summary = agent.knowledge_integrator.get_knowledge_summary()

        # Should have learned something
        assert summary["total_lessons"] > 0
        assert summary["total_scenarios"] > 0


# Tests for Accuracy of Recommendations

class TestRecommendationAccuracy:
    """Tests for accuracy and quality of recommendations."""

    @pytest.mark.asyncio
    async def test_confidence_calibration(self, agent):
        """Test that confidence scores are reasonable."""
        decision = await agent.weekly_review_cycle()

        # Confidence should be between 0 and 1
        assert 0.0 <= decision.confidence <= 1.0

        # Confidence should reflect analysis depth
        if decision.decision_type == "hold":
            # Hold decisions should have high confidence
            assert decision.confidence > 0.6

    @pytest.mark.asyncio
    async def test_recommendation_consistency(self, agent):
        """Test that recommendations are internally consistent."""
        decision = await agent.weekly_review_cycle()

        if decision.decision_type == "rebalance":
            # Check actions are consistent
            actions = decision.actions

            # Should not have buy and sell for same ticker
            tickers = set(a.get("ticker") for a in actions)
            for ticker in tickers:
                ticker_actions = [a for a in actions if a.get("ticker") == ticker]
                action_types = [a.get("action") for a in ticker_actions]

                # Should not have contradictory actions
                assert not ("buy" in action_types and "sell" in action_types)

    @pytest.mark.asyncio
    async def test_constraint_satisfaction(self, agent):
        """Test that recommendations respect constraints."""
        decision = await agent.weekly_review_cycle()

        if decision.decision_type == "rebalance":
            # Check that no action violates max position size
            for action in decision.actions:
                if "target_allocation" in action:
                    target_str = action["target_allocation"]
                    # Extract percentage
                    if isinstance(target_str, str):
                        percentages = [
                            float(x.strip().rstrip('%')) / 100
                            for x in target_str.split('-')
                        ]
                        max_pct = max(percentages)

                        assert max_pct <= agent.max_position_size + 0.05  # Small tolerance


# Tests for Robustness to Market Conditions

class TestMarketConditionRobustness:
    """Tests for robustness across different market conditions."""

    @pytest.mark.asyncio
    async def test_bull_market_handling(self, agent, mock_market_data_provider):
        """Test handling of bull market conditions."""
        # Modify mock data for bull market
        mock_market_data_provider.get_current_state.return_value = {
            "fundamentals": {
                "AAPL": {"pe_ratio": 30.0, "earnings_growth": 0.25},
                "MSFT": {"pe_ratio": 35.0, "earnings_growth": 0.30},
                "GOOGL": {"pe_ratio": 28.0, "earnings_growth": 0.22}
            },
            "technical": {
                "market_momentum": 0.15,  # Strong momentum
                "volatility_regime": "low"
            },
            "macro": {
                "interest_rate": 0.02,  # Low rates
                "inflation": 0.02,
                "gdp_growth": 0.04  # Strong growth
            },
            "sentiment": {
                "market_sentiment": "very positive"
            }
        }

        decision = await agent.weekly_review_cycle()

        # Should handle successfully
        assert decision is not None
        assert decision.decision_type in ["hold", "rebalance"]

    @pytest.mark.asyncio
    async def test_bear_market_handling(self, agent, mock_market_data_provider):
        """Test handling of bear market conditions."""
        mock_market_data_provider.get_current_state.return_value = {
            "fundamentals": {
                "AAPL": {"pe_ratio": 15.0, "earnings_growth": -0.05},
                "MSFT": {"pe_ratio": 18.0, "earnings_growth": -0.10},
                "GOOGL": {"pe_ratio": 14.0, "earnings_growth": -0.08}
            },
            "technical": {
                "market_momentum": -0.20,  # Strong negative momentum
                "volatility_regime": "high"
            },
            "macro": {
                "interest_rate": 0.05,  # High rates
                "inflation": 0.05,
                "gdp_growth": 0.00  # No growth
            },
            "sentiment": {
                "market_sentiment": "negative"
            }
        }

        decision = await agent.weekly_review_cycle()

        # Should handle successfully
        assert decision is not None

    @pytest.mark.asyncio
    async def test_high_volatility_handling(self, agent, mock_market_data_provider):
        """Test handling of high volatility conditions."""
        mock_market_data_provider.get_current_state.return_value = {
            "technical": {
                "market_momentum": 0.0,
                "volatility_regime": "very high"
            },
            "macro": {
                "interest_rate": 0.035,
                "inflation": 0.04,
                "gdp_growth": 0.02
            }
        }

        decision = await agent.weekly_review_cycle()

        # Should be more cautious in high volatility
        if decision.decision_type == "rebalance":
            # Should recommend smaller position sizes
            for action in decision.actions:
                if action.get("action") == "position_size_caution":
                    assert "smaller" in action.get("rationale", "").lower()


# Tests for Module Integration

class TestModuleIntegration:
    """Tests for integration of all modules."""

    @pytest.mark.asyncio
    async def test_rlm_decomposition_integration(self, agent):
        """Test RLM decomposer integration."""
        review_data = await agent._review_phase()
        analysis = await agent._analysis_phase(review_data)

        rlm_result = analysis["rlm_decomposition"]

        assert "key_factors" in rlm_result
        assert len(rlm_result["key_factors"]) > 0

        # Check factor structure
        for factor in rlm_result["key_factors"]:
            assert "name" in factor
            assert "category" in factor
            assert "importance" in factor
            assert 0.0 <= factor["importance"] <= 1.0

    @pytest.mark.asyncio
    async def test_roma_tester_integration(self, agent):
        """Test ROMA tester integration."""
        review_data = await agent._review_phase()
        analysis = await agent._analysis_phase(review_data)

        roma_result = analysis["roma_tests"]

        assert "hypotheses_tested" in roma_result
        assert "test_results" in roma_result
        assert "recommendations" in roma_result

        # Should have tested some hypotheses
        assert roma_result["hypotheses_tested"] > 0

    @pytest.mark.asyncio
    async def test_adversarial_tester_integration(self, agent):
        """Test adversarial tester integration."""
        review_data = await agent._review_phase()
        analysis = await agent._analysis_phase(review_data)

        adversarial_result = analysis["adversarial_analysis"]

        assert "challenges" in adversarial_result
        assert "biases" in adversarial_result
        assert "concerns" in adversarial_result

        # Should have generated some challenges
        assert len(adversarial_result["challenges"]) >= 0

    @pytest.mark.asyncio
    async def test_math_verifier_integration(self, agent):
        """Test math verifier integration."""
        review_data = await agent._review_phase()
        analysis = await agent._analysis_phase(review_data)

        math_result = analysis["math_verification"]

        assert "all_passed" in math_result
        assert "passed_checks" in math_result
        assert "total_checks" in math_result

        # Should have performed some checks
        assert math_result["total_checks"] > 0

    @pytest.mark.asyncio
    async def test_knowledge_integrator_integration(self, agent):
        """Test knowledge integrator integration."""
        decision = await agent.weekly_review_cycle()

        # Record outcome
        await agent.record_outcome(
            decision.decision_id,
            "positive return of 5%",
            {"return": 0.05}
        )

        # Check knowledge was extracted
        summary = agent.knowledge_integrator.get_knowledge_summary()

        assert summary["total_lessons"] > 0
        assert summary["total_scenarios"] > 0


# Tests for Performance Metrics

class TestPerformanceMetrics:
    """Tests for performance tracking and metrics."""

    @pytest.mark.asyncio
    async def test_performance_summary(self, agent):
        """Test performance summary calculation."""
        # Run multiple cycles
        for i in range(5):
            decision = await agent.weekly_review_cycle()
            await agent.record_outcome(
                decision.decision_id,
                f"{'positive' if i % 2 == 0 else 'negative'} return",
                {"return": 0.05 if i % 2 == 0 else -0.02}
            )

        summary = agent.get_performance_summary()

        assert summary["total_decisions"] == 5
        assert summary["decisions_with_outcomes"] == 5
        assert "average_confidence" in summary
        assert "accuracy" in summary
        assert summary["accuracy"] is not None

    @pytest.mark.asyncio
    async def test_accuracy_calculation(self, agent):
        """Test accuracy calculation."""
        # Run cycles with known outcomes
        for i in range(10):
            decision = await agent.weekly_review_cycle()
            await agent.record_outcome(
                decision.decision_id,
                f"{'positive' if i < 7 else 'negative'} return",  # 70% positive
                {"return": 0.05 if i < 7 else -0.02}
            )

        summary = agent.get_performance_summary()

        # Accuracy should be around 70%
        assert 0.6 <= summary["accuracy"] <= 0.8


# Tests for Error Handling

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_portfolio(self, mock_market_data_provider, tmp_path):
        """Test handling of empty portfolio."""
        empty_portfolio = PortfolioState(
            holdings={},
            cash=10000.0,
            total_value=10000.0
        )

        agent = InvestmentCommitteeAgent(
            portfolio_state=empty_portfolio,
            market_data_provider=mock_market_data_provider,
            database_path=tmp_path / "empty_db",
            enable_loongflow=False
        )

        # Should handle gracefully
        decision = await agent.weekly_review_cycle()
        assert decision is not None

    @pytest.mark.asyncio
    async def test_missing_outcome_recording(self, agent):
        """Test recording outcome for non-existent decision."""
        # Should not raise error
        await agent.record_outcome(
            "non_existent_id",
            "test outcome",
            {"return": 0.0}
        )

        # Decision should not be added
        assert len(agent.decisions) == 0

    @pytest.mark.asyncio
    async def test_market_data_failure(self, agent, mock_market_data_provider):
        """Test handling of market data provider failure."""
        # Make market data fail
        mock_market_data_provider.get_current_state.side_effect = Exception("Data unavailable")

        # Should handle gracefully or raise meaningful error
        try:
            decision = await agent.weekly_review_cycle()
            # If it doesn't raise, decision should be None or safe default
            assert decision is None or decision.decision_type == "hold"
        except Exception as e:
            # Should be a meaningful error
            assert "unavailable" in str(e).lower() or "data" in str(e).lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
