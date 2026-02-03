#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Alpha Signal Evolver

Comprehensive tests for alpha signal discovery including:
- Crisis survival validation
- Delisting avoidance
- Feature importance tracking
- Signal combination
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List

from openevolve.finance.verticals.hedge_fund.alpha_evolver import AlphaSignalEvolver
from openevolve.finance.verticals.hedge_fund.feature_importance import FeatureImportanceTracker
from openevolve.finance.verticals.hedge_fund.signal_combiner import SignalCombiner
from openevolve.finance.verticals.hedge_fund.schemas import (
    SignalConstraints,
    AlphaSignal,
    AlphaSource,
    BacktestResult,
    CrisisPerformance,
    CrisisPeriod,
    FeatureSet
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def signal_constraints():
    """Create standard signal constraints for testing."""
    return SignalConstraints(
        min_market_cap=500_000_000,  # $500M min
        max_turnover=50,  # 50% annual turnover
        crisis_periods=["dotcom", "gfc", "covid"],
        min_sharpe=1.0,
        min_information_ratio=0.5,
        max_delisting_rate=0.01,  # Max 1% delisting rate
        max_drawdown=-0.30,  # Max 30% drawdown
        sector_neutral=False,
        beta_neutral=False
    )


@pytest.fixture
def mock_alpha_signal():
    """Create a mock alpha signal for testing."""
    # Create mock backtest result
    backtest = BacktestResult(
        signal_id="test_signal_1",
        universe="russell_3000",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2026, 1, 1),
        include_delisted=True,
        total_return=1.5,
        annual_return=0.08,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        information_ratio=1.0,
        alpha=0.04,
        beta=1.0,
        tracking_error=0.08,
        max_drawdown=-0.20,
        volatility=0.15,
        var_95=-0.02,
        avg_market_cap=2_000_000_000,
        turnover=0.4,
        trading_costs=0.002,
        delisting_rate=0.005,
        survivorship_bias=0.0,
        crisis_performance={
            "dotcom": CrisisPerformance(
                crisis=CrisisPeriod.DOTCOM,
                start_date=datetime(2000, 3, 1),
                end_date=datetime(2002, 10, 1),
                return_pct=0.05,
                alpha_pct=0.03,
                max_drawdown_pct=-0.15,
                volatility_pct=0.18,
                survived=True,
                notes="Survived dotcom crisis"
            ),
            "gfc": CrisisPerformance(
                crisis=CrisisPeriod.GFC,
                start_date=datetime(2008, 9, 1),
                end_date=datetime(2009, 3, 1),
                return_pct=0.02,
                alpha_pct=0.02,
                max_drawdown_pct=-0.20,
                volatility_pct=0.20,
                survived=True,
                notes="Survived GFC"
            ),
            "covid": CrisisPerformance(
                crisis=CrisisPeriod.COVID,
                start_date=datetime(2020, 2, 1),
                end_date=datetime(2020, 12, 1),
                return_pct=0.08,
                alpha_pct=0.05,
                max_drawdown_pct=-0.12,
                volatility_pct=0.25,
                survived=True,
                notes="Survived COVID"
            )
        },
        returns_by_period={}
    )

    return AlphaSignal(
        signal_id="test_signal_1",
        name="Test Value Signal",
        description="Test value factor signal",
        formula="ev_to_fcf",
        features=FeatureSet(
            features={"ev_to_fcf": "enterprise_value / free_cash_flow"},
            feature_correlations={"ev_to_fcf": 0.12}
        ),
        alpha_source=AlphaSource.RISK_BASED,
        rationale="Value stocks have higher expected returns",
        backtest=backtest,
        sharpe_ratio=1.5,
        information_ratio=1.0,
        crisis_performance=backtest.crisis_performance,
        feature_correlation={"ev_to_fcf": 0.12}
    )


@pytest.fixture
def multiple_mock_signals(mock_alpha_signal):
    """Create multiple mock signals for testing."""
    signals = [mock_alpha_signal]

    # Create a second signal
    backtest2 = BacktestResult(
        signal_id="test_signal_2",
        universe="russell_3000",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2026, 1, 1),
        include_delisted=True,
        total_return=1.3,
        annual_return=0.07,
        sharpe_ratio=1.3,
        sortino_ratio=1.6,
        information_ratio=0.8,
        alpha=0.03,
        beta=0.9,
        tracking_error=0.08,
        max_drawdown=-0.22,
        volatility=0.14,
        var_95=-0.02,
        avg_market_cap=1_500_000_000,
        turnover=0.5,
        trading_costs=0.003,
        delisting_rate=0.008,
        survivorship_bias=0.0,
        crisis_performance={
            "dotcom": CrisisPerformance(
                crisis=CrisisPeriod.DOTCOM,
                start_date=datetime(2000, 3, 1),
                end_date=datetime(2002, 10, 1),
                return_pct=0.04,
                alpha_pct=0.02,
                max_drawdown_pct=-0.18,
                volatility_pct=0.19,
                survived=True,
                notes="Survived dotcom"
            ),
            "gfc": CrisisPerformance(
                crisis=CrisisPeriod.GFC,
                start_date=datetime(2008, 9, 1),
                end_date=datetime(2009, 3, 1),
                return_pct=0.01,
                alpha_pct=0.01,
                max_drawdown_pct=-0.22,
                volatility_pct=0.22,
                survived=True,
                notes="Survived GFC"
            ),
            "covid": CrisisPerformance(
                crisis=CrisisPeriod.COVID,
                start_date=datetime(2020, 2, 1),
                end_date=datetime(2020, 12, 1),
                return_pct=0.06,
                alpha_pct=0.04,
                max_drawdown_pct=-0.14,
                volatility_pct=0.26,
                survived=True,
                notes="Survived COVID"
            )
        },
        returns_by_period={}
    )

    signal2 = AlphaSignal(
        signal_id="test_signal_2",
        name="Test Momentum Signal",
        description="Test momentum factor signal",
        formula="momentum_12m",
        features=FeatureSet(
            features={"momentum_12m": "total_return[-252:-21]"},
            feature_correlations={"momentum_12m": 0.15}
        ),
        alpha_source=AlphaSource.BEHAVIORAL,
        rationale="Investors underreact to information",
        backtest=backtest2,
        sharpe_ratio=1.3,
        information_ratio=0.8,
        crisis_performance=backtest2.crisis_performance,
        feature_correlation={"momentum_12m": 0.15}
    )

    signals.append(signal2)
    return signals


# ============================================================================
# TESTS: ALPHA SIGNAL EVOLVER
# ============================================================================

@pytest.mark.asyncio
async def test_alpha_evolver_initialization():
    """Test that AlphaSignalEvolver initializes correctly."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    assert evolver.data_source == "MOCK"
    assert evolver.include_delisted == True
    assert evolver.backtester is not None
    assert evolver.feature_importance is not None
    assert evolver.signal_combiner is not None


@pytest.mark.asyncio
async def test_discover_alpha_signals_basic(signal_constraints):
    """Test basic alpha signal discovery."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    result = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=5
    )

    # Check that we got results
    assert result is not None
    assert len(result.surviving_signals) >= 0
    assert result.ensemble_signal is not None
    assert isinstance(result.feature_importance, dict)
    assert result.total_hypotheses_tested > 0


@pytest.mark.asyncio
async def test_alpha_signals_respect_constraints(signal_constraints):
    """Test that discovered signals respect constraints."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    result = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=3
    )

    # Check that surviving signals meet constraints
    for signal in result.surviving_signals:
        # Market cap constraint
        assert signal.backtest.avg_market_cap >= signal_constraints.min_market_cap, \
            f"Signal {signal.signal_id} violates market cap constraint"

        # Turnover constraint
        assert signal.backtest.turnover <= signal_constraints.max_turnover / 100, \
            f"Signal {signal.signal_id} violates turnover constraint"

        # Sharpe ratio constraint
        assert signal.sharpe_ratio >= signal_constraints.min_sharpe, \
            f"Signal {signal.signal_id} violates Sharpe constraint"

        # Max drawdown constraint
        assert signal.backtest.max_drawdown >= signal_constraints.max_drawdown, \
            f"Signal {signal.signal_id} violates drawdown constraint"


@pytest.mark.asyncio
async def test_alpha_signals_survive_crises(signal_constraints):
    """Test that signals survive required crisis periods."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    result = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=3
    )

    # Check crisis survival for each signal
    for signal in result.surviving_signals:
        for crisis_name in signal_constraints.crisis_periods:
            assert crisis_name in signal.crisis_performance, \
                f"Signal {signal.signal_id} missing crisis data for {crisis_name}"

            crisis_perf = signal.crisis_performance[crisis_name]
            assert crisis_perf.survived, \
                f"Signal {signal.signal_id} did not survive {crisis_name}"

            # Check that alpha wasn't terrible in crisis
            assert crisis_perf.alpha_pct > -0.10, \
                f"Signal {signal.signal_id} had bad alpha in {crisis_name}: {crisis_perf.alpha_pct}"


@pytest.mark.asyncio
async def test_alpha_signals_avoid_delistings(signal_constraints):
    """Test that signals avoid overfitting to delisted stocks."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    # Set stricter delisting constraint
    signal_constraints.max_delisting_rate = 0.005  # Max 0.5%

    result = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=3
    )

    # Check delisting rates
    for signal in result.surviving_signals:
        assert signal.backtest.delisting_rate < signal_constraints.max_delisting_rate, \
            f"Signal {signal.signal_id} has high delisting rate: {signal.backtest.delisting_rate}"


@pytest.mark.asyncio
async def test_feature_hypotheses_generation(signal_constraints):
    """Test that feature hypotheses are generated correctly."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    # Get predefined hypotheses
    hypotheses = evolver._get_predefined_hypotheses(
        universe="russell_3000",
        constraints=signal_constraints
    )

    # Check that we got hypotheses
    assert len(hypotheses) > 0

    # Check hypothesis structure
    for hyp in hypotheses:
        assert hyp.hypothesis_id is not None
        assert hyp.feature_name is not None
        assert hyp.feature_definition is not None
        assert hyp.alpha_source in AlphaSource
        assert hyp.rationale is not None
        assert len(hyp.failure_modes) > 0
        assert 0 <= hyp.expected_correlation <= 1
        assert 0 <= hyp.confidence <= 1


# ============================================================================
# TESTS: FEATURE IMPORTANCE TRACKER
# ============================================================================

@pytest.mark.asyncio
async def test_feature_importance_extraction(multiple_mock_signals):
    """Test feature importance extraction from signals."""
    tracker = FeatureImportanceTracker()

    importance = await tracker.extract_importance(multiple_mock_signals)

    # Check that we got importance scores
    assert len(importance) > 0

    # Check that scores are between 0 and 1
    for feature_name, score in importance.items():
        assert 0 <= score <= 1, f"Invalid score for {feature_name}: {score}"


@pytest.mark.asyncio
async def test_feature_importance_ranking(multiple_mock_signals):
    """Test feature ranking by importance."""
    tracker = FeatureImportanceTracker()

    # Extract importance
    await tracker.extract_importance(multiple_mock_signals)

    # Get ranked features
    ranked = tracker.get_ranked_features(top_n=5, min_frequency=1)

    # Check ranking
    assert len(ranked) <= 5

    # Check that scores are in descending order
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"


@pytest.mark.asyncio
async def test_crisis_robust_features(multiple_mock_signals):
    """Test identification of crisis-robust features."""
    tracker = FeatureImportanceTracker()

    # Extract importance
    await tracker.extract_importance(multiple_mock_signals)

    # Get crisis-robust features for GFC
    robust_features = tracker.get_crisis_robust_features(
        crisis="gfc",
        min_success_rate=0.5
    )

    # Check that we got results
    assert isinstance(robust_features, list)


@pytest.mark.asyncio
async def test_redundant_feature_detection(multiple_mock_signals):
    """Test detection of redundant (highly correlated) features."""
    tracker = FeatureImportanceTracker()

    # Extract importance
    await tracker.extract_importance(multiple_mock_signals)

    # Find redundant features
    redundant = tracker.find_redundant_features(threshold=0.8)

    # Check that we got a list
    assert isinstance(redundant, list)


# ============================================================================
# TESTS: SIGNAL COMBINER
# ============================================================================

@pytest.mark.asyncio
async def test_weighted_signal_combination(multiple_mock_signals):
    """Test weighted signal combination."""
    combiner = SignalCombiner()

    ensemble = await combiner.combine(
        signals=multiple_mock_signals,
        method="weighted",
        ensemble_name="Test Weighted Ensemble"
    )

    # Check ensemble properties
    assert ensemble is not None
    assert ensemble.ensemble_id is not None
    assert ensemble.name == "Test Weighted Ensemble"
    assert ensemble.combination_method == "information_ratio_weighted"
    assert len(ensemble.signals) == len(multiple_mock_signals)
    assert len(ensemble.weights) == len(multiple_mock_signals)

    # Check that weights sum to 1 (approximately)
    total_weight = sum(ensemble.weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"


@pytest.mark.asyncio
async def test_rank_signal_combination(multiple_mock_signals):
    """Test rank-based signal combination."""
    combiner = SignalCombiner()

    ensemble = await combiner.combine(
        signals=multiple_mock_signals,
        method="rank",
        ensemble_name="Test Rank Ensemble"
    )

    # Check ensemble properties
    assert ensemble is not None
    assert ensemble.combination_method == "rank_based"

    # Check that weights are equal (rank combination)
    weights = list(ensemble.weights.values())
    expected_weight = 1.0 / len(multiple_mock_signals)
    for weight in weights:
        assert abs(weight - expected_weight) < 0.01, \
            f"Expected equal weights, got {weight}"


@pytest.mark.asyncio
async def test_ensemble_diversification_ratio(multiple_mock_signals):
    """Test that ensemble has reasonable diversification ratio."""
    combiner = SignalCombiner()

    ensemble = await combiner.combine(
        signals=multiple_mock_signals,
        method="weighted"
    )

    # Diversification ratio should be >= 1.0
    assert ensemble.diversification_ratio >= 1.0, \
        f"Diversification ratio {ensemble.diversification_ratio} < 1.0"


@pytest.mark.asyncio
async def test_hierarchical_combination(multiple_mock_signals):
    """Test hierarchical risk parity combination."""
    combiner = SignalCombiner()

    ensemble = await combiner.combine(
        signals=multiple_mock_signals,
        method="hierarchical"
    )

    # Check ensemble
    assert ensemble is not None
    assert ensemble.combination_method == "hierarchical_risk_parity"

    # Check weights sum to 1
    total_weight = sum(ensemble.weights.values())
    assert abs(total_weight - 1.0) < 0.01


# ============================================================================
# TESTS: EXAMPLE SIGNALS
# ============================================================================

def test_example_signals_exist():
    """Test that example signals are defined."""
    from openevolve.finance.verticals.hedge_fund.examples import get_example_signals

    signals = get_example_signals()

    assert len(signals) > 0, "No example signals found"

    # Check signal structure
    for signal in signals:
        assert "signal_id" in signal
        assert "name" in signal
        assert "description" in signal
        assert "formula" in signal
        assert "alpha_source" in signal
        assert "information_ratio" in signal
        assert "sharpe_ratio" in signal


def test_example_signal_retrieval():
    """Test retrieving specific example signals."""
    from openevolve.finance.verticals.hedge_fund.examples import get_signal_by_id

    # Test retrieving a known signal
    signal = get_signal_by_id("earnings_credit_momentum")

    assert signal is not None
    assert signal["signal_id"] == "earnings_credit_momentum"
    assert "credit" in signal["name"].lower()


# ============================================================================
# TESTS: INTEGRATION
# ============================================================================

@pytest.mark.asyncio
async def test_full_alpha_discovery_pipeline(signal_constraints):
    """Test the complete alpha discovery pipeline."""
    # Initialize evolver
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    # Discover signals
    result = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=5
    )

    # Check that we got a complete result
    assert result is not None
    assert len(result.surviving_signals) >= 0
    assert result.ensemble_signal is not None
    assert isinstance(result.feature_importance, dict)
    assert isinstance(result.crisis_performance, dict)

    # Check that ensemble is valid
    ensemble = result.ensemble_signal
    assert ensemble.backtest is not None
    assert ensemble.sharpe_ratio > 0
    assert ensemble.information_ratio > 0


@pytest.mark.asyncio
async def test_evolution_persistence_across_cycles(signal_constraints):
    """Test that feature importance persists across evolution cycles."""
    evolver = AlphaSignalEvolver(config={
        "data_source": "MOCK",
        "include_delisted": True
    })

    # Run first evolution
    result1 = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=3
    )

    # Run second evolution
    result2 = await evolver.discover_alpha_signals(
        universe="russell_3000",
        constraints=signal_constraints,
        num_hypotheses=3
    )

    # Check that feature importance tracker has learned
    assert evolver.feature_importance.evolution_count == 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
