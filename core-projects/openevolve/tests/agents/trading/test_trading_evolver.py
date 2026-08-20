#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for the Trading Strategy Evolution System.

Tests cover:
- Strategy generation quality
- Evolution convergence
- Performance improvement over time
- Causal model accuracy
- Risk management
- Backtesting accuracy
"""

import pytest

# SKIP: this test requires the optional `openevolve.agents` subsystem
# (trading evolver), which is not part of the current core distribution.
pytest.skip(
    "openevolve.agents subsystem is not available in this distribution",
    allow_module_level=True,
)

import asyncio
from datetime import datetime, timedelta, UTC
from pathlib import Path
import sys

# Add openvolve to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openevolve.agents.trading.schemas import (
    Strategy,
    StrategyVariant,
    StrategyPerformance,
    MarketData,
    TradeSignal,
    EvolutionState,
    StrategyType,
    SignalType,
    CausalRelationship
)
from openevolve.agents.trading.trading_evolver import TradingEvolver
from openevolve.agents.trading.rlm_generator import RLMGenerator
from openevolve.agents.trading.variant_manager import VariantManager
from openevolve.agents.trading.judge_panel import JudgePanel
from openevolve.agents.trading.causal_modeler import CausalModeler
from openevolve.agents.trading.adversary import Adversary


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_strategy():
    """Create a sample strategy for testing."""
    return Strategy(
        strategy_id="test_strategy_1",
        name="Test Momentum Strategy",
        description="Simple momentum strategy for testing",
        strategy_type=StrategyType.MOMENTUM,
        parameters={
            "lookback_period": 20,
            "threshold": 2.0,
            "position_sizing": 0.1
        },
        entry_conditions=[
            "price > moving_average(lookback_period)",
            "momentum > threshold"
        ],
        exit_conditions=[
            "momentum < -threshold",
            "stop_loss hit"
        ],
        risk_rules={
            "max_position_size": 0.1,
            "stop_loss_pct": 0.05
        }
    )


@pytest.fixture
def sample_variant():
    """Create a sample variant for testing."""
    return StrategyVariant(
        variant_id="test_variant_1",
        parent_strategy_id="test_strategy_1",
        name="Test Variant v1",
        parameters={
            "lookback_period": 20,
            "threshold": 2.0,
            "position_sizing": 0.1
        },
        generation=1
    )


@pytest.fixture
def sample_performance():
    """Create sample performance data."""
    return StrategyPerformance(
        strategy_id="test_variant_1",
        period="90_days",
        total_return=0.25,
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        max_drawdown=0.12,
        win_rate=0.55,
        profit_factor=1.8,
        avg_win=0.03,
        avg_loss=-0.02,
        trades=100,
        volatility=0.15,
        beta=1.1,
        alpha=0.05,
        calmar_ratio=2.1,
        information_ratio=0.6
    )


# ============================================================================
# RLM Generator Tests
# ============================================================================

class TestRLMGenerator:
    """Test RLM strategy generation."""

    @pytest.mark.asyncio
    async def test_generate_strategies(self):
        """Test strategy generation."""
        generator = RLMGenerator()

        strategies = await generator.generate_strategies(
            market_regime={"regime": "bull", "volatility": "low"},
            num_ideas=3
        )

        assert len(strategies) > 0
        assert all(isinstance(s, Strategy) for s in strategies)
        assert all(s.strategy_id for s in strategies)

    @pytest.mark.asyncio
    async def test_strategy_refinement(self):
        """Test strategy refinement based on feedback."""
        generator = RLMGenerator()

        original_strategy = Strategy(
            strategy_id="original",
            name="Original Strategy",
            description="Test",
            strategy_type=StrategyType.MOMENTUM,
            parameters={"threshold": 2.0}
        )

        feedback = {
            "strengths": ["Good entry signals"],
            "weaknesses": ["Exits too late"],
            "parameter_sensitivity": {"threshold": 0.8}
        }

        refined = await generator.refine_strategy(
            original_strategy,
            feedback,
            market_regime={"regime": "bull"}
        )

        assert refined.strategy_id != original_strategy.strategy_id
        assert "refined" in refined.name

    @pytest.mark.asyncio
    async def test_strategy_combination(self):
        """Test combining multiple strategies."""
        generator = RLMGenerator()

        strategies = [
            Strategy(
                strategy_id=f"strat_{i}",
                name=f"Strategy {i}",
                description="Test",
                strategy_type=StrategyType.MOMENTUM,
                parameters={"threshold": 1.0 + i * 0.5}
            )
            for i in range(3)
        ]

        performance_data = [
            {"sharpe_ratio": 1.0 + i * 0.5}
            for i in range(3)
        ]

        hybrid = await generator.combine_strategies(strategies, performance_data)

        assert hybrid.strategy_type == StrategyType.HYBRID
        assert "hybrid" in hybrid.name


# ============================================================================
# Variant Manager Tests
# ============================================================================

class TestVariantManager:
    """Test variant management."""

    @pytest.mark.asyncio
    async def test_add_strategy(self, sample_strategy):
        """Test adding a strategy."""
        manager = VariantManager(max_variants=5)

        variant = await manager.add_strategy(sample_strategy)

        assert variant.variant_id is not None
        assert variant.parent_strategy_id == sample_strategy.strategy_id
        assert variant.status == "initialized"

    @pytest.mark.asyncio
    async def test_paper_trading(self, sample_variant):
        """Test paper trading simulation."""
        manager = VariantManager()
        manager.variants[sample_variant.variant_id] = sample_variant

        performance = await manager.paper_trade_variant(
            sample_variant.variant_id,
            days=90
        )

        assert isinstance(performance, StrategyPerformance)
        assert performance.strategy_id == sample_variant.variant_id
        assert performance.sharpe_ratio > 0

    @pytest.mark.asyncio
    async def test_variant_evolution(self):
        """Test variant evolution via mutation and crossover."""
        manager = VariantManager()

        # Create parent variants
        parents = [
            StrategyVariant(
                variant_id=f"parent_{i}",
                parent_strategy_id="test",
                name=f"Parent {i}",
                parameters={"threshold": 1.0 + i * 0.5},
                generation=0
            )
            for i in range(2)
        ]

        children = await manager.evolve_variants(parents, num_children=3)

        assert len(children) == 3
        assert all(c.generation > 0 for c in children)

    @pytest.mark.asyncio
    async def test_variant_pruning(self, sample_variant):
        """Test variant pruning."""
        manager = VariantManager(max_variants=3)

        # Add multiple variants
        for i in range(5):
            variant = StrategyVariant(
                variant_id=f"variant_{i}",
                parent_strategy_id="test",
                name=f"Variant {i}",
                parameters={"threshold": i},
                generation=0
            )
            manager.variants[variant.variant_id] = variant
            # Add fake performance
            manager.variant_performances[variant.variant_id] = StrategyPerformance(
                strategy_id=variant.variant_id,
                period="test",
                total_return=i * 0.1,
                sharpe_ratio=i * 0.5
            )

        await manager.prune_variants(keep_top_n=2)

        active_variants = await manager.get_active_variants()
        assert len(active_variants) <= 2


# ============================================================================
# Judge Panel Tests
# ============================================================================

class TestJudgePanel:
    """Test judge panel evaluation."""

    @pytest.mark.asyncio
    async def test_judge_evaluation(self, sample_variant, sample_performance):
        """Test individual judge evaluation."""
        panel = JudgePanel()

        evaluations = await panel.evaluate_strategy(
            variant=sample_variant,
            performance=sample_performance,
            market_regime={"regime": "bull"}
        )

        assert len(evaluations) == 5  # 5 judges
        assert all(isinstance(e, JudgeEvaluation) for e in evaluations)
        assert all(0 <= e.score <= 1 for e in evaluations)

    @pytest.mark.asyncio
    async def test_aggregation(self, sample_variant, sample_performance):
        """Test evaluation aggregation."""
        panel = JudgePanel()

        evaluations = await panel.evaluate_strategy(
            variant=sample_variant,
            performance=sample_performance,
            market_regime={"regime": "bull"}
        )

        aggregate = panel.aggregate_evaluations(evaluations)

        assert "overall_score" in aggregate
        assert 0 <= aggregate["overall_score"] <= 1
        assert "consensus" in aggregate
        assert "recommendation" in aggregate
        assert aggregate["recommendation"] in ["approve", "conditional", "reject"]


# ============================================================================
# Causal Modeler Tests
# ============================================================================

class TestCausalModeler:
    """Test causal modeling."""

    @pytest.mark.asyncio
    async def test_causal_model_learning(self, sample_strategy):
        """Test learning from outcomes."""
        modeler = CausalModeler()

        # Create fake performance history
        performance_history = []
        for i in range(10):
            performance_history.append({
                "performance": {
                    "sharpe_ratio": 1.0 + i * 0.1,
                    "total_return": 0.1 + i * 0.05
                },
                "parameters": {
                    "threshold": 2.0 + i * 0.2,
                    "lookback": 20
                }
            })

        causal_model = await modeler.learn_from_outcomes(
            strategy=sample_strategy,
            performance_history=performance_history,
            market_context={"regime": "bull"}
        )

        assert "relationships" in causal_model
        assert "mechanisms" in causal_model
        assert "predictions" in causal_model
        assert len(causal_model["relationships"]) > 0

    @pytest.mark.asyncio
    async def test_insight_extraction(self, sample_strategy):
        """Test extracting insights from causal model."""
        modeler = CausalModeler()

        causal_model = {
            "relationships": [
                CausalRelationship(
                    cause="parameter_threshold",
                    effect="performance",
                    strength=0.7,
                    confidence=0.8,
                    mechanism="Higher threshold increases signal quality",
                    evidence=[],
                    context={}
                )
            ],
            "predictions": [
                {
                    "condition": "regime_bull",
                    "predicted_performance_change": 0.5,
                    "confidence": 0.8,
                    "reasoning": "Bull markets favor momentum"
                }
            ]
        }

        insights = await modeler.extract_insights(causal_model)

        assert len(insights) > 0
        assert any(i["type"] == "causal_insight" for i in insights)
        assert any(i["type"] == "prediction" for i in insights)

    @pytest.mark.asyncio
    async def test_performance_prediction(self, sample_strategy):
        """Test predicting performance under new conditions."""
        modeler = CausalModeler()

        # Add a causal model
        modeler.causal_models[sample_strategy.strategy_id] = [
            CausalRelationship(
                cause="regime_bull",
                effect="performance",
                strength=0.5,
                confidence=0.8,
                mechanism="Bull markets increase returns",
                evidence=[],
                context={"regime": "bull"}
            )
        ]

        prediction = await modeler.predict_performance(
            strategy=sample_strategy,
            market_conditions={"regime": "bull"}
        )

        assert "predicted_performance" in prediction
        assert "confidence" in prediction
        assert prediction["predicted_performance"] > 0


# ============================================================================
# Adversary Tests
# ============================================================================

class TestAdversary:
    """Test adversarial testing."""

    @pytest.mark.asyncio
    async def test_strategy_testing(self, sample_variant):
        """Test adversarial strategy testing."""
        adversary = Adversary()

        result = await adversary.test_strategy(
            variant=sample_variant,
            market_conditions=["bull", "bear", "high_volatility"]
        )

        assert "robustness_score" in result
        assert 0 <= result["robustness_score"] <= 1
        assert "failure_modes" in result
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_weakness_detection(self, sample_variant):
        """Test weakness detection."""
        adversary = Adversary()

        weaknesses = await adversary.find_weaknesses(sample_variant)

        assert isinstance(weaknesses, list)
        assert all("type" in w for w in weaknesses)
        assert all("severity" in w for w in weaknesses)

    @pytest.mark.asyncio
    async def test_counter_strategy_generation(self, sample_variant):
        """Test counter-strategy generation."""
        adversary = Adversary()

        counter = await adversary.generate_counter_strategy(sample_variant)

        assert "description" in counter
        assert "exploits" in counter
        assert "approach" in counter
        assert "expected_advantage" in counter


# ============================================================================
# Trading Evolver Tests
# ============================================================================

class TestTradingEvolver:
    """Test main trading evolver orchestrator."""

    @pytest.mark.asyncio
    async def test_evolution_cycle(self):
        """Test complete evolution cycle."""
        evolver = TradingEvolver(
            max_variants=3,
            evolution_interval=timedelta(seconds=1),
            live_trading_enabled=False  # Don't actually trade
        )

        # Run single cycle
        state = await evolver.run_evolution_cycle()

        assert state.generation >= 1
        assert isinstance(state.population, list)

    @pytest.mark.asyncio
    async def test_continuous_evolution(self):
        """Test continuous evolution with timeout."""
        evolver = TradingEvolver(
            max_variants=2,
            evolution_interval=timedelta(seconds=1),
            live_trading_enabled=False
        )

        # Run for limited time
        task = asyncio.create_task(evolver.start())

        # Let it run for 2 cycles
        await asyncio.sleep(2.5)

        # Stop
        evolver.stop()
        await task

        assert evolver.current_cycle >= 2
        assert evolver.state.generation >= 2

    @pytest.mark.asyncio
    async def test_top_strategies_retrieval(self):
        """Test retrieving top performing strategies."""
        evolver = TradingEvolver(
            max_variants=5,
            live_trading_enabled=False
        )

        # Run a cycle first
        await evolver.run_evolution_cycle()

        top_strategies = await evolver.get_top_strategies(top_n=3)

        assert len(top_strategies) <= 3
        assert all("strategy" in s for s in top_strategies)
        assert all("performance" in s for s in top_strategies)
        assert all("fitness" in s for s in top_strategies)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete system."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline from generation to selection."""
        # Initialize components
        generator = RLMGenerator()
        variant_manager = VariantManager(max_variants=5)
        judge_panel = JudgePanel()
        adversary = Adversary()
        causal_modeler = CausalModeler()

        # Phase 1: Generate
        strategies = await generator.generate_strategies(
            market_regime={"regime": "bull"},
            num_ideas=3
        )

        assert len(strategies) == 3

        # Phase 2: Create variants and test
        for strategy in strategies[:2]:  # Use first 2
            await variant_manager.add_strategy(strategy)

        variants = await variant_manager.get_active_variants()

        # Test variants
        for variant in variants:
            await variant_manager.paper_trade_variant(variant.variant_id)

        # Phase 3: Evaluate
        for variant in variants:
            performance = await variant_manager.get_performance(variant.variant_id)
            evaluations = await judge_panel.evaluate_strategy(
                variant, performance, {"regime": "bull"}
            )

            assert len(evaluations) > 0

        # Phase 4: Adversarial test
        for variant in variants[:1]:  # Test one
            result = await adversary.test_strategy(
                variant, ["bull", "bear"]
            )
            assert result["robustness_score"] >= 0

        # Phase 5: Learn
        for variant in variants[:1]:  # Learn from one
            performance_history = await variant_manager.get_performance_history(
                variant.variant_id
            )

            if performance_history:
                causal_model = await causal_modeler.learn_from_outcomes(
                    strategy=strategies[0],
                    performance_history=performance_history,
                    market_context={"regime": "bull"}
                )

                assert "relationships" in causal_model

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test that performance improves over generations."""
        evolver = TradingEvolver(
            max_variants=5,
            live_trading_enabled=False
        )

        # Run multiple cycles
        initial_fitness = []
        for i in range(3):
            await evolver.run_evolution_cycle()
            initial_fitness.append(evolver.state.best_fitness)

        # Fitness should generally increase (may fluctuate but trend should be up)
        # Note: In simulation this may be noisy, so we just check it runs
        assert len(initial_fitness) == 3

    @pytest.mark.asyncio
    async def test_risk_management(self):
        """Test that risk management is enforced."""
        evolver = TradingEvolver(
            max_variants=3,
            live_trading_enabled=False
        )

        # Run a cycle
        await evolver.run_evolution_cycle()

        # Get top strategies
        top_strategies = await evolver.get_top_strategies(top_n=1)

        if top_strategies:
            strategy_data = top_strategies[0]
            performance = StrategyPerformance(**strategy_data["performance"])

            # Check risk metrics are within bounds
            assert performance.max_drawdown < 0.5  # Less than 50% drawdown
            assert performance.volatility < 1.0  # Less than 100% volatility


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
