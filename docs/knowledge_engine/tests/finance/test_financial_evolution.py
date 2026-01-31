"""
Comprehensive tests for Financial Evolution Bridge.

Tests the core integration between LoongFlow (high-level reasoning)
and OpenEvolve (low-level evolution) for financial applications.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from knowledge_engine.finance import (
    FinancialEvolutionAgent,
    FinancialEvolutionMemory,
    CrisisAwareFitness,
    SurvivorshipBacktester
)

from knowledge_engine.finance.schemas import (
    Strategy,
    BacktestResult,
    CrisisLesson,
    EvolutionObjective,
    EvolutionBudget,
    CrisisType,
    StrategyType,
    MarketConditions,
    DelistingEvent
)


# =============================================================================
# Test FinancialEvolutionMemory
# =============================================================================

class TestFinancialEvolutionMemory:
    """Test financial evolution memory system"""

    @pytest.fixture
    def memory(self):
        """Create memory instance"""
        return FinancialEvolutionMemory()

    @pytest.fixture
    def sample_lesson(self):
        """Create sample crisis lesson"""
        return CrisisLesson(
            lesson_id="test_lesson_1",
            crisis=CrisisType.GFC,
            strategy_type=StrategyType.MOMENTUM,
            successful=True,
            lesson="Momentum strategies failed during GFC due to trend reversals",
            feature_importance={"volatility": 0.8, "trend": 0.9},
            boost_amount=0.5,
            conditions_met={
                "volatility_threshold": 0.25,
                "max_drawdown_threshold": 0.20,
                "trend_requirement": "negative"
            }
        )

    def test_store_lesson(self, memory, sample_lesson):
        """Test storing a lesson"""
        memory.store_lesson(sample_lesson)

        # Verify stored in crisis bucket
        assert CrisisType.GFC in memory.crisis_lessons
        assert len(memory.crisis_lessons[CrisisType.GFC]) == 1
        assert memory.crisis_lessons[CrisisType.GFC][0].lesson_id == "test_lesson_1"

    def test_get_relevant_lessons(self, memory, sample_lesson):
        """Test retrieving relevant lessons"""
        memory.store_lesson(sample_lesson)

        conditions = MarketConditions(
            volatility=0.30,
            trend="down",
            resembles_crisis=CrisisType.GFC
        )

        relevant = memory.get_relevant_lessons(conditions)

        assert len(relevant) >= 1
        assert any(l.lesson_id == "test_lesson_1" for l in relevant)

    def test_condition_matching(self, memory, sample_lesson):
        """Test lesson condition matching"""
        memory.store_lesson(sample_lesson)

        conditions = MarketConditions(
            volatility=0.30,  # Above threshold
            trend="down",
            resembles_crisis=CrisisType.GFC
        )

        relevant = memory.get_relevant_lessons(conditions)

        # Should match because volatility > 0.25 threshold
        assert len(relevant) >= 1

    def test_strategy_lineage(self, memory):
        """Test strategy lineage tracking"""
        memory.add_strategy_lineage(
            parent_id=None,
            child_id="strategy_1",
            strategy_type=StrategyType.MOMENTUM
        )

        memory.add_strategy_lineage(
            parent_id="strategy_1",
            child_id="strategy_2",
            strategy_type=StrategyType.MOMENTUM
        )

        lineage = memory.get_strategy_lineage("strategy_2")

        assert len(lineage) >= 2


# =============================================================================
# Test CrisisAwareFitness
# =============================================================================

class TestCrisisAwareFitness:
    """Test crisis-aware fitness function"""

    @pytest.fixture
    def fitness(self, memory):
        """Create fitness instance"""
        crisis_periods = [
            ("2007-09-01", "2009-03-31", CrisisType.GFC),
            ("2020-02-01", "2020-04-30", CrisisType.COVID)
        ]

        return CrisisAwareFitness(
            crisis_periods=crisis_periods,
            memory=memory
        )

    @pytest.fixture
    def memory(self):
        """Create memory instance"""
        return FinancialEvolutionMemory()

    @pytest.fixture
    def sample_backtest(self):
        """Create sample backtest result"""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.02, 0.01, 0.01]
        drawdowns = [0.0, -0.01, -0.02, -0.01, -0.005, -0.015, -0.01, -0.005, 0.0, 0.0]

        return BacktestResult(
            strategy_id="test_strategy",
            returns=returns,
            drawdowns=drawdowns,
            delistings=[],
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            final_wealth=1.2,
            volatility=0.15,
            total_trades=10,
            win_rate=0.6,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )

    def test_base_fitness_calculation(self, fitness, sample_backtest):
        """Test base fitness calculation"""
        score = fitness.evaluate(sample_backtest)

        assert score.total_score > 0
        assert score.base_score > 0
        assert len(score.components) > 0

    def test_component_scores(self, fitness, sample_backtest):
        """Test fitness component scores"""
        score = fitness.evaluate(sample_backtest)

        # Check key components
        assert "sharpe_ratio" in score.components
        assert "max_drawdown" in score.components
        assert "final_wealth" in score.components
        assert "volatility" in score.components

    def test_learned_boost(self, fitness, memory, sample_backtest):
        """Test learned boost from memory"""
        # Store a lesson
        lesson = CrisisLesson(
            lesson_id="boost_lesson",
            crisis=CrisisType.GFC,
            strategy_type=StrategyType.MOMENTUM,
            successful=True,
            lesson="Test lesson",
            feature_importance={"volatility": 0.9},
            boost_amount=0.5,
            conditions_met={
                "volatility_threshold": 0.10,
                "trend_requirement": "positive"
            }
        )
        memory.store_lesson(lesson)

        # Evaluate with matching conditions
        conditions = MarketConditions(
            volatility=0.15,
            trend="up",
            resembles_crisis=CrisisType.GFC
        )

        score = fitness.evaluate(sample_backtest, conditions)

        # Should have boost
        assert score.learned_boost >= 0

    def test_lesson_creation(self, fitness, sample_backtest):
        """Test creating lessons from results"""
        lesson = fitness.update_lesson_from_result(
            result=sample_backtest,
            crisis_type=CrisisType.GFC,
            successful=True
        )

        assert lesson.crisis == CrisisType.GFC
        assert lesson.successful == True
        assert len(lesson.feature_importance) > 0
        assert len(lesson.conditions_met) > 0


# =============================================================================
# Test SurvivorshipBacktester
# =============================================================================

class TestSurvivorshipBacktester:
    """Test survivorship-aware backtester"""

    @pytest.fixture
    def backtester(self):
        """Create backtester instance"""
        return SurvivorshipBacktester(
            data_source="CRSP_SIMULATED",
            include_delisted=True
        )

    @pytest.fixture
    def sample_strategy(self):
        """Create sample strategy"""
        return Strategy(
            strategy_id="momentum_3m",
            strategy_type=StrategyType.MOMENTUM,
            parameters={"lookback": 3, "alpha": 0.01, "beta": 1.2},
            description="3-month momentum strategy"
        )

    @pytest.mark.asyncio
    async def test_backtest_execution(self, backtester, sample_strategy):
        """Test running a backtest"""
        result = await backtester.run(
            strategy=sample_strategy,
            period="2020-01-01:2020-12-31"
        )

        assert result.strategy_id == "momentum_3m"
        assert len(result.returns) > 0
        assert len(result.drawdowns) > 0
        assert result.sharpe_ratio != 0

    @pytest.mark.asyncio
    async def test_delisting_tracking(self, backtester):
        """Test delisting event tracking"""
        # Create strategy that might trigger delistings
        strategy = Strategy(
            strategy_id="risky_strategy",
            strategy_type=StrategyType.MOMENTUM,
            parameters={"lookback": 1, "beta": 2.0},  # High risk
            description="High-risk momentum"
        )

        result = await backtester.run(
            strategy=strategy,
            period="2007-01-01:2009-12-31"  # GFC period
        )

        # May have delistings
        assert isinstance(result.delistings, list)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, backtester):
        """Test parallel backtesting"""
        strategies = [
            Strategy(
                strategy_id=f"strategy_{i}",
                strategy_type=StrategyType.MOMENTUM,
                parameters={"lookback": i+1, "alpha": 0.01},
                description=f"Strategy {i}"
            )
            for i in range(5)
        ]

        results = await backtester.run_parallel(
            strategies=strategies,
            period="2020-01-01:2020-12-31"
        )

        assert len(results) == 5
        assert all(isinstance(r, BacktestResult) for r in results)


# =============================================================================
# Test FinancialEvolutionAgent (Integration Tests)
# =============================================================================

class TestFinancialEvolutionAgent:
    """Integration tests for financial evolution agent"""

    @pytest.fixture
    def agent(self):
        """Create agent instance"""
        config = {
            "backtester": {
                "data_source": "CRSP_SIMULATED",
                "include_delisted": True
            },
            "fitness": {
                "sharpe_weight": 2.0,
                "drawdown_weight": -5.0,
                "wealth_weight": 3.0,
                "crisis_weight": 5.0
            }
        }

        return FinancialEvolutionAgent(config=config)

    @pytest.fixture
    def sample_objective(self):
        """Create sample evolution objective"""
        return EvolutionObjective(
            universe="test_equities",
            crisis_periods=[
                ("2007-09-01", "2009-03-31", CrisisType.GFC),
                ("2020-02-01", "2020-04-30", CrisisType.COVID)
            ],
            survival_constraints={
                "max_drawdown": 0.30,
                "min_equity_final": 1.0,
                "delisting_penalty": -1000
            }
        )

    @pytest.fixture
    def sample_budget(self):
        """Create sample evolution budget"""
        return EvolutionBudget(
            iterations=10,  # Small for testing
            cost_cap=50,
            strategies_per_iteration=5
        )

    @pytest.mark.asyncio
    async def test_evolution_loop(self, agent, sample_objective, sample_budget):
        """Test full evolution loop"""
        result = await agent.evolve_strategies(
            objective=sample_objective,
            budget=sample_budget
        )

        # Validate results
        assert len(result.best_strategies) > 0
        assert result.iterations_completed > 0
        assert result.final_cost <= sample_budget.cost_cap
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_crisis_survivor_evolution(self, agent, sample_objective, sample_budget):
        """Test evolution of crisis-surviving strategies"""
        result = await agent.evolve_strategies(
            objective=sample_objective,
            budget=sample_budget
        )

        # Check that best strategies exist
        assert len(result.best_strategies) > 0

        # Check that lessons were learned
        assert len(result.lessons_learned) > 0

        # Check budget constraint
        assert result.final_cost <= sample_budget.cost_cap

    @pytest.mark.asyncio
    async def test_budget_enforcement(self, agent, sample_objective):
        """Test that budget cap is enforced"""
        # Set very low budget
        budget = EvolutionBudget(
            iterations=100,
            cost_cap=1.0,  # Very low
            strategies_per_iteration=10
        )

        result = await agent.evolve_strategies(
            objective=sample_objective,
            budget=budget
        )

        # Should stop early due to budget
        assert result.final_cost <= budget.cost_cap
        assert result.iterations_completed < budget.iterations

    @pytest.mark.asyncio
    async def test_single_strategy_evaluation(self, agent):
        """Test evaluating a single strategy"""
        strategy = Strategy(
            strategy_id="test_strategy",
            strategy_type=StrategyType.MOMENTUM,
            parameters={"lookback": 3},
            description="Test momentum strategy"
        )

        result, score = await agent.evaluate_strategy(strategy)

        assert isinstance(result, BacktestResult)
        assert isinstance(score, float)
        assert len(result.returns) > 0

    def test_memory_persistence(self, agent, tmp_path):
        """Test memory persistence to disk"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            import os
            persistence_path = os.path.join(tmp_dir, "memory.json")

            # Create agent with persistence
            agent_with_persistence = FinancialEvolutionAgent(
                config={"memory": {"persistence_path": persistence_path}}
            )

            # Store a lesson
            lesson = CrisisLesson(
                lesson_id="test_lesson",
                crisis=CrisisType.GFC,
                strategy_type=StrategyType.MOMENTUM,
                successful=True,
                lesson="Test lesson",
                feature_importance={},
                boost_amount=0.5
            )

            agent_with_persistence.memory.store_lesson(lesson)

            # Create new agent with same path
            new_agent = FinancialEvolutionAgent(
                config={"memory": {"persistence_path": persistence_path}}
            )

            # Should have loaded the lesson
            assert CrisisType.GFC in new_agent.memory.crisis_lessons


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_parallel_backtesting_performance(self):
        """Test parallel backtesting performance"""
        import time

        backtester = SurvivorshipBacktester()

        strategies = [
            Strategy(
                strategy_id=f"perf_test_{i}",
                strategy_type=StrategyType.MOMENTUM,
                parameters={"lookback": i % 10 + 1},
                description=f"Performance test {i}"
            )
            for i in range(20)
        ]

        start = time.time()
        results = await backtester.run_parallel(
            strategies=strategies,
            period="2020-01-01:2020-12-31"
        )
        elapsed = time.time() - start

        assert len(results) == 20
        # Should complete in reasonable time
        assert elapsed < 30  # 30 seconds for 20 strategies

    @pytest.mark.asyncio
    async def test_memory_scaling(self):
        """Test memory performance with many lessons"""
        memory = FinancialEvolutionMemory()

        # Add many lessons
        for i in range(1000):
            lesson = CrisisLesson(
                lesson_id=f"lesson_{i}",
                crisis=CrisisType.GFC if i % 2 == 0 else CrisisType.COVID,
                strategy_type=StrategyType.MOMENTUM,
                successful=i % 3 == 0,  # 1/3 successful
                lesson=f"Lesson {i}",
                feature_importance={"feature_1": 0.5},
                boost_amount=0.1
            )
            memory.store_lesson(lesson)

        # Test retrieval performance
        conditions = MarketConditions(
            volatility=0.3,
            trend="down",
            resembles_crisis=CrisisType.GFC
        )

        import time
        start = time.time()
        relevant = memory.get_relevant_lessons(conditions)
        elapsed = time.time() - start

        # Should retrieve quickly
        assert elapsed < 1.0  # 1 second max
        assert len(relevant) > 0
