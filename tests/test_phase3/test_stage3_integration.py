"""
Integration Tests for Stage 3 Monte Carlo Nest

Tests for complete integration of Γ₁ (ACI), Γ₂ (MCTS), and Γ₃ (Statistical Validation)

Author: Agent D2 (Γ₂/Γ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Testing
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase3.stage3_integration import (
    MonteCarloNest, NestConfig, NestResult, AgentResult,
    AgentStrategy, quick_nest_search
)
from phase3.mcts_search import MCTSState


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleOptimizationState(MCTSState):
    """Simple state for integration testing"""

    def __init__(self, value=0, depth=0, max_depth=10):
        self._value = value
        self._depth = depth
        self._max_depth = max_depth
        super().__init__(
            variables={'value': value},
            unassigned=[],
            domains={},
            satisfied=(depth >= max_depth),
            depth=depth
        )

    @property
    def value(self):
        return self._value

    def is_terminal(self):
        return self._depth >= self._max_depth


@pytest.fixture
def nest_config():
    """Standard nest configuration for testing"""
    return NestConfig(
        num_agents=4,
        mcts_iterations=100,
        mcts_playout_depth=10,
        verbose=False,
        validate_results=True
    )


@pytest.fixture
def initial_state():
    """Initial state for testing"""
    return SimpleOptimizationState(value=0, depth=0)


@pytest.fixture
def action_generator():
    """Generate actions"""
    def generate(state):
        if state.is_terminal():
            return []
        return ['+1', '-1', '+2', '-2']
    return generate


@pytest.fixture
def state_transition():
    """Transition function"""
    def transition(state, action):
        new_value = state._value + int(action)
        new_depth = state._depth + 1
        return SimpleOptimizationState(new_value, new_depth, state._max_depth)
    return transition


@pytest.fixture
def value_function():
    """Value function"""
    def evaluate(state):
        return state._value
    return evaluate


# ============================================================================
# Monte Carlo Nest Tests
# ============================================================================

class TestMonteCarloNest:
    """Tests for Monte Carlo Nest"""

    def test_nest_initialization(self, nest_config):
        """Test nest initialization"""
        nest = MonteCarloNest(config=nest_config)
        assert nest.config == nest_config
        assert nest.validator is not None

    def test_basic_search(self, nest_config, initial_state,
                         action_generator, state_transition, value_function):
        """Test basic nest search"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert isinstance(result, NestResult)
        assert result.best_agent_result is not None
        assert len(result.all_agent_results) == nest_config.num_agents
        assert result.elapsed_time > 0

    def test_agent_diversity(self, nest_config, initial_state,
                            action_generator, state_transition, value_function):
        """Test that agents use different strategies"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        strategies = [r.strategy for r in result.all_agent_results]

        # Should have different strategies
        assert len(set(strategies)) > 1

    def test_aggregation(self, nest_config, initial_state,
                       action_generator, state_transition, value_function):
        """Test result aggregation"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Aggregated value should be reasonable
        all_values = [r.best_value for r in result.all_agent_results]
        assert min(all_values) <= result.aggregated_value <= max(all_values)

    def test_parallel_vs_sequential(self, initial_state,
                                   action_generator, state_transition, value_function):
        """Compare parallel and sequential execution"""
        # Parallel
        config_parallel = NestConfig(
            num_agents=4,
            mcts_iterations=100,
            parallel_agents=True,
            verbose=False
        )
        nest_parallel = MonteCarloNest(config=config_parallel)
        result_parallel = nest_parallel.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Sequential
        config_sequential = NestConfig(
            num_agents=4,
            mcts_iterations=100,
            parallel_agents=False,
            verbose=False
        )
        nest_sequential = MonteCarloNest(config=config_sequential)
        result_sequential = nest_sequential.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Both should complete successfully
        assert result_parallel.best_agent_result.best_value is not None
        assert result_sequential.best_agent_result.best_value is not None

        # Parallel should be faster (usually)
        # But we don't assert this due to test environment variability


# ============================================================================
# Agent Strategy Tests
# ============================================================================

class TestAgentStrategies:
    """Tests for different agent strategies"""

    def test_exploit_strategy(self, initial_state,
                             action_generator, state_transition, value_function):
        """Test exploit-heavy agent"""
        config = NestConfig(
            num_agents=1,
            agent_strategies=[AgentStrategy.EXPLOIT],
            mcts_iterations=100,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert result.all_agent_results[0].strategy == AgentStrategy.EXPLOIT

    def test_explore_strategy(self, initial_state,
                             action_generator, state_transition, value_function):
        """Test explore-heavy agent"""
        config = NestConfig(
            num_agents=1,
            agent_strategies=[AgentStrategy.EXPLORE],
            mcts_iterations=100,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert result.all_agent_results[0].strategy == AgentStrategy.EXPLORE

    def test_mixed_strategies(self, initial_state,
                             action_generator, state_transition, value_function):
        """Test mixed strategies"""
        config = NestConfig(
            num_agents=4,
            agent_strategies=[
                AgentStrategy.EXPLOIT,
                AgentStrategy.EXPLORE,
                AgentStrategy.BALANCED,
                AgentStrategy.ADAPTIVE
            ],
            mcts_iterations=100,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert len(result.all_agent_results) == 4
        strategies = [r.strategy for r in result.all_agent_results]
        assert AgentStrategy.EXPLOIT in strategies
        assert AgentStrategy.BALANCED in strategies
        assert AgentStrategy.ADAPTIVE in strategies


# ============================================================================
# Validation Tests
# ============================================================================

class TestNestValidation:
    """Tests for validation in nest context"""

    def test_validation_enabled(self, initial_state,
                                action_generator, state_transition, value_function):
        """Test with validation enabled"""
        config = NestConfig(
            num_agents=2,
            mcts_iterations=100,
            validate_results=True,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # All agents should have validation results
        for agent_result in result.all_agent_results:
            assert agent_result.validation is not None

    def test_validation_disabled(self, initial_state,
                                action_generator, state_transition, value_function):
        """Test with validation disabled"""
        config = NestConfig(
            num_agents=2,
            mcts_iterations=100,
            validate_results=False,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # No validation results
        for agent_result in result.all_agent_results:
            assert agent_result.validation is None

    def test_confident_result_selection(self, initial_state,
                                      action_generator, state_transition, value_function):
        """Test that confident results are preferred"""
        config = NestConfig(
            num_agents=4,
            mcts_iterations=100,
            validate_results=True,
            verbose=False
        )

        nest = MonteCarloNest(config=config)
        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Best result should be from confident agents if available
        confident_results = [r for r in result.all_agent_results if r.is_confident]
        if confident_results:
            # Best should be one of the confident ones
            assert result.best_agent_result in confident_results or \
                   result.best_agent_result.best_value >= max(r.best_value for r in confident_results)


# ============================================================================
# Performance Tests
# ============================================================================

class TestNestPerformance:
    """Performance tests for Monte Carlo Nest"""

    def test_scalability_with_agents(self, initial_state,
                                    action_generator, state_transition, value_function):
        """Test scalability with different numbers of agents"""
        results = []

        for num_agents in [1, 2, 4]:
            config = NestConfig(
                num_agents=num_agents,
                mcts_iterations=50,
                parallel_agents=True,
                verbose=False
            )

            nest = MonteCarloNest(config=config)
            result = nest.search(
                initial_state,
                action_generator,
                state_transition,
                value_function
            )

            results.append((num_agents, result.elapsed_time))

        # More agents shouldn't drastically increase time (if parallel)
        # But we don't assert strictly due to test environment variability
        assert len(results) == 3

    def test_large_search(self):
        """Test with larger search space"""
        initial = SimpleOptimizationState(value=0, depth=0, max_depth=20)

        config = NestConfig(
            num_agents=4,
            mcts_iterations=200,
            mcts_playout_depth=20,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            initial,
            lambda s: ['+1', '-1', '+2', '-2', '+3', '-3'] if not s.is_terminal() else [],
            lambda s, a: SimpleOptimizationState(s._value + int(a), s._depth + 1, s._max_depth),
            lambda s: s._value
        )

        # Should complete successfully
        assert result.best_agent_result.best_value is not None
        assert result.elapsed_time < 60  # Should complete within 1 minute


# ============================================================================
# Result Analysis Tests
# ============================================================================

class TestResultAnalysis:
    """Tests for result analysis and reporting"""

    def test_result_summary(self, nest_config, initial_state,
                          action_generator, state_transition, value_function):
        """Test result summary generation"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        summary = result.summary()

        assert "Best Agent" in summary
        assert "Best Value" in summary
        assert "Aggregated Value" in summary
        assert "Confidence" in summary
        assert "Elapsed Time" in summary

    def test_metadata(self, nest_config, initial_state,
                    action_generator, state_transition, value_function):
        """Test result metadata"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert 'initial_aci' in result.metadata
        assert 'num_agents' in result.metadata
        assert 'agent_strategies' in result.metadata
        assert len(result.metadata['agent_strategies']) == nest_config.num_agents


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_quick_nest_search(self, initial_state,
                              action_generator, state_transition, value_function):
        """Test quick nest search convenience function"""
        result = quick_nest_search(
            initial_state,
            action_generator,
            state_transition,
            value_function,
            num_agents=4,
            iterations_per_agent=100
        )

        assert isinstance(result, NestResult)
        assert len(result.all_agent_results) == 4


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests"""

    def test_complete_pipeline(self):
        """Test complete pipeline with all components"""
        # Create more complex problem
        initial = SimpleOptimizationState(value=0, depth=0, max_depth=15)

        config = NestConfig(
            num_agents=4,
            mcts_iterations=150,
            validate_results=True,
            parallel_agents=True,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            initial,
            lambda s: [f'+{i}' for i in range(1, 4)] + [f'-{i}' for i in range(1, 4)]
                      if not s.is_terminal() else [],
            lambda s, a: SimpleOptimizationState(s._value + int(a), s._depth + 1, s._max_depth),
            lambda s: s._value
        )

        # Complete pipeline should work
        assert result.best_agent_result is not None
        assert result.aggregated_value is not None
        assert result.metadata is not None
        assert result.elapsed_time > 0

        # All agents should have completed
        assert len(result.all_agent_results) == config.num_agents

        # Validation should have been performed
        if config.validate_results:
            assert all(r.validation is not None for r in result.all_agent_results)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
