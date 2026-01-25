"""
Comprehensive unit tests for Stage 3 Integration (Monte Carlo Nest)

Tests the integration of Γ₁ (ACI Analyzer), Γ₂ (MCTS Search), and Γ₃ (Statistical Validator)
in the Monte Carlo Nest architecture.

Author: Agent D2 (Γ₂/Γ₃ Integration Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch

# Try to import stage3 integration
try:
    from rese.phase3.stage3_integration import (
        MonteCarloNest,
        NestConfig,
        NestResult,
        AgentResult,
        AgentStrategy,
        MCTSState,
        quick_nest_search
    )
except ImportError:
    pytest.skip("Stage 3 integration module not available", allow_module_level=True)

# Try to import MCTS components
try:
    from rese.phase3.mcts_search import (
        MCTSConfig,
        MCTSSearch,
        MCTSNode,
        PlayoutStrategy
    )
except ImportError:
    MCTSConfig = None
    MCTSSearch = None
    MCTSNode = None
    PlayoutStrategy = None


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_state():
    """Create simple MCTS state"""
    if MCTSState:
        return MCTSState(
            variables={'x': 0},
            unassigned=['y', 'z'],
            depth=0,
            satisfied=False
        )
    return None


@pytest.fixture
def nest_config():
    """Create Monte Carlo Nest configuration"""
    return NestConfig(
        num_agents=3,
        mcts_iterations=100,
        mcts_playout_depth=20,
        validate_results=True,
        verbose=False
    )


@pytest.fixture
def mock_aci_analyzer():
    """Create mock ACI analyzer"""
    analyzer = Mock()
    analyzer.calculate.return_value = {
        'ACI': 0.65,
        'confidence': 0.9,
        'disorder_entropy': 0.5,
        'causal_coherence': 0.7
    }
    return analyzer


@pytest.fixture
def basic_action_generator():
    """Simple action generator for testing"""
    def actions(state):
        if state.is_terminal():
            return []
        return ['action1', 'action2', 'action3']
    return actions


@pytest.fixture
def basic_state_transition():
    """Simple state transition for testing"""
    def transition(state, action):
        if MCTSState:
            new_vars = state.variables.copy()
            new_vars['x'] = new_vars.get('x', 0) + 1
            return MCTSState(
                variables=new_vars,
                unassigned=state.unassigned[1:] if state.unassigned else [],
                depth=state.depth + 1
            )
        return state
    return transition


@pytest.fixture
def basic_value_function():
    """Simple value function for testing"""
    def value_fn(state):
        return float(state.variables.get('x', 0))
    return value_fn


# =============================================================================
# NestConfig Tests
# =============================================================================

class TestNestConfig:
    """Test NestConfig functionality"""

    def test_default_values(self):
        """Test default configuration"""
        config = NestConfig()

        assert config.num_agents == 4
        assert config.mcts_iterations == 500
        assert config.mcts_playout_depth == 50
        assert config.aci_guided == True
        assert config.early_stopping == True
        assert config.validate_results == True
        assert config.confidence_level == 0.95
        assert config.parallel_agents == True
        assert config.max_workers == 4
        assert config.max_time_seconds == 300.0
        assert config.convergence_required == True

    def test_custom_values(self):
        """Test custom configuration"""
        config = NestConfig(
            num_agents=2,
            mcts_iterations=200,
            validate_results=False,
            verbose=True
        )

        assert config.num_agents == 2
        assert config.mcts_iterations == 200
        assert config.validate_results == False
        assert config.verbose == True


# =============================================================================
# AgentResult Tests
# =============================================================================

class TestAgentResult:
    """Test AgentResult functionality"""

    def test_is_confident_with_validation(self):
        """Test is_confident with confident validation"""
        from rese.phase3.statistical_validator import ValidationResult, ConfidenceInterval, ConvergenceResult

        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.8,
            level=0.95,
            method=None,
            width=0.1
        )

        conv = ConvergenceResult(
            converged=True,
            method=None,
            iteration=100,
            confidence=0.95,
            details={}
        )

        validation = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        result = AgentResult(
            agent_id=0,
            strategy=AgentStrategy.EXPLOIT,
            best_value=0.75,
            best_node=None,
            search_info={},
            validation=validation
        )

        assert result.is_confident()

    def test_is_confident_without_validation(self):
        """Test is_confident without validation"""
        result = AgentResult(
            agent_id=0,
            strategy=AgentStrategy.EXPLOIT,
            best_value=0.75,
            best_node=None,
            search_info={}
        )

        assert not result.is_confident()

    def test_is_confident_not_converged(self):
        """Test is_confident when not converged"""
        from rese.phase3.statistical_validator import ValidationResult, ConfidenceInterval, ConvergenceResult

        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.8,
            level=0.95,
            method=None,
            width=0.1
        )

        conv = ConvergenceResult(
            converged=False,
            method=None,
            iteration=50,
            confidence=0.5,
            details={}
        )

        validation = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        result = AgentResult(
            agent_id=0,
            strategy=AgentStrategy.EXPLOIT,
            best_value=0.75,
            best_node=None,
            search_info={},
            validation=validation
        )

        assert not result.is_confident()


# =============================================================================
# MonteCarloNest Tests
# =============================================================================

class TestMonteCarloNest:
    """Test MonteCarloNest functionality"""

    def test_initialization(self, nest_config):
        """Test nest initialization"""
        nest = MonteCarloNest(config=nest_config)

        assert nest.config == nest_config
        assert nest.aci_analyzer is None
        assert nest.validator is not None

    def test_initialization_with_aci_analyzer(self, nest_config, mock_aci_analyzer):
        """Test nest initialization with ACI analyzer"""
        nest = MonteCarloNest(
            config=nest_config,
            aci_analyzer=mock_aci_analyzer
        )

        assert nest.aci_analyzer == mock_aci_analyzer

    def test_search_basic(self, nest_config, simple_state, basic_action_generator,
                         basic_state_transition, basic_value_function):
        """Test basic nest search"""
        nest = MonteCarloNest(config=nest_config)

        result = nest.search(
            initial_state=simple_state,
            action_generator=basic_action_generator,
            state_transition=basic_state_transition,
            value_function=basic_value_function
        )

        assert isinstance(result, NestResult)
        assert result.best_agent_result is not None
        assert len(result.all_agent_results) > 0
        assert result.elapsed_time > 0
        assert isinstance(result.aggregated_value, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.converged, bool)

    def test_search_with_aci_analyzer(self, nest_config, simple_state,
                                    basic_action_generator, basic_state_transition,
                                    basic_value_function, mock_aci_analyzer):
        """Test nest search with ACI analyzer"""
        nest = MonteCarloNest(
            config=nest_config,
            aci_analyzer=mock_aci_analyzer
        )

        result = nest.search(
            initial_state=simple_state,
            action_generator=basic_action_generator,
            state_transition=basic_state_transition,
            value_function=basic_value_function
        )

        # ACI analyzer should have been called
        assert mock_aci_analyzer.calculate.called
        assert 'initial_aci' in result.metadata

    def test_calculate_aci_without_analyzer(self, nest_config, simple_state):
        """Test ACI calculation without analyzer returns defaults"""
        nest = MonteCarloNest(config=nest_config)

        aci_result = nest._calculate_aci(simple_state)

        assert aci_result['ACI'] == 0.5
        assert aci_result['confidence'] == 0.0

    def test_calculate_aci_with_analyzer_error(self, nest_config, simple_state):
        """Test ACI calculation with analyzer error"""
        mock_analyzer = Mock()
        mock_analyzer.calculate.side_effect = Exception("ACI calculation failed")

        nest = MonteCarloNest(
            config=nest_config,
            aci_analyzer=mock_analyzer
        )

        aci_result = nest._calculate_aci(simple_state)

        # Should return defaults on error
        assert aci_result['ACI'] == 0.5

    def test_create_agent_configs(self, nest_config):
        """Test creating agent configurations"""
        nest = MonteCarloNest(config=nest_config)

        aci_result = {'ACI': 0.65}
        configs = nest._create_agent_configs(aci_result)

        assert len(configs) == len(nest_config.agent_strategies)

        # Each config should be an MCTSConfig
        for config in configs:
            assert isinstance(config, MCTSConfig)

    def test_create_config_for_exploit_strategy(self, nest_config):
        """Test config for EXPLOIT strategy"""
        nest = MonteCarloNest(config=nest_config)

        config = nest._create_config_for_strategy(
            AgentStrategy.EXPLOIT,
            aci_score=0.65
        )

        assert config.exploration_constant < 1.0  # Low C
        assert not config.adaptive_c
        assert config.playout_strategy == PlayoutStrategy.CAUSALLY_GUIDED

    def test_create_config_for_explore_strategy(self, nest_config):
        """Test config for EXPLORE strategy"""
        nest = MonteCarloNest(config=nest_config)

        config = nest._create_config_for_strategy(
            AgentStrategy.EXPLORE,
            aci_score=0.65
        )

        assert config.exploration_constant > 1.0  # High C
        assert not config.adaptive_c
        assert config.playout_strategy == PlayoutStrategy.RANDOM

    def test_create_config_for_balanced_strategy(self, nest_config):
        """Test config for BALANCED strategy"""
        nest = MonteCarloNest(config=nest_config)

        config = nest._create_config_for_strategy(
            AgentStrategy.BALANCED,
            aci_score=0.65
        )

        assert config.exploration_constant == 1.41
        assert not config.adaptive_c
        assert config.playout_strategy == PlayoutStrategy.HEURISTIC_GUIDED

    def test_create_config_for_adaptive_strategy(self, nest_config):
        """Test config for ADAPTIVE strategy"""
        nest = MonteCarloNest(config=nest_config)

        config = nest._create_config_for_strategy(
            AgentStrategy.ADAPTIVE,
            aci_score=0.65
        )

        assert config.adaptive_c
        assert config.playout_strategy == PlayoutStrategy.ADAPTIVE

    def test_run_agents_sequential(self, nest_config, simple_state,
                                  basic_action_generator, basic_state_transition,
                                  basic_value_function):
        """Test sequential agent execution"""
        config = NestConfig(
            num_agents=2,
            parallel_agents=False,
            verbose=False
        )
        nest = MonteCarloNest(config=config)

        mcts_config = MCTSConfig(
            max_iterations=10,
            verbose=False
        )

        configs = [mcts_config, mcts_config]

        results = nest._run_agents_sequential(
            simple_state,
            basic_action_generator,
            basic_state_transition,
            basic_value_function,
            configs
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, AgentResult)
            assert result.agent_id in [0, 1]
            assert result.strategy in nest_config.agent_strategies

    def test_validate_agents(self, nest_config):
        """Test agent result validation"""
        nest = MonteCarloNest(config=nest_config)

        # Create mock results
        agent_results = [
            AgentResult(
                agent_id=0,
                strategy=AgentStrategy.EXPLOIT,
                best_value=0.75,
                best_node=None,
                search_info={'value_history': [0.5, 0.6, 0.7, 0.75]}
            ),
            AgentResult(
                agent_id=1,
                strategy=AgentStrategy.EXPLORE,
                best_value=0.73,
                best_node=None,
                search_info={'value_history': [0.5, 0.6, 0.65, 0.73]}
            )
        ]

        validated = nest._validate_agents(agent_results)

        assert len(validated) == 2
        for result in validated:
            assert result.validation is not None

    def test_aggregate_results(self, nest_config):
        """Test result aggregation"""
        nest = MonteCarloNest(config=nest_config)

        # Create mock results
        agent_results = [
            AgentResult(
                agent_id=0,
                strategy=AgentStrategy.EXPLOIT,
                best_value=0.75,
                best_node=None,
                search_info={'converged': True}
            ),
            AgentResult(
                agent_id=1,
                strategy=AgentStrategy.EXPLORE,
                best_value=0.80,
                best_node=None,
                search_info={'converged': True}
            ),
            AgentResult(
                agent_id=2,
                strategy=AgentStrategy.BALANCED,
                best_value=0.70,
                best_node=None,
                search_info={'converged': False}
            )
        ]

        best, aggregated = nest._aggregate_results(agent_results)

        assert best.agent_id == 1  # Highest value
        assert aggregated['value'] > 0
        assert aggregated['confidence'] >= 0
        assert not aggregated['converged']  # Not all converged
        assert 'validation_summary' in aggregated


# =============================================================================
# NestResult Tests
# =============================================================================

class TestNestResult:
    """Test NestResult functionality"""

    def test_summary_string(self):
        """Test summary string generation"""
        from rese.phase3.statistical_validator import ValidationResult, ConfidenceInterval, ConvergenceResult

        ci = ConfidenceInterval(
            lower=0.7,
            upper=0.8,
            level=0.95,
            method=None,
            width=0.1
        )

        conv = ConvergenceResult(
            converged=True,
            method=None,
            iteration=100,
            confidence=0.95,
            details={}
        )

        validation = ValidationResult(
            confidence_interval=ci,
            convergence=conv
        )

        best_result = AgentResult(
            agent_id=0,
            strategy=AgentStrategy.EXPLOIT,
            best_value=0.75,
            best_node=None,
            search_info={},
            validation=validation
        )

        result = NestResult(
            best_agent_result=best_result,
            all_agent_results=[best_result],
            aggregated_value=0.75,
            confidence=0.9,
            elapsed_time=1.5,
            converged=True
        )

        summary = result.summary()

        assert 'Monte Carlo Nest Result' in summary
        assert 'Best Agent' in summary
        assert 'Best Value' in summary
        assert 'Converged' in summary


# =============================================================================
# Quick Nest Search Tests
# =============================================================================

class TestQuickNestSearch:
    """Test quick_nest_search convenience function"""

    def test_quick_search(self, simple_state):
        """Test quick nest search"""
        def actions(state):
            return ['a1', 'a2'] if not state.is_terminal() else []

        def transition(state, action):
            if MCTSState:
                return MCTSState(
                    variables={'x': state.variables.get('x', 0) + 1},
                    depth=state.depth + 1
                )
            return state

        def value_fn(state):
            return float(state.variables.get('x', 0))

        result = quick_nest_search(
            simple_state,
            actions,
            transition,
            value_fn,
            num_agents=2,
            iterations_per_agent=50
        )

        assert isinstance(result, NestResult)
        assert len(result.all_agent_results) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_pipeline_with_all_components(self, nest_config, simple_state,
                                             basic_action_generator,
                                             basic_state_transition,
                                             basic_value_function,
                                             mock_aci_analyzer):
        """Test complete pipeline with all components"""
        nest = MonteCarloNest(
            config=nest_config,
            aci_analyzer=mock_aci_analyzer
        )

        result = nest.search(
            initial_state=simple_state,
            action_generator=basic_action_generator,
            state_transition=basic_state_transition,
            value_function=basic_value_function
        )

        # Verify all components ran
        assert mock_aci_analyzer.calculate.called

        # Verify structure
        assert result.best_agent_result is not None
        assert len(result.all_agent_results) == nest_config.num_agents

        # Verify metadata
        assert 'initial_aci' in result.metadata
        assert 'num_agents' in result.metadata
        assert 'agent_strategies' in result.metadata

    def test_parallel_execution(self, simple_state, basic_action_generator,
                              basic_state_transition, basic_value_function):
        """Test parallel agent execution"""
        config = NestConfig(
            num_agents=4,
            parallel_agents=True,
            max_workers=2,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            initial_state=simple_state,
            action_generator=basic_action_generator,
            state_transition=basic_state_transition,
            value_function=basic_value_function
        )

        assert len(result.all_agent_results) == 4

    def test_with_validation_disabled(self, simple_state, basic_action_generator,
                                    basic_state_transition, basic_value_function):
        """Test with validation disabled"""
        config = NestConfig(
            num_agents=2,
            validate_results=False,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            initial_state=simple_state,
            action_generator=basic_action_generator,
            state_transition=basic_state_transition,
            value_function=basic_value_function
        )

        # Results should not have validation
        for agent_result in result.all_agent_results:
            assert agent_result.validation is None


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_action_list(self, simple_state):
        """Test with no available actions"""
        config = NestConfig(num_agents=1, verbose=False)
        nest = MonteCarloNest(config=config)

        def actions(state):
            return []

        def transition(state, action):
            return state

        def value_fn(state):
            return 0.0

        # Should handle gracefully
        result = nest.search(
            simple_state,
            actions,
            transition,
            value_fn
        )

        assert result is not None

    def test_single_agent(self, simple_state, basic_action_generator,
                         basic_state_transition, basic_value_function):
        """Test with single agent"""
        config = NestConfig(
            num_agents=1,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            simple_state,
            basic_action_generator,
            basic_state_transition,
            basic_value_function
        )

        assert len(result.all_agent_results) == 1

    def test_very_small_iterations(self, simple_state, basic_action_generator,
                                  basic_state_transition, basic_value_function):
        """Test with very small iteration count"""
        config = NestConfig(
            num_agents=2,
            mcts_iterations=5,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        result = nest.search(
            simple_state,
            basic_action_generator,
            basic_state_transition,
            basic_value_function
        )

        # Should still complete
        assert result is not None

    def test_agent_failure_handling(self, simple_state):
        """Test handling of agent failures"""
        config = NestConfig(
            num_agents=2,
            verbose=False
        )

        nest = MonteCarloNest(config=config)

        def actions(state):
            return ['a1']

        def transition(state, action):
            if state.depth > 0:
                raise RuntimeError("Transition error")
            return MCTSState(
                variables={'x': 1},
                depth=state.depth + 1
            ) if MCTSState else state

        def value_fn(state):
            return 1.0

        # Should handle error gracefully
        try:
            result = nest.search(
                simple_state,
                actions,
                transition,
                value_fn
            )
            # If it doesn't raise, verify structure
            if result:
                assert isinstance(result, NestResult)
        except RuntimeError:
            # Expected for some implementations
            pass
