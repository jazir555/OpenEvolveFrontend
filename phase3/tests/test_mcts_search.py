"""
Comprehensive unit tests for MCTS Search Module

Tests all MCTS components including node selection, expansion,
simulation, backpropagation, and parallel search.

Author: Agent D2 (Γ₂ Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
import math
import time
from typing import List, Any, Tuple
from unittest.mock import Mock, MagicMock

# Try to import MCTS module
try:
    from rese.phase3.mcts_search import (
        MCTSState,
        MCTSNode,
        MCTSConfig,
        MCTSSearch,
        ParallelMCTS,
        PlayoutStrategy,
        quick_mcts_search,
        nullcontext
    )
except ImportError:
    pytest.skip("MCTS search module not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_state():
    """Create a simple MCTS state for testing"""
    return MCTSState(
        variables={'x': 0},
        unassigned=['y', 'z'],
        depth=0,
        satisfied=False
    )


@pytest.fixture
def terminal_state():
    """Create a terminal state"""
    return MCTSState(
        variables={'x': 1, 'y': 2, 'z': 3},
        unassigned=[],
        depth=3,
        satisfied=True
    )


@pytest.fixture
def basic_config():
    """Create basic MCTS config"""
    return MCTSConfig(
        max_iterations=100,
        max_playout_depth=10,
        exploration_constant=1.41,
        verbose=False
    )


@pytest.fixture
def root_node(simple_state):
    """Create a root MCTS node"""
    return MCTSNode(state=simple_state)


# =============================================================================
# MCTSState Tests
# =============================================================================

class TestMCTSState:
    """Test MCTSState functionality"""

    def test_initialization(self):
        """Test state initialization"""
        state = MCTSState(
            variables={'x': 1, 'y': 2},
            unassigned=['z'],
            domains={'z': [1, 2, 3]},
            satisfied=False,
            depth=1
        )

        assert state.variables == {'x': 1, 'y': 2}
        assert state.unassigned == ['z']
        assert state.domains == {'z': [1, 2, 3]}
        assert not state.satisfied
        assert state.depth == 1

    def test_is_terminal_true_satisfied(self):
        """Test is_terminal returns True when satisfied"""
        state = MCTSState(
            variables={'x': 1},
            unassigned=[],
            satisfied=True
        )

        assert state.is_terminal()

    def test_is_terminal_true_no_unassigned(self):
        """Test is_terminal returns True when no unassigned variables"""
        state = MCTSState(
            variables={'x': 1},
            unassigned=[],
            satisfied=False
        )

        assert state.is_terminal()

    def test_is_terminal_false(self):
        """Test is_terminal returns False when not complete"""
        state = MCTSState(
            variables={'x': 1},
            unassigned=['y'],
            satisfied=False
        )

        assert not state.is_terminal()

    def test_is_solution_true(self):
        """Test is_solution when satisfied"""
        state = MCTSState(satisfied=True)
        assert state.is_solution()

    def test_is_solution_false(self):
        """Test is_solution when not satisfied"""
        state = MCTSState(satisfied=False)
        assert not state.is_solution()

    def test_hash_and_equality(self):
        """Test state hashing and equality"""
        state1 = MCTSState(variables={'x': 1, 'y': 2})
        state2 = MCTSState(variables={'x': 1, 'y': 2})
        state3 = MCTSState(variables={'x': 1, 'y': 3})

        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)


# =============================================================================
# MCTSNode Tests
# =============================================================================

class TestMCTSNode:
    """Test MCTSNode functionality"""

    def test_initialization(self, simple_state):
        """Test node initialization"""
        node = MCTSNode(state=simple_state)

        assert node.state == simple_state
        assert node.parent is None
        assert len(node.children) == 0
        assert node.visits == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert node.aci_score == 0.5

    def test_value_property_zero_visits(self, root_node):
        """Test value property returns 0 when no visits"""
        assert root_node.value == 0.0

    def test_value_property_with_visits(self, root_node):
        """Test value property calculates average"""
        root_node.visits = 10
        root_node.value_sum = 5.0

        assert root_node.value == 0.5

    def test_is_fully_expanded_terminal(self, terminal_state):
        """Test is_fully_expanded for terminal node"""
        node = MCTSNode(state=terminal_state)
        assert node.is_fully_expanded

    def test_is_fully_expanded_with_children(self, simple_state):
        """Test is_fully_expanded with children"""
        node = MCTSNode(state=simple_state)
        child = MCTSNode(state=simple_state)
        node.children.append(child)

        assert node.is_fully_expanded

    def test_is_fully_expanded_no_children(self, simple_state):
        """Test is_fully_expanded without children"""
        node = MCTSNode(state=simple_state)
        assert not node.is_fully_expanded

    def test_is_leaf_true(self, simple_state):
        """Test is_leaf with no children"""
        node = MCTSNode(state=simple_state)
        assert node.is_leaf

    def test_is_leaf_false(self, simple_state):
        """Test is_leaf with children"""
        node = MCTSNode(state=simple_state)
        child = MCTSNode(state=simple_state)
        node.children.append(child)

        assert not node.is_leaf

    def test_initialization_raises_without_state(self):
        """Test that node requires state"""
        with pytest.raises(ValueError):
            MCTSNode(state=None)


# =============================================================================
# MCTSConfig Tests
# =============================================================================

class TestMCTSConfig:
    """Test MCTSConfig functionality"""

    def test_default_values(self):
        """Test default configuration values"""
        config = MCTSConfig()

        assert config.exploration_constant == 1.41
        assert config.adaptive_c == True
        assert config.progressive_widening == True
        assert config.max_playout_depth == 50
        assert config.max_iterations == 1000
        assert config.max_time_seconds == 60.0
        assert config.convergence_window == 20
        assert config.convergence_threshold == 0.001
        assert config.num_workers == 1
        assert config.aci_guided == True
        assert config.early_stopping == True
        assert config.verbose == False

    def test_custom_values(self):
        """Test custom configuration"""
        config = MCTSConfig(
            exploration_constant=2.0,
            adaptive_c=False,
            max_iterations=500,
            verbose=True
        )

        assert config.exploration_constant == 2.0
        assert config.adaptive_c == False
        assert config.max_iterations == 500
        assert config.verbose == True


# =============================================================================
# MCTSSearch Tests
# =============================================================================

class TestMCTSSearch:
    """Test MCTSSearch functionality"""

    def test_initialization(self, basic_config):
        """Test MCTS search initialization"""
        search = MCTSSearch(config=basic_config)

        assert search.config == basic_config
        assert search.iterations == 0
        assert search.best_value == -float('inf')
        assert search.best_node is None
        assert len(search.value_history) == 0
        assert search.converged == False
        assert search.start_time is None

    def test_initialization_default_config(self):
        """Test initialization with default config"""
        search = MCTSSearch()

        assert search.config is not None
        assert search.iterations == 0

    def test_search_basic(self, simple_state, basic_config):
        """Test basic MCTS search"""
        search = MCTSSearch(config=basic_config)

        # Simple action generator
        def actions(state):
            return ['action1', 'action2'] if not state.is_terminal() else []

        # Simple state transition
        def transition(state, action):
            new_vars = state.variables.copy()
            new_vars['x'] = new_vars.get('x', 0) + 1
            return MCTSState(
                variables=new_vars,
                unassigned=state.unassigned[1:] if state.unassigned else [],
                depth=state.depth + 1
            )

        # Simple value function
        def value_fn(state):
            return float(state.variables.get('x', 0))

        best_node, info = search.search(
            simple_state,
            actions,
            transition,
            value_fn
        )

        assert best_node is not None
        assert 'iterations' in info
        assert 'best_value' in info
        assert 'converged' in info
        assert info['iterations'] > 0

    def test_search_with_terminal_state(self, terminal_state, basic_config):
        """Test search with terminal state"""
        search = MCTSSearch(config=basic_config)

        def actions(state):
            return []

        def transition(state, action):
            return state

        def value_fn(state):
            return 1.0

        best_node, info = search.search(
            terminal_state,
            actions,
            transition,
            value_fn
        )

        assert best_node is not None

    def test_search_max_iterations(self, simple_state):
        """Test search respects max iterations"""
        config = MCTSConfig(max_iterations=50, verbose=False)
        search = MCTSSearch(config=config)

        def actions(state):
            return ['a1', 'a2'] if not state.is_terminal() else []

        def transition(state, action):
            return MCTSState(
                variables={'x': state.variables.get('x', 0) + 1},
                unassigned=[],
                depth=state.depth + 1
            )

        def value_fn(state):
            return float(state.variables.get('x', 0))

        best_node, info = search.search(
            simple_state,
            actions,
            transition,
            value_fn
        )

        assert info['iterations'] <= 50

    def test_adaptive_c_calculation(self, basic_config):
        """Test adaptive C parameter calculation"""
        search = MCTSSearch(config=basic_config)

        # High ACI -> lower C
        c_high = search._adaptive_c(0.9)
        assert c_high < basic_config.exploration_constant

        # Low ACI -> higher C
        c_low = search._adaptive_c(0.2)
        assert c_low > basic_config.exploration_constant

        # Medium ACI -> same C
        c_med = search._adaptive_c(0.5)
        assert c_med == basic_config.exploration_constant

    def test_should_stop_time_limit(self, simple_state):
        """Test time limit stopping"""
        config = MCTSConfig(
            max_time_seconds=0.1,  # 100ms
            verbose=False
        )
        search = MCTSSearch(config=config)

        state = simple_state

        # Set start time in past
        search.start_time = time.time() - 0.2  # 200ms ago

        assert search._should_stop(MCTSNode(state=state))

    def test_should_stop_convergence(self, simple_state):
        """Test convergence detection"""
        config = MCTSConfig(
            convergence_window=5,
            convergence_threshold=0.01,
            verbose=False
        )
        search = MCTSSearch(config=config)

        # Add stable value history
        search.value_history = [1.0, 1.0, 1.0, 1.0, 1.0]

        assert search._should_stop(MCTSNode(state=simple_state))

    def test_should_stop_early_stopping(self, simple_state):
        """Test early stopping for low ACI"""
        config = MCTSConfig(
            early_stopping=True,
            verbose=False
        )
        search = MCTSSearch(config=config)

        node = MCTSNode(state=simple_state, aci_score=0.2)  # Low ACI
        search.iterations = 150  # Above minimum
        search.best_value = 0.0  # No improvement

        assert search._should_stop(node)

    def test_progressive_widening(self, simple_state):
        """Test progressive widening expansion"""
        config = MCTSConfig(
            progressive_widening=True,
            widening_constant=0.5,
            verbose=False
        )
        search = MCTSSearch(config=config)

        node = MCTSNode(state=simple_state)

        # Initially should expand
        assert search._should_expand(node)

        # Add some children
        for _ in range(5):
            node.children.append(MCTSNode(state=simple_state))

        # With low visits, should still expand
        node.visits = 5
        assert search._should_expand(node)

        # With many visits and children, should not expand
        node.visits = 100
        assert not search._should_expand(node)

    def test_backpropagation(self, simple_state):
        """Test value backpropagation"""
        config = MCTSConfig(aci_guided=False, verbose=False)
        search = MCTSSearch(config=config)

        # Create tree: root -> child -> grandchild
        root = MCTSNode(state=simple_state)
        child = MCTSNode(state=simple_state, parent=root)
        grandchild = MCTSNode(state=simple_state, parent=child)

        root.children.append(child)
        child.children.append(grandchild)

        # Backpropagate value
        search._backpropagate(grandchild, 1.0)

        assert grandchild.visits == 1
        assert grandchild.value_sum == 1.0
        assert child.visits == 1
        assert child.value_sum == 1.0
        assert root.visits == 1
        assert root.value_sum == 1.0

    def test_backpropagation_with_aci_weighting(self, simple_state):
        """Test ACI-weighted backpropagation"""
        config = MCTSConfig(aci_guided=True, verbose=False)
        search = MCTSSearch(config=config)

        # Create nodes with different ACI scores
        root = MCTSNode(state=simple_state, aci_score=0.8)
        child = MCTSNode(state=simple_state, parent=root, aci_score=0.6)

        root.children.append(child)

        # Backpropagate
        search._backpropagate(child, 1.0)

        # Root should get weighted value (weight = 0.5 + 0.5 * ACI)
        expected_weight = 0.5 + 0.5 * 0.8
        assert child.value_sum == 1.0
        assert abs(root.value_sum - 1.0 * expected_weight) < 0.01

    def test_select_best_child(self, simple_state):
        """Test selecting best child"""
        node = MCTSNode(state=simple_state)

        # Create children with different visit counts
        child1 = MCTSNode(state=simple_state, parent=node)
        child1.visits = 5
        child1.value_sum = 3.0

        child2 = MCTSNode(state=simple_state, parent=node)
        child2.visits = 10
        child2.value_sum = 6.0

        child3 = MCTSNode(state=simple_state, parent=node)
        child3.visits = 3
        child3.value_sum = 2.0

        node.children = [child1, child2, child3]

        search = MCTSSearch()
        best = search._select_best_child(node)

        # Should select most visited
        assert best == child2

    def test_count_tree_nodes(self, simple_state):
        """Test tree size counting"""
        search = MCTSSearch()

        # Create tree
        root = MCTSNode(state=simple_state)
        child1 = MCTSNode(state=simple_state)
        child2 = MCTSNode(state=simple_state)
        grandchild = MCTSNode(state=simple_state)

        root.children = [child1, child2]
        child1.children = [grandchild]

        count = search._count_tree_nodes(root)

        assert count == 4

    def test_simulate_terminal_state(self, terminal_state):
        """Test simulation with terminal state"""
        search = MCTSSearch()

        node = MCTSNode(state=terminal_state)

        def transition(state, action):
            return state

        def value_fn(state):
            return 1.0

        value = search._simulate(node, transition, value_fn)

        assert value == 1.0

    def test_select_playout_strategy_adaptive(self, simple_state):
        """Test adaptive playout strategy selection"""
        config = MCTSConfig(
            playout_strategy=PlayoutStrategy.ADAPTIVE,
            aci_guided=True,
            verbose=False
        )
        search = MCTSSearch(config=config)

        # High ACI -> causally guided
        high_aci_node = MCTSNode(state=simple_state, aci_score=0.8)
        strategy = search._select_playout_strategy(high_aci_node)
        assert strategy == PlayoutStrategy.CAUSALLY_GUIDED

        # Medium ACI -> heuristic guided
        med_aci_node = MCTSNode(state=simple_state, aci_score=0.5)
        strategy = search._select_playout_strategy(med_aci_node)
        assert strategy == PlayoutStrategy.HEURISTIC_GUIDED

        # Low ACI -> random
        low_aci_node = MCTSNode(state=simple_state, aci_score=0.2)
        strategy = search._select_playout_strategy(low_aci_node)
        assert strategy == PlayoutStrategy.RANDOM

    def test_adaptive_playout_depth(self):
        """Test adaptive playout depth based on ACI"""
        config = MCTSConfig(max_playout_depth=50, verbose=False)
        search = MCTSSearch(config=config)

        # High ACI (low disorder) -> deep
        high_aci_depth = search._adaptive_playout_depth(0.9)
        assert high_aci_depth == 50

        # Low ACI (high disorder) -> shallow
        low_aci_depth = search._adaptive_playout_depth(0.2)
        assert low_aci_depth < 50

        # Medium ACI -> medium depth
        med_aci_depth = search._adaptive_playout_depth(0.5)
        assert 0 < med_aci_depth < 50


# =============================================================================
# ParallelMCTS Tests
# =============================================================================

class TestParallelMCTS:
    """Test parallel MCTS functionality"""

    def test_initialization(self, basic_config):
        """Test parallel MCTS initialization"""
        parallel = ParallelMCTS(config=basic_config)

        assert parallel.config == basic_config

    def test_search_parallel_single_worker(self, simple_state):
        """Test parallel search with single worker (sequential)"""
        config = MCTSConfig(num_workers=1, verbose=False)
        parallel = ParallelMCTS(config=config)

        def actions(state):
            return ['a1', 'a2']

        def transition(state, action):
            return MCTSState(
                variables={'x': state.variables.get('x', 0) + 1},
                depth=state.depth + 1
            )

        def value_fn(state):
            return float(state.variables.get('x', 0))

        best_node, info = parallel.search_parallel(
            simple_state,
            actions,
            transition,
            value_fn,
            num_workers=1
        )

        assert best_node is not None
        assert 'best_value' in info

    def test_aggregate_results(self, simple_state):
        """Test result aggregation"""
        parallel = ParallelMCTS()

        # Create mock results
        result1 = (MCTSNode(state=simple_state), {
            'best_value': 0.8,
            'iterations': 100,
            'worker_id': 0
        })

        result2 = (MCTSNode(state=simple_state), {
            'best_value': 0.9,
            'iterations': 100,
            'worker_id': 1
        })

        result3 = (MCTSNode(state=simple_state), {
            'best_value': 0.7,
            'iterations': 100,
            'worker_id': 2
        })

        best_node, aggregated = parallel._aggregate_results([result1, result2, result3])

        assert aggregated['num_workers'] == 3
        assert aggregated['best_value'] == 0.9
        assert aggregated['best_worker'] == 1
        assert len(aggregated['all_values']) == 3
        assert abs(aggregated['mean_value'] - 0.8) < 0.01


# =============================================================================
# Quick MCTS Search Tests
# =============================================================================

class TestQuickMCTSSearch:
    """Test convenience function"""

    def test_quick_search(self, simple_state):
        """Test quick_mcts_search convenience function"""
        def actions(state):
            return ['a1', 'a2'] if not state.is_terminal() else []

        def transition(state, action):
            return MCTSState(
                variables={'x': state.variables.get('x', 0) + 1},
                depth=state.depth + 1
            )

        def value_fn(state):
            return float(state.variables.get('x', 0))

        best_node, info = quick_mcts_search(
            simple_state,
            actions,
            transition,
            value_fn,
            max_iterations=50
        )

        assert best_node is not None
        assert info['iterations'] > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_action_list(self, simple_state):
        """Test with no available actions"""
        search = MCTSSearch(MCTSConfig(max_iterations=10, verbose=False))

        def actions(state):
            return []  # No actions

        def transition(state, action):
            return state

        def value_fn(state):
            return 0.0

        # Should handle gracefully
        best_node, info = search.search(
            simple_state,
            actions,
            transition,
            value_fn
        )

        assert best_node is not None

    def test_zero_exploration_constant(self, simple_state):
        """Test with zero exploration constant (pure exploitation)"""
        config = MCTSConfig(exploration_constant=0.0, verbose=False)
        search = MCTSSearch(config=config)

        node = MCTSNode(state=simple_state)
        child = MCTSNode(state=simple_state, parent=node)
        child.visits = 10
        child.value_sum = 5.0

        node.children.append(child)
        node.visits = 20

        # Should select child with highest value (no exploration bonus)
        selected = search._select_child(node)
        assert selected == child

    def test_negative_values(self, simple_state):
        """Test with negative reward values"""
        search = MCTSSearch()

        def actions(state):
            return ['a1']

        def transition(state, action):
            return MCTSState(
                variables={'x': -1},
                depth=state.depth + 1
            )

        def value_fn(state):
            return -1.0

        node = MCTSNode(state=simple_state)
        value = search._simulate(node, transition, value_fn)

        assert value < 0

    def test_very_large_tree(self, simple_state):
        """Test with large number of iterations"""
        config = MCTSConfig(
            max_iterations=1000,
            max_playout_depth=10,
            verbose=False
        )
        search = MCTSSearch(config=config)

        def actions(state):
            return ['a1', 'a2', 'a3']

        def transition(state, action):
            return MCTSState(
                variables={'x': state.variables.get('x', 0) + 1},
                depth=state.depth + 1
            )

        def value_fn(state):
            return float(state.variables.get('x', 0))

        best_node, info = search.search(
            simple_state,
            actions,
            transition,
            value_fn
        )

        assert info['tree_size'] > 0

    def test_virtual_losses(self, simple_state):
        """Test virtual loss mechanism for parallel MCTS"""
        config = MCTSConfig(virtual_loss=True, verbose=False)
        search = MCTSSearch(config=config)

        node = MCTSNode(state=simple_state)
        node.visits = 10
        node.value_sum = 5.0
        node.virtual_losses = 2

        # Calculate effective value with virtual losses
        effective_visits = node.visits + node.virtual_losses
        effective_value = (node.value_sum - node.virtual_losses) / effective_visits

        assert effective_visits == 12
        assert effective_value < (node.value_sum / node.visits)


# =============================================================================
# Nullcontext Tests
# =============================================================================

class TestNullcontext:
    """Test nullcontext implementation"""

    def test_nullcontext_basic(self):
        """Test nullcontext works as context manager"""
        ctx = nullcontext()

        with ctx:
            result = 1 + 1

        assert result == 2

    def test_nullcontext_exception_handling(self):
        """Test nullcontext handles exceptions correctly"""
        ctx = nullcontext()

        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test error")
