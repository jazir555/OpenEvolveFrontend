"""
Unit Tests for MCTS Search Module (Γ₂)

Tests for Monte Carlo Tree Search implementation including:
- UCB node selection
- Progressive widening
- ACI-guided search
- Parallel execution

Author: Agent D2 (Γ₂/Γ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Testing
"""

import pytest
import numpy as np
import random
from typing import List, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase3.mcts_search import (
    MCTSState, MCTSNode, MCTSConfig, MCTSSearch,
    ParallelMCTS, PlayoutStrategy,
    quick_mcts_search, nullcontext
)


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleOptimizationState(MCTSState):
    """Simple state for testing: maximize value"""

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

    def __repr__(self):
        return f"State(value={self._value}, depth={self._depth})"


@pytest.fixture
def simple_config():
    """Standard MCTS configuration for testing"""
    return MCTSConfig(
        max_iterations=100,
        max_playout_depth=10,
        verbose=False,
        exploration_constant=1.41,
        progressive_widening=True,
        aci_guided=False  # Disable for basic tests
    )


@pytest.fixture
def initial_state():
    """Initial state for testing"""
    return SimpleOptimizationState(value=0, depth=0)


@pytest.fixture
def action_generator():
    """Generate actions: increment or decrement"""
    def generate(state: SimpleOptimizationState):
        if state.is_terminal():
            return []
        return ['+1', '-1']
    return generate


@pytest.fixture
def state_transition():
    """Transition function"""
    def transition(state: SimpleOptimizationState, action: str):
        new_value = state._value + (1 if action == '+1' else -1)
        new_depth = state._depth + 1
        return SimpleOptimizationState(new_value, new_depth, state._max_depth)
    return transition


@pytest.fixture
def value_function():
    """Value function (prefer higher values)"""
    def evaluate(state: SimpleOptimizationState):
        return state._value
    return evaluate


# ============================================================================
# MCTSState Tests
# ============================================================================

class TestMCTSState:
    """Tests for MCTSState base class"""

    def test_state_creation(self):
        """Test creating a basic state"""
        state = MCTSState()
        assert state.variables == {}
        assert state.unassigned == []
        assert state.domains == {}
        assert not state.satisfied
        assert state.depth == 0

    def test_state_with_data(self):
        """Test state with variable assignments"""
        state = MCTSState(
            variables={'x': 1, 'y': 2},
            unassigned=['z'],
            domains={'z': [1, 2, 3]},
            satisfied=False,
            depth=2
        )
        assert len(state.variables) == 2
        assert len(state.unassigned) == 1
        assert len(state.domains) == 1
        assert state.depth == 2

    def test_state_terminal(self):
        """Test terminal state detection"""
        state = MCTSState(satisfied=True)
        assert state.is_terminal()

    def test_state_hashable(self):
        """Test that states are hashable (for caching)"""
        state1 = MCTSState(variables={'x': 1})
        state2 = MCTSState(variables={'x': 1})

        assert hash(state1) == hash(state2)
        assert state1 == state2


# ============================================================================
# MCTSNode Tests
# ============================================================================

class TestMCTSNode:
    """Tests for MCTSNode"""

    def test_node_creation(self, initial_state):
        """Test creating a node"""
        node = MCTSNode(state=initial_state)
        assert node.state == initial_state
        assert node.visits == 0
        assert node.value_sum == 0.0
        assert node.value == 0.0
        assert len(node.children) == 0
        assert node.parent is None

    def test_node_statistics(self, initial_state):
        """Test node statistics calculation"""
        node = MCTSNode(state=initial_state)

        # Simulate visits
        node.visits = 10
        node.value_sum = 7.5

        assert node.value == 0.75

    def test_node_parent_child(self, initial_state):
        """Test parent-child relationships"""
        parent = MCTSNode(state=initial_state)
        child_state = SimpleOptimizationState(value=1, depth=1)
        child = MCTSNode(state=child_state, parent=parent)

        assert child.parent == parent
        assert len(parent.children) == 0  # Not automatically added

    def test_leaf_detection(self, initial_state):
        """Test leaf node detection"""
        node = MCTSNode(state=initial_state)
        assert node.is_leaf

        # Add child
        child_state = SimpleOptimizationState(value=1, depth=1)
        child = MCTSNode(state=child_state, parent=node)
        node.children.append(child)

        assert not node.is_leaf


# ============================================================================
# MCTSSearch Tests
# ============================================================================

class TestMCTSSearch:
    """Tests for MCTS search algorithm"""

    def test_mcts_initialization(self, simple_config):
        """Test MCTS initialization"""
        mcts = MCTSSearch(config=simple_config)
        assert mcts.config == simple_config
        assert mcts.iterations == 0
        assert mcts.best_value == -float('inf')
        assert len(mcts.value_history) == 0

    def test_mcts_basic_search(self, simple_config, initial_state,
                               action_generator, state_transition, value_function):
        """Test basic MCTS search"""
        mcts = MCTSSearch(config=simple_config)

        best_node, info = mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert best_node is not None
        assert info['iterations'] > 0
        assert info['best_value'] is not None
        assert info['tree_size'] > 0

    def test_mcts_finds_better_solution(self, simple_config, initial_state,
                                       action_generator, state_transition, value_function):
        """Test that MCTS finds better than random solution"""
        mcts = MCTSSearch(config=simple_config)

        best_node, info = mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Should find positive value (always increment)
        assert info['best_value'] >= 0

    def test_mcts_convergence_detection(self, initial_state,
                                       action_generator, state_transition, value_function):
        """Test convergence detection"""
        config = MCTSConfig(
            max_iterations=500,
            convergence_window=10,
            convergence_threshold=0.01,
            verbose=False
        )

        mcts = MCTSSearch(config=config)
        best_node, info = mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Check convergence info
        assert 'converged' in info

    def test_mucs_early_stopping(self):
        """Test early stopping with time limit"""
        config = MCTSConfig(
            max_iterations=10000,
            max_time_seconds=0.1,  # 100ms
            verbose=False
        )

        mcts = MCTSSearch(config=config)

        initial = SimpleOptimizationState(value=0, depth=0)
        best_node, info = mcts.search(
            initial,
            lambda s: ['+1', '-1'] if not s.is_terminal() else [],
            lambda s, a: SimpleOptimizationState(s._value + (1 if a == '+1' else -1),
                                                s._depth + 1, s._max_depth),
            lambda s: s._value
        )

        # Should stop due to time limit
        assert info['elapsed_time'] >= 0.1
        assert info['elapsed_time'] < 0.2  # Should not take much longer


# ============================================================================
# ACI-Guided Search Tests
# ============================================================================

class TestACIGuidedSearch:
    """Tests for ACI-guided MCTS"""

    def test_adaptive_c_parameter(self, simple_config):
        """Test adaptive C parameter calculation"""
        mcts = MCTSSearch(config=simple_config)

        # High ACI → low C (exploit)
        c_high = mcts._adaptive_c(0.9)
        assert c_high < simple_config.exploration_constant

        # Medium ACI → standard C
        c_med = mcts._adaptive_c(0.5)
        assert abs(c_med - simple_config.exploration_constant) < 0.01

        # Low ACI → high C (explore)
        c_low = mcts._adaptive_c(0.2)
        assert c_low > simple_config.exploration_constant

    def test_adaptive_playout_depth(self, simple_config):
        """Test adaptive playout depth"""
        mcts = MCTSSearch(config=simple_config)

        # High ACI (low disorder) → deep playouts
        depth_high = mcts._adaptive_playout_depth(0.9)
        assert depth_high > simple_config.max_playout_depth * 0.7

        # Low ACI (high disorder) → shallow playouts
        depth_low = mcts._adaptive_playout_depth(0.2)
        assert depth_low < simple_config.max_playout_depth * 0.5

    def test_progressive_widening(self, simple_config):
        """Test progressive widening logic"""
        mcts = MCTSSearch(config=simple_config)

        # Create node
        initial = SimpleOptimizationState(value=0, depth=0)
        node = MCTSNode(state=initial)

        # Initially should expand
        assert mcts._should_expand(node)

        # Simulate visits
        node.visits = 100

        # Should continue expanding until n^C > k
        for i in range(50):
            node.children.append(MCTSNode(state=SimpleOptimizationState(value=0, depth=1)))

        # Eventually should stop expanding
        if len(node.children) > node.visits ** simple_config.widening_constant:
            assert not mcts._should_expand(node)


# ============================================================================
# Parallel MCTS Tests
# ============================================================================

class TestParallelMCTS:
    """Tests for parallel MCTS execution"""

    def test_parallel_mcts_initialization(self, simple_config):
        """Test parallel MCTS initialization"""
        parallel = ParallelMCTS(config=simple_config)
        assert parallel.config == simple_config

    def test_parallel_search(self, initial_state,
                            action_generator, state_transition, value_function):
        """Test parallel MCTS search"""
        config = MCTSConfig(
            max_iterations=100,
            num_workers=4,
            verbose=False
        )

        parallel = ParallelMCTS(config=config)

        best_node, info = parallel.search_parallel(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        assert best_node is not None
        assert info['num_workers'] == 4
        assert 'all_values' in info
        assert len(info['all_values']) == 4

    def test_parallel_vs_sequential(self, initial_state,
                                   action_generator, state_transition, value_function):
        """Compare parallel and sequential results"""
        config = MCTSConfig(
            max_iterations=200,
            verbose=False
        )

        # Sequential
        sequential = MCTSSearch(config=config)
        _, seq_info = sequential.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Parallel
        config_parallel = MCTSConfig(
            max_iterations=50,  # 200/4
            num_workers=4,
            verbose=False
        )
        parallel = ParallelMCTS(config=config_parallel)
        _, par_info = parallel.search_parallel(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Parallel should do similar total iterations
        total_par_iters = par_info['total_iterations']
        assert abs(total_par_iters - seq_info['iterations']) < seq_info['iterations'] * 0.2


# ============================================================================
# Utility Tests
# ============================================================================

class TestUtilities:
    """Tests for utility functions"""

    def test_quick_mcts_search(self, initial_state,
                              action_generator, state_transition, value_function):
        """Test convenience function"""
        best_node, info = quick_mcts_search(
            initial_state,
            action_generator,
            state_transition,
            value_function,
            max_iterations=100
        )

        assert best_node is not None
        assert info['iterations'] == 100


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests"""

    def test_large_search_tree(self):
        """Test with larger search tree"""
        initial = SimpleOptimizationState(value=0, depth=0, max_depth=20)

        config = MCTSConfig(
            max_iterations=500,
            verbose=False
        )

        mcts = MCTSSearch(config=config)

        best_node, info = mcts.search(
            initial,
            lambda s: ['+1', '-1', '+2', '-2'] if not s.is_terminal() else [],
            lambda s, a: SimpleOptimizationState(
                s._value + int(a), s._depth + 1, s._max_depth),
            lambda s: s._value
        )

        # Should complete without error
        assert info['iterations'] > 0
        assert info['tree_size'] > 100

    def test_mcts_scalability(self):
        """Test MCTS scaling with iterations"""
        initial = SimpleOptimizationState(value=0, depth=0)

        for max_iters in [50, 100, 200]:
            mcts = MCTSSearch(config=MCTSConfig(max_iterations=max_iters))

            best_node, info = mcts.search(
                initial,
                lambda s: ['+1', '-1'] if not s.is_terminal() else [],
                lambda s, a: SimpleOptimizationState(s._value + (1 if a == '+1' else -1),
                                                    s._depth + 1, s._max_depth),
                lambda s: s._value
            )

            assert info['iterations'] == max_iters


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
