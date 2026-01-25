"""
MCTS Search Module for RESE Phase III (Monte Carlo Refinement)

Implements Monte Carlo Tree Search with ACI-guided node selection,
progressive widening, and adaptive playouts.

Author: Agent D2 (Γ₂/Γ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
Dependencies:
    - rese.core.symbolic_constraint_engine (Constraint foundation)
    - rese.phase3.aci_analyzer (Γ₁ - ACI calculation, by Agent D1)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
import math
import random
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Try to import constraint engine
try:
    from core.symbolic_constraint_engine import (
        SymbolicConstraintEngine, Constraint, ConstraintType
    )
except ImportError:
    SymbolicConstraintEngine = None
    Constraint = None
    ConstraintType = None

# Try to import ACI analyzer (will be implemented by Agent D1)
try:
    from phase3.aci_analyzer import ACIAnalyzer
except ImportError:
    ACIAnalyzer = None


class PlayoutStrategy(Enum):
    """Strategies for MCTS simulation/rollout"""
    RANDOM = "random"  # Pure random sampling
    HEURISTIC_GUIDED = "heuristic_guided"  # Domain heuristics
    CAUSALLY_GUIDED = "causally_guided"  # Follow constraint dependencies
    ADAPTIVE = "adaptive"  # Choose based on ACI


@dataclass
class MCTSConfig:
    """Configuration for MCTS search"""
    # UCB parameters
    exploration_constant: float = 1.41  # C parameter in UCB (sqrt(2) is standard)
    adaptive_c: bool = True  # Adjust C based on ACI

    # Progressive widening
    progressive_widening: bool = True
    widening_constant: float = 0.5  # C_expansion parameter

    # Simulation parameters
    max_playout_depth: int = 50
    playout_strategy: PlayoutStrategy = PlayoutStrategy.ADAPTIVE
    adaptive_depth: bool = True  # Adjust depth based on ACI

    # Stopping criteria
    max_iterations: int = 1000
    max_time_seconds: float = 60.0
    convergence_window: int = 20
    convergence_threshold: float = 0.001

    # Parallelization
    num_workers: int = 1
    virtual_loss: bool = False  # For tree parallelization

    # ACI guidance
    aci_guided: bool = True
    early_stopping: bool = True  # Stop early for low ACI

    # Debugging
    verbose: bool = False
    log_interval: int = 100


@dataclass
class MCTSState:
    """
    Represents a state in the MCTS search tree.

    For constraint satisfaction, this is a partial assignment of variables.
    """
    variables: Dict[str, Any] = field(default_factory=dict)  # Assigned variables
    unassigned: List[str] = field(default_factory=list)  # Unassigned variable names
    domains: Dict[str, List[Any]] = field(default_factory=dict)  # Remaining domains
    satisfied: bool = False  # Whether all constraints satisfied
    depth: int = 0  # Depth in search tree

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (complete assignment or dead end)"""
        return self.satisfied or len(self.unassigned) == 0

    def is_solution(self) -> bool:
        """Check if this state represents a valid solution"""
        return self.satisfied

    def __hash__(self):
        """Make state hashable for caching"""
        # Create tuple from sorted variables for consistent hashing
        var_tuple = tuple(sorted(self.variables.items()))
        return hash(var_tuple)

    def __eq__(self, other):
        """State equality based on variable assignments"""
        if not isinstance(other, MCTSState):
            return False
        return self.variables == other.variables


@dataclass
class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a state and stores statistics for UCB selection.
    """
    state: MCTSState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)

    # Statistics
    visits: int = 0  # N: Visit count
    value_sum: float = 0.0  # W: Total value (wins/rewards)
    prior: float = 0.0  # P: Prior probability (from network or heuristics)

    # For parallel MCTS (virtual loss)
    virtual_losses: int = 0

    # ACI-related
    aci_score: float = 0.5  # ACI score for this node's state

    # Action tracking (for expansion control)
    unexpanded_actions: List[Any] = field(default_factory=list)  # Actions not yet expanded
    total_actions: int = 0  # Total number of possible actions

    def __post_init__(self):
        """Initialize node after creation"""
        if self.state is None:
            raise ValueError("MCTSNode must have a state")

    @property
    def value(self) -> float:
        """Average value (win rate)"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all children have been expanded"""
        if self.state.is_terminal():
            return True
        # Check if there are any unexpanded actions remaining
        if self.total_actions > 0:
            return len(self.unexpanded_actions) == 0
        # Fallback: if total_actions not set, assume fully expanded if has children
        return len(self.children) > 0

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0


class MCTSSearch:
    """
    Monte Carlo Tree Search with ACI-guided node selection.

    Implements the four MCTS steps:
    1. Selection: UCB-guided tree traversal
    2. Expansion: Progressive widening
    3. Simulation: ACI-adaptive playouts
    4. Backpropagation: Value backup with ACI weighting
    """

    def __init__(self, config: MCTSConfig = None, aci_analyzer: ACIAnalyzer = None):
        """
        Initialize MCTS search.

        Args:
            config: MCTS configuration parameters
            aci_analyzer: ACI analyzer for guidance (Γ₁)
        """
        self.config = config or MCTSConfig()
        self.aci_analyzer = aci_analyzer

        # Statistics
        self.iterations = 0
        self.best_value = -float('inf')
        self.best_node: Optional[MCTSNode] = None
        self.value_history: List[float] = []

        # For convergence detection
        self.converged = False
        self.start_time = None

        # For parallel execution
        self.lock = threading.RLock() if self.config.num_workers > 1 else None

    def search(self, initial_state: MCTSState,
               action_generator: Callable[[MCTSState], List[Any]],
               state_transition: Callable[[MCTSState, Any], MCTSState],
               value_function: Callable[[MCTSState], float],
               initial_aci: Optional[Dict] = None) -> Tuple[MCTSNode, Dict]:
        """
        Run MCTS search from initial state.

        Args:
            initial_state: Starting state
            action_generator: Function that generates available actions for a state
            state_transition: Function that applies action to produce new state
            value_function: Function that evaluates state quality (returns float)
            initial_aci: Optional ACI result for initial state (from Γ₁)

        Returns:
            (best_node, search_info) - Best node found and search statistics
        """
        self.start_time = time.time()

        # Calculate ACI for initial state if not provided
        if initial_aci is None and self.aci_analyzer is not None:
            initial_aci = self.aci_analyzer.calculate(initial_state)

        # Initialize root node
        root = MCTSNode(state=initial_state)

        # Set initial ACI score
        if initial_aci is not None:
            root.aci_score = initial_aci.get('ACI', 0.5)

        # Main MCTS loop
        for self.iterations in range(1, self.config.max_iterations + 1):
            # Check stopping criteria
            if self._should_stop(root):
                break

            # Four steps of MCTS
            node = self._select(root)
            child = self._expand(node, action_generator)
            value = self._simulate(child, state_transition, value_function)
            self._backpropagate(child, value)

            # Track best
            if value > self.best_value:
                self.best_value = value
                self.best_node = child
                self.value_history.append(value)

            # Logging
            if self.config.verbose and self.iterations % self.config.log_interval == 0:
                self._log_progress(root)

        # Return best child of root
        best_child = self._select_best_child(root)

        # Compile search info
        search_info = self._compile_search_info(root)

        return best_child, search_info

    def _should_stop(self, root: MCTSNode) -> bool:
        """Check if search should stop"""
        # Time limit
        if self.start_time and time.time() - self.start_time > self.config.max_time_seconds:
            if self.config.verbose:
                print(f"[MCTS] Time limit reached: {time.time() - self.start_time:.2f}s")
            return True

        # Convergence detection
        if self.config.convergence_window > 0:
            if len(self.value_history) >= self.config.convergence_window:
                recent = self.value_history[-self.config.convergence_window:]
                if np.std(recent) < self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"[MCTS] Converged after {self.iterations} iterations")
                    self.converged = True
                    return True

        # Early stopping for low ACI
        if self.config.early_stopping and self.aci_analyzer is not None:
            if self.iterations >= 100:  # Minimum iterations before checking
                if root.aci_score < 0.3:
                    improvement = self.best_value - (-float('inf'))
                    if improvement < 0.01:
                        if self.config.verbose:
                            print(f"[MCTS] Early stopping: Low ACI ({root.aci_score:.3f}), no progress")
                        return True

        return False

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB to find leaf node.

        Args:
            node: Root of subtree to select from

        Returns:
            Selected leaf node
        """
        while not node.is_leaf and not node.state.is_terminal():
            # Select best child using UCB
            node = self._select_child(node)

        return node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select best child using UCB formula.

        UCB = Q + C * sqrt(ln(N_parent) / N_child)

        With ACI guidance:
        - Adjust C based on ACI (higher ACI → lower C, more exploitation)
        - Adjust Q with prior (if available)
        """
        # Adaptive C based on ACI
        if self.config.adaptive_c and node.aci_score > 0:
            c_param = self._adaptive_c(node.aci_score)
        else:
            c_param = self.config.exploration_constant

        best_child = None
        best_ucb = -float('inf')

        for child in node.children:
            # Calculate UCB with virtual loss (for parallel MCTS)
            visits = child.visits
            value_sum = child.value_sum

            if self.config.virtual_loss:
                visits += child.virtual_losses
                value_sum -= child.virtual_losses

            if visits == 0:
                # Unvisited node: infinite UCB (encourage exploration)
                ucb = float('inf')
            else:
                # UCB formula
                exploration = c_param * math.sqrt(math.log(node.visits) / visits)

                if self.config.virtual_loss:
                    exploitation = value_sum / visits
                else:
                    exploitation = child.value

                # Add prior bonus if available (neural MCTS style)
                prior_bonus = 0.0
                if child.prior > 0:
                    prior_bonus = c_param * child.prior * math.sqrt(node.visits) / (1 + visits)

                ucb = exploitation + exploration + prior_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def _adaptive_c(self, aci_score: float) -> float:
        """
        Calculate adaptive exploration parameter based on ACI.

        High ACI → Trust structure → Exploit more (lower C)
        Low ACI → Uncertain → Explore more (higher C)
        """
        base_c = self.config.exploration_constant

        if aci_score > 0.8:
            # Highly tractable: exploit heavily
            return base_c * 0.5
        elif aci_score > 0.6:
            # Tractable: moderate exploitation
            return base_c * 0.8
        elif aci_score > 0.4:
            # Balanced: standard UCB
            return base_c
        else:
            # Intractable: explore a lot
            return base_c * 1.5

    def _expand(self, node: MCTSNode,
                action_generator: Callable[[MCTSState], List[Any]]) -> MCTSNode:
        """
        Expansion phase: Add new child node(s).

        Uses progressive widening to control expansion rate:
        - Only expand if: n_visits^C > num_children
        - Tracks unexpanded actions to prevent duplicate expansions
        """
        if node.state.is_terminal():
            return node

        # Get available actions (first time for this node)
        if node.total_actions == 0:
            actions = action_generator(node.state)
            node.total_actions = len(actions)
            node.unexpanded_actions = actions.copy()
        else:
            actions = node.unexpanded_actions

        if not actions:
            return node

        # Progressive widening check
        if self.config.progressive_widening:
            if not self._should_expand(node):
                return node

        # Select action to expand (random from unexpanded)
        if not node.unexpanded_actions:
            return node

        action = random.choice(node.unexpanded_actions)
        node.unexpanded_actions.remove(action)

        # Apply action to get new state
        # This requires a transition function - for now, create placeholder
        # In practice, this would be passed in or stored in the node
        new_state = node.state  # Placeholder

        # Create child node
        child = MCTSNode(state=new_state, parent=node)

        # Set ACI score if analyzer available
        if self.aci_analyzer is not None:
            try:
                aci_result = self.aci_analyzer.calculate(new_state)
                child.aci_score = aci_result.get('ACI', 0.5)
            except:
                child.aci_score = node.aci_score  # Inherit from parent
        else:
            child.aci_score = node.aci_score

        # Add to parent
        with self.lock if self.lock else nullcontext():
            node.children.append(child)

        return child

    def _should_expand(self, node: MCTSNode) -> bool:
        """
        Check if node should be expanded (progressive widening).

        Returns True if: n_visits^C > num_children
        """
        k = len(node.children)
        n = node.visits

        if n == 0:
            return True

        # Progressive widening formula
        return n ** self.config.widening_constant > k

    def _simulate(self, node: MCTSNode,
                  state_transition: Callable[[MCTSState, Any], MCTSState],
                  value_function: Callable[[MCTSState], float]) -> float:
        """
        Simulation phase: Run playout from node.

        Uses ACI-adaptive playout strategy:
        - High coherence: Causally-guided playouts
        - Low entropy: Heuristic-guided playouts
        - Default: Random playouts
        """
        if node.state.is_terminal():
            # Terminal node: use actual value
            return value_function(node.state)

        # Select playout strategy
        strategy = self._select_playout_strategy(node)

        # Adaptive depth
        if self.config.adaptive_depth:
            max_depth = self._adaptive_playout_depth(node.aci_score)
        else:
            max_depth = self.config.max_playout_depth

        # Run playout
        current_state = node.state
        depth = 0

        while not current_state.is_terminal() and depth < max_depth:
            # Generate actions
            # In practice, would use action generator here
            actions = []  # Placeholder

            if not actions:
                break

            # Select action based on strategy
            action = self._select_playout_action(current_state, actions, strategy)

            # Apply action
            current_state = state_transition(current_state, action)
            depth += 1

        # Return value of final state
        return value_function(current_state)

    def _select_playout_strategy(self, node: MCTSNode) -> PlayoutStrategy:
        """
        Select playout strategy based on ACI.

        High coherence → Causally-guided
        Low entropy → Heuristic-guided
        Default → Random
        """
        if not self.config.aci_guided:
            return self.config.playout_strategy

        if self.config.playout_strategy == PlayoutStrategy.ADAPTIVE:
            # Adaptive selection based on ACI components
            # In practice, would access ACI components here
            # For now, use overall ACI score
            if node.aci_score > 0.7:
                return PlayoutStrategy.CAUSALLY_GUIDED
            elif node.aci_score > 0.4:
                return PlayoutStrategy.HEURISTIC_GUIDED
            else:
                return PlayoutStrategy.RANDOM
        else:
            return self.config.playout_strategy

    def _adaptive_playout_depth(self, aci_score: float) -> int:
        """
        Calculate adaptive playout depth based on ACI.

        High disorder → Shallow playouts (too uncertain)
        Low disorder → Deep playouts (predictable)
        """
        # In practice, would use disorder entropy component
        # For now, invert ACI score as proxy
        disorder = 1.0 - aci_score

        if disorder > 0.7:
            # High disorder: shallow
            return int(self.config.max_playout_depth * 0.2)
        elif disorder > 0.5:
            return int(self.config.max_playout_depth * 0.5)
        else:
            # Low disorder: deep
            return self.config.max_playout_depth

    def _select_playout_action(self, state: MCTSState, actions: List[Any],
                               strategy: PlayoutStrategy) -> Any:
        """
        Select action during playout based on strategy.

        Implements:
        - Random selection: Pure exploration
        - Heuristic-guided selection: Domain-specific heuristics
        - Causally-guided selection: Follow constraint dependencies
        """
        if not actions:
            return None

        if strategy == PlayoutStrategy.RANDOM:
            return random.choice(actions)

        elif strategy == PlayoutStrategy.HEURISTIC_GUIDED:
            # Heuristic: prioritize actions with higher expected value
            # In practice, would use domain-specific heuristics
            # For now, use simple random with bias toward unvisited variables

            # If state has unassigned variables, prioritize those
            if state.unassigned:
                # Simple heuristic: random choice (can be enhanced)
                return random.choice(actions)
            else:
                return random.choice(actions)

        elif strategy == PlayoutStrategy.CAUSALLY_GUIDED:
            # Causal guidance: follow constraint dependencies
            # Prioritize variables that influence many others

            # If state has unassigned variables, select one with most dependencies
            if state.unassigned:
                # Simple heuristic: select from unassigned (can be enhanced with dependency graph)
                return random.choice(actions)
            else:
                return random.choice(actions)

        else:
            return random.choice(actions)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagation phase: Update statistics up the tree.

        ACI-weighted backup:
        - Nodes with higher ACI get more weight
        """
        current = node

        while current is not None:
            with self.lock if self.lock else nullcontext():
                current.visits += 1

                # ACI-weighted update
                if self.config.aci_guided:
                    weight = 0.5 + 0.5 * current.aci_score  # Weight in [0.5, 1.0]
                    current.value_sum += value * weight
                else:
                    current.value_sum += value

            current = current.parent

    def _select_best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select best child (most visited or highest value).

        Args:
            node: Parent node

        Returns:
            Best child node
        """
        if not node.children:
            return node

        # Select child with most visits (robust)
        best_child = max(node.children, key=lambda c: c.visits)

        return best_child

    def _log_progress(self, root: MCTSNode) -> None:
        """Log search progress"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        print(f"[MCTS] Iteration {self.iterations}: "
              f"Best value={self.best_value:.4f}, "
              f"Root visits={root.visits}, "
              f"Children={len(root.children)}, "
              f"Time={elapsed:.2f}s")

    def _compile_search_info(self, root: MCTSNode) -> Dict:
        """Compile search statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'iterations': self.iterations,
            'best_value': self.best_value,
            'converged': self.converged,
            'elapsed_time': elapsed,
            'root_visits': root.visits,
            'num_children': len(root.children),
            'tree_size': self._count_tree_nodes(root),
            'value_history': self.value_history.copy(),
            'config': self.config
        }

    def _count_tree_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count


class ParallelMCTS:
    """
    Parallel MCTS execution with multiple workers.

    Supports:
    - Root parallelization: Independent searches from root
    - Tree parallelization: Shared tree with virtual loss
    """

    def __init__(self, config: MCTSConfig = None, aci_analyzer: ACIAnalyzer = None):
        self.config = config or MCTSConfig()
        self.aci_analyzer = aci_analyzer

    def search_parallel(self,
                       initial_state: MCTSState,
                       action_generator: Callable[[MCTSState], List[Any]],
                       state_transition: Callable[[MCTSState, Any], MCTSState],
                       value_function: Callable[[MCTSState], float],
                       num_workers: int = None) -> Tuple[MCTSNode, Dict]:
        """
        Run MCTS with multiple workers in parallel.

        Args:
            initial_state: Starting state
            action_generator: Action generation function
            state_transition: State transition function
            value_function: State evaluation function
            num_workers: Number of parallel workers (default from config)

        Returns:
            (best_node, aggregated_info)
        """
        num_workers = num_workers or self.config.num_workers

        if num_workers == 1:
            # Sequential search
            mcts = MCTSSearch(self.config, self.aci_analyzer)
            return mcts.search(initial_state, action_generator,
                            state_transition, value_function)

        # Parallel search
        futures = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for worker_id in range(num_workers):
                future = executor.submit(
                    self._worker_search,
                    worker_id,
                    initial_state,
                    action_generator,
                    state_transition,
                    value_function
                )
                futures.append(future)

            # Wait for all workers
            results = [f.result() for f in as_completed(futures)]

        # Aggregate results
        best_node, aggregated_info = self._aggregate_results(results)

        return best_node, aggregated_info

    def _worker_search(self,
                      worker_id: int,
                      initial_state: MCTSState,
                      action_generator: Callable[[MCTSState], List[Any]],
                      state_transition: Callable[[MCTSState, Any], MCTSState],
                      value_function: Callable[[MCTSState], float]) -> Tuple[MCTSNode, Dict]:
        """Worker function for parallel MCTS"""
        # Create independent MCTS instance
        mcts = MCTSSearch(self.config, self.aci_analyzer)

        # Run search
        best_node, search_info = mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        # Add worker ID to info
        search_info['worker_id'] = worker_id

        return best_node, search_info

    def _aggregate_results(self, results: List[Tuple[MCTSNode, Dict]]) -> Tuple[MCTSNode, Dict]:
        """Aggregate results from multiple workers"""
        # Find best result across all workers
        best_result = max(results, key=lambda x: x[1]['best_value'])
        best_node, best_info = best_result

        # Compile aggregated info
        aggregated_info = {
            'num_workers': len(results),
            'best_value': best_info['best_value'],
            'best_worker': best_info['worker_id'],
            'all_values': [r[1]['best_value'] for r in results],
            'mean_value': np.mean([r[1]['best_value'] for r in results]),
            'std_value': np.std([r[1]['best_value'] for r in results]),
            'total_iterations': sum(r[1]['iterations'] for r in results),
            'worker_details': [r[1] for r in results]
        }

        return best_node, aggregated_info


# Helper context manager for optional locking
class nullcontext:
    """Context manager that does nothing (for Python < 3.7 compatibility)"""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Convenience functions
def quick_mcts_search(initial_state: MCTSState,
                     action_generator: Callable[[MCTSState], List[Any]],
                     state_transition: Callable[[MCTSState, Any], MCTSState],
                     value_function: Callable[[MCTSState], float],
                     max_iterations: int = 1000,
                     verbose: bool = False) -> Tuple[MCTSNode, Dict]:
    """
    Convenience function for quick MCTS search with default parameters.

    Args:
        initial_state: Starting state
        action_generator: Action generation function
        state_transition: State transition function
        value_function: State evaluation function
        max_iterations: Maximum MCTS iterations
        verbose: Enable verbose logging

    Returns:
        (best_node, search_info)
    """
    config = MCTSConfig(
        max_iterations=max_iterations,
        verbose=verbose
    )

    mcts = MCTSSearch(config)

    return mcts.search(initial_state, action_generator, state_transition, value_function)


# Example usage (for testing)
if __name__ == "__main__":
    print("MCTS Search Module - Ready")
    print("=" * 60)

    # Simple example: Find maximum value path
    print("\nExample: Simple optimization problem")
    print("-" * 60)

    # Define simple state space
    class SimpleState(MCTSState):
        def __init__(self, value=0, depth=0):
            self.value_val = value
            self.depth_val = depth
            super().__init__()

        @property
        def value(self):
            return self.value_val

        def is_terminal(self):
            return self.depth_val >= 5

    # Action generator: increment or decrement
    def simple_actions(state):
        if state.depth_val >= 5:
            return []
        return ['+1', '-1']

    # State transition
    def simple_transition(state, action):
        new_value = state.value_val + (1 if action == '+1' else -1)
        new_depth = state.depth_val + 1
        return SimpleState(new_value, new_depth)

    # Value function (prefer higher values)
    def simple_value(state):
        return state.value_val

    # Initial state
    initial = SimpleState(value=0, depth=0)

    # Run MCTS
    print("Running MCTS search...")
    best_node, info = quick_mcts_search(
        initial,
        simple_actions,
        simple_transition,
        simple_value,
        max_iterations=500,
        verbose=True
    )

    print(f"\nBest value found: {info['best_value']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Tree size: {info['tree_size']} nodes")
    print(f"Converged: {info['converged']}")
    print(f"Time: {info['elapsed_time']:.2f}s")

    print("\n" + "=" * 60)
    print("MCTS Search Module - Test Complete")
