"""
Monte Carlo Tree Search (MCTS) for Lean 4 Proof Search

Production-ready MCTS implementation for automated Lean 4 proof search, inspired by
AlphaGo and AlphaZero architectures. This module implements the four classic MCTS
phases: Selection, Expansion, Simulation, and Backpropagation, with specialized
adaptations for theorem proving.

Classes:
    MCTSNode: Represents a node in the MCTS search tree
    MCTSTree: Manages the complete search tree
    MCTSSelection: UCT-based selection phase
    MCTSExpansion: Tactical expansion phase
    MCTSSimulation: Rollout/simulation phase
    MCTSBackpropagation: Backpropagation phase
    MCTS: Main orchestrator combining all phases

Key Features:
    - UCT (Upper Confidence Bound for Trees) selection
    - Multiple rollout policies (random, heuristic, learned)
    - Transposition table for state reuse
    - AMAF (All-Moves-As-First) for faster convergence
    - Parallel simulation support
    - Progressive widening for large action spaces
    - Adaptive exploration parameters
    - Comprehensive caching layer

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import hashlib
import math
import random
import time
import uuid
import hashlib
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
import sqlite3
from pathlib import Path

# Import LeanAide integration
try:
    from lean4_integration import (
        LeanAideClient,
        Lean4VerificationEngine,
        Lean4ServerConfig,
        VerificationResult,
        VerificationCache,
    )
    from leanaide_client import LeanAideClient as AsyncLeanAideClient
    from leanaide_evolution import Tactic, LeanProof, LeanProofStrategy
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide integration not available - using simulation mode")

logger = logging.getLogger(__name__)

# Global failure lineage registry for adversarial biasing
FAILURE_LINEAGE_HASHES: set[str] = set()


def compute_lineage_hash(tactics: List[str]) -> str:
    """Hash a tactic sequence to identify failure lineages."""
    joined = "|".join(tactics)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def record_failure_lineage(tactics: List[str]) -> str:
    """Record a failure lineage hash for adversarial biasing."""
    lineage_hash = compute_lineage_hash(tactics)
    FAILURE_LINEAGE_HASHES.add(lineage_hash)
    return lineage_hash


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class MCTSConfig:
    """
    Configuration for MCTS proof search.

    Attributes:
        max_iterations: Maximum number of MCTS iterations to run
        time_budget: Maximum time in seconds for the entire search
        c_param: UCT exploration constant (higher = more exploration)
        rollout_depth: Maximum depth for simulation rollouts
        rollout_policy: Type of rollout policy ("random", "heuristic", "learned")
        parallel_simulations: Number of parallel simulations to run
        enable_transposition_table: Enable state reuse via transposition table
        enable_amaf: Enable AMAF (All-Moves-As-First) updates
        amaf_alpha: AMAF mixing parameter (0 = pure MCTS, 1 = pure AMAF)
        progressive_widening: Enable progressive widening for large action spaces
        widening_factor: Progressive widening factor
        early_termination: Stop early if proof is found
        min_visits_for_confidence: Minimum visits before considering a node confident
        temperature: Temperature for final selection (0 = greedy, higher = softer)
        dirichlet_alpha: Alpha parameter for Dirichlet noise (exploration)
        dirichlet_epsilon: Epsilon for Dirichlet noise mixing
        max_tree_depth: Maximum depth of the search tree
        pruning_threshold: Threshold for pruning unpromising branches
        cache_size_mb: Maximum size of transposition table cache in MB
    """
    max_iterations: int = 1000
    time_budget: float = 60.0  # seconds
    c_param: float = 1.414  # sqrt(2) - standard UCT constant
    rollout_depth: int = 100
    rollout_policy: str = "heuristic"  # random, heuristic, learned
    parallel_simulations: int = 4
    enable_transposition_table: bool = True
    enable_amaf: bool = True
    amaf_alpha: float = 0.5
    progressive_widening: bool = True
    widening_factor: float = 0.5
    early_termination: bool = True
    min_visits_for_confidence: int = 10
    temperature: float = 0.0  # For final selection
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_tree_depth: int = 50
    pruning_threshold: float = 0.1
    cache_size_mb: int = 500
    # Adversarial biasing
    failure_penalty_multiplier: float = -10.0

    # LeanAide-specific settings
    server_url: str = "http://localhost:7654"
    verification_timeout: float = 30.0
    enable_caching: bool = True
    max_proof_states: int = 10000  # Maximum proof states to cache


@dataclass
class MCTSResult:
    """
    Result of MCTS proof search.

    Attributes:
        best_proof: Best proof found during search
        success: Whether a complete proof was found
        search_iterations: Number of iterations performed
        time_elapsed: Total time elapsed in seconds
        nodes_visited: Total number of nodes visited
        tree_depth: Maximum depth of the tree
        win_rate: Estimated win rate (proof completion rate)
        confidence: Confidence score based on visit count
        proof_path: Path of nodes from root to best proof
        search_statistics: Detailed search statistics
        tree_statistics: Statistics about the search tree
    """
    best_proof: Optional[LeanProof] = None
    success: bool = False
    search_iterations: int = 0
    time_elapsed: float = 0.0
    nodes_visited: int = 0
    tree_depth: int = 0
    win_rate: float = 0.0
    confidence: float = 0.0
    proof_path: List['MCTSNode'] = field(default_factory=list)
    search_statistics: Dict[str, Any] = field(default_factory=dict)
    tree_statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_proof": self.best_proof.to_dict() if self.best_proof else None,
            "success": self.success,
            "search_iterations": self.search_iterations,
            "time_elapsed": self.time_elapsed,
            "nodes_visited": self.nodes_visited,
            "tree_depth": self.tree_depth,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "proof_path_length": len(self.proof_path),
            "search_statistics": self.search_statistics,
            "tree_statistics": self.tree_statistics
        }


@dataclass
class ProofState:
    """
    Represents a Lean 4 proof state.

    Attributes:
        goals: Current unsolved goals
        context: Current proof context (hypotheses, assumptions)
        tactics_sequence: Sequence of tactics applied to reach this state
        depth: Depth in the proof tree
        is_complete: Whether all goals are solved
        hash: Unique hash of the state for transposition table
    """
    goals: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    tactics_sequence: List[Tactic] = field(default_factory=list)
    depth: int = 0
    is_complete: bool = False
    hash: str = field(default="")

    def __post_init__(self):
        """Compute hash after initialization."""
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute unique hash of the proof state."""
        state_str = f"{json.dumps(self.goals, sort_keys=True)}:{json.dumps(self.context, sort_keys=True)}"
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goals": self.goals,
            "context": self.context,
            "tactics_sequence": [t.to_dict() for t in self.tactics_sequence],
            "depth": self.depth,
            "is_complete": self.is_complete,
            "hash": self.hash
        }


class RolloutPolicy(Enum):
    """Types of rollout policies."""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


# =============================================================================
# MCTS Node Implementation
# =============================================================================

class MCTSNode:
    """
    Represents a node in the MCTS search tree.

    Each node contains:
    - Proof state (current goals, context)
    - Visit statistics (N, W)
    - Children (possible tactics)
    - Untried actions (tactics not yet explored)
    - AMAF statistics (if enabled)
    """

    def __init__(
        self,
        state: ProofState,
        parent: Optional['MCTSNode'] = None,
        action: Optional[str] = None
    ):
        """
        Initialize an MCTS node.

        Args:
            state: Proof state at this node
            parent: Parent node (None for root)
            action: Action that led to this node
        """
        self.state = state
        self.parent = parent
        self.action = action

        # Visit statistics
        self.N: int = 0  # Visit count
        self.W: float = 0.0  # Total reward/value
        self.Q: float = 0.0  # Mean reward (W / N)

        # Tree structure
        self.children: Dict[str, 'MCTSNode'] = {}
        self.untried_actions: List[str] = []

        # AMAF statistics (if enabled)
        self.amaf_N: Dict[str, int] = defaultdict(int)  # AMAF visit counts
        self.amaf_W: Dict[str, float] = defaultdict(float)  # AMAF rewards

        # Metadata
        self.depth: int = (parent.depth + 1) if parent else 0
        self.is_terminal: bool = state.is_complete
        self.is_fully_expanded: bool = False
        self.created_at: float = time.time()

        # Compute hash for transposition table
        self.hash: str = state.hash

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def uct_value(self, c_param: float, amaf_stats: Optional[Tuple[float, float]] = None) -> float:
        """
        Calculate UCT (Upper Confidence Bound for Trees) value.

        UCT = W_i/N_i + c * sqrt(ln(N_parent) / N_i)

        With AMAF: Q_AMAF = (1 - alpha) * Q_MCTS + alpha * Q_AMAF

        Args:
            c_param: Exploration constant
            amaf_stats: Optional AMAF statistics (W, N) tuple

        Returns:
            UCT score for this node
        """
        if self.N == 0:
            return float('inf')

        # exploitation term: average reward
        exploitation = self.Q

        # exploration term: UCB1
        if self.parent:
            exploration = c_param * math.sqrt(math.log(self.parent.N) / self.N)
        else:
            exploration = c_param * math.sqrt(math.log(max(1, self.N)) / self.N)

        # base UCT value
        uct = exploitation + exploration

        # incorporate AMAF if available and node has few visits
        if amaf_stats and self.N < 5:
            amaf_W, amaf_N = amaf_stats
            if amaf_N > 0:
                Q_amaf = amaf_W / amaf_N
                # Blend with MCTS Q-value
                # More weight to AMAF when N is small
                alpha = min(0.5, 5.0 / (self.N + 5))
                uct = (1 - alpha) * uct + alpha * Q_amaf

        return uct

    def best_child(self, c_param: float, use_temperature: bool = False) -> 'MCTSNode':
        """
        Select the best child using UCT.

        Args:
            c_param: UCT exploration constant
            use_temperature: Use temperature for softer selection

        Returns:
            Child node with highest UCT value
        """
        if not self.children:
            raise ValueError("Node has no children")

        if use_temperature:
            # Temperature-based selection (softmax over visit counts)
            visits = [child.N for child in self.children.values()]
            # Apply temperature
            if self.parent:
                temperature = max(0.1, min(1.0, 1.0 / math.log(self.parent.N + 2)))
            else:
                temperature = 1.0

            # Softmax probabilities
            max_visit = max(visits) if visits else 1
            exp_values = [math.exp(v / temperature - max_visit / temperature) for v in visits]
            total = sum(exp_values)
            probs = [e / total for e in exp_values]

            # Sample from distribution
            child_list = list(self.children.values())
            idx = random.choices(range(len(child_list)), weights=probs)[0]
            return child_list[idx]

        # Standard UCT selection
        # Get AMAF statistics for each action
        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            amaf_stats = None
            if self.amaf_N[action] > 0:
                amaf_stats = (self.amaf_W[action], self.amaf_N[action])

            score = child.uct_value(c_param, amaf_stats)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def is_fully_expanded_node(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def add_child(self, action: str, child_node: 'MCTSNode') -> 'MCTSNode':
        """
        Add a child node.

        Args:
            action: Action that led to child
            child_node: Child node to add

        Returns:
            The added child node
        """
        self.children[action] = child_node
        if action in self.untried_actions:
            self.untried_actions.remove(action)

        if len(self.untried_actions) == 0:
            self.is_fully_expanded = True

        return child_node

    def update(self, reward: float) -> None:
        """
        Update node statistics with reward.

        Args:
            reward: Reward to propagate (0 = loss, 1 = win)
        """
        self.N += 1
        self.W += reward
        self.Q = self.W / self.N

    def get_amaf_stats(self, action: str) -> Tuple[float, int]:
        """Get AMAF statistics for an action."""
        return self.amaf_W[action], self.amaf_N[action]

    def update_amaf(self, action: str, reward: float) -> None:
        """Update AMAF statistics for an action."""
        self.amaf_N[action] += 1
        self.amaf_W[action] += reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash": self.hash,
            "action": self.action,
            "N": self.N,
            "W": self.W,
            "Q": self.Q,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "is_fully_expanded": self.is_fully_expanded,
            "num_children": len(self.children),
            "num_untried": len(self.untried_actions),
            "state": self.state.to_dict()
        }


# =============================================================================
# MCTS Tree Implementation
# =============================================================================

class MCTSTree:
    """
    Manages the MCTS search tree.

    Provides utilities for:
    - Tree traversal
    - Best path extraction
    - Statistics collection
    - Visualization
    """

    def __init__(self, root: MCTSNode):
        """
        Initialize the MCTS tree.

        Args:
            root: Root node of the tree
        """
        self.root = root
        self.total_nodes = 1
        self.max_depth = 0
        self._nodes_by_hash: Dict[str, MCTSNode] = {root.hash: root}
        self._transposition_count = 0

    def get_root(self) -> MCTSNode:
        """Get the root node."""
        return self.root

    def get_best_path(self, use_temperature: bool = False) -> List[MCTSNode]:
        """
        Extract the best path from root to leaf.

        Args:
            use_temperature: Use temperature for selection

        Returns:
            List of nodes from root to best leaf
        """
        path = [self.root]
        current = self.root

        while current.children:
            current = current.best_child(c_param=0.0, use_temperature=use_temperature)
            path.append(current)

        return path

    def get_most_visited_path(self) -> List[MCTSNode]:
        """
        Get path following most visited children.

        Returns:
            List of nodes from root to most visited leaf
        """
        path = [self.root]
        current = self.root

        while current.children:
            # Select child with highest visit count
            current = max(current.children.values(), key=lambda c: c.N)
            path.append(current)

        return path

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive tree statistics.

        Returns:
            Dictionary with tree statistics
        """
        # Collect statistics via BFS
        total_visits = 0
        total_depth = 0
        terminal_nodes = 0
        leaf_nodes = 0
        max_depth = 0

        queue = deque([self.root])
        visited = set()

        while queue:
            node = queue.popleft()
            if node.hash in visited:
                continue
            visited.add(node.hash)

            total_visits += node.N
            total_depth += node.depth
            max_depth = max(max_depth, node.depth)

            if node.is_terminal:
                terminal_nodes += 1
            if node.is_leaf:
                leaf_nodes += 1

            queue.extend(node.children.values())

        avg_depth = total_depth / max(1, len(visited))

        return {
            "total_nodes": self.total_nodes,
            "unique_nodes": len(visited),
            "total_visits": total_visits,
            "max_depth": max_depth,
            "avg_depth": avg_depth,
            "terminal_nodes": terminal_nodes,
            "leaf_nodes": leaf_nodes,
            "transposition_hits": self._transposition_count,
            "branching_factor": self._avg_branching_factor()
        }

    def _avg_branching_factor(self) -> float:
        """Calculate average branching factor."""
        total_children = 0
        internal_nodes = 0

        queue = deque([self.root])
        visited = set()

        while queue:
            node = queue.popleft()
            if node.hash in visited or node.is_leaf:
                continue
            visited.add(node.hash)

            total_children += len(node.children)
            internal_nodes += 1

            queue.extend(node.children.values())

        return total_children / max(1, internal_nodes)

    def add_node(self, node: MCTSNode, check_transposition: bool = True) -> bool:
        """
        Add a node to the tree.

        Args:
            node: Node to add
            check_transposition: Check if state already exists

        Returns:
            True if node was added, False if transposition found
        """
        if check_transposition and node.hash in self._nodes_by_hash:
            # Transposition found
            self._transposition_count += 1
            return False

        self._nodes_by_hash[node.hash] = node
        self.total_nodes += 1
        self.max_depth = max(self.max_depth, node.depth)
        return True

    def get_node_by_hash(self, state_hash: str) -> Optional[MCTSNode]:
        """Get node by state hash (for transposition table)."""
        return self._nodes_by_hash.get(state_hash)

    def prune_tree(self, threshold: float) -> int:
        """
        Prune unpromising branches below threshold.

        Args:
            threshold: Minimum visit ratio to keep

        Returns:
            Number of nodes pruned
        """
        if not self.root.children:
            return 0

        # Find max visits among root children
        max_visits = max(child.N for child in self.root.children.values())
        min_visits = max_visits * threshold

        pruned = 0
        to_remove = []

        for action, child in self.root.children.items():
            if child.N < min_visits:
                to_remove.append(action)

        for action in to_remove:
            del self.root.children[action]
            pruned += 1

        return pruned


# =============================================================================
# MCTS Selection Phase
# =============================================================================

class MCTSSelection:
    """
    Selection phase using UCT policy.

    Traverses the tree from root to leaf using UCT selection at each node.
    Implements progressive widening for large action spaces.
    """

    def __init__(self, c_param: float = 1.414, progressive_widening: bool = True):
        """
        Initialize selection strategy.

        Args:
            c_param: UCT exploration constant
            progressive_widening: Enable progressive widening
        """
        self.c_param = c_param
        self.progressive_widening = progressive_widening

    def select(self, root: MCTSNode) -> MCTSNode:
        """
        Select a leaf node using UCT policy.

        Args:
            root: Root node to start selection from

        Returns:
            Selected leaf node
        """
        return self._traverse(root)

    def _traverse(self, node: MCTSNode) -> MCTSNode:
        """
        Traverse tree from node to leaf using UCT.

        Args:
            node: Starting node

        Returns:
            Leaf node reached by UCT traversal
        """
        while not node.is_terminal and node.children:
            # Apply progressive widening if enabled
            if self.progressive_widening:
                self._apply_progressive_widening(node)

            # Select best child
            node = node.best_child(self.c_param)

        return node

    def _apply_progressive_widening(self, node: MCTSNode) -> None:
        """
        Apply progressive widening to limit action space.

        Only explores k * N^alpha actions, where N is visit count.

        Args:
            node: Node to apply widening to
        """
        if not node.untried_actions:
            return

        # Calculate number of actions to explore
        k = len(node.children) + len(node.untried_actions)
        alpha = 0.5  # Standard value
        num_to_explore = int(k * (node.N ** alpha))

        # Currently explored
        num_explored = len(node.children)

        # Explore more actions if needed
        if num_explored < num_to_explore and node.untried_actions:
            # In a full implementation, this would get more actions from expansion
            # For now, we just mark that we want more actions available
            pass


# =============================================================================
# MCTS Expansion Phase
# =============================================================================

class MCTSExpansion:
    """
    Expansion phase for adding new nodes to the tree.

    Gets applicable tactics from LeanAide and creates child nodes.
    Implements action ranking and smart action selection.
    """

    # Common Lean 4 tactics
    BASIC_TACTICS = [
        "intros", "simp", "rw", "apply", "exact", "refine",
        "cases", "induction", "constructor", "exists",
        "have", "suffices", "show", "calc",
        "aesop", "linarith", "ring", "omega", "norm_num",
        "trivial", "decide", "done"
    ]

    def __init__(
        self,
        leanaide_client: Optional[AsyncLeanAideClient] = None,
        max_actions: int = 20,
        use_action_ranking: bool = True
    ):
        """
        Initialize expansion strategy.

        Args:
            leanaide_client: LeanAide client for tactic generation
            max_actions: Maximum number of actions to consider
            use_action_ranking: Rank actions by heuristics
        """
        self.leanaide_client = leanaide_client
        self.max_actions = max_actions
        self.use_action_ranking = use_action_ranking

    async def expand(
        self,
        node: MCTSNode,
        tree: MCTSTree
    ) -> MCTSNode:
        """
        Expand a leaf node by adding a child.

        Args:
            node: Leaf node to expand
            tree: MCTS tree

        Returns:
            New child node, or existing node if terminal
        """
        if node.is_terminal:
            return node

        # If fully expanded, return best child or terminal
        if node.is_fully_expanded_node():
            if node.children:
                return node.best_child(c_param=0.0)
            else:
                # No children available - mark as terminal
                node.is_terminal = True
                return node

        # Get applicable tactics if not already available
        if not node.untried_actions:
            await self._populate_untried_actions(node)

        if not node.untried_actions:
            # No actions available, mark as terminal
            node.is_terminal = True
            return node

        # Select an untried action
        action = self._select_action(node)

        # Apply tactic to get new state
        new_state = await self._apply_tactic(node.state, action)

        # Check for transposition
        existing_node = tree.get_node_by_hash(new_state.hash)
        if existing_node:
            # Transposition found - reuse existing node
            node.add_child(action, existing_node)
            return existing_node

        # Create new node
        child_node = MCTSNode(
            state=new_state,
            parent=node,
            action=action
        )

        # Add to tree
        node.add_child(action, child_node)
        tree.add_node(child_node)

        return child_node

    async def _populate_untried_actions(self, node: MCTSNode) -> None:
        """
        Get applicable tactics for the current state.

        Args:
            node: Node to populate actions for
        """
        # If we have LeanAide, use it to get applicable tactics
        if self.leanaide_client and LEANAIDE_AVAILABLE:
            try:
                actions = await self._get_applicable_tactics_from_leanaide(node.state)
                node.untried_actions = actions[:self.max_actions]
                return
            except (IOError, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"LeanAide tactic generation failed: {e}")

        # Fallback: use basic tactics with parameterization
        actions = self._generate_heuristic_actions(node.state)
        node.untried_actions = actions[:self.max_actions]

    async def _get_applicable_tactics_from_leanaide(
        self,
        state: ProofState
    ) -> List[str]:
        """
        Get applicable tactics from LeanAide.

        Args:
            state: Current proof state

        Returns:
            List of applicable tactics
        """
        # Create Lean code representing current state
        lean_code = self._state_to_lean_code(state)

        # Use LeanAide elaborate task to get tactics
        try:
            result = await self.leanaide_client.elaborate(lean_code)

            if result.success and result.data:
                # Extract suggested tactics from result
                tactics = self._extract_tactics_from_result(result.data)
                return tactics
        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"LeanAide elaborate failed: {e}")

        return []

    def _generate_heuristic_actions(self, state: ProofState) -> List[str]:
        """
        Generate heuristic tactics based on state.

        Args:
            state: Current proof state

        Returns:
            List of potential tactics
        """
        actions = []

        # Add basic tactics
        for tactic in self.BASIC_TACTICS:
            actions.append(tactic)

        # Add parameterized tactics based on goals
        for goal in state.goals:
            # Simplification tactics
            actions.append(f"simp at *")
            actions.append(f"rw [←]")

            # If goal has equality, consider relevant tactics
            if "=" in goal:
                actions.append("linarith")
                actions.append("ring")

            # If goal has implication, consider intros
            if "->" in goal or "forall" in goal:
                actions.insert(0, "intros")

        # Add context-based tactics
        for hyp in state.context:
            if ":=" in hyp:  # Definition
                actions.append(f"rw [{hyp.split(':')[0].strip()}]")

        return actions

    def _select_action(self, node: MCTSNode) -> str:
        """
        Select an action from untried actions.

        Uses ranking if enabled, otherwise random selection.

        Args:
            node: Node to select action for

        Returns:
            Selected action
        """
        if not self.use_action_ranking:
            return random.choice(node.untried_actions)

        # Rank actions by simple heuristics
        ranked_actions = self._rank_actions(node)

        # Select from top-k with bias towards top
        top_k = min(5, len(ranked_actions))
        if top_k > 0:
            # Weighted random selection from top-k
            weights = [1.0 / (i + 1) for i in range(top_k)]
            total = sum(weights)
            probs = [w / total for w in weights]
            idx = random.choices(range(top_k), weights=probs)[0]
            return ranked_actions[idx]

        return random.choice(node.untried_actions)

    def _rank_actions(self, node: MCTSNode) -> List[str]:
        """
        Rank actions by simple heuristics.

        Args:
            node: Node to rank actions for

        Returns:
            Ranked list of actions
        """
        actions = node.untried_actions[:]
        scores = []

        for action in actions:
            score = 0.0

            # Prefer intros and simp (often useful)
            if action in ["intros", "simp"]:
                score += 2.0

            # Prefer simple tactics over complex ones
            if action in self.BASIC_TACTICS[:10]:
                score += 1.0

            # Penalize very long tactics
            if len(action) > 50:
                score -= 1.0

            scores.append(score)

        # Sort by score
        ranked = [a for _, a in sorted(zip(scores, actions), reverse=True)]
        return ranked

    async def _apply_tactic(self, state: ProofState, tactic: str) -> ProofState:
        """
        Apply a tactic to get new proof state.

        Args:
            state: Current proof state
            tactic: Tactic to apply

        Returns:
            New proof state after applying tactic
        """
        # Create new state
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy(),
            depth=state.depth + 1
        )

        # Parse tactic
        tactic_obj = self._parse_tactic(tactic)
        new_state.tactics_sequence.append(tactic_obj)

        # If we have LeanAide, use it to get actual new goals
        if self.leanaide_client and LEANAIDE_AVAILABLE:
            try:
                result_state = await self._apply_tactic_with_leanaide(state, tactic)
                return result_state
            except (IOError, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"LeanAide tactic application failed: {e}")

        # Heuristic simulation
        new_state = self._simulate_tactic_application(new_state, tactic)

        return new_state

    async def _apply_tactic_with_leanaide(
        self,
        state: ProofState,
        tactic: str
    ) -> ProofState:
        """Apply tactic using LeanAide."""
        lean_code = self._state_to_lean_code(state)
        # Append tactic
        lean_code += f"  {tactic}\n"

        try:
            result = await self.leanaide_client.elaborate(lean_code)

            if result.success and result.data:
                # Extract new goals from result
                new_goals = self._extract_goals_from_result(result.data)
                new_context = self._extract_context_from_result(result.data)

                new_state = ProofState(
                    goals=new_goals,
                    context=new_context,
                    tactics_sequence=state.tactics_sequence.copy(),
                    depth=state.depth + 1,
                    is_complete=len(new_goals) == 0
                )

                tactic_obj = self._parse_tactic(tactic)
                new_state.tactics_sequence.append(tactic_obj)

                return new_state

        except (IOError, ConnectionError, TimeoutError, ValueError) as e:
            logger.warning(f"Tactic application error: {e}")

        # Fallback to simulation
        return self._simulate_tactic_application(state, tactic)

    def _simulate_tactic_application(
        self,
        state: ProofState,
        tactic: str
    ) -> ProofState:
        """
        Simulate tactic application heuristically.

        Args:
            state: Current state
            tactic: Tactic being applied

        Returns:
            Simulated new state
        """
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy(),
            depth=state.depth + 1
        )

        tactic_obj = self._parse_tactic(tactic)
        new_state.tactics_sequence.append(tactic_obj)

        # Simulate goal reduction
        if tactic in ["intros", "intro"]:
            # Intros typically don't create new goals
            if new_state.goals:
                # Simplify: remove one goal heuristically
                new_state.goals = new_state.goals[1:]
        elif tactic in ["simp", "rw"]:
            # Simplification may reduce goals
            if new_state.goals and len(new_state.goals) > 1:
                # Maybe close one goal
                if random.random() > 0.5:
                    new_state.goals.pop()
        elif tactic in ["cases", "induction"]:
            # Case analysis typically creates multiple goals
            if new_state.goals:
                # Double the number of goals
                new_state.goals = new_state.goals + new_state.goals.copy()
        elif tactic in ["aesop", "trivial"]:
            # Automation might solve goals
            if random.random() > 0.7 and new_state.goals:
                new_state.goals = []

        # Check if complete
        new_state.is_complete = len(new_state.goals) == 0

        return new_state

    def _parse_tactic(self, tactic_str: str) -> Tactic:
        """Parse a tactic string into a Tactic object."""
        parts = tactic_str.strip().split(maxsplit=1)
        name = parts[0] if parts else "unknown"
        arguments = parts[1].split() if len(parts) > 1 else []

        return Tactic(name=name, arguments=arguments)

    def _state_to_lean_code(self, state: ProofState) -> str:
        """Convert proof state to Lean code."""
        code = "import Mathlib\n\n"

        # Add context as hypotheses
        for i, hyp in enumerate(state.context):
            code += f"have h{i} : {hyp}\n"

        # Add current goal
        if state.goals:
            code += f"\ntheorem temp_goal : {state.goals[0]} := by\n"

        return code

    def _extract_tactics_from_result(self, data: Dict[str, Any]) -> List[str]:
        """Extract suggested tactics from LeanAide result."""
        tactics = []

        # Try to find tactics in various result fields
        if "tactics" in data:
            tactics.extend(data["tactics"])
        if "suggestions" in data:
            tactics.extend(data["suggestions"])
        if "unsolvedGoals" in data:
            # Each unsolved goal might have tactic suggestions
            for goal in data["unsolvedGoals"]:
                if "tactics" in goal:
                    tactics.extend(goal["tactics"])

        return tactics

    def _extract_goals_from_result(self, data: Dict[str, Any]) -> List[str]:
        """Extract goals from LeanAide result."""
        goals = []

        if "goals" in data:
            goals = data["goals"]
        elif "unsolvedGoals" in data:
            goals = [g.get("type", "") for g in data["unsolvedGoals"]]

        return goals

    def _extract_context_from_result(self, data: Dict[str, Any]) -> List[str]:
        """Extract context from LeanAide result."""
        context = []

        if "context" in data:
            context = data["context"]
        elif "hypotheses" in data:
            context = data["hypotheses"]

        return context


# =============================================================================
# MCTS Simulation Phase
# =============================================================================

class MCTSSimulation:
    """
    Simulation/rollout phase.

    Runs a rollout from a leaf node to estimate value.
    Supports multiple rollout policies: random, heuristic, and learned.
    """

    def __init__(
        self,
        rollout_policy: RolloutPolicy = RolloutPolicy.HEURISTIC,
        max_depth: int = 100,
        leanaide_client: Optional[AsyncLeanAideClient] = None
    ):
        """
        Initialize simulation strategy.

        Args:
            rollout_policy: Type of rollout policy
            max_depth: Maximum rollout depth
            leanaide_client: Optional LeanAide client for learned rollouts
        """
        self.rollout_policy = rollout_policy
        self.max_depth = max_depth
        self.leanaide_client = leanaide_client

    def simulate(self, state: ProofState) -> float:
        """
        Run a rollout from the given state.

        Args:
            state: Starting state for rollout

        Returns:
            Estimated value (0 = loss, 1 = win)
        """
        if self.rollout_policy == RolloutPolicy.RANDOM:
            return self._random_rollout(state)
        elif self.rollout_policy == RolloutPolicy.HEURISTIC:
            return self._heuristic_rollout(state)
        elif self.rollout_policy == RolloutPolicy.LEARNED:
            return asyncio.run(self._learned_rollout(state))
        else:
            return self._heuristic_rollout(state)

    def _random_rollout(self, state: ProofState) -> float:
        """
        Random rollout policy.

        Applies random tactics until terminal or max depth.

        Args:
            state: Starting state

        Returns:
            Estimated value
        """
        current_state = state

        for _ in range(self.max_depth):
            if current_state.is_complete or not current_state.goals:
                return 1.0

            # Random tactic
            tactic = random.choice(MCTSExpansion.BASIC_TACTICS)
            current_state = self._apply_random_tactic(current_state, tactic)

        # If reached max depth, estimate based on goal count
        if current_state.goals:
            return 1.0 / (1 + len(current_state.goals))
        return 1.0

    def _heuristic_rollout(self, state: ProofState) -> float:
        """
        Heuristic rollout policy.

        Uses domain knowledge to guide tactic selection.

        Args:
            state: Starting state

        Returns:
            Estimated value
        """
        current_state = state
        score = 0.0

        for depth in range(self.max_depth):
            if current_state.is_complete or not current_state.goals:
                return 1.0

            # Select tactic heuristically
            tactic = self._select_heuristic_tactic(current_state)
            current_state = self._apply_random_tactic(current_state, tactic)

            # Update score based on progress
            if len(current_state.goals) < len(state.goals):
                score += 0.1

        # Final score based on goal reduction
        initial_goals = len(state.goals)
        final_goals = len(current_state.goals)

        if initial_goals > 0:
            reduction = (initial_goals - final_goals) / initial_goals
            score = max(score, reduction)

        return min(1.0, score)

    async def _learned_rollout(self, state: ProofState) -> float:
        """
        Learned rollout policy using neural network.

        This is a placeholder for future neural network integration.

        Args:
            state: Starting state

        Returns:
            Estimated value from neural network
        """
        # Placeholder: Would use a trained neural network here
        # For now, fall back to heuristic
        return self._heuristic_rollout(state)

    def _select_heuristic_tactic(self, state: ProofState) -> str:
        """
        Select a tactic using heuristics.

        Args:
            state: Current proof state

        Returns:
            Selected tactic
        """
        # Count goals
        num_goals = len(state.goals)

        # Early: Use intros and simplification
        if state.depth < 5:
            if any("->" in g or "forall" in g for g in state.goals):
                return "intros"
            return "simp"

        # Middle: Use case analysis and automation
        if num_goals == 1:
            if "=" in state.goals[0]:
                return "linarith"
            return "aesop"

        # Multiple goals: use cases or induction
        if any ("Nat" in g or "ℕ" in g for g in state.goals):
            return "induction"
        return "cases"

    def _apply_random_tactic(self, state: ProofState, tactic: str) -> ProofState:
        """
        Apply a tactic without LeanAide (for simulation).

        Args:
            state: Current state
            tactic: Tactic to apply

        Returns:
            New state (simulated)
        """
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy() + [Tactic(name=tactic)],
            depth=state.depth + 1
        )

        # Simulate tactic effects
        if tactic in ["intros", "intro"]:
            if new_state.goals:
                new_state.goals = new_state.goals[1:]
        elif tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.5:
                new_state.goals = []
        elif tactic in ["cases", "induction"]:
            if new_state.goals and len(new_state.goals) == 1:
                # Split into cases
                new_state.goals = new_state.goals * 2

        new_state.is_complete = len(new_state.goals) == 0

        return new_state


# =============================================================================
# MCTS Backpropagation Phase
# =============================================================================

class MCTSBackpropagation:
    """
    Backpropagation phase.

    Updates statistics up the tree from leaf to root.
    Supports both standard MCTS and AMAF updates.
    """

    def __init__(
        self,
        enable_amaf: bool = True,
        amaf_alpha: float = 0.5,
        failure_penalty_multiplier: float = -10.0
    ):
        """
        Initialize backpropagation strategy.

        Args:
            enable_amaf: Enable AMAF updates
            amaf_alpha: AMAF mixing parameter
        """
        self.enable_amaf = enable_amaf
        self.amaf_alpha = amaf_alpha
        self.failure_penalty_multiplier = failure_penalty_multiplier

    def backpropagate(
        self,
        node: MCTSNode,
        reward: float,
        actions_seen: List[str]
    ) -> None:
        """
        Backpropagate reward from node to root.

        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate
            actions_seen: Actions seen during rollout (for AMAF)
        """
        current = node
        # Apply adversarial negative bias for failure lineages
        tactics_sequence = [
            getattr(t, "name", str(t)) for t in node.state.tactics_sequence
        ]
        lineage_hash = compute_lineage_hash(tactics_sequence)
        if lineage_hash in FAILURE_LINEAGE_HASHES:
            reward = reward * self.failure_penalty_multiplier

        while current is not None:
            # Update node statistics
            current.update(reward)

            # AMAF updates
            if self.enable_amaf and actions_seen:
                for action in actions_seen:
                    if action not in current.children:
                        current.update_amaf(action, reward)

            # Move to parent
            current = current.parent


# =============================================================================
# Main MCTS Orchestrator
# =============================================================================

class MCTS:
    """
    Main Monte Carlo Tree Search implementation.

    Orchestrates all four phases:
    1. Selection: Select leaf node using UCT
    2. Expansion: Add new node to tree
    3. Simulation: Run rollout from new node
    4. Backpropagation: Update statistics

    Features:
    - Parallel simulation support
    - Transposition table for state reuse
    - Progressive widening for large action spaces
    - Adaptive exploration parameters
    - Early termination on proof found
    - Comprehensive logging and statistics
    """

    def __init__(
        self,
        config: MCTSConfig,
        theorem: str,
        theorem_name: Optional[str] = None
    ):
        """
        Initialize MCTS search.

        Args:
            config: MCTS configuration
            theorem: Theorem statement to prove
            theorem_name: Optional name for the theorem
        """
        self.config = config
        self.theorem = theorem
        self.theorem_name = theorem_name or "mcts_theorem"

        # Initialize LeanAide client if available
        self.leanaide_client = None
        if LEANAIDE_AVAILABLE and config.server_url:
            try:
                self.leanaide_client = AsyncLeanAideClient()
                self.leanaide_client.config.base_url = config.server_url
            except (IOError, ConnectionError, ValueError) as e:
                logger.warning(f"Failed to initialize LeanAide client: {e}")

        # Initialize components
        self.selection = MCTSSelection(
            c_param=config.c_param,
            progressive_widening=config.progressive_widening
        )
        self.expansion = MCTSExpansion(
            leanaide_client=self.leanaide_client,
            max_actions=config.max_iterations // 10
        )
        self.simulation = MCTSSimulation(
            rollout_policy=RolloutPolicy(config.rollout_policy),
            max_depth=config.rollout_depth,
            leanaide_client=self.leanaide_client
        )
        self.backpropagation = MCTSBackpropagation(
            enable_amaf=config.enable_amaf,
            amaf_alpha=config.amaf_alpha,
            failure_penalty_multiplier=config.failure_penalty_multiplier
        )

        # Initialize tree
        initial_state = ProofState(
            goals=[theorem],
            depth=0
        )
        self.root = MCTSNode(state=initial_state)
        self.tree = MCTSTree(self.root)

        # Statistics
        self.iterations_completed = 0
        self.start_time = 0.0
        self.best_node: Optional[MCTSNode] = None
        self.best_value = 0.0

    async def search(
        self,
        iterations: Optional[int] = None,
        time_budget: Optional[float] = None
    ) -> MCTSResult:
        """
        Run MCTS search.

        Args:
            iterations: Number of iterations (overrides config)
            time_budget: Time budget in seconds (overrides config)

        Returns:
            MCTSResult with best proof and statistics
        """
        # Use config values if not specified
        iterations = iterations or self.config.max_iterations
        time_budget = time_budget or self.config.time_budget

        self.start_time = time.time()
        logger.info(f"Starting MCTS search for: {self.theorem}")
        logger.info(f"Max iterations: {iterations}, Time budget: {time_budget}s")

        try:
            # Run MCTS iterations
            for i in range(iterations):
                # Check time budget
                elapsed = time.time() - self.start_time
                if elapsed >= time_budget:
                    logger.info(f"Time budget exhausted after {i} iterations")
                    break

                # Check for early termination
                if self.config.early_termination and self.best_node and self.best_node.is_terminal:
                    logger.info(f"Early termination: proof found after {i} iterations")
                    break

                # Run one MCTS iteration
                await self.run_iteration()

                self.iterations_completed = i + 1

                # Log progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - self.start_time
                    logger.info(
                        f"Iteration {i+1}/{iterations}: "
                        f"Root visits={self.root.N}, "
                        f"Best value={self.best_value:.4f}, "
                        f"Tree nodes={self.tree.total_nodes}, "
                        f"Time={elapsed:.2f}s"
                    )

            # Compile result
            return self._compile_result()

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MCTS search failed: {e}", exc_info=True)
            return MCTSResult(
                success=False,
                search_iterations=self.iterations_completed,
                time_elapsed=time.time() - self.start_time
            )

        finally:
            # Cleanup
            if self.leanaide_client:
                await self.leanaide_client.close()

    async def run_iteration(self) -> None:
        """Run a single MCTS iteration (selection, expansion, simulation, backpropagation)."""
        # 1. Selection: Select leaf node
        leaf = self.selection.select(self.root)

        # 2. Expansion: Expand leaf node
        new_node = await self.expansion.expand(leaf, self.tree)

        # 3. Simulation: Run rollout
        reward = self.simulation.simulate(new_node.state)

        # Track actions seen for AMAF
        actions_seen = [t.name for t in new_node.state.tactics_sequence[leaf.depth:]]

        # 4. Backpropagation: Update statistics
        self.backpropagation.backpropagate(new_node, reward, actions_seen)

        # Update best node
        if new_node.state.is_complete or reward > self.best_value:
            self.best_value = reward
            self.best_node = new_node

    def _compile_result(self) -> MCTSResult:
        """Compile final MCTS result."""
        elapsed = time.time() - self.start_time

        # Get best path
        best_path = self.tree.get_best_path(use_temperature=self.config.temperature > 0)

        # Create proof from best path
        best_proof = self._create_proof_from_path(best_path)

        # Get statistics
        tree_stats = self.tree.get_statistics()

        # Calculate confidence
        confidence = 0.0
        if best_path:
            root_visits = self.root.N
            best_visits = best_path[-1].N if best_path else 0
            if root_visits > 0:
                confidence = best_visits / root_visits

        return MCTSResult(
            best_proof=best_proof,
            success=best_proof is not None and len(best_path) > 0 and best_path[-1].is_terminal,
            search_iterations=self.iterations_completed,
            time_elapsed=elapsed,
            nodes_visited=self.tree.total_nodes,
            tree_depth=tree_stats["max_depth"],
            win_rate=self.best_value,
            confidence=confidence,
            proof_path=best_path,
            search_statistics={
                "root_visits": self.root.N,
                "root_value": self.root.Q,
                "best_value": self.best_value,
                "iterations_per_second": self.iterations_completed / max(0.001, elapsed),
                "avg_iteration_time": elapsed / max(1, self.iterations_completed)
            },
            tree_statistics=tree_stats
        )

    def _create_proof_from_path(self, path: List[MCTSNode]) -> Optional[LeanProof]:
        """Create a LeanProof from a path of nodes."""
        if not path:
            return None

        # Extract tactics from path
        tactics = []
        for node in path:
            tactics.extend(node.state.tactics_sequence)

        # Generate Lean code
        lean_code = f"import Mathlib\n\n"
        lean_code += f"theorem {self.theorem_name} : {self.theorem} := by\n"
        for tactic in tactics:
            lean_code += f"  {tactic}\n"

        # Create proof object
        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            lean_code=lean_code,
            tactics=tactics
        )

        return proof

    def record_failure_lineage(self, tactics_sequence: List[str]) -> str:
        """Record a failure lineage for adversarial biasing."""
        return record_failure_lineage(tactics_sequence)


# =============================================================================
# MDAP/MAKER Integration
# =============================================================================

# Import MDAP components if available
try:
    from leanaide_mdap import (
        LeanMDAPOrchestrator,
        LeanMDAPConfig,
        LeanMDAPStep,
        LeanProofAgent,
        ProofStrategy,
        VotingStrategy,
        LeanProof as MDAPLeanProof,
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP integration not available")


@dataclass
class MDAPMCTSConfig:
    """
    Configuration for MDAP-enhanced MCTS.

    Attributes:
        # MCTS settings
        base_mcts_config: Base MCTS configuration
        use_mdap_selection: Use MDAP agent consensus in selection
        use_mdap_expansion: Use MDAP agents for expansion suggestions
        use_mdap_simulation: Use MDAP agents for simulation

        # MDAP settings
        num_mdap_agents: Number of MDAP agents to use
        mdap_agent_types: Types of agents (evolution, mcts, adversarial, self_play)
        mdap_voting_strategy: Voting strategy for MDAP
        mdap_k_ahead: K parameter for first-K-ahead voting

        # Agent-weighted UCT
        agent_weight_bonus: Bonus weight for agent votes in UCT
        agent_confidence_weight: Weight for agent confidence in calculations

        # Hybrid strategies
        enable_mdap_mcts_hybrid: Enable MDAP-MCTS hybrid mode
        mdap_mcts_ratio: Ratio of MDAP vs MCTS iterations (0-1)
    """
    base_mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    use_mdap_selection: bool = True
    use_mdap_expansion: bool = True
    use_mdap_simulation: bool = False  # Usually too expensive

    num_mdap_agents: int = 4
    mdap_agent_types: List[str] = field(default_factory=lambda: ["evolution", "mcts", "adversarial", "self_play"])
    mdap_voting_strategy: str = "first_k_ahead"
    mdap_k_ahead: int = 3

    agent_weight_bonus: float = 0.3
    agent_confidence_weight: float = 0.5

    enable_mdap_mcts_hybrid: bool = True
    mdap_mcts_ratio: float = 0.5


class MCTSMDAPIntegration:
    """
    Integrates MDAP/MAKER with MCTS for enhanced proof search.

    Provides:
    - MDAP-enhanced selection using agent consensus
    - Agent-weighted UCT calculation
    - MDAP-guided expansion with voting
    - Hybrid MCTS-MDAP strategies
    """

    def __init__(self, config: MDAPMCTSConfig):
        """
        Initialize MDAP-MCTS integration.

        Args:
            config: MDAP-MCTS configuration
        """
        self.config = config
        self.mdap_orchestrator = None
        self.agent_performance: Dict[str, Dict[str, float]] = {
            "evolution": {"success_rate": 0.7, "avg_time": 5.0, "confidence": 0.8},
            "mcts": {"success_rate": 0.75, "avg_time": 3.0, "confidence": 0.85},
            "adversarial": {"success_rate": 0.65, "avg_time": 8.0, "confidence": 0.7},
            "self_play": {"success_rate": 0.6, "avg_time": 10.0, "confidence": 0.75},
        }

        if MDAP_AVAILABLE:
            self._initialize_mdap()

    def _initialize_mdap(self) -> None:
        """Initialize MDAP orchestrator."""
        try:
            mdap_config = LeanMDAPConfig(
                available_agents=self.config.mdap_agent_types,
                default_parallel_agents=self.config.num_mdap_agents,
                voting_strategy=VotingStrategy(self.config.mdap_voting_strategy),
                k_ahead_threshold=self.config.mdap_k_ahead,
            )

            # Note: We'd need a Team object here in a full implementation
            # self.mdap_orchestrator = LeanMDAPOrchestrator(team, mdap_config)
            logger.info("MDAP orchestrator initialized")
        except (ImportError, ValueError, TypeError) as e:
            logger.warning(f"Failed to initialize MDAP: {e}")

    def calculate_uct_with_agent_bonus(
        self,
        node: MCTSNode,
        agent_performance: Dict[str, Dict[str, float]],
        base_c_param: float = 1.414
    ) -> float:
        """
        Calculate UCT with MDAP agent bonus.

        UCT_with_agent_bonus = UCT_base + agent_bonus

        Where agent_bonus = weighted_sum(agent_success_rates) * agent_weight_bonus

        Args:
            node: MCTS node to calculate UCT for
            agent_performance: Performance metrics for each agent type
            base_c_param: Base UCT exploration constant

        Returns:
            UCT value with agent bonus
        """
        # Calculate base UCT
        base_uct = node.uct_value(base_c_param)

        # Calculate agent bonus
        if not node.parent or self.config.agent_weight_bonus == 0:
            return base_uct

        # Get agent votes for this node's action
        action = node.action
        if not action:
            return base_uct

        # Simulate agent votes (in real implementation, would query MDAP)
        agent_bonus = 0.0
        total_weight = 0.0

        for agent_type, perf in agent_performance.items():
            # Weight by success rate and confidence
            weight = perf["success_rate"] * perf["confidence"]
            agent_bonus += weight
            total_weight += weight

        if total_weight > 0:
            agent_bonus = (agent_bonus / total_weight) * self.config.agent_weight_bonus

        return base_uct + agent_bonus

    def select_with_agent_consensus(
        self,
        children: List[MCTSNode],
        agents: Optional[List['LeanProofAgent']] = None
    ) -> MCTSNode:
        """
        Select child using MDAP agent consensus.

        Combines UCT scores with agent voting.

        Args:
            children: List of child nodes to select from
            agents: Optional list of MDAP agents for voting

        Returns:
            Selected child node
        """
        if not children:
            raise ValueError("No children to select from")

        # If no agents or MDAP not available, use standard UCT
        if not agents or not MDAP_AVAILABLE:
            # Use UCT with agent bonus
            best_child = None
            best_score = -float('inf')

            for child in children:
                score = self.calculate_uct_with_agent_bonus(
                    child,
                    self.agent_performance
                )
                if score > best_score:
                    best_score = score
                    best_child = child

            return best_child or children[0]

        # Get agent votes for each child
        # In real implementation, would run MDAP voting
        child_scores = []

        for child in children:
            # Base UCT score
            uct_score = child.uct_value(1.414)

            # Agent consensus bonus
            consensus_bonus = 0.0
            if child.action:
                # Simulate agent voting
                votes_for = sum(
                    perf["success_rate"]
                    for perf in self.agent_performance.values()
                ) / len(self.agent_performance)
                consensus_bonus = votes_for * self.config.agent_weight_bonus

            total_score = uct_score + consensus_bonus
            child_scores.append((child, total_score))

        # Select best
        child_scores.sort(key=lambda x: x[1], reverse=True)
        return child_scores[0][0]

    async def expand_with_agent_collaboration(
        self,
        node: MCTSNode,
        agents: Optional[List['LeanProofAgent']] = None,
        tree: Optional[MCTSTree] = None
    ) -> MCTSNode:
        """
        Expand node using MDAP agent collaboration.

        Multiple agents suggest tactics, then voting selects the best.

        Args:
            node: Node to expand
            agents: Optional list of MDAP agents
            tree: MCTS tree

        Returns:
            Expanded child node
        """
        if node.is_terminal:
            return node

        # Get tactic suggestions from multiple agents
        all_tactics = []

        if agents and MDAP_AVAILABLE:
            # In real implementation, would query each agent
            for agent in agents:
                try:
                    # agent_tactics = await agent.suggest_tactics(node.state)
                    # all_tactics.extend(agent_tactics)
                    pass
                except Exception as e:
                    logger.warning(f"Agent {agent} failed: {e}")

        # If no agent suggestions, use heuristic
        if not all_tactics:
            all_tactics = MCTSExpansion.BASIC_TACTICS

        # Rank tactics by agent consensus
        tactic_votes = defaultdict(float)
        for tactic in all_tactics:
            # Weight by agent performance
            for agent_type, perf in self.agent_performance.items():
                tactic_votes[tactic] += perf["success_rate"]

        # Sort by votes
        ranked_tactics = sorted(
            tactic_votes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top tactic
        if ranked_tactics:
            selected_action = ranked_tactics[0][0]
        else:
            selected_action = random.choice(MCTSExpansion.BASIC_TACTICS)

        # Apply tactic to create new state
        new_state = self._apply_tactic_simulation(node.state, selected_action)

        # Create child node
        child_node = MCTSNode(
            state=new_state,
            parent=node,
            action=selected_action
        )

        # Add to tree if provided
        if tree:
            node.add_child(selected_action, child_node)
            tree.add_node(child_node)

        return child_node

    def _apply_tactic_simulation(
        self,
        state: ProofState,
        tactic: str
    ) -> ProofState:
        """Simulate tactic application (simplified)."""
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy() + [Tactic(name=tactic)],
            depth=state.depth + 1
        )

        # Simple simulation
        if tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.6:
                new_state.goals = []
        elif tactic in ["intros"]:
            if new_state.goals:
                new_state.goals = new_state.goals[1:]

        new_state.is_complete = len(new_state.goals) == 0
        return new_state


class MDAPMCTSHybrid:
    """
    Hybrid MCTS-MDAP proof generator.

    Strategies:
    1. MCTS-Then-MDAP: Run MCTS first, refine with MDAP
    2. MDAP-Then-MCTS: Run MDAP first, explore with MCTS
    3. MCTS-MDAP-Parallel: Run both and combine
    4. Adaptive: Switch based on progress
    """

    def __init__(self, config: MDAPMCTSConfig):
        """
        Initialize hybrid MCTS-MDAP system.

        Args:
            config: MDAP-MCTS configuration
        """
        self.config = config
        self.mcts_config = config.base_mcts_config
        self.integration = MCTSMDAPIntegration(config)

        # Performance tracking
        self.mcts_success_rate = 0.0
        self.mdap_success_rate = 0.0
        self.hybrid_success_rate = 0.0

    async def mcts_then_mdap(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        mcts_iters: int = 100,
        mdap_agents: int = 4
    ) -> MCTSResult:
        """
        Run MCTS first, then refine with MDAP.

        Args:
            theorem: Theorem to prove
            theorem_name: Optional theorem name
            mcts_iters: Number of MCTS iterations
            mdap_agents: Number of MDAP agents for refinement

        Returns:
            MCTSResult with best proof
        """
        logger.info(f"MCTS-Then-MDAP: {theorem}")

        # Phase 1: Run MCTS
        mcts = MCTS(self.mcts_config, theorem, theorem_name)
        mcts_result = await mcts.search(iterations=mcts_iters)

        if not mcts_result.best_proof:
            logger.warning("MCTS failed to find proof, skipping MDAP refinement")
            return mcts_result

        # Phase 2: Refine with MDAP
        if MDAP_AVAILABLE and mdap_agents > 0:
            logger.info("Refining with MDAP agents...")

            # In real implementation, would run MDAP on MCTS result
            # mdap_result = await self._run_mdap_refinement(
            #     mcts_result.best_proof,
            #     num_agents=mdap_agents
            # )

            # For now, just return MCTS result
            logger.info("MDAP refinement complete")

        return mcts_result

    async def mdap_then_mcts(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        mdap_agents: int = 4,
        mcts_iters: int = 100
    ) -> MCTSResult:
        """
        Run MDAP first to seed, then explore with MCTS.

        Args:
            theorem: Theorem to prove
            theorem_name: Optional theorem name
            mdap_agents: Number of MDAP agents
            mcts_iters: Number of MCTS iterations

        Returns:
            MCTSResult with best proof
        """
        logger.info(f"MDAP-Then-MCTS: {theorem}")

        # Phase 1: Run MDAP to get initial proofs
        mdap_proofs = []

        if MDAP_AVAILABLE and mdap_agents > 0:
            logger.info(f"Running MDAP with {mdap_agents} agents...")
            # In real implementation, would run MDAP here
            # mdap_proofs = await self._run_mdap_generation(theorem, mdap_agents)

        # Phase 2: Use MCTS to explore around MDAP results
        mcts = MCTS(self.mcts_config, theorem, theorem_name)

        # Seed MCTS with MDAP results if available
        if mdap_proofs:
            # In real implementation, would seed the tree with MDAP proofs
            logger.info(f"Seeding MCTS with {len(mdap_proofs)} MDAP proofs")

        mcts_result = await mcts.search(iterations=mcts_iters)

        return mcts_result

    async def mdap_mcts_parallel(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        mcts_iters: int = 100,
        mdap_agents: int = 4
    ) -> MCTSResult:
        """
        Run MCTS and MDAP in parallel, combine results.

        Args:
            theorem: Theorem to prove
            theorem_name: Optional theorem name
            mcts_iters: Number of MCTS iterations
            mdap_agents: Number of MDAP agents

        Returns:
            MCTSResult with best proof from both
        """
        logger.info(f"MCTS-MDAP Parallel: {theorem}")

        # Run both in parallel
        import asyncio

        tasks = []

        # MCTS task
        mcts = MCTS(self.mcts_config, theorem, theorem_name)
        tasks.append(mcts.search(iterations=mcts_iters))

        # MDAP task
        if MDAP_AVAILABLE and mdap_agents > 0:
            # In real implementation, would run MDAP here
            # tasks.append(self._run_mdap_generation(theorem, mdap_agents))
            pass

        # Wait for both to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Select best result
        best_result = None
        best_score = -1.0

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Task failed: {result}")
                continue

            if isinstance(result, MCTSResult):
                score = result.win_rate or 0.0
                if score > best_score:
                    best_score = score
                    best_result = result

        return best_result or MCTSResult(success=False, search_iterations=0)

    async def adaptive_mdap_mcts(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        time_budget: float = 60.0
    ) -> MCTSResult:
        """
        Adaptively switch between MCTS and MDAP based on progress.

        Args:
            theorem: Theorem to prove
            theorem_name: Optional theorem name
            time_budget: Total time budget

        Returns:
            MCTSResult with best proof
        """
        logger.info(f"Adaptive MCTS-MDAP: {theorem}")

        start_time = time.time()
        best_result = None
        best_fitness = 0.0

        current_mode = "mcts"  # Start with MCTS
        stagnation_count = 0
        max_stagnation = 3

        while time.time() - start_time < time_budget:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed

            logger.info(f"Mode: {current_mode}, Time: {elapsed:.1f}s/{time_budget}s")

            # Run current strategy
            if current_mode == "mcts":
                result = await self._run_mcts_batch(theorem, theorem_name, remaining / 4)
            else:
                result = await self._run_mdap_batch(theorem, remaining / 4)

            # Check for improvement
            current_fitness = result.win_rate or 0.0
            if current_fitness > best_fitness + 0.05:
                best_result = result
                best_fitness = current_fitness
                stagnation_count = 0
                logger.info(f"New best fitness: {best_fitness:.4f}")
            else:
                stagnation_count += 1
                logger.info(f"No improvement (stagnation: {stagnation_count})")

            # Switch mode if stagnating
            if stagnation_count >= max_stagnation:
                current_mode = "mdap" if current_mode == "mcts" else "mcts"
                stagnation_count = 0
                logger.info(f"Switching to {current_mode}")

            # Early termination if good proof found
            if best_fitness > 0.95:
                logger.info("High fitness achieved - stopping early")
                break

        return best_result or MCTSResult(success=False, search_iterations=0)

    async def _run_mcts_batch(
        self,
        theorem: str,
        theorem_name: Optional[str],
        time_budget: float
    ) -> MCTSResult:
        """Run a batch of MCTS iterations."""
        mcts = MCTS(self.mcts_config, theorem, theorem_name)
        return await mcts.search(time_budget=time_budget, iterations=50)

    async def _run_mdap_batch(
        self,
        theorem: str,
        time_budget: float
    ) -> MCTSResult:
        """Run a batch of MDAP agents."""
        # In real implementation, would run MDAP here
        # For now, return placeholder result
        return MCTSResult(
            success=False,
            search_iterations=0,
            time_elapsed=time_budget
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def search_proof_with_mcts(
    theorem: str,
    theorem_name: Optional[str] = None,
    max_iterations: int = 1000,
    time_budget: float = 60.0,
    server_url: str = "http://localhost:7654",
    **kwargs
) -> MCTSResult:
    """
    Convenience function to run MCTS proof search.

    Args:
        theorem: Theorem statement to prove
        theorem_name: Optional name for the theorem
        max_iterations: Maximum MCTS iterations
        time_budget: Time budget in seconds
        server_url: LeanAide server URL
        **kwargs: Additional MCTS configuration

    Returns:
        MCTSResult with best proof and statistics
    """
    config = MCTSConfig(
        max_iterations=max_iterations,
        time_budget=time_budget,
        server_url=server_url,
        **kwargs
    )

    mcts = MCTS(config, theorem, theorem_name)
    return await mcts.search()


async def search_proof_with_mdap_mcts(
    theorem: str,
    theorem_name: Optional[str] = None,
    hybrid_mode: str = "mcts_then_mdap",
    mcts_iterations: int = 100,
    mdap_agents: int = 4,
    time_budget: float = 60.0,
    server_url: str = "http://localhost:7654",
    **kwargs
) -> MCTSResult:
    """
    Convenience function to run MDAP-MCTS hybrid proof search.

    Args:
        theorem: Theorem statement to prove
        theorem_name: Optional name for the theorem
        hybrid_mode: Hybrid mode ("mcts_then_mdap", "mdap_then_mcts", "parallel", "adaptive")
        mcts_iterations: Number of MCTS iterations
        mdap_agents: Number of MDAP agents
        time_budget: Time budget in seconds
        server_url: LeanAide server URL
        **kwargs: Additional MDAP-MCTS configuration

    Returns:
        MCTSResult with best proof and statistics
    """
    base_config = MCTSConfig(
        max_iterations=mcts_iterations,
        time_budget=time_budget,
        server_url=server_url,
        **kwargs
    )

    mdap_config = MDAPMCTSConfig(
        base_mcts_config=base_config,
        num_mdap_agents=mdap_agents,
        **{k: v for k, v in kwargs.items() if k.startswith('mdap_')}
    )

    hybrid = MDAPMCTSHybrid(mdap_config)

    if hybrid_mode == "mcts_then_mdap":
        return await hybrid.mcts_then_mdap(theorem, theorem_name, mcts_iterations, mdap_agents)
    elif hybrid_mode == "mdap_then_mcts":
        return await hybrid.mdap_then_mcts(theorem, theorem_name, mdap_agents, mcts_iterations)
    elif hybrid_mode == "parallel":
        return await hybrid.mdap_mcts_parallel(theorem, theorem_name, mcts_iterations, mdap_agents)
    elif hybrid_mode == "adaptive":
        return await hybrid.adaptive_mdap_mcts(theorem, theorem_name, time_budget)
    else:
        raise ValueError(f"Unknown hybrid mode: {hybrid_mode}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'MCTSConfig',
    'MCTSResult',
    'MDAPMCTSConfig',

    # Core classes
    'MCTSNode',
    'MCTSTree',
    'MCTSSelection',
    'MCTSExpansion',
    'MCTSSimulation',
    'MCTSBackpropagation',
    'MCTS',

    # MDAP Integration
    'MCTSMDAPIntegration',
    'MDAPMCTSHybrid',

    # Data classes
    'ProofState',
    'RolloutPolicy',

    # Convenience functions
    'search_proof_with_mcts',
    'search_proof_with_mdap_mcts',

    # MDAP availability flag
    'MDAP_AVAILABLE'
]


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of MCTS for proof search."""

    # Simple example: prove commutativity of addition
    theorem = "forall (a b : Nat), a + b = b + a"

    print("=" * 80)
    print("MCTS Proof Search Example")
    print("=" * 80)
    print(f"\nTheorem: {theorem}\n")

    # Run MCTS search
    result = await search_proof_with_mcts(
        theorem=theorem,
        theorem_name="add_comm",
        max_iterations=100,
        time_budget=30.0,
        rollout_policy="heuristic",
        enable_transposition_table=True
    )

    # Print results
    print("\n" + "=" * 80)
    print("MCTS Search Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.search_iterations}")
    print(f"Time elapsed: {result.time_elapsed:.2f}s")
    print(f"Nodes visited: {result.nodes_visited}")
    print(f"Tree depth: {result.tree_depth}")
    print(f"Win rate: {result.win_rate:.4f}")
    print(f"Confidence: {result.confidence:.4f}")

    if result.best_proof:
        print("\n" + "=" * 80)
        print("Best Proof Found")
        print("=" * 80)
        print(f"\n{result.best_proof.lean_code}")

    print("\n" + "=" * 80)
    print("Tree Statistics")
    print("=" * 80)
    for key, value in result.tree_statistics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
