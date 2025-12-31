"""
LeanAide MCTS Strategy Library

A comprehensive library of MCTS-specific strategies for Lean 4 proof search.
This library extends the base strategy system with specialized Monte Carlo Tree Search
strategies for automated proof generation.

Features:
- Rollout policies: Random, Heuristic, and Learned
- Selection strategies: UCT, Adaptive UCT, Thompson Sampling
- Expansion strategies: Standard, Progressive Widening, Tree Policy
- Backpropagation strategies: Standard, AMAF/RAVE
- Domain-specific strategies: Induction, Algebraic, Logical
- Strategy factory for composition
- Performance tracking and analytics

Author: LeanAide MCTS System
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Union, Callable, Any, Set
)
from enum import Enum
import json
import random
import math
import logging
import time
from collections import defaultdict

# Import base strategy library
try:
    from leanaide_strategies import (
        LeanTacticLibrary,
        TacticMetadata,
        TacticCategory,
        ProofContext,
        ProofDifficulty
    )
except ImportError:
    logging.warning("Base strategy library not available - using standalone mode")
    LeanTacticLibrary = None

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class RolloutPolicyType(Enum):
    """Types of rollout policies"""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    LEARNED = "learned"


class SelectionStrategyType(Enum):
    """Types of selection strategies"""
    UCT = "uct"
    ADAPTIVE_UCT = "adaptive_uct"
    THOMPSON_SAMPLING = "thompson_sampling"


class ExpansionStrategyType(Enum):
    """Types of expansion strategies"""
    STANDARD = "standard"
    PROGRESSIVE_WIDENING = "progressive_widening"
    TREE_POLICY = "tree_policy"


class BackpropagationStrategyType(Enum):
    """Types of backpropagation strategies"""
    STANDARD = "standard"
    AMAF = "amaf"  # All-Moves-As-First


class DomainType(Enum):
    """Mathematical domains for specialized strategies"""
    INDUCTION = "induction"
    ALGEBRAIC = "algebraic"
    LOGICAL = "logical"
    ANALYSIS = "analysis"
    COMBINATORICS = "combinatorics"
    GENERAL = "general"


@dataclass
class MCTSNode:
    """Represents a node in the MCTS search tree"""
    state: Dict[str, Any]  # Proof state
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    action: Optional[str] = None  # Tactic that led to this node
    visits: int = 0
    value: float = 0.0  # Cumulative value
    mean_value: float = 0.0  # Average value
    untried_actions: List[str] = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False
    is_solved: bool = False

    # AMAF statistics
    amaf_visits: Dict[str, int] = field(default_factory=dict)
    amaf_values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.parent:
            self.depth = self.parent.depth + 1

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> Optional['MCTSNode']:
        """Get best child using UCT"""
        if not self.children:
            return None

        def uct(node: MCTSNode) -> float:
            if node.visits == 0:
                return float('inf')
            exploitation = node.mean_value
            exploration = c_param * math.sqrt(math.log(self.visits) / node.visits)
            return exploitation + exploration

        return max(self.children, key=uct)

    def update(self, reward: float) -> None:
        """Update node statistics with new reward"""
        self.visits += 1
        self.value += reward
        self.mean_value = self.value / self.visits


@dataclass
class MCTSSearchResult:
    """Result of an MCTS search"""
    best_proof: Optional[str] = None
    tactics_sequence: List[str] = field(default_factory=list)
    search_time: float = 0.0
    nodes_visited: int = 0
    tree_depth: int = 0
    success: bool = False
    value: float = 0.0
    strategy_used: Optional[str] = None


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    success_rate: float = 0.0
    avg_search_time: float = 0.0
    avg_tree_depth: float = 0.0
    avg_nodes_visited: float = 0.0
    proof_quality_score: float = 0.0
    total_uses: int = 0
    last_used: float = 0.0


# ============================================================================
# Rollout Policies
# ============================================================================

class RolloutPolicy(ABC):
    """Abstract base class for rollout policies"""

    def __init__(self, name: str):
        self.name = name
        self.performance = StrategyPerformance()

    @abstractmethod
    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """Select a tactic during rollout"""
        pass

    @abstractmethod
    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """Perform a rollout from the given state"""
        pass

    def record_performance(self, success: bool, time_taken: float, depth: int, nodes: int) -> None:
        """Record performance metrics"""
        self.performance.total_uses += 1
        self.performance.last_used = time.time()

        # Update running averages
        alpha = 1.0 / self.performance.total_uses
        self.performance.success_rate = (1 - alpha) * self.performance.success_rate + alpha * (1.0 if success else 0.0)
        self.performance.avg_search_time = (1 - alpha) * self.performance.avg_search_time + alpha * time_taken
        self.performance.avg_tree_depth = (1 - alpha) * self.performance.avg_tree_depth + alpha * depth
        self.performance.avg_nodes_visited = (1 - alpha) * self.performance.avg_nodes_visited + alpha * nodes


class RandomRolloutPolicy(RolloutPolicy):
    """
    Random rollout policy: select tactics uniformly at random.

    Fast but provides low-quality estimates. Useful as a baseline
    and for very broad exploration.
    """

    def __init__(self):
        super().__init__(name="random_rollout")

    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """Select a random tactic"""
        if not tactics:
            return "sorry"
        return random.choice(tactics)

    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """
        Perform a random rollout.

        Returns:
            Estimated value of the state (0.0 to 1.0)
        """
        depth = 0
        current_value = 0.5  # Neutral starting value

        while depth < max_depth and not state.get("is_solved", False):
            # Get available tactics
            available_tactics = state.get("available_tactics", [])
            if not available_tactics:
                break

            # Select random tactic
            tactic = self.select_tactic(available_tactics, state)

            # Simulate tactic application
            if state.get("is_terminal", False):
                # Terminal state: check if solved
                current_value = 1.0 if state.get("is_solved", False) else 0.0
                break

            depth += 1

            # Heuristic: prefer constructive tactics
            if any(t in tactic for t in ["intro", "existsi", "use", "constructor"]):
                current_value += 0.1

        return min(1.0, max(0.0, current_value))


class HeuristicRolloutPolicy(RolloutPolicy):
    """
    Heuristic rollout policy: use domain knowledge to guide tactic selection.

    Heuristics:
    - Prefer constructive tactics (intro, existsi, constructor)
    - Prefer simplification tactics (simp, aesop, norm_num)
    - Avoid case analysis on large types
    - Prefer domain-specific tactics based on goal
    - Prefer tactics with high historical success rates
    """

    def __init__(self, tactic_library: Optional[LeanTacticLibrary] = None):
        super().__init__(name="heuristic_rollout")
        self.tactic_library = tactic_library

        # Tactic preferences by category
        self.tactic_preferences = {
            TacticCategory.CONSTRUCTIVE: 1.5,
            TacticCategory.SIMPLIFICATION: 1.3,
            TacticCategory.STRUCTURAL: 1.2,
            TacticCategory.ALGEBRAIC: 1.2,
            TacticCategory.ARITHMETIC: 1.1,
            TacticCategory.LOGICAL: 1.0,
            TacticCategory.INDUCTIVE: 0.9,  # Use sparingly in rollouts
            TacticCategory.ADVANCED: 0.8,
        }

        # Bonus tactics (always get extra consideration)
        self.bonus_tactics = {
            "intro": 1.3,
            "simp": 1.2,
            "aesop": 1.2,
            "constructor": 1.3,
            "existsi": 1.4,
            "use": 1.3,
            "assumption": 1.5,
            "rfl": 1.4,
            "norm_num": 1.3,
        }

    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """Select a tactic using heuristics"""
        if not tactics:
            return "sorry"

        # Score each tactic
        scored_tactics = []
        for tactic in tactics:
            score = self.score_tactic(tactic, state)
            scored_tactics.append((tactic, score))

        # Sort by score
        scored_tactics.sort(key=lambda x: x[1], reverse=True)

        # Use weighted random selection from top k
        top_k = min(5, len(scored_tactics))
        top_tactics = scored_tactics[:top_k]

        # Weighted random selection
        weights = [score for _, score in top_tactics]
        total_weight = sum(weights)

        if total_weight == 0:
            return random.choice(tactics)

        rand_val = random.uniform(0, total_weight)
        cumulative = 0.0
        for tactic, weight in top_tactics:
            cumulative += weight
            if rand_val <= cumulative:
                return tactic

        return top_tactics[0][0]

    def score_tactic(self, tactic: str, state: Dict[str, Any]) -> float:
        """Score a tactic based on heuristics"""
        score = 1.0  # Base score

        # Extract tactic name (without arguments)
        tactic_name = tactic.split()[0] if tactic else ""

        # Bonus tactics
        if tactic_name in self.bonus_tactics:
            score *= self.bonus_tactics[tactic_name]

        # Check tactic category if library is available
        if self.tactic_library:
            metadata = self.tactic_library.get_tactic(tactic_name)
            if metadata:
                preference = self.tactic_preferences.get(metadata.category, 1.0)
                score *= preference

                # Factor in success rate
                score *= (0.5 + metadata.success_rate)

        # Domain-specific bonuses
        goal = state.get("goal", "").lower()
        domain = state.get("domain", "general")

        # Induction proofs
        if domain == "induction" or "induction" in goal:
            if tactic_name in ["induction", "cases"]:
                score *= 1.5

        # Algebraic proofs
        if domain == "algebraic" or any(op in goal for op in ["=", "+", "*"]):
            if tactic_name in ["ring", "simp", "norm_num", "linarith"]:
                score *= 1.4

        # Logical proofs
        if domain == "logical" or any(log in goal for log in ["forall", "exists", "implies"]):
            if tactic_name in ["intro", "apply", "exact", "constructor"]:
                score *= 1.3

        # Penalize case analysis on large types
        if tactic_name == "cases":
            context_size = len(state.get("context", []))
            if context_size > 10:
                score *= 0.5

        # Prefer safe tactics
        if self.tactic_library:
            metadata = self.tactic_library.get_tactic(tactic_name)
            if metadata and metadata.is_safe:
                score *= 1.1

        return score

    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """
        Perform a heuristic-guided rollout.

        Returns:
            Estimated value of the state (0.0 to 1.0)
        """
        depth = 0
        current_value = 0.5

        while depth < max_depth and not state.get("is_solved", False):
            available_tactics = state.get("available_tactics", [])
            if not available_tactics:
                break

            # Select tactic using heuristics
            tactic = self.select_tactic(available_tactics, state)

            # Update value based on tactic choice
            if state.get("is_terminal", False):
                current_value = 1.0 if state.get("is_solved", False) else 0.0
                break

            # Bonus for good tactical choices
            if any(t in tactic for t in ["intro", "simp", "constructor", "assumption"]):
                current_value += 0.15
            elif "cases" in tactic:
                current_value -= 0.05  # Slight penalty for case analysis

            depth += 1

        return min(1.0, max(0.0, current_value))


class LearnedRolloutPolicy(RolloutPolicy):
    """
    Learned rollout policy: use a trained model for tactic selection.

    This is an advanced/optional feature that requires:
    - A trained model (e.g., neural network)
    - Model weights file
    - Feature extraction from proof states

    The policy can fall back to heuristics if the model is unavailable.
    """

    def __init__(self, model_path: Optional[str] = None, fallback_policy: Optional[RolloutPolicy] = None):
        super().__init__(name="learned_rollout")
        self.model_path = model_path
        self.model = None
        self.fallback_policy = fallback_policy or HeuristicRolloutPolicy()

        # Try to load model
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load a trained model from file"""
        try:
            # Placeholder for actual model loading
            # In practice, this would load a neural network or other ML model
            logger.info(f"Loading learned rollout model from {model_path}")
            # self.model = load_model(model_path)
            self.model = "mock_model"  # Placeholder
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using fallback policy.")
            self.model = None

    def extract_features(self, state: Dict[str, Any]) -> List[float]:
        """Extract features from proof state for model input"""
        features = []

        # Goal features
        goal = state.get("goal", "")
        features.append(len(goal))
        features.append(goal.count("forall"))
        features.append(goal.count("exists"))
        features.append(goal.count("="))
        features.append(goal.count("->"))

        # Context features
        context = state.get("context", [])
        features.append(len(context))

        # Domain features
        domain = state.get("domain", "general")
        domain_features = {
            "induction": [1, 0, 0, 0, 0],
            "algebraic": [0, 1, 0, 0, 0],
            "logical": [0, 0, 1, 0, 0],
            "analysis": [0, 0, 0, 1, 0],
            "general": [0, 0, 0, 0, 1],
        }
        features.extend(domain_features.get(domain, domain_features["general"]))

        return features

    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """Select a tactic using the learned model"""
        if not self.model or not tactics:
            return self.fallback_policy.select_tactic(tactics, state)

        try:
            # Extract features
            features = self.extract_features(state)

            # Get model predictions
            # tactic_scores = self.model.predict(features, tactics)
            # Placeholder: return heuristic selection
            return self.fallback_policy.select_tactic(tactics, state)

        except Exception as e:
            logger.warning(f"Model prediction failed: {e}. Using fallback.")
            return self.fallback_policy.select_tactic(tactics, state)

    def predict_value(self, state: Dict[str, Any]) -> float:
        """Predict the value of a state using the learned model"""
        if not self.model:
            return 0.5

        try:
            features = self.extract_features(state)
            # value = self.model.predict_value(features)
            # Placeholder: return neutral value
            return 0.5
        except Exception as e:
            logger.warning(f"Value prediction failed: {e}")
            return 0.5

    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """Perform a rollout using the learned policy"""
        # Use learned policy with fallback
        return self.fallback_policy.rollout(state, max_depth)


# ============================================================================
# Selection Strategies
# ============================================================================

class SelectionStrategy(ABC):
    """Abstract base class for MCTS selection strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_child(self, children: List[MCTSNode], **kwargs) -> Optional[MCTSNode]:
        """Select a child node during tree traversal"""
        pass


class UCTSelection(SelectionStrategy):
    """
    Standard UCT (Upper Confidence Bound for Trees) selection.

    Uses the classic UCT formula:
    UCT = mean_value + c * sqrt(log(parent_visits) / child_visits)

    Where c (c_param) balances exploration vs exploitation.
    Typical value: c = sqrt(2) ≈ 1.414
    """

    def __init__(self, c_param: float = 1.414):
        super().__init__(name="uct")
        self.c_param = c_param

    def calculate_uct(self, node: MCTSNode, parent_visits: int) -> float:
        """Calculate UCT value for a node"""
        if node.visits == 0:
            return float('inf')

        exploitation = node.mean_value
        exploration = self.c_param * math.sqrt(math.log(parent_visits) / node.visits)

        return exploitation + exploration

    def select_child(self, children: List[MCTSNode], **kwargs) -> Optional[MCTSNode]:
        """Select child using UCT"""
        if not children:
            return None

        parent_visits = sum(child.visits for child in children)

        def uct_score(node: MCTSNode) -> float:
            return self.calculate_uct(node, parent_visits)

        return max(children, key=uct_score)


class AdaptiveUCTSelection(SelectionStrategy):
    """
    Adaptive UCT selection with dynamic c_param based on tree depth.

    Adjusts exploration parameter based on:
    - Tree depth (deeper nodes get more exploration)
    - Visit count (visited nodes get more exploitation)
    - Node value variance (high variance gets more exploration)

    Formula:
    c_adaptive = base_c * (1 + depth_factor) * (1 + variance_factor)
    """

    def __init__(self, base_c: float = 1.414):
        super().__init__(name="adaptive_uct")
        self.base_c = base_c
        self.depth_factor = 0.1
        self.variance_factor = 0.5

    def adaptive_c_param(self, node: MCTSNode, depth: int) -> float:
        """Calculate adaptive c parameter"""
        # Depth-based adjustment
        depth_multiplier = 1.0 + (depth * self.depth_factor)

        # Variance-based adjustment (if we have enough visits)
        variance_multiplier = 1.0
        if node.visits > 1:
            # Estimate variance from value range
            value_range = abs(node.mean_value - 0.5) * 2  # Higher if further from 0.5
            variance_multiplier = 1.0 + (value_range * self.variance_factor)

        return self.base_c * depth_multiplier * variance_multiplier

    def calculate_uct(self, node: MCTSNode, parent_visits: int, c_param: float) -> float:
        """Calculate UCT with adaptive c parameter"""
        if node.visits == 0:
            return float('inf')

        exploitation = node.mean_value
        exploration = c_param * math.sqrt(math.log(parent_visits) / node.visits)

        return exploitation + exploration

    def select_child(self, children: List[MCTSNode], **kwargs) -> Optional[MCTSNode]:
        """Select child using adaptive UCT"""
        if not children:
            return None

        depth = kwargs.get('depth', 0)
        parent_visits = sum(child.visits for child in children)

        def adaptive_uct_score(node: MCTSNode) -> float:
            c_param = self.adaptive_c_param(node, depth)
            return self.calculate_uct(node, parent_visits, c_param)

        return max(children, key=adaptive_uct_score)


class ThompsonSamplingSelection(SelectionStrategy):
    """
    Thompson Sampling (Bayesian) selection.

    Models each child's value as a Beta distribution and samples from it.
    Naturally balances exploration and exploitation through Bayesian inference.

    Beta distribution parameters:
    - alpha: number of successes + 1
    - beta: number of failures + 1

    Advantages:
    - Theoretically sound Bayesian approach
    - Automatically adapts based on evidence
    - Good for non-stationary reward distributions
    """

    def __init__(self):
        super().__init__(name="thompson_sampling")

    def sample_value(self, node: MCTSNode) -> float:
        """Sample a value from the node's Beta distribution"""
        # Convert mean value and visits to Beta parameters
        # mean = alpha / (alpha + beta)
        # alpha + beta = visits (pseudocounts)

        if node.visits == 0:
            # Unvisited node: maximum variance sample
            return random.betavariate(1, 1)

        # Use soft counts for Beta parameters
        # Ensure alpha and beta are always > 0
        alpha = max(0.1, 1 + node.mean_value * node.visits)
        beta = max(0.1, 1 + (1 - node.mean_value) * node.visits)

        return random.betavariate(alpha, beta)

    def select_child(self, children: List[MCTSNode], **kwargs) -> Optional[MCTSNode]:
        """Select child using Thompson sampling"""
        if not children:
            return None

        # Sample from each child's distribution
        def thompson_score(node: MCTSNode) -> float:
            return self.sample_value(node)

        return max(children, key=thompson_score)


# ============================================================================
# Expansion Strategies
# ============================================================================

class ExpansionStrategy(ABC):
    """Abstract base class for MCTS expansion strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def expand(self, node: MCTSNode, **kwargs) -> Optional[MCTSNode]:
        """Expand a node and return the new child"""
        pass


class StandardExpansion(ExpansionStrategy):
    """
    Standard MCTS expansion: expand single untried action.

    Selects one untried action and creates a child node for it.
    This is the most common expansion strategy.
    """

    def __init__(self):
        super().__init__(name="standard_expansion")

    def expand(self, node: MCTSNode, **kwargs) -> Optional[MCTSNode]:
        """Expand node with one untried action"""
        if node.is_fully_expanded() or node.is_terminal:
            return None

        # Select random untried action
        if not node.untried_actions:
            return None

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Create child node
        child_state = self._apply_action(node.state, action)

        child = MCTSNode(
            state=child_state,
            parent=node,
            action=action,
            untried_actions=child_state.get("available_tactics", []),
            depth=node.depth + 1
        )

        node.children.append(child)
        return child

    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply a tactic to the state (simplified)"""
        # Placeholder: in practice, this would call Lean to apply the tactic
        new_state = state.copy()
        new_state["last_tactic"] = action
        new_state["available_tactics"] = state.get("available_tactics", [])

        # Simulate state change
        if action == "sorry":
            new_state["is_terminal"] = True
            new_state["is_solved"] = False

        return new_state


class ProgressiveWidening(ExpansionStrategy):
    """
    Progressive widening expansion strategy.

    Limits the number of children based on visit count to prevent
    premature expansion of the action space.

    Expands node i when:
    visits >= C * (num_children)^D

    Where:
    - C: widening parameter (typically 1-10)
    - D: widening exponent (typically 0.5-1.0)

    Reference: Progressive Widening for MCTS in large action spaces.
    """

    def __init__(self, widening_param: float = 3.0, widening_exponent: float = 0.5):
        super().__init__(name="progressive_widening")
        self.widening_param = widening_param
        self.widening_exponent = widening_exponent

    def should_expand_child(self, node: MCTSNode, child_count: int) -> bool:
        """Check if we should expand another child"""
        threshold = self.widening_param * (child_count ** self.widening_exponent)
        return node.visits >= threshold

    def expand(self, node: MCTSNode, **kwargs) -> Optional[MCTSNode]:
        """Expand with progressive widening"""
        if node.is_terminal:
            return None

        num_children = len(node.children)

        # Check if we should expand another child
        if not self.should_expand_child(node, num_children):
            return None

        # Check if there are untried actions
        if not node.untried_actions:
            return None

        # Select and apply action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        child_state = self._apply_action(node.state, action)

        child = MCTSNode(
            state=child_state,
            parent=node,
            action=action,
            untried_actions=child_state.get("available_tactics", []),
            depth=node.depth + 1
        )

        node.children.append(child)
        return child

    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply a tactic to the state"""
        new_state = state.copy()
        new_state["last_tactic"] = action
        new_state["available_tactics"] = state.get("available_tactics", [])
        return new_state


class TreePolicyExpansion(ExpansionStrategy):
    """
    Tree policy expansion: expand most promising untried action.

    Uses heuristic evaluation to select which action to expand first,
    rather than choosing randomly.

    Can use:
    - Domain knowledge
    - Tactic success rates
    - Goal characteristics
    """

    def __init__(self, heuristic_fn: Optional[Callable] = None):
        super().__init__(name="tree_policy_expansion")
        self.heuristic_fn = heuristic_fn or self._default_heuristic

    def evaluate_child_potential(self, node: MCTSNode, action: str) -> float:
        """Evaluate the potential of an untried action"""
        # Use heuristic function
        return self.heuristic_fn(node.state, action)

    def _default_heuristic(self, state: Dict[str, Any], action: str) -> float:
        """Default heuristic for action evaluation"""
        score = 1.0

        # Prefer safe tactics
        safe_tactics = ["simp", "norm_num", "assumption", "rfl", "trivial"]
        if any(t in action for t in safe_tactics):
            score *= 1.5

        # Prefer constructive tactics
        constructive = ["intro", "constructor", "existsi", "use"]
        if any(t in action for t in constructive):
            score *= 1.3

        # Penalize complex tactics early
        if state.get("depth", 0) < 3:
            if action in ["induction", "cases", "by_contradiction"]:
                score *= 0.7

        return score + random.uniform(0, 0.1)  # Add noise for exploration

    def expand(self, node: MCTSNode, **kwargs) -> Optional[MCTSNode]:
        """Expand most promising action"""
        if node.is_fully_expanded() or node.is_terminal:
            return None

        if not node.untried_actions:
            return None

        # Score all untried actions
        scored_actions = [
            (action, self.evaluate_child_potential(node, action))
            for action in node.untried_actions
        ]

        # Select best action
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        best_action = scored_actions[0][0]

        node.untried_actions.remove(best_action)

        # Create child
        child_state = self._apply_action(node.state, best_action)

        child = MCTSNode(
            state=child_state,
            parent=node,
            action=best_action,
            untried_actions=child_state.get("available_tactics", []),
            depth=node.depth + 1
        )

        node.children.append(child)
        return child

    def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply a tactic to the state"""
        new_state = state.copy()
        new_state["last_tactic"] = action
        new_state["available_tactics"] = state.get("available_tactics", [])
        return new_state


# ============================================================================
# Backpropagation Strategies
# ============================================================================

class BackpropagationStrategy(ABC):
    """Abstract base class for backpropagation strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def backpropagate(self, node: MCTSNode, reward: float, **kwargs) -> None:
        """Backpropagate reward from node to root"""
        pass


class StandardBackpropagation(BackpropagationStrategy):
    """
    Standard MCTS backpropagation.

    Updates node statistics along the path from leaf to root.
    Each node's value is updated with the reward.
    """

    def __init__(self):
        super().__init__(name="standard_backpropagation")

    def backpropagate(self, node: MCTSNode, reward: float, **kwargs) -> None:
        """Backpropagate reward up the tree"""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent


class AMAFBackpropagation(BackpropagationStrategy):
    """
    All-Moves-As-First (AMAF/RAVE) backpropagation.

    Updates not just the visited nodes, but also sibling nodes that
    have the same action available. This accelerates learning by
    sharing statistics across the tree.

    Also known as RAVE (Rapid Action Value Estimation).

    Formula combines tree value with AMAF value:
    Q_combined = (1 - beta) * Q_tree + beta * Q_AMAF

    Where beta decreases as visits increase:
    beta = sqrt(C / (3 * visits + C))
    """

    def __init__(self, amaf_weight: float = 1000.0):
        super().__init__(name="amaf_backpropagation")
        self.amaf_weight = amaf_weight

    def backpropagate(self, node: MCTSNode, reward: float, **kwargs) -> None:
        """Backpropagate with AMAF updates"""
        action = kwargs.get('action')
        visited_nodes = kwargs.get('visited_nodes', [])

        # Standard backpropagation
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

        # AMAF updates
        if action:
            self._update_amaf_stats(node, action, reward, visited_nodes)

    def _update_amaf_stats(
        self,
        node: MCTSNode,
        action: str,
        reward: float,
        visited_nodes: List[MCTSNode]
    ) -> None:
        """Update AMAF statistics for nodes with the same action"""
        # Walk up the tree and update AMAF stats
        current = node.parent
        while current is not None:
            # Update AMAF for this action if it was available
            if action in current.untried_actions or any(c.action == action for c in current.children):
                if action not in current.amaf_visits:
                    current.amaf_visits[action] = 0
                    current.amaf_values[action] = 0.0

                current.amaf_visits[action] += 1
                current.amaf_values[action] += reward

            current = current.parent

    def get_amaf_value(self, node: MCTSNode, action: str) -> float:
        """Get AMAF value for an action at a node"""
        if action not in node.amaf_visits or node.amaf_visits[action] == 0:
            return 0.5  # Neutral value

        return node.amaf_values[action] / node.amaf_visits[action]

    def get_combined_value(self, node: MCTSNode, action: str) -> float:
        """Get combined tree + AMAF value for an action"""
        # Find child with this action
        child = next((c for c in node.children if c.action == action), None)

        if not child:
            return self.get_amaf_value(node, action)

        # Tree value
        q_tree = child.mean_value
        tree_visits = child.visits

        # AMAF value
        q_amaf = self.get_amaf_value(node, action)
        amaf_visits = node.amaf_visits.get(action, 0)

        # Beta parameter (decreases with visits)
        beta = math.sqrt(self.amaf_weight / (3 * tree_visits + self.amaf_weight))

        # Combined value
        return (1 - beta) * q_tree + beta * q_amaf


# ============================================================================
# Domain-Specific Strategies
# ============================================================================

class DomainSpecificStrategy(ABC):
    """Abstract base class for domain-specific MCTS strategies"""

    def __init__(self, domain: DomainType, name: str):
        self.domain = domain
        self.name = name

    @abstractmethod
    def score_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Score tactics based on domain knowledge"""
        pass

    @abstractmethod
    def rollout_bias(self, state: Dict[str, Any]) -> float:
        """Provide bias value for rollouts in this domain"""
        pass


class InductionMCTS(DomainSpecificStrategy):
    """
    Specialized MCTS strategy for induction proofs.

    Biases search toward induction tactics and patterns:
    - Prefers induction on natural numbers
    - Favors base case simplification
    - Prefers inductive hypothesis application
    """

    def __init__(self):
        super().__init__(domain=DomainType.INDUCTION, name="induction_mcts")

        # Induction-favored tactics
        self.induction_tactics = {
            "induction": 2.0,
            "cases": 1.5,
            "simp": 1.3,
            "norm_num": 1.2,
            "ring": 1.2,
            "linarith": 1.1,
        }

    def score_induction_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Score tactics for induction proofs"""
        scores = {}
        available_tactics = state.get("available_tactics", [])

        for tactic in available_tactics:
            base_score = 1.0
            tactic_name = tactic.split()[0]

            # Apply induction bonus
            if tactic_name in self.induction_tactics:
                base_score *= self.induction_tactics[tactic_name]

            # Check for induction opportunities
            goal = state.get("goal", "").lower()

            # Base case indicators
            if any(kw in goal for kw in ["zero", "0", "base", "nil"]):
                if tactic_name in ["simp", "norm_num", "rfl"]:
                    base_score *= 1.5

            # Inductive step indicators
            if any(kw in goal for kw in ["succ", "n + 1", "cons", "ih"]):
                if tactic_name in ["simp", "rw", "apply"]:
                    base_score *= 1.4

            scores[tactic] = base_score

        return scores

    def prefer_induction_rollout(self, state: Dict[str, Any]) -> float:
        """Provide induction-biased value for rollouts"""
        base_value = 0.5
        goal = state.get("goal", "").lower()

        # Bonus for induction structure
        if "induction" in goal:
            base_value += 0.2

        # Bonus for natural numbers
        if any(kw in goal for kw in ["nat", "natural", "ℕ"]):
            base_value += 0.1

        # Bonus for recursive structure
        if any(kw in goal for kw in ["factorial", "fibonacci", "+", "*"]):
            base_value += 0.1

        return min(1.0, base_value)

    def score_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        return self.score_induction_tactics(state)

    def rollout_bias(self, state: Dict[str, Any]) -> float:
        return self.prefer_induction_rollout(state)


class AlgebraicMCTS(DomainSpecificStrategy):
    """
    Specialized MCTS strategy for algebraic proofs.

    Biases toward algebraic manipulation:
    - Prefers ring, simp, norm_num
    - Favors calculation tactics
    - Prefers equational reasoning
    """

    def __init__(self):
        super().__init__(domain=DomainType.ALGEBRAIC, name="algebraic_mcts")

        self.algebraic_tactics = {
            "ring": 2.0,
            "ring_nf": 1.8,
            "simp": 1.5,
            "norm_num": 1.5,
            "linarith": 1.4,
            "nlinarith": 1.3,
            "calc": 1.6,
            "ac_rfl": 1.7,
        }

    def score_algebraic_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Score tactics for algebraic proofs"""
        scores = {}
        available_tactics = state.get("available_tactics", [])

        for tactic in available_tactics:
            base_score = 1.0
            tactic_name = tactic.split()[0]

            # Apply algebraic bonus
            if tactic_name in self.algebraic_tactics:
                base_score *= self.algebraic_tactics[tactic_name]

            # Check goal characteristics
            goal = state.get("goal", "").lower()

            # Ring operations
            if any(op in goal for op in ["+", "*", "-", "/"]):
                if tactic_name in ["ring", "simp", "norm_num"]:
                    base_score *= 1.3

            # Equalities
            if "=" in goal:
                if tactic_name in ["ring", "ac_rfl", "rfl"]:
                    base_score *= 1.4

            # Inequalities
            if any(op in goal for op in ["<", ">", "≤", ">="]):
                if tactic_name in ["linarith", "nlinarith"]:
                    base_score *= 1.5

            # Complex expressions
            if goal.count("(") > 2 or goal.count("+") > 2:
                if tactic_name in ["ring", "simp"]:
                    base_score *= 1.2

            scores[tactic] = base_score

        return scores

    def prefer_calc_rollout(self, state: Dict[str, Any]) -> float:
        """Provide algebraic-biased value for rollouts"""
        base_value = 0.5
        goal = state.get("goal", "").lower()

        # Bonus for algebraic operations
        if any(op in goal for op in ["+", "*", "-", "/"]):
            base_value += 0.15

        # Bonus for equalities
        if "=" in goal:
            base_value += 0.1

        # Bonus for inequalities
        if any(op in goal for op in ["<", ">", "≤", ">="]):
            base_value += 0.1

        return min(1.0, base_value)

    def score_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        return self.score_algebraic_tactics(state)

    def rollout_bias(self, state: Dict[str, Any]) -> float:
        return self.prefer_calc_rollout(state)


class LogicalMCTS(DomainSpecificStrategy):
    """
    Specialized MCTS strategy for logical proofs.

    Biases toward logical reasoning:
    - Prefers intro, apply, exact
    - Favors constructive logic
    - Prefers structural reasoning
    """

    def __init__(self):
        super().__init__(domain=DomainType.LOGICAL, name="logical_mcts")

        self.logical_tactics = {
            "intro": 2.0,
            "intros": 1.8,
            "apply": 1.7,
            "exact": 1.7,
            "refine": 1.5,
            "constructor": 1.6,
            "existsi": 1.8,
            "use": 1.7,
            "cases": 1.4,
            "rcases": 1.3,
        }

    def score_logical_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Score tactics for logical proofs"""
        scores = {}
        available_tactics = state.get("available_tactics", [])

        for tactic in available_tactics:
            base_score = 1.0
            tactic_name = tactic.split()[0]

            # Apply logical bonus
            if tactic_name in self.logical_tactics:
                base_score *= self.logical_tactics[tactic_name]

            # Check goal characteristics
            goal = state.get("goal", "").lower()

            # Universal quantifier
            if "forall" in goal or "∀" in state.get("goal", ""):
                if tactic_name in ["intro", "intros"]:
                    base_score *= 1.5

            # Existential quantifier
            if "exists" in goal or "∃" in state.get("goal", ""):
                if tactic_name in ["existsi", "use", "apply"]:
                    base_score *= 1.6

            # Implication
            if "→" in state.get("goal", "") or "implies" in goal:
                if tactic_name in ["intro", "apply"]:
                    base_score *= 1.4

            # Conjunction
            if "∧" in state.get("goal", "") or "/\\" in goal:
                if tactic_name in ["constructor", "split"]:
                    base_score *= 1.5

            # Disjunction
            if "∨" in state.get("goal", "") or "\\/" in goal:
                if tactic_name in ["cases", "left", "right"]:
                    base_score *= 1.5

            scores[tactic] = base_score

        return scores

    def prefer_by_contradiction_rollout(self, state: Dict[str, Any]) -> float:
        """Provide logical-biased value for rollouts"""
        base_value = 0.5
        goal = state.get("goal", "").lower()

        # Bonus for logical structure
        if any(q in goal for q in ["forall", "exists", "implies"]):
            base_value += 0.15

        # Bonus for quantifiers
        if any(q in state.get("goal", "") for q in ["∀", "∃", "→", "∧", "∨"]):
            base_value += 0.15

        # Bonus for constructive structure
        if any(t in goal for t in ["intro", "apply", "constructor"]):
            base_value += 0.1

        return min(1.0, base_value)

    def score_tactics(self, state: Dict[str, Any]) -> Dict[str, float]:
        return self.score_logical_tactics(state)

    def rollout_bias(self, state: Dict[str, Any]) -> float:
        return self.prefer_by_contradiction_rollout(state)


# ============================================================================
# Strategy Factory
# ============================================================================

class MCTSStrategyFactory:
    """
    Factory for creating MCTS strategies and their components.

    Provides a unified interface for strategy creation with composition support.
    """

    @staticmethod
    def create_rollout_policy(
        policy_type: RolloutPolicyType,
        **kwargs
    ) -> RolloutPolicy:
        """
        Create a rollout policy.

        Args:
            policy_type: Type of rollout policy
            **kwargs: Policy-specific parameters

        Returns:
            RolloutPolicy instance
        """
        if policy_type == RolloutPolicyType.RANDOM:
            return RandomRolloutPolicy()

        elif policy_type == RolloutPolicyType.HEURISTIC:
            tactic_library = kwargs.get('tactic_library')
            return HeuristicRolloutPolicy(tactic_library=tactic_library)

        elif policy_type == RolloutPolicyType.LEARNED:
            model_path = kwargs.get('model_path')
            fallback = kwargs.get('fallback_policy')
            return LearnedRolloutPolicy(model_path=model_path, fallback_policy=fallback)

        else:
            raise ValueError(f"Unknown rollout policy type: {policy_type}")

    @staticmethod
    def create_selection_strategy(
        strategy_type: SelectionStrategyType,
        **kwargs
    ) -> SelectionStrategy:
        """
        Create a selection strategy.

        Args:
            strategy_type: Type of selection strategy
            **kwargs: Strategy-specific parameters

        Returns:
            SelectionStrategy instance
        """
        if strategy_type == SelectionStrategyType.UCT:
            c_param = kwargs.get('c_param', 1.414)
            return UCTSelection(c_param=c_param)

        elif strategy_type == SelectionStrategyType.ADAPTIVE_UCT:
            base_c = kwargs.get('base_c', 1.414)
            return AdaptiveUCTSelection(base_c=base_c)

        elif strategy_type == SelectionStrategyType.THOMPSON_SAMPLING:
            return ThompsonSamplingSelection()

        else:
            raise ValueError(f"Unknown selection strategy type: {strategy_type}")

    @staticmethod
    def create_expansion_strategy(
        strategy_type: ExpansionStrategyType,
        **kwargs
    ) -> ExpansionStrategy:
        """
        Create an expansion strategy.

        Args:
            strategy_type: Type of expansion strategy
            **kwargs: Strategy-specific parameters

        Returns:
            ExpansionStrategy instance
        """
        if strategy_type == ExpansionStrategyType.STANDARD:
            return StandardExpansion()

        elif strategy_type == ExpansionStrategyType.PROGRESSIVE_WIDENING:
            widening_param = kwargs.get('widening_param', 3.0)
            widening_exponent = kwargs.get('widening_exponent', 0.5)
            return ProgressiveWidening(
                widening_param=widening_param,
                widening_exponent=widening_exponent
            )

        elif strategy_type == ExpansionStrategyType.TREE_POLICY:
            heuristic_fn = kwargs.get('heuristic_fn')
            return TreePolicyExpansion(heuristic_fn=heuristic_fn)

        else:
            raise ValueError(f"Unknown expansion strategy type: {strategy_type}")

    @staticmethod
    def create_backpropagation_strategy(
        strategy_type: BackpropagationStrategyType,
        **kwargs
    ) -> BackpropagationStrategy:
        """
        Create a backpropagation strategy.

        Args:
            strategy_type: Type of backpropagation strategy
            **kwargs: Strategy-specific parameters

        Returns:
            BackpropagationStrategy instance
        """
        if strategy_type == BackpropagationStrategyType.STANDARD:
            return StandardBackpropagation()

        elif strategy_type == BackpropagationStrategyType.AMAF:
            amaf_weight = kwargs.get('amaf_weight', 1000.0)
            return AMAFBackpropagation(amaf_weight=amaf_weight)

        else:
            raise ValueError(f"Unknown backpropagation strategy type: {strategy_type}")

    @staticmethod
    def create_domain_strategy(
        domain: DomainType
    ) -> DomainSpecificStrategy:
        """
        Create a domain-specific strategy.

        Args:
            domain: Mathematical domain

        Returns:
            DomainSpecificStrategy instance
        """
        if domain == DomainType.INDUCTION:
            return InductionMCTS()
        elif domain == DomainType.ALGEBRAIC:
            return AlgebraicMCTS()
        elif domain == DomainType.LOGICAL:
            return LogicalMCTS()
        else:
            # Return general strategy (algebraic as default)
            return AlgebraicMCTS()

    @staticmethod
    def create_composite_strategy(
        rollout_policy: Union[RolloutPolicyType, RolloutPolicy],
        selection_strategy: Union[SelectionStrategyType, SelectionStrategy],
        expansion_strategy: Union[ExpansionStrategyType, ExpansionStrategy],
        backpropagation_strategy: Union[BackpropagationStrategyType, BackpropagationStrategy],
        domain_strategy: Optional[Union[DomainType, DomainSpecificStrategy]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a composite MCTS strategy from components.

        Args:
            rollout_policy: Rollout policy (type or instance)
            selection_strategy: Selection strategy (type or instance)
            expansion_strategy: Expansion strategy (type or instance)
            backpropagation_strategy: Backpropagation strategy (type or instance)
            domain_strategy: Optional domain-specific strategy
            **kwargs: Additional parameters

        Returns:
            Dictionary with all strategy components
        """
        # Create rollout policy
        if isinstance(rollout_policy, RolloutPolicyType):
            rollout = MCTSStrategyFactory.create_rollout_policy(
                rollout_policy,
                **kwargs
            )
        else:
            rollout = rollout_policy

        # Create selection strategy
        if isinstance(selection_strategy, SelectionStrategyType):
            selection = MCTSStrategyFactory.create_selection_strategy(
                selection_strategy,
                **kwargs
            )
        else:
            selection = selection_strategy

        # Create expansion strategy
        if isinstance(expansion_strategy, ExpansionStrategyType):
            expansion = MCTSStrategyFactory.create_expansion_strategy(
                expansion_strategy,
                **kwargs
            )
        else:
            expansion = expansion_strategy

        # Create backpropagation strategy
        if isinstance(backpropagation_strategy, BackpropagationStrategyType):
            backpropagation = MCTSStrategyFactory.create_backpropagation_strategy(
                backpropagation_strategy,
                **kwargs
            )
        else:
            backpropagation = backpropagation_strategy

        # Create domain strategy if specified
        domain = None
        if domain_strategy:
            if isinstance(domain_strategy, DomainType):
                domain = MCTSStrategyFactory.create_domain_strategy(domain_strategy)
            else:
                domain = domain_strategy

        return {
            'rollout_policy': rollout,
            'selection_strategy': selection,
            'expansion_strategy': expansion,
            'backpropagation_strategy': backpropagation,
            'domain_strategy': domain,
        }

    @staticmethod
    def create_preset_strategy(preset_name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a preset strategy combination.

        Available presets:
        - 'balanced': Balanced exploration/exploitation
        - 'exploratory': High exploration
        - 'exploitative': High exploitation
        - 'fast': Fast rollouts with simple policies
        - 'accurate': High-quality rollouts with heuristics
        - 'induction': Specialized for induction proofs
        - 'algebraic': Specialized for algebraic proofs
        - 'logical': Specialized for logical proofs

        Args:
            preset_name: Name of the preset
            **kwargs: Override parameters

        Returns:
            Dictionary with all strategy components
        """
        presets = {
            'balanced': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.UCT,
                'expansion_strategy': ExpansionStrategyType.STANDARD,
                'backpropagation_strategy': BackpropagationStrategyType.STANDARD,
                'c_param': 1.414,
            },
            'exploratory': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.UCT,
                'expansion_strategy': ExpansionStrategyType.PROGRESSIVE_WIDENING,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
                'c_param': 2.0,
            },
            'exploitative': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.ADAPTIVE_UCT,
                'expansion_strategy': ExpansionStrategyType.TREE_POLICY,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
                'base_c': 1.0,
            },
            'fast': {
                'rollout_policy': RolloutPolicyType.RANDOM,
                'selection_strategy': SelectionStrategyType.UCT,
                'expansion_strategy': ExpansionStrategyType.STANDARD,
                'backpropagation_strategy': BackpropagationStrategyType.STANDARD,
                'c_param': 1.414,
            },
            'accurate': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.THOMPSON_SAMPLING,
                'expansion_strategy': ExpansionStrategyType.TREE_POLICY,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
            },
            'induction': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.ADAPTIVE_UCT,
                'expansion_strategy': ExpansionStrategyType.TREE_POLICY,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
                'domain_strategy': DomainType.INDUCTION,
            },
            'algebraic': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.UCT,
                'expansion_strategy': ExpansionStrategyType.TREE_POLICY,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
                'domain_strategy': DomainType.ALGEBRAIC,
            },
            'logical': {
                'rollout_policy': RolloutPolicyType.HEURISTIC,
                'selection_strategy': SelectionStrategyType.THOMPSON_SAMPLING,
                'expansion_strategy': ExpansionStrategyType.TREE_POLICY,
                'backpropagation_strategy': BackpropagationStrategyType.AMAF,
                'domain_strategy': DomainType.LOGICAL,
            },
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        # Get preset config and apply overrides
        config = presets[preset_name].copy()
        config.update(kwargs)

        # Extract components
        rollout_policy = config.pop('rollout_policy')
        selection_strategy = config.pop('selection_strategy')
        expansion_strategy = config.pop('expansion_strategy')
        backpropagation_strategy = config.pop('backpropagation_strategy')
        domain_strategy = config.pop('domain_strategy', None)

        # Create composite strategy with remaining kwargs
        return MCTSStrategyFactory.create_composite_strategy(
            rollout_policy=rollout_policy,
            selection_strategy=selection_strategy,
            expansion_strategy=expansion_strategy,
            backpropagation_strategy=backpropagation_strategy,
            domain_strategy=domain_strategy,
            **config
        )


# ============================================================================
# Performance Tracker
# ============================================================================

class MCTSPerformanceTracker:
    """
    Tracks performance metrics for MCTS strategies.

    Records:
    - Success rate per strategy
    - Average search time
    - Average tree depth
    - Proof quality metrics
    - Strategy comparisons
    """

    def __init__(self):
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.search_history: List[Dict[str, Any]] = []
        self.comparison_results: Dict[str, Dict[str, float]] = {}

    def record_search(
        self,
        strategy_name: str,
        result: MCTSSearchResult,
        proof_quality: float = 0.0
    ) -> None:
        """Record performance of a strategy search"""
        # Initialize performance tracking if needed
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance()

        performance = self.strategy_performance[strategy_name]

        # Record metrics
        performance.total_uses += 1
        performance.last_used = time.time()

        # Update running averages
        alpha = 1.0 / performance.total_uses
        performance.success_rate = (
            (1 - alpha) * performance.success_rate +
            alpha * (1.0 if result.success else 0.0)
        )
        performance.avg_search_time = (
            (1 - alpha) * performance.avg_search_time +
            alpha * result.search_time
        )
        performance.avg_tree_depth = (
            (1 - alpha) * performance.avg_tree_depth +
            alpha * result.tree_depth
        )
        performance.avg_nodes_visited = (
            (1 - alpha) * performance.avg_nodes_visited +
            alpha * result.nodes_visited
        )
        performance.proof_quality_score = (
            (1 - alpha) * performance.proof_quality_score +
            alpha * proof_quality
        )

        # Record in history
        self.search_history.append({
            'strategy': strategy_name,
            'success': result.success,
            'time': result.search_time,
            'depth': result.tree_depth,
            'nodes': result.nodes_visited,
            'quality': proof_quality,
            'timestamp': time.time(),
        })

    def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance statistics for a strategy"""
        return self.strategy_performance.get(strategy_name)

    def compare_strategies(self, strategy_names: List[str]) -> Dict[str, float]:
        """Compare strategies and return rankings"""
        stats = {}
        for name in strategy_names:
            if name in self.strategy_performance:
                perf = self.strategy_performance[name]
                # Combined score: success rate - time penalty + quality bonus
                score = (
                    perf.success_rate * 10.0 -
                    min(perf.avg_search_time / 10.0, 1.0) +
                    perf.proof_quality_score * 2.0
                )
                stats[name] = score

        # Sort by score
        sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
        self.comparison_results = sorted_stats

        return sorted_stats

    def get_best_strategy(self, domain: Optional[DomainType] = None) -> Optional[str]:
        """Get the best performing strategy"""
        if not self.strategy_performance:
            return None

        # Filter by domain if specified
        candidates = list(self.strategy_performance.keys())

        if domain:
            candidates = [s for s in candidates if domain.value in s.lower()]

        if not candidates:
            return None

        # Get best by success rate (with minimum uses threshold)
        qualified = [
            s for s in candidates
            if self.strategy_performance[s].total_uses >= 3
        ]

        if not qualified:
            qualified = candidates

        return max(
            qualified,
            key=lambda s: self.strategy_performance[s].success_rate
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Export performance metrics"""
        return {
            'strategy_performance': {
                name: {
                    'success_rate': perf.success_rate,
                    'avg_search_time': perf.avg_search_time,
                    'avg_tree_depth': perf.avg_tree_depth,
                    'avg_nodes_visited': perf.avg_nodes_visited,
                    'proof_quality': perf.proof_quality_score,
                    'total_uses': perf.total_uses,
                }
                for name, perf in self.strategy_performance.items()
            },
            'total_searches': len(self.search_history),
            'strategy_rankings': self.comparison_results,
        }


# ============================================================================
# MDAP-Aware MCTS Strategies
# =============================================================================

# Import MDAP components if available
try:
    from leanaide_mdap import (
        LeanMDAPOrchestrator,
        LeanMDAPConfig,
        LeanProofAgent,
        ProofStrategy,
        VotingStrategy,
        MDAP_AVAILABLE,
    )
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP components not available - MDAP-aware strategies will be limited")


class MDAPAwareRolloutPolicy(RolloutPolicy):
    """
    MDAP-aware rollout policy using agent consensus.

    Uses multiple MDAP agents to vote on tactic selection during rollouts.
    Combines agent votes with heuristic scoring for informed rollouts.
    """

    def __init__(
        self,
        num_agents: int = 4,
        agent_types: Optional[List[str]] = None,
        voting_strategy: str = "weighted",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize MDAP-aware rollout policy.

        Args:
            num_agents: Number of MDAP agents to use
            agent_types: Types of agents (evolution, mcts, adversarial, self_play)
            voting_strategy: Strategy for combining agent votes
            confidence_threshold: Minimum confidence for agent consensus
        """
        super().__init__(name="mdap_aware_rollout")
        self.num_agents = num_agents
        self.agent_types = agent_types or ["evolution", "mcts", "adversarial", "self_play"]
        self.voting_strategy = voting_strategy
        self.confidence_threshold = confidence_threshold

        # Agent performance weights (simulated - in real implementation would be learned)
        self.agent_weights = {
            "evolution": 0.75,
            "mcts": 0.80,
            "adversarial": 0.70,
            "self_play": 0.65,
        }

        if MDAP_AVAILABLE:
            self._initialize_mdap()

    def _initialize_mdap(self) -> None:
        """Initialize MDAP orchestrator."""
        try:
            mdap_config = LeanMDAPConfig(
                available_agents=self.agent_types,
                default_parallel_agents=self.num_agents,
                voting_strategy=VotingStrategy(self.voting_strategy),
            )
            # Note: Would need Team object in full implementation
            logger.info("MDAP orchestrator initialized for rollout policy")
        except Exception as e:
            logger.warning(f"Failed to initialize MDAP for rollout: {e}")

    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """
        Select tactic using MDAP agent consensus.

        Args:
            tactics: Available tactics
            state: Current proof state

        Returns:
            Selected tactic
        """
        if not tactics:
            return ""

        # Get agent votes for each tactic
        tactic_scores = defaultdict(float)

        for tactic in tactics:
            for agent_type in self.agent_types:
                # Simulate agent vote (in real implementation, would query agent)
                weight = self.agent_weights.get(agent_type, 0.5)
                tactic_scores[tactic] += weight * self._tactic_agent_score(tactic, state, agent_type)

        # Normalize scores
        max_score = max(tactic_scores.values()) if tactic_scores else 1.0
        if max_score > 0:
            tactic_scores = {t: s / max_score for t, s in tactic_scores.items()}

        # Select tactic with highest score
        best_tactic = max(tactic_scores.items(), key=lambda x: x[1])[0]

        return best_tactic

    def _tactic_agent_score(self, tactic: str, state: Dict[str, Any], agent_type: str) -> float:
        """
        Get agent-specific score for a tactic.

        Args:
            tactic: Tactic to score
            state: Current proof state
            agent_type: Type of agent

        Returns:
            Score from 0 to 1
        """
        # Simulate agent-specific scoring
        goal = state.get("goal", "").lower()
        domain = state.get("domain", "general")

        score = 0.5  # Base score

        # Agent-type-specific preferences
        if agent_type == "evolution":
            if tactic in ["simp", "rw", "apply"]:
                score += 0.2
        elif agent_type == "mcts":
            if tactic in ["intros", "simp", "cases"]:
                score += 0.25
        elif agent_type == "adversarial":
            if tactic in ["by_contradiction", "push_neg"]:
                score += 0.3
        elif agent_type == "self_play":
            if tactic in ["aesop", "linarith"]:
                score += 0.2

        # Domain-specific adjustments
        if "nat" in goal or "induction" in domain:
            if tactic == "induction":
                score += 0.3
        elif "=" in goal or "algebra" in domain:
            if tactic in ["ring", "linarith"]:
                score += 0.2

        return min(1.0, score)

    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """
        Perform MDAP-aware rollout.

        Args:
            state: Starting state
            max_depth: Maximum rollout depth

        Returns:
            Estimated value (0 to 1)
        """
        current_state = state
        total_value = 0.0
        depth = 0

        for _ in range(max_depth):
            if self._is_terminal(current_state):
                return 1.0

            # Get available tactics
            tactics = current_state.get("available_tactics", [])
            if not tactics:
                break

            # Select tactic using MDAP consensus
            selected_tactic = self.select_tactic(tactics, current_state)

            # Apply tactic (simulate)
            current_state = self._apply_tactic(current_state, selected_tactic)
            depth += 1

            # Update value based on progress
            goals_remaining = len(current_state.get("goals", [current_state.get("goal", "")]))
            total_value += 1.0 / (goals_remaining + 1)

        return min(1.0, total_value / max_depth)

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if state is terminal."""
        return state.get("is_solved", False) or len(state.get("goals", [])) == 0

    def _apply_tactic(self, state: Dict[str, Any], tactic: str) -> Dict[str, Any]:
        """Apply tactic and return new state (simplified simulation)."""
        new_state = state.copy()

        # Simplified tactic application
        if tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.5:
                new_state["is_solved"] = True
        elif tactic == "intros":
            goals = new_state.get("goals", [])
            if goals:
                new_state["goals"] = goals[1:]

        return new_state


class AgentWeightedSelection(SelectionStrategy):
    """
    Agent-weighted selection using UCT with MDAP agent performance weights.

    UCT_with_agent_weights = UCT_base + weighted_agent_bonus
    """

    def __init__(
        self,
        c_param: float = 1.414,
        agent_weights: Optional[Dict[str, float]] = None,
        weight_bonus: float = 0.3
    ):
        """
        Initialize agent-weighted selection.

        Args:
            c_param: UCT exploration constant
            agent_weights: Weights for each agent type
            weight_bonus: Bonus multiplier for agent weights
        """
        super().__init__(name="agent_weighted_selection")
        self.c_param = c_param
        self.agent_weights = agent_weights or {
            "evolution": 0.75,
            "mcts": 0.80,
            "adversarial": 0.70,
            "self_play": 0.65,
        }
        self.weight_bonus = weight_bonus

    def select_child(self, children: List[MCTSNode]) -> MCTSNode:
        """
        Select child using agent-weighted UCT.

        Args:
            children: List of child nodes

        Returns:
            Selected child
        """
        if not children:
            raise ValueError("No children to select from")

        def agent_weighted_uct(node: MCTSNode) -> float:
            """Calculate UCT with agent weight bonus."""
            if node.visits == 0:
                return float('inf')

            # Standard UCT
            exploitation = node.mean_value
            if node.parent and node.parent.visits > 0:
                exploration = self.c_param * math.sqrt(math.log(node.parent.visits) / node.visits)
            else:
                exploration = self.c_param

            base_uct = exploitation + exploration

            # Add agent weight bonus based on action
            if node.action:
                # Get agent consensus for this action (simulated)
                agent_bonus = self._get_agent_bonus(node.action)
                base_uct += agent_bonus * self.weight_bonus

            return base_uct

        return max(children, key=agent_weighted_uct)

    def _get_agent_bonus(self, action: str) -> float:
        """
        Get agent consensus bonus for an action.

        Args:
            action: Tactic action

        Returns:
            Bonus score from 0 to 1
        """
        # Simulate agent votes
        total_weight = 0.0
        for weight in self.agent_weights.values():
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Action-specific agent preferences
        action_bonus = 0.0
        for agent_type, weight in self.agent_weights.items():
            if agent_type == "evolution" and action in ["simp", "rw"]:
                action_bonus += weight
            elif agent_type == "mcts" and action in ["intros", "cases"]:
                action_bonus += weight
            elif agent_type == "adversarial" and action in ["by_contradiction"]:
                action_bonus += weight
            elif agent_type == "self_play" and action in ["aesop"]:
                action_bonus += weight

        return action_bonus / total_weight


class VotingBasedExpansion(ExpansionStrategy):
    """
    Voting-based expansion using MDAP agent consensus.

    Multiple agents suggest child nodes, then voting selects which to expand.
    """

    def __init__(
        self,
        num_agents: int = 4,
        agent_types: Optional[List[str]] = None,
        voting_method: str = "majority",
        expansion_budget: int = 5
    ):
        """
        Initialize voting-based expansion.

        Args:
            num_agents: Number of MDAP agents
            agent_types: Types of agents to use
            voting_method: Method for voting (majority, weighted, unanimous)
            expansion_budget: Maximum children to expand
        """
        super().__init__(name="voting_based_expansion")
        self.num_agents = num_agents
        self.agent_types = agent_types or ["evolution", "mcts", "adversarial", "self_play"]
        self.voting_method = voting_method
        self.expansion_budget = expansion_budget

    def expand(self, node: MCTSNode, available_actions: List[str]) -> List[str]:
        """
        Select actions to expand using voting.

        Args:
            node: Node to expand
            available_actions: Available actions/tactics

        Returns:
            Selected actions for expansion
        """
        if not available_actions:
            return []

        # Get agent votes for each action
        action_votes = defaultdict(float)

        for action in available_actions:
            votes = self._get_agent_votes(action, node)
            action_votes[action] = votes

        # Select top actions based on voting method
        if self.voting_method == "majority":
            # Select actions with majority support
            threshold = self.num_agents / 2.0
            selected = [a for a, v in action_votes.items() if v >= threshold]
        elif self.voting_method == "weighted":
            # Select top actions by weighted votes
            sorted_actions = sorted(action_votes.items(), key=lambda x: x[1], reverse=True)
            selected = [a for a, _ in sorted_actions[:self.expansion_budget]]
        else:  # unanimous
            # Select actions with unanimous support
            selected = [a for a, v in action_votes.items() if v >= self.num_agents]

        return selected[:self.expansion_budget]

    def _get_agent_votes(self, action: str, node: MCTSNode) -> float:
        """
        Get agent votes for an action.

        Args:
            action: Tactic action
            node: Current node

        Returns:
            Weighted vote count
        """
        votes = 0.0

        # Simulate agent voting based on action and node state
        state = node.state

        for agent_type in self.agent_types:
            # Agent-type-specific voting
            if agent_type == "evolution":
                if action in ["simp", "rw", "apply"]:
                    votes += 1.0
            elif agent_type == "mcts":
                if action in ["intros", "cases", "induction"]:
                    votes += 1.0
            elif agent_type == "adversarial":
                if action in ["by_contradiction", "push_neg"]:
                    votes += 1.0
            elif agent_type == "self_play":
                if action in ["aesop", "linarith", "ring"]:
                    votes += 1.0

        # Context-aware voting
        goal = state.get("goal", "").lower()
        if "nat" in goal and action == "induction":
            votes += 1.0
        elif "=" in goal and action in ["ring", "linarith"]:
            votes += 0.5

        return votes


class ConsensusRolloutPolicy(RolloutPolicy):
    """
    Consensus rollout using MAKER voting mechanism.

    Uses first-K-ahead-by-K voting for more robust rollout decisions.
    """

    def __init__(
        self,
        k_ahead: int = 3,
        num_agents: int = 4,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize consensus rollout policy.

        Args:
            k_ahead: K parameter for first-K-ahead voting
            num_agents: Number of agents for voting
            confidence_threshold: Minimum confidence to accept consensus
        """
        super().__init__(name="consensus_rollout")
        self.k_ahead = k_ahead
        self.num_agents = num_agents
        self.confidence_threshold = confidence_threshold

    def select_tactic(self, tactics: List[str], state: Dict[str, Any]) -> str:
        """
        Select tactic using first-K-ahead consensus.

        Args:
            tactics: Available tactics
            state: Current proof state

        Returns:
            Selected tactic
        """
        if not tactics:
            return ""

        # Get top-K tactics by initial scoring
        tactic_scores = [(t, self._score_tactic(t, state)) for t in tactics]
        tactic_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-K tactics
        top_k = tactic_scores[:self.k_ahead]

        # Check if there's a clear winner (consensus)
        if len(top_k) > 0:
            best_tactic, best_score = top_k[0]
            if best_score >= self.confidence_threshold:
                return best_tactic

        # If no clear consensus, use weighted random selection from top-K
        if top_k:
            total_score = sum(score for _, score in top_k)
            if total_score > 0:
                probs = [score / total_score for _, score in top_k]
                selected_idx = random.choices(range(len(top_k)), weights=probs)[0]
                return top_k[selected_idx][0]

        # Fallback to best scoring
        return top_k[0][0] if top_k else tactics[0]

    def _score_tactic(self, tactic: str, state: Dict[str, Any]) -> float:
        """
        Score a tactic based on state context.

        Args:
            tactic: Tactic to score
            state: Current proof state

        Returns:
            Score from 0 to 1
        """
        score = 0.5  # Base score

        goal = state.get("goal", "").lower()
        domain = state.get("domain", "general")
        depth = state.get("depth", 0)

        # Depth-based preferences
        if depth < 5:
            # Early in proof: prefer intro tactics
            if tactic in ["intros", "intro"]:
                score += 0.3
        elif depth >= 10:
            # Later in proof: prefer automation
            if tactic in ["aesop", "simp", "trivial"]:
                score += 0.3

        # Goal-based preferences
        if "nat" in goal:
            if tactic == "induction":
                score += 0.4
            elif tactic in ["simp", "norm_num"]:
                score += 0.2
        elif "=" in goal:
            if tactic in ["ring", "linarith", "rw"]:
                score += 0.3
        elif "->" in goal or "forall" in goal:
            if tactic == "intros":
                score += 0.3

        # Domain-based preferences
        if domain == "induction":
            if tactic in ["induction", "cases"]:
                score += 0.3
        elif domain == "algebraic":
            if tactic in ["ring", "simp", "calc"]:
                score += 0.3

        return min(1.0, score)

    def rollout(self, state: Dict[str, Any], max_depth: int) -> float:
        """
        Perform consensus-based rollout.

        Args:
            state: Starting state
            max_depth: Maximum rollout depth

        Returns:
            Estimated value
        """
        current_state = state
        depth = 0

        for _ in range(max_depth):
            if self._is_terminal(current_state):
                return 1.0

            tactics = current_state.get("available_tactics", [])
            if not tactics:
                break

            selected = self.select_tactic(tactics, current_state)
            current_state = self._apply_tactic(current_state, selected)
            depth += 1

        # Estimate value based on progress
        goals_remaining = len(current_state.get("goals", [current_state.get("goal", "")]))
        progress = 1.0 - (goals_remaining / max(1, len(state.get("goals", [state.get("goal", "")]))))

        return max(0.0, min(1.0, progress))

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if state is terminal."""
        return state.get("is_solved", False)

    def _apply_tactic(self, state: Dict[str, Any], tactic: str) -> Dict[str, Any]:
        """Apply tactic (simplified)."""
        new_state = state.copy()
        if tactic in ["simp", "aesop"] and random.random() > 0.5:
            new_state["is_solved"] = True
        return new_state


# ============================================================================
# Example Usage and Tests
# ============================================================================

def example_rollout_policy():
    """Example: Create and use rollout policies"""
    print("=== Rollout Policy Examples ===\n")

    # Random rollout
    random_policy = RandomRolloutPolicy()
    print(f"Random policy: {random_policy.name}")

    # Heuristic rollout
    heuristic_policy = HeuristicRolloutPolicy()
    print(f"Heuristic policy: {heuristic_policy.name}")

    # Test tactic selection
    test_state = {
        "goal": "∀ n : Nat, n + 0 = n",
        "domain": "induction",
        "available_tactics": ["simp", "intro", "induction", "cases", "apply"],
        "context": [],
    }

    tactics = test_state["available_tactics"]

    print("\nTactic selection test:")
    print(f"State: {test_state['goal']}")

    selected_random = random_policy.select_tactic(tactics, test_state)
    print(f"Random policy selected: {selected_random}")

    selected_heuristic = heuristic_policy.select_tactic(tactics, test_state)
    print(f"Heuristic policy selected: {selected_heuristic}")

    # Test rollout
    print("\nRollout test:")
    value = heuristic_policy.rollout(test_state, max_depth=10)
    print(f"Heuristic rollout value: {value:.3f}")


def example_selection_strategy():
    """Example: Create and use selection strategies"""
    print("\n=== Selection Strategy Examples ===\n")

    # Create test nodes
    root = MCTSNode(state={"test": "root"}, visits=100)

    for i in range(5):
        child = MCTSNode(
            state={"test": f"child_{i}"},
            parent=root,
            visits=random.randint(10, 50),
            value=random.uniform(5, 25)
        )
        child.mean_value = child.value / child.visits if child.visits > 0 else 0
        root.children.append(child)

    # UCT selection
    uct = UCTSelection(c_param=1.414)
    selected_uct = uct.select_child(root.children)
    print(f"UCT selected child with value: {selected_uct.mean_value:.3f}, visits: {selected_uct.visits}")

    # Adaptive UCT
    adaptive = AdaptiveUCTSelection(base_c=1.414)
    selected_adaptive = adaptive.select_child(root.children, depth=5)
    print(f"Adaptive UCT selected child with value: {selected_adaptive.mean_value:.3f}")

    # Thompson sampling
    thompson = ThompsonSamplingSelection()
    selected_ts = thompson.select_child(root.children)
    print(f"Thompson sampling selected child with value: {selected_ts.mean_value:.3f}")


def example_strategy_factory():
    """Example: Use the strategy factory"""
    print("\n=== Strategy Factory Examples ===\n")

    # Create individual components
    print("Creating individual components:")
    rollout = MCTSStrategyFactory.create_rollout_policy(RolloutPolicyType.HEURISTIC)
    print(f"  Rollout: {rollout.name}")

    selection = MCTSStrategyFactory.create_selection_strategy(SelectionStrategyType.UCT, c_param=1.5)
    print(f"  Selection: {selection.name}")

    expansion = MCTSStrategyFactory.create_expansion_strategy(ExpansionStrategyType.PROGRESSIVE_WIDENING)
    print(f"  Expansion: {expansion.name}")

    backprop = MCTSStrategyFactory.create_backpropagation_strategy(BackpropagationStrategyType.AMAF)
    print(f"  Backpropagation: {backprop.name}")

    # Create composite strategy
    print("\nCreating composite strategy:")
    composite = MCTSStrategyFactory.create_composite_strategy(
        rollout_policy=RolloutPolicyType.HEURISTIC,
        selection_strategy=SelectionStrategyType.ADAPTIVE_UCT,
        expansion_strategy=ExpansionStrategyType.TREE_POLICY,
        backpropagation_strategy=BackpropagationStrategyType.AMAF,
        base_c=1.3,
    )
    print(f"  Composite created with {len(composite)} components")

    # Create preset strategies
    print("\nCreating preset strategies:")
    presets = ['balanced', 'fast', 'induction', 'algebraic', 'logical']

    for preset in presets:
        strategy = MCTSStrategyFactory.create_preset_strategy(preset)
        print(f"  {preset.capitalize()}: {len(strategy)} components")


def example_domain_strategies():
    """Example: Use domain-specific strategies"""
    print("\n=== Domain Strategy Examples ===\n")

    # Induction
    induction = InductionMCTS()
    print(f"Induction strategy: {induction.name}")

    test_state = {
        "goal": "∀ n : Nat, n + 0 = n",
        "domain": "induction",
        "available_tactics": ["simp", "intro", "induction", "cases", "apply", "norm_num"],
        "depth": 0,
    }

    induction_scores = induction.score_tactics(test_state)
    print("Induction tactic scores:")
    for tactic, score in sorted(induction_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tactic}: {score:.2f}")

    # Algebraic
    algebraic = AlgebraicMCTS()
    print(f"\nAlgebraic strategy: {algebraic.name}")

    algebraic_state = {
        "goal": "∀ a b : ℝ, (a + b)² = a² + 2ab + b²",
        "domain": "algebraic",
        "available_tactics": ["ring", "simp", "calc", "norm_num", "linarith"],
        "depth": 0,
    }

    algebraic_scores = algebraic.score_tactics(algebraic_state)
    print("Algebraic tactic scores:")
    for tactic, score in sorted(algebraic_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tactic}: {score:.2f}")


def example_performance_tracker():
    """Example: Use performance tracker"""
    print("\n=== Performance Tracker Example ===\n")

    tracker = MCTSPerformanceTracker()

    # Simulate some searches
    strategies = ['uct_heuristic', 'adaptive_uct_amaf', 'thompson_tree']

    for i in range(10):
        strategy = random.choice(strategies)

        result = MCTSSearchResult(
            success=random.random() > 0.6,  # 40% success rate
            search_time=random.uniform(1.0, 10.0),
            tree_depth=random.randint(5, 20),
            nodes_visited=random.randint(50, 200),
            value=random.uniform(0.0, 1.0),
        )

        quality = random.uniform(0.5, 1.0) if result.success else 0.0
        tracker.record_search(strategy, result, quality)

    # Get statistics
    print("Strategy performance:")
    for name, perf in tracker.strategy_performance.items():
        print(f"\n{name}:")
        print(f"  Success rate: {perf.success_rate:.2%}")
        print(f"  Avg time: {perf.avg_search_time:.2f}s")
        print(f"  Avg depth: {perf.avg_tree_depth:.1f}")
        print(f"  Uses: {perf.total_uses}")

    # Compare strategies
    print("\nStrategy rankings:")
    rankings = tracker.compare_strategies(strategies)
    for rank, (strategy, score) in enumerate(rankings.items(), 1):
        print(f"  {rank}. {strategy}: {score:.2f}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("LeanAide MCTS Strategy Library")
    print("=" * 50)

    # Run examples
    example_rollout_policy()
    example_selection_strategy()
    example_strategy_factory()
    example_domain_strategies()
    example_performance_tracker()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
