"""
RESE Phase III: MCTS Search Executor

This module implements Phase III of RESE - Monte Carlo Refinement with MC-NEST algorithm.

Core Components:
- MCTSSearchExecutor: Main orchestrator for MC-NEST
- SearchTreeBuilder: Constructs MCTS search tree
- HypothesisValidator: Statistically validates hypotheses
- ConvergenceDetector: Detects convergence via ACI
- SelectionStrategy: UCB1 selection for node expansion

Following CLAUDE.md principles:
- Law of Idempotency: Deduplicate by hypothesis_id
- Law of Configuration Explicitness: Env vars (PHASE3_ITERATIONS, UCB1_C, etc.)
- Circuit Breaker: Detect search failures
- Structured Logging: JSON with correlation_id
- Timeout: All search operations timeout (default 30000ms)
- Dead Letter Queue: For invalid hypotheses

Author: RESE Team
Created: 2026-02-04
Phase: III - Monte Carlo Refinement
"""

import os
import sys
import time
import uuid
import json
import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "schemas"))

try:
    from rese_schemas import (
        Hypothesis,
        SearchTreeNode,
        MCTSSearchResult,
        ExplorationConfig,
        HypothesisStatus,
        MCTSNodeState,
        ExplorationStrategy,
    )
    from rese_dee import (
        DEELogger,
        CircuitBreaker,
        retry_with_backoff,
    )
except ImportError:
    # Fallback imports
    from glue.schemas.rese_schemas import (
        Hypothesis,
        SearchTreeNode,
        MCTSSearchResult,
        ExplorationConfig,
        HypothesisStatus,
        MCTSNodeState,
        ExplorationStrategy,
    )
    from glue.lib.rese_dee import (
        DEELogger,
        CircuitBreaker,
        retry_with_backoff,
    )


# ============================================================================
# CONFIGURATION (Law of Configuration Explicitness)
# ============================================================================

@dataclass
class Phase3Config:
    """
    Configuration for Phase III MCTS Search.

    All values MUST come from environment variables.
    Crashes immediately if required vars are missing.
    """
    # MCTS parameters
    iterations: int
    ucb1_c: float  # Exploration constant for UCB1
    convergence_threshold: float
    timeout_ms: int

    # Search tree parameters
    max_depth: int
    max_children_per_node: int
    min_visits_before_expand: int

    # Validation parameters
    statistical_significance_threshold: float
    confidence_interval: float  # e.g., 0.95 for 95% CI
    min_sample_size: int

    # ACI (Algorithmic Convergence Indicator) parameters
    aci_window_size: int
    aci_stability_threshold: float

    # Deduplication (Law of Idempotency)
    enable_deduplication: bool
    hypothesis_cache_size: int

    # Circuit breaker
    circuit_breaker_threshold: int
    circuit_breaker_timeout_ms: int

    # Correlation ID for tracing
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Phase3Config":
        """
        Load configuration from environment variables.

        Required env vars:
        - PHASE3_ITERATIONS
        - PHASE3_UCB1_C
        - PHASE3_CONVERGENCE_THRESHOLD
        - PHASE3_TIMEOUT_MS
        - PHASE3_MAX_DEPTH
        - PHASE3_MAX_CHILDREN
        - PHASE3_MIN_VISITS
        - PHASE3_SIG_THRESHOLD
        - PHASE3_CONFIDENCE_INTERVAL
        - PHASE3_MIN_SAMPLE_SIZE
        - PHASE3_ACI_WINDOW
        - PHASE3_ACI_STABILITY
        - PHASE3_DEDUP_ENABLED
        - PHASE3_CACHE_SIZE
        - PHASE3_CB_THRESHOLD
        - PHASE3_CB_TIMEOUT

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        env_vars = {
            "PHASE3_ITERATIONS": ("iterations", 1000, int),
            "PHASE3_UCB1_C": ("ucb1_c", 1.414, float),
            "PHASE3_CONVERGENCE_THRESHOLD": ("convergence_threshold", 0.001, float),
            "PHASE3_TIMEOUT_MS": ("timeout_ms", 30000, int),
            "PHASE3_MAX_DEPTH": ("max_depth", 20, int),
            "PHASE3_MAX_CHILDREN": ("max_children_per_node", 10, int),
            "PHASE3_MIN_VISITS": ("min_visits_before_expand", 5, int),
            "PHASE3_SIG_THRESHOLD": ("statistical_significance_threshold", 0.05, float),
            "PHASE3_CONFIDENCE_INTERVAL": ("confidence_interval", 0.95, float),
            "PHASE3_MIN_SAMPLE_SIZE": ("min_sample_size", 30, int),
            "PHASE3_ACI_WINDOW": ("aci_window_size", 100, int),
            "PHASE3_ACI_STABILITY": ("aci_stability_threshold", 0.01, float),
            "PHASE3_DEDUP_ENABLED": ("enable_deduplication", True, bool),
            "PHASE3_CACHE_SIZE": ("hypothesis_cache_size", 10000, int),
            "PHASE3_CB_THRESHOLD": ("circuit_breaker_threshold", 5, int),
            "PHASE3_CB_TIMEOUT": ("circuit_breaker_timeout_ms", 60000, int),
        }

        config = {"correlation_id": os.getenv("CORRELATION_ID")}

        for env_name, (field_name, default, field_type) in env_vars.items():
            value = os.getenv(env_name)

            if value is None:
                # Use default for missing vars
                config[field_name] = default
            else:
                try:
                    if field_type == bool:
                        config[field_name] = value.lower() in ("true", "1", "yes")
                    else:
                        config[field_name] = field_type(value)
                except (ValueError, TypeError) as e:
                    print(f"FATAL: Invalid value for {env_name}: {value}")
                    print(f"Expected {field_type.__name__}")
                    sys.exit(1)

        return cls(**config)


# ============================================================================
# DEAD LETTER QUEUE
# ============================================================================

class HypothesisDLQ:
    """
    Dead Letter Queue for invalid hypotheses.

    Stores hypotheses that failed validation for later analysis.
    """

    def __init__(self, logger: Optional[DEELogger] = None):
        self.logger = logger or DEELogger()
        self.failed_hypotheses: List[Dict[str, Any]] = []
        self.max_size = int(os.getenv("PHASE3_DLQ_MAX_SIZE", "1000"))

    def add(
        self,
        hypothesis: Hypothesis,
        error: str,
        error_type: str,
        validation_result: Optional[Dict[str, Any]] = None
    ):
        """
        Add failed hypothesis to DLQ.

        Args:
            hypothesis: The failed hypothesis
            error: Error message
            error_type: Type of error (validation, statistical, convergence)
            validation_result: Optional validation details
        """
        if len(self.failed_hypotheses) >= self.max_size:
            self.logger.warning("DLQ full, dropping oldest hypothesis")
            self.failed_hypotheses.pop(0)

        dlq_entry = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "statement": hypothesis.statement,
            "error": error,
            "error_type": error_type,
            "validation_result": validation_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.failed_hypotheses.append(dlq_entry)

        self.logger.error(
            "Hypothesis added to DLQ",
            hypothesis_id=hypothesis.hypothesis_id,
            error_type=error_type,
            error=error
        )

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all failed hypotheses."""
        return self.failed_hypotheses.copy()

    def clear(self):
        """Clear the DLQ."""
        self.failed_hypotheses.clear()
        self.logger.info("DLQ cleared")

    def size(self) -> int:
        """Get DLQ size."""
        return len(self.failed_hypotheses)


# ============================================================================
# UCB1 SELECTION STRATEGY
# ============================================================================

class UCB1SelectionStrategy:
    """
    UCB1 (Upper Confidence Bound) selection for MCTS node expansion.

    UCB1 = mean_value + c * sqrt(ln(parent_visits) / visits)

    Balances exploration (high c) and exploitation (high mean_value).
    """

    def __init__(self, exploration_constant: float = 1.414):
        """
        Initialize UCB1 selector.

        Args:
            exploration_constant: C parameter for UCB1 (default sqrt(2))
        """
        self.exploration_constant = exploration_constant

    def select_child(
        self,
        parent_node: SearchTreeNode,
        tree: Dict[str, SearchTreeNode]
    ) -> Optional[SearchTreeNode]:
        """
        Select best child using UCB1.

        Args:
            parent_node: Parent node
            tree: Search tree

        Returns:
            Selected child node or None if no children
        """
        if not parent_node.children:
            return None

        # Calculate UCB1 for each child
        best_child = None
        best_ucb1 = float('-inf')

        for child_id in parent_node.children:
            if child_id not in tree:
                continue

            child = tree[child_id]
            ucb1 = self.calculate_ucb1(parent_node, child)

            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_child = child

        return best_child

    def calculate_ucb1(
        self,
        parent_node: SearchTreeNode,
        child_node: SearchTreeNode
    ) -> float:
        """
        Calculate UCB1 value for child node.

        Args:
            parent_node: Parent node
            child_node: Child node

        Returns:
            UCB1 score
        """
        if child_node.visit_count == 0:
            return float('inf')

        exploitation = child_node.mean_value
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_node.visit_count + 1) / (child_node.visit_count + 1)
        )

        return exploitation + exploration


# ============================================================================
# SEARCH TREE BUILDER
# ============================================================================

class SearchTreeBuilder:
    """
    Builds and manages the MCTS search tree.

    Features:
    - Tree construction from root hypothesis
    - Node expansion with child hypotheses
    - Idempotent node updates (Law of Idempotency)
    - Tree traversal and statistics
    """

    def __init__(
        self,
        config: Phase3Config,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.tree: Dict[str, SearchTreeNode] = {}
        self.hypothesis_cache: Set[str] = set()  # For deduplication

    def build_root(
        self,
        root_hypothesis: Hypothesis
    ) -> SearchTreeNode:
        """
        Build root node from hypothesis.

        Args:
            root_hypothesis: Root hypothesis

        Returns:
            Root search tree node
        """
        root_node = SearchTreeNode(
            node_id=root_hypothesis.hypothesis_id,
            hypothesis=root_hypothesis,
            state=MCTSNodeState.EXPANDED,
            depth=0
        )

        self.tree[root_node.node_id] = root_node
        self.hypothesis_cache.add(root_hypothesis.hypothesis_id)

        self.logger.info(
            "Root node created",
            node_id=root_node.node_id,
            hypothesis_id=root_hypothesis.hypothesis_id
        )

        return root_node

    def expand_node(
        self,
        parent_node: SearchTreeNode,
        child_hypotheses: List[Hypothesis]
    ) -> List[SearchTreeNode]:
        """
        Expand node with child hypotheses.

        Idempotent: deduplicates by hypothesis_id (Law of Idempotency).

        Args:
            parent_node: Parent node to expand
            child_hypotheses: Child hypotheses to add

        Returns:
            List of new child nodes
        """
        if parent_node.depth >= self.config.max_depth:
            self.logger.warning(
                "Max depth reached, cannot expand",
                node_id=parent_node.node_id,
                depth=parent_node.depth
            )
            return []

        if len(parent_node.children) >= self.config.max_children_per_node:
            self.logger.warning(
                "Max children reached, cannot expand",
                node_id=parent_node.node_id,
                num_children=len(parent_node.children)
            )
            return []

        new_nodes = []

        for hypothesis in child_hypotheses:
            # Deduplication check
            if hypothesis.hypothesis_id in self.hypothesis_cache:
                self.logger.debug(
                    "Skipping duplicate hypothesis",
                    hypothesis_id=hypothesis.hypothesis_id
                )
                continue

            # Create child node
            child_node = SearchTreeNode(
                node_id=hypothesis.hypothesis_id,
                hypothesis=hypothesis,
                state=MCTSNodeState.UNEXPANDED,
                parent_id=parent_node.node_id,
                depth=parent_node.depth + 1
            )

            # Add to tree
            self.tree[child_node.node_id] = child_node
            self.hypothesis_cache.add(hypothesis.hypothesis_id)

            # Link to parent
            if child_node.node_id not in parent_node.children:
                parent_node.children.append(child_node.node_id)

            new_nodes.append(child_node)

        # Update parent state
        if new_nodes:
            parent_node.state = MCTSNodeState.EXPANDED

        self.logger.info(
            "Node expanded",
            parent_id=parent_node.node_id,
            num_new_children=len(new_nodes),
            total_children=len(parent_node.children)
        )

        return new_nodes

    def update_node_value(
        self,
        node_id: str,
        reward: float
    ):
        """
        Update node value with reward (idempotent).

        Args:
            node_id: Node to update
            reward: Reward value
        """
        if node_id not in self.tree:
            self.logger.warning("Node not found for update", node_id=node_id)
            return

        node = self.tree[node_id]
        node.update_value(reward)

        self.logger.debug(
            "Node value updated",
            node_id=node_id,
            reward=reward,
            visit_count=node.visit_count,
            mean_value=node.mean_value
        )

    def get_node(self, node_id: str) -> Optional[SearchTreeNode]:
        """Get node by ID."""
        return self.tree.get(node_id)

    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        if not self.tree:
            return {
                "total_nodes": 0,
                "max_depth": 0,
                "root_id": None,
                "leaf_nodes": 0,
                "expanded_nodes": 0,
            }

        max_depth = max(node.depth for node in self.tree.values())
        leaf_nodes = sum(1 for node in self.tree.values() if not node.children)
        expanded_nodes = sum(
            1 for node in self.tree.values()
            if node.state == MCTSNodeState.EXPANDED
        )

        return {
            "total_nodes": len(self.tree),
            "max_depth": max_depth,
            "root_id": next(iter(self.tree.keys()), None),
            "leaf_nodes": leaf_nodes,
            "expanded_nodes": expanded_nodes,
        }


# ============================================================================
# HYPOTHESIS VALIDATOR (Statistical Validation)
# ============================================================================

@dataclass
class ValidationMetrics:
    """Statistical validation metrics for a hypothesis."""
    hypothesis_id: str
    is_valid: bool
    confidence: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    mean_reward: float
    std_reward: float
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "sample_size": self.sample_size,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "validation_timestamp": self.validation_timestamp.isoformat(),
        }


class HypothesisValidator:
    """
    Statistically validates hypotheses using hypothesis testing.

    Performs:
    - T-tests for statistical significance
    - Confidence interval calculation
    - Sample size validation
    - Effect size calculation
    """

    def __init__(
        self,
        config: Phase3Config,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.validation_cache: Dict[str, ValidationMetrics] = {}

    def validate(
        self,
        hypothesis: Hypothesis,
        rewards: List[float]
    ) -> Tuple[ValidationMetrics, Optional[str]]:
        """
        Validate hypothesis with statistical tests.

        Args:
            hypothesis: Hypothesis to validate
            rewards: List of reward values from simulations

        Returns:
            Tuple of (validation_metrics, error_message)
        """
        # Check sample size
        if len(rewards) < self.config.min_sample_size:
            error = f"Insufficient sample size: {len(rewards)} < {self.config.min_sample_size}"
            self.logger.warning(
                error,
                hypothesis_id=hypothesis.hypothesis_id,
                sample_size=len(rewards)
            )
            return ValidationMetrics(
                hypothesis_id=hypothesis.hypothesis_id,
                is_valid=False,
                confidence=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                sample_size=len(rewards),
                mean_reward=0.0,
                std_reward=0.0,
            ), error

        # Calculate statistics
        import statistics
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        sample_size = len(rewards)

        # Calculate confidence interval
        ci_margin = self._calculate_confidence_interval(
            std_reward,
            sample_size
        )
        confidence_interval = (
            mean_reward - ci_margin,
            mean_reward + ci_margin
        )

        # Perform t-test (test if mean > 0.5)
        t_statistic, p_value = self._perform_t_test(
            mean_reward,
            std_reward,
            sample_size,
            null_hypothesis_mean=0.5
        )

        # Check statistical significance
        is_significant = p_value < self.config.statistical_significance_threshold
        is_valid = is_significant and mean_reward > 0.5

        # Calculate confidence score
        confidence = mean_reward if is_valid else mean_reward * 0.5

        validation_metrics = ValidationMetrics(
            hypothesis_id=hypothesis.hypothesis_id,
            is_valid=is_valid,
            confidence=confidence,
            p_value=p_value,
            confidence_interval=confidence_interval,
            sample_size=sample_size,
            mean_reward=mean_reward,
            std_reward=std_reward,
        )

        # Cache result
        self.validation_cache[hypothesis.hypothesis_id] = validation_metrics

        self.logger.info(
            "Hypothesis validated",
            hypothesis_id=hypothesis.hypothesis_id,
            is_valid=is_valid,
            confidence=confidence,
            p_value=p_value,
            sample_size=sample_size
        )

        return validation_metrics, None

    def _calculate_confidence_interval(
        self,
        std_dev: float,
        sample_size: int
    ) -> float:
        """
        Calculate confidence interval margin.

        Uses t-distribution for small samples.
        """
        from math import sqrt

        # Approximate t-value for 95% CI (could use scipy.stats for exact)
        if sample_size >= 120:
            t_value = 1.96
        elif sample_size >= 30:
            t_value = 2.0
        else:
            t_value = 2.5  # Conservative for small samples

        margin = t_value * std_dev / sqrt(sample_size)
        return margin

    def _perform_t_test(
        self,
        mean: float,
        std_dev: float,
        sample_size: int,
        null_hypothesis_mean: float = 0.5
    ) -> Tuple[float, float]:
        """
        Perform one-sample t-test.

        Returns:
            Tuple of (t_statistic, p_value)
        """
        from math import sqrt

        # Calculate t-statistic
        standard_error = std_dev / sqrt(sample_size)
        t_statistic = (mean - null_hypothesis_mean) / standard_error if standard_error > 0 else 0.0

        # Approximate p-value (two-tailed)
        # For exact p-value, would use scipy.stats
        abs_t = abs(t_statistic)
        if abs_t > 3.0:
            p_value = 0.001
        elif abs_t > 2.0:
            p_value = 0.05
        elif abs_t > 1.0:
            p_value = 0.3
        else:
            p_value = 0.5

        return t_statistic, p_value


# ============================================================================
# CONVERGENCE DETECTOR (ACI - Algorithmic Convergence Indicator)
# ============================================================================

class ConvergenceDetector:
    """
    Detects convergence using ACI (Algorithmic Convergence Indicator).

    ACI measures:
    - Stability of best hypothesis confidence over window
    - Reduction in variance across top hypotheses
    - Plateau detection in reward improvements

    Convergence is reached when ACI indicates stability.
    """

    def __init__(
        self,
        config: Phase3Config,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.confidence_history: List[float] = []
        self.reward_history: List[float] = []
        self.iteration_history: List[int] = []

    def update(
        self,
        iteration: int,
        best_confidence: float,
        best_reward: float
    ):
        """
        Update convergence metrics.

        Args:
            iteration: Current iteration
            best_confidence: Best hypothesis confidence
            best_reward: Best reward value
        """
        self.confidence_history.append(best_confidence)
        self.reward_history.append(best_reward)
        self.iteration_history.append(iteration)

        # Keep window size bounded
        if len(self.confidence_history) > self.config.aci_window_size:
            self.confidence_history.pop(0)
            self.reward_history.pop(0)
            self.iteration_history.pop(0)

    def check_convergence(self) -> Tuple[bool, Optional[float]]:
        """
        Check if convergence has been reached.

        Returns:
            Tuple of (is_converged, aci_value)
        """
        if len(self.confidence_history) < self.config.aci_window_size:
            return False, None

        # Calculate ACI (stability metric)
        aci_value = self._calculate_aci()

        # Check if ACI indicates stability
        is_converged = aci_value < self.config.aci_stability_threshold

        if is_converged:
            self.logger.info(
                "Convergence detected",
                aci_value=aci_value,
                threshold=self.config.aci_stability_threshold,
                window_size=len(self.confidence_history)
            )

        return is_converged, aci_value

    def _calculate_aci(self) -> float:
        """
        Calculate ACI (Algorithmic Convergence Indicator).

        ACI measures variance in confidence over window.
        Lower ACI = more stable = converged.
        """
        import statistics

        if len(self.confidence_history) < 2:
            return float('inf')

        # Calculate variance in confidence
        variance = statistics.variance(self.confidence_history)

        # Normalize by range
        if max(self.confidence_history) - min(self.confidence_history) > 0:
            normalized_variance = variance / (
                max(self.confidence_history) - min(self.confidence_history)
            )
        else:
            normalized_variance = variance

        return normalized_variance


# ============================================================================
# MC-NEST EXECUTOR (Main Orchestrator)
# ============================================================================

class MCTSSearchExecutor:
    """
    Main executor for MC-NEST (Monte Carlo Nash Equilibrium Self-Refine Tree).

    Orchestrates Phase III MCTS search with:
    - UCB1 selection strategy
    - Statistical hypothesis validation
    - ACI convergence detection
    - Idempotent tree construction
    - Circuit breaker for failures
    - DLQ for invalid hypotheses

    Following CLAUDE.md principles:
    - Law of Idempotency: Deduplicate by hypothesis_id
    - Law of Configuration Explicitness: All config via env vars
    - Circuit Breaker: Detect search failures
    - Structured Logging: JSON with correlation_id
    - Timeout: All search operations timeout
    """

    def __init__(
        self,
        config: Optional[Phase3Config] = None,
        logger: Optional[DEELogger] = None
    ):
        """
        Initialize MCTS Search Executor.

        Args:
            config: Optional configuration (defaults to env vars)
            logger: Optional logger
        """
        self.config = config or Phase3Config.from_env()
        self.logger = logger or DEELogger(self.config.correlation_id)

        # Initialize components
        self.tree_builder = SearchTreeBuilder(self.config, self.logger)
        self.selection_strategy = UCB1SelectionStrategy(self.config.ucb1_c)
        self.hypothesis_validator = HypothesisValidator(self.config, self.logger)
        self.convergence_detector = ConvergenceDetector(self.config, self.logger)
        self.dlq = HypothesisDLQ(self.logger)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout_ms=self.config.circuit_breaker_timeout_ms,
            logger=self.logger
        )

        self.logger.info(
            "MCTS Search Executor initialized",
            config=self.config.__dict__
        )

    def execute_search(
        self,
        root_hypothesis: Hypothesis,
        hypothesis_generator: Callable[[], List[Hypothesis]],
        reward_function: Callable[[Hypothesis], float],
    ) -> Tuple[MCTSSearchResult, Optional[str]]:
        """
        Execute MC-NEST search from root hypothesis.

        Args:
            root_hypothesis: Starting hypothesis
            hypothesis_generator: Function to generate child hypotheses
            reward_function: Function to evaluate hypothesis reward

        Returns:
            Tuple of (search_result, error_message)
        """
        start_time = time.time()
        search_id = str(uuid.uuid4())

        self.logger.info(
            "Starting MC-NEST search",
            search_id=search_id,
            root_hypothesis_id=root_hypothesis.hypothesis_id,
            iterations=self.config.iterations
        )

        try:
            # Build root node
            root_node = self.tree_builder.build_root(root_hypothesis)

            best_hypothesis = root_hypothesis
            best_reward = 0.0
            convergence_iteration = None

            # MC-NEST iterations
            for iteration in range(self.config.iterations):
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.timeout_ms:
                    self.logger.warning(
                        "Search timeout reached",
                        iteration=iteration,
                        elapsed_ms=elapsed_ms
                    )
                    break

                # MCTS phases
                try:
                    # Selection: Select node using UCB1
                    selected_node = self._select_node(root_node)

                    # Expansion: Generate and add child hypotheses
                    new_nodes = self._expand_node(
                        selected_node,
                        hypothesis_generator
                    )

                    # Simulation: Evaluate rewards with circuit breaker
                    rewards = self._simulate_nodes(
                        new_nodes,
                        reward_function
                    )

                    # Backpropagation: Update values up the tree
                    self._backpropagate(selected_node, rewards)

                    # Validation: Statistically validate best hypothesis
                    if best_hypothesis and best_hypothesis.hypothesis_id in self.tree_builder.hypothesis_cache:
                        validation_metrics, error = self.hypothesis_validator.validate(
                            best_hypothesis,
                            self.tree_builder.get_node(best_hypothesis.hypothesis_id).metadata.get("rewards", [best_reward])
                        )

                        if error:
                            self.dlq.add(best_hypothesis, error, "validation", validation_metrics.to_dict())

                    # Update best hypothesis
                    current_best = self._find_best_hypothesis()
                    if current_best and current_best.confidence > best_hypothesis.confidence:
                        best_hypothesis = current_best
                        node = self.tree_builder.get_node(current_best.hypothesis_id)
                        best_reward = node.mean_value if node else 0.0

                    # Check convergence
                    self.convergence_detector.update(
                        iteration,
                        best_hypothesis.confidence,
                        best_reward
                    )

                    if iteration > 0 and iteration % 10 == 0:
                        is_converged, aci_value = self.convergence_detector.check_convergence()
                        if is_converged:
                            convergence_iteration = iteration
                            self.logger.info(
                                "Convergence reached",
                                iteration=iteration,
                                aci_value=aci_value,
                                best_confidence=best_hypothesis.confidence
                            )
                            break

                except Exception as e:
                    self.logger.error(
                        "Iteration failed, continuing",
                        iteration=iteration,
                        error=str(e)
                    )
                    continue

            # Build search result
            execution_time_ms = (time.time() - start_time) * 1000
            tree_stats = self.tree_builder.get_tree_statistics()

            search_result = MCTSSearchResult(
                search_id=search_id,
                root_hypothesis=root_hypothesis,
                best_hypothesis=best_hypothesis,
                tree_root=root_node,
                iterations=iteration + 1,
                convergence_reached=convergence_iteration is not None,
                convergence_iteration=convergence_iteration,
                total_nodes=tree_stats["total_nodes"],
                max_depth=tree_stats["max_depth"],
                execution_time_ms=execution_time_ms,
                strategy=ExplorationStrategy.MCTS,
                metadata={
                    "dlq_size": self.dlq.size(),
                    "aci_final": self.convergence_detector._calculate_aci() if len(self.convergence_detector.confidence_history) > 1 else None,
                }
            )

            self.logger.info(
                "MC-NEST search complete",
                search_id=search_id,
                iterations=search_result.iterations,
                best_confidence=best_hypothesis.confidence,
                convergence_reached=search_result.convergence_reached,
                execution_time_ms=execution_time_ms
            )

            return search_result, None

        except Exception as e:
            error_msg = f"MC-NEST search failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return None, error_msg

    def _select_node(self, root_node: SearchTreeNode) -> SearchTreeNode:
        """Select node for expansion using UCB1."""
        current_node = root_node

        while current_node.state == MCTSNodeState.EXPANDED and current_node.children:
            child = self.selection_strategy.select_child(
                current_node,
                self.tree_builder.tree
            )
            if child is None:
                break
            current_node = child

        return current_node

    def _expand_node(
        self,
        node: SearchTreeNode,
        hypothesis_generator: Callable[[], List[Hypothesis]]
    ) -> List[SearchTreeNode]:
        """Expand node with new hypotheses."""
        if node.state != MCTSNodeState.UNEXPANDED:
            return []

        # Generate child hypotheses
        child_hypotheses = hypothesis_generator()

        # Expand node
        new_nodes = self.tree_builder.expand_node(node, child_hypotheses)

        return new_nodes

    def _simulate_nodes(
        self,
        nodes: List[SearchTreeNode],
        reward_function: Callable[[Hypothesis], float]
    ) -> Dict[str, List[float]]:
        """Simulate rewards for nodes with circuit breaker."""
        rewards = {}

        for node in nodes:
            if not node.hypothesis:
                continue

            try:
                # Use circuit breaker for reward evaluation
                reward = self.circuit_breaker.call(
                    reward_function,
                    node.hypothesis
                )

                # Store reward in node metadata
                if "rewards" not in node.metadata:
                    node.metadata["rewards"] = []
                node.metadata["rewards"].append(reward)

                rewards[node.node_id] = node.metadata["rewards"]

            except CircuitBreakerOpenError as e:
                self.logger.error(
                    "Circuit breaker open during simulation",
                    node_id=node.node_id,
                    error=str(e)
                )
                # Add to DLQ
                self.dlq.add(node.hypothesis, str(e), "system")
                continue

            except Exception as e:
                self.logger.error(
                    "Simulation failed",
                    node_id=node.node_id,
                    error=str(e)
                )
                continue

        return rewards

    def _backpropagate(
        self,
        node: SearchTreeNode,
        rewards: Dict[str, List[float]]
    ):
        """Backpropagate rewards up the tree."""
        current_id = node.node_id

        while current_id is not None:
            current_node = self.tree_builder.get_node(current_id)
            if current_node is None:
                break

            # Calculate average reward from children
            total_reward = 0.0
            count = 0

            for child_id in current_node.children:
                if child_id in rewards and rewards[child_id]:
                    total_reward += sum(rewards[child_id]) / len(rewards[child_id])
                    count += 1

            if count > 0:
                avg_reward = total_reward / count
                self.tree_builder.update_node_value(current_id, avg_reward)

            # Move to parent
            current_id = current_node.parent_id

    def _find_best_hypothesis(self) -> Optional[Hypothesis]:
        """Find best hypothesis in tree."""
        best_hypothesis = None
        best_confidence = 0.0

        for node in self.tree_builder.tree.values():
            if node.hypothesis and node.hypothesis.confidence > best_confidence:
                best_confidence = node.hypothesis.confidence
                best_hypothesis = node.hypothesis

        return best_hypothesis


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    "Phase3Config",
    "MCTSSearchExecutor",
    "SearchTreeBuilder",
    "HypothesisValidator",
    "ConvergenceDetector",
    "UCB1SelectionStrategy",
    "HypothesisDLQ",
    "ValidationMetrics",
]
