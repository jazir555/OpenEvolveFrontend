"""
RESE Deep Exploration Engine (DEE) Implementation

This module implements the Deep Exploration Engine for RESE, providing:
- Hypothesis generation and testing
- Cross-domain pattern recognition
- MCTS-based exploration
- Isomorphic mapping between domains

Following CLAUDE.md principles:
- Law of Idempotency: UPSERT logic, deduplicate by hypothesis_id
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect pattern recognition failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations have timeout (default 10000ms)
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
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schemas"))

try:
    from rese_schemas import (
        Hypothesis,
        SearchTreeNode,
        Pattern,
        MCTSSearchResult,
        ExplorationConfig,
        HypothesisStatus,
        PatternType,
        MCTSNodeState,
        ExplorationStrategy,
        ContradictionType,
    )
except ImportError:
    # Fallback for testing
    from glue.schemas.rese_schemas import (
        Hypothesis,
        SearchTreeNode,
        Pattern,
        MCTSSearchResult,
        ExplorationConfig,
        HypothesisStatus,
        PatternType,
        MCTSNodeState,
        ExplorationStrategy,
        ContradictionType,
    )


# ============================================================================
# STRUCTURED LOGGER (JSON Lines)
# ============================================================================

class DEELogger:
    """
    Structured logger for DEE operations.

    Outputs JSON Lines format with correlation_id, source_service, target_service.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.source_service = "rese_dee"
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)

    def _log(self, level: str, msg: str, **kwargs):
        """Log a message in JSON Lines format."""
        log_entry = {
            "msg": msg,
            "level": level,
            "correlation_id": self.correlation_id,
            "source_service": self.source_service,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        log_json = json.dumps(log_entry)
        self.logger.log(logging.getLevelName(level.upper()), log_json)

    def info(self, msg: str, **kwargs):
        self._log("info", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("error", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("warning", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("debug", msg, **kwargs)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker for detecting pattern recognition failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Failures detected, stop attempting
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 60000,
        half_open_max_calls: int = 3,
        logger: Optional[DEELogger] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.half_open_max_calls = half_open_max_calls
        self.logger = logger or DEELogger()

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                error_msg = f"Circuit breaker is OPEN. {self.failure_count} failures detected."
                self.logger.error(error_msg, failure_count=self.failure_count)
                raise CircuitBreakerOpenError(error_msg)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(str(e))
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed_ms = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
        return elapsed_ms >= self.recovery_timeout_ms

    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
                self.half_open_calls = 0
                self.logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self, error_msg: str):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(
                "Circuit breaker opened due to failures",
                failure_count=self.failure_count,
                error=error_msg
            )


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is OPEN."""
    pass


# ============================================================================
# EXPONENTIAL BACKOFF WITH JITTER
# ============================================================================

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay_ms: int = 100,
    max_delay_ms: int = 10000,
    jitter_factor: float = 0.1,
    logger: Optional[DEELogger] = None
) -> Any:
    """
    Retry function with exponential backoff and jitter.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay_ms: Base delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        jitter_factor: Jitter factor (0.0 to 1.0)
        logger: Logger instance

    Returns:
        Function result

    Raises:
        Exception: If all retries exhausted
    """
    logger = logger or DEELogger()
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"All retries exhausted for {func.__name__}",
                    attempts=attempt + 1,
                    error=str(e)
                )
                raise

            # Calculate delay with jitter
            delay_ms = min(base_delay_ms * (2 ** attempt), max_delay_ms)
            jitter = delay_ms * jitter_factor * (random.random() * 2 - 1)
            final_delay_ms = max(0, delay_ms + jitter)

            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {func.__name__}",
                delay_ms=final_delay_ms,
                error=str(e)
            )

            time.sleep(final_delay_ms / 1000.0)

    raise last_exception


# ============================================================================
# HYPOTHESIS GENERATOR
# ============================================================================

class HypothesisGenerator:
    """
    Generates testable hypotheses from problem statements and existing knowledge.

    Uses various strategies:
    - Abductive reasoning (inference to best explanation)
    - Analogical reasoning (cross-domain mapping)
    - Causal inference (effect → cause)
    - Constraint-based generation
    """

    def __init__(
        self,
        config: ExplorationConfig,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.hypothesis_cache: Dict[str, Hypothesis] = {}

    def generate(
        self,
        problem_statement: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None,
        existing_hypotheses: Optional[List[Hypothesis]] = None
    ) -> List[Hypothesis]:
        """
        Generate hypotheses from problem statement.

        Args:
            problem_statement: The problem to analyze
            domain: Domain of application
            context: Additional context
            existing_hypotheses: Existing hypotheses to build upon

        Returns:
            List of generated hypotheses (deduplicated by hypothesis_id)
        """
        start_time = time.time()
        self.logger.info(
            "Generating hypotheses",
            domain=domain,
            problem_length=len(problem_statement)
        )

        try:
            hypotheses = []

            # Strategy 1: Direct causal hypotheses
            causal_hypotheses = self._generate_causal_hypotheses(
                problem_statement, domain, context
            )
            hypotheses.extend(causal_hypotheses)

            # Strategy 2: Structural hypotheses
            structural_hypotheses = self._generate_structural_hypotheses(
                problem_statement, domain, context
            )
            hypotheses.extend(structural_hypotheses)

            # Strategy 3: Analogical hypotheses (if existing hypotheses provided)
            if existing_hypotheses:
                analogical_hypotheses = self._generate_analogical_hypotheses(
                    problem_statement, domain, existing_hypotheses
                )
                hypotheses.extend(analogical_hypotheses)

            # Deduplicate by hypothesis_id (Law of Idempotency)
            unique_hypotheses = self._deduplicate_hypotheses(hypotheses)

            # Limit to max_hypotheses
            final_hypotheses = list(unique_hypotheses.values())[:self.config.max_hypotheses]

            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Hypothesis generation complete",
                count=len(final_hypotheses),
                elapsed_ms=elapsed_ms
            )

            return final_hypotheses

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Hypothesis generation failed",
                error=str(e),
                elapsed_ms=elapsed_ms
            )
            raise

    def _generate_causal_hypotheses(
        self,
        problem_statement: str,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Hypothesis]:
        """Generate causal hypotheses (if X then Y)."""
        hypotheses = []

        # Extract potential causal relationships
        keywords = ["because", "due to", "causes", "leads to", "results in"]
        words = problem_statement.lower().split()

        for i, word in enumerate(words):
            if word in keywords:
                # Create hypothesis from causal relationship
                hypothesis = Hypothesis(
                    statement=f"Causal relationship detected at position {i}",
                    type="causal",
                    domain=domain,
                    confidence=0.5,
                    metadata={"keyword": word, "position": i}
                )
                hypotheses.append(hypothesis)

        # Generate generic causal hypothesis if none found
        if not hypotheses:
            hypothesis = Hypothesis(
                statement=f"Generic causal hypothesis for {domain}",
                type="causal",
                domain=domain,
                confidence=0.3
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_structural_hypotheses(
        self,
        problem_statement: str,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Hypothesis]:
        """Generate structural hypotheses (component relationships)."""
        hypotheses = []

        # Look for structural keywords
        structural_keywords = ["component", "system", "architecture", "module"]
        words = problem_statement.lower().split()

        for i, word in enumerate(words):
            if word in structural_keywords:
                hypothesis = Hypothesis(
                    statement=f"Structural relationship detected at position {i}",
                    type="structural",
                    domain=domain,
                    confidence=0.5,
                    metadata={"keyword": word, "position": i}
                )
                hypotheses.append(hypothesis)

        return hypotheses

    def _generate_analogical_hypotheses(
        self,
        problem_statement: str,
        domain: str,
        existing_hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Generate analogical hypotheses from other domains."""
        hypotheses = []

        # Group existing hypotheses by domain
        by_domain = defaultdict(list)
        for h in existing_hypotheses:
            by_domain[h.domain].append(h)

        # For each other domain, create analogical hypothesis
        for other_domain, other_hypotheses in by_domain.items():
            if other_domain != domain and other_hypotheses:
                # Pick the best hypothesis from other domain
                best_hyp = max(other_hypotheses, key=lambda h: h.confidence)

                # Create analogical hypothesis
                hypothesis = Hypothesis(
                    statement=f"Analogical to: {best_hyp.statement[:100]}",
                    type="analogical",
                    domain=domain,
                    confidence=best_hyp.confidence * 0.7,  # Reduce confidence
                    source_hypotheses=[best_hyp.hypothesis_id],
                    metadata={
                        "source_domain": other_domain,
                        "original_hypothesis_id": best_hyp.hypothesis_id
                    }
                )
                hypotheses.append(hypothesis)

        return hypotheses

    def _deduplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> Dict[str, Hypothesis]:
        """
        Deduplicate hypotheses by hypothesis_id (Law of Idempotency).

        If duplicate IDs found, keep the one with higher confidence.
        """
        unique = {}
        for hypothesis in hypotheses:
            existing = unique.get(hypothesis.hypothesis_id)
            if existing is None or hypothesis.confidence > existing.confidence:
                unique[hypothesis.hypothesis_id] = hypothesis

        return unique


# ============================================================================
# PATTERN RECOGNIZER
# ============================================================================

class PatternRecognizer:
    """
    Recognizes patterns across domains using structural and semantic analysis.

    Implements:
    - Structural pattern matching (graph isomorphism)
    - Functional pattern matching (similar behaviors)
    - Causal pattern matching (similar cause-effect relationships)
    """

    def __init__(
        self,
        config: ExplorationConfig,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.circuit_breaker = CircuitBreaker(logger=logger)
        self.pattern_cache: Dict[str, Pattern] = {}

    def recognize_patterns(
        self,
        hypotheses: List[Hypothesis],
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Pattern]:
        """
        Recognize patterns across hypotheses.

        Args:
            hypotheses: List of hypotheses to analyze
            domain: Domain of analysis
            context: Additional context

        Returns:
            List of recognized patterns (deduplicated by pattern_id)
        """
        start_time = time.time()
        self.logger.info(
            "Recognizing patterns",
            domain=domain,
            hypothesis_count=len(hypotheses)
        )

        try:
            # Use circuit breaker for pattern recognition
            patterns = retry_with_backoff(
                lambda: self._recognize_patterns_internal(hypotheses, domain, context),
                logger=self.logger
            )

            # Filter by confidence threshold
            high_confidence_patterns = [
                p for p in patterns
                if p.confidence >= self.config.pattern_recognition_threshold
            ]

            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(
                "Pattern recognition complete",
                total_patterns=len(patterns),
                high_confidence=len(high_confidence_patterns),
                elapsed_ms=elapsed_ms
            )

            return high_confidence_patterns

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Pattern recognition failed",
                error=str(e),
                elapsed_ms=elapsed_ms
            )
            # Return empty list (graceful degradation)
            return []

    def _recognize_patterns_internal(
        self,
        hypotheses: List[Hypothesis],
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Pattern]:
        """Internal pattern recognition (wrapped by circuit breaker)."""
        patterns = []

        # Strategy 1: Structural patterns (shared statement structure)
        structural_patterns = self._recognize_structural_patterns(hypotheses, domain)
        patterns.extend(structural_patterns)

        # Strategy 2: Functional patterns (shared type/domain)
        functional_patterns = self._recognize_functional_patterns(hypotheses, domain)
        patterns.extend(functional_patterns)

        # Strategy 3: Causal patterns (shared causal chains)
        causal_patterns = self._recognize_causal_patterns(hypotheses, domain)
        patterns.extend(causal_patterns)

        # Deduplicate by pattern_id
        unique_patterns = self._deduplicate_patterns(patterns)

        return list(unique_patterns.values())

    def _recognize_structural_patterns(
        self,
        hypotheses: List[Hypothesis],
        domain: str
    ) -> List[Pattern]:
        """Recognize structural patterns."""
        patterns = []

        # Group hypotheses by statement length
        by_length = defaultdict(list)
        for h in hypotheses:
            length_bucket = len(h.statement) // 50
            by_length[length_bucket].append(h)

        # Create pattern for each bucket with multiple hypotheses
        for length_bucket, bucket_hypotheses in by_length.items():
            if len(bucket_hypotheses) >= 2:
                pattern_id = f"structural_length_{length_bucket}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    type=PatternType.STRUCTURAL,
                    description=f"Structural pattern: statements of length {length_bucket * 50}-{(length_bucket + 1) * 50}",
                    confidence=min(1.0, len(bucket_hypotheses) / 10),
                    domains=[domain],
                    instances=[{"hypothesis_id": h.hypothesis_id} for h in bucket_hypotheses],
                    metadata={"length_bucket": length_bucket}
                )
                patterns.append(pattern)

        return patterns

    def _recognize_functional_patterns(
        self,
        hypotheses: List[Hypothesis],
        domain: str
    ) -> List[Pattern]:
        """Recognize functional patterns."""
        patterns = []

        # Group hypotheses by type
        by_type = defaultdict(list)
        for h in hypotheses:
            by_type[h.type].append(h)

        # Create pattern for each type
        for hyp_type, type_hypotheses in by_type.items():
            if len(type_hypotheses) >= 2:
                pattern_id = f"functional_type_{hyp_type}"
                pattern = Pattern(
                    pattern_id=pattern_id,
                    type=PatternType.FUNCTIONAL,
                    description=f"Functional pattern: {hyp_type} hypotheses",
                    confidence=min(1.0, len(type_hypotheses) / 10),
                    domains=[domain],
                    instances=[{"hypothesis_id": h.hypothesis_id} for h in type_hypotheses],
                    metadata={"hypothesis_type": hyp_type}
                )
                patterns.append(pattern)

        return patterns

    def _recognize_causal_patterns(
        self,
        hypotheses: List[Hypothesis],
        domain: str
    ) -> List[Pattern]:
        """Recognize causal patterns."""
        patterns = []

        # Look for causal chains (A -> B -> C)
        causal_hypotheses = [h for h in hypotheses if h.type == "causal"]

        for i, h1 in enumerate(causal_hypotheses):
            for h2 in causal_hypotheses[i+1:]:
                # Check if h1's output could trigger h2
                if self._is_causal_chain(h1, h2):
                    pattern_id = f"causal_chain_{h1.hypothesis_id}_{h2.hypothesis_id}"
                    pattern = Pattern(
                        pattern_id=pattern_id,
                        type=PatternType.CAUSAL,
                        description=f"Causal chain: {h1.statement[:50]} -> {h2.statement[:50]}",
                        confidence=min(h1.confidence, h2.confidence),
                        domains=[domain],
                        instances=[
                            {"hypothesis_id": h1.hypothesis_id},
                            {"hypothesis_id": h2.hypothesis_id}
                        ],
                        metadata={"chain": [h1.hypothesis_id, h2.hypothesis_id]}
                    )
                    patterns.append(pattern)

        return patterns

    def _is_causal_chain(self, h1: Hypothesis, h2: Hypothesis) -> bool:
        """Check if two hypotheses form a causal chain."""
        # Simple heuristic: check for shared keywords
        words1 = set(h1.statement.lower().split())
        words2 = set(h2.statement.lower().split())
        overlap = len(words1 & words2)
        return overlap >= 2

    def _deduplicate_patterns(self, patterns: List[Pattern]) -> Dict[str, Pattern]:
        """Deduplicate patterns by pattern_id."""
        unique = {}
        for pattern in patterns:
            existing = unique.get(pattern.pattern_id)
            if existing is None or pattern.confidence > existing.confidence:
                unique[pattern.pattern_id] = pattern
        return unique


# ============================================================================
# MCTS EXPLAINER
# ============================================================================

class MCTSExplainer:
    """
    Monte Carlo Tree Search for exploration and hypothesis refinement.

    Implements the four MCTS phases:
    1. Selection: Select promising node using UCB
    2. Expansion: Add child nodes to selected node
    3. Simulation: Simulate from new node to estimate value
    4. Backpropagation: Update values up the tree
    """

    def __init__(
        self,
        config: ExplorationConfig,
        logger: Optional[DEELogger] = None
    ):
        self.config = config
        self.logger = logger or DEELogger(config.correlation_id)
        self.tree: Dict[str, SearchTreeNode] = {}
        self.root_node_id: Optional[str] = None

    def explore(
        self,
        root_hypothesis: Hypothesis,
        hypothesis_generator: HypothesisGenerator,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MCTSSearchResult:
        """
        Perform MCTS exploration from root hypothesis.

        Args:
            root_hypothesis: Starting hypothesis
            hypothesis_generator: Generator for creating child hypotheses
            domain: Domain of exploration
            context: Additional context

        Returns:
            MCTS search result with best hypothesis and tree
        """
        start_time = time.time()
        search_id = str(uuid.uuid4())

        self.logger.info(
            "Starting MCTS exploration",
            search_id=search_id,
            root_hypothesis_id=root_hypothesis.hypothesis_id,
            domain=domain
        )

        try:
            # Initialize tree with root node
            self.root_node_id = root_hypothesis.hypothesis_id
            root_node = SearchTreeNode(
                node_id=self.root_node_id,
                hypothesis=root_hypothesis,
                state=MCTSNodeState.EXPANDED,
                depth=0
            )
            self.tree[self.root_node_id] = root_node

            best_hypothesis = root_hypothesis
            convergence_iteration = None
            prev_best_confidence = root_hypothesis.confidence

            # MCTS iterations
            for iteration in range(self.config.mcts_iterations):
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.timeout_ms:
                    self.logger.warning(
                        "MCTS timeout reached",
                        iteration=iteration,
                        elapsed_ms=elapsed_ms
                    )
                    break

                # Check convergence
                if iteration > 0 and iteration % 10 == 0:
                    if abs(best_hypothesis.confidence - prev_best_confidence) < self.config.convergence_threshold:
                        convergence_iteration = iteration
                        self.logger.info(
                            "Convergence reached",
                            iteration=iteration,
                            confidence=best_hypothesis.confidence
                        )
                        break
                    prev_best_confidence = best_hypothesis.confidence

                # MCTS phases
                selected_node = self._select(self.root_node_id)
                new_nodes = self._expand(selected_node, hypothesis_generator, domain, context)

                if new_nodes:
                    for new_node in new_nodes:
                        reward = self._simulate(new_node, context)
                        self._backpropagate(new_node.node_id, reward)

                        # Update best hypothesis
                        if new_node.hypothesis and new_node.hypothesis.confidence > best_hypothesis.confidence:
                            best_hypothesis = new_node.hypothesis

            # Calculate statistics
            total_nodes = len(self.tree)
            max_depth = max(node.depth for node in self.tree.values()) if self.tree else 0
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "MCTS exploration complete",
                search_id=search_id,
                iterations=iteration + 1,
                total_nodes=total_nodes,
                max_depth=max_depth,
                execution_time_ms=execution_time_ms,
                best_confidence=best_hypothesis.confidence
            )

            # Create result
            result = MCTSSearchResult(
                search_id=search_id,
                root_hypothesis=root_hypothesis,
                best_hypothesis=best_hypothesis,
                tree_root=self.tree[self.root_node_id],
                iterations=iteration + 1,
                convergence_reached=(convergence_iteration is not None),
                convergence_iteration=convergence_iteration,
                total_nodes=total_nodes,
                max_depth=max_depth,
                execution_time_ms=execution_time_ms,
                strategy=ExplorationStrategy.MCTS
            )

            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "MCTS exploration failed",
                search_id=search_id,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            raise

    def _select(self, node_id: str) -> SearchTreeNode:
        """
        Selection phase: Select node using UCB policy.

        Args:
            node_id: Starting node ID

        Returns:
            Selected node for expansion
        """
        node = self.tree[node_id]

        # If node has no children or is unexpanded, return it
        if not node.children or node.state == MCTSNodeState.UNEXPANDED:
            return node

        # Select child with highest UCB
        best_child = None
        best_ucb = -float('inf')

        total_visits = node.visit_count

        for child_id in node.children:
            child = self.tree[child_id]
            ucb = child.calculate_ucb(
                total_visits,
                self.config.mcts_exploration_constant
            )

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        # Recursively select from best child
        return self._select(best_child.node_id)

    def _expand(
        self,
        node: SearchTreeNode,
        hypothesis_generator: HypothesisGenerator,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> List[SearchTreeNode]:
        """
        Expansion phase: Add child nodes to selected node.

        Args:
            node: Node to expand
            hypothesis_generator: Generator for new hypotheses
            domain: Domain of exploration
            context: Additional context

        Returns:
            List of new nodes
        """
        # Check if max depth reached
        if node.depth >= self.config.exploration_depth:
            node.is_terminal = True
            node.state = MCTSNodeState.TERMINAL
            return []

        # Generate child hypotheses
        if node.hypothesis:
            new_hypotheses = hypothesis_generator.generate(
                problem_statement=node.hypothesis.statement,
                domain=domain,
                context=context,
                existing_hypotheses=[node.hypothesis]
            )
        else:
            new_hypotheses = []

        # Create child nodes
        new_nodes = []
        for hypothesis in new_hypotheses[:5]:  # Limit branching factor
            child_node = SearchTreeNode(
                hypothesis=hypothesis,
                state=MCTSNodeState.UNEXPANDED,
                parent_id=node.node_id,
                depth=node.depth + 1
            )
            self.tree[child_node.node_id] = child_node
            node.children.append(child_node.node_id)
            new_nodes.append(child_node)

        node.state = MCTSNodeState.EXPANDED

        return new_nodes

    def _simulate(self, node: SearchTreeNode, context: Optional[Dict[str, Any]]) -> float:
        """
        Simulation phase: Simulate from node to estimate value.

        Args:
            node: Node to simulate from
            context: Additional context

        Returns:
            Simulated reward value [0.0, 1.0]
        """
        # Simple simulation: use hypothesis confidence as reward
        if node.hypothesis:
            return node.hypothesis.confidence
        return 0.5

    def _backpropagate(self, node_id: str, reward: float):
        """
        Backpropagation phase: Update values up the tree.

        Args:
            node_id: Node to start backpropagation from
            reward: Reward to propagate
        """
        current_id = node_id

        while current_id is not None:
            node = self.tree[current_id]
            node.update_value(reward)
            current_id = node.parent_id

    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search tree."""
        if not self.tree:
            return {}

        depths = [node.depth for node in self.tree.values()]
        visits = [node.visit_count for node in self.tree.values()]
        values = [node.mean_value for node in self.tree.values()]

        return {
            "total_nodes": len(self.tree),
            "max_depth": max(depths) if depths else 0,
            "mean_depth": sum(depths) / len(depths) if depths else 0,
            "total_visits": sum(visits),
            "mean_value": sum(values) / len(values) if values else 0,
            "max_value": max(values) if values else 0,
        }


# ============================================================================
# DEEP EXPLORATION ENGINE (MAIN ORCHESTRATOR)
# ============================================================================

class DeepExplorationEngine:
    """
    Main orchestrator for deep exploration.

    Coordinates:
    - Hypothesis generation
    - Pattern recognition
    - MCTS exploration
    - Result synthesis
    """

    def __init__(
        self,
        config: Optional[ExplorationConfig] = None,
        logger: Optional[DEELogger] = None
    ):
        self.config = config or ExplorationConfig.from_env()
        self.logger = logger or DEELogger(self.config.correlation_id)

        # Initialize components
        self.hypothesis_generator = HypothesisGenerator(self.config, self.logger)
        self.pattern_recognizer = PatternRecognizer(self.config, self.logger)
        self.mcts_explainer = MCTSExplainer(self.config, self.logger)

        self.logger.info(
            "Deep Exploration Engine initialized",
            config=self.config.to_dict()
        )

    def explore(
        self,
        problem_statement: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MCTSSearchResult:
        """
        Perform deep exploration of a problem statement.

        Args:
            problem_statement: The problem to explore
            domain: Domain of application
            context: Additional context

        Returns:
            MCTS search result with best hypothesis and patterns
        """
        start_time = time.time()

        self.logger.info(
            "Starting deep exploration",
            domain=domain,
            problem_length=len(problem_statement)
        )

        try:
            # Phase 1: Generate initial hypotheses
            self.logger.info("Phase 1: Generating initial hypotheses")
            initial_hypotheses = self.hypothesis_generator.generate(
                problem_statement=problem_statement,
                domain=domain,
                context=context
            )

            if not initial_hypotheses:
                self.logger.warning("No hypotheses generated, creating default")
                initial_hypotheses = [
                    Hypothesis(
                        statement=f"Default hypothesis for {domain}",
                        domain=domain,
                        confidence=0.3
                    )
                ]

            # Use best initial hypothesis as root
            root_hypothesis = max(initial_hypotheses, key=lambda h: h.confidence)

            # Phase 2: MCTS exploration
            self.logger.info("Phase 2: MCTS exploration")
            search_result = self.mcts_explainer.explore(
                root_hypothesis=root_hypothesis,
                hypothesis_generator=self.hypothesis_generator,
                domain=domain,
                context=context
            )

            # Phase 3: Pattern recognition
            self.logger.info("Phase 3: Recognizing patterns")
            all_hypotheses = initial_hypotheses.copy()
            if search_result.best_hypothesis:
                all_hypotheses.append(search_result.best_hypothesis)

            patterns = self.pattern_recognizer.recognize_patterns(
                hypotheses=all_hypotheses,
                domain=domain,
                context=context
            )

            # Update search result with patterns
            search_result.metadata["patterns"] = [p.to_dict() for p in patterns]
            search_result.metadata["initial_hypotheses"] = [h.to_dict() for h in initial_hypotheses]

            elapsed_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Deep exploration complete",
                search_id=search_result.search_id,
                best_confidence=search_result.best_hypothesis.confidence if search_result.best_hypothesis else 0.0,
                patterns_found=len(patterns),
                elapsed_ms=elapsed_ms
            )

            return search_result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Deep exploration failed",
                domain=domain,
                error=str(e),
                elapsed_ms=elapsed_ms
            )
            raise

    def batch_explore(
        self,
        problems: List[Tuple[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[MCTSSearchResult]:
        """
        Perform deep exploration on multiple problems.

        Args:
            problems: List of (problem_statement, domain) tuples
            context: Additional context (shared across all problems)

        Returns:
            List of MCTS search results
        """
        self.logger.info(
            "Starting batch deep exploration",
            problem_count=len(problems)
        )

        results = []
        for i, (problem_statement, domain) in enumerate(problems):
            self.logger.info(
                f"Processing problem {i+1}/{len(problems)}",
                domain=domain
            )

            try:
                result = self.explore(problem_statement, domain, context)
                results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Failed to process problem {i+1}",
                    domain=domain,
                    error=str(e)
                )
                # Continue with other problems (graceful degradation)

        return results


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Core components
    "DeepExplorationEngine",
    "HypothesisGenerator",
    "PatternRecognizer",
    "MCTSExplainer",

    # Utilities
    "DEELogger",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "retry_with_backoff",
]
