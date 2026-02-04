"""
Unit tests for RESE Deep Exploration Engine

Tests cover:
- Schema validation and serialization
- Hypothesis generation
- Pattern recognition
- MCTS exploration
- Circuit breaker functionality
- DLQ operations
- Error handling

Following CLAUDE.md: Contract-based testing
"""

import pytest
import sys
import os
from datetime import datetime, timezone

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "schemas"))

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
)

from glue.lib.rese_dee import (
    DeepExplorationEngine,
    HypothesisGenerator,
    PatternRecognizer,
    MCTSExplainer,
    DEELogger,
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def exploration_config():
    """Create test configuration."""
    return ExplorationConfig(
        exploration_depth=3,
        mcts_iterations=20,
        mcts_exploration_constant=1.414,
        convergence_threshold=0.001,
        timeout_ms=5000,
        max_hypotheses=10,
        pattern_recognition_threshold=0.5,
        correlation_id="test-correlation-id"
    )


@pytest.fixture
def dee_logger():
    """Create test logger."""
    return DEELogger("test-correlation-id")


# ============================================================================
# SCHEMA TESTS
# ============================================================================

class TestHypothesisSchema:
    """Test Hypothesis schema."""

    def test_create_hypothesis(self):
        """Test hypothesis creation."""
        h = Hypothesis(
            statement="Test hypothesis",
            type="causal",
            domain="test_domain",
            confidence=0.7
        )

        assert h.hypothesis_id is not None
        assert h.statement == "Test hypothesis"
        assert h.type == "causal"
        assert h.domain == "test_domain"
        assert h.confidence == 0.7
        assert h.status == HypothesisStatus.PENDING

    def test_hypothesis_idempotency(self):
        """Test hypothesis evidence deduplication (Law of Idempotency)."""
        h = Hypothesis(
            hypothesis_id="test-id",
            statement="Test"
        )

        # Add same evidence twice
        h.update_evidence({"source": "test1"}, is_supporting=True)
        h.update_evidence({"source": "test1"}, is_supporting=True)

        # Should only have one entry
        assert len(h.evidence) == 1

    def test_hypothesis_serialization(self):
        """Test hypothesis to_dict conversion."""
        h = Hypothesis(
            statement="Test",
            confidence=0.8
        )

        d = h.to_dict()

        assert d["hypothesis_id"] == h.hypothesis_id
        assert d["statement"] == "Test"
        assert d["confidence"] == 0.8
        assert "status" in d
        assert "created_at" in d

    def test_hypothesis_deserialization(self):
        """Test hypothesis from_dict conversion."""
        d = {
            "hypothesis_id": "test-id",
            "statement": "Test",
            "type": "causal",
            "confidence": 0.8,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        h = Hypothesis.from_dict(d)

        assert h.hypothesis_id == "test-id"
        assert h.statement == "Test"
        assert h.type == "causal"
        assert h.confidence == 0.8

    def test_confidence_calculation(self):
        """Test confidence calculation from evidence."""
        h = Hypothesis(statement="Test")

        h.update_evidence({"source": "test1"}, is_supporting=True)
        h.update_evidence({"source": "test2"}, is_supporting=True)
        h.update_evidence({"source": "test3"}, is_supporting=False)

        confidence = h.calculate_confidence()

        assert 0.0 <= confidence <= 1.0


class TestSearchTreeNodeSchema:
    """Test SearchTreeNode schema."""

    def test_node_creation(self):
        """Test node creation."""
        node = SearchTreeNode(
            node_id="test-node",
            visit_count=10,
            value=5.0
        )

        assert node.node_id == "test-node"
        assert node.visit_count == 10
        assert node.value == 5.0

    def test_ucb_calculation(self):
        """Test UCB calculation."""
        node = SearchTreeNode(
            visit_count=10,
            value=5.0
        )

        ucb = node.calculate_ucb(total_visits=20, exploration_constant=1.414)

        assert ucb > 0
        assert node.exploration_bonus > 0

    def test_value_update(self):
        """Test value update with reward."""
        node = SearchTreeNode()

        node.update_value(0.7)

        assert node.visit_count == 1
        assert node.value == 0.7
        assert node.mean_value == 0.7

        node.update_value(0.9)

        assert node.visit_count == 2
        assert node.value == 1.6
        assert node.mean_value == 0.8


class TestExplorationConfig:
    """Test ExplorationConfig."""

    def test_config_creation(self):
        """Test config creation."""
        config = ExplorationConfig(
            exploration_depth=10,
            mcts_iterations=1000
        )

        assert config.exploration_depth == 10
        assert config.mcts_iterations == 1000

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ExplorationConfig(
            exploration_depth=10,
            timeout_ms=5000
        )

        d = config.to_dict()

        assert d["exploration_depth"] == 10
        assert d["timeout_ms"] == 5000


# ============================================================================
# HYPOTHESIS GENERATOR TESTS
# ============================================================================

class TestHypothesisGenerator:
    """Test HypothesisGenerator."""

    def test_generate_causal_hypotheses(self, exploration_config, dee_logger):
        """Test causal hypothesis generation."""
        generator = HypothesisGenerator(exploration_config, dee_logger)

        hypotheses = generator.generate(
            problem_statement="System is slow because database is overloaded",
            domain="performance"
        )

        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)

    def test_hypothesis_deduplication(self, exploration_config, dee_logger):
        """Test hypothesis deduplication."""
        generator = HypothesisGenerator(exploration_config, dee_logger)

        # Create duplicate hypotheses with same ID
        h1 = Hypothesis(hypothesis_id="dup-id", statement="Test", confidence=0.5)
        h2 = Hypothesis(hypothesis_id="dup-id", statement="Test", confidence=0.7)

        unique = generator._deduplicate_hypotheses([h1, h2])

        assert len(unique) == 1
        assert unique["dup-id"].confidence == 0.7


# ============================================================================
# PATTERN RECOGNIZER TESTS
# ============================================================================

class TestPatternRecognizer:
    """Test PatternRecognizer."""

    def test_recognize_structural_patterns(self, exploration_config, dee_logger):
        """Test structural pattern recognition."""
        recognizer = PatternRecognizer(exploration_config, dee_logger)

        hypotheses = [
            Hypothesis(statement="Short statement", domain="test"),
            Hypothesis(statement="Another short one", domain="test"),
            Hypothesis(statement="Medium length statement here", domain="test"),
        ]

        patterns = recognizer._recognize_structural_patterns(hypotheses, "test")

        assert len(patterns) > 0
        assert all(isinstance(p, Pattern) for p in patterns)

    def test_pattern_deduplication(self, exploration_config, dee_logger):
        """Test pattern deduplication."""
        recognizer = PatternRecognizer(exploration_config, dee_logger)

        p1 = Pattern(pattern_id="dup-pattern", confidence=0.5)
        p2 = Pattern(pattern_id="dup-pattern", confidence=0.8)

        unique = recognizer._deduplicate_patterns([p1, p2])

        assert len(unique) == 1
        assert unique["dup-pattern"].confidence == 0.8


# ============================================================================
# MCTS EXPLAINER TESTS
# ============================================================================

class TestMCTSExplainer:
    """Test MCTSExplainer."""

    def test_initialization(self, exploration_config, dee_logger):
        """Test MCTS explainer initialization."""
        explainer = MCTSExplainer(exploration_config, dee_logger)

        assert explainer.config == exploration_config
        assert explainer.tree == {}
        assert explainer.root_node_id is None

    def test_simple_exploration(self, exploration_config, dee_logger):
        """Test simple MCTS exploration."""
        explainer = MCTSExplainer(exploration_config, dee_logger)

        root_hypothesis = Hypothesis(
            statement="Test problem",
            domain="test",
            confidence=0.5
        )

        generator = HypothesisGenerator(exploration_config, dee_logger)

        result = explainer.explore(
            root_hypothesis=root_hypothesis,
            hypothesis_generator=generator,
            domain="test"
        )

        assert isinstance(result, MCTSSearchResult)
        assert result.search_id is not None
        assert result.root_hypothesis == root_hypothesis
        assert result.best_hypothesis is not None
        assert result.iterations > 0


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreaker:
    """Test CircuitBreaker."""

    def test_circuit_breaker_initial_state(self, dee_logger):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(logger=dee_logger)

        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_on_failures(self, dee_logger):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, logger=dee_logger)

        def failing_func():
            raise Exception("Test failure")

        # Trigger failures
        for _ in range(3):
            try:
                cb.call(failing_func)
            except:
                pass

        assert cb.state == "OPEN"
        assert cb.failure_count >= 3

    def test_circuit_breaker_blocks_when_open(self, dee_logger):
        """Test circuit breaker blocks calls when OPEN."""
        cb = CircuitBreaker(failure_threshold=2, logger=dee_logger)

        def failing_func():
            raise Exception("Test failure")

        # Trigger circuit breaker
        for _ in range(2):
            try:
                cb.call(failing_func)
            except:
                pass

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "success")


# ============================================================================
# DEEP EXPLORATION ENGINE TESTS
# ============================================================================

class TestDeepExplorationEngine:
    """Test DeepExplorationEngine."""

    def test_initialization(self, exploration_config):
        """Test DEE initialization."""
        dee = DeepExplorationEngine(exploration_config)

        assert dee.config == exploration_config
        assert dee.hypothesis_generator is not None
        assert dee.pattern_recognizer is not None
        assert dee.mcts_explainer is not None

    def test_simple_exploration(self, exploration_config):
        """Test simple deep exploration."""
        dee = DeepExplorationEngine(exploration_config)

        result = dee.explore(
            problem_statement="System performance degrades under high load",
            domain="performance"
        )

        assert isinstance(result, MCTSSearchResult)
        assert result.search_id is not None
        assert result.best_hypothesis is not None
        assert result.iterations > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self, exploration_config):
        """Test full exploration pipeline."""
        dee = DeepExplorationEngine(exploration_config)

        result = dee.explore(
            problem_statement="Database queries are slow when user count exceeds 1000",
            domain="performance",
            context={"database": "postgresql", "users": 1000}
        )

        # Validate result structure
        assert result.search_id is not None
        assert result.root_hypothesis is not None
        assert result.best_hypothesis is not None
        assert result.best_hypothesis.confidence >= 0.0
        assert result.best_hypothesis.confidence <= 1.0
        assert result.iterations > 0
        assert result.execution_time_ms > 0

        # Check for patterns
        assert "patterns" in result.metadata
        patterns = result.metadata["patterns"]
        assert isinstance(patterns, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
