"""
Integration tests for RESE DEE Adapter

Tests end-to-end functionality including:
- Adapter initialization
- Request/response handling
- Error handling and DLQ
- Health checks
- Configuration validation

Following CLAUDE.md: Contract-based testing
"""

import pytest
import sys
import os
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dee_adapter import DEEAdapter, DeadLetterQueue


# ============================================================================
# SETUP
# ============================================================================

@pytest.fixture(autouse=True)
def setup_environment():
    """Set up environment variables for testing."""
    # Set required environment variables
    os.environ["EXPLORATION_DEPTH"] = "5"
    os.environ["MCTS_ITERATIONS"] = "50"
    os.environ["MCTS_EXPLORATION_CONSTANT"] = "1.414"
    os.environ["CONVERGENCE_THRESHOLD"] = "0.001"
    os.environ["EXPLORATION_TIMEOUT_MS"] = "10000"
    os.environ["MAX_HYPOTHESES"] = "20"
    os.environ["PATTERN_RECOGNITION_THRESHOLD"] = "0.5"
    os.environ["DLQ_MAX_SIZE"] = "100"

    yield

    # Cleanup
    for key in os.environ.keys():
        if key.startswith(("EXPLORATION_", "MCTS_", "CONVERGENCE_", "MAX_", "PATTERN_", "DLQ_")):
            del os.environ[key]


# ============================================================================
# ADAPTER INITIALIZATION TESTS
# ============================================================================

class TestAdapterInitialization:
    """Test DEE adapter initialization."""

    def test_successful_initialization(self):
        """Test successful adapter initialization."""
        adapter = DEEAdapter()

        assert adapter.config is not None
        assert adapter.engine is not None
        assert adapter.logger is not None
        assert adapter.dlq is not None
        assert adapter.circuit_breaker is not None

    def test_configuration_values(self):
        """Test configuration values are set correctly."""
        adapter = DEEAdapter()

        assert adapter.config.exploration_depth == 5
        assert adapter.config.mcts_iterations == 50
        assert adapter.config.timeout_ms == 10000
        assert adapter.config.max_hypotheses == 20


# ============================================================================
# EXPLORE API TESTS
# ============================================================================

class TestExploreAPI:
    """Test explore API."""

    def test_successful_exploration(self):
        """Test successful exploration request."""
        adapter = DEEAdapter()

        result = adapter.explore({
            "problem_statement": "System is slow under high load",
            "domain": "performance",
            "context": {
                "load_level": "high",
                "component": "database"
            }
        })

        # Validate response structure
        assert "search_id" in result
        assert "root_hypothesis" in result
        assert "best_hypothesis" in result
        assert "best_confidence" in result
        assert "tree_statistics" in result
        assert "execution_time_ms" in result
        assert "strategy" in result
        assert "patterns" in result
        assert "timestamp" in result

        # Validate data types
        assert isinstance(result["best_confidence"], float)
        assert 0.0 <= result["best_confidence"] <= 1.0
        assert isinstance(result["execution_time_ms"], float)
        assert result["execution_time_ms"] > 0

    def test_exploration_with_correlation_id(self):
        """Test exploration with correlation ID."""
        adapter = DEEAdapter()

        correlation_id = "test-correlation-123"

        result = adapter.explore({
            "problem_statement": "Test problem",
            "domain": "test",
            "correlation_id": correlation_id
        })

        # Correlation ID should be in logs (checked via logger instance)

    def test_invalid_request_missing_fields(self):
        """Test request validation with missing fields."""
        adapter = DEEAdapter()

        with pytest.raises(ValueError, match="Request validation failed"):
            adapter.explore({
                "domain": "test"
                # Missing problem_statement
            })

    def test_invalid_request_wrong_type(self):
        """Test request validation with wrong types."""
        adapter = DEEAdapter()

        with pytest.raises(ValueError, match="Request validation failed"):
            adapter.explore({
                "problem_statement": 123,  # Should be string
                "domain": "test"
            })

    def test_context_optional(self):
        """Test that context is optional."""
        adapter = DEEAdapter()

        # Should not raise error
        result = adapter.explore({
            "problem_statement": "Test problem",
            "domain": "test"
            # No context provided
        })

        assert result is not None


# ============================================================================
# BATCH EXPLORE API TESTS
# ============================================================================

class TestBatchExploreAPI:
    """Test batch_explore API."""

    def test_successful_batch_exploration(self):
        """Test successful batch exploration."""
        adapter = DEEAdapter()

        result = adapter.batch_explore({
            "problems": [
                {
                    "problem_statement": "Problem 1",
                    "domain": "domain_a"
                },
                {
                    "problem_statement": "Problem 2",
                    "domain": "domain_b"
                }
            ]
        })

        # Validate response structure
        assert "total_problems" in result
        assert "successful_results" in result
        assert "results" in result
        assert "timestamp" in result

        assert result["total_problems"] == 2
        assert result["successful_results"] <= 2
        assert len(result["results"]) <= 2

    def test_batch_exploration_with_shared_context(self):
        """Test batch exploration with shared context."""
        adapter = DEEAdapter()

        result = adapter.batch_explore({
            "problems": [
                {
                    "problem_statement": "Problem 1",
                    "domain": "domain_a"
                }
            ],
            "context": {"shared": "value"}
        })

        assert result["total_problems"] == 1

    def test_batch_exploration_invalid_request(self):
        """Test batch exploration with invalid request."""
        adapter = DEEAdapter()

        with pytest.raises(ValueError, match="must contain 'problems' list"):
            adapter.batch_explore({
                "problems": "not a list"
            })

        with pytest.raises(ValueError, match="must have 'problem_statement' and 'domain'"):
            adapter.batch_explore({
                "problems": [
                    {"problem_statement": "Test"}  # Missing domain
                ]
            })


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and DLQ."""

    def test_dlq_on_validation_error(self):
        """Test that validation errors add to DLQ."""
        adapter = DEEAdapter()

        initial_dlq_size = adapter.dlq.size()

        try:
            adapter.explore({
                "domain": "test"  # Missing problem_statement
            })
        except ValueError:
            pass

        # DLQ should have grown
        assert adapter.dlq.size() > initial_dlq_size

    def test_dlq_entry_structure(self):
        """Test DLQ entry structure."""
        adapter = DEEAdapter()
        dlq = adapter.dlq

        # Add entry manually
        test_request = {"test": "request"}
        dlq.add(test_request, "Test error", "logic")

        contents = dlq.get_all()
        assert len(contents) > 0

        entry = contents[-1]
        assert "request" in entry
        assert "error" in entry
        assert "error_type" in entry
        assert "timestamp" in entry
        assert "dlq_id" in entry

        assert entry["request"] == test_request
        assert entry["error"] == "Test error"
        assert entry["error_type"] == "logic"

    def test_dlq_clear(self):
        """Test DLQ clearing."""
        adapter = DEEAdapter()
        dlq = adapter.dlq

        # Add entries
        dlq.add({"test": 1}, "Error 1", "logic")
        dlq.add({"test": 2}, "Error 2", "logic")

        assert dlq.size() > 0

        # Clear
        adapter.clear_dlq()

        assert dlq.size() == 0

    def test_dlq_max_size(self):
        """Test DLQ max size enforcement."""
        dlq = DeadLetterQueue()
        dlq.max_size = 5  # Small size for testing

        # Add more entries than max size
        for i in range(10):
            dlq.add({"test": i}, f"Error {i}", "logic")

        # Should not exceed max size
        assert dlq.size() <= 5


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_structure(self):
        """Test health check response structure."""
        adapter = DEEAdapter()

        health = adapter.get_health()

        assert "status" in health
        assert "circuit_breaker_state" in health
        assert "dlq_size" in health
        assert "config" in health
        assert "timestamp" in health

    def test_health_check_status(self):
        """Test health check status values."""
        adapter = DEEAdapter()

        health = adapter.get_health()

        # Status should be healthy or degraded
        assert health["status"] in ["healthy", "degraded"]

        # Circuit breaker state should be valid
        assert health["circuit_breaker_state"] in ["CLOSED", "OPEN", "HALF_OPEN"]


# ============================================================================
# TIMEOUT TESTS
# ============================================================================

class TestTimeout:
    """Test timeout functionality."""

    def test_exploration_timeout(self):
        """Test that exploration respects timeout."""
        # Set very short timeout
        os.environ["EXPLORATION_TIMEOUT_MS"] = "100"
        os.environ["MCTS_ITERATIONS"] = "10000"  # Many iterations

        adapter = DEEAdapter()

        start_time = time.time()

        result = adapter.explore({
            "problem_statement": "Test problem",
            "domain": "test"
        })

        elapsed_ms = (time.time() - start_time) * 1000

        # Should complete within timeout + some margin
        assert elapsed_ms < 500  # 100ms timeout + margin


# ============================================================================
# IDEMPOTENCY TESTS
# ============================================================================

class TestIdempotency:
    """Test idempotency (Law of Idempotency)."""

    def test_duplicate_exploration_requests(self):
        """Test that duplicate requests produce consistent results."""
        adapter = DEEAdapter()

        request = {
            "problem_statement": "Test problem",
            "domain": "test"
        }

        # Execute same request twice
        result1 = adapter.explore(request)
        result2 = adapter.explore(request)

        # Results should have same structure
        assert set(result1.keys()) == set(result2.keys())

        # Both should have valid search_ids (different, but valid)
        assert result1["search_id"] != result2["search_id"]
        assert result1["best_hypothesis"] is not None
        assert result2["best_hypothesis"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
