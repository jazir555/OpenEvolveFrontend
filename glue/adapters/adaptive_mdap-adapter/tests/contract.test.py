"""
Contract Tests for Adaptive MDAP/MAKER Adapter

Federation Constitution - Section 4: The Proof of Work (The Vibe Check)

Contract tests verify that the API returns the specific fields we rely on.
If the contract is violated (API changed), the adapter refuses to start
to prevent data corruption.

These tests run on container startup. If any test fails, the adapter
will not start, preventing corrupted data from entering the system.
"""

import os
import sys
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import after path manipulation
from adaptive_mdap_adapter import (
    AdaptiveMDAPAdapter,
    AdaptiveMDAPAdapterConfig,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    CanonicalStrategy,
    CanonicalRequest,
    CanonicalResponse,
    ProcessingDomain,
    AdaptationMode,
    TaskStatus,
    CircuitState
)

from maker_adapter import (
    MakerAdapter,
    CanonicalMakerConfig,
    CanonicalMakerStep,
    CanonicalMakerResult,
    VotingMode,
    RedFlagSeverity
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def adapter_config():
    """Create test adapter configuration."""
    return AdaptiveMDAPAdapterConfig(
        timeout_ms=5000,
        max_retries=2,
        retry_delay_ms=100,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout_ms=10000,
        log_level="DEBUG"
    )


@pytest.fixture
def adapter(adapter_config):
    """Create adapter instance with test config."""
    return AdaptiveMDAPAdapter(adapter_config)


@pytest.fixture
def maker_adapter(adapter_config):
    """Create MAKER adapter instance with test config."""
    return MakerAdapter(adapter_config)


@pytest.fixture
def sample_subproblem():
    """Create sample subproblem for testing."""
    return CanonicalSubProblem(
        id="test-sub-001",
        description="Implement a secure authentication system with OAuth2 support",
        domain="security",
        depth=3,
        dependencies=["dep-001", "dep-002"],
        metadata={"priority": "high", "tags": ["auth", "oauth2"]}
    )


@pytest.fixture
def sample_maker_step():
    """Create sample MAKER step for testing."""
    return CanonicalMakerStep(
        step_id="maker-step-001",
        prompt_template="Analyze this: {state}",
        task_type="analysis",
        priority=1,
        system_prompt="You are a security expert",
        expected_schema={"type": "object"},
        metadata={"domain": "security"}
    )


# ============================================================================
# Contract Tests: Adaptive MDAP Adapter
# ============================================================================

class TestAdaptiveMDAPAdapterContracts:
    """Contract tests for Adaptive MDAP Adapter."""

    def test_adapter_initialization_contract(self, adapter):
        """
        CONTRACT: Adapter must initialize with all required attributes
        """
        assert hasattr(adapter, "config"), "Adapter must have config attribute"
        assert hasattr(adapter, "logger"), "Adapter must have logger attribute"
        assert hasattr(adapter, "circuit_breaker"), "Adapter must have circuit_breaker attribute"
        assert hasattr(adapter, "metrics"), "Adapter must have metrics attribute"
        assert adapter.mdap_available in [True, False], "mdap_available must be boolean"

    def test_config_from_env_contract(self):
        """
        CONTRACT: Configuration must fail fast if required env vars missing
        """
        # Save original env
        original_timeout = os.environ.get("ADAPTIVE_MDAP_TIMEOUT_MS")

        try:
            # Remove required env var
            if "ADAPTIVE_MDAP_TIMEOUT_MS" in os.environ:
                del os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"]

            # Should raise AdapterConfigError
            with pytest.raises(Exception) as exc_info:
                AdaptiveMDAPAdapterConfig.from_env()

            assert "ADAPTIVE_MDAP_TIMEOUT_MS" in str(exc_info.value)

        finally:
            # Restore original env
            if original_timeout is not None:
                os.environ["ADAPTIVE_MDAP_TIMEOUT_MS"] = original_timeout

    def test_canonical_subproblem_contract(self, sample_subproblem):
        """
        CONTRACT: CanonicalSubProblem must have specific fields
        """
        assert hasattr(sample_subproblem, "id"), "Must have id field"
        assert hasattr(sample_subproblem, "description"), "Must have description field"
        assert hasattr(sample_subproblem, "domain"), "Must have domain field"
        assert hasattr(sample_subproblem, "depth"), "Must have depth field"
        assert hasattr(sample_subproblem, "dependencies"), "Must have dependencies field"
        assert hasattr(sample_subproblem, "metadata"), "Must have metadata field"

        # Type checks
        assert isinstance(sample_subproblem.id, str), "id must be string"
        assert isinstance(sample_subproblem.description, str), "description must be string"
        assert isinstance(sample_subproblem.domain, str), "domain must be string"
        assert isinstance(sample_subproblem.depth, int), "depth must be integer"
        assert isinstance(sample_subproblem.dependencies, list), "dependencies must be list"
        assert isinstance(sample_subproblem.metadata, dict), "metadata must be dict"

    def test_canonical_complexity_score_contract(self):
        """
        CONTRACT: CanonicalComplexityScore must have specific fields
        """
        score = CanonicalComplexityScore(
            overall_score=0.75,
            text_length_score=0.6,
            dependency_score=0.8,
            depth_score=0.9
        )

        assert hasattr(score, "overall_score"), "Must have overall_score"
        assert hasattr(score, "text_length_score"), "Must have text_length_score"
        assert hasattr(score, "dependency_score"), "Must have dependency_score"
        assert hasattr(score, "depth_score"), "Must have depth_score"
        assert hasattr(score, "timestamp"), "Must have timestamp (UTC ISO-8601)"

        # Type checks
        assert isinstance(score.overall_score, float), "overall_score must be float"
        assert 0 <= score.overall_score <= 1, "overall_score must be between 0 and 1"
        assert isinstance(score.timestamp, str), "timestamp must be string"

        # Verify UTC format
        datetime.fromisoformat(score.timestamp)  # Should not raise

    def test_canonical_strategy_contract(self):
        """
        CONTRACT: CanonicalStrategy must have specific fields
        """
        strategy = CanonicalStrategy(
            strategy="MAKER_ULTRA",
            n_agents=5,
            k_ahead=3,
            max_retries=3,
            timeout_ms=120000
        )

        assert hasattr(strategy, "strategy"), "Must have strategy field"
        assert hasattr(strategy, "n_agents"), "Must have n_agents field"
        assert hasattr(strategy, "k_ahead"), "Must have k_ahead field"
        assert hasattr(strategy, "max_retries"), "Must have max_retries field"
        assert hasattr(strategy, "timeout_ms"), "Must have timeout_ms field"

        # Type checks
        assert isinstance(strategy.strategy, str), "strategy must be string"
        assert isinstance(strategy.n_agents, int), "n_agents must be integer"
        assert isinstance(strategy.k_ahead, int), "k_ahead must be integer"
        assert isinstance(strategy.max_retries, int), "max_retries must be integer"
        assert isinstance(strategy.timeout_ms, int), "timeout_ms must be integer"

    def test_canonical_response_contract(self):
        """
        CONTRACT: CanonicalResponse must have specific fields
        """
        response = CanonicalResponse(
            task_id="task-001",
            status=TaskStatus.COMPLETED,
            complexity_score=CanonicalComplexityScore(overall_score=0.7)
        )

        assert hasattr(response, "task_id"), "Must have task_id"
        assert hasattr(response, "status"), "Must have status"
        assert hasattr(response, "complexity_score"), "Must have complexity_score"
        assert hasattr(response, "strategy"), "Must have strategy"
        assert hasattr(response, "result"), "Must have result"
        assert hasattr(response, "error"), "Must have error"
        assert hasattr(response, "correlation_id"), "Must have correlation_id"
        assert hasattr(response, "timestamp"), "Must have timestamp (UTC ISO-8601)"
        assert hasattr(response, "execution_time_ms"), "Must have execution_time_ms"

        # Type checks
        assert isinstance(response.task_id, str), "task_id must be string"
        assert isinstance(response.status, TaskStatus), "status must be TaskStatus enum"
        assert isinstance(response.timestamp, str), "timestamp must be string"

        # Verify UTC format
        datetime.fromisoformat(response.timestamp)  # Should not raise

    def test_acl_transformation_subproblem_contract(self, adapter, sample_subproblem):
        """
        CONTRACT: ACL must transform external subproblem to canonical format
        """
        # Test with dict input
        dict_input = {
            "id": "dict-001",
            "description": "Test description",
            "domain": "test",
            "depth": 2,
            "dependencies": [],
            "metadata": {}
        }

        result = adapter.to_canonical_subproblem(dict_input)
        assert isinstance(result, CanonicalSubProblem), "Result must be CanonicalSubProblem"
        assert result.id == "dict-001", "ID must be preserved"

    def test_acl_transformation_complexity_contract(self, adapter):
        """
        CONTRACT: ACL must transform external complexity to canonical format
        """
        # Test with dict input
        dict_input = {
            "overall_score": 0.65,
            "text_length_score": 0.7,
            "dependency_score": 0.6
        }

        result = adapter.to_canonical_complexity(dict_input)
        assert isinstance(result, CanonicalComplexityScore), "Result must be CanonicalComplexityScore"
        assert result.overall_score == 0.65, "overall_score must be preserved"

    def test_circuit_breaker_contract(self, adapter):
        """
        CONTRACT: Circuit breaker must prevent cascading failures
        """
        # Initial state should be CLOSED
        assert adapter.circuit_breaker.get_state() == CircuitState.CLOSED

        # Record failures up to threshold
        for _ in range(adapter.config.circuit_breaker_threshold):
            adapter.circuit_breaker.record_failure()

        # Circuit should be OPEN
        assert adapter.circuit_breaker.get_state() == CircuitState.OPEN

        # Requests should be denied
        assert not adapter.circuit_breaker.allow_request()

    def test_health_check_contract(self, adapter):
        """
        CONTRACT: Health check must return specific fields
        """
        health = adapter.health_check()

        assert "status" in health, "Health check must include status"
        assert "circuit_breaker_state" in health, "Health check must include circuit_breaker_state"
        assert "mdap_available" in health, "Health check must include mdap_available"
        assert "metrics" in health, "Health check must include metrics"
        assert "timestamp" in health, "Health check must include timestamp (UTC ISO-8601)"

        # Verify UTC format
        datetime.fromisoformat(health["timestamp"])  # Should not raise


# ============================================================================
# Contract Tests: MAKER Adapter
# ============================================================================

class TestMakerAdapterContracts:
    """Contract tests for MAKER Adapter."""

    def test_maker_adapter_initialization_contract(self, maker_adapter):
        """
        CONTRACT: MAKER adapter must initialize with all required attributes
        """
        assert hasattr(maker_adapter, "config"), "Adapter must have config attribute"
        assert hasattr(maker_adapter, "logger"), "Adapter must have logger attribute"
        assert hasattr(maker_adapter, "circuit_breaker"), "Adapter must have circuit_breaker attribute"
        assert hasattr(maker_adapter, "metrics"), "Adapter must have metrics attribute"
        assert maker_adapter.maker_available in [True, False], "maker_available must be boolean"

    def test_canonical_maker_config_contract(self):
        """
        CONTRACT: CanonicalMakerConfig must have specific fields
        """
        config = CanonicalMakerConfig(
            k_min=2,
            k_max=7,
            max_votes_per_step=50,
            timeout_seconds=90
        )

        assert hasattr(config, "k_min"), "Must have k_min field"
        assert hasattr(config, "k_max"), "Must have k_max field"
        assert hasattr(config, "max_votes_per_step"), "Must have max_votes_per_step field"
        assert hasattr(config, "timeout_seconds"), "Must have timeout_seconds field"

        # Type checks
        assert isinstance(config.k_min, int), "k_min must be integer"
        assert isinstance(config.k_max, int), "k_max must be integer"
        assert isinstance(config.max_votes_per_step, int), "max_votes_per_step must be integer"

    def test_canonical_maker_step_contract(self, sample_maker_step):
        """
        CONTRACT: CanonicalMakerStep must have specific fields
        """
        assert hasattr(sample_maker_step, "step_id"), "Must have step_id"
        assert hasattr(sample_maker_step, "prompt_template"), "Must have prompt_template"
        assert hasattr(sample_maker_step, "task_type"), "Must have task_type"
        assert hasattr(sample_maker_step, "priority"), "Must have priority"
        assert hasattr(sample_maker_step, "system_prompt"), "Must have system_prompt"

        # Type checks
        assert isinstance(sample_maker_step.step_id, str), "step_id must be string"
        assert isinstance(sample_maker_step.prompt_template, str), "prompt_template must be string"

    def test_canonical_maker_result_contract(self):
        """
        CONTRACT: CanonicalMakerResult must have specific fields
        """
        result = CanonicalMakerResult(
            success=True,
            steps_completed=5,
            votes_cast=15,
            red_flags_detected=0
        )

        assert hasattr(result, "success"), "Must have success"
        assert hasattr(result, "steps_completed"), "Must have steps_completed"
        assert hasattr(result, "votes_cast"), "Must have votes_cast"
        assert hasattr(result, "red_flags_detected"), "Must have red_flags_detected"
        assert hasattr(result, "final_action"), "Must have final_action"
        assert hasattr(result, "agent_votes"), "Must have agent_votes"
        assert hasattr(result, "red_flags"), "Must have red_flags"
        assert hasattr(result, "metrics"), "Must have metrics"
        assert hasattr(result, "timestamp"), "Must have timestamp (UTC ISO-8601)"

        # Type checks
        assert isinstance(result.success, bool), "success must be boolean"
        assert isinstance(result.steps_completed, int), "steps_completed must be integer"
        assert isinstance(result.timestamp, str), "timestamp must be string"

        # Verify UTC format
        datetime.fromisoformat(result.timestamp)  # Should not raise


# ============================================================================
# Contract Tests: Integration
# ============================================================================

class TestIntegrationContracts:
    """Contract tests for MDAP/MAKER integration."""

    def test_adapter_correlation_tracking(self, adapter):
        """
        CONTRACT: All operations must track correlation IDs for distributed tracing
        """
        subproblem = CanonicalSubProblem(
            id="corr-test-001",
            description="Test correlation tracking",
            domain="test"
        )

        response = adapter.analyze_complexity(subproblem, correlation_id="test-corr-123")

        # Response must include correlation ID
        assert response.correlation_id == "test-corr-123", "Correlation ID must be preserved"

    def test_utc_timestamps_contract(self, adapter):
        """
        CONTRACT: Law 6 - All timestamps must be in UTC ISO-8601 format
        """
        health = adapter.health_check()

        # Must be parseable as ISO-8601
        dt = datetime.fromisoformat(health["timestamp"])

        # Must be UTC (no timezone info means UTC in our implementation)
        assert dt.tzinfo is None or dt.tzinfo == timezone.utc, "Timestamp must be UTC"

    def test_idempotency_contract(self, adapter, sample_subproblem):
        """
        CONTRACT: Law 4 - All operations must be safe to retry
        """
        # First call
        response1 = adapter.analyze_complexity(sample_subproblem)

        # Second call (should not cause issues)
        response2 = adapter.analyze_complexity(sample_subproblem)

        # Both should succeed
        assert response1.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response2.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    def test_graceful_degradation_contract(self, adapter):
        """
        CONTRACT: Adapter must degrade gracefully if MDAP unavailable
        """
        # Even if MDAP is unavailable, adapter should not crash
        response = adapter.analyze_complexity(
            CanonicalSubProblem(
                id="degrade-test",
                description="Test graceful degradation",
                domain="test"
            )
        )

        # Should return valid response
        assert isinstance(response, CanonicalResponse)
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
