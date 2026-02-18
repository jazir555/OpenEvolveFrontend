"""
Integration Tests for Adaptive MDAP/MAKER Adapter

Tests the full integration between the adapter, core MDAP/MAKER modules,
and the BubbleLab API client.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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

from bubblelab_api_client import (
    BubbleLabAPIClient,
    BubbleLabAPIClientConfig,
    BubbleLabAPIResponseError,
    BubbleLabAPIConnectionError
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
    """Create adapter instance."""
    return AdaptiveMDAPAdapter(adapter_config)


@pytest.fixture
def maker_adapter(adapter_config):
    """Create MAKER adapter instance."""
    return MakerAdapter(adapter_config)


@pytest.fixture
def bubblelab_config():
    """Create BubbleLab API client config."""
    return BubbleLabAPIClientConfig(
        api_url="http://localhost:8000",
        api_key="test-key",
        timeout_ms=5000,
        max_retries=2
    )


@pytest.fixture
def bubblelab_client(bubblelab_config):
    """Create BubbleLab API client."""
    return BubbleLabAPIClient(bubblelab_config)


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestAdaptiveMDAPIntegration:
    """End-to-end integration tests for Adaptive MDAP adapter."""

    def test_full_complexity_analysis_workflow(self, adapter):
        """
        Test complete workflow: subproblem → analysis → allocation → result
        """
        # Create subproblem
        subproblem = CanonicalSubProblem(
            id="integration-test-001",
            description="Implement OAuth2 authentication with PKCE flow",
            domain="security",
            depth=4,
            dependencies=["token-storage", "user-directory"],
            metadata={"protocol": "oauth2", "flow": "pkce"}
        )

        # Analyze complexity
        analysis_response = adapter.analyze_complexity(
            subproblem=subproblem,
            correlation_id="integration-test-001"
        )

        # Verify response structure
        assert isinstance(analysis_response, CanonicalResponse)
        assert analysis_response.task_id == subproblem.id
        assert analysis_response.correlation_id == "integration-test-001"
        assert isinstance(analysis_response.timestamp, str)

        # If MDAP is available, verify complexity score
        if analysis_response.status == TaskStatus.COMPLETED:
            assert analysis_response.complexity_score is not None
            assert 0 <= analysis_response.complexity_score.overall_score <= 1
            assert analysis_response.execution_time_ms is not None
            assert analysis_response.execution_time_ms > 0

            # Allocate resources based on complexity
            allocation_response = adapter.allocate_resources(
                complexity_score=analysis_response.complexity_score,
                correlation_id="integration-test-002"
            )

            assert isinstance(allocation_response, CanonicalResponse)
            if allocation_response.status == TaskStatus.COMPLETED:
                assert allocation_response.strategy is not None
                assert allocation_response.strategy.n_agents >= 1
                assert allocation_response.strategy.timeout_ms > 0

    def test_circuit_breaker_integration(self, adapter):
        """
        Test that circuit breaker works in the full workflow
        """
        # Record failures to open circuit
        for _ in range(adapter.config.circuit_breaker_threshold):
            adapter.circuit_breaker.record_failure()

        # Circuit should be open
        assert adapter.circuit_breaker.get_state() == CircuitState.OPEN

        # Request should be rejected
        subproblem = CanonicalSubProblem(
            id="circuit-test",
            description="Test",
            domain="test"
        )

        response = adapter.analyze_complexity(subproblem)

        assert response.status == TaskStatus.FAILED
        assert response.error["code"] == "CIRCUIT_BREAKER_OPEN"

    def test_health_check_integration(self, adapter):
        """
        Test health check returns complete information
        """
        health = adapter.health_check()

        assert "status" in health
        assert "circuit_breaker_state" in health
        assert "mdap_available" in health
        assert "metrics" in health
        assert "timestamp" in health

        # Verify metrics structure
        assert "requests_total" in health["metrics"]
        assert "requests_success" in health["metrics"]
        assert "requests_failed" in health["metrics"]


class TestMakerAdapterIntegration:
    """End-to-end integration tests for MAKER adapter."""

    def test_maker_step_execution_workflow(self, maker_adapter):
        """
        Test complete MAKER step execution workflow
        """
        step = CanonicalMakerStep(
            step_id="integration-maker-001",
            prompt_template="Analyze: {state}",
            task_type="analysis",
            priority=1,
            system_prompt="You are a helpful assistant",
            expected_schema={"type": "object"},
            metadata={"domain": "general"}
        )

        # Create mock team
        mock_team = Mock()
        mock_team.name = "test-team"
        mock_team.members = []

        current_state = {"problem": "test problem"}
        history = []

        # Execute step
        response = maker_adapter.execute_maker_step(
            step=step,
            current_state=current_state,
            history=history,
            team=mock_team,
            correlation_id="maker-integration-001"
        )

        # Verify response structure
        assert isinstance(response, CanonicalMakerResult)
        assert response.steps_completed >= 0
        assert response.votes_cast >= 0
        assert response.red_flags_detected >= 0
        assert isinstance(response.terminated_reason, str)
        assert response.correlation_id == "maker-integration-001"
        assert isinstance(response.timestamp, str)


class TestBubbleLabAPIIntegration:
    """Integration tests for BubbleLab API client."""

    @patch('bubblelab_api_client.requests')
    def test_bubblelab_mdap_maker_status_workflow(self, mock_requests, bubblelab_client):
        """
        Test complete workflow for fetching MDAP/MAKER status
        """
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "mdap_available": True,
            "maker_available": True,
            "associative_available": True,
            "ground_truth_available": False,
            "full_system_available": True
        }
        mock_requests.request.return_value = mock_response

        # Fetch status
        status = bubblelab_client.get_mdap_maker_status()

        assert status["mdap_available"] == True
        assert status["maker_available"] == True
        assert status["full_system_available"] == True

    @patch('bubblelab_api_client.requests')
    def test_bubblelab_solve_workflow(self, mock_requests, bubblelab_client):
        """
        Test complete workflow for solving with MDAP/MAKER
        """
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": {
                "final_solution": "Test solution",
                "metadata": {"agents_used": 3}
            }
        }
        mock_requests.request.return_value = mock_response

        # Solve problem
        result = bubblelab_client.solve_with_mdap_maker(
            problem_statement="Implement secure authentication",
            use_mdap=True,
            num_mdap_agents=3
        )

        assert result["success"] == True
        assert "result" in result

    @patch('bubblelab_api_client.requests')
    def test_bubblelab_error_handling(self, mock_requests, bubblelab_client):
        """
        Test error handling in BubbleLab API client
        """
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.text = "Database connection failed"
        mock_requests.request.return_value = mock_response

        # Should raise API response error
        with pytest.raises(BubbleLabAPIResponseError) as exc_info:
            bubblelab_client.get_mdap_maker_status()

        assert exc_info.value.status_code == 500
        assert "Database connection failed" in exc_info.value.response_body

    @patch('bubblelab_api_client.requests')
    def test_bubblelab_retry_logic(self, mock_requests, bubblelab_client):
        """
        Test that retry logic works correctly
        """
        # Mock timeout followed by success
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                import requests
                raise requests.exceptions.Timeout("Connection timeout")
            else:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"mdap_available": True}
                return mock_response

        mock_requests.request.side_effect = side_effect

        # Should retry and succeed
        status = bubblelab_client.get_mdap_maker_status()

        assert call_count[0] == 2  # Initial call + 1 retry
        assert status["mdap_available"] == True


class TestMultiAdapterIntegration:
    """Integration tests using multiple adapters together."""

    def test_mdap_maker_combined_workflow(self, adapter, maker_adapter):
        """
        Test workflow that uses both MDAP and MAKER adapters
        """
        # Step 1: Analyze complexity with MDAP
        subproblem = CanonicalSubProblem(
            id="combined-test-001",
            description="Design a distributed voting protocol",
            domain="distributed_systems",
            depth=4
        )

        mdap_response = adapter.analyze_complexity(subproblem)

        # Step 2: Use complexity to determine MAKER config
        if mdap_response.status == TaskStatus.COMPLETED:
            complexity = mdap_response.complexity_score.overall_score

            # Higher complexity → more aggressive MAKER settings
            if complexity > 0.7:
                k_min = 3
                k_max = 7
            else:
                k_min = 2
                k_max = 5

            # Verify MAKER adapter can handle this config
            maker_config = CanonicalMakerConfig(
                k_min=k_min,
                k_max=k_max,
                max_votes_per_step=30
            )

            assert maker_config.k_min >= 2
            assert maker_config.k_max <= 8

    def test_health_check_aggregation(self, adapter, maker_adapter):
        """
        Test aggregating health checks from multiple adapters
        """
        mdap_health = adapter.health_check()
        maker_health = maker_adapter.health_check()

        # Aggregate status
        overall_health = {
            "mdap_adapter": mdap_health["status"],
            "maker_adapter": maker_health["status"],
            "overall": "healthy" if all([
                mdap_health["status"] == "healthy",
                maker_health["status"] == "healthy"
            ]) else "degraded",
            "timestamp": mdap_health["timestamp"]
        }

        assert overall_health["overall"] in ["healthy", "degraded"]
        assert "timestamp" in overall_health


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
