"""Tests for the ResilientLLMClient."""

import time
import unittest
from unittest.mock import Mock, patch

import pytest

from ace import DummyLLMClient
from ace.llm import LLMResponse
from ace.llm_providers.resilient_client import CallInfo, ResilientLLMClient


@pytest.mark.unit
class ResilientLLMClientTest(unittest.TestCase):
    """Test suite for ResilientLLMClient."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.base_client = DummyLLMClient()
        self.resilient_client = ResilientLLMClient(
            base_client=self.base_client,
            max_retries=3,
            base_sleep=0.1,  # Short sleep for tests
            timeout=10.0,
        )

    def test_successful_call_returns_response_and_call_info(self) -> None:
        """Test that a successful call returns both response and CallInfo."""
        # Queue a successful response
        self.base_client.queue('{"final_answer": "42"}')

        # Make the call
        response, call_info = self.resilient_client.complete(
            prompt="What is the answer?",
            role="agent",
            call_id="test_call_1",
        )

        # Verify response
        self.assertEqual(response, '{"final_answer": "42"}')

        # Verify CallInfo
        self.assertIsInstance(call_info, CallInfo)
        self.assertEqual(call_info.role, "agent")
        self.assertEqual(call_info.call_id, "test_call_1")
        self.assertEqual(call_info.prompt, "What is the answer?")
        self.assertEqual(call_info.response, '{"final_answer": "42"}')
        self.assertGreaterEqual(call_info.total_time, 0)  # Can be 0 for very fast calls
        self.assertIsNone(call_info.error)

    def test_retry_on_timeout(self) -> None:
        """Test that timeout errors trigger retry with exponential backoff."""
        # Create a mock client that fails twice then succeeds
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # First two attempts fail with timeout
        mock_client.complete.side_effect = [
            TimeoutError("Request timed out"),
            TimeoutError("Request timed out"),
            # Third attempt succeeds
            LLMResponse(text="Success after retries"),
        ]

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=3,
            base_sleep=0.05,  # Very short for tests
            timeout=10.0,
        )

        # Make the call
        start_time = time.time()
        response, call_info = resilient_client.complete(
            prompt="Test prompt",
            role="agent",
            call_id="timeout_test",
        )
        elapsed = time.time() - start_time

        # Verify success after retries
        self.assertEqual(response, "Success after retries")
        self.assertEqual(call_info.response, "Success after retries")
        self.assertIsNone(call_info.error)

        # Verify it called complete 3 times (initial + 2 retries)
        self.assertEqual(mock_client.complete.call_count, 3)

        # Verify exponential backoff was applied (should take some time)
        # With base_sleep=0.05, attempts 1 and 2 should have delays
        self.assertGreater(elapsed, 0.05)

    def test_retry_on_rate_limit(self) -> None:
        """Test that rate limit errors trigger retry with longer backoff."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Simulate rate limit errors
        rate_limit_error = Exception("429 Rate limit exceeded")

        mock_client.complete.side_effect = [
            rate_limit_error,
            rate_limit_error,
            LLMResponse(text="Success after rate limit"),
        ]

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=3,
            base_sleep=0.05,
            timeout=10.0,
        )

        # Make the call
        start_time = time.time()
        response, call_info = resilient_client.complete(
            prompt="Test prompt",
            role="reflector",
            call_id="rate_limit_test",
        )
        elapsed = time.time() - start_time

        # Verify success
        self.assertEqual(response, "Success after rate limit")
        self.assertEqual(call_info.response, "Success after rate limit")
        self.assertIsNone(call_info.error)

        # Verify it retried correctly
        self.assertEqual(mock_client.complete.call_count, 3)

        # Rate limit errors should have longer backoff (multiplier=2.0)
        # Should take more time than regular timeout errors
        self.assertGreater(elapsed, 0.2)

    def test_no_retry_on_auth_error(self) -> None:
        """Test that authentication errors do not trigger retry."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Simulate auth error
        auth_error = Exception("401 Unauthorized - Invalid API key")
        mock_client.complete.side_effect = auth_error

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=3,
            base_sleep=0.1,
            timeout=10.0,
        )

        # Make the call - should fail immediately
        with self.assertRaises(Exception) as context:
            resilient_client.complete(
                prompt="Test prompt",
                role="agent",
                call_id="auth_test",
            )

        # Verify the auth error was raised
        self.assertIn("401", str(context.exception))

        # Verify it only called once (no retries)
        self.assertEqual(mock_client.complete.call_count, 1)

    def test_exponential_backoff(self) -> None:
        """Test that exponential backoff increases wait time between retries."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Fail 3 times with timeout
        mock_client.complete.side_effect = [
            TimeoutError("Timeout 1"),
            TimeoutError("Timeout 2"),
            TimeoutError("Timeout 3"),
            LLMResponse(text="Finally succeeded"),
        ]

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=4,
            base_sleep=0.05,
            timeout=10.0,
        )

        # Make the call
        start_time = time.time()
        response, call_info = resilient_client.complete(
            prompt="Test prompt",
            role="skill_manager",
            call_id="backoff_test",
        )
        elapsed = time.time() - start_time

        # Verify success
        self.assertEqual(response, "Finally succeeded")

        # Verify 4 attempts (initial + 3 retries)
        self.assertEqual(mock_client.complete.call_count, 4)

        # With exponential backoff:
        # Attempt 1: immediate
        # Attempt 2: ~0.05s (base_sleep * 1^1 * jitter)
        # Attempt 3: ~0.05s (base_sleep * 1^2 * jitter)
        # Attempt 4: ~0.05s (base_sleep * 1^3 * jitter)
        # Total should be > 0.1s due to cumulative delays
        self.assertGreater(elapsed, 0.1)

    def test_retry_on_empty_response(self) -> None:
        """Test that empty responses trigger a retry."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Return empty responses first, then a real response
        mock_client.complete.side_effect = [
            LLMResponse(text=""),  # Empty
            LLMResponse(text="   "),  # Whitespace only
            LLMResponse(text="Valid response"),
        ]

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=3,
            base_sleep=0.05,
            timeout=10.0,
        )

        # Make the call
        response, call_info = resilient_client.complete(
            prompt="Test prompt",
            role="agent",
            call_id="empty_response_test",
        )

        # Verify success after retries
        self.assertEqual(response, "Valid response")
        self.assertEqual(call_info.response, "Valid response")

        # Verify it retried for empty responses
        self.assertEqual(mock_client.complete.call_count, 3)

    def test_max_retries_exceeded(self) -> None:
        """Test that max retries limit is respected."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Always fail
        mock_client.complete.side_effect = TimeoutError("Always times out")

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=2,  # Only 2 retries
            base_sleep=0.05,
            timeout=10.0,
        )

        # Make the call - should fail after max retries
        with self.assertRaises(TimeoutError):
            resilient_client.complete(
                prompt="Test prompt",
                role="agent",
                call_id="max_retries_test",
            )

        # Verify it tried max_retries + 1 times (initial + retries)
        self.assertEqual(mock_client.complete.call_count, 3)

    def test_timeout_exceeded(self) -> None:
        """Test that overall timeout is respected."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Always fail with slow response
        mock_client.complete.side_effect = TimeoutError("Slow timeout")

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=10,  # High retry count
            base_sleep=0.1,
            timeout=0.2,  # Very short timeout
        )

        # Make the call - should fail due to timeout
        start_time = time.time()
        with self.assertRaises(TimeoutError):
            resilient_client.complete(
                prompt="Test prompt",
                role="agent",
                call_id="timeout_exceeded_test",
            )
        elapsed = time.time() - start_time

        # Should not take longer than timeout + some overhead
        self.assertLess(elapsed, 0.5)

    def test_error_classification_rate_limit(self) -> None:
        """Test error classification for rate limit errors."""
        client = ResilientLLMClient(self.base_client)

        # Test various rate limit error messages
        errors = [
            Exception("429 Rate limit exceeded"),
            Exception("Rate limit reached"),
            Exception("Quota exceeded"),
        ]

        for error in errors:
            error_type = client._classify_error(error)
            self.assertEqual(
                error_type, "rate_limit", f"Failed for error: {error}"
            )

    def test_error_classification_timeout(self) -> None:
        """Test error classification for timeout errors."""
        client = ResilientLLMClient(self.base_client)

        errors = [
            TimeoutError("Request timed out"),
            Exception("Connection timeout"),
            Exception("Network timeout error"),
        ]

        for error in errors:
            error_type = client._classify_error(error)
            self.assertEqual(
                error_type, "timeout", f"Failed for error: {error}"
            )

    def test_error_classification_server_error(self) -> None:
        """Test error classification for server errors."""
        client = ResilientLLMClient(self.base_client)

        errors = [
            Exception("500 Internal Server Error"),
            Exception("502 Bad Gateway"),
            Exception("503 Service Unavailable"),
        ]

        for error in errors:
            error_type = client._classify_error(error)
            self.assertEqual(
                error_type, "server_error", f"Failed for error: {error}"
            )

    def test_error_classification_auth_error(self) -> None:
        """Test error classification for authentication errors."""
        client = ResilientLLMClient(self.base_client)

        errors = [
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("Authentication failed"),
            Exception("Invalid API key"),
        ]

        for error in errors:
            error_type = client._classify_error(error)
            self.assertEqual(
                error_type, "auth_error", f"Failed for error: {error}"
            )

    def test_error_classification_unknown(self) -> None:
        """Test error classification for unknown errors."""
        client = ResilientLLMClient(self.base_client)

        error = Exception("Some unknown error")
        error_type = client._classify_error(error)
        self.assertEqual(error_type, "unknown")

    def test_backoff_calculation(self) -> None:
        """Test backoff calculation for different error types."""
        client = ResilientLLMClient(self.base_client, base_sleep=1.0)

        # Test rate limit backoff (multiplier 2.0)
        backoff = client._calculate_backoff("rate_limit", 1)
        # Should be around 1.0 * 2.0^1 * jitter(0.5-1.5) = 1.0-3.0
        self.assertGreater(backoff, 0.5)
        self.assertLess(backoff, 4.5)

        # Test server error backoff (multiplier 1.5)
        backoff = client._calculate_backoff("server_error", 1)
        # Should be around 1.0 * 1.5^1 * jitter = 0.75-2.25
        self.assertGreater(backoff, 0.5)
        self.assertLess(backoff, 3.5)

        # Test timeout backoff (multiplier 1.0)
        backoff = client._calculate_backoff("timeout", 1)
        # Should be around 1.0 * 1.0^1 * jitter = 0.5-1.5
        self.assertGreater(backoff, 0.4)
        self.assertLess(backoff, 2.0)

    def test_backoff_increases_with_attempts(self) -> None:
        """Test that backoff increases exponentially with attempt number."""
        client = ResilientLLMClient(self.base_client, base_sleep=1.0)

        # For rate limit errors (multiplier=2.0), backoff should increase significantly
        backoffs_1 = [client._calculate_backoff("rate_limit", 1) for _ in range(20)]
        backoffs_2 = [client._calculate_backoff("rate_limit", 2) for _ in range(20)]
        backoffs_3 = [client._calculate_backoff("rate_limit", 3) for _ in range(20)]

        avg_1 = sum(backoffs_1) / len(backoffs_1)
        avg_2 = sum(backoffs_2) / len(backoffs_2)
        avg_3 = sum(backoffs_3) / len(backoffs_3)

        # With multiplier=2.0:
        # Attempt 1: 1.0 * 2.0^1 * jitter = 1.0-3.0 (avg ~2.0)
        # Attempt 2: 1.0 * 2.0^2 * jitter = 2.0-6.0 (avg ~4.0)
        # Attempt 3: 1.0 * 2.0^3 * jitter = 4.0-12.0 (avg ~8.0)
        self.assertGreater(avg_2, avg_1)
        self.assertGreater(avg_3, avg_2)

    def test_token_tracking(self) -> None:
        """Test that token usage is tracked in CallInfo."""
        mock_client = Mock(spec=DummyLLMClient)
        mock_client.model = "test-model"

        # Return response with usage data
        response_with_usage = LLMResponse(
            text="Test response",
            raw={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
            },
        )
        mock_client.complete.return_value = response_with_usage

        resilient_client = ResilientLLMClient(
            base_client=mock_client,
            max_retries=3,
            base_sleep=0.1,
            timeout=10.0,
        )

        # Make the call
        response, call_info = resilient_client.complete(
            prompt="Test prompt",
            role="agent",
            call_id="token_test",
        )

        # Verify token tracking
        self.assertEqual(call_info.prompt_tokens, 100)
        self.assertEqual(call_info.response_tokens, 50)

    def test_call_info_fields(self) -> None:
        """Test that all CallInfo fields are properly populated."""
        self.base_client.queue("Test response")

        response, call_info = self.resilient_client.complete(
            prompt="What is ACE?",
            role="agent",
            call_id="field_test_123",
        )

        # Verify all fields
        self.assertEqual(call_info.role, "agent")
        self.assertEqual(call_info.call_id, "field_test_123")
        self.assertEqual(call_info.model, "dummy")  # DummyLLMClient model
        self.assertEqual(call_info.prompt, "What is ACE?")
        self.assertEqual(call_info.response, "Test response")
        self.assertGreaterEqual(call_info.total_time, 0)  # Can be 0 for very fast calls
        self.assertEqual(call_info.prompt_tokens, 0)
        self.assertEqual(call_info.response_tokens, 0)
        self.assertIsNone(call_info.error)


if __name__ == "__main__":
    unittest.main()
