"""
Extra tests for Sovereign-Grade Problem Decomposition System - Reliability & Error Handling
"""

import unittest
import time
from unittest.mock import Mock, patch

from sovereign_reliability import (
    with_retry,
    CircuitBreaker,
    RateLimiter,
    AdaptiveRetryStrategy,
    ResilientComponent,
    ErrorSeverity,
)


class TestSovereignReliabilityExtra(unittest.TestCase):
    """Extra tests for reliability components."""

    def test_retry_with_fallback(self):
        """Test that the fallback function is called after all retries fail."""
        mock_func = Mock(side_effect=ValueError("test error"))
        fallback_func = Mock(return_value="fallback_result")

        decorated_func = with_retry(max_attempts=3, fallback=fallback_func)(mock_func)

        result = decorated_func()

        self.assertEqual(mock_func.call_count, 3)
        fallback_func.assert_called_once()
        self.assertEqual(result, "fallback_result")

    def test_circuit_breaker_half_open_state(self):
        """Test the half-open state of the circuit breaker."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        mock_func = Mock()

        # Open the circuit
        with self.assertRaises(ValueError):
            circuit_breaker.call(Mock(side_effect=ValueError))
        with self.assertRaises(ValueError):
            circuit_breaker.call(Mock(side_effect=ValueError))
        self.assertEqual(circuit_breaker.state, "open")

        # Wait for the timeout
        time.sleep(0.1)

        # The circuit should be half-open now
        circuit_breaker.call(mock_func)
        self.assertEqual(circuit_breaker.state, "closed")

    def test_rate_limiter_wait_time(self):
        """Test the wait time calculation of the rate limiter."""
        rate_limiter = RateLimiter(max_requests=2, time_window=0.1)
        self.assertTrue(rate_limiter.is_allowed())
        self.assertTrue(rate_limiter.is_allowed())
        self.assertFalse(rate_limiter.is_allowed())

        wait_time = rate_limiter.get_wait_time()
        self.assertGreater(wait_time, 0)

    def test_adaptive_retry_strategy(self):
        """Test the adaptive retry strategy."""
        strategy = AdaptiveRetryStrategy(initial_delay=0.01)
        delay1 = strategy.get_delay(0)

        # Simulate failures
        for _ in range(4):
            strategy.record_failure()

        delay2 = strategy.get_delay(0)
        self.assertGreater(delay2, delay1)

    @patch("sovereign_reliability.trace.get_tracer")
    def test_resilient_component_tracing(self, mock_get_tracer):
        """Test that the resilient component creates spans for tracing."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        resilient_component = ResilientComponent()
        mock_operation = Mock(return_value="success")

        resilient_component.safe_execute(mock_operation)

        mock_tracer.start_as_current_span.assert_called_once_with("mock_operation")


if __name__ == "__main__":
    unittest.main()
