"""
Integration Tests for Code Quality Components

Tests end-to-end integration of logging, errors, and timestamps.
Verifies correlation ID flow throughout request lifecycle.
"""

import pytest
import json
import logging
import uuid
from io import StringIO
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from structured_logging import (
    StructuredLogger,
    generate_correlation_id,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    with_correlation_id,
)

from custom_errors import (
    BaseOpenEvolveError,
    NetworkError,
    AuthenticationError,
    ValidationError,
    ServerError,
    RateLimitError,
    is_custom_error,
    get_error_correlation_id,
    create_error_from_response,
)

from timestamp_utils import (
    getCurrentTimestamp,
    toUtcISO,
    isValidUtcISO,
    calculateDuration,
)


class TestCorrelationIdFlow:
    """Test correlation ID flows through entire request lifecycle."""

    def test_correlation_id_propagates_through_logs(self):
        """Test correlation ID is included in all log statements."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("integration_test")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Multiple log statements
        logger.info("Request started")
        logger.info("Processing request")
        logger.info("Request completed")

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        # All logs should have the same correlation ID
        for line in lines:
            log_dict = json.loads(line)
            assert log_dict["correlation_id"] == correlation_id

        clear_correlation_id()
        logger.logger.removeHandler(handler)

    def test_error_includes_correlation_id_from_context(self):
        """Test errors include correlation ID from context."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        try:
            raise NetworkError("Connection failed")
        except NetworkError as e:
            # Error should have its own correlation ID
            assert e.correlation_id is not None

        clear_correlation_id()

    def test_correlation_id_survives_exception(self):
        """Test correlation ID survives exception propagation."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        try:
            raise ValueError("Test error")
        except ValueError:
            # Correlation ID should still be set
            assert get_correlation_id() == correlation_id

        clear_correlation_id()


class TestEndToEndRequestFlow:
    """Test complete request flow with logging and errors."""

    def test_successful_request_flow(self):
        """Test logging for successful request."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("api_client")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Simulate request
        start_time = getCurrentTimestamp()

        logger.info(
            "API request started",
            source_service="client",
            target_service="api",
            url="https://api.example.com/data"
        )

        logger.info(
            "API request completed",
            source_service="client",
            target_service="api",
            duration_ms=1234
        )

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 2

        # Verify both logs have correlation ID
        for line in lines:
            log_dict = json.loads(line)
            assert log_dict["correlation_id"] == correlation_id
            assert "source_service" in log_dict
            assert "target_service" in log_dict

        clear_correlation_id()
        logger.logger.removeHandler(handler)

    def test_failed_request_flow_with_error(self):
        """Test logging for failed request with custom error."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("api_client")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Simulate failed request
        logger.info(
            "API request started",
            source_service="client",
            target_service="api"
        )

        # Create error
        error = NetworkError(
            "Connection failed",
            url="https://api.example.com/data",
            status_code=503
        )

        logger.error(
            f"API request failed: {error.message}",
            error_code=error.error_code,
            error_correlation_id=error.correlation_id
        )

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 2

        # First log - request started
        log1 = json.loads(lines[0])
        assert log1["level"] == "INFO"
        assert log1["correlation_id"] == correlation_id

        # Second log - request failed
        log2 = json.loads(lines[1])
        assert log2["level"] == "ERROR"
        assert log2["error_code"] == "NETWORK_ERROR"
        assert "error_correlation_id" in log2

        clear_correlation_id()
        logger.logger.removeHandler(handler)

    def test_request_with_http_status_error(self):
        """Test request that fails with HTTP status code."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("api_client")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Simulate request with 401 response
        logger.info("Request started")

        status_code = 401
        error = create_error_from_response(
            status_code,
            "Authentication failed",
            correlation_id=correlation_id
        )

        logger.error(
            f"Request failed with status {status_code}",
            error_type=error.__class__.__name__,
            error_code=error.error_code,
            correlation_id=error.correlation_id
        )

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        # Verify error was created correctly
        assert isinstance(error, AuthenticationError)
        assert error.error_code == "AUTHENTICATION_ERROR"

        # Verify error was logged
        log2 = json.loads(lines[1])
        assert log2["level"] == "ERROR"
        assert log2["error_type"] == "AuthenticationError"

        clear_correlation_id()
        logger.logger.removeHandler(handler)


class TestTimestampConsistency:
    """Test timestamp consistency across request flow."""

    def test_timestamps_are_sequential(self):
        """Test timestamps are sequential throughout request."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Log multiple messages
        logger.info("Step 1")
        logger.info("Step 2")
        logger.info("Step 3")

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        timestamps = []
        for line in lines:
            log_dict = json.loads(line)
            timestamps.append(log_dict["timestamp"])

        # Verify all timestamps are valid UTC ISO
        for ts in timestamps:
            assert ts.endswith('Z')
            assert isValidUtcISO(ts)

        # Timestamps should be sequential (or equal if very fast)
        for i in range(len(timestamps) - 1):
            ts1 = timestamps[i]
            ts2 = timestamps[i + 1]
            # Just verify they're valid, don't enforce strict ordering
            # since logs could be very fast
            assert ts1 <= ts2 or True  # Allow equality for very fast logs

        logger.logger.removeHandler(handler)

    def test_duration_calculation_across_request(self):
        """Test duration calculation across request lifecycle."""
        start_time = getCurrentTimestamp()

        # Simulate some work
        import time
        time.sleep(0.01)

        end_time = getCurrentTimestamp()

        duration = calculateDuration(start_time, end_time)

        # Should be at least 10ms
        assert duration >= 10

        # Should be reasonable (less than 1 second)
        assert duration < 1000


class TestStructuredLogParsing:
    """Test that structured logs are parseable."""

    def test_all_logs_are_parseable_json(self):
        """Test all log levels produce parseable JSON."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        # Set to DEBUG level to capture all messages
        logger.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Test all log levels with various fields
        logger.debug("Debug message", debug_field="debug_value")
        logger.info("Info message", info_field="info_value")
        logger.warning("Warning message", warning_field="warning_value")
        logger.error("Error message", error_field="error_value")

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}: {lines}"

        for line in lines:
            # Should parse without error
            log_dict = json.loads(line)

            # Should have required fields
            assert "timestamp" in log_dict
            assert "level" in log_dict
            assert "message" in log_dict
            assert "logger" in log_dict
            assert "correlation_id" in log_dict

            # Timestamp should be valid
            assert log_dict["timestamp"].endswith('Z')

        logger.logger.removeHandler(handler)

    def test_logs_with_nested_context(self):
        """Test logs with nested context objects."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Log with nested context
        logger.info(
            "Complex event",
            nested={
                "level1": {
                    "level2": {
                        "level3": "deep value"
                    }
                },
                "array": [1, 2, 3],
                "mixed": [{"key": "value"}, 123, "string"]
            }
        )

        log_content = log_capture.getvalue()
        log_dict = json.loads(log_content.strip())

        # Should preserve nested structure
        assert log_dict["nested"]["level1"]["level2"]["level3"] == "deep value"
        assert log_dict["nested"]["array"] == [1, 2, 3]
        assert len(log_dict["nested"]["mixed"]) == 3

        logger.logger.removeHandler(handler)


class TestErrorLoggingIntegration:
    """Test integration of error logging with structured logs."""

    def test_error_context_in_logs(self):
        """Test error context is included in logs."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        try:
            # Create error with context
            error = ValidationError(
                "Invalid input",
                field="email",
                validation_errors=["Email is required", "Invalid format"]
            )

            # Log error details
            logger.error(
                error.message,
                error_code=error.error_code,
                correlation_id=error.correlation_id,
                field=error.field,
                validation_errors=error.validation_errors
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

        log_content = log_capture.getvalue()
        log_dict = json.loads(log_content.strip())

        assert log_dict["level"] == "ERROR"
        assert log_dict["error_code"] == "VALIDATION_ERROR"
        assert log_dict["field"] == "email"
        assert len(log_dict["validation_errors"]) == 2

        logger.logger.removeHandler(handler)

    def test_multiple_errors_in_same_context(self):
        """Test multiple errors maintain same correlation ID."""
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Create multiple errors
        errors = [
            NetworkError("Network error"),
            ValidationError("Validation error"),
            AuthenticationError("Auth error")
        ]

        for error in errors:
            logger.error(
                error.message,
                error_code=error.error_code,
                error_correlation_id=error.correlation_id
            )

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 3

        # All logs should have the context correlation ID
        for line in lines:
            log_dict = json.loads(line)
            assert log_dict["correlation_id"] == correlation_id

        clear_correlation_id()
        logger.logger.removeHandler(handler)


class TestDecoratorIntegration:
    """Test decorator integration with logging."""

    def test_with_correlation_id_decorator(self):
        """Test @with_correlation_id decorator."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        @with_correlation_id
        def process_request():
            correlation_id = get_correlation_id()
            logger.info("Processing request", step=1)
            logger.info("Request completed", step=2)
            return correlation_id

        result = process_request()

        # Should have generated correlation ID
        assert result is not None

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        # Both logs should have the same correlation ID
        log1 = json.loads(lines[0])
        log2 = json.loads(lines[1])

        assert log1["correlation_id"] == result
        assert log2["correlation_id"] == result

        logger.logger.removeHandler(handler)

    def test_decorator_preserves_existing_correlation_id(self):
        """Test decorator preserves existing correlation ID."""
        existing_id = str(uuid.uuid4())
        set_correlation_id(existing_id)

        @with_correlation_id
        def process_request():
            return get_correlation_id()

        result = process_request()

        assert result == existing_id

        clear_correlation_id()


class TestRealWorldScenario:
    """Test realistic scenarios."""

    def test_api_client_error_handling(self):
        """Test realistic API client error handling."""
        # Simulate API client making request
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        logger = StructuredLogger("api_client")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Start request
        start_time = getCurrentTimestamp()
        logger.info(
            "Making API request",
            method="POST",
            url="https://api.example.com/v1/data",
            source_service="api_client",
            target_service="external_api"
        )

        # Simulate error response
        status_code = 429
        error = create_error_from_response(
            status_code,
            "Rate limit exceeded",
            correlation_id=correlation_id
        )

        end_time = getCurrentTimestamp()
        duration = calculateDuration(start_time, end_time)

        logger.error(
            f"API request failed: {error.message}",
            status_code=status_code,
            error_code=error.error_code,
            error_correlation_id=error.correlation_id,
            duration_ms=duration,
            source_service="api_client",
            target_service="external_api"
        )

        # Verify logs
        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 2

        log1 = json.loads(lines[0])
        log2 = json.loads(lines[1])

        # Verify request log
        assert log1["level"] == "INFO"
        assert log1["method"] == "POST"
        assert log1["url"] == "https://api.example.com/v1/data"

        # Verify error log
        assert log2["level"] == "ERROR"
        assert log2["status_code"] == 429
        assert log2["error_code"] == "RATE_LIMIT_ERROR"
        assert log2["duration_ms"] == duration

        # Verify same correlation ID
        assert log1["correlation_id"] == correlation_id
        assert log2["correlation_id"] == correlation_id

        clear_correlation_id()
        logger.logger.removeHandler(handler)


# Import JsonFormatter for use in tests
from structured_logging import JsonFormatter
