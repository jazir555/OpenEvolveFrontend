"""
Comprehensive Logging Tests (Bug #21)

Tests for structured logging with correlation IDs and JSON Lines format.
Follows the Federation Constitution's observability requirements.
"""

import pytest
import json
import logging
import uuid
import re
from io import StringIO
from typing import Dict, Any

import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from structured_logging import (
    StructuredLogger,
    JsonFormatter,
    generate_correlation_id,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    with_correlation_id,
)


class TestCorrelationIdGeneration:
    """Test correlation ID generation and validation."""

    def test_generate_correlation_id_returns_uuid_v4(self):
        """Test that generate_correlation_id returns valid UUID v4 format."""
        correlation_id = generate_correlation_id()

        # Should be valid UUID v4 format
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        assert uuid_pattern.match(correlation_id), f"Invalid UUID v4 format: {correlation_id}"

    def test_generate_correlation_id_unique(self):
        """Test that each generated correlation ID is unique."""
        ids = [generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100, "Generated IDs are not unique"

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID in context."""
        test_id = str(uuid.uuid4())

        set_correlation_id(test_id)
        retrieved_id = get_correlation_id()

        assert retrieved_id == test_id, f"Expected {test_id}, got {retrieved_id}"

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id(str(uuid.uuid4()))
        assert get_correlation_id() is not None

        clear_correlation_id()
        assert get_correlation_id() is None


class TestStructuredLogger:
    """Test StructuredLogger functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.logger = StructuredLogger("test_service")
        self.log_capture = StringIO()

        # Set logger to DEBUG level to capture all messages
        self.logger.logger.setLevel(logging.DEBUG)

        # Add handler to capture logs
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(JsonFormatter())
        self.logger.logger.addHandler(self.handler)

    def teardown_method(self):
        """Cleanup after tests."""
        self.logger.logger.removeHandler(self.handler)

    def _get_last_log(self) -> Dict[str, Any]:
        """Get last log entry as dictionary."""
        log_content = self.log_capture.getvalue()
        if not log_content.strip():
            return {}

        # Get last line
        last_line = log_content.strip().split('\n')[-1]
        return json.loads(last_line)

    def test_logger_initialization_requires_name(self):
        """Test that logger requires a name."""
        with pytest.raises(ValueError, match="Logger name must be provided"):
            StructuredLogger("")

        with pytest.raises(ValueError, match="Logger name must be provided"):
            StructuredLogger(None)

    def test_logger_has_all_log_levels(self):
        """Test logger has info, warn, warning, error, debug methods."""
        assert hasattr(self.logger, 'info')
        assert hasattr(self.logger, 'warning')
        assert hasattr(self.logger, 'warn')
        assert hasattr(self.logger, 'error')
        assert hasattr(self.logger, 'debug')

    def test_info_log_includes_required_fields(self):
        """Test info log includes required fields."""
        self.logger.info("Test message")

        log_dict = self._get_last_log()

        # Required fields
        assert "timestamp" in log_dict
        assert "level" in log_dict
        assert "message" in log_dict
        assert "logger" in log_dict
        assert "correlation_id" in log_dict

        # Check values
        assert log_dict["level"] == "INFO"
        assert log_dict["message"] == "Test message"
        assert log_dict["logger"] == "test_service"

    def test_warning_log_includes_required_fields(self):
        """Test warning log includes required fields."""
        self.logger.warning("Warning message")

        log_dict = self._get_last_log()

        assert log_dict["level"] == "WARNING"
        assert log_dict["message"] == "Warning message"

    def test_warn_alias(self):
        """Test warn is alias for warning."""
        self.logger.warn("Warning message")

        log_dict = self._get_last_log()
        assert log_dict["level"] == "WARNING"

    def test_error_log_includes_required_fields(self):
        """Test error log includes required fields."""
        self.logger.error("Error message")

        log_dict = self._get_last_log()

        assert log_dict["level"] == "ERROR"
        assert log_dict["message"] == "Error message"

    def test_debug_log_includes_required_fields(self):
        """Test debug log includes required fields."""
        self.logger.debug("Debug message")

        log_dict = self._get_last_log()

        assert log_dict["level"] == "DEBUG"
        assert log_dict["message"] == "Debug message"

    def test_logs_include_source_and_target_service(self):
        """Test logs include source_service and target_service when provided."""
        self.logger.info(
            "API call",
            source_service="client",
            target_service="api"
        )

        log_dict = self._get_last_log()

        assert log_dict["source_service"] == "client"
        assert log_dict["target_service"] == "api"

    def test_logs_include_correlation_id_when_provided(self):
        """Test logs include provided correlation ID."""
        test_id = str(uuid.uuid4())
        self.logger.info("Test message", correlation_id=test_id)

        log_dict = self._get_last_log()

        assert log_dict["correlation_id"] == test_id

    def test_logs_use_context_correlation_id_when_not_provided(self):
        """Test logs use correlation ID from context when not explicitly provided."""
        context_id = str(uuid.uuid4())
        set_correlation_id(context_id)

        self.logger.info("Test message")

        log_dict = self._get_last_log()
        assert log_dict["correlation_id"] == context_id

        clear_correlation_id()

    def test_logs_include_extra_fields(self):
        """Test logs include extra kwargs as fields."""
        self.logger.info(
            "Test message",
            extra_field="extra_value",
            number=42,
            nested={"key": "value"}
        )

        log_dict = self._get_last_log()

        assert log_dict["extra_field"] == "extra_value"
        assert log_dict["number"] == 42
        assert log_dict["nested"]["key"] == "value"


class TestJsonFormatter:
    """Test JsonFormatter produces valid JSON."""

    def test_formatter_outputs_valid_json(self):
        """Test formatter outputs valid JSON."""
        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)

        # Should be valid JSON
        log_dict = json.loads(formatted)

        assert "timestamp" in log_dict
        assert "level" in log_dict
        assert "message" in log_dict
        assert "logger" in log_dict


class TestJsonLinesFormat:
    """Test logs follow JSON Lines format."""

    def test_multiple_logs_each_on_new_line(self):
        """Test multiple log entries are on separate lines."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Log multiple messages
        logger.info("Message 1")
        logger.info("Message 2")
        logger.info("Message 3")

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            log_dict = json.loads(line)
            assert "message" in log_dict

        logger.logger.removeHandler(handler)


class TestTimestampFormat:
    """Test log timestamps are in correct format."""

    def test_timestamp_ends_with_z(self):
        """Test timestamps end with Z (UTC indicator)."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        logger.info("Test message")

        log_content = log_capture.getvalue()
        log_dict = json.loads(log_content.strip().split('\n')[-1])

        assert log_dict["timestamp"].endswith('Z'), \
            f"Timestamp should end with Z: {log_dict['timestamp']}"

    def test_timestamp_iso8601_format(self):
        """Test timestamps are in ISO-8601 format."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        logger.info("Test message")

        log_content = log_capture.getvalue()
        log_dict = json.loads(log_content.strip().split('\n')[-1])

        # Should be valid ISO-8601
        timestamp = log_dict["timestamp"].replace('Z', '')
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            pytest.fail(f"Invalid ISO-8601 timestamp: {log_dict['timestamp']}")

    def test_timestamp_no_timezone_offset(self):
        """Test timestamps don't include timezone offset (only Z)."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        logger.info("Test message")

        log_content = log_capture.getvalue()
        log_dict = json.loads(log_content.strip().split('\n')[-1])

        # Should not have + or - offset indicators
        timestamp = log_dict["timestamp"]
        # Remove the Z and check for timezone offset patterns
        timestamp_without_z = timestamp[:-1]
        # Should not have +HH:MM or -HH:MM at the end
        assert not re.search(r'[+-]\d{2}:\d{2}$', timestamp_without_z), \
            f"Timestamp should not have timezone offset: {timestamp}"


class TestWithCorrelationIdDecorator:
    """Test with_correlation_id decorator."""

    def test_decorator_generates_correlation_id_if_not_set(self):
        """Test decorator generates correlation ID if not already set."""
        @with_correlation_id
        def test_function():
            return get_correlation_id()

        clear_correlation_id()
        result = test_function()

        assert result is not None
        assert re.match(r'^[0-9a-f]{8}-', result, re.IGNORECASE)

    def test_decorator_preserves_existing_correlation_id(self):
        """Test decorator preserves existing correlation ID."""
        existing_id = str(uuid.uuid4())

        @with_correlation_id
        def test_function():
            return get_correlation_id()

        set_correlation_id(existing_id)
        result = test_function()

        assert result == existing_id
        clear_correlation_id()


class TestLogParseability:
    """Test logs are parseable as JSON."""

    def test_all_log_levels_parseable(self):
        """Test all log levels produce parseable JSON."""
        logger = StructuredLogger("test_service")
        log_capture = StringIO()

        # Set to DEBUG level to capture all messages
        logger.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(JsonFormatter())
        logger.logger.addHandler(handler)

        # Test all levels
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")

        log_content = log_capture.getvalue()
        lines = log_content.strip().split('\n')

        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}: {lines}"

        for line in lines:
            # Should parse without error
            log_dict = json.loads(line)
            assert isinstance(log_dict, dict)

        logger.logger.removeHandler(handler)
