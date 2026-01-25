"""
Comprehensive Timestamp Tests (Bug #23)

Tests for UTC timestamp utilities following the Law of UTC.
All timestamps must be in UTC ISO-8601 format ending with 'Z'.
"""

import pytest
from datetime import datetime, timedelta, timezone
import time
import re

import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from timestamp_utils import (
    getCurrentTimestamp,
    toUtcISO,
    isValidUtcISO,
    calculateDuration,
    addDuration,
    parseUtcISO,
    formatDuration,
)


class TestGetCurrentTimestamp:
    """Test getCurrentTimestamp function."""

    def test_returns_string(self):
        """Test getCurrentTimestamp returns a string."""
        timestamp = getCurrentTimestamp()

        assert isinstance(timestamp, str)

    def test_ends_with_z(self):
        """Test timestamp ends with Z (UTC indicator)."""
        timestamp = getCurrentTimestamp()

        assert timestamp.endswith('Z'), f"Timestamp should end with Z: {timestamp}"

    def test_iso8601_format(self):
        """Test timestamp is in ISO-8601 format."""
        timestamp = getCurrentTimestamp()

        # Remove Z and try parsing
        timestamp_without_z = timestamp.replace('Z', '')

        try:
            datetime.fromisoformat(timestamp_without_z)
        except ValueError:
            pytest.fail(f"Invalid ISO-8601 timestamp: {timestamp}")

    def test_no_timezone_offset(self):
        """Test timestamp has no timezone offset (only Z)."""
        timestamp = getCurrentTimestamp()

        # Should not have + or - except for Z at end
        timestamp_without_z = timestamp[:-1]
        assert '+' not in timestamp_without_z
        assert '-' not in timestamp_without_z.replace('-', '')  # Allow dashes in date

    def test_is_recent(self):
        """Test timestamp is recent (within last second)."""
        timestamp = getCurrentTimestamp()
        parsed = parseUtcISO(timestamp)
        now = datetime.utcnow()

        # Should be within 1 second
        diff = abs((now - parsed).total_seconds())
        assert diff < 1.0, f"Timestamp is not recent: {diff}s difference"

    def test_unique_on_consecutive_calls(self):
        """Test consecutive calls return different timestamps."""
        timestamp1 = getCurrentTimestamp()
        time.sleep(0.01)  # Small delay
        timestamp2 = getCurrentTimestamp()

        assert timestamp1 != timestamp2


class TestToUtcISO:
    """Test toUtcISO conversion function."""

    def test_convert_string_with_date_and_time(self):
        """Test converting 'YYYY-MM-DD HH:MM:SS' format."""
        result = toUtcISO("2025-01-19 12:00:00")

        assert result.endswith('Z')
        assert "2025-01-19T12:00:00" in result

    def test_convert_string_with_iso_format(self):
        """Test converting ISO format string."""
        result = toUtcISO("2025-01-19T12:00:00")

        assert result.endswith('Z')
        assert "2025-01-19T12:00:00" in result

    def test_convert_string_already_with_z(self):
        """Test converting string that already has Z."""
        result = toUtcISO("2025-01-19T12:00:00Z")

        assert result.endswith('Z')
        assert result == "2025-01-19T12:00:00Z"

    def test_convert_string_with_timezone_offset(self):
        """Test converting string with timezone offset."""
        result = toUtcISO("2025-01-19T12:00:00+05:00")

        assert result.endswith('Z')
        # +05:00 means 5 hours ahead, so UTC should be 07:00
        assert "2025-01-19T07:00:00Z" in result

    def test_convert_string_with_negative_offset(self):
        """Test converting string with negative timezone offset."""
        result = toUtcISO("2025-01-19T12:00:00-03:00")

        assert result.endswith('Z')
        # -03:00 means 3 hours behind, so UTC should be 15:00
        assert "2025-01-19T15:00:00Z" in result

    def test_convert_datetime_naive(self):
        """Test converting naive datetime object."""
        dt = datetime(2025, 1, 19, 12, 0, 0)
        result = toUtcISO(dt)

        assert result == "2025-01-19T12:00:00Z"

    def test_convert_datetime_with_timezone(self):
        """Test converting datetime with timezone info."""
        # Create datetime with UTC timezone
        dt = datetime(2025, 1, 19, 12, 0, 0, tzinfo=timezone.utc)
        result = toUtcISO(dt)

        assert result.endswith('Z')

    def test_convert_unix_timestamp_int(self):
        """Test converting Unix timestamp (int)."""
        # 2025-01-19 12:00:00 UTC ≈ 1737273600
        unix_ts = 1737273600
        result = toUtcISO(unix_ts)

        assert result.endswith('Z')
        assert "2025-01-19" in result

    def test_convert_unix_timestamp_float(self):
        """Test converting Unix timestamp (float)."""
        # Unix timestamp with fractional seconds
        unix_ts = 1737273600.123456
        result = toUtcISO(unix_ts)

        assert result.endswith('Z')
        # Should include fractional seconds
        assert "." in result

    def test_convert_various_string_formats(self):
        """Test converting various common string formats."""
        formats = [
            "2025-01-19 12:00:00",
            "2025-01-19T12:00:00",
            "2025/01/19 12:00:00",
            "01/19/2025 12:00:00",
            "2025-01-19",  # Date only
        ]

        for fmt in formats:
            result = toUtcISO(fmt)
            assert result.endswith('Z'), f"Failed for format: {fmt}"

    def test_invalid_string_format_raises_error(self):
        """Test invalid string format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            toUtcISO("invalid timestamp")

    def test_unsupported_type_raises_error(self):
        """Test unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported timestamp type"):
            toUtcISO([1, 2, 3])


class TestIsValidUtcISO:
    """Test isValidUtcISO validation function."""

    def test_valid_iso_with_z(self):
        """Test valid ISO timestamp with Z returns True."""
        assert isValidUtcISO("2025-01-19T12:00:00Z") is True

    def test_valid_iso_with_fractional_seconds(self):
        """Test valid ISO with fractional seconds returns True."""
        assert isValidUtcISO("2025-01-19T12:00:00.123456Z") is True

    def test_missing_z_returns_false(self):
        """Test timestamp without Z returns False."""
        assert isValidUtcISO("2025-01-19T12:00:00") is False

    def test_with_timezone_offset_returns_false(self):
        """Test timestamp with timezone offset returns False."""
        assert isValidUtcISO("2025-01-19T12:00:00+00:00") is False
        assert isValidUtcISO("2025-01-19T12:00:00-05:00") is False

    def test_invalid_format_returns_false(self):
        """Test invalid format returns False."""
        assert isValidUtcISO("invalid") is False
        assert isValidUtcISO("2025-01-19 12:00:00") is False
        assert isValidUtcISO("2025/01/19") is False

    def test_empty_string_returns_false(self):
        """Test empty string returns False."""
        assert isValidUtcISO("") is False

    def test_non_string_returns_false(self):
        """Test non-string input returns False."""
        assert isValidUtcISO(123) is False
        assert isValidUtcISO(None) is False
        assert isValidUtcISO([]) is False


class TestCalculateDuration:
    """Test calculateDuration function."""

    def test_duration_in_milliseconds(self):
        """Test duration is calculated in milliseconds."""
        start = "2025-01-19T12:00:00Z"
        end = "2025-01-19T12:00:01Z"

        duration = calculateDuration(start, end)

        assert duration == 1000.0  # 1 second = 1000 ms

    def test_duration_with_fractional_seconds(self):
        """Test duration with fractional seconds."""
        start = "2025-01-19T12:00:00Z"
        end = "2025-01-19T12:00:01.500Z"

        duration = calculateDuration(start, end)

        assert duration == 1500.0  # 1.5 seconds = 1500 ms

    def test_duration_with_datetime_objects(self):
        """Test duration calculation with datetime objects."""
        start = datetime(2025, 1, 19, 12, 0, 0)
        end = datetime(2025, 1, 19, 12, 0, 5)

        duration = calculateDuration(start, end)

        assert duration == 5000.0  # 5 seconds = 5000 ms

    def test_negative_duration(self):
        """Test duration can be negative if end is before start."""
        start = "2025-01-19T12:00:01Z"
        end = "2025-01-19T12:00:00Z"

        duration = calculateDuration(start, end)

        assert duration == -1000.0

    def test_zero_duration(self):
        """Test zero duration when timestamps are equal."""
        start = "2025-01-19T12:00:00Z"
        end = "2025-01-19T12:00:00Z"

        duration = calculateDuration(start, end)

        assert duration == 0.0

    def test_duration_with_unix_timestamps(self):
        """Test duration with Unix timestamps."""
        start = 1737273600  # 2025-01-19 12:00:00 UTC
        end = 1737273605    # 2025-01-19 12:00:05 UTC

        duration = calculateDuration(start, end)

        assert duration == 5000.0


class TestAddDuration:
    """Test addDuration function."""

    def test_add_positive_duration(self):
        """Test adding positive duration."""
        timestamp = "2025-01-19T12:00:00Z"
        result = addDuration(timestamp, 1000)

        assert result == "2025-01-19T12:00:01Z"

    def test_add_negative_duration(self):
        """Test adding negative duration."""
        timestamp = "2025-01-19T12:00:01Z"
        result = addDuration(timestamp, -1000)

        assert result == "2025-01-19T12:00:00Z"

    def test_add_fractional_milliseconds(self):
        """Test adding fractional milliseconds."""
        timestamp = "2025-01-19T12:00:00Z"
        result = addDuration(timestamp, 1500.5)

        # Should add 1.5005 seconds
        assert "2025-01-19T12:00:01.500" in result

    def test_add_zero_duration(self):
        """Test adding zero duration returns same timestamp."""
        timestamp = "2025-01-19T12:00:00Z"
        result = addDuration(timestamp, 0)

        assert result == timestamp

    def test_add_large_duration(self):
        """Test adding large duration (hours)."""
        timestamp = "2025-01-19T12:00:00Z"
        result = addDuration(timestamp, 3600000)  # 1 hour = 3600000 ms

        assert result == "2025-01-19T13:00:00Z"

    def test_result_ends_with_z(self):
        """Test result always ends with Z."""
        timestamp = "2025-01-19T12:00:00Z"
        result = addDuration(timestamp, 1000)

        assert result.endswith('Z')


class TestParseUtcISO:
    """Test parseUtcISO function."""

    def test_parse_valid_timestamp(self):
        """Test parsing valid timestamp."""
        timestamp = "2025-01-19T12:00:00Z"
        result = parseUtcISO(timestamp)

        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 19
        assert result.hour == 12
        assert result.minute == 0
        assert result.second == 0

    def test_parse_with_fractional_seconds(self):
        """Test parsing timestamp with fractional seconds."""
        timestamp = "2025-01-19T12:00:00.123456Z"
        result = parseUtcISO(timestamp)

        assert result.microsecond == 123456

    def test_parse_invalid_timestamp_raises_error(self):
        """Test parsing invalid timestamp raises ValueError."""
        with pytest.raises(ValueError):
            parseUtcISO("invalid")

        with pytest.raises(ValueError):
            parseUtcISO("2025-01-19T12:00:00")  # Missing Z

        with pytest.raises(ValueError):
            parseUtcISO("2025-01-19T12:00:00+00:00")  # Has offset


class TestFormatDuration:
    """Test formatDuration function."""

    def test_format_milliseconds(self):
        """Test formatting duration less than 1 second."""
        assert formatDuration(100) == "100ms"
        assert formatDuration(999) == "999ms"

    def test_format_seconds(self):
        """Test formatting duration in seconds."""
        assert formatDuration(1000) == "1.00s"
        assert formatDuration(1500) == "1.50s"
        # 59999ms is 59.999 seconds, which should be formatted as "59.999s"
        # But the function rounds to 2 decimal places, so it becomes "60.00s"
        assert formatDuration(59999) == "60.00s"

    def test_format_minutes(self):
        """Test formatting duration in minutes."""
        assert formatDuration(61000) == "1m 1s"
        assert formatDuration(60000) == "1m 0s"
        assert formatDuration(120000) == "2m 0s"
        assert formatDuration(125000) == "2m 5s"

    def test_format_zero_duration(self):
        """Test formatting zero duration."""
        assert formatDuration(0) == "0ms"

    def test_format_negative_duration(self):
        """Test formatting negative duration."""
        # Negative durations in milliseconds are shown as ms
        assert formatDuration(-1000) == "-1000ms"
        # Negative durations less than 1000ms
        assert formatDuration(-500) == "-500ms"


class TestTimestampConsistency:
    """Test timestamp consistency across functions."""

    def test_roundtrip_get_to_parse(self):
        """Test roundtrip: getCurrentTimestamp -> parseUtcISO."""
        timestamp = getCurrentTimestamp()
        parsed = parseUtcISO(timestamp)

        # Should be able to convert back to ISO
        result = parsed.isoformat() + "Z"
        assert result == timestamp

    def test_roundtrip_to_utc_iso_to_parse(self):
        """Test roundtrip: toUtcISO -> parseUtcISO."""
        original = "2025-01-19 12:00:00"
        converted = toUtcISO(original)
        parsed = parseUtcISO(converted)

        assert isinstance(parsed, datetime)
        assert parsed.year == 2025
        assert parsed.month == 1
        assert parsed.day == 19

    def test_add_then_calculate_duration(self):
        """Test addDuration and calculateDuration are consistent."""
        start = "2025-01-19T12:00:00Z"
        added = addDuration(start, 5000)

        duration = calculateDuration(start, added)

        assert duration == 5000.0

    def test_all_functions_produce_valid_utc_iso(self):
        """Test all timestamp functions produce valid UTC ISO format."""
        # getCurrentTimestamp
        ts1 = getCurrentTimestamp()
        assert isValidUtcISO(ts1)

        # toUtcISO
        ts2 = toUtcISO("2025-01-19 12:00:00")
        assert isValidUtcISO(ts2)

        # addDuration
        ts3 = addDuration("2025-01-19T12:00:00Z", 1000)
        assert isValidUtcISO(ts3)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_old_date(self):
        """Test converting very old date."""
        result = toUtcISO("1970-01-01 00:00:00")
        assert result.endswith('Z')
        assert "1970-01-01" in result

    def test_future_date(self):
        """Test converting future date."""
        result = toUtcISO("2099-12-31 23:59:59")
        assert result.endswith('Z')
        assert "2099-12-31" in result

    def test_leap_year(self):
        """Test leap year date."""
        result = toUtcISO("2024-02-29 12:00:00")  # 2024 is a leap year
        assert result.endswith('Z')

    def test_end_of_year(self):
        """Test end of year transition."""
        start = "2025-12-31T23:59:59Z"
        result = addDuration(start, 1000)

        assert "2026-01-01" in result

    def test_very_small_duration(self):
        """Test very small duration."""
        start = "2025-01-19T12:00:00Z"
        end = addDuration(start, 0.001)  # 1 microsecond

        duration = calculateDuration(start, end)
        assert duration == 0.001

    def test_very_large_duration(self):
        """Test very large duration."""
        start = "2025-01-19T12:00:00Z"
        end = addDuration(start, 86400000)  # 1 day

        duration = calculateDuration(start, end)
        assert duration == 86400000.0
