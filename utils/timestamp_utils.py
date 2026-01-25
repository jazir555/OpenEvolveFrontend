"""
Timestamp Utilities for OpenEvolve

Provides UTC timestamp handling following the Law of UTC.
All timestamps are in UTC ISO-8601 format.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import re
import isodate


def getCurrentTimestamp() -> str:
    """
    Get current UTC timestamp in ISO-8601 format.

    Returns:
        UTC timestamp string ending with 'Z' (e.g., "2025-01-19T12:00:00.000000Z")

    Example:
        >>> getCurrentTimestamp()
        '2025-01-19T12:00:00.123456Z'
    """
    return datetime.utcnow().isoformat() + "Z"


def toUtcISO(timestamp: Union[str, datetime, int, float]) -> str:
    """
    Convert various timestamp formats to UTC ISO-8601.

    Args:
        timestamp: Input timestamp (string, datetime, or Unix timestamp)

    Returns:
        UTC ISO-8601 string ending with 'Z'

    Raises:
        ValueError: If timestamp format is invalid

    Examples:
        >>> toUtcISO("2025-01-19 12:00:00")
        '2025-01-19T12:00:00Z'
        >>> toUtcISO(datetime(2025, 1, 19, 12, 0, 0))
        '2025-01-19T12:00:00Z'
        >>> toUtcISO(1705689600)  # Unix timestamp
        '2025-01-19T20:00:00Z'
    """
    if isinstance(timestamp, str):
        # Try parsing various string formats
        timestamp = timestamp.strip()

        # Already has Z (UTC)
        if timestamp.endswith('Z'):
            # Ensure ISO format
            try:
                # Parse and reformat to ensure consistency
                dt = datetime.fromisoformat(timestamp.replace('Z', ''))
                return dt.isoformat() + "Z"
            except ValueError:
                # Try parsing with timezone info
                try:
                    dt = isodate.parse_datetime(timestamp)
                    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                except:
                    raise ValueError(f"Invalid timestamp format: {timestamp}")

        # Has timezone offset (+/-)
        if re.search(r'[+-]\d{2}:\d{2}$', timestamp):
            try:
                dt = isodate.parse_datetime(timestamp)
                return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            except:
                raise ValueError(f"Invalid timestamp format: {timestamp}")

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp, fmt)
                return dt.isoformat() + "Z"
            except ValueError:
                continue

        # Try parsing as ISO format without timezone
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.isoformat() + "Z"
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

    elif isinstance(timestamp, datetime):
        # Convert datetime to UTC ISO format
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.isoformat() + "Z"

    elif isinstance(timestamp, (int, float)):
        # Unix timestamp
        dt = datetime.utcfromtimestamp(timestamp)
        return dt.isoformat() + "Z"

    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def isValidUtcISO(timestamp: str) -> bool:
    """
    Validate if string is a valid UTC ISO-8601 timestamp.

    Args:
        timestamp: String to validate

    Returns:
        True if valid UTC ISO-8601 format, False otherwise

    Examples:
        >>> isValidUtcISO("2025-01-19T12:00:00Z")
        True
        >>> isValidUtcISO("2025-01-19T12:00:00+00:00")
        False  # Has offset instead of Z
        >>> isValidUtcISO("2025-01-19 12:00:00")
        False
    """
    if not isinstance(timestamp, str):
        return False

    timestamp = timestamp.strip()

    # Must end with Z
    if not timestamp.endswith('Z'):
        return False

    # Remove Z and parse
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', ''))
        # Ensure it's a valid datetime
        return True
    except ValueError:
        return False


def calculateDuration(start_time: Union[str, datetime], end_time: Union[str, datetime]) -> float:
    """
    Calculate duration between two timestamps in milliseconds.

    Args:
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        Duration in milliseconds

    Examples:
        >>> start = "2025-01-19T12:00:00Z"
        >>> end = "2025-01-19T12:00:01Z"
        >>> calculateDuration(start, end)
        1000.0
    """
    # Convert to UTC ISO if needed
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', ''))
    elif not isinstance(start_time, datetime):
        start_time = datetime.utcfromtimestamp(start_time)

    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace('Z', ''))
    elif not isinstance(end_time, datetime):
        end_time = datetime.utcfromtimestamp(end_time)

    duration = end_time - start_time
    return duration.total_seconds() * 1000  # Convert to milliseconds


def addDuration(timestamp: Union[str, datetime], milliseconds: float) -> str:
    """
    Add duration to timestamp.

    Args:
        timestamp: Base timestamp
        milliseconds: Milliseconds to add (can be negative)

    Returns:
        New UTC ISO-8601 timestamp

    Examples:
        >>> addDuration("2025-01-19T12:00:00Z", 1000)
        '2025-01-19T12:00:01Z'
        >>> addDuration("2025-01-19T12:00:00Z", -1000)
        '2025-01-19T11:59:59Z'
    """
    # Convert to datetime
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', ''))
    else:
        dt = timestamp
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)

    # Add duration
    delta = timedelta(milliseconds=milliseconds)
    new_dt = dt + delta

    # Return UTC ISO format
    return new_dt.isoformat() + "Z"


def parseUtcISO(timestamp: str) -> datetime:
    """
    Parse UTC ISO-8601 timestamp to datetime object.

    Args:
        timestamp: UTC ISO-8601 string

    Returns:
        Naive datetime object (assumes UTC)

    Raises:
        ValueError: If timestamp format is invalid

    Examples:
        >>> parseUtcISO("2025-01-19T12:00:00Z")
        datetime.datetime(2025, 1, 19, 12, 0, 0)
    """
    if not isValidUtcISO(timestamp):
        raise ValueError(f"Invalid UTC ISO-8601 timestamp: {timestamp}")

    return datetime.fromisoformat(timestamp.replace('Z', ''))


def formatDuration(milliseconds: float) -> str:
    """
    Format duration in milliseconds to human-readable string.

    Args:
        milliseconds: Duration in milliseconds

    Returns:
        Formatted duration string

    Examples:
        >>> formatDuration(1000)
        '1.00s'
        >>> formatDuration(1500)
        '1.50s'
        >>> formatDuration(100)
        '100ms'
        >>> formatDuration(61000)
        '1m 1s'
    """
    if milliseconds < 1000:
        return f"{int(milliseconds)}ms"
    elif milliseconds < 60000:
        return f"{milliseconds / 1000:.2f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


# Alias for compatibility
getCurrentTimestamp = getCurrentTimestamp
toUtcISO = toUtcISO
isValidUtcISO = isValidUtcISO
calculateDuration = calculateDuration
addDuration = addDuration
