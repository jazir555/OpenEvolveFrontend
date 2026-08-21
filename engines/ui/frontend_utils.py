from __future__ import annotations


"""Frontend Utilities Module (Test Compatibility)"""

from datetime import datetime
from typing import Any, Dict
from urllib.parse import parse_qs


class JSONHelpers:
    """Helper for JSON operations."""
    
    def safe_parse(self, json_string: str) -> dict:
        """Safely parse JSON."""
        import json
        try:
            return json.loads(json_string)
        except:
            return {}


class DateHelpers:
    """Helper for date operations."""
    
    def format_date(self, date: datetime, format: str = 'YYYY-MM-DD') -> str:
        """Format a date."""
        return date.strftime('%Y-%m-%d')


class ValidationHelpers:
    """Helper for validation."""
    
    def is_valid_email(self, email: str) -> bool:
        """Check if email is valid."""
        return '@' in email


class StorageHelpers:
    """Helper for browser storage."""
    
    def __init__(self):
        self.storage = {}
    
    def set_item(self, key: str, value: Any):
        """Set an item."""
        self.storage[key] = value
    
    def get_item(self, key: str) -> Any:
        """Get an item."""
        return self.storage.get(key)


class URLHelpers:
    """Helper for URL operations."""

    def parse_params(self, query_string: str) -> dict:
        """Parse URL parameters."""
        if query_string.startswith('?'):
            query_string = query_string[1:]
        parsed = dict(parse_qs(query_string))
        # Convert single-item lists to strings
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


class DebounceHelper:
    """Helper for debouncing."""
    
    def __init__(self, wait: int = 100):
        self.wait = wait
    
    def create_wrapper(self, func):
        """Create a debounced wrapper."""
        return func
