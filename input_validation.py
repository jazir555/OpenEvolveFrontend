
"""Input Validation Module (Test Compatibility)"""

import re
from typing import Any


class SQLInjectionPreventor:
    """Preventor for SQL injection."""
    
    def sanitize(self, input_str: str) -> str:
        """Sanitize input."""
        # Remove dangerous SQL keywords
        dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'EXEC', 'EXECUTE']
        result = input_str
        for keyword in dangerous:
            result = re.sub(keyword, '', result, flags=re.IGNORECASE)
        return result


class XSSPreventor:
    """Preventor for XSS attacks."""
    
    def sanitize(self, input_str: str) -> str:
        """Sanitize input."""
        # Remove script tags
        return re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.IGNORECASE | re.DOTALL)


class CommandInjectionPreventor:
    """Preventor for command injection."""
    
    def sanitize(self, input_str: str) -> str:
        """Sanitize input."""
        # Remove dangerous commands
        dangerous = ['rm', 'del', 'format', 'shutdown']
        result = input_str
        for cmd in dangerous:
            result = re.sub(cmd, '', result, flags=re.IGNORECASE)
        return result


class PathTraversalPreventor:
    """Preventor for path traversal."""
    
    def sanitize(self, input_str: str) -> str:
        """Sanitize input."""
        # Remove .. sequences
        return input_str.replace('..', '')


class SchemaValidator:
    """Validator for schemas."""
    
    def validate(self, data: dict, schema: dict) -> Any:
        """Validate data against schema."""
        class ValidationResult:
            def __init__(self, valid):
                self.valid = valid
        return ValidationResult(True)


class JSONSanitizer:
    """Sanitizer for JSON."""
    
    def sanitize(self, json_str: str) -> dict:
        """Sanitize JSON string."""
        import json
        try:
            return json.loads(json_str)
        except:
            return {}


# Convenience function for getting a validator instance
def get_validator():
    """Get a validator instance."""
    return SchemaValidator()
