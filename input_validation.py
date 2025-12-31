"""
Sovereign-Grade Problem Decomposition System - Input Validation and Sanitization
Implements comprehensive validation for all user inputs and external data.
"""

import re
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import html
import urllib.parse
import json
import logging
import bleach
from dataclasses import is_dataclass, fields


logger = logging.getLogger(__name__)


class ValidationRule(Enum):
    """Types of validation rules available"""
    NOT_EMPTY = "not_empty"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    TYPE = "type"
    RANGE = "range"
    EMAIL = "email"
    URL = "url"
    SANITIZE_HTML = "sanitize_html"
    NO_SCRIPT = "no_script"


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for field '{field}': {message}")


class ValidationRuleConfig:
    """Configuration for a validation rule"""
    
    def __init__(self, rule: ValidationRule, value: Any = None, params: Optional[Dict[str, Any]] = None):
        self.rule = rule
        self.value = value
        self.params = params or {}


class InputValidator:
    """Main input validation class"""
    
    def __init__(self):
        # Common patterns for validation
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'username': r'^[a-zA-Z0-9_]{3,20}$',  # 3-20 chars, alphanumeric and underscore only
            'password': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$',  # At least 8 chars with letter and number
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'alphanumeric': r'^[a-zA-Z0-9]+$',
            'no_special_chars': r'^[a-zA-Z0-9\s\-_]+$',
        }
        
        # HTML sanitization settings
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img',
            'table', 'thead', 'tbody', 'tr', 'th', 'td'
        ]
        self.allowed_attributes = {
            'a': ['href', 'title', 'target'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'code': ['class'],
            'pre': ['class']
        }
    
    def validate(self, data: Any, field_name: str, rules: List[ValidationRuleConfig]) -> Any:
        """Validate a single field against multiple rules"""
        errors = []
        
        for rule_config in rules:
            try:
                if rule_config.rule == ValidationRule.NOT_EMPTY:
                    self._validate_not_empty(data, field_name)
                elif rule_config.rule == ValidationRule.MIN_LENGTH:
                    self._validate_min_length(data, field_name, rule_config.value)
                elif rule_config.rule == ValidationRule.MAX_LENGTH:
                    self._validate_max_length(data, field_name, rule_config.value)
                elif rule_config.rule == ValidationRule.PATTERN:
                    self._validate_pattern(data, field_name, rule_config.value)
                elif rule_config.rule == ValidationRule.TYPE:
                    data = self._validate_type(data, field_name, rule_config.value)
                elif rule_config.rule == ValidationRule.RANGE:
                    self._validate_range(data, field_name, rule_config.params.get('min'), rule_config.params.get('max'))
                elif rule_config.rule == ValidationRule.EMAIL:
                    self._validate_email(data, field_name)
                elif rule_config.rule == ValidationRule.URL:
                    self._validate_url(data, field_name)
                elif rule_config.rule == ValidationRule.SANITIZE_HTML:
                    data = self._sanitize_html(data, field_name)
                elif rule_config.rule == ValidationRule.NO_SCRIPT:
                    data = self._remove_script_tags(data)
            except ValidationError as e:
                errors.append(str(e))
        
        if errors:
            raise ValidationError(field_name, "; ".join(errors), data)
        
        return data
    
    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, List[ValidationRuleConfig]]) -> Dict[str, Any]:
        """Validate entire data schema"""
        validated_data = {}
        errors = {}
        
        for field_name, rules in schema.items():
            field_value = data.get(field_name)
            
            try:
                validated_value = self.validate(field_value, field_name, rules)
                validated_data[field_name] = validated_value
            except ValidationError as e:
                errors[field_name] = str(e)
        
        if errors:
            raise ValidationError("schema", json.dumps(errors))
        
        return validated_data
    
    def _validate_not_empty(self, value: Any, field_name: str):
        """Validate that value is not empty"""
        if value is None:
            raise ValidationError(field_name, "Value cannot be null")
        if isinstance(value, (str, list, dict)):
            if len(value) == 0:
                raise ValidationError(field_name, "Value cannot be empty")
        elif value == "":
            raise ValidationError(field_name, "Value cannot be empty string")
    
    def _validate_min_length(self, value: Any, field_name: str, min_length: int):
        """Validate minimum length"""
        if value is not None and hasattr(value, '__len__'):
            if len(value) < min_length:
                raise ValidationError(field_name, f"Value must be at least {min_length} characters")
    
    def _validate_max_length(self, value: Any, field_name: str, max_length: int):
        """Validate maximum length"""
        if value is not None and hasattr(value, '__len__'):
            if len(value) > max_length:
                raise ValidationError(field_name, f"Value must be no more than {max_length} characters")
    
    def _validate_pattern(self, value: str, field_name: str, pattern: str):
        """Validate against a regex pattern"""
        if value is not None:
            if not re.match(pattern, str(value)):
                raise ValidationError(field_name, f"Value does not match required pattern: {pattern}")
    
    def _validate_type(self, value: Any, field_name: str, expected_type: type) -> Any:
        """Validate type and convert if necessary"""
        if value is None:
            return value  # Allow None values
        
        if not isinstance(value, expected_type):
            try:
                # Try to convert to expected type
                if expected_type == int:
                    return int(value)
                elif expected_type == float:
                    return float(value)
                elif expected_type == str:
                    return str(value)
                elif expected_type == bool:
                    if isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes', 'on')
                    return bool(value)
                else:
                    raise ValidationError(field_name, f"Cannot convert value to {expected_type.__name__}")
            except (ValueError, TypeError):
                raise ValidationError(field_name, f"Value must be of type {expected_type.__name__}")
        
        return value
    
    def _validate_range(self, value: Union[int, float], field_name: str, min_val: Union[int, float], max_val: Union[int, float]):
        """Validate numeric range"""
        if value is not None:
            if not isinstance(value, (int, float)):
                raise ValidationError(field_name, "Value must be numeric for range validation")
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                raise ValidationError(field_name, f"Value must be between {min_val} and {max_val}")
    
    def _validate_email(self, value: str, field_name: str):
        """Validate email format"""
        if value is not None:
            value = str(value)
            if not re.match(self.patterns['email'], value):
                raise ValidationError(field_name, "Invalid email format")
    
    def _validate_url(self, value: str, field_name: str):
        """Validate URL format"""
        if value is not None:
            value = str(value)
            if not re.match(self.patterns['url'], value):
                raise ValidationError(field_name, "Invalid URL format")
    
    def _sanitize_html(self, value: str, field_name: str) -> str:
        """Sanitize HTML content"""
        if value is not None:
            value = str(value)
            # Use bleach to sanitize HTML
            sanitized = bleach.clean(
                value, 
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
            return sanitized
        return value
    
    def _remove_script_tags(self, value: str) -> str:
        """Remove script tags and other potentially dangerous content"""
        if value is not None:
            value = str(value)
            # Remove script tags (case-insensitive)
            value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
            # Remove event handlers
            value = re.sub(r'on\w+\s*=', 'safe_', value, flags=re.IGNORECASE)
            # Remove javascript: protocol
            value = re.sub(r'javascript:', 'safe_javascript:', value, flags=re.IGNORECASE)
            return value
        return value
    
    def validate_problem_definition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete problem definition"""
        schema = {
            'id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'title': [
                ValidationRuleConfig(ValidationRule.NOT_EMPTY),
                ValidationRuleConfig(ValidationRule.MIN_LENGTH, 5),
                ValidationRuleConfig(ValidationRule.MAX_LENGTH, 200),
                ValidationRuleConfig(ValidationRule.SANITIZE_HTML)
            ],
            'description': [
                ValidationRuleConfig(ValidationRule.NOT_EMPTY),
                ValidationRuleConfig(ValidationRule.MIN_LENGTH, 10),
                ValidationRuleConfig(ValidationRule.MAX_LENGTH, 10000),
                ValidationRuleConfig(ValidationRule.SANITIZE_HTML)
            ],
            'problem_type': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'domain_context': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'complexity_score': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'constraints': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'success_criteria': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'stakeholders': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'resources_available': [ValidationRuleConfig(ValidationRule.TYPE, dict)],
            'deadline': [ValidationRuleConfig(ValidationRule.TYPE, str)],  # Could be ISO date string
            'created_at': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'updated_at': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'metadata': [ValidationRuleConfig(ValidationRule.TYPE, dict)]
        }
        
        return self.validate_schema(data, schema)
    
    def validate_decomposition_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a decomposition plan"""
        schema = {
            'id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'problem_id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'strategy': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'sub_problems': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'dependency_graph': [ValidationRuleConfig(ValidationRule.TYPE, dict)],
            'validation_checkpoints': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'quality_scores': [ValidationRuleConfig(ValidationRule.TYPE, dict)],
            'confidence_level': [ValidationRuleConfig(ValidationRule.RANGE, params={'min': 0.0, 'max': 1.0})],
            'created_by': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'approved_by': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'status': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'created_at': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'updated_at': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'metadata': [ValidationRuleConfig(ValidationRule.TYPE, dict)]
        }
        
        return self.validate_schema(data, schema)
    
    def validate_solution_attempt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a solution attempt"""
        schema = {
            'id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'sub_problem_id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'approach': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'solution_content': [
                ValidationRuleConfig(ValidationRule.NOT_EMPTY),
                ValidationRuleConfig(ValidationRule.SANITIZE_HTML)
            ],
            'team_id': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'confidence_score': [ValidationRuleConfig(ValidationRule.RANGE, params={'min': 0.0, 'max': 1.0})],
            'validation_results': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'feedback': [ValidationRuleConfig(ValidationRule.TYPE, list)],
            'status': [ValidationRuleConfig(ValidationRule.NOT_EMPTY)],
            'created_at': [ValidationRuleConfig(ValidationRule.TYPE, str)],
            'metadata': [ValidationRuleConfig(ValidationRule.TYPE, dict)]
        }
        
        return self.validate_schema(data, schema)
    
    def sanitize_json_input(self, json_str: str) -> str:
        """Sanitize JSON input for potential malicious content"""
        try:
            # Parse the JSON to validate structure
            parsed = json.loads(json_str)
            
            # Sanitize and validate the parsed data
            sanitized = self._sanitize_recursive(parsed)
            
            # Return sanitized JSON string
            return json.dumps(sanitized)
        except json.JSONDecodeError:
            raise ValidationError("json_input", "Invalid JSON format")
    
    def _sanitize_recursive(self, obj: Any) -> Any:
        """Recursively sanitize an object"""
        if isinstance(obj, dict):
            return {k: self._sanitize_recursive(v) for k, v in obj.items() if not self._is_dangerous_key(k)}
        elif isinstance(obj, list):
            return [self._sanitize_recursive(item) for item in obj]
        elif isinstance(obj, str):
            # Sanitize string content
            sanitized = self._remove_script_tags(obj)
            return bleach.clean(sanitized, tags=self.allowed_tags, attributes=self.allowed_attributes, strip=True)
        else:
            return obj
    
    def _is_dangerous_key(self, key: str) -> bool:
        """Check if a key name is potentially dangerous"""
        dangerous_patterns = [
            r'\$.*',  # MongoDB operators
            r'.*password.*',  # Password fields
            r'.*secret.*',  # Secret fields
            r'.*token.*',  # Token fields
            r'.*key.*'  # Key fields
        ]
        
        key_lower = key.lower()
        for pattern in dangerous_patterns:
            if re.match(pattern, key_lower):
                return True
        return False


class Sanitizer:
    """Data sanitization utilities"""
    
    def __init__(self):
        self.validator = InputValidator()
    
    def sanitize_text(self, text: str) -> str:
        """Basic text sanitization"""
        if not isinstance(text, str):
            return str(text)
        
        # HTML escape
        sanitized = html.escape(text)
        # Remove potentially dangerous patterns
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal"""
        # Remove path traversal attempts
        filename = re.sub(r'\.\./', '', filename)
        filename = re.sub(r'\.\.\\', '', filename)
        
        # Only allow safe characters in filename
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        return filename
    
    def sanitize_url(self, url: str) -> str:
        """Sanitize URL to prevent open redirect attacks"""
        try:
            parsed = urllib.parse.urlparse(url)
            # Only allow http and https protocols
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError('url', 'Invalid URL scheme. Only http and https are allowed.')
            return url
        except Exception:
            raise ValidationError('url', 'Invalid URL format.')
    
    def sanitize_dataclass(self, obj: Any) -> Any:
        """Sanitize a dataclass object"""
        if not is_dataclass(obj):
            return obj
        
        for field in fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, str):
                setattr(obj, field.name, self.sanitize_text(value))
            elif isinstance(value, (list, tuple)):
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_list.append(self.sanitize_text(item))
                    else:
                        sanitized_list.append(self.sanitize_dataclass(item))
                setattr(obj, field.name, type(value)(sanitized_list))
            elif is_dataclass(value):
                setattr(obj, field.name, self.sanitize_dataclass(value))
        
        return obj


# Global validator instance
_input_validator = None
_sanitizer = None


def get_validator() -> InputValidator:
    """Get the input validator instance"""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_sanitizer() -> Sanitizer:
    """Get the sanitizer instance"""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = Sanitizer()
    return _sanitizer


# Example usage
if __name__ == "__main__":
    validator = get_validator()
    sanitizer = get_sanitizer()
    
    # Example: validate a problem definition
    problem_data = {
        'id': 'prob_123',
        'title': 'Example Problem Title',
        'description': '<p>This is a problem description with <strong>HTML</strong>.</p><script>alert("malicious")</script>',
        'problem_type': 'research',
        'domain_context': {'domain': 'software_engineering'},
        'complexity_score': {'overall_complexity': 7.5},
        'constraints': [],
        'success_criteria': [],
        'stakeholders': [],
        'resources_available': {},
        'created_at': '2023-01-01T00:00:00',
        'updated_at': '2023-01-01T00:00:00',
        'metadata': {}
    }
    
    try:
        validated_problem = validator.validate_problem_definition(problem_data)
        print("Problem validation successful!")
        print(f"Sanitized description: {validated_problem['description']}")
    except ValidationError as e:
        print(f"Validation failed: {e}")
    
    # Example: sanitize text
    malicious_text = '<script>alert("XSS")</script><p>Normal content</p>'
    sanitized_text = sanitizer.sanitize_text(malicious_text)
    print(f"Original: {malicious_text}")
    print(f"Sanitized: {sanitized_text}")