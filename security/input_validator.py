"""
RESE Security: Input Validation and Sanitization

Comprehensive input validation, sanitization, and security checking for all RESE components.

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import re
import json
import html
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Security configuration
MAX_STRING_LENGTH = 10000
MAX_DICT_DEPTH = 20
MAX_LIST_LENGTH = 10000
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

# Allowed patterns
SAFE_STRING_PATTERN = re.compile(r'^[\w\s\-.,!?;:()\[\]{}"\'/@#&%$*=+<>|~`]+$')
SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
SAFE_NUMERIC_PATTERN = re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$')


class SecuritySeverity(Enum):
    """Security issue severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Security issue detected during validation"""
    severity: SecuritySeverity
    category: str
    field: str
    message: str
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'severity': self.severity.value,
            'category': self.category,
            'field': self.field,
            'message': self.message,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat()
        }


class InputValidator:
    """
    Comprehensive input validation for all RESE inputs.

    Validates and sanitizes:
    - Problem descriptions
    - Constraints
    - Variables
    - File uploads
    - API requests
    - Lean 4 code
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize input validator.

        Args:
            strict_mode: If True, reject on any security issue
        """
        self.strict_mode = strict_mode
        self.issues: List[SecurityIssue] = []

        # Dangerous patterns to detect
        self.dangerous_patterns = {
            'sql_injection': [
                r"(\b(ALTER|CREATE|DELETE|DROP|EXEC|EXECUTE|INSERT|INTO|SELECT|UPDATE|UNION)\b)",
                r"(--|;|/\*|\*/|xp_|sp_)",
                r"('('|'')|'(\|)|or\s+1\s*=\s*1)"
            ],
            'code_injection': [
                r"(eval\s*\(|exec\s*\(|__import__\(|compile\s*\()",
                r"(\$\{.*\}|@\(.*\)|#\{.*\})",
                r"(<script|javascript:|on\w+\s*=)"
            ],
            'path_traversal': [
                r"(\.\./|\.\.\|\\|~)",
                r"(%2e%2e|%5c|%252e)"
            ],
            'xss': [
                r"(<script|<iframe|<object|<embed)",
                r"(javascript:|vbscript:|data:)",
                r"(onerror|onload|onclick|onmouseover)\s*="
            ]
        }

        # Compile dangerous patterns
        self.compiled_patterns = {}
        for category, patterns in self.dangerous_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def validate_problem_input(
        self,
        description: str,
        constraints: List[Dict[str, Any]],
        variables: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityIssue]]:
        """
        Validate complete problem input.

        Args:
            description: Problem description
            constraints: List of constraints
            variables: Problem variables

        Returns:
            Tuple of (is_valid, issues)
        """
        self.issues = []

        # Validate description
        self._validate_string(description, "description", max_length=MAX_STRING_LENGTH)

        # Validate constraints
        self._validate_constraints(constraints)

        # Validate variables
        self._validate_variables(variables)

        is_valid = len(self.issues) == 0 or not self.strict_mode

        return is_valid, self.issues

    def _validate_string(
        self,
        value: str,
        field_name: str,
        max_length: int = MAX_STRING_LENGTH,
        allow_html: bool = False
    ) -> None:
        """
        Validate string input.

        Args:
            value: String to validate
            field_name: Field name for reporting
            max_length: Maximum allowed length
            allow_html: Whether HTML is allowed
        """
        if not isinstance(value, str):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="type_validation",
                field=field_name,
                message=f"Expected string, got {type(value).__name__}",
                recommendation="Ensure field is a string"
            ))
            return

        # Check length
        if len(value) > max_length:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="length_validation",
                field=field_name,
                message=f"String exceeds maximum length of {max_length}",
                recommendation=f"Shorten {field_name} to {max_length} characters or less"
            ))

        # Check for dangerous patterns
        if not allow_html:
            self._check_dangerous_patterns(value, field_name)

        # Check for null bytes
        if '\x00' in value:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="null_byte_injection",
                field=field_name,
                message="String contains null bytes",
                recommendation="Remove null bytes from input"
            ))

    def _validate_constraints(self, constraints: List[Dict[str, Any]]) -> None:
        """
        Validate constraint list.

        Args:
            constraints: List of constraint dictionaries
        """
        if not isinstance(constraints, list):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="type_validation",
                field="constraints",
                message=f"Expected list, got {type(constraints).__name__}",
                recommendation="Ensure constraints is a list"
            ))
            return

        # Check list length
        if len(constraints) > MAX_LIST_LENGTH:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="length_validation",
                field="constraints",
                message=f"Too many constraints (max: {MAX_LIST_LENGTH})",
                recommendation=f"Reduce constraints to {MAX_LIST_LENGTH} or fewer"
            ))
            return

        # Validate each constraint
        for idx, constraint in enumerate(constraints):
            field_name = f"constraints[{idx}]"
            self._validate_dict(constraint, field_name, max_depth=3)

            # Validate constraint structure
            if isinstance(constraint, dict):
                # Validate ID if present
                if 'id' in constraint:
                    self._validate_identifier(constraint['id'], f"{field_name}.id")

                # Validate type if present
                if 'type' in constraint:
                    valid_types = ['HARD', 'SOFT', 'PREFERENCE']
                    if constraint['type'] not in valid_types:
                        self.issues.append(SecurityIssue(
                            severity=SecuritySeverity.MEDIUM,
                            category="enum_validation",
                            field=f"{field_name}.type",
                            message=f"Invalid constraint type: {constraint['type']}",
                            recommendation=f"Use one of: {valid_types}"
                        ))

                # Validate description
                if 'description' in constraint:
                    self._validate_string(
                        constraint['description'],
                        f"{field_name}.description",
                        max_length=5000
                    )

                # Validate formalization (Lean 4 code)
                if 'formalization' in constraint:
                    self._validate_lean4_code(
                        constraint['formalization'],
                        f"{field_name}.formalization"
                    )

    def _validate_variables(self, variables: Dict[str, Any], depth: int = 0) -> None:
        """
        Validate variables dictionary.

        Args:
            variables: Variables dictionary
            depth: Current nesting depth
        """
        if depth > MAX_DICT_DEPTH:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="depth_validation",
                field="variables",
                message=f"Maximum nesting depth exceeded: {depth}",
                recommendation="Reduce nesting depth"
            ))
            return

        if not isinstance(variables, dict):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="type_validation",
                field="variables",
                message=f"Expected dict, got {type(variables).__name__}",
                recommendation="Ensure variables is a dictionary"
            ))
            return

        # Validate each variable
        for key, value in variables.items():
            # Validate key
            self._validate_identifier(key, f"variables.{key}")

            # Validate value based on type
            if isinstance(value, str):
                self._validate_string(value, f"variables.{key}")
            elif isinstance(value, (int, float)):
                self._validate_numeric(value, f"variables.{key}")
            elif isinstance(value, list):
                if len(value) > MAX_LIST_LENGTH:
                    self.issues.append(SecurityIssue(
                        severity=SecuritySeverity.MEDIUM,
                        category="length_validation",
                        field=f"variables.{key}",
                        message=f"List too long (max: {MAX_LIST_LENGTH})",
                        recommendation="Reduce list length"
                    ))
                elif len(value) > 100:  # Check each element if not too long
                    for idx, item in enumerate(value):
                        if isinstance(item, str):
                            self._validate_string(item, f"variables.{key}[{idx}]", max_length=1000)
            elif isinstance(value, dict):
                self._validate_variables(value, depth + 1)

    def _validate_dict(
        self,
        value: Any,
        field_name: str,
        max_depth: int = MAX_DICT_DEPTH
    ) -> None:
        """
        Validate dictionary structure.

        Args:
            value: Value to validate
            field_name: Field name for reporting
            max_depth: Maximum nesting depth
        """
        if not isinstance(value, dict):
            return

        if len(value) > 1000:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="size_validation",
                field=field_name,
                message=f"Dictionary too large (max: 1000 keys)",
                recommendation="Reduce dictionary size"
            ))

        # Recursively validate nested structures
        for key, val in value.items():
            if isinstance(val, dict):
                self._validate_dict(val, f"{field_name}.{key}", max_depth - 1)
            elif isinstance(val, list):
                if len(val) > 100:
                    self.issues.append(SecurityIssue(
                        severity=SecuritySeverity.MEDIUM,
                        category="size_validation",
                        field=f"{field_name}.{key}",
                        message=f"List too long (max: 100)",
                        recommendation="Reduce list length"
                    ))

    def _validate_identifier(self, value: str, field_name: str) -> None:
        """
        Validate identifier (variable name, constraint ID, etc.).

        Args:
            value: Identifier to validate
            field_name: Field name for reporting
        """
        if not isinstance(value, str):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="type_validation",
                field=field_name,
                message=f"Expected string identifier, got {type(value).__name__}",
                recommendation="Ensure identifier is a string"
            ))
            return

        if len(value) > 100:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="length_validation",
                field=field_name,
                message=f"Identifier too long (max: 100 characters)",
                recommendation="Shorten identifier"
            ))
            return

        if not SAFE_IDENTIFIER_PATTERN.match(value):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="identifier_validation",
                field=field_name,
                message=f"Invalid identifier format: {value}",
                recommendation="Use alphanumeric characters and underscores only, starting with letter or underscore"
            ))

    def _validate_numeric(self, value: Union[int, float], field_name: str) -> None:
        """
        Validate numeric value.

        Args:
            value: Numeric value to validate
            field_name: Field name for reporting
        """
        if not isinstance(value, (int, float)):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="type_validation",
                field=field_name,
                message=f"Expected numeric, got {type(value).__name__}",
                recommendation="Ensure value is numeric"
            ))
            return

        # Check for NaN or Inf
        if isinstance(value, float):
            if value != value or abs(value) == float('inf'):
                self.issues.append(SecurityIssue(
                    severity=SecuritySeverity.MEDIUM,
                    category="numeric_validation",
                    field=field_name,
                    message=f"Invalid numeric value: {value}",
                    recommendation="Use finite numeric values only"
                ))

        # Check range (adjust as needed)
        if abs(value) > 1e308:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="range_validation",
                field=field_name,
                message=f"Numeric value out of range: {value}",
                recommendation="Use values within IEEE 754 range"
            ))

    def _validate_lean4_code(self, code: str, field_name: str) -> None:
        """
        Validate Lean 4 code for safety.

        Args:
            code: Lean 4 code to validate
            field_name: Field name for reporting
        """
        if not isinstance(code, str):
            return

        # Basic sanity checks
        if len(code) > 100000:  # 100KB limit per code block
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.MEDIUM,
                category="length_validation",
                field=field_name,
                message=f"Lean 4 code too long (max: 100KB)",
                recommendation="Split into smaller code blocks"
            ))

        # Check for potentially dangerous Lean constructs
        dangerous_lean_patterns = [
            (r'\b(eval|run|meta\s)', "metaprogramming"),
            (r'#(eval|check|reduce)', "command evaluation"),
            (r'\b(sorry|admit)', "proof admission"),
        ]

        for pattern, category in dangerous_lean_patterns:
            if re.search(pattern, code):
                self.issues.append(SecurityIssue(
                    severity=SecuritySeverity.LOW,
                    category="lean4_validation",
                    field=field_name,
                    message=f"Contains potentially unsafe Lean 4 construct: {category}",
                    recommendation=f"Review {category} usage for safety"
                ))

    def _check_dangerous_patterns(self, value: str, field_name: str) -> None:
        """
        Check for dangerous injection patterns.

        Args:
            value: String to check
            field_name: Field name for reporting
        """
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(value)
                if matches:
                    self.issues.append(SecurityIssue(
                        severity=SecuritySeverity.HIGH,
                        category=category,
                        field=field_name,
                        message=f"Detected potentially dangerous pattern: {category}",
                        recommendation="Remove or sanitize suspicious input"
                    ))

    def validate_file_upload(self, file_path: Path, max_size: int = MAX_FILE_SIZE_BYTES) -> Tuple[bool, List[SecurityIssue]]:
        """
        Validate uploaded file.

        Args:
            file_path: Path to uploaded file
            max_size: Maximum allowed file size

        Returns:
            Tuple of (is_valid, issues)
        """
        self.issues = []

        # Check file exists
        if not file_path.exists():
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="file_validation",
                field="file",
                message="File does not exist",
                recommendation="Ensure file path is correct"
            ))
            return False, self.issues

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.HIGH,
                category="file_validation",
                field="file",
                message=f"File too large: {file_size} bytes (max: {max_size})",
                recommendation=f"Reduce file size to {max_size} bytes or less"
            ))

        # Check file extension
        dangerous_extensions = {'.exe', '.bat', '.sh', '.cmd', '.scr', '.pif'}
        if file_path.suffix.lower() in dangerous_extensions:
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.CRITICAL,
                category="file_validation",
                field="file",
                message=f"Dangerous file extension: {file_path.suffix}",
                recommendation="Use safe file extensions only (.lean, .json, .txt)"
            ))

        # Check for path traversal in filename
        if '..' in str(file_path) or str(file_path).startswith('/'):
            self.issues.append(SecurityIssue(
                severity=SecuritySeverity.CRITICAL,
                category="path_traversal",
                field="file",
                message="Suspicious file path detected",
                recommendation="Use simple filenames without path traversal"
            ))

        is_valid = len([i for i in self.issues if i.severity == SecuritySeverity.CRITICAL]) == 0

        return is_valid, self.issues

    def sanitize_html(self, value: str) -> str:
        """
        Sanitize HTML by escaping dangerous characters.

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return value

        # Escape HTML
        sanitized = html.escape(value)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        return sanitized

    def sanitize_path(self, path: str) -> str:
        """
        Sanitize file path to prevent path traversal.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        if not isinstance(path, str):
            return ""

        # Remove path traversal attempts
        sanitized = path.replace('..', '').replace('\\', '/')

        # Remove leading slashes
        sanitized = sanitized.lstrip('/')

        # Only allow safe characters
        sanitized = re.sub(r'[^\w\s\-./]', '', sanitized)

        return sanitized


class SchemaValidator:
    """
    Validate data against JSON schemas.
    """

    def __init__(self):
        """Initialize schema validator"""
        self.schemas = {}

    def load_schema(self, schema_name: str, schema_path: Path) -> None:
        """
        Load validation schema from file.

        Args:
            schema_name: Name for the schema
            schema_path: Path to schema file
        """
        try:
            with open(schema_path, 'r') as f:
                self.schemas[schema_name] = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load schema {schema_name}: {e}")

    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema_name: Name of schema to use

        Returns:
            Tuple of (is_valid, errors)
        """
        if schema_name not in self.schemas:
            return False, [f"Schema not found: {schema_name}"]

        schema = self.schemas[schema_name]
        errors = []

        # Basic validation (for full JSON Schema validation, use jsonschema library)
        self._validate_schema(data, schema, "", errors)

        return len(errors) == 0, errors

    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[str]
    ) -> None:
        """
        Recursively validate data against schema.

        Args:
            data: Data to validate
            schema: Schema to validate against
            path: Current path in data
            errors: List to collect errors
        """
        # Type validation
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'string' and not isinstance(data, str):
                errors.append(f"{path}: Expected string, got {type(data).__name__}")
            elif expected_type == 'number' and not isinstance(data, (int, float)):
                errors.append(f"{path}: Expected number, got {type(data).__name__}")
            elif expected_type == 'integer' and not isinstance(data, int):
                errors.append(f"{path}: Expected integer, got {type(data).__name__}")
            elif expected_type == 'boolean' and not isinstance(data, bool):
                errors.append(f"{path}: Expected boolean, got {type(data).__name__}")
            elif expected_type == 'array' and not isinstance(data, list):
                errors.append(f"{path}: Expected array, got {type(data).__name__}")
            elif expected_type == 'object' and not isinstance(data, dict):
                errors.append(f"{path}: Expected object, got {type(data).__name__}")

        # Range validation
        if 'minimum' in schema and isinstance(data, (int, float)):
            if data < schema['minimum']:
                errors.append(f"{path}: Value {data} below minimum {schema['minimum']}")

        if 'maximum' in schema and isinstance(data, (int, float)):
            if data > schema['maximum']:
                errors.append(f"{path}: Value {data} above maximum {schema['maximum']}")

        # String length validation
        if 'minLength' in schema and isinstance(data, str):
            if len(data) < schema['minLength']:
                errors.append(f"{path}: String length {len(data)} below minimum {schema['minLength']}")

        if 'maxLength' in schema and isinstance(data, str):
            if len(data) > schema['maxLength']:
                errors.append(f"{path}: String length {len(data)} above maximum {schema['maxLength']}")


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_input(
    description: str,
    constraints: List[Dict[str, Any]],
    variables: Dict[str, Any],
    strict_mode: bool = True
) -> Tuple[bool, List[SecurityIssue]]:
    """
    Validate complete problem input.

    Args:
        description: Problem description
        constraints: List of constraints
        variables: Problem variables
        strict_mode: If True, reject on any issue

    Returns:
        Tuple of (is_valid, issues)
    """
    validator = InputValidator(strict_mode=strict_mode)
    return validator.validate_problem_input(description, constraints, variables)


def sanitize_input(value: str, allow_html: bool = False) -> str:
    """
    Sanitize user input string.

    Args:
        value: String to sanitize
        allow_html: Whether HTML is allowed

    Returns:
        Sanitized string
    """
    validator = InputValidator()
    if not allow_html:
        return validator.sanitize_html(value)
    return value


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'InputValidator',
    'SchemaValidator',
    'SecurityIssue',
    'SecuritySeverity',
    'validate_input',
    'sanitize_input',
]
