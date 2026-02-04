"""Output validators for quality assurance.

Validates LLM outputs against various criteria including type safety,
pattern matching, content quality, and safety checks.

Following CLAUDE.md patterns:
- UTC timestamps for all validation events
- Structured logging with correlation_id
- Fail-safe defaults (validation fails on error)
- SSOT pattern for validation state
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Type, Union

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check.
    
    SSOT Pattern: This is the single source of truth for validation outcomes.
    """
    is_valid: bool
    validator_name: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    fixed_output: Optional[Any] = None
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for structured logging."""
        return {
            "is_valid": self.is_valid,
            "validator_name": self.validator_name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "details": self.details,
            "fixed_output": self.fixed_output,
            "suggestions": self.suggestions
        }


class Validator(ABC):
    """Base class for all validators.
    
    Implements fail-safe pattern: validation fails on error unless
    explicitly configured otherwise.
    """
    
    def __init__(self, name: Optional[str] = None, fail_safe: bool = True):
        self.name = name or self.__class__.__name__
        self.fail_safe = fail_safe  # If True, validation fails on error
        self._validation_count = 0
        self._failure_count = 0
        
    @abstractmethod
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate the output.
        
        Args:
            output: The output to validate
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ValidationResult with validation status
        """
        raise NotImplementedError
        
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Attempt to fix invalid output.
        
        Args:
            output: The invalid output
            validation_result: The validation result explaining the failure
            
        Returns:
            Fixed output or None if cannot fix
        """
        return None
        
    def explain(self, output: Any, result: ValidationResult) -> str:
        """Explain why validation failed.
        
        Args:
            output: The output that failed validation
            result: The validation result
            
        Returns:
            Human-readable explanation
        """
        return f"{self.name}: {result.message}"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "validator_name": self.name,
            "total_validations": self._validation_count,
            "failure_count": self._failure_count,
            "success_rate": 1.0 - (self._failure_count / max(1, self._validation_count))
        }
        
    def _create_result(
        self,
        is_valid: bool,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        fixed_output: Optional[Any] = None,
        suggestions: Optional[List[str]] = None
    ) -> ValidationResult:
        """Create a validation result."""
        self._validation_count += 1
        if not is_valid:
            self._failure_count += 1
            
        return ValidationResult(
            is_valid=is_valid,
            validator_name=self.name,
            message=message,
            severity=severity,
            correlation_id=correlation_id,
            details=details or {},
            fixed_output=fixed_output,
            suggestions=suggestions or []
        )


class TypeValidator(Validator):
    """Validate output type matches expected type."""
    
    def __init__(
        self,
        expected_type: Union[Type, Tuple[Type, ...]],
        allow_none: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.expected_type = expected_type
        self.allow_none = allow_none
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate output is of expected type."""
        try:
            if output is None:
                if self.allow_none:
                    return self._create_result(
                        True,
                        "Output is None (allowed)",
                        ValidationSeverity.INFO,
                        correlation_id
                    )
                return self._create_result(
                    False,
                    "Output is None but expected type " + str(self.expected_type),
                    ValidationSeverity.ERROR,
                    correlation_id,
                    suggestions=[f"Provide a value of type {self.expected_type}"]
                )
                
            if isinstance(output, self.expected_type):
                return self._create_result(
                    True,
                    f"Output is valid type: {self.expected_type}",
                    ValidationSeverity.INFO,
                    correlation_id,
                    details={"actual_type": type(output).__name__}
                )
                
            return self._create_result(
                False,
                f"Expected type {self.expected_type}, got {type(output).__name__}",
                ValidationSeverity.ERROR,
                correlation_id,
                details={
                    "expected_type": str(self.expected_type),
                    "actual_type": type(output).__name__
                },
                suggestions=[f"Convert output to {self.expected_type}"]
            )
            
        except Exception as e:
            logger.error({
                "msg": "Type validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Attempt type conversion."""
        if output is None:
            return None
            
        try:
            if self.expected_type == str:
                return str(output)
            elif self.expected_type == int:
                return int(output)
            elif self.expected_type == float:
                return float(output)
            elif self.expected_type == bool:
                return bool(output)
            elif self.expected_type == list and isinstance(output, (str, tuple, set)):
                return list(output)
            elif self.expected_type == dict and isinstance(output, str):
                return json.loads(output)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
            
        return None


class RegexValidator(Validator):
    """Validate output matches regex pattern."""
    
    def __init__(
        self,
        pattern: Union[str, Pattern],
        must_match: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.must_match = must_match  # If True, pattern must be found
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate output matches regex pattern."""
        try:
            if not isinstance(output, str):
                return self._create_result(
                    False,
                    f"Regex validation requires string input, got {type(output).__name__}",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    suggestions=["Ensure output is a string before regex validation"]
                )
                
            matches = self.pattern.findall(output)
            has_match = len(matches) > 0
            
            if self.must_match and not has_match:
                return self._create_result(
                    False,
                    f"Pattern '{self.pattern.pattern}' not found in output",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={"pattern": self.pattern.pattern, "output_length": len(output)},
                    suggestions=[f"Ensure output contains pattern: {self.pattern.pattern}"]
                )
                
            if not self.must_match and has_match:
                return self._create_result(
                    False,
                    f"Forbidden pattern '{self.pattern.pattern}' found in output",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={"pattern": self.pattern.pattern, "matches": matches},
                    suggestions=["Remove forbidden pattern from output"]
                )
                
            return self._create_result(
                True,
                f"Pattern validation passed",
                ValidationSeverity.INFO,
                correlation_id,
                details={"matches": matches}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Regex validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Regex validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise


class LengthValidator(Validator):
    """Validate output length constraints."""
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.min_length = min_length
        self.max_length = max_length
        
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length")
            
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate output length constraints."""
        try:
            length = len(output) if hasattr(output, '__len__') else None
            
            if length is None:
                return self._create_result(
                    False,
                    f"Output type {type(output).__name__} does not support length check",
                    ValidationSeverity.ERROR,
                    correlation_id
                )
                
            violations = []
            
            if self.min_length is not None and length < self.min_length:
                violations.append(f"Length {length} is below minimum {self.min_length}")
                
            if self.max_length is not None and length > self.max_length:
                violations.append(f"Length {length} exceeds maximum {self.max_length}")
                
            if violations:
                return self._create_result(
                    False,
                    "; ".join(violations),
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "length": length,
                        "min_length": self.min_length,
                        "max_length": self.max_length
                    },
                    suggestions=self._generate_fix_suggestions(length)
                )
                
            return self._create_result(
                True,
                f"Length {length} is within valid range",
                ValidationSeverity.INFO,
                correlation_id,
                details={"length": length}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Length validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Length validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def _generate_fix_suggestions(self, current_length: int) -> List[str]:
        """Generate suggestions for fixing length violations."""
        suggestions = []
        if self.min_length is not None and current_length < self.min_length:
            suggestions.append(f"Add {self.min_length - current_length} characters")
        if self.max_length is not None and current_length > self.max_length:
            suggestions.append(f"Remove {current_length - self.max_length} characters")
        return suggestions
        
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Attempt to fix length violations."""
        if not isinstance(output, (str, list)):
            return None
            
        try:
            if isinstance(output, str):
                if self.max_length is not None and len(output) > self.max_length:
                    return output[:self.max_length]
                if self.min_length is not None and len(output) < self.min_length:
                    return output + " " * (self.min_length - len(output))
            elif isinstance(output, list):
                if self.max_length is not None and len(output) > self.max_length:
                    return output[:self.max_length]
                # Cannot easily extend lists
        except Exception:
            pass
            
        return None


class RangeValidator(Validator):
    """Validate numeric range constraints."""
    
    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_int_only: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_int_only = allow_int_only
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate numeric range."""
        try:
            if not isinstance(output, (int, float)):
                return self._create_result(
                    False,
                    f"Expected numeric value, got {type(output).__name__}",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    suggestions=["Ensure output is a number"]
                )
                
            if self.allow_int_only and isinstance(output, float) and not output.is_integer():
                return self._create_result(
                    False,
                    f"Expected integer, got float: {output}",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    suggestions=[f"Round to nearest integer: {round(output)}"],
                    fixed_output=round(output)
                )
                
            violations = []
            
            if self.min_value is not None and output < self.min_value:
                violations.append(f"Value {output} is below minimum {self.min_value}")
                
            if self.max_value is not None and output > self.max_value:
                violations.append(f"Value {output} exceeds maximum {self.max_value}")
                
            if violations:
                return self._create_result(
                    False,
                    "; ".join(violations),
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "value": output,
                        "min_value": self.min_value,
                        "max_value": self.max_value
                    }
                )
                
            return self._create_result(
                True,
                f"Value {output} is within valid range",
                ValidationSeverity.INFO,
                correlation_id,
                details={"value": output}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Range validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Range validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Clamp value to valid range."""
        if not isinstance(output, (int, float)):
            return None
            
        result = output
        if self.min_value is not None:
            result = max(result, self.min_value)
        if self.max_value is not None:
            result = min(result, self.max_value)
            
        return int(result) if self.allow_int_only else result


class EnumValidator(Validator):
    """Validate output is one of allowed values."""
    
    def __init__(
        self,
        allowed_values: Set[Any],
        case_sensitive: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate output is in allowed values."""
        try:
            check_value = output if self.case_sensitive else str(output).lower()
            allowed = self.allowed_values if self.case_sensitive else {str(v).lower() for v in self.allowed_values}
            
            if check_value in allowed:
                return self._create_result(
                    True,
                    f"Value '{output}' is in allowed set",
                    ValidationSeverity.INFO,
                    correlation_id,
                    details={"value": output}
                )
                
            # Try to find closest match
            suggestions = self._find_similar(str(output))
            
            return self._create_result(
                False,
                f"Value '{output}' is not in allowed values: {self.allowed_values}",
                ValidationSeverity.ERROR,
                correlation_id,
                details={
                    "value": output,
                    "allowed_values": list(self.allowed_values)
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error({
                "msg": "Enum validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Enum validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def _find_similar(self, value: str) -> List[str]:
        """Find similar allowed values for suggestions."""
        # Simple edit distance for suggestions
        def edit_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        scored = [(v, edit_distance(value.lower(), str(v).lower())) for v in self.allowed_values]
        scored.sort(key=lambda x: x[1])
        return [f"Did you mean: '{v}'?" for v, score in scored[:3] if score <= 3]


class PIIValidator(Validator):
    """Detect personally identifiable information."""
    
    # Common PII patterns
    PII_PATTERNS = {
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    }
    
    def __init__(
        self,
        detect_types: Optional[List[str]] = None,
        block_on_detection: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.detect_types = set(detect_types) if detect_types else set(self.PII_PATTERNS.keys())
        self.block_on_detection = block_on_detection
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Detect PII in output."""
        try:
            if not isinstance(output, str):
                return self._create_result(
                    True,
                    "PII check skipped for non-string output",
                    ValidationSeverity.INFO,
                    correlation_id
                )
                
            detections = {}
            for pii_type in self.detect_types:
                if pii_type in self.PII_PATTERNS:
                    matches = self.PII_PATTERNS[pii_type].findall(output)
                    if matches:
                        detections[pii_type] = matches
                        
            if detections:
                message = f"PII detected: {list(detections.keys())}"
                if self.block_on_detection:
                    return self._create_result(
                        False,
                        message,
                        ValidationSeverity.CRITICAL,
                        correlation_id,
                        details={
                            "detections": {k: len(v) for k, v in detections.items()},
                            "detected_types": list(detections.keys())
                        },
                        suggestions=["Remove PII from output", "Use redaction to mask sensitive data"]
                    )
                else:
                    return self._create_result(
                        True,
                        message + " (allowed in permissive mode)",
                        ValidationSeverity.WARNING,
                        correlation_id,
                        details={"detections": list(detections.keys())}
                    )
                    
            return self._create_result(
                True,
                "No PII detected",
                ValidationSeverity.INFO,
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "PII validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"PII validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Redact detected PII."""
        if not isinstance(output, str):
            return None
            
        result = output
        for pii_type in self.detect_types:
            if pii_type in self.PII_PATTERNS:
                result = self.PII_PATTERNS[pii_type].sub(f"[REDACTED_{pii_type.upper()}]", result)
                
        return result


class ToxicityValidator(Validator):
    """Detect toxic or harmful content."""
    
    # Toxic keywords/patterns (simplified implementation)
    TOXIC_PATTERNS = {
        "hate_speech": re.compile(r'\b(hate|kill|die|destroy)\b', re.IGNORECASE),
        "harassment": re.compile(r'\b(stupid|idiot|loser|worthless)\b', re.IGNORECASE),
        "profanity": re.compile(r'\b(damn|hell|crap|stupid)\b', re.IGNORECASE),
    }
    
    def __init__(
        self,
        sensitivity: str = "medium",  # low, medium, high
        block_categories: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.sensitivity = sensitivity
        self.block_categories = set(block_categories) if block_categories else set(self.TOXIC_PATTERNS.keys())
        
        # Adjust patterns based on sensitivity
        if sensitivity == "low":
            self.block_categories = {"hate_speech"}
        elif sensitivity == "high":
            # Add more strict patterns for high sensitivity
            self.TOXIC_PATTERNS["strict"] = re.compile(r'\b(ugly|dumb|bad|wrong)\b', re.IGNORECASE)
            
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Detect toxic content in output."""
        try:
            if not isinstance(output, str):
                return self._create_result(
                    True,
                    "Toxicity check skipped for non-string output",
                    ValidationSeverity.INFO,
                    correlation_id
                )
                
            detections = {}
            for category in self.block_categories:
                if category in self.TOXIC_PATTERNS:
                    matches = self.TOXIC_PATTERNS[category].findall(output)
                    if matches:
                        detections[category] = len(matches)
                        
            if detections:
                return self._create_result(
                    False,
                    f"Potentially toxic content detected: {list(detections.keys())}",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "detections": detections,
                        "sensitivity": self.sensitivity
                    },
                    suggestions=["Rephrase content to be more neutral", "Remove flagged language"]
                )
                
            return self._create_result(
                True,
                "No toxic content detected",
                ValidationSeverity.INFO,
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "Toxicity validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Toxicity validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Attempt to clean toxic content."""
        if not isinstance(output, str):
            return None
            
        result = output
        # Replace toxic words with asterisks
        for category in self.block_categories:
            if category in self.TOXIC_PATTERNS:
                pattern = self.TOXIC_PATTERNS[category]
                result = pattern.sub(lambda m: '*' * len(m.group()), result)
                
        return result


class JSONValidator(Validator):
    """Validate JSON structure."""
    
    def __init__(self, allow_partial: bool = False, name: Optional[str] = None):
        super().__init__(name=name)
        self.allow_partial = allow_partial
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate JSON structure."""
        try:
            if isinstance(output, (dict, list)):
                return self._create_result(
                    True,
                    "Output is already a valid Python object",
                    ValidationSeverity.INFO,
                    correlation_id
                )
                
            if not isinstance(output, str):
                return self._create_result(
                    False,
                    f"JSON validation requires string or dict/list, got {type(output).__name__}",
                    ValidationSeverity.ERROR,
                    correlation_id
                )
                
            # Try to parse
            parsed = json.loads(output)
            
            return self._create_result(
                True,
                "Valid JSON structure",
                ValidationSeverity.INFO,
                correlation_id,
                details={"type": type(parsed).__name__}
            )
            
        except json.JSONDecodeError as e:
            suggestions = ["Check for trailing commas", "Verify quotes are properly escaped", "Validate brackets are balanced"]
            
            if self.allow_partial:
                # Try to extract valid JSON subset
                fixed = self._extract_json(output)
                if fixed:
                    return self._create_result(
                        True,
                        f"Partial JSON found and fixed: {str(e)}",
                        ValidationSeverity.WARNING,
                        correlation_id,
                        fixed_output=fixed,
                        suggestions=suggestions
                    )
                    
            return self._create_result(
                False,
                f"Invalid JSON: {str(e)}",
                ValidationSeverity.ERROR,
                correlation_id,
                details={"error": str(e)},
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error({
                "msg": "JSON validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"JSON validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def _extract_json(self, text: str) -> Optional[Any]:
        """Attempt to extract valid JSON from text."""
        # Try to find JSON objects or arrays
        for pattern in [r'\{[\s\S]*?\}', r'\[[\s\S]*?\]']:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        return None
        
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Try to fix JSON."""
        if not isinstance(output, str):
            return None
        return self._extract_json(output)


class SchemaValidator(Validator):
    """Validate against JSON Schema."""
    
    def __init__(
        self,
        schema: Dict[str, Any],
        strict: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.schema = schema
        self.strict = strict
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate output against JSON schema."""
        try:
            # First ensure it's valid JSON/object
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                except json.JSONDecodeError as e:
                    return self._create_result(
                        False,
                        f"Invalid JSON: {str(e)}",
                        ValidationSeverity.ERROR,
                        correlation_id
                    )
            else:
                data = output
                
            # Validate schema requirements
            violations = self._validate_schema(data, self.schema)
            
            if violations:
                return self._create_result(
                    False,
                    f"Schema validation failed: {len(violations)} violations",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "violations": violations,
                        "schema": self.schema
                    },
                    suggestions=[f"Fix: {v}" for v in violations[:5]]
                )
                
            return self._create_result(
                True,
                "Schema validation passed",
                ValidationSeverity.INFO,
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "Schema validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Schema validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def _validate_schema(self, data: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
        """Basic schema validation implementation."""
        violations = []
        
        # Check type
        schema_type = schema.get("type")
        if schema_type:
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict
            }
            expected = type_map.get(schema_type)
            if expected and not isinstance(data, expected):
                violations.append(f"{path}: Expected {schema_type}, got {type(data).__name__}")
                
        # Check required properties for objects
        if schema_type == "object" and "properties" in schema:
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    violations.append(f"{path}: Missing required property '{prop}'")
                    
        # Check enum
        if "enum" in schema and data not in schema["enum"]:
            violations.append(f"{path}: Value '{data}' not in enum {schema['enum']}")
            
        # Check nested properties
        if schema_type == "object" and isinstance(data, dict):
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop in data:
                    violations.extend(
                        self._validate_schema(data[prop], prop_schema, f"{path}.{prop}")
                    )
                    
        # Check array items
        if schema_type == "array" and isinstance(data, list):
            item_schema = schema.get("items")
            if item_schema:
                for i, item in enumerate(data):
                    violations.extend(
                        self._validate_schema(item, item_schema, f"{path}[{i}]")
                    )
                    
        return violations


class QualityValidator(Validator):
    """Validate quality score threshold."""
    
    def __init__(
        self,
        min_quality_score: float = 0.7,
        max_quality_score: float = 1.0,
        quality_metric: str = "overall",
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.min_quality_score = min_quality_score
        self.max_quality_score = max_quality_score
        self.quality_metric = quality_metric
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Validate quality score."""
        try:
            # Extract quality score from output
            score = self._extract_quality_score(output)
            
            if score is None:
                return self._create_result(
                    False,
                    f"Could not extract quality score using metric '{self.quality_metric}'",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    suggestions=["Ensure output contains quality metrics"]
                )
                
            if not isinstance(score, (int, float)):
                return self._create_result(
                    False,
                    f"Quality score must be numeric, got {type(score).__name__}",
                    ValidationSeverity.ERROR,
                    correlation_id
                )
                
            if score < self.min_quality_score:
                return self._create_result(
                    False,
                    f"Quality score {score:.2f} below minimum {self.min_quality_score}",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "score": score,
                        "min_score": self.min_quality_score,
                        "max_score": self.max_quality_score,
                        "metric": self.quality_metric
                    },
                    suggestions=[f"Improve quality to reach {self.min_quality_score}"]
                )
                
            if score > self.max_quality_score:
                return self._create_result(
                    False,
                    f"Quality score {score:.2f} exceeds maximum {self.max_quality_score}",
                    ValidationSeverity.WARNING,
                    correlation_id,
                    details={"score": score, "max_score": self.max_quality_score}
                )
                
            return self._create_result(
                True,
                f"Quality score {score:.2f} is within valid range",
                ValidationSeverity.INFO,
                correlation_id,
                details={
                    "score": score,
                    "metric": self.quality_metric
                }
            )
            
        except Exception as e:
            logger.error({
                "msg": "Quality validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Quality validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def _extract_quality_score(self, output: Any) -> Optional[float]:
        """Extract quality score from output."""
        if isinstance(output, dict):
            # Try common quality score keys
            keys = [self.quality_metric, "quality", "score", "quality_score", "rating"]
            for key in keys:
                if key in output:
                    return output[key]
                # Try nested paths like "quality.overall"
                if "." in self.quality_metric:
                    parts = self.quality_metric.split(".")
                    val = output
                    for part in parts:
                        if isinstance(val, dict) and part in val:
                            val = val[part]
                        else:
                            break
                    else:
                        return val
        elif isinstance(output, (int, float)):
            return float(output)
            
        return None


class CompositeValidator(Validator):
    """Combine multiple validators with AND/OR logic."""
    
    def __init__(
        self,
        validators: List[Validator],
        mode: str = "all",  # "all" = AND, "any" = OR
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.validators = validators
        self.mode = mode
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Run all validators and combine results."""
        results = []
        for validator in self.validators:
            result = validator.validate(output, correlation_id)
            results.append(result)
            
        if self.mode == "all":
            # AND mode - all must pass
            failures = [r for r in results if not r.is_valid]
            if failures:
                return self._create_result(
                    False,
                    f"Composite validation failed: {len(failures)}/{len(results)} validators failed",
                    ValidationSeverity.ERROR,
                    correlation_id,
                    details={
                        "failed_validators": [r.validator_name for r in failures],
                        "total_validators": len(results)
                    },
                    suggestions=list(set(sum([r.suggestions for r in failures], [])))
                )
            return self._create_result(
                True,
                f"All {len(results)} validators passed",
                ValidationSeverity.INFO,
                correlation_id
            )
        else:
            # OR mode - any can pass
            passes = [r for r in results if r.is_valid]
            if passes:
                return self._create_result(
                    True,
                    f"Composite validation passed: {len(passes)}/{len(results)} validators passed",
                    ValidationSeverity.INFO,
                    correlation_id
                )
            return self._create_result(
                False,
                f"All {len(results)} validators failed",
                ValidationSeverity.ERROR,
                correlation_id,
                suggestions=list(set(sum([r.suggestions for r in results], [])))
            )


class CustomValidator(Validator):
    """Custom validator with user-defined validation function."""
    
    def __init__(
        self,
        validate_fn: callable,
        fix_fn: Optional[callable] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.validate_fn = validate_fn
        self.fix_fn = fix_fn
        
    def validate(self, output: Any, correlation_id: Optional[str] = None) -> ValidationResult:
        """Run custom validation function."""
        try:
            return self.validate_fn(output, correlation_id)
        except Exception as e:
            logger.error({
                "msg": "Custom validation error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            if self.fail_safe:
                return self._create_result(
                    False,
                    f"Custom validation error: {str(e)}",
                    ValidationSeverity.CRITICAL,
                    correlation_id
                )
            raise
            
    def fix(self, output: Any, validation_result: ValidationResult) -> Optional[Any]:
        """Run custom fix function."""
        if self.fix_fn:
            try:
                return self.fix_fn(output, validation_result)
            except Exception:
                pass
        return None
