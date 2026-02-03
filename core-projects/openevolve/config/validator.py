"""
Configuration Validator

Validates configuration parameters with detailed error messages and suggestions.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """
    Represents a single validation error.

    Attributes:
        parameter: Name of the parameter that failed validation
        value: The invalid value
        message: Error message
        severity: 'error' or 'warning'
        suggestion: Optional suggestion for fixing the error
    """
    parameter: str
    value: Any
    message: str
    severity: str = 'error'
    suggestion: Optional[str] = None

    def __repr__(self) -> str:
        return f"ValidationError({self.parameter}={self.value}: {self.message})"


@dataclass
class ValidationResult:
    """
    Result of configuration validation.

    Attributes:
        is_valid: True if no errors found (warnings don't affect this)
        errors: List of validation errors
        warnings: List of validation warnings
    """
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError) -> None:
        """Add an error to the result"""
        self.errors.append(error)
        if error.severity == 'error':
            self.is_valid = False

    def add_warning(self, warning: ValidationError) -> None:
        """Add a warning to the result"""
        self.warnings.append(warning)

    def get_error_messages(self) -> List[str]:
        """Get error messages as strings"""
        return [f"{e.parameter}: {e.message}" for e in self.errors]

    def get_warning_messages(self) -> List[str]:
        """Get warning messages as strings"""
        return [f"{w.parameter}: {w.message}" for w in self.warnings]

    def __repr__(self) -> str:
        return f"ValidationResult(valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class ConfigValidator:
    """
    Validate configuration with detailed error messages.

    Features:
    - Type checking
    - Range validation
    - Dependency checking
    - Logical consistency checks
    - Helpful error messages with suggestions
    """

    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize ConfigValidator.

        Args:
            schema: Optional schema dict with parameter definitions
                    If None, uses default schema
        """
        self.schema = schema or self._get_default_schema()

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # Validate each parameter
        for param_name, value in config.items():
            self._validate_parameter(param_name, value, result)

        # Check dependencies
        dependency_errors = self.check_dependencies(config)
        for error in dependency_errors:
            result.add_error(error)

        # Check logical consistency
        consistency_errors = self._check_consistency(config)
        for error in consistency_errors:
            result.add_error(error)

        logger.info(
            f"Validation complete: {len(result.errors)} errors, "
            f"{len(result.warnings)} warnings"
        )

        return result

    def validate_parameter(
        self,
        name: str,
        value: Any,
        schema: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a single parameter.

        Args:
            name: Parameter name
            value: Parameter value
            schema: Optional schema for this parameter

        Returns:
            Tuple of (is_valid, error_message)
        """
        param_schema = schema or self.schema.get(name, {})

        # Check if parameter is known
        if not param_schema:
            return True, None  # Unknown parameters are OK (flexibility)

        # Type check
        if 'type' in param_schema:
            expected_type = param_schema['type']
            if not isinstance(value, expected_type):
                return False, f"Expected type {expected_type.__name__}, got {type(value).__name__}"

        # Range check for numeric values
        if 'range' in param_schema and isinstance(value, (int, float)):
            min_val, max_val = param_schema['range']
            if not (min_val <= value <= max_val):
                return False, f"Value {value} out of range [{min_val}, {max_val}]"

        # Choice check
        if 'choices' in param_schema:
            if value not in param_schema['choices']:
                return False, f"Value must be one of {param_schema['choices']}, got '{value}'"

        return True, None

    def check_dependencies(self, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Check parameter dependencies.

        Example: If enable_memory=True, memory_type must be set

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Memory dependencies
        if config.get('enable_memory', False):
            if 'memory_type' not in config:
                errors.append(ValidationError(
                    parameter='memory_type',
                    value=None,
                    message='memory_type is required when enable_memory=True',
                    suggestion="Set memory_type to 'episodic', 'semantic', or 'working'"
                ))

        # QD dependencies
        if config.get('qd_enabled', False):
            if 'qd_algorithm' not in config:
                errors.append(ValidationError(
                    parameter='qd_algorithm',
                    value=None,
                    message='qd_algorithm is required when qd_enabled=True',
                    suggestion="Set qd_algorithm to 'map_elites', 'cvt_map_elites', etc."
                ))

        # Planning dependencies
        if config.get('enable_planning', False):
            if 'planner_model' not in config:
                errors.append(ValidationError(
                    parameter='planner_model',
                    value=None,
                    message='planner_model is required when enable_planning=True',
                    suggestion="Set planner_model to a valid model ID"
                ))

        # Parallel execution dependencies
        if config.get('parallel_workers', 1) > 1:
            if not config.get('enable_ray', False):
                errors.append(ValidationError(
                    parameter='enable_ray',
                    value=False,
                    message='enable_ray=True is recommended when parallel_workers > 1',
                    severity='warning',
                    suggestion="Set enable_ray=True for parallel execution"
                ))

        # Early stopping dependencies
        if config.get('early_stopping', False):
            if 'early_stopping_patience' not in config:
                errors.append(ValidationError(
                    parameter='early_stopping_patience',
                    value=None,
                    message='early_stopping_patience is required when early_stopping=True',
                    suggestion="Set early_stopping_patience to a positive integer"
                ))

        return errors

    def suggest_fixes(self, errors: List[ValidationError]) -> List[str]:
        """
        Generate suggestions for fixing validation errors.

        Args:
            errors: List of validation errors

        Returns:
            List of suggestion strings
        """
        suggestions = []

        for error in errors:
            if error.suggestion:
                suggestions.append(f"{error.parameter}: {error.suggestion}")
            else:
                # Generate automatic suggestions
                suggestion = self._generate_suggestion(error)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    def _validate_parameter(
        self,
        name: str,
        value: Any,
        result: ValidationResult
    ) -> None:
        """Validate a single parameter and add to result"""
        is_valid, error_message = self.validate_parameter(name, value)

        if not is_valid:
            result.add_error(ValidationError(
                parameter=name,
                value=value,
                message=error_message,
                suggestion=self._generate_suggestion_from_message(error_message)
            ))

    def _check_consistency(self, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Check logical consistency of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check that temperature is in valid range
        if 'temperature' in config:
            temp = config['temperature']
            if not (0.0 <= temp <= 2.0):
                errors.append(ValidationError(
                    parameter='temperature',
                    value=temp,
                    message=f'temperature must be between 0.0 and 2.0, got {temp}',
                    suggestion='Set temperature between 0.0 (deterministic) and 2.0 (creative)'
                ))

        # Check that population_size <= max_iterations for efficiency
        if 'population_size' in config and 'max_iterations' in config:
            pop = config['population_size']
            max_iter = config['max_iterations']
            if pop > max_iter:
                errors.append(ValidationError(
                    parameter='population_size',
                    value=pop,
                    message=f'population_size ({pop}) > max_iterations ({max_iter}) is inefficient',
                    severity='warning',
                    suggestion=f'Consider reducing population_size to <= {max_iter}'
                ))

        # Check mutation + crossover rate sum
        if 'mutation_rate' in config and 'crossover_rate' in config:
            mut = config['mutation_rate']
            cross = config['crossover_rate']
            if mut + cross > 1.0:
                errors.append(ValidationError(
                    parameter='crossover_rate',
                    value=cross,
                    message=f'mutation_rate + crossover_rate = {mut + cross} > 1.0',
                    severity='warning',
                    suggestion='Consider reducing one or both rates so sum <= 1.0'
                ))

        # Check log level
        if 'log_level' in config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            level = config['log_level'].upper()
            if level not in valid_levels:
                errors.append(ValidationError(
                    parameter='log_level',
                    value=level,
                    message=f'Invalid log level: {level}',
                    suggestion=f"Use one of: {', '.join(valid_levels)}"
                ))

        return errors

    def _generate_suggestion(self, error: ValidationError) -> Optional[str]:
        """Generate suggestion for an error"""
        if 'out of range' in error.message:
            # Extract range from error message
            import re
            match = re.search(r'\[([-\d.]+),\s*([-\d.]+)\]', error.message)
            if match:
                min_val, max_val = match.groups()
                return f"Set {error.parameter} between {min_val} and {max_val}"

        return None

    def _generate_suggestion_from_message(self, message: str) -> Optional[str]:
        """Generate suggestion from error message"""
        if 'out of range' in message:
            import re
            match = re.search(r'\[([-\d.]+),\s*([-\d.]+)\]', message)
            if match:
                return f"Value must be between {match.group(1)} and {match.group(2)}"

        return None

    def _get_default_schema(self) -> Dict:
        """
        Get default parameter schema.

        Returns:
            Dictionary defining parameter constraints
        """
        from .env_mappings import ENV_MAPPINGS, ENV_RANGES

        schema = {}

        for param_name, (env_name, param_type) in ENV_MAPPINGS.items():
            schema[param_name] = {
                'type': param_type
            }

            # Add range if available
            if param_name in ENV_RANGES:
                schema[param_name]['range'] = ENV_RANGES[param_name]

        return schema


class StrictConfigValidator(ConfigValidator):
    """
    Strict validator that rejects unknown parameters.

    Use this when you want to ensure only known parameters are used.
    """

    def validate_parameter(
        self,
        name: str,
        value: Any,
        schema: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter strictly (rejects unknown parameters).

        Args:
            name: Parameter name
            value: Parameter value
            schema: Optional schema

        Returns:
            Tuple of (is_valid, error_message)
        """
        param_schema = schema or self.schema.get(name)

        if not param_schema:
            return False, f"Unknown parameter: {name}"

        return super().validate_parameter(name, value, schema)


class LenientConfigValidator(ConfigValidator):
    """
    Lenient validator that only logs warnings, never errors.

    Useful for development and testing.
    """

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration leniently (all errors become warnings).

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult with warnings only (always valid)
        """
        result = ValidationResult()

        for param_name, value in config.items():
            is_valid, error_message = self.validate_parameter(param_name, value)

            if not is_valid:
                # Convert error to warning
                result.add_warning(ValidationError(
                    parameter=param_name,
                    value=value,
                    message=error_message,
                    severity='warning'
                ))

        # Always return valid
        result.is_valid = True

        return result
