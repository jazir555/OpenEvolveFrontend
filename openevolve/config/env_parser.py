"""
Environment Variable Configuration Parser

Parses environment variables into configuration parameters.

All environment variables use the EVOLVE_ prefix:
- EVOLVE_MAX_ITERATIONS=100 → config.max_iterations=100
- EVOLVE_ENABLE_PLANNING=true → config.enable_planning=True
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class EnvConfigError(Exception):
    """Base error for environment variable parsing"""
    pass


class ValidationError:
    """Represents a single validation error"""

    def __init__(self, parameter: str, value: str, message: str):
        self.parameter = parameter
        self.value = value
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.parameter}={self.value}: {self.message})"


class EnvConfigParser:
    """
    Parse environment variables into configuration.

    Features:
    - Maps env vars to config parameters
    - Type conversion (int, float, bool, str, list)
    - Validation of env var names and values
    - Detailed error reporting
    """

    # Prefix for all environment variables
    ENV_PREFIX = "EVOLVE_"

    # Type mapping for type conversion
    TYPE_CONVERTERS = {
        int: lambda x: int(float(x)),  # Handle "1.0" as int
        float: float,
        str: str,
        bool: lambda x: x.lower() in ('true', '1', 'yes', 'on', 't', 'y'),
        list: lambda x: [item.strip() for item in x.split(',')]
    }

    def __init__(self, env_mappings: Optional[Dict[str, Tuple[str, Type]]] = None):
        """
        Initialize EnvConfigParser.

        Args:
            env_mappings: Optional dict mapping parameter names to (env_var_name, type) tuples
                          If None, uses default mappings from env_mappings module
        """
        # Lazy import to avoid circular dependency
        if env_mappings is None:
            try:
                from .env_mappings import ENV_MAPPINGS
                self.env_mappings = ENV_MAPPINGS
            except ImportError:
                logger.warning("env_mappings module not found - no default mappings")
                self.env_mappings = {}
        else:
            self.env_mappings = env_mappings

        self._parsed_cache: Optional[Dict[str, Any]] = None

    def parse_env(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse all environment variables with EVOLVE_ prefix.

        Args:
            prefix: Optional custom prefix (default: uses ENV_PREFIX)

        Returns:
            Dictionary of configuration parameters from environment
        """
        if prefix is None:
            prefix = self.ENV_PREFIX

        config = {}
        prefix_len = len(prefix)

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Convert EVOLVE_MAX_ITERATIONS → max_iterations
                param_name = self._env_to_param_name(env_var[prefix_len:])

                # Try to convert value to appropriate type
                converted_value = self._convert_value(param_name, value)

                config[param_name] = converted_value

        logger.debug(f"Parsed {len(config)} parameters from environment variables")
        return config

    def get_env_value(self, param_name: str, default: Any = None) -> Optional[Any]:
        """
        Get environment variable value for a specific parameter.

        Args:
            param_name: Configuration parameter name (e.g., 'max_iterations')
            default: Default value if env var not set

        Returns:
            Converted value or default
        """
        env_var = self.param_to_env_name(param_name)
        value = os.environ.get(env_var)

        if value is None:
            return default

        return self._convert_value(param_name, value)

    def env_to_config(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse environment variables and return as config dict.

        Args:
            prefix: Optional custom prefix (default: uses ENV_PREFIX)

        Returns:
            Dictionary suitable for creating UnifiedConfiguration
        """
        return self.parse_env(prefix)

    def validate_env_vars(self) -> List[ValidationError]:
        """
        Validate all EVOLVE_ environment variables.

        Checks:
        - Type conversion works
        - Values are in valid ranges
        - Required vars are set (if we have required list)

        Returns:
            List of ValidationErrors (empty if all valid)
        """
        errors = []
        prefix = self.ENV_PREFIX

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                param_name = self._env_to_param_name(env_var[prefix_len:])

                # Try to convert
                try:
                    converted = self._convert_value(param_name, value)
                except Exception as e:
                    errors.append(ValidationError(
                        parameter=param_name,
                        value=value,
                        message=f"Type conversion failed: {e}"
                    ))
                    continue

                # Validate range if we have type info
                if param_name in self.env_mappings:
                    env_name, param_type = self.env_mappings[param_name]

                    # Check type
                    if not isinstance(converted, param_type):
                        errors.append(ValidationError(
                            parameter=param_name,
                            value=value,
                            message=f"Expected {param_type.__name__}, got {type(converted).__name__}"
                        ))

        return errors

    def _env_to_param_name(self, env_suffix: str) -> str:
        """
        Convert environment variable suffix to parameter name.

        Example: MAX_ITERATIONS → max_iterations

        Args:
            env_suffix: Environment variable name after prefix

        Returns:
            Parameter name in snake_case
        """
        # Convert to lowercase and handle underscores
        return env_suffix.lower()

    def _convert_value(self, param_name: str, value: str) -> Any:
        """
        Convert string value to appropriate type.

        Args:
            param_name: Parameter name (for type lookup)
            value: String value from environment

        Returns:
            Converted value

        Raises:
            ValueError: If conversion fails
        """
        # Look up type from mappings
        if param_name in self.env_mappings:
            env_name, param_type = self.env_mappings[param_name]

            try:
                return self.TYPE_CONVERTERS[param_type](value)
            except (ValueError, KeyError) as e:
                raise ValueError(f"Cannot convert {value} to {param_type.__name__}: {e}")

        # Try to auto-detect type
        # Boolean
        if value.lower() in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
            return self.TYPE_CONVERTERS[bool](value)

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # Default to string
        return value

    @staticmethod
    def param_to_env_name(param_name: str) -> str:
        """
        Convert parameter name to environment variable name.

        Example: max_iterations → EVOLVE_MAX_ITERATIONS

        Args:
            param_name: Parameter name in snake_case

        Returns:
            Environment variable name with EVOLVE_ prefix
        """
        return f"EVOLVE_{param_name.upper()}"

    def list_env_vars(self) -> List[str]:
        """
        List all EVOLVE_ environment variables.

        Returns:
            List of environment variable names
        """
        return [env for env in os.environ if env.startswith(self.ENV_PREFIX)]

    def clear_cache(self) -> None:
        """Clear parsed cache"""
        self._parsed_cache = None

    def get_mapping_info(self) -> Dict[str, Tuple[str, Type]]:
        """
        Get all parameter mappings.

        Returns:
            Dictionary mapping parameter names to (env_var_name, type) tuples
        """
        return self.env_mappings.copy()


class TypedEnvConfigParser(EnvConfigParser):
    """
    Environment variable parser with strict type checking.

    Raises errors if type conversion fails or values are out of range.
    """

    def __init__(self, env_mappings: Dict[str, Tuple[str, Type]], ranges: Optional[Dict[str, Tuple[Any, Any]]] = None):
        """
        Initialize typed parser.

        Args:
            env_mappings: Dict mapping parameter names to (env_var_name, type)
            ranges: Optional dict mapping parameter names to (min, max) ranges
        """
        super().__init__(env_mappings)
        self.ranges = ranges or {}

    def _convert_value(self, param_name: str, value: str) -> Any:
        """
        Convert value with strict type checking and range validation.

        Args:
            param_name: Parameter name
            value: String value

        Returns:
            Converted value

        Raises:
            ValueError: If conversion fails or value out of range
        """
        converted = super()._convert_value(param_name, value)

        # Check range if specified
        if param_name in self.ranges:
            min_val, max_val = self.ranges[param_name]

            if isinstance(converted, (int, float)):
                if not (min_val <= converted <= max_val):
                    raise ValueError(
                        f"Value {converted} out of range [{min_val}, {max_val}]"
                    )

        return converted


# Convenience function for quick parsing
def parse_env_vars(prefix: str = "EVOLVE_") -> Dict[str, Any]:
    """
    Quick function to parse environment variables.

    Args:
        prefix: Environment variable prefix

    Returns:
        Dictionary of parsed parameters
    """
    parser = EnvConfigParser()
    parser.ENV_PREFIX = prefix
    return parser.parse_env()


def get_env_var(param_name: str, default: Any = None) -> Any:
    """
    Quick function to get single environment variable value.

    Args:
        param_name: Parameter name (e.g., 'max_iterations')
        default: Default value if not set

    Returns:
        Converted value or default
    """
    parser = EnvConfigParser()
    return parser.get_env_value(param_name, default)
