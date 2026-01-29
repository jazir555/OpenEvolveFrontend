"""
Configuration Loader for OpenEvolve Integrations

This module provides configuration loading capabilities for all integrations.
It supports YAML and JSON formats with environment variable interpolation.

Author: Agent 8 (Integration Orchestrator)
Created: 2026-01-02
Status: ✅ Complete
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigLoadError(Exception):
    """Exception raised when configuration loading fails."""
    message: str
    path: str
    details: Optional[str] = None


class ConfigLoader:
    """
    Configuration loader for integration configurations.

    Features:
    - YAML and JSON support
    - Environment variable interpolation
    - Configuration validation
    - Default value merging
    - Configuration caching
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the configuration loader.

        Args:
            cache_enabled: Whether to cache loaded configurations
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(
        self,
        path: Union[str, Path],
        interpolate_env: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            path: Path to configuration file (YAML or JSON)
            interpolate_env: Whether to interpolate environment variables
            validate: Whether to validate the configuration

        Returns:
            Configuration dictionary

        Raises:
            ConfigLoadError: If loading fails
        """
        path = str(path)

        # Check cache
        if self.cache_enabled and path in self._cache:
            logger.debug(f"Loading config from cache: {path}")
            return self._cache[path]

        try:
            # Determine file type and load
            if path.endswith('.yaml') or path.endswith('.yml'):
                config = self._load_yaml(path)
            elif path.endswith('.json'):
                config = self._load_json(path)
            else:
                raise ConfigLoadError(
                    message="Unsupported file format",
                    path=path,
                    details="Supported formats: .yaml, .yml, .json"
                )

            # Interpolate environment variables
            if interpolate_env:
                config = self._interpolate_env_vars(config)

            # Validate configuration
            if validate:
                self._validate_config(config, path)

            # Cache if enabled
            if self.cache_enabled:
                self._cache[path] = config

            logger.info(f"Loaded configuration from: {path}")
            return config

        except ConfigLoadError:
            raise
        except Exception as e:
            raise ConfigLoadError(
                message=f"Failed to load configuration",
                path=path,
                details=str(e)
            )

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ConfigLoadError(
                message="PyYAML not installed",
                path=path,
                details="Install with: pip install pyyaml"
            )

        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigLoadError(
                message="Invalid YAML syntax",
                path=path,
                details=str(e)
            )
        except FileNotFoundError:
            raise ConfigLoadError(
                message="Configuration file not found",
                path=path
            )

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadError(
                message="Invalid JSON syntax",
                path=path,
                details=str(e)
            )
        except FileNotFoundError:
            raise ConfigLoadError(
                message="Configuration file not found",
                path=path
            )

    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Recursively interpolate environment variables in configuration.

        Supports:
        - ${VAR_NAME} - simple substitution
        - ${VAR_NAME:default} - with default value

        Args:
            config: Configuration value (can be nested)

        Returns:
            Configuration with interpolated values
        """
        if isinstance(config, dict):
            return {
                key: self._interpolate_env_vars(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [
                self._interpolate_env_vars(item)
                for item in config
            ]
        elif isinstance(config, str):
            return self._substitute_env_var(config)
        else:
            return config

    def _substitute_env_var(self, value: str) -> Any:
        """
        Substitute environment variables in a string.

        Args:
            value: String potentially containing environment variable references

        Returns:
            Substituted value (converted to appropriate type)
        """
        # Pattern for ${VAR} or ${VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""

            # Get environment variable or default
            env_value = os.environ.get(var_name, default)

            if not env_value:
                logger.warning(f"Environment variable {var_name} not set and no default provided")

            return env_value

        # Substitute all environment variables
        result = re.sub(pattern, replacer, value)

        # Try to convert to appropriate type
        return self._convert_type(result)

    def _convert_type(self, value: str) -> Any:
        """
        Convert string to appropriate type.

        Args:
            value: String value

        Returns:
            Converted value (int, float, bool, or string)
        """
        # Boolean
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False

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

        # String
        return value

    def _validate_config(self, config: Dict[str, Any], path: str) -> None:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary
            path: Path to configuration file (for error messages)

        Raises:
            ConfigLoadError: If validation fails
        """
        # Check for required top-level keys
        if not isinstance(config, dict):
            raise ConfigLoadError(
                message="Configuration must be a dictionary",
                path=path
            )

        # Validate project section if present
        if 'project' in config:
            project = config['project']
            required_keys = ['name', 'version']
            for key in required_keys:
                if key not in project:
                    logger.warning(f"Missing recommended key in project section: {key}")

        # Validate connection section if present
        if 'connection' in config:
            connection = config['connection']
            # URL or host should be present
            if not any(key in connection for key in ['url', 'host', 'uri']):
                logger.warning("Connection section missing url/host/uri")

        # Validate integration section if present
        if 'integration' in config:
            integration = config['integration']
            # Check for boolean flags
            bool_flags = ['auto_start', 'cache_enabled', 'fallback_on_error']
            for flag in bool_flags:
                if flag in integration and not isinstance(integration[flag], bool):
                    raise ConfigLoadError(
                        message=f"Integration flag '{flag}' must be boolean",
                        path=path
                    )

    def merge_with_defaults(
        self,
        config: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge configuration with defaults.

        Args:
            config: User configuration
            defaults: Default configuration values

        Returns:
            Merged configuration
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Deep merge two dictionaries."""
            result = base.copy()

            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        return deep_merge(defaults, config)

    def save(
        self,
        config: Dict[str, Any],
        path: Union[str, Path],
        format: Optional[str] = None
    ) -> None:
        """
        Save configuration to a file.

        Args:
            config: Configuration dictionary
            path: Path to save file
            format: Format to save ('yaml' or 'json'). If None, inferred from path
        """
        path = str(path)

        # Determine format
        if format is None:
            if path.endswith('.yaml') or path.endswith('.yml'):
                format = 'yaml'
            elif path.endswith('.json'):
                format = 'json'
            else:
                format = 'yaml'  # Default

        # Create directory if needed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # Save file
        if format == 'yaml':
            self._save_yaml(config, path)
        elif format == 'json':
            self._save_json(config, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved configuration to: {path}")

    def _save_yaml(self, config: Dict[str, Any], path: str) -> None:
        """Save YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ConfigLoadError(
                message="PyYAML not installed",
                path=path,
                details="Install with: pip install pyyaml"
            )

        with open(path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    def _save_json(self, config: Dict[str, Any], path: str) -> None:
        """Save JSON configuration file."""
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def clear_cache(self, path: Optional[str] = None) -> None:
        """
        Clear configuration cache.

        Args:
            path: Optional specific path to clear. If None, clears all cache.
        """
        if path:
            if path in self._cache:
                del self._cache[path]
                logger.debug(f"Cleared cache for: {path}")
        else:
            self._cache.clear()
            logger.debug("Cleared all configuration cache")

    def get_default_config(self, integration_type: str) -> Dict[str, Any]:
        """
        Get default configuration for an integration type.

        Args:
            integration_type: Type of integration (e.g., 'graphiti', 'oneke')

        Returns:
            Default configuration dictionary
        """
        defaults = {
            'project': {
                'name': integration_type,
                'version': '1.0.0',
                'enabled': True
            },
            'connection': {
                'timeout': 30,
                'retries': 3
            },
            'integration': {
                'auto_start': True,
                'cache_enabled': True,
                'cache_ttl': 3600,
                'fallback_on_error': True
            },
            'performance': {
                'max_workers': 4,
                'batch_size': 100
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

        return defaults

    def create_example_config(self, integration_type: str, path: Union[str, Path]) -> None:
        """
        Create an example configuration file for an integration.

        Args:
            integration_type: Type of integration
            path: Path to save example config
        """
        example_config = {
            'project': {
                'name': integration_type,
                'version': '1.0.0',
                'description': f'Example configuration for {integration_type}',
                'enabled': True
            },
            'connection': {
                'url': 'localhost',
                'port': 7687,
                'api_key': '${API_KEY}',  # Environment variable example
                'timeout': 30,
                'retries': 3
            },
            'features': {
                'feature_1': True,
                'feature_2': False
            },
            'integration': {
                'auto_start': True,
                'cache_enabled': True,
                'cache_ttl': 3600,
                'fallback_on_error': True
            },
            'performance': {
                'max_workers': 4,
                'timeout': 30,
                'batch_size': 100
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

        self.save(example_config, path)
        logger.info(f"Created example configuration at: {path}")


# Convenience functions

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load(path)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Convenience function to save a configuration file.

    Args:
        config: Configuration dictionary
        path: Path to save file
    """
    loader = ConfigLoader()
    loader.save(config, path)
