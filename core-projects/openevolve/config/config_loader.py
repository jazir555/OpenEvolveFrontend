"""
Configuration File Loader

Supports loading and saving configuration from multiple file formats:
- YAML (.yaml, .yml)
- JSON (.json)
- TOML (.toml)

Auto-detects format based on file extension.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoaderError(Exception):
    """Base error for ConfigLoader operations"""
    pass


class ConfigFormatError(ConfigLoaderError):
    """Raised when file format is not supported"""
    pass


class ConfigParseError(ConfigLoaderError):
    """Raised when file parsing fails"""
    pass


class ConfigLoader:
    """
    Load configuration from YAML, JSON, TOML files.

    Features:
    - Auto-detect file format from extension
    - Load and save all supported formats
    - Handle file not found gracefully
    - Provide detailed error messages
    """

    SUPPORTED_FORMATS = {
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.toml': 'toml'
    }

    def __init__(self):
        """Initialize ConfigLoader"""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """
        Check which format libraries are available.

        Sets self._has_yaml, self._has_toml flags
        """
        self._has_yaml = False
        self._has_toml = False

        try:
            import yaml
            self._has_yaml = True
            self._yaml = yaml
        except ImportError:
            logger.warning("PyYAML not installed - YAML files not supported")

        try:
            import tomli
            self._toml_reader = tomli
            self._has_toml = True
        except ImportError:
            try:
                import tomllib
                self._toml_reader = tomllib
                self._has_toml = True
            except ImportError:
                logger.warning("tomli/tomllib not available - TOML files not supported (Python 3.11+ has built-in tomllib)")

    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Dictionary of configuration parameters

        Raises:
            ConfigFormatError: If YAML library not available
            ConfigParseError: If YAML parsing fails
            FileNotFoundError: If file doesn't exist
        """
        if not self._has_yaml:
            raise ConfigFormatError("YAML format not supported - install PyYAML")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                config = self._yaml.safe_load(f) or {}

            logger.debug(f"Loaded YAML config from {filepath} with {len(config)} parameters")
            return config

        except self._yaml.YAMLError as e:
            raise ConfigParseError(f"Failed to parse YAML file {filepath}: {e}")

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary of configuration parameters

        Raises:
            ConfigParseError: If JSON parsing fails
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            logger.debug(f"Loaded JSON config from {filepath} with {len(config)} parameters")
            return config

        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Failed to parse JSON file {filepath}: {e}")

    def load_toml(self, filepath: str) -> Dict[str, Any]:
        """
        Load configuration from TOML file.

        Args:
            filepath: Path to TOML file

        Returns:
            Dictionary of configuration parameters

        Raises:
            ConfigFormatError: If TOML library not available
            ConfigParseError: If TOML parsing fails
            FileNotFoundError: If file doesn't exist
        """
        if not self._has_toml:
            raise ConfigFormatError("TOML format not supported - install tomli (or use Python 3.11+)")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                config = self._toml_reader.load(f)

            logger.debug(f"Loaded TOML config from {filepath} with {len(config)} parameters")
            return config

        except Exception as e:
            raise ConfigParseError(f"Failed to parse TOML file {filepath}: {e}")

    def load_auto(self, filepath: str) -> Dict[str, Any]:
        """
        Auto-detect format and load configuration file.

        Args:
            filepath: Path to configuration file

        Returns:
            Dictionary of configuration parameters

        Raises:
            ConfigFormatError: If format not supported
            ConfigParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        ext = Path(filepath).suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ConfigFormatError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        format_type = self.SUPPORTED_FORMATS[ext]

        if format_type == 'yaml':
            return self.load_yaml(filepath)
        elif format_type == 'json':
            return self.load_json(filepath)
        elif format_type == 'toml':
            return self.load_toml(filepath)
        else:
            raise ConfigFormatError(f"Unknown format: {format_type}")

    def save_yaml(self, config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            filepath: Path to save file

        Raises:
            ConfigFormatError: If YAML library not available
            IOError: If file write fails
        """
        if not self._has_yaml:
            raise ConfigFormatError("YAML format not supported - install PyYAML")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        try:
            with open(filepath, 'w') as f:
                self._yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved YAML config to {filepath} with {len(config)} parameters")

        except Exception as e:
            raise IOError(f"Failed to save YAML file {filepath}: {e}")

    def save_json(self, config: Dict[str, Any], filepath: str, pretty: bool = True) -> None:
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            filepath: Path to save file
            pretty: Whether to format JSON prettily (default: True)

        Raises:
            IOError: If file write fails
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        try:
            with open(filepath, 'w') as f:
                if pretty:
                    json.dump(config, f, indent=2)
                else:
                    json.dump(config, f)

            logger.info(f"Saved JSON config to {filepath} with {len(config)} parameters")

        except Exception as e:
            raise IOError(f"Failed to save JSON file {filepath}: {e}")

    def save_toml(self, config: Dict[str, Any], filepath: str) -> None:
        """
        Save configuration to TOML file.

        Args:
            config: Configuration dictionary
            filepath: Path to save file

        Raises:
            ConfigFormatError: If TOML library not available
            IOError: If file write fails
        """
        if not self._has_toml:
            # Try tomli_w for writing
            try:
                import tomli_w
            except ImportError:
                raise ConfigFormatError("TOML write not supported - install tomli_w")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        try:
            with open(filepath, 'wb') as f:
                tomli_w.dump(config, f)

            logger.info(f"Saved TOML config to {filepath} with {len(config)} parameters")

        except Exception as e:
            raise IOError(f"Failed to save TOML file {filepath}: {e}")

    def exists(self, filepath: str) -> bool:
        """
        Check if configuration file exists.

        Args:
            filepath: Path to check

        Returns:
            True if file exists, False otherwise
        """
        return os.path.isfile(filepath)

    def get_format(self, filepath: str) -> Optional[str]:
        """
        Get file format from extension.

        Args:
            filepath: Path to file

        Returns:
            Format string ('yaml', 'json', 'toml') or None if unknown
        """
        ext = Path(filepath).suffix.lower()
        return self.SUPPORTED_FORMATS.get(ext)

    def is_supported(self, filepath: str) -> bool:
        """
        Check if file format is supported.

        Args:
            filepath: Path to file

        Returns:
            True if format is supported, False otherwise
        """
        return Path(filepath).suffix.lower() in self.SUPPORTED_FORMATS
