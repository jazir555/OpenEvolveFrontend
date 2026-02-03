"""
Hierarchical Configuration System

Manages configuration priority from multiple sources:

Priority (highest to lowest):
1. Runtime overrides (direct function arguments)
2. Environment variables
3. Config file (local)
4. Profile
5. User config (~/.evolve/config.yaml)
6. Global config (/etc/evolve/config.yaml)
7. Defaults
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigHierarchy:
    """
    Resolve configuration from multiple hierarchical sources.

    Implements the 7-level priority system for configuration resolution.
    """

    # Config file locations in priority order (low to high)
    GLOBAL_CONFIG_PATH = '/etc/evolve/config.yaml'
    USER_CONFIG_PATH = '~/.evolve/config.yaml'
    LOCAL_CONFIG_PATH = './config.yaml'
    LOCAL_CONFIG_ALT = './config.yml'

    def __init__(self):
        """Initialize ConfigHierarchy"""
        from .config_loader import ConfigLoader
        from .env_parser import EnvConfigParser
        from .profiles import ProfileManager

        self.loader = ConfigLoader()
        self.env_parser = EnvConfigParser()
        self.profile_manager = ProfileManager()

    def resolve_config(
        self,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        env_vars: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        profile: Optional[str] = None,
        use_global_config: bool = True,
        use_user_config: bool = True,
        use_local_config: bool = True
    ) -> Dict[str, Any]:
        """
        Resolve configuration from all sources with proper priority.

        Args:
            runtime_overrides: Highest priority - direct function arguments
            env_vars: Environment variables
            config_file: Specific config file to load
            profile: Profile name to load
            use_global_config: Whether to load global config (/etc/evolve/config.yaml)
            use_user_config: Whether to load user config (~/.evolve/config.yaml)
            use_local_config: Whether to load local config (./config.yaml)

        Returns:
            Merged configuration dictionary
        """
        # Start with empty config
        config = {}

        # Level 7: Defaults (implicitly applied by config classes)
        # These will be applied by the config class itself
        logger.debug("Level 7: Defaults will be applied by config class")

        # Level 6: Global config
        if use_global_config and self._exists(self.GLOBAL_CONFIG_PATH):
            global_config = self._load_config_safe(self.GLOBAL_CONFIG_PATH)
            config.update(global_config)
            logger.debug(f"Level 6: Loaded global config from {self.GLOBAL_CONFIG_PATH}")

        # Level 5: User config
        if use_user_config and self._exists(self.USER_CONFIG_PATH):
            user_config = self._load_config_safe(self.USER_CONFIG_PATH)
            config.update(user_config)
            logger.debug(f"Level 5: Loaded user config from {self.USER_CONFIG_PATH}")

        # Level 4: Profile
        if profile:
            profile_config = self.profile_manager.load_profile(profile)
            config.update(profile_config)
            logger.debug(f"Level 4: Loaded profile '{profile}'")

        # Level 3: Config file
        if config_file:
            file_config = self.loader.load_auto(config_file)
            config.update(file_config)
            logger.debug(f"Level 3: Loaded config file '{config_file}'")
        elif use_local_config:
            # Try local config files
            local_file = self._find_local_config()
            if local_file:
                file_config = self.loader.load_auto(local_file)
                config.update(file_config)
                logger.debug(f"Level 3: Loaded local config '{local_file}'")

        # Level 2: Environment variables
        if env_vars is None:
            env_vars = self.env_parser.parse_env()
        if env_vars:
            config.update(env_vars)
            logger.debug(f"Level 2: Applied {len(env_vars)} environment variables")

        # Level 1: Runtime overrides (highest priority)
        if runtime_overrides:
            config.update(runtime_overrides)
            logger.debug(f"Level 1: Applied {len(runtime_overrides)} runtime overrides")

        logger.info(f"Resolved configuration with {len(config)} parameters from {self._count_sources()} sources")
        return config

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.

        Later configs override earlier ones.

        Args:
            *configs: Variable number of config dictionaries

        Returns:
            Merged configuration
        """
        merged = {}

        for config in configs:
            merged.update(config)

        return merged

    def apply_overrides(
        self,
        base: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply overrides to base configuration.

        Args:
            base: Base configuration
            overrides: Override parameters

        Returns:
            New configuration with overrides applied
        """
        # Create copy to avoid mutating base
        result = base.copy()
        result.update(overrides)
        return result

    def get_config_sources(self) -> List[Dict[str, Any]]:
        """
        Get all available configuration sources.

        Returns:
            List of dicts with source info (name, path, exists)
        """
        sources = [
            {'name': 'global', 'path': self.GLOBAL_CONFIG_PATH, 'exists': self._exists(self.GLOBAL_CONFIG_PATH)},
            {'name': 'user', 'path': self.USER_CONFIG_PATH, 'exists': self._exists(self.USER_CONFIG_PATH)},
            {'name': 'local', 'path': self.LOCAL_CONFIG_PATH, 'exists': self._exists(self.LOCAL_CONFIG_PATH)},
            {'name': 'local_alt', 'path': self.LOCAL_CONFIG_ALT, 'exists': self._exists(self.LOCAL_CONFIG_ALT)},
        ]

        return sources

    def find_config_file(self) -> Optional[str]:
        """
        Find the highest priority config file that exists.

        Returns:
            Path to config file or None
        """
        # Check in priority order (high to low for this function)
        paths = [
            self.LOCAL_CONFIG_PATH,
            self.LOCAL_CONFIG_ALT,
            self.USER_CONFIG_PATH,
            self.GLOBAL_CONFIG_PATH,
        ]

        for path in paths:
            if self._exists(path):
                return path

        return None

    def _exists(self, path: str) -> bool:
        """Check if config file exists"""
        expanded = os.path.expanduser(path)
        return os.path.isfile(expanded)

    def _load_config_safe(self, path: str) -> Dict[str, Any]:
        """
        Load config file safely, returning empty dict on failure.

        Args:
            path: Path to config file

        Returns:
            Config dict or empty dict
        """
        try:
            expanded = os.path.expanduser(path)
            return self.loader.load_auto(expanded)
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            return {}

    def _find_local_config(self) -> Optional[str]:
        """Find local config file"""
        if self._exists(self.LOCAL_CONFIG_PATH):
            return self.LOCAL_CONFIG_PATH
        elif self._exists(self.LOCAL_CONFIG_ALT):
            return self.LOCAL_CONFIG_ALT
        return None

    def _count_sources(self) -> int:
        """Count how many sources contributed to current resolution"""
        # This would need to be tracked during resolve_config
        # For now, return a placeholder
        return 0


class ConfigMerge:
    """
    Advanced configuration merging utilities.

    Supports:
    - Shallow merge (default dict.update)
    - Deep merge (recursive dict merge)
    - List merge (append or replace)
    - Conditional merge (merge only if conditions met)
    """

    @staticmethod
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Nested dicts are merged recursively.
        Lists are replaced (not appended).

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Deep merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigMerge.deep_merge(result[key], value)
            else:
                # Replace value
                result[key] = value

        return result

    @staticmethod
    def conditional_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
        condition: callable
    ) -> Dict[str, Any]:
        """
        Conditionally merge based on predicate function.

        Args:
            base: Base dictionary
            override: Override dictionary
            condition: Function(key, base_value, override_value) -> bool

        Returns:
            Conditionally merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            base_value = base.get(key)
            if condition(key, base_value, value):
                result[key] = value

        return result

    @staticmethod
    def merge_if_missing(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge only keys that are missing from base.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary (only new keys added)
        """
        result = base.copy()

        for key, value in override.items():
            if key not in base:
                result[key] = value

        return result

    @staticmethod
    def merge_if_present(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge only keys that exist in base.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary (only existing keys updated)
        """
        result = base.copy()

        for key, value in override.items():
            if key in base:
                result[key] = value

        return result


class ConfigSnapshot:
    """
    Snapshot and restore configuration state.

    Useful for:
    - Testing with temporary config changes
    - Rolling back to previous config
    - Comparing config states
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Create config snapshot.

        Args:
            config: Configuration to snapshot
        """
        self.config = config.copy()
        self.timestamp = self._get_timestamp()

    def restore(self) -> Dict[str, Any]:
        """Restore snapshot"""
        return self.config.copy()

    def diff(self, other: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get differences between snapshot and other config.

        Args:
            other: Other config to compare

        Returns:
            Dict of differences
        """
        diff = {}

        # Keys in snapshot but not in other
        for key in self.config:
            if key not in other:
                diff[key] = {'old': self.config[key], 'new': None}
            elif self.config[key] != other[key]:
                diff[key] = {'old': self.config[key], 'new': other[key]}

        # Keys in other but not in snapshot
        for key in other:
            if key not in self.config:
                diff[key] = {'old': None, 'new': other[key]}

        return diff

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class ConfigSource:
    """
    Track where each configuration value came from.

    Useful for debugging and understanding config resolution.
    """

    def __init__(self):
        """Initialize ConfigSource tracker"""
        self.sources: Dict[str, str] = {}

    def add_source(self, key: str, source: str) -> None:
        """Track source for a config key"""
        self.sources[key] = source

    def get_source(self, key: str) -> Optional[str]:
        """Get source for a config key"""
        return self.sources.get(key)

    def get_all_sources(self) -> Dict[str, str]:
        """Get all tracked sources"""
        return self.sources.copy()

    def group_by_source(self) -> Dict[str, List[str]]:
        """Group keys by their source"""
        groups: Dict[str, List[str]] = {}

        for key, source in self.sources.items():
            if source not in groups:
                groups[source] = []
            groups[source].append(key)

        return groups
