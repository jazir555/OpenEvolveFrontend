"""
Configuration Manager - Unified Interface

Main configuration manager providing a single interface for all config operations.
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""

    def __init__(self, errors: List[str], warnings: List[str] = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__(f"Configuration validation failed with {len(errors)} errors")


class ConfigManager:
    """
    Main configuration manager - single interface for all config operations.

    Features:
    - Load config from multiple sources with proper priority
    - Save config to files
    - Validate configurations
    - Enable hot-reload
    - Manage profiles
    - Query parameters

    Usage:
        manager = ConfigManager()
        config = manager.load_config(
            config_file='config.yaml',
            profile='development',
            env_override=True
        )
    """

    def __init__(self):
        """Initialize ConfigManager"""
        from .config_loader import ConfigLoader
        from .env_parser import EnvConfigParser
        from .validator import ConfigValidator
        from .profiles import ProfileManager
        from .hierarchy import ConfigHierarchy

        self.loader = ConfigLoader()
        self.env_parser = EnvConfigParser()
        self.validator = ConfigValidator()
        self.profile_manager = ProfileManager()
        self.hierarchy = ConfigHierarchy()
        self.hot_reload = None

        logger.debug("ConfigManager initialized")

    def load_config(
        self,
        config_file: Optional[str] = None,
        profile: Optional[str] = None,
        env_override: bool = True,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        use_global_config: bool = True,
        use_user_config: bool = True,
        use_local_config: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from all sources with proper priority.

        Priority (highest to lowest):
        1. Runtime overrides
        2. Environment variables
        3. Config file
        4. Profile
        5. User config
        6. Global config
        7. Defaults

        Args:
            config_file: Optional path to config file
            profile: Optional profile name
            env_override: Whether to apply environment variables (default: True)
            runtime_overrides: Optional runtime parameter overrides
            use_global_config: Whether to load global config
            use_user_config: Whether to load user config
            use_local_config: Whether to load local config
            validate: Whether to validate the final config (default: True)

        Returns:
            Configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If specified config file doesn't exist
        """
        # Load environment variables if enabled
        env_vars = self.env_parser.parse_env() if env_override else None

        # Resolve from hierarchy
        config = self.hierarchy.resolve_config(
            runtime_overrides=runtime_overrides,
            env_vars=env_vars,
            config_file=config_file,
            profile=profile,
            use_global_config=use_global_config,
            use_user_config=use_user_config,
            use_local_config=use_local_config
        )

        # Validate if requested
        if validate:
            validation_result = self.validator.validate(config)
            if not validation_result.is_valid:
                raise ConfigValidationError(
                    validation_result.get_error_messages(),
                    validation_result.get_warning_messages()
                )

            # Log warnings if any
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Config validation warning: {warning}")

        logger.info(
            f"Loaded configuration with {len(config)} parameters "
            f"(profile={profile}, file={config_file})"
        )

        return config

    def save_config(
        self,
        config: Dict[str, Any],
        filepath: str,
        format: str = 'yaml',
        pretty: bool = True
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            filepath: Path to save file
            format: File format ('yaml', 'json', 'toml')
            pretty: Whether to format output prettily (for JSON)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        if format == 'yaml':
            self.loader.save_yaml(config, filepath)
        elif format == 'json':
            self.loader.save_json(config, filepath, pretty=pretty)
        elif format == 'toml':
            self.loader.save_toml(config, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml', 'json', or 'toml'")

        logger.info(f"Saved configuration to {filepath} (format={format})")

    def enable_hot_reload(
        self,
        config_file: str,
        callback: Callable[[Any], None],
        poll_interval: float = 1.0
    ) -> None:
        """
        Enable hot-reload for config file.

        Args:
            config_file: Path to config file to watch
            callback: Function to call when config changes
            poll_interval: How often to check for changes (seconds)

        Example:
            def on_config_change(event):
                print(f"Config changed: {event.changes}")
                # Apply new config...

            manager.enable_hot_reload('config.yaml', on_config_change)
        """
        from .hot_reload import ConfigHotReload

        self.hot_reload = ConfigHotReload(
            config_file=config_file,
            callback=callback,
            poll_interval=poll_interval
        )
        self.hot_reload.start()

        logger.info(f"Enabled hot-reload for {config_file}")

    def disable_hot_reload(self) -> None:
        """Disable hot-reload"""
        if self.hot_reload:
            self.hot_reload.stop()
            self.hot_reload = None
            logger.info("Disabled hot-reload")

    def create_profile(
        self,
        name: str,
        base: str = 'quickstart',
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create new configuration profile.

        Args:
            name: Name for new profile
            base: Base profile to start from
            overrides: Parameter overrides

        Returns:
            New profile parameters
        """
        return self.profile_manager.create_profile(name, base, overrides)

    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        return self.profile_manager.list_profiles()

    def get_profile_info(self, profile_name: str):
        """Get information about a profile"""
        return self.profile_manager.get_profile_info(profile_name)

    def delete_profile(self, name: str) -> None:
        """Delete a custom profile"""
        self.profile_manager.delete_profile(name)

    def validate_config(self, config: Dict[str, Any]) -> Any:
        """
        Validate a configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult object

        Raises:
            ConfigValidationError: If validation fails
        """
        validation_result = self.validator.validate(config)

        if not validation_result.is_valid:
            raise ConfigValidationError(
                validation_result.get_error_messages(),
                validation_result.get_warning_messages()
            )

        return validation_result

    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dict with parameter info or None if not found
        """
        from .env_mappings import ENV_MAPPINGS, ENV_RANGES

        if param_name not in ENV_MAPPINGS:
            return None

        env_name, param_type = ENV_MAPPINGS[param_name]

        info = {
            'name': param_name,
            'env_var': env_name,
            'type': param_type.__name__,
        }

        if param_name in ENV_RANGES:
            info['range'] = ENV_RANGES[param_name]

        return info

    def list_all_parameters(self) -> List[str]:
        """Get list of all known parameter names"""
        from .env_mappings import ENV_MAPPINGS
        return sorted(ENV_MAPPINGS.keys())

    def get_env_var_for_param(self, param_name: str) -> Optional[str]:
        """
        Get environment variable name for a parameter.

        Args:
            param_name: Parameter name

        Returns:
            Environment variable name or None if not found
        """
        from .env_mappings import ENV_MAPPINGS
        if param_name in ENV_MAPPINGS:
            return ENV_MAPPINGS[param_name][0]
        return None

    def export_env_vars(
        self,
        config: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Export config as environment variable script.

        Args:
            config: Configuration dictionary
            output_file: Optional file to write script to

        Returns:
            Shell script content
        """
        from .env_mappings import ENV_MAPPINGS

        lines = ["# OpenEvolve Configuration Environment Variables", ""]

        for param_name, value in config.items():
            if param_name in ENV_MAPPINGS:
                env_name = ENV_MAPPINGS[param_name][0]
                lines.append(f"export {env_name}={value}")

        script = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(script)
            logger.info(f"Exported environment variables to {output_file}")

        return script

    def compare_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        label1: str = "config1",
        label2: str = "config2"
    ) -> Dict[str, Any]:
        """
        Compare two configurations.

        Args:
            config1: First configuration
            config2: Second configuration
            label1: Label for first config
            label2: Label for second config

        Returns:
            Dict with differences
        """
        differences = {
            'only_in_first': {},
            'only_in_second': {},
            'different_values': {}
        }

        # Keys only in first
        for key in config1:
            if key not in config2:
                differences['only_in_first'][key] = config1[key]

        # Keys only in second
        for key in config2:
            if key not in config1:
                differences['only_in_second'][key] = config2[key]

        # Different values
        for key in config1:
            if key in config2 and config1[key] != config2[key]:
                differences['different_values'][key] = {
                    label1: config1[key],
                    label2: config2[key]
                }

        return differences

    def merge_configs(
        self,
        *configs: Dict[str, Any],
        strategy: str = 'override'
    ) -> Dict[str, Any]:
        """
        Merge multiple configurations.

        Args:
            *configs: Variable number of config dicts
            strategy: Merge strategy ('override', 'deep', 'if_missing', 'if_present')

        Returns:
            Merged configuration
        """
        from .hierarchy import ConfigMerge

        if strategy == 'override':
            return self.hierarchy.merge_configs(*configs)
        elif strategy == 'deep':
            result = {}
            for config in configs:
                result = ConfigMerge.deep_merge(result, config)
            return result
        elif strategy == 'if_missing':
            result = {}
            for config in configs:
                result = ConfigMerge.merge_if_missing(result, config)
            return result
        elif strategy == 'if_present':
            result = {}
            for config in configs:
                result = ConfigMerge.merge_if_present(result, config)
            return result
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    def get_config_sources(self) -> List[Dict[str, Any]]:
        """Get all available configuration sources"""
        return self.hierarchy.get_config_sources()

    def find_config_file(self) -> Optional[str]:
        """Find the highest priority config file that exists"""
        return self.hierarchy.find_config_file()


# Convenience functions for quick operations

def load_config(
    config_file: Optional[str] = None,
    profile: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to load configuration.

    Args:
        config_file: Optional config file path
        profile: Optional profile name
        **kwargs: Additional arguments for ConfigManager.load_config

    Returns:
        Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(config_file=config_file, profile=profile, **kwargs)


def save_config(
    config: Dict[str, Any],
    filepath: str,
    format: str = 'yaml'
) -> None:
    """
    Quick function to save configuration.

    Args:
        config: Configuration dictionary
        filepath: Path to save
        format: File format ('yaml', 'json', 'toml')
    """
    manager = ConfigManager()
    manager.save_config(config, filepath, format)


def create_config(
    profile: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """
    Quick function to create configuration from profile with overrides.

    Args:
        profile: Optional profile name
        **overrides: Parameter overrides

    Returns:
        Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(profile=profile, runtime_overrides=overrides)


def list_profiles() -> List[str]:
    """Quick function to list all profiles"""
    manager = ConfigManager()
    return manager.list_profiles()


def validate_config(config: Dict[str, Any]) -> Any:
    """
    Quick function to validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        ValidationResult
    """
    manager = ConfigManager()
    return manager.validate_config(config)
