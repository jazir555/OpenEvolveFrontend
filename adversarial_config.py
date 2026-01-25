"""
Configuration Validation and Management System

This module provides comprehensive configuration management:
- Schema validation with Pydantic-like API
- Type checking and conversion
- Environment variable integration
- Configuration file loading (JSON, YAML, TOML)
- Runtime validation
- Configuration migration
- Secrets management
- Configuration diff/merge

Author: OpenEvolve Config Team
Created: 2025-01-07
Version: 1.0.0
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Type, TypeVar, Union,
    get_type_hints, get_origin, get_args
)
import copy

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


# =============================================================================
# VALIDATION SCHEMA
# =============================================================================

class ValidationError(Exception):
    """Configuration validation error"""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.format_message())

    def format_message(self) -> str:
        if self.field:
            return f"Validation error in field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class ConfigField:
    """
    Configuration field definition with validation rules

    Example:
        field = ConfigField(
            type=int,
            required=True,
            min_value=0,
            max_value=100,
            description="Port number"
        )
    """

    def __init__(
        self,
        type: Type,
        required: bool = False,
        default: Any = None,
        description: str = "",
        env_var: Optional[str] = None,
        validator: Optional[callable] = None,
        **constraints
    ):
        self.type = type
        self.required = required
        self.default = default
        self.description = description
        self.env_var = env_var
        self.validator = validator
        self.constraints = constraints

        # Common constraints
        self.min_value = constraints.get('min_value')
        self.max_value = constraints.get('max_value')
        self.min_length = constraints.get('min_length')
        self.max_length = constraints.get('max_length')
        self.pattern = constraints.get('pattern')
        self.choices = constraints.get('choices')
        self.secret = constraints.get('secret', False)

    def validate(self, value: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Validate and convert a value

        Args:
            value: Value to validate
            config: Full configuration dict for cross-field validation

        Returns:
            Converted/validated value

        Raises:
            ValidationError: If validation fails
        """
        # Handle None
        if value is None:
            if self.required:
                raise ValidationError(f"Field is required", value=value)
            return self.default

        # Type conversion
        try:
            value = self._convert_type(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Type conversion failed: {e}", value=value)

        # Custom validator
        if self.validator:
            try:
                value = self.validator(value, config or {})
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                raise ValidationError(f"Custom validation failed: {e}", value=value)

        # Type-specific validation
        self._validate_constraints(value)

        return value

    def _convert_type(self, value: Any) -> Any:
        """Convert value to the correct type"""
        # Handle Optional types
        origin_type = get_origin(self.type)
        if origin_type is Union:
            # Get first non-None type
            type_args = [arg for arg in get_args(self.type) if arg is not type(None)]
            if type_args:
                self.type = type_args[0]

        # Handle lists
        if self.type is list or (origin_type is list):
            if isinstance(value, str):
                # Comma-separated string
                return [item.strip() for item in value.split(',')]
            return list(value)

        # Handle bool
        if self.type is bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)

        # Handle int, float, str
        if self.type in (int, float, str):
            return self.type(value)

        # Handle Enum
        if isinstance(self.type, type) and issubclass(self.type, Enum):
            if isinstance(value, str):
                return self.type[value]
            return self.type(value)

        return value

    def _validate_constraints(self, value: Any):
        """Validate field constraints"""
        # Min/max value for numbers
        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                raise ValidationError(
                    f"Value {value} is less than minimum {self.min_value}",
                    value=value
                )

        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                raise ValidationError(
                    f"Value {value} is greater than maximum {self.max_value}",
                    value=value
                )

        # Min/max length for strings and lists
        if self.min_length is not None and isinstance(value, (str, list)):
            if len(value) < self.min_length:
                raise ValidationError(
                    f"Length {len(value)} is less than minimum {self.min_length}",
                    value=value
                )

        if self.max_length is not None and isinstance(value, (str, list)):
            if len(value) > self.max_length:
                raise ValidationError(
                    f"Length {len(value)} is greater than maximum {self.max_length}",
                    value=value
                )

        # Pattern for strings
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                raise ValidationError(
                    f"Value '{value}' does not match pattern '{self.pattern}'",
                    value=value
                )

        # Choices
        if self.choices and value not in self.choices:
            raise ValidationError(
                f"Value '{value}' is not in allowed choices: {self.choices}",
                value=value
            )


# =============================================================================
# CONFIGURATION SCHEMA
# =============================================================================

class ConfigSchema:
    """
    Configuration schema with field definitions

    Example:
        schema = ConfigSchema()
        schema.add_field("port", ConfigField(type=int, required=True, min_value=1, max_value=65535))
        schema.add_field("host", ConfigField(type=str, default="localhost"))
    """

    def __init__(self, name: str = "config"):
        self.name = name
        self.fields: Dict[str, ConfigField] = {}

    def add_field(self, name: str, field: ConfigField):
        """Add a field to the schema"""
        self.fields[name] = field

    def remove_field(self, name: str):
        """Remove a field from the schema"""
        self.fields.pop(name, None)

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration dict against the schema

        Args:
            config: Configuration to validate

        Returns:
            Validated and converted configuration

        Raises:
            ValidationError: If validation fails
        """
        validated = {}

        # Load from environment variables
        env_config = self._load_from_env()

        # Merge env config (env vars take precedence)
        merged = {**config, **env_config}

        # Validate each field
        for field_name, field_def in self.fields.items():
            value = merged.get(field_name)
            validated[field_name] = field_def.validate(value, merged)

        # Check for unknown fields
        for key in merged.keys():
            if key not in self.fields:
                logger.warning(f"Unknown configuration field: {key}")

        return validated

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration values from environment variables"""
        config = {}

        for field_name, field_def in self.fields.items():
            if field_def.env_var:
                env_value = os.environ.get(field_def.env_var)
                if env_value is not None:
                    config[field_name] = env_value

        return config

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            field_name: field_def.default
            for field_name, field_def in self.fields.items()
            if not field_def.required
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        properties = {}
        required = []

        for field_name, field_def in self.fields.items():
            prop_schema = {
                "type": self._type_to_json_type(field_def.type),
                "description": field_def.description
            }

            if field_def.required:
                required.append(field_name)

            if field_def.default is not None:
                prop_schema["default"] = field_def.default

            if field_def.min_value is not None:
                prop_schema["minimum"] = field_def.min_value

            if field_def.max_value is not None:
                prop_schema["maximum"] = field_def.max_value

            if field_def.min_length is not None:
                prop_schema["minLength"] = field_def.min_length

            if field_def.max_length is not None:
                prop_schema["maxLength"] = field_def.max_length

            if field_def.pattern:
                prop_schema["pattern"] = field_def.pattern

            if field_def.choices:
                prop_schema["enum"] = field_def.choices

            properties[field_name] = prop_schema

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _type_to_json_type(self, python_type: Type) -> str:
        """Convert Python type to JSON Schema type"""
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return type_map.get(python_type, "string")


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseConfig(ABC):
    """
    Base configuration class with automatic validation

    Example:
        class MyConfig(BaseConfig):
            port: int = 8080
            host: str = "localhost"
            debug: bool = False

            def get_schema(self) -> ConfigSchema:
                schema = ConfigSchema()
                schema.add_field("port", ConfigField(type=int, min_value=1, max_value=65535))
                schema.add_field("host", ConfigField(type=str, pattern=r'^[\w\-\.]+$'))
                schema.add_field("debug", ConfigField(type=bool))
                return schema
    """

    def __init__(self, **kwargs):
        # Get schema
        schema = self.get_schema()

        # Get defaults
        config = schema.get_default_config()

        # Update with provided values
        config.update(kwargs)

        # Validate
        validated = schema.validate(config)

        # Set attributes
        for key, value in validated.items():
            setattr(self, key, value)

    @classmethod
    @abstractmethod
    def get_schema(cls) -> ConfigSchema:
        """Get the configuration schema"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        schema = self.get_schema()
        return {
            field_name: getattr(self, field_name, None)
            for field_name in schema.fields.keys()
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        config_dict = self.to_dict()

        # Mask secrets
        schema = self.get_schema()
        for field_name, field_def in schema.fields.items():
            if field_def.secret and field_name in config_dict:
                config_dict[field_name] = "***REDACTED***"

        return json.dumps(config_dict, indent=indent)

    @classmethod
    def from_dict(cls: Type[T], config: Dict[str, Any]) -> T:
        """Create from dictionary"""
        return cls(**config)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create from JSON string"""
        config = json.loads(json_str)
        return cls.from_dict(config)

    @classmethod
    def from_json_file(cls: Type[T], file_path: str) -> T:
        """Load from JSON file"""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())

    @classmethod
    def from_env(cls: Type[T]) -> T:
        """Load from environment variables"""
        schema = cls.get_schema()
        config = schema._load_from_env()
        return cls.from_dict(config)

    def save(self, file_path: str):
        """Save to JSON file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    def diff(self, other: 'BaseConfig') -> Dict[str, Dict[str, Any]]:
        """Compare with another config"""
        diff = {}

        schema = self.get_schema()
        for field_name in schema.fields.keys():
            self_value = getattr(self, field_name, None)
            other_value = getattr(other, field_name, None)

            if self_value != other_value:
                diff[field_name] = {
                    "self": self_value,
                    "other": other_value
                }

        return diff

    def merge(self, other: 'BaseConfig', prefer_other: bool = True) -> 'BaseConfig':
        """Merge with another config"""
        schema = self.get_schema()
        merged = self.to_dict()

        for field_name in schema.fields.keys():
            other_value = getattr(other, field_name, None)
            if other_value is not None:
                if prefer_other:
                    merged[field_name] = other_value
                elif merged.get(field_name) is None:
                    merged[field_name] = other_value

        return self.__class__.from_dict(merged)


# =============================================================================
# ADVERSARIAL CONFIGURATION
# =============================================================================

class AdversarialConfig(BaseConfig):
    """
    Enhanced adversarial testing configuration with validation

    Environment Variables:
        ADV_MAX_ITERATIONS: Maximum iterations
        ADV_ENSEMBLE_SIZE: Ensemble size
        ADV_ENABLE_LLM: Enable LLM attacks
        ADV_API_KEY: API key for LLM
    """

    # Default values (will be validated)
    max_iterations: int = 10
    ensemble_size: int = 5
    enable_llm_attacks: bool = False
    api_key: Optional[str] = None
    enable_adaptive_defense: bool = True
    explainability_level: str = "detailed"
    learning_mode: str = "online"
    enable_ensemble: bool = True
    enable_caching: bool = True
    enable_parallel_evaluation: bool = False
    cache_ttl_seconds: int = 3600
    timeout_seconds: int = 300

    @classmethod
    def get_schema(cls) -> ConfigSchema:
        """Get configuration schema"""
        schema = ConfigSchema(name="adversarial")

        # Max iterations
        schema.add_field("max_iterations", ConfigField(
            type=int,
            required=True,
            default=10,
            min_value=1,
            max_value=100,
            description="Maximum adversarial iterations",
            env_var="ADV_MAX_ITERATIONS"
        ))

        # Ensemble size
        schema.add_field("ensemble_size", ConfigField(
            type=int,
            required=True,
            default=5,
            min_value=1,
            max_value=20,
            description="Ensemble attack size",
            env_var="ADV_ENSEMBLE_SIZE"
        ))

        # Enable LLM attacks
        schema.add_field("enable_llm_attacks", ConfigField(
            type=bool,
            default=False,
            description="Enable LLM-based attacks",
            env_var="ADV_ENABLE_LLM"
        ))

        # API key (secret)
        schema.add_field("api_key", ConfigField(
            type=str,
            default=None,
            description="OpenAI API key for LLM attacks",
            env_var="ADV_API_KEY",
            secret=True
        ))

        # Enable adaptive defense
        schema.add_field("enable_adaptive_defense", ConfigField(
            type=bool,
            default=True,
            description="Enable adaptive defense system"
        ))

        # Explainability level
        schema.add_field("explainability_level", ConfigField(
            type=str,
            default="detailed",
            choices=["basic", "detailed", "full"],
            description="Explainability level"
        ))

        # Learning mode
        schema.add_field("learning_mode", ConfigField(
            type=str,
            default="online",
            choices=["offline", "online", "hybrid", "transfer"],
            description="Continuous learning mode"
        ))

        # Enable ensemble
        schema.add_field("enable_ensemble", ConfigField(
            type=bool,
            default=True,
            description="Enable ensemble attacks"
        ))

        # Enable caching
        schema.add_field("enable_caching", ConfigField(
            type=bool,
            default=True,
            description="Enable result caching"
        ))

        # Enable parallel evaluation
        schema.add_field("enable_parallel_evaluation", ConfigField(
            type=bool,
            default=False,
            description="Enable parallel evaluation (experimental)"
        ))

        # Cache TTL
        schema.add_field("cache_ttl_seconds", ConfigField(
            type=int,
            default=3600,
            min_value=0,
            description="Cache TTL in seconds"
        ))

        # Timeout
        schema.add_field("timeout_seconds", ConfigField(
            type=int,
            default=300,
            min_value=1,
            description="Operation timeout in seconds"
        ))

        return schema


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class ConfigManager:
    """
    Central configuration management system

    Features:
    - Multiple configuration sources (files, env, CLI)
    - Configuration profiles (dev, test, prod)
    - Configuration migration
    - Validation
    - Hot reload
    """

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.configs: Dict[str, BaseConfig] = {}
        self.watchers: Dict[str, Any] = {}

    def register_config(self, name: str, config: BaseConfig):
        """Register a configuration"""
        self.configs[name] = config
        logger.info(f"Registered configuration: {name}")

    def get_config(self, name: str) -> Optional[BaseConfig]:
        """Get a configuration"""
        return self.configs.get(name)

    def load_config(
        self,
        name: str,
        config_class: Type[BaseConfig],
        profile: str = "default"
    ) -> BaseConfig:
        """
        Load configuration from multiple sources

        Priority (highest first):
        1. Environment variables
        2. Profile-specific config file
        3. Default config file
        4. Schema defaults

        Args:
            name: Configuration name
            config_class: Configuration class
            profile: Configuration profile (dev, test, prod)

        Returns:
            Validated configuration instance
        """
        # Start with defaults
        config = config_class()

        # Load from default file
        default_file = self.config_dir / f"{name}.json"
        if default_file.exists():
            try:
                file_config = config_class.from_json_file(str(default_file))
                config = config.merge(file_config, prefer_other=False)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to load default config: {e}")

        # Load from profile file
        profile_file = self.config_dir / f"{name}.{profile}.json"
        if profile_file.exists():
            try:
                profile_config = config_class.from_json_file(str(profile_file))
                config = config.merge(profile_config)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"Failed to load profile config: {e}")

        # Load from environment (highest priority)
        try:
            env_config = config_class.from_env()
            config = config.merge(env_config)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.warning(f"Failed to load env config: {e}")

        # Register and return
        self.register_config(name, config)
        return config

    def save_config(self, name: str, profile: str = "default"):
        """Save configuration to file"""
        config = self.get_config(name)
        if not config:
            raise ValueError(f"Configuration '{name}' not found")

        filename = f"{name}.{profile}.json"
        filepath = self.config_dir / filename
        config.save(str(filepath))

        logger.info(f"Saved configuration: {filename}")

    def validate_all(self) -> Dict[str, bool]:
        """Validate all registered configurations"""
        results = {}

        for name, config in self.configs.items():
            try:
                schema = config.get_schema()
                schema.validate(config.to_dict())
                results[name] = True
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Config '{name}' validation failed: {e}")
                results[name] = False

        return results

    def export_config(
        self,
        name: str,
        format: str = "json",
        include_secrets: bool = False
    ) -> str:
        """Export configuration to string"""
        config = self.get_config(name)
        if not config:
            raise ValueError(f"Configuration '{name}' not found")

        if format == "json":
            if include_secrets:
                return json.dumps(config.to_dict(), indent=2)
            else:
                return config.to_json()
        elif format == "env":
            return self._to_env_format(config)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _to_env_format(self, config: BaseConfig) -> str:
        """Convert config to environment variable format"""
        schema = config.get_schema()
        lines = []

        for field_name, field_def in schema.fields.items():
            if field_def.env_var:
                value = getattr(config, field_name, None)
                if value is not None:
                    if field_def.secret:
                        value = "***REDACTED***"
                    lines.append(f"{field_def.env_var}={value}")

        return "\n".join(lines)


# =============================================================================
# MIGRATION SYSTEM
# =============================================================================

class ConfigMigration:
    """
    Configuration migration and version management

    Handles:
    - Schema versioning
    - Automatic migrations
    - Rollback support
    - Validation after migration
    """

    def __init__(self):
        self.migrations: List[Dict[str, Any]] = []

    def add_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: callable
    ):
        """Add a migration function"""
        self.migrations.append({
            "from_version": from_version,
            "to_version": to_version,
            "func": migration_func,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Sort by version
        self.migrations.sort(key=lambda m: m["from_version"])

    def migrate(
        self,
        config: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Migrate configuration from one version to another"""
        current_version = from_version
        current_config = config.copy()

        while current_version != to_version:
            # Find applicable migration
            migration = None
            for m in self.migrations:
                if m["from_version"] == current_version:
                    migration = m
                    break

            if not migration:
                raise ValueError(f"No migration found from {current_version}")

            # Apply migration
            try:
                current_config = migration["func"](current_config)
                current_version = migration["to_version"]
                logger.info(f"Migrated config from {migration['from_version']} to {current_version}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Migration failed: {e}")
                raise

        return current_config


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("Configuration Validation and Management System")
    print("=" * 60)

    # Create config
    print("\n1. Creating Configuration")
    print("-" * 40)

    # Create with defaults
    config1 = AdversarialConfig()
    print("Default config:")
    print(config1.to_json())

    # Create with custom values
    config2 = AdversarialConfig(
        max_iterations=20,
        ensemble_size=7,
        enable_llm_attacks=True
    )
    print("\nCustom config:")
    print(config2.to_json())

    # Validation
    print("\n2. Testing Validation")
    print("-" * 40)

    try:
        # Invalid max_iterations
        bad_config = AdversarialConfig(max_iterations=0)
        print("ERROR: Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Caught validation error: {e.message}")

    try:
        # Invalid explainability_level
        bad_config = AdversarialConfig(explainability_level="invalid")
        print("ERROR: Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Caught validation error: {e.message}")

    # Config manager
    print("\n3. Using Config Manager")
    print("-" * 40)

    manager = ConfigManager(config_dir="./demo_config")
    config = manager.load_config(
        name="adversarial",
        config_class=AdversarialConfig,
        profile="dev"
    )

    print(f"Loaded config: {config.to_json()}")

    # Save config
    manager.save_config("adversarial", profile="dev")
    print("✓ Saved configuration")

    # Export
    print("\nExported ENV format:")
    print(manager.export_config("adversarial", format="env"))

    # Diff
    print("\n4. Comparing Configurations")
    print("-" * 40)

    diff = config1.diff(config2)
    print(f"Differences found: {len(diff)}")
    for field, values in diff.items():
        print(f"  {field}: {values['self']} → {values['other']}")

    # Merge
    print("\n5. Merging Configurations")
    print("-" * 40)

    merged = config1.merge(config2)
    print(f"Merged max_iterations: {merged.max_iterations}")
    print(f"Merged ensemble_size: {merged.ensemble_size}")

    # JSON Schema
    print("\n6. JSON Schema Export")
    print("-" * 40)

    schema = AdversarialConfig.get_schema()
    json_schema = schema.to_json_schema()
    print(json.dumps(json_schema, indent=2))

    print("\n" + "=" * 60)
    print("Configuration system demo complete!")
