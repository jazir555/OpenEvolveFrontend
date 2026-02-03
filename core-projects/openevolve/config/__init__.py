"""
OpenEvolve Configuration System

A comprehensive, highly configurable system supporting:
- Multiple file formats (YAML, JSON, TOML)
- Environment variable parsing
- Configuration profiles (dev, test, prod)
- Hierarchical overrides
- Hot-reload capability
- Comprehensive validation

Usage:
    from openevolve.config import ConfigManager

    manager = ConfigManager()
    config = manager.load_config(
        config_file='config.yaml',
        profile='development',
        env_override=True
    )
"""

from .config_loader import ConfigLoader
from .env_parser import EnvConfigParser
from .validator import ConfigValidator, ValidationResult, ValidationError
from .profiles import ProfileManager, DevelopmentProfile, TestingProfile, ProductionProfile
from .hierarchy import ConfigHierarchy
from .hot_reload import ConfigHotReload
from .manager import ConfigManager
from .env_mappings import ENV_MAPPINGS, env_name_to_config, config_to_env_name

__all__ = [
    'ConfigLoader',
    'EnvConfigParser',
    'ConfigValidator',
    'ValidationResult',
    'ValidationError',
    'ProfileManager',
    'DevelopmentProfile',
    'TestingProfile',
    'ProductionProfile',
    'ConfigHierarchy',
    'ConfigHotReload',
    'ConfigManager',
    'ENV_MAPPINGS',
    'env_name_to_config',
    'config_to_env_name',
]

__version__ = '1.0.0'
