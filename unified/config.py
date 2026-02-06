"""Unified Configuration Module

Centralized configuration management for unified components.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class UnifiedConfig:
    """Configuration for unified system."""
    
    # General settings
    debug: bool = False
    log_level: str = "INFO"
    
    # Evolution settings
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_timeout: int = 30
    
    # Storage settings
    storage_backend: str = "memory"
    storage_path: Optional[str] = None
    
    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    
    # Extensions
    extensions: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manager for unified configuration.
    
    Provides centralized access to configuration settings
    with support for loading from different sources.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[UnifiedConfig] = None
    
    def __new__(cls):
        """Singleton pattern for config manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = UnifiedConfig()
        return cls._instance
    
    @property
    def config(self) -> UnifiedConfig:
        """Get current configuration.
        
        Returns:
            UnifiedConfig instance
        """
        if self._config is None:
            self._config = UnifiedConfig()
        return self._config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.extensions[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return self.config.extensions.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        if hasattr(self.config, key):
            setattr(self.config, key, value)
        else:
            self.config.extensions[key] = value


def get_config() -> UnifiedConfig:
    """Get global configuration.
    
    Returns:
        UnifiedConfig instance
    """
    return ConfigManager().config


def load_config(config_dict: Dict[str, Any]) -> None:
    """Load configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
    """
    ConfigManager().load_from_dict(config_dict)


__all__ = [
    "UnifiedConfig",
    "ConfigManager",
    "get_config",
    "load_config"
]
