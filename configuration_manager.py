import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Manages loading and providing configuration settings from config.yaml.
    """
    _instance = None

    def __new__(cls, config_path: str = "config.yaml", env: str = "default"):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config.yaml", env: str = "default"):
        if self._initialized:
            return
        self.config_path = config_path
        self.env = env
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """
        Loads configuration from the YAML file.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Load default settings
        self._config = full_config.get("default", {})

        # Override with environment-specific settings
        if self.env != "default":
            env_config = full_config.get(self.env, {})
            self._config.update(env_config)
        
        logger.info(f"Configuration loaded for environment: {self.env}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by key.
        """
        return self._config.get(key, default)

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Returns the performance optimization configuration.
        """
        return self.get_config("performance_optimization", {})

    def get_reliability_config(self) -> Dict[str, Any]:
        """
        Returns the reliability configuration.
        """
        return self.get_config("reliability", {})

    def get_openevolve_config(self) -> Dict[str, Any]:
        """
        Returns the OpenEvolve client configuration.
        """
        return self.get_config("openevolve_client", {})

# Global instance for easy access
config_manager = ConfigurationManager()
