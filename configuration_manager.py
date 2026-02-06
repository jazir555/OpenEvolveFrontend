import yaml
import os
import threading
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Manages loading and providing configuration settings from config.yaml.
    Thread-safe singleton pattern with double-checked locking.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str = "config.yaml", env: str = "default"):
        # Double-checked locking for thread-safe singleton
        if cls._instance is None:
            with cls._lock:
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
        Uses default configuration if file is not found.
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}, using defaults")
            self._config = self._get_default_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f) or {}

            # Load default settings
            self._config = full_config.get("default", {})

            # Override with environment-specific settings
            if self.env != "default":
                env_config = full_config.get(self.env, {})
                self._config.update(env_config)

            logger.info(f"Configuration loaded for environment: {self.env}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}, using defaults")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Returns default configuration when config file is not available.
        """
        return {
            "performance_optimization": {
                "enable_caching": True,
                "cache_ttl_seconds": 300
            },
            "reliability": {
                "enable_retries": True,
                "max_retries": 3
            },
            "openevolve_client": {
                "timeout_seconds": 30
            }
        }

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
