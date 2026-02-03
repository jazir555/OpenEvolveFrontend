"""
Visualization Configuration

Environment-based configuration following CLAUDE.md principles.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class VisualizationConfig:
    """
    Configuration for visualization system.

    All values loaded from environment variables with validation.
    Crashes immediately if required config is missing (CLAUDE.md compliance).
    """

    # Default values (must be overridden by env vars in production)
    DEFAULTS = {
        'VISUALIZATION_OUTPUT_DIR': 'data/visualizations',
        'VISUALIZATION_CACHE_DIR': 'data/visualization_cache',
        'VISUALIZATION_MAX_NODES': '10000',
        'VISUALIZATION_MAX_EDGES': '50000',
        'VISUALIZATION_DEFAULT_WIDTH': '1200',
        'VISUALIZATION_DEFAULT_HEIGHT': '800',
        'VISUALIZATION_CACHE_TTL': '3600',
        'VISUALIZATION_EXPORT_TIMEOUT': '30',
        'VISUALIZATION_ENABLE_CACHING': 'true',
    }

    def __init__(self):
        """Initialize configuration from environment variables."""
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        self._ensure_directories()

        logger.info({
            'event': 'visualization_config_loaded',
            'output_dir': self.output_dir,
            'cache_enabled': self.enable_caching,
            'max_nodes': self.max_nodes,
            'timestamp': datetime.utcnow().isoformat()
        })

    def _load_config(self):
        """Load configuration from environment with defaults."""
        for key, default in self.DEFAULTS.items():
            value = os.getenv(key, default)
            self._config[key] = value

    def _validate_config(self):
        """
        Validate configuration.

        Following CLAUDE.md: crash immediately if misconfigured.
        """
        # Validate paths
        try:
            Path(self._config['VISUALIZATION_OUTPUT_DIR'])
            Path(self._config['VISUALIZATION_CACHE_DIR'])
        except Exception as e:
            logger.error({
                'event': 'config_validation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise ValueError(f"Invalid path configuration: {e}")

        # Validate numeric values
        try:
            int(self._config['VISUALIZATION_MAX_NODES'])
            int(self._config['VISUALIZATION_MAX_EDGES'])
            int(self._config['VISUALIZATION_DEFAULT_WIDTH'])
            int(self._config['VISUALIZATION_DEFAULT_HEIGHT'])
            int(self._config['VISUALIZATION_CACHE_TTL'])
            int(self._config['VISUALIZATION_EXPORT_TIMEOUT'])
        except ValueError as e:
            logger.error({
                'event': 'config_validation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise ValueError(f"Invalid numeric configuration: {e}")

        # Validate boolean
        cache_enabled = self._config['VISUALIZATION_ENABLE_CACHING'].lower()
        if cache_enabled not in ['true', 'false']:
            raise ValueError(f"VISUALIZATION_ENABLE_CACHING must be 'true' or 'false'")

    def _ensure_directories(self):
        """Create output and cache directories if they don't exist."""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error({
                'event': 'directory_creation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise

    @property
    def output_dir(self) -> str:
        return self._config['VISUALIZATION_OUTPUT_DIR']

    @property
    def cache_dir(self) -> str:
        return self._config['VISUALIZATION_CACHE_DIR']

    @property
    def max_nodes(self) -> int:
        return int(self._config['VISUALIZATION_MAX_NODES'])

    @property
    def max_edges(self) -> int:
        return int(self._config['VISUALIZATION_MAX_EDGES'])

    @property
    def default_width(self) -> int:
        return int(self._config['VISUALIZATION_DEFAULT_WIDTH'])

    @property
    def default_height(self) -> int:
        return int(self._config['VISUALIZATION_DEFAULT_HEIGHT'])

    @property
    def cache_ttl(self) -> int:
        return int(self._config['VISUALIZATION_CACHE_TTL'])

    @property
    def export_timeout(self) -> int:
        return int(self._config['VISUALIZATION_EXPORT_TIMEOUT'])

    @property
    def enable_caching(self) -> bool:
        return self._config['VISUALIZATION_ENABLE_CACHING'].lower() == 'true'

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.copy()


# Singleton instance
_config_instance: Optional[VisualizationConfig] = None


def get_visualization_config() -> VisualizationConfig:
    """Get singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = VisualizationConfig()
    return _config_instance
