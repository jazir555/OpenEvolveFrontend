"""
Configuration Management for Knowledge Engine

Loads configuration from YAML files with environment variable overrides.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def load_config(
    config_path: Optional[str] = None,
    env_prefix: str = "KNOWLEDGE_ENGINE_"
) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides

    Args:
        config_path: Path to configuration file (optional)
        env_prefix: Prefix for environment variables

    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'causal_modeling': {
            'use_causal_learn': True,
            'default_algorithm': 'pc',
            'alpha': 0.05,
            'min_confidence': 0.7
        }
    }

    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Override with environment variables
    # Example: KNOWLEDGE_ENGINE_CAUSAL_MODELING__ALPHA=0.01
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_key = key[len(env_prefix):].lower().replace('__', '.')
            # Simple value parsing (could be enhanced)
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.lower() in ('true', 'yes'):
                value = True
            elif value.lower() in ('false', 'no'):
                value = False

            # Set nested config value
            keys = config_key.split('.')
            config = default_config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            logger.info(f"Override config: {config_key} = {value}")

    return default_config


def load_causal_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load causal modeling configuration

    Args:
        config_path: Path to causal config file (optional)

    Returns:
        Causal modeling configuration
    """
    if config_path is None:
        # Default path
        config_dir = Path(__file__).parent
        config_path = config_dir / "causal_config.yaml"

    config = load_config(str(config_path), env_prefix="CAUSAL_")
    return config.get('causal_modeling', config)
