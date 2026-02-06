"""Configuration profiles for different use cases."""

import os
from enum import Enum
from typing import Dict, Any
from pathlib import Path

# Try to import yaml, fallback to JSON if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    import json


class ConfigProfile(Enum):
    """Predefined configuration profiles."""
    CONSERVATIVE = "conservative"  # Favor quality over cost
    BALANCED = "balanced"          # Default balance
    AGGRESSIVE = "aggressive"      # Favor cost over quality
    CLOUD_CONSERVATIVE = "cloud_conservative"
    CLOUD_BALANCED = "cloud_balanced"
    CLOUD_AGGRESSIVE = "cloud_aggressive"


def get_profile_config(profile: ConfigProfile) -> Dict[str, Any]:
    """Get configuration for a specific profile with granular strategies."""
    
    base = {
        "classifier": {
            "embedding_model": "all-MiniLM-L6-v2",
            "feature_weights": {
                "text_length": 0.15,
                "domain_rarity": 0.20,
                "depth": 0.15,
                "historical_error": 0.20,
                "dependency": 0.10,
                "keyword_complexity": 0.10,
                "constraint_density": 0.10
            },
            "cache_ttl_hours": 24,
        },
        "strategies": {
            "direct": {"n_agents": 1, "k_ahead": 0, "max_retries": 1, "timeout_ms": 30000},
            "mdap_light": {"n_agents": 3, "k_ahead": 1, "max_retries": 2, "timeout_ms": 60000},
            "mdap_medium": {"n_agents": 5, "k_ahead": 1, "max_retries": 2, "timeout_ms": 90000},
            "maker_full": {"n_agents": 5, "k_ahead": 2, "max_retries": 3, "timeout_ms": 120000},
            "maker_ultra": {"n_agents": 7, "k_ahead": 3, "max_retries": 4, "timeout_ms": 180000},
        },
        "monitoring": {
            "enabled": True,
            "metrics_export_format": "json",
            "log_level": "INFO",
            "enable_structured_logging": True,
        },
    }
    
    if profile == ConfigProfile.CONSERVATIVE:
        base["allocator"] = {
            "thresholds": [0.1, 0.3, 0.5, 0.7],  # Lower thresholds = favor quality early
            "enable_learning": False,
        }
    elif profile == ConfigProfile.BALANCED:
        base["allocator"] = {
            "thresholds": [0.2, 0.4, 0.6, 0.8],  # Standard balanced distribution
            "enable_learning": False,
        }
    elif profile == ConfigProfile.AGGRESSIVE:
        base["allocator"] = {
            "thresholds": [0.3, 0.5, 0.7, 0.9],  # Higher thresholds = favor cost savings
            "enable_learning": False,
        }
    elif profile == ConfigProfile.CLOUD_CONSERVATIVE:
        base["allocator"] = {
            "thresholds": [0.1, 0.2, 0.4, 0.6],
            "enable_learning": True,
        }
        # Extend timeouts for cloud
        for s in base["strategies"]:
            base["strategies"][s]["timeout_ms"] = int(base["strategies"][s]["timeout_ms"] * 1.5)
    elif profile == ConfigProfile.CLOUD_BALANCED:
        base["allocator"] = {
            "thresholds": [0.2, 0.4, 0.6, 0.8],
            "enable_learning": True,
        }
    elif profile == ConfigProfile.CLOUD_AGGRESSIVE:
        base["allocator"] = {
            "thresholds": [0.4, 0.6, 0.8, 0.95],
            "enable_learning": True,
        }
    
    return base


def load_profile(profile: ConfigProfile, save_to: Path = None) -> Dict[str, Any]:
    """Load a configuration profile and optionally save to file."""
    config = get_profile_config(profile)
    
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, 'w') as f:
            if YAML_AVAILABLE:
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
    
    return config


__all__ = ["ConfigProfile", "get_profile_config", "load_profile"]
