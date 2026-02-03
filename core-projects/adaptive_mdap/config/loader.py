"""Configuration loader with environment variable overrides."""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Loader
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from adaptive_mdap.core.errors import ConfigurationError


@dataclass
class ClassifierConfig:
    """Configuration for complexity classifier."""
    embedding_model: str = "all-MiniLM-L6-v2"
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "text_length": 0.2,
        "domain_rarity": 0.2,
        "depth": 0.2,
        "historical_error": 0.2,
        "dependency": 0.2,
    })
    cache_dir: str = ".cache/adaptive_mdap"
    cache_ttl_hours: int = 24


@dataclass
class AllocatorConfig:
    """Configuration for resource allocator."""
    thresholds: list = field(default_factory=lambda: [0.3, 0.7])
    enable_learning: bool = False
    default_strategy: str = "maker_full"


@dataclass
class StrategyConfig:
    """Configuration for solving strategies."""
    direct: Dict[str, Any] = field(default_factory=lambda: {
        "n_agents": 1,
        "k_ahead": 0,
        "max_retries": 1,
        "timeout_ms": 30000,
    })
    mdap_light: Dict[str, Any] = field(default_factory=lambda: {
        "n_agents": 3,
        "k_ahead": 1,
        "max_retries": 2,
        "timeout_ms": 60000,
    })
    maker_full: Dict[str, Any] = field(default_factory=lambda: {
        "n_agents": 5,
        "k_ahead": 2,
        "max_retries": 3,
        "timeout_ms": 120000,
    })


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    enabled: bool = True
    metrics_export_format: str = "json"  # json, prometheus, csv
    log_level: str = "INFO"
    enable_structured_logging: bool = True


@dataclass
class AdaptiveMDAPConfig:
    """Main configuration for Adaptive MDAP."""
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveMDAPConfig":
        """Create configuration from dictionary."""
        return cls(
            classifier=ClassifierConfig(**data.get("classifier", {})),
            allocator=AllocatorConfig(**data.get("allocator", {})),
            strategies=StrategyConfig(**data.get("strategies", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
        )


class ConfigLoader:
    """Loads configuration from YAML files with environment overrides."""
    
    DEFAULT_CONFIG_PATH = Path("config/adaptive_mdap.yaml")
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config: Optional[AdaptiveMDAPConfig] = None
    
    def load(self) -> AdaptiveMDAPConfig:
        """Load configuration from file."""
        if self._config is not None:
            return self._config
        
        # Load from file if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        
        # Apply environment variable overrides
        data = self._apply_env_overrides(data)
        
        # Validate
        self._validate(data)
        
        self._config = AdaptiveMDAPConfig.from_dict(data)
        return self._config
    
    def _apply_env_overrides(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Classifier overrides
        if env_model := os.getenv("ADAPTIVE_MDAP_EMBEDDING_MODEL"):
            data.setdefault("classifier", {})["embedding_model"] = env_model
        
        if env_cache := os.getenv("ADAPTIVE_MDAP_CACHE_DIR"):
            data.setdefault("classifier", {})["cache_dir"] = env_cache
        
        # Allocator overrides
        if env_thresholds := os.getenv("ADAPTIVE_MDAP_THRESHOLDS"):
            try:
                thresholds = [float(x.strip()) for x in env_thresholds.split(",")]
                data.setdefault("allocator", {})["thresholds"] = thresholds
            except ValueError:
                raise ConfigurationError(
                    f"Invalid ADAPTIVE_MDAP_THRESHOLDS: {env_thresholds}",
                    config_key="allocator.thresholds"
                )
        
        # Monitoring overrides
        if env_log_level := os.getenv("ADAPTIVE_MDAP_LOG_LEVEL"):
            data.setdefault("monitoring", {})["log_level"] = env_log_level
        
        return data
    
    def _validate(self, data: Dict[str, Any]) -> None:
        """Validate configuration."""
        # Validate allocator thresholds
        if thresholds := data.get("allocator", {}).get("thresholds"):
            if len(thresholds) != 2:
                raise ConfigurationError(
                    f"Thresholds must have exactly 2 values, got {len(thresholds)}",
                    config_key="allocator.thresholds"
                )
            if not (0.0 <= thresholds[0] < thresholds[1] <= 1.0):
                raise ConfigurationError(
                    f"Thresholds must satisfy 0.0 <= t0 < t1 <= 1.0, got {thresholds}",
                    config_key="allocator.thresholds"
                )
        
        # Validate feature weights
        if weights := data.get("classifier", {}).get("feature_weights"):
            total = sum(weights.values())
            if not 0.99 <= total <= 1.01:  # Allow small floating point errors
                raise ConfigurationError(
                    f"Feature weights must sum to 1.0, got {total}",
                    config_key="classifier.feature_weights"
                )
    
    def reload(self) -> AdaptiveMDAPConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load()
