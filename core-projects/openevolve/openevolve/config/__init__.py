"""
Runtime Configuration System

This package provides dynamic configuration capabilities for OpenEvolve:
- Runtime parameter updates
- Configuration file watching (hot-reload)
- Dynamic strategy switching
- Adaptive configuration
- Resource-aware configuration
- Configuration metrics
"""

# Import old config for backward compatibility
import sys
from pathlib import Path

# Add parent directory to path to import config.py
_config_path = Path(__file__).parent.parent / "config.py"
if _config_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("openevolve._config_module", _config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["openevolve._config_module"] = config_module
    spec.loader.exec_module(config_module)

    # Export all classes from original config.py
    Config = config_module.Config
    load_config = config_module.load_config
    LLMModelConfig = config_module.LLMModelConfig
    LLMConfig = config_module.LLMConfig
    PromptConfig = config_module.PromptConfig
    DatabaseConfig = config_module.DatabaseConfig
    EvaluatorConfig = config_module.EvaluatorConfig
    EvolutionTraceConfig = config_module.EvolutionTraceConfig

# Import new runtime config modules
from .runtime_config import RuntimeConfigUpdater, ConfigUpdate, ConfigWatcherCallback, SimpleConfigValidator, ValidationResult
from .config_watcher import ConfigFileWatcher, MultiConfigWatcher
from .dynamic_strategy import DynamicStrategySwitcher, SystemMode, StateMigrator
from .adaptive_config import AdaptiveConfigurator, PerformanceMetrics, AutoTuner
from .resource_config import ResourceAwareConfigurator, ResourceInfo, ResourceLimits
from .config_metrics import ConfigurationMetrics, ConfigComparison, hash_config

__all__ = [
    # Legacy exports from original config.py
    "Config",
    "load_config",
    "LLMModelConfig",
    "LLMConfig",
    "PromptConfig",
    "DatabaseConfig",
    "EvaluatorConfig",
    "EvolutionTraceConfig",

    # Runtime config
    "RuntimeConfigUpdater",
    "ConfigUpdate",
    "ConfigWatcherCallback",
    "SimpleConfigValidator",
    "ValidationResult",

    # Config watcher
    "ConfigFileWatcher",
    "MultiConfigWatcher",

    # Dynamic strategy
    "DynamicStrategySwitcher",
    "SystemMode",
    "StateMigrator",

    # Adaptive config
    "AdaptiveConfigurator",
    "PerformanceMetrics",
    "AutoTuner",

    # Resource config
    "ResourceAwareConfigurator",
    "ResourceInfo",
    "ResourceLimits",

    # Metrics
    "ConfigurationMetrics",
    "ConfigComparison",
    "hash_config",
]
