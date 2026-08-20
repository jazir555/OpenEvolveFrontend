"""
Unified Configuration System for OpenEvolve

This module provides a unified configuration schema that works across:
- OpenEvolve's 272+ parameters (QD, MO, Adversarial, etc.)
- LoongFlow PES's ~50 parameters
- New unified parameters

The unified config provides:
1. Single source of truth for all evolutionary modes
2. Type-safe validation with Pydantic
3. Easy serialization/deserialization
4. Domain-specific presets
5. Automatic validation and conflict detection
"""

from .config import (
    UnifiedEvolutionConfig,
    CommonConfig,
    LLMConfig,
    LLMModelConfig,
    DatabaseConfig,
    EvaluatorConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig,
    EvolutionMode,
    DomainType,
)
from .config_mapper import ConfigMapper
from .config_validator import ConfigValidator
from .defaults import (
    get_finance_config,
    get_trading_config,
    get_scientific_config,
    get_engineering_config,
    get_pharmaceutical_config,
    get_web_design_config,
    get_domain_config,
    list_domains,
)

__all__ = [
    # Main config classes
    "UnifiedEvolutionConfig",
    "CommonConfig",
    "LLMConfig",
    "LLMModelConfig",
    "DatabaseConfig",
    "EvaluatorConfig",
    "PESConfig",
    "QDConfig",
    "MOConfig",
    "AdversarialConfig",
    "OpenEvolveConfig",
    "EvolutionMode",
    "DomainType",
    # Utilities
    "ConfigMapper",
    "ConfigValidator",
    # Domain presets
    "get_finance_config",
    "get_trading_config",
    "get_scientific_config",
    "get_engineering_config",
    "get_pharmaceutical_config",
    "get_web_design_config",
    "get_domain_config",
    "list_domains",
]
