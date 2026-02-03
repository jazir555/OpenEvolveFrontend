"""
Unified Evolutionary Optimization System
Supports OpenEvolve + LoongFlow PES integration

This module provides a unified configuration schema that supports:
- PES (Plan-Execute-Summarize) mode from LoongFlow
- QD (Quality-Diversity) mode from OpenEvolve (MAP-Elites)
- MO (Multi-Objective) optimization
- Adversarial co-evolution
- Standard evolutionary algorithms

Author: AI Architecture Team
Date: 2026-01-30
"""

# Export main classes
from .config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig,
    LLMConfig,
    DatabaseConfig,
    EvaluatorConfig
)
from .config_mapper import ConfigMapper
from .config_validator import ConfigValidator

# Export unified evolution API
from .unified_evolution_api import (
    UnifiedEvolutionAPI,
    evolve,
    evolve_openevolve_only,
    evolve_with_loongflow,
    quick_evolve,
    evolve_no_gauntlet,
    evolve_batch,
    EvolutionResult,
    SystemMode,
    ProgressUpdate
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    'UnifiedEvolutionConfig',
    'EvolutionMode',
    'DomainType',
    'PESConfig',
    'QDConfig',
    'MOConfig',
    'AdversarialConfig',
    'OpenEvolveConfig',
    'LLMConfig',
    'DatabaseConfig',
    'EvaluatorConfig',
    'ConfigMapper',
    'ConfigValidator',

    # Unified Evolution API
    'UnifiedEvolutionAPI',
    'evolve',
    'evolve_openevolve_only',
    'evolve_with_loongflow',
    'quick_evolve',
    'evolve_no_gauntlet',
    'evolve_batch',
    'EvolutionResult',
    'SystemMode',
    'ProgressUpdate'
]
