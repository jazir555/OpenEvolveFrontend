"""
OpenEvolve Unified Package

This package provides unified APIs for the OpenEvolve system.
"""

# Import all symbols from unified_evolution_api
from .unified_evolution_api import (
    UnifiedEvolutionAPI,
    EvolutionResult,
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    ProgressUpdate,
    StrategyUsed,
    create_unified_api,
    evolve,
    quick_evolve,
    evolve_no_gauntlet,
    evolve_batch,
)

# Export symbols
__all__ = [
    'UnifiedEvolutionAPI',
    'EvolutionResult',
    'UnifiedEvolutionConfig',
    'EvolutionMode',
    'DomainType',
    'PESConfig',
    'ProgressUpdate',
    'StrategyUsed',
    'create_unified_api',
    'evolve',
    'quick_evolve',
    'evolve_no_gauntlet',
    'evolve_batch',
]
