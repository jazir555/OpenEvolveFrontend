"""
OpenEvolve Unified Package

This package provides unified APIs for the OpenEvolve system.
"""

import sys
import os

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

# Import config classes for backward compatibility with core-projects
from .config import (
    MOConfig,
    QDConfig,
    AdversarialConfig,
    LLMConfig,
    EvaluatorConfig,
    DatabaseConfig,
)

# Inject glue layer's config into sys.modules so core-project uses it
# This ensures compatibility when core-projects/openevolve/openevolve/unified/config
# is imported via relative imports
try:
    # Import the glue layer's config module
    from . import config as glue_config_module

    # Calculate the core-project's config module path
    import openevolve
    core_base = os.path.dirname(openevolve.__file__)
    core_unified_config_path = os.path.join(core_base, 'core-projects', 'openevolve', 'openevolve', 'unified', 'config')

    # We can't directly override relative imports, but we can ensure the glue layer
    # is imported first and available in sys.modules
    sys.modules.setdefault('openevolve.unified.config', glue_config_module)
except Exception:
    pass

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
    # Config classes for core-project compatibility
    'MOConfig',
    'QDConfig',
    'AdversarialConfig',
    'LLMConfig',
    'EvaluatorConfig',
    'DatabaseConfig',
]
