"""
Bug Fix Adapters for Core Projects

This module contains glue code adapters that fix bugs in core projects
without modifying the core project files themselves.

All fixes follow the Anti-Corruption Layer pattern:
- Wrap core classes/functions
- Provide corrected behavior
- Do not modify core source code

Core Projects (DO NOT EDIT):
- crewai, openevolve, roma, bubblelab, datapizza, claudiomiro
- graphiti, global-chem, deep-research-agent, all kg/*, leanaide
- curie, PAMI, research-quest, ragbits, steer, ACE, uqsa

Bug Fixes:
- CrewAIConfigOverride: Fixes invalid paths in crewai config
- crewaiConfigOverride: Legacy fixes for crewai config (deprecated)
- EvolutionConfigurationWrapper: Fixes duplicate dataclass fields
- AdversarialImportResolver: Fixes circular import issues
- ConfigProvider: Provides configuration overrides

Author: OpenEvolve Frontend Team
Date: 2026-01-21
"""

from .crewai_config_fix import CrewAIConfigOverride
from .crewai_config_fix import crewaiConfigOverride
from .evolution_wrapper import EvolutionConfigurationWrapper
from .adversarial_import_resolver import (
    AdversarialImportResolver,
    RedTeamStrategyProxy,
    get_red_team_strategy,
    get_default_strategy,
)
# NOTE: config_provider.py does not exist in this package. The only
# ConfigProvider in the repo is engines/config/config_provider.py, which is an
# empty stub (`class ConfigProvider: pass`) and does NOT implement the
# get_env()/validate_config() API that test_fixes.py exercises. The import is
# therefore optional so the four working adapters above remain importable.
# Do not treat ConfigProvider as available until a real implementation lands.
try:
    from .config_provider import ConfigProvider
    _HAS_CONFIG_PROVIDER = True
except ImportError:  # pragma: no cover - module is currently missing
    _HAS_CONFIG_PROVIDER = False

__all__ = [
    'CrewAIConfigOverride',
    'crewaiConfigOverride',  # Legacy - kept for backwards compatibility
    'EvolutionConfigurationWrapper',
    'AdversarialImportResolver',
    'RedTeamStrategyProxy',
    'get_red_team_strategy',
    'get_default_strategy',
]

if _HAS_CONFIG_PROVIDER:
    __all__.append('ConfigProvider')