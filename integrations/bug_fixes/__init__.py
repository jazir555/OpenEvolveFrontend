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
- HephaestusConfigOverride: Legacy fixes for hephaestus config (deprecated)
- EvolutionConfigurationWrapper: Fixes duplicate dataclass fields
- AdversarialImportResolver: Fixes circular import issues
- ConfigProvider: Provides configuration overrides

Author: OpenEvolve Frontend Team
Date: 2026-01-21
"""

from .crewai_config_fix import CrewAIConfigOverride
from .hephaestus_config_fix import HephaestusConfigOverride
from .evolution_wrapper import EvolutionConfigurationWrapper
from .adversarial_import_resolver import (
    AdversarialImportResolver,
    RedTeamStrategyProxy,
    get_red_team_strategy,
    get_default_strategy,
)
from .config_provider import ConfigProvider

__all__ = [
    'CrewAIConfigOverride',
    'HephaestusConfigOverride',  # Legacy - kept for backwards compatibility
    'EvolutionConfigurationWrapper',
    'AdversarialImportResolver',
    'RedTeamStrategyProxy',
    'get_red_team_strategy',
    'get_default_strategy',
    'ConfigProvider',
]
