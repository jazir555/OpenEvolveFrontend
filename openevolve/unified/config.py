"""
Configuration module for OpenEvolve Unified API.

Provides configuration classes for evolution operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class EvolutionMode(Enum):
    """Evolution operation modes."""
    STANDARD = "standard"
    PES = "pes"
    QUALITY_DIVERSITY = "quality_diversity"
    MULTI_OBJECTIVE = "multi_objective"
    ADVERSARIAL = "adversarial"


class DomainType(Enum):
    """Supported domains for evolution."""
    GENERAL = "general"
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB = "web"


class SystemMode(Enum):
    """System operation modes."""
    STANDALONE = "standalone"
    INTEGRATED = "integrated"
    DISTRIBUTED = "distributed"


@dataclass
class PESConfig:
    """Configuration for PES (Plan-Execute-Summarize) mode."""
    enabled: bool = True
    enable_planning: bool = True
    enable_memory: bool = True
    max_rounds: int = 3


@dataclass
class UnifiedEvolutionConfig:
    """Configuration for unified evolution."""
    domain: DomainType = DomainType.GENERAL
    evolution_mode: EvolutionMode = EvolutionMode.STANDARD
    max_iterations: int = 50
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: int = 5
    pes: PESConfig = field(default_factory=PESConfig)
    run_gauntlet: bool = True
    store_knowledge: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)


# Re-export from unified_evolution_api for convenience
try:
    from .unified_evolution_api import (
        EvolutionResult,
        ProgressUpdate,
        StrategyUsed,
    )
except ImportError:
    # Fallback if circular import
    pass

__all__ = [
    'EvolutionMode',
    'DomainType',
    'SystemMode',
    'PESConfig',
    'UnifiedEvolutionConfig',
]
