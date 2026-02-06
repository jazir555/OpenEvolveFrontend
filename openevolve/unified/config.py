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
class MOConfig:
    """Configuration for Multi-Objective optimization."""
    optimization_strategy: str = "nsga2"  # nsga2, moead, spea2
    objectives: List[str] = field(default_factory=list)
    weights: Optional[List[float]] = None
    hypervolume_target: Optional[float] = None


@dataclass
class QDConfig:
    """Configuration for Quality-Diversity optimization."""
    archive_size: int = 100
    niche_grid_size: int = 10
    measure_dimensions: int = 2
    novelty_threshold: float = 0.5


@dataclass
class AdversarialConfig:
    """Configuration for Adversarial evolution."""
    adversary_population_size: int = 50
    adversary_mutation_rate: float = 0.2
    adversarial_rounds: int = 5
    stress_test_intensity: float = 0.7


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3
    api_key: Optional[str] = None


@dataclass
class EvaluatorConfig:
    """Configuration for evaluation."""
    evaluation_timeout: int = 60
    parallel_evaluations: int = 4
    early_stopping: bool = True
    validation_threshold: float = 0.8
    cache_results: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for database storage."""
    backend: str = "sqlite"  # sqlite, postgresql, mongodb
    connection_string: Optional[str] = None
    checkpoint_interval: int = 10
    max_checkpoints: int = 100
    compress_checkpoints: bool = True


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
    mo: MOConfig = field(default_factory=MOConfig)
    qd: QDConfig = field(default_factory=QDConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
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
    'MOConfig',
    'QDConfig',
    'AdversarialConfig',
    'LLMConfig',
    'EvaluatorConfig',
    'DatabaseConfig',
    'UnifiedEvolutionConfig',
]
