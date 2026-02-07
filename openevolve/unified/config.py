"""
Configuration module for OpenEvolve Unified API.

Provides configuration classes for evolution operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Import enums from unified_evolution_api to ensure compatibility
from .unified_evolution_api import EvolutionMode, DomainType, SystemMode


@dataclass
class PESConfig:
    """Configuration for PES (Plan-Execute-Summarize) mode."""
    enabled: bool = True
    enable_planning: bool = True
    enable_memory: bool = True
    use_memory: bool = True  # Alias for enable_memory
    max_rounds: int = 3
    max_plans: int = 5  # Added for core-projects compatibility
    plan_iterations: int = 3
    parallel_candidates: int = 1
    enable_summary: bool = True
    memory_top_k: int = 5
    # Additional parameters for backward compatibility
    planning_temperature: float = 0.7
    planning_iterations: int = 3
    max_refinement_iterations: int = 3


@dataclass
class MOConfig:
    """Configuration for Multi-Objective optimization."""
    enabled: bool = False
    optimization_strategy: str = "nsga2"  # nsga2, moead, spea2
    algorithm: str = "nsga2"  # Alias for optimization_strategy
    objectives: List[str] = field(default_factory=list)
    weights: Optional[List[float]] = None
    hypervolume_target: Optional[float] = None
    objective_weights: Optional[Dict[str, float]] = None
    optimization_direction: Dict[str, str] = field(default_factory=lambda: {"score": "maximize"})
    use_pareto: bool = True
    pareto_archive_size: int = 100
    pareto_size: int = 100  # Alias for pareto_archive_size
    selection_method: str = "nsga2"
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1


@dataclass
class QDConfig:
    """Configuration for Quality-Diversity optimization."""
    enabled: bool = False
    archive_size: int = 100
    niche_grid_size: int = 10
    grid_resolution: int = 10  # Alias for niche_grid_size
    measure_dimensions: int = 2
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    novelty_threshold: float = 0.5


@dataclass
class AdversarialConfig:
    """Configuration for Adversarial evolution."""
    enabled: bool = False
    enable_adversarial: bool = False  # Alias for enabled (core-projects uses this)
    adversary_population_size: int = 50
    adversary_mutation_rate: float = 0.2
    adversarial_rounds: int = 5
    stress_test_intensity: float = 0.7
    # Additional parameters for backward compatibility
    red_team_models: Optional[List[str]] = None
    blue_team_models: Optional[List[str]] = None
    num_adversaries: int = 2
    adversarial_mode: str = "generator_discriminator"
    robustness_threshold: float = 0.8


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    max_retries: int = 3
    api_key: Optional[str] = None
    # Additional parameters for backward compatibility
    plan_temperature: float = 0.7
    summary_temperature: float = 0.7
    retries: int = 3
    top_p: float = 0.95


@dataclass
class EvaluatorConfig:
    """Configuration for evaluation."""
    evaluation_timeout: int = 60
    parallel_evaluations: int = 4
    early_stopping: bool = True
    validation_threshold: float = 0.8
    cache_results: bool = True
    # Additional parameters for backward compatibility
    timeout: int = 60
    max_retries: int = 3
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.01
    convergence_threshold: float = 0.01
    early_stopping_metric: str = "fitness"


@dataclass
class DatabaseConfig:
    """Configuration for database storage."""
    backend: str = "sqlite"  # sqlite, postgresql, mongodb
    connection_string: Optional[str] = None
    checkpoint_interval: int = 10
    max_checkpoints: int = 100
    compress_checkpoints: bool = True
    # Additional parameters for backward compatibility
    enable_memory: bool = True
    adaptive_exploration: bool = True
    population_size: int = 1000
    elite_archive_size: int = 100
    num_islands: int = 5
    archive_size: int = 100  # Alias for elite_archive_size
    diversity_metric: str = "edit_distance"  # Diversity metric for QD


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
