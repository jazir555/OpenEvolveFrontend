"""
Unified configuration system for OpenEvolve + LoongFlow PES integration.

Combines OpenEvolve's 272+ parameters with PES-specific configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto


class StrategySelectionMode(Enum):
    """How to select evolution strategy."""
    AUTO = auto()      # Automatically select based on problem
    MANUAL = auto()    # User-specified strategy
    PES_ONLY = auto()  # Always use PES-enhanced mode


class CostModel(Enum):
    """Cost model for budget estimation."""
    TOKEN_BASED = auto()   # Based on LLM tokens
    TIME_BASED = auto()    # Based on compute time
    HYBRID = auto()        # Combined model


class AdaptationTrigger(Enum):
    """What triggers parameter adaptation."""
    PLATEAU = auto()       # Fitness plateau
    BUDGET = auto()        # Budget threshold
    SCHEDULED = auto()     # Regular intervals


class KnowledgeMode(Enum):
    """How to use knowledge base."""
    ACTIVE = auto()        # Actively query and update
    PASSIVE = auto()       # Only query, don't update
    DISABLED = auto()      # Don't use knowledge


@dataclass
class BudgetConfig:
    """Budget configuration for cost-aware evolution."""
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    max_api_calls: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    
    # Budget allocation ratios (must sum to <= 0.9, leaving 10% contingency)
    planning_budget_ratio: float = 0.05
    evolution_budget_ratio: float = 0.85
    verification_budget_ratio: float = 0.10
    contingency_reserve_ratio: float = 0.10
    
    def __post_init__(self):
        total = (self.planning_budget_ratio + self.evolution_budget_ratio + 
                self.verification_budget_ratio + self.contingency_reserve_ratio)
        if total > 1.0:
            raise ValueError(f"Budget ratios sum to {total}, must be <= 1.0")


@dataclass
class PESConfig:
    """PES-specific configuration."""
    # Enablement
    enable_pes_planning: bool = True
    enable_cost_optimization: bool = True
    enable_adaptive_execution: bool = True
    enable_summarization: bool = True
    
    # Planning
    planning_depth: int = 3
    use_historical_patterns: bool = True
    
    # Cost model
    cost_model: CostModel = CostModel.TOKEN_BASED
    
    # Adaptation
    adaptation_interval: int = 10
    adaptation_trigger: AdaptationTrigger = AdaptationTrigger.PLATEAU
    
    # Early stopping
    pes_early_stopping: bool = True
    convergence_sensitivity: float = 0.01
    min_improvement_window: int = 5
    
    # Knowledge
    knowledge_integration_mode: KnowledgeMode = KnowledgeMode.ACTIVE
    knowledge_similarity_threshold: float = 0.7
    
    # Directed search
    enable_directed_mutation: bool = True
    directed_mutation_weight: float = 0.7
    
    # Summarization
    extract_patterns: bool = True
    update_knowledge_base: bool = True


@dataclass
class UnifiedEvolutionConfig:
    """
    Unified configuration combining OpenEvolve and PES parameters.
    
    This class integrates:
    - OpenEvolve's 272+ evolution parameters
    - LoongFlow PES planning parameters
    - Cost optimization parameters
    - Strategy selection parameters
    """
    
    # ============================================================
    # OpenEvolve Core Parameters (subset of key parameters)
    # ============================================================
    
    # Core evolution
    evolution_mode: str = "standard"  # standard, qd, mo, adversarial, pes_enhanced
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: bool = True
    diversity_maintenance: bool = True
    
    # Model configuration
    model_id: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    
    # Quality Diversity (MAP-Elites)
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    archive_size: int = 100
    novelty_threshold: float = 0.1
    
    # Multi-objective (NSGA-II)
    objectives: List[str] = field(default_factory=list)
    pareto_front_size: int = 50
    
    # Evaluation
    evaluation_budget: int = 10000
    cascade_evaluation: bool = True
    parallel_evaluations: int = 4
    
    # Resources
    cost_limit_usd: float = 10.0
    token_limit: int = 100000
    api_call_limit: int = 1000
    max_time: int = 1800
    
    # Language agnostic
    language: str = "python"
    file_suffix: str = ".py"
    
    # ============================================================
    # PES Integration Parameters
    # ============================================================
    
    strategy_selection_mode: StrategySelectionMode = StrategySelectionMode.AUTO
    pes_config: PESConfig = field(default_factory=PESConfig)
    budget_config: BudgetConfig = field(default_factory=BudgetConfig)
    
    # ============================================================
    # Integration Parameters
    # ============================================================
    
    enable_pes_callbacks: bool = True
    enable_budget_monitoring: bool = True
    fallback_to_standard: bool = True
    
    def to_openevolve_config(self) -> Dict[str, Any]:
        """Convert to OpenEvolve-compatible configuration dictionary."""
        return {
            "evolution_mode": self.evolution_mode,
            "max_iterations": self.max_iterations,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism": self.elitism,
            "diversity_maintenance": self.diversity_maintenance,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "feature_dimensions": self.feature_dimensions,
            "feature_bins": self.feature_bins,
            "archive_size": self.archive_size,
            "objectives": self.objectives,
            "evaluation_budget": self.evaluation_budget,
            "cost_limit_usd": self.cost_limit_usd,
            "language": self.language,
        }
    
    @classmethod
    def from_openevolve_config(
        cls,
        config: Dict[str, Any],
        enable_pes: bool = True
    ) -> "UnifiedEvolutionConfig":
        """Create from existing OpenEvolve configuration."""
        unified = cls()
        
        # Copy OpenEvolve parameters
        for key, value in config.items():
            if hasattr(unified, key):
                setattr(unified, key, value)
        
        # Enable PES
        if enable_pes:
            unified.pes_config.enable_pes_planning = True
            if unified.evolution_mode == "standard":
                unified.evolution_mode = "pes_enhanced"
        
        return unified
