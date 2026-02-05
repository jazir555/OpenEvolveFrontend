"""Configuration for PES Enhanced - non-invasive enhancement layer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CostOptimizationConfig:
    """Cost optimization settings - extracted from LoongFlow best practices."""
    
    # Budget limits
    max_cost_usd: float = 10.0
    max_tokens: int = 100000
    max_time_seconds: int = 1800
    
    # Token pricing (customizable per provider)
    prompt_token_price: float = 0.00001  # GPT-4o: $0.01 per 1K tokens
    completion_token_price: float = 0.00003  # GPT-4o: $0.03 per 1K tokens
    
    # Budget allocation
    planning_budget_pct: float = 0.05
    evolution_budget_pct: float = 0.85
    verification_budget_pct: float = 0.10
    
    # Alert thresholds
    warning_threshold: float = 0.70
    critical_threshold: float = 0.90
    
    # Cost-quality tradeoff
    use_cheap_models_for_execution: bool = True
    cheap_model: str = "gpt-3.5-turbo"
    expensive_model: str = "gpt-4o"


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration - addresses OpenEvolve gap."""
    
    enabled: bool = True
    patience: int = 5
    min_improvement: float = 0.01
    improvement_window: int = 10
    
    # Convergence detection
    convergence_threshold: float = 0.95
    plateau_threshold: float = 0.001
    diversity_window: int = 20
    
    # Time-based stopping
    max_duration_ms: int = 300000  # 5 minutes default
    
    # Evaluation budget stopping
    max_evaluations: int = 10000


@dataclass
class PlanningConfig:
    """Planning phase configuration - adds LoongFlow-style planning."""
    
    enabled: bool = True
    planning_iterations: int = 1
    
    # Strategy selection
    auto_select_strategy: bool = True
    consider_cost: bool = True
    consider_complexity: bool = True
    
    # Resource estimation
    estimate_evaluations: bool = True
    estimate_cost: bool = True
    estimate_duration: bool = True


@dataclass
class SummarizationConfig:
    """Summarization configuration - extracts learning from runs."""
    
    enabled: bool = True
    extract_patterns: bool = True
    analyze_success_factors: bool = True
    analyze_failure_modes: bool = True
    
    # Metrics to track
    track_efficiency: bool = True
    track_convergence: bool = True
    track_cost_breakdown: bool = True


@dataclass
class PESEnhancedConfig:
    """Main configuration for PES Enhanced layer."""
    
    # Sub-configs
    cost: CostOptimizationConfig = field(default_factory=CostOptimizationConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    
    # Enhancement toggles (all default to False to preserve existing behavior)
    enable_cost_optimization: bool = False
    enable_early_stopping: bool = False
    enable_planning: bool = False
    enable_summarization: bool = False
    enable_adaptive_parameters: bool = False
    
    # Pass-through to existing implementation
    preserve_existing_behavior: bool = True
    fallback_on_error: bool = True
    
    @classmethod
    def enable_all(cls) -> "PESEnhancedConfig":
        """Create config with all enhancements enabled."""
        config = cls()
        config.enable_cost_optimization = True
        config.enable_early_stopping = True
        config.enable_planning = True
        config.enable_summarization = True
        config.enable_adaptive_parameters = True
        return config
    
    @classmethod
    def cost_aware(cls, max_cost_usd: float = 5.0) -> "PESEnhancedConfig":
        """Create config focused on cost optimization."""
        config = cls()
        config.enable_cost_optimization = True
        config.enable_early_stopping = True
        config.cost.max_cost_usd = max_cost_usd
        return config
