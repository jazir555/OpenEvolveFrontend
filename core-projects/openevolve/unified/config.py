"""
Unified Configuration Schema
Supports all evolutionary modes: OpenEvolve + LoongFlow PES

Total Parameters: ~90+ (consolidated from OpenEvolve's 51 + LoongFlow's 20+)

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvolutionMode(str, Enum):
    """Supported evolution modes"""
    PES = "pes"                    # Plan-Execute-Summarize (LoongFlow)
    QD = "qd"                      # Quality-Diversity MAP-Elites (OpenEvolve)
    MO = "mo"                      # Multi-Objective optimization
    ADVERSARIAL = "adversarial"    # Adversarial co-evolution
    STANDARD = "standard"          # Traditional evolutionary algorithm
    AUTO = "auto"                  # Auto-select based on configuration


class DomainType(str, Enum):
    """Problem domains"""
    GENERAL = "general"
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB = "web"
    MATH = "math"
    ML = "ml"


# ============================================================================
# SUB-CONFIGURATION CLASSES
# ============================================================================

class LLMModelConfig(BaseModel):
    """Configuration for a single LLM in the ensemble"""
    name: str = Field(description="Model name (e.g., 'gpt-4', 'claude-3-opus')")
    weight: float = Field(default=1.0, ge=0.0, description="Ensemble weight")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    api_key: Optional[str] = Field(default=None, description="API key (or use env var)")
    api_version: Optional[str] = Field(default=None, description="API version")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Override temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Override max tokens")
    provider: str = Field(default="openai", description="API provider: openai, anthropic, azure, etc.")


class LLMConfig(BaseModel):
    """
    LLM Configuration
    Supports both OpenEvolve mutation and LoongFlow PES planning
    """
    # Model ensemble
    models: List[LLMModelConfig] = Field(
        default=[],
        description="Models for mutation/execution (weighted ensemble)"
    )
    evaluator_models: List[LLMModelConfig] = Field(
        default=[],
        description="Models for evaluation/feedback (if using LLM evaluator)"
    )
    planner_models: List[LLMModelConfig] = Field(
        default=[],
        description="Models for PES planning phase (LoongFlow)"
    )
    summary_models: List[LLMModelConfig] = Field(
        default=[],
        description="Models for PES summary phase (LoongFlow)"
    )

    # Generation parameters (OpenEvolve)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Mutation creativity")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum output tokens")

    # Request parameters
    timeout: int = Field(default=60, ge=1, description="Request timeout (seconds)")
    retries: int = Field(default=3, ge=0, description="Number of retries")
    retry_delay: int = Field(default=5, ge=0, description="Delay between retries (seconds)")

    # Reproducibility
    random_seed: Optional[int] = Field(default=42, description="Random seed for LLM sampling")

    # Reasoning (for o1 models)
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort for o1 models")

    # PES-specific temperatures (LoongFlow)
    plan_temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for planning")
    summary_temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for summary")


class DatabaseConfig(BaseModel):
    """
    Database / Population Configuration
    Supports OpenEvolve MAP-Elites + LoongFlow evolutionary memory
    """
    # Population management (OpenEvolve)
    population_size: int = Field(default=1000, ge=1, le=100000, description="Total population size")
    archive_size: int = Field(default=100, ge=1, le=10000, description="Elite archive size")
    num_islands: int = Field(default=5, ge=1, le=50, description="Number of parallel islands")

    # Selection parameters (OpenEvolve)
    elite_selection_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of elites")
    exploration_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Exploration sampling ratio")
    exploitation_ratio: float = Field(default=0.7, ge=0.0, le=1.0, description="Exploitation sampling ratio")

    # MAP-Elites configuration (OpenEvolve QD)
    feature_dimensions: List[str] = Field(
        default=["complexity", "diversity"],
        description="Behavioral feature dimensions for MAP-Elites"
    )
    feature_bins: Union[int, Dict[str, int]] = Field(
        default=10,
        description="Bins per dimension (int or dict per dimension)"
    )

    # Island migration (OpenEvolve)
    migration_interval: int = Field(default=50, ge=1, description="Generations between migrations")
    migration_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Migration fraction")
    migration_topology: str = Field(
        default="ring",
        description="Migration topology: ring, fully_connected, random"
    )

    # Diversity (OpenEvolve)
    diversity_metric: str = Field(default="edit_distance", description="Diversity metric")
    diversity_reference_size: int = Field(default=20, ge=1, description="Reference set size for diversity")

    # LoongFlow evolutionary memory
    enable_memory: bool = Field(default=True, description="Enable PES evolutionary memory")
    memory_path: Optional[str] = Field(default=None, description="Path to memory database")
    exploration_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Base exploration rate")
    adaptive_exploration: bool = Field(default=True, description="Adaptive exploration (local optima detection)")

    # Logging
    log_prompts: bool = Field(default=True, description="Log prompts to database")
    log_artifacts: bool = Field(default=True, description="Log evaluation artifacts")


class EvaluatorConfig(BaseModel):
    """
    Evaluator Configuration
    Supports cascade evaluation, parallel evaluation, and gauntlet integration
    """
    # General
    timeout: int = Field(default=300, ge=1, description="Evaluation timeout (seconds)")
    max_retries: int = Field(default=3, ge=0, description="Max evaluation retries")

    # Cascade evaluation (OpenEvolve)
    cascade_evaluation: bool = Field(default=True, description="Enable multi-stage cascade")
    cascade_thresholds: List[float] = Field(
        default=[0.5, 0.75, 0.9],
        description="Score thresholds for each cascade stage"
    )

    # Parallel evaluation (OpenEvolve)
    parallel_evaluations: int = Field(default=4, ge=1, le=100, description="Parallel evaluations")
    parallel_batch_size: int = Field(default=10, ge=1, description="Batch size for parallel eval")

    # LLM feedback (OpenEvolve)
    use_llm_feedback: bool = Field(default=False, description="Use LLM for evaluation feedback")
    llm_feedback_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="LLM feedback weight")

    # Gauntlet integration
    enable_gauntlets: bool = Field(default=True, description="Run gauntlets on solutions")
    gauntlet_strictness: str = Field(
        default="standard",
        description="Gauntlet strictness: lenient, standard, strict"
    )
    gauntlet_id: Optional[str] = Field(default=None, description="Specific gauntlet to run")

    # Artifacts
    enable_artifacts: bool = Field(default=True, description="Enable artifact collection")
    max_artifact_storage: int = Field(
        default=100 * 1024 * 1024,
        ge=0,
        description="Max artifact storage (bytes)"
    )

    # Run generated programs in an isolated, resource-limited subprocess via
    # SecureCodeExecutor instead of in-process (defence against runaway code).
    secure_execution: bool = Field(
        default=False,
        description="Execute generated programs in a resource-limited subprocess"
    )

    # Early stopping (LoongFlow PES)
    early_stopping: bool = Field(default=True, description="Enable early stopping on improvement")
    early_stopping_patience: int = Field(default=5, ge=1, description="Attempts before giving up")
    early_stopping_threshold: float = Field(default=0.01, ge=0.0, description="Improvement threshold")


class PESConfig(BaseModel):
    """
    Plan-Execute-Summarize Configuration (LoongFlow)
    """
    # Enable PES mode
    enabled: bool = Field(default=False, description="Enable PES mode")

    # Planning phase
    enable_planning: bool = Field(default=True, description="Enable planning phase")
    max_plans: int = Field(default=1, ge=1, description="Number of plans to generate")
    plan_iterations: int = Field(default=1, ge=1, description="Planning iterations")

    # Execution phase
    max_rounds: int = Field(default=3, ge=1, description="Max execution rounds per iteration")
    parallel_candidates: int = Field(default=1, ge=1, le=10, description="Parallel candidates per round")

    # Summarization phase
    enable_summary: bool = Field(default=True, description="Enable summary/reflection phase")
    summary_iterations: int = Field(default=1, ge=1, description="Summary iterations")

    # Memory integration
    use_memory: bool = Field(default=True, description="Use evolutionary memory in planning")
    memory_top_k: int = Field(default=5, ge=1, description="Number of top solutions to retrieve")


class QDConfig(BaseModel):
    """
    Quality-Diversity Configuration (OpenEvolve MAP-Elites)
    """
    # Enable QD mode
    enabled: bool = Field(default=False, description="Enable QD mode")

    # MAP-Elites grid
    grid_resolution: int = Field(default=10, ge=2, le=100, description="MAP-Elites grid resolution")
    feature_dimensions: Optional[List[str]] = Field(
        default=None,
        description="Override feature dimensions for QD"
    )
    archive_size: int = Field(default=1000, ge=10, le=100000, description="MAP-Elites archive size")

    # QD variants
    use_cvt_map_elites: bool = Field(default=False, description="Use CVT-MAP-Elites (centroidal voronoi)")
    cvt_samples: int = Field(default=10000, ge=100, description="CVT samples for centroids")


class MOConfig(BaseModel):
    """
    Multi-Objective Optimization Configuration
    """
    # Enable MO mode
    enabled: bool = Field(default=False, description="Enable multi-objective optimization")

    # Objectives
    objectives: Optional[List[str]] = Field(
        default=None,
        description="List of objective names (e.g., ['return', 'risk', 'liquidity'])"
    )
    objective_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for weighted sum aggregation"
    )

    # Algorithm selection
    algorithm: str = Field(
        default="nsga2",
        description="MO algorithm: nsga2, spea2, moead, weighted_sum"
    )
    pareto_size: int = Field(default=100, ge=10, le=1000, description="Pareto front size")

    # Dominance
    use_constraint_domination: bool = Field(default=True, description="Use constrained domination")


class AdversarialConfig(BaseModel):
    """
    Adversarial Co-evolution Configuration
    """
    # Enable adversarial mode
    enabled: bool = Field(default=False, description="Enable adversarial co-evolution")

    # Rounds
    adversarial_rounds: int = Field(default=20, ge=1, le=1000, description="Number of adversarial rounds")

    # Teams
    red_team_models: List[str] = Field(
        default=["gpt-4", "claude-3-opus"],
        description="Models to use for red team (attack generation)"
    )
    blue_team_models: List[str] = Field(
        default=["gpt-4", "claude-3-opus"],
        description="Models to use for blue team (defense)"
    )

    # Evaluation
    robustness_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Robustness pass threshold")


# ============================================================================
# MAIN UNIFIED CONFIGURATION
# ============================================================================

class UnifiedEvolutionConfig(BaseModel):
    """
    Unified configuration for all evolutionary modes

    Maps OpenEvolve's 51 parameters + LoongFlow's 20+ parameters
    into a single coherent configuration system (~90+ total parameters)

    Usage:
        # PES mode (LoongFlow)
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True),
            # ... other params
        )

        # QD mode (OpenEvolve MAP-Elites)
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(enabled=True),
            # ... other params
        )
    """

    # =========================================================================
    # COMMON PARAMETERS (all modes)
    # =========================================================================

    # Evolution control
    max_iterations: int = Field(default=10000, ge=1, le=1000000, description="Maximum iterations")
    checkpoint_interval: int = Field(default=100, ge=1, description="Checkpoint frequency")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    # Time limits
    time_limit_seconds: Optional[int] = Field(default=None, ge=1, description="Max execution time (seconds)")
    target_fitness: Optional[float] = Field(default=None, description="Stop when fitness reaches target")

    # Domain
    domain: DomainType = Field(default=DomainType.GENERAL, description="Problem domain")

    # Code/program settings
    language: Optional[str] = Field(default=None, description="Programming language (auto-detect if None)")
    max_code_length: int = Field(default=10000, ge=100, description="Maximum code length")
    diff_based_evolution: bool = Field(default=True, description="Use diff-based evolution (vs full rewrite)")

    # Convergence
    early_stopping_patience: Optional[int] = Field(default=None, ge=1, description="Early stopping patience")
    convergence_threshold: float = Field(default=0.001, ge=0.0, description="Convergence threshold")
    early_stopping_metric: str = Field(default="combined_score", description="Metric for early stopping")

    # =========================================================================
    # MODE SELECTION
    # =========================================================================

    evolution_mode: EvolutionMode = Field(
        default=EvolutionMode.AUTO,
        description="Evolution mode (auto-select if AUTO)"
    )

    # =========================================================================
    # SUB-CONFIGURATIONS
    # =========================================================================

    # Core subsystems
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database/population config")
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig, description="Evaluator config")

    # Mode-specific configs
    pes: PESConfig = Field(default_factory=PESConfig, description="PES mode config (LoongFlow)")
    qd: QDConfig = Field(default_factory=QDConfig, description="QD mode config (OpenEvolve)")
    mo: MOConfig = Field(default_factory=MOConfig, description="Multi-objective config")
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig, description="Adversarial config")

    # =========================================================================
    # KNOWLEDGE ENGINE INTEGRATION
    # =========================================================================

    enable_knowledge_extraction: bool = Field(
        default=True,
        description="Extract learning to Knowledge Engine"
    )
    enable_strategy_learning: bool = Field(
        default=True,
        description="Learn strategy recommendations from history"
    )
    knowledge_engine_path: Optional[str] = Field(
        default=None,
        description="Path to Knowledge Engine instance"
    )

    # =========================================================================
    # LOONGFLOW OPTIONAL CONTROL
    # =========================================================================

    enable_loongflow: bool = Field(
        default=True,
        description="Enable LoongFlow PES system. If False, only OpenEvolve modes will be used."
    )

    loongflow_fallback_enabled: bool = Field(
        default=True,
        description="Allow fallback to OpenEvolve if LoongFlow is unavailable or fails."
    )

    require_loongflow: bool = Field(
        default=False,
        description="Require LoongFlow to be available. If True and LoongFlow is unavailable, raise an error instead of falling back to OpenEvolve."
    )

    # =========================================================================
    # OUTPUT & LOGGING
    # =========================================================================

    output_dir: str = Field(default="./evolution_output", description="Output directory")
    verbose: bool = Field(default=True, description="Verbose logging")
    trace_enabled: bool = Field(default=False, description="Enable evolution trace")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @field_validator('evolution_mode', mode='before')
    @classmethod
    def auto_select_mode(cls, v, info):
        """Auto-select mode based on configuration if AUTO"""
        if v != EvolutionMode.AUTO:
            return v

        # Auto-detect based on enabled configs
        values = info.data if hasattr(info, 'data') else {}
        if values.get('pes', PESConfig()).enabled:
            return EvolutionMode.PES
        elif values.get('qd', QDConfig()).enabled:
            return EvolutionMode.QD
        elif values.get('mo', MOConfig()).enabled:
            return EvolutionMode.MO
        elif values.get('adversarial', AdversarialConfig()).enabled:
            return EvolutionMode.ADVERSARIAL
        else:
            return EvolutionMode.STANDARD

    @field_validator('database')
    @classmethod
    def validate_database(cls, v):
        """Ensure selection ratios sum to <= 1.0"""
        total = v.elite_selection_ratio + v.exploration_ratio + v.exploitation_ratio
        if total > 1.0:
            raise ValueError(f"Selection ratios sum to {total} > 1.0")
        return v

    @model_validator(mode='after')
    def validate_loongflow_settings(self):
        """Validate LoongFlow settings are consistent"""
        enable_loongflow = self.enable_loongflow
        require_loongflow = self.require_loongflow

        # If require_loongflow is True, fallback must be disabled
        if require_loongflow and not enable_loongflow:
            raise ValueError(
                "require_loongflow=True but enable_loongflow=False is contradictory. "
                "Either set enable_loongflow=True or require_loongflow=False"
            )

        # If require_loongflow and LoongFlow not available, error will be raised later
        # This is intentional - user wants strict requirement

        return self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def is_loongflow_enabled(self) -> bool:
        """Check if LoongFlow should be used"""
        return self.enable_loongflow

    def should_use_loongflow(self) -> bool:
        """
        Check if LoongFlow should be used considering availability
        Returns True if LoongFlow is enabled and available
        """
        if not self.enable_loongflow:
            return False

        if self.require_loongflow:
            # Will check availability and raise error if not found
            available = self._check_loongflow_availability()
            if not available:
                raise RuntimeError(
                    "require_loongflow=True but LoongFlow is not available. "
                    "Please install LoongFlow or set require_loongflow=False."
                )
            return True

        if self.loongflow_fallback_enabled:
            # Check if LoongFlow is available
            available = self._check_loongflow_availability()
            if not available:
                logger.warning(
                    "LoongFlow is enabled but not available. Falling back to OpenEvolve modes. "
                    "Set loongflow_fallback_enabled=False to require LoongFlow."
                )
            return available

        return self._check_loongflow_availability()

    def _check_loongflow_availability(self) -> bool:
        """Check if LoongFlow package is available"""
        try:
            import loongflow
            return True
        except ImportError:
            return False

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    @staticmethod
    def openevolve_only(**kwargs) -> "UnifiedEvolutionConfig":
        """
        Create OpenEvolve-only configuration (LoongFlow disabled)

        This is a convenience method for users who want to use only OpenEvolve.

        Example:
            config = UnifiedEvolutionConfig.openevolve_only(
                max_iterations=100,
                domain="finance"
            )
        """
        return UnifiedEvolutionConfig(
            enable_loongflow=False,
            loongflow_fallback_enabled=False,
            require_loongflow=False,
            **kwargs
        )

    @staticmethod
    def loongflow_required(**kwargs) -> "UnifiedEvolutionConfig":
        """
        Create configuration that requires LoongFlow (no fallback)

        This is for users who want to ensure LoongFlow is used.

        Example:
            config = UnifiedEvolutionConfig.loongflow_required(
                domain="finance"
            )
        """
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            loongflow_fallback_enabled=False,
            **kwargs
        )

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        use_enum_values = True


# ============================================================================
# LEGACY CONFIGURATION CLASSES (for backward compatibility)
# ============================================================================

class OpenEvolveConfig(BaseModel):
    """
    Legacy OpenEvolve configuration wrapper
    Converts to UnifiedEvolutionConfig
    """
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    random_seed: Optional[int] = 42
    diff_based_evolution: bool = True
    max_code_length: int = 10000
    language: Optional[str] = None

    # Early stopping
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    def to_unified(self) -> UnifiedEvolutionConfig:
        """Convert to unified config"""
        return UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD if self.random_seed else EvolutionMode.STANDARD,
            max_iterations=self.max_iterations,
            checkpoint_interval=self.checkpoint_interval,
            random_seed=self.random_seed,
            diff_based_evolution=self.diff_based_evolution,
            max_code_length=self.max_code_length,
            language=self.language,
            early_stopping_patience=self.early_stopping_patience,
            convergence_threshold=self.convergence_threshold,
            early_stopping_metric=self.early_stopping_metric,
            qd=QDConfig(enabled=True)  # OpenEvolve defaults to QD mode
        )
