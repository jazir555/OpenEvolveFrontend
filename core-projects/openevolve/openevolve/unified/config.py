"""
Unified Configuration Schema for All Evolutionary Modes

This module defines the complete unified configuration system that combines:
- OpenEvolve's 272+ parameters
- LoongFlow PES's ~50 parameters
- New unified evolutionary parameters

Total Parameters Documented: 322+
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
import json
import logging

logger = logging.getLogger(__name__)


class EvolutionMode(str, Enum):
    """Supported evolution modes."""

    PES = "pes"
    QD = "qd"
    MO = "mo"
    ADVERSARIAL = "adversarial"
    STANDARD = "standard"
    HYBRID = "hybrid"
    OPENEVOLVE = "openevolve"
    AUTO = "auto"


class DomainType(str, Enum):
    """Problem domains."""

    GENERAL = "general"
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB = "web"
    WEB_DESIGN = "web_design"
    LEGAL = "legal"
    MANUFACTURING = "manufacturing"
    MATH = "math"
    ML = "ml"


class CommonConfig(BaseModel):
    """Configuration common to ALL evolutionary modes"""

    # === Core Evolution Parameters (11 parameters) ===
    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum number of evolution iterations to run"
    )
    random_seed: Optional[int] = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (None = random)"
    )
    checkpoint_interval: int = Field(
        default=50,
        ge=1,
        description="Save checkpoints every N iterations"
    )

    # === Logging Configuration (6 parameters) ===
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_dir: Optional[str] = Field(
        default=None,
        description="Custom directory for logs (default: workspace/logs)"
    )
    log_to_console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )
    log_rotation: str = Field(
        default="H",
        description="Log rotation: S=seconds, M=minutes, H=hours, D=days"
    )
    log_backup_count: int = Field(
        default=0,
        ge=0,
        description="Number of backup logs to keep (0 = unlimited)"
    )

    # === Workspace Configuration (3 parameters) ===
    workspace_path: str = Field(
        default="./evolve_run_output",
        description="Root directory for all outputs and artifacts"
    )
    task_name: str = Field(
        default="evolution_task",
        description="Name of the evolution task (used for logging/filing)"
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Detailed description of the evolution task"
    )

    # === Concurrency Configuration (2 parameters) ===
    concurrency: int = Field(
        default=5,
        ge=1,
        description="Number of concurrent evaluations to run"
    )
    timeout: int = Field(
        default=300,
        ge=1,
        description="Default timeout in seconds for operations"
    )


class LLMModelConfig(BaseModel):
    """Configuration for a single LLM model in ensemble"""

    model_config = {"protected_namespaces": ()}

    # === Identity (3 parameters) ===
    name: str = Field(
        ...,
        description="Model name (e.g., 'gpt-4o', 'gemini-2.0-flash')"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for this model in ensemble sampling"
    )

    # === API Configuration (3 parameters) ===
    api_base: Optional[str] = Field(
        default=None,
        description="API base URL (defaults to provider's standard endpoint)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    provider: Optional[str] = Field(
        default=None,
        description="Model provider: openai, azure, anthropic, google, etc."
    )

    # === Generation Parameters (5 parameters) ===
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)"
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        ge=1,
        description="Maximum tokens to generate"
    )
    context_length: int = Field(
        default=65536,
        ge=1,
        description="Model's maximum context window size"
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for models that support it (low/medium/high)"
    )

    # === Request Parameters (3 parameters) ===
    timeout: Optional[int] = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds"
    )
    retries: Optional[int] = Field(
        default=3,
        ge=0,
        description="Number of retries for failed requests"
    )
    retry_delay: Optional[int] = Field(
        default=5,
        ge=0,
        description="Delay between retries in seconds"
    )

    # === System Message (1 parameter) ===
    system_message: Optional[str] = Field(
        default=None,
        description="System message/instruction for the model"
    )


class LLMConfig(BaseModel):
    """Configuration for LLM ensemble models"""

    # === Evolution Models (10 parameters) ===
    models: List[LLMModelConfig] = Field(
        default_factory=lambda: [LLMModelConfig(name="gpt-4o", weight=1.0)],
        description="List of models for evolution generation"
    )

    # === Evaluator Models (10 parameters) ===
    evaluator_models: List[LLMModelConfig] = Field(
        default_factory=list,
        description="List of models for evaluation/feedback (defaults to evolution models)"
    )

    # === Default API Configuration (3 parameters) ===
    default_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Default API base URL"
    )
    default_api_key: Optional[str] = Field(
        default=None,
        description="Default API key (can override with env var)"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for all models"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Global temperature override"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Global top-p override"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Global max tokens override"
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description="Global timeout override (seconds)"
    )
    retries: int = Field(
        default=3,
        ge=0,
        description="Global retries for LLM calls"
    )
    retry_delay: int = Field(
        default=5,
        ge=0,
        description="Global retry delay (seconds)"
    )
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for sampling"
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for supported models"
    )
    plan_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for PES planning phase"
    )
    summary_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for PES summarization phase"
    )

    @property
    def api_base(self) -> str:
        return self.default_api_base

    @api_base.setter
    def api_base(self, value: str) -> None:
        self.default_api_base = value

    @property
    def api_key(self) -> Optional[str]:
        return self.default_api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self.default_api_key = value


class DatabaseConfig(BaseModel):
    """Configuration for evolutionary database/memory"""

    # === Storage Configuration (5 parameters) ===
    storage_type: str = Field(
        default="in_memory",
        description="Storage backend: in_memory, redis, file, database"
    )
    db_path: Optional[str] = Field(
        default=None,
        description="Path for file-based storage (None = in-memory)"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for distributed storage"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path for checkpoints and outputs"
    )
    checkpoint_interval: int = Field(
        default=50,
        ge=1,
        description="Save checkpoint every N iterations"
    )

    # === Population Parameters (5 parameters) ===
    population_size: int = Field(
        default=1000,
        ge=10,
        description="Maximum population size per island"
    )
    elite_archive_size: int = Field(
        default=100,
        ge=1,
        description="Size of elite archive for best solutions"
    )
    num_islands: int = Field(
        default=5,
        ge=1,
        description="Number of islands for parallel evolution"
    )
    use_sampling_weight: bool = Field(
        default=True,
        description="Use weighted sampling based on fitness"
    )
    sampling_weight_power: float = Field(
        default=1.0,
        ge=0.0,
        description="Power for sampling weight (higher = more exploitation)"
    )

    # === Island Migration Parameters (2 parameters) ===
    migration_interval: int = Field(
        default=50,
        ge=1,
        description="Migrate solutions between islands every N iterations"
    )
    migration_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of population to migrate"
    )

    # === Selection Parameters (4 parameters) ===
    elite_selection_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of elite solutions to select"
    )
    exploration_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability of random exploration (vs exploitation)"
    )
    exploitation_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of exploitation vs random selection"
    )
    boltzmann_temperature: float = Field(
        default=1.0,
        ge=0.0,
        description="Temperature for Boltzmann sampling (higher = more random)"
    )

    # === MAP-Elites Feature Map Parameters (5 parameters) ===
    feature_dimensions: List[str] = Field(
        default_factory=lambda: ["complexity", "diversity"],
        description="Feature dimensions for MAP-Elites grid (built-in: complexity, diversity, score)"
    )
    feature_bins: Union[int, Dict[str, int]] = Field(
        default=10,
        description="Number of bins per feature dimension (int or dict per dimension)"
    )
    feature_scaling_method: str = Field(
        default="minmax",
        description="Feature scaling: minmax, standard, robust, none"
    )
    diversity_reference_size: int = Field(
        default=20,
        ge=1,
        description="Size of reference set for diversity calculation"
    )
    diversity_metric: str = Field(
        default="edit_distance",
        description="Diversity metric: edit_distance, feature_based, semantic"
    )

    # === Logging (2 parameters) ===
    log_prompts: bool = Field(
        default=True,
        description="Log all prompts and responses to database"
    )
    enable_artifacts: bool = Field(
        default=True,
        description="Store evaluation artifacts in database"
    )


class EvaluatorConfig(BaseModel):
    """Configuration for solution evaluation"""

    # === General Settings (5 parameters) ===
    timeout: int = Field(
        default=300,
        ge=1,
        description="Maximum evaluation time in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries for failed evaluations"
    )
    early_stopping_patience: Optional[int] = Field(
        default=None,
        ge=1,
        description="Early stopping patience for evaluator loops"
    )
    convergence_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum improvement to reset evaluator patience"
    )
    early_stopping_metric: str = Field(
        default="fitness",
        description="Metric to track for evaluator early stopping"
    )
    evaluate_code: str = Field(
        default="",
        description="Python code string for evaluation"
    )
    evolve_target: Optional[str] = Field(
        default=None,
        description="Specific target or goal for evolution"
    )
    workspace_path: Optional[str] = Field(
        default=None,
        description="Path for evaluation workspace"
    )

    # === Resource Limits (2 parameters) ===
    memory_limit_mb: Optional[int] = Field(
        default=None,
        ge=1,
        description="Memory limit in MB per evaluation"
    )
    cpu_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="CPU limit per evaluation (0.0-1.0 for fraction, >1.0 for cores)"
    )

    # === Evaluation Strategies (4 parameters) ===
    cascade_evaluation: bool = Field(
        default=True,
        description="Use cascade evaluation to filter bad solutions early"
    )
    cascade_thresholds: List[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9],
        description="Thresholds for cascade evaluation stages"
    )
    parallel_evaluations: int = Field(
        default=4,
        ge=1,
        description="Number of parallel evaluations"
    )
    distributed: bool = Field(
        default=False,
        description="Enable distributed evaluation across workers"
    )

    # === LLM Feedback (2 parameters) ===
    use_llm_feedback: bool = Field(
        default=False,
        description="Use LLM to provide code quality feedback"
    )
    llm_feedback_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight of LLM feedback in final score"
    )

    # === Artifact Handling (2 parameters) ===
    enable_artifacts: bool = Field(
        default=True,
        description="Enable artifact storage from evaluations"
    )
    max_artifact_storage: int = Field(
        default=100 * 1024 * 1024,
        ge=0,
        description="Maximum artifact storage per program in bytes"
    )


class PESConfig(BaseModel):
    """LoongFlow PES (Plan-Evolve-Summarize) Specific Configuration (22 parameters)"""

    # Enable PES mode
    enabled: bool = Field(default=False, description="Enable PES mode")

    # === Planning Configuration (6 parameters) ===
    enable_planning: bool = Field(
        default=True,
        description="Enable planning phase before evolution"
    )
    planner_type: str = Field(
        default="evolve_planner",
        description="Type of planner: evolve_planner, react_planner, chat_planner"
    )
    planning_iterations: int = Field(
        default=1,
        ge=1,
        description="Number of planning iterations"
    )
    planning_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for planning LLM calls"
    )
    use_refinement: bool = Field(
        default=True,
        description="Enable plan refinement based on feedback"
    )
    max_refinement_iterations: int = Field(
        default=3,
        ge=0,
        description="Maximum refinement iterations for plans"
    )

    # === Execution Configuration (5 parameters) ===
    executor_type: str = Field(
        default="evolve_executor",
        description="Type of executor: evolve_executor, react_executor, chat_executor"
    )
    execution_mode: str = Field(
        default="sequential",
        description="Execution mode: sequential, parallel, adaptive"
    )
    enable_code_execution: bool = Field(
        default=True,
        description="Enable actual code execution during evolution"
    )
    execution_timeout: int = Field(
        default=300,
        ge=1,
        description="Timeout per code execution in seconds"
    )
    sandbox_mode: bool = Field(
        default=True,
        description="Run code in sandboxed environment"
    )

    # === Summarization Configuration (5 parameters) ===
    enable_summary: bool = Field(
        default=True,
        description="Enable summarization after evolution"
    )
    summary_type: str = Field(
        default="evolve_summary",
        description="Type of summarizer: evolve_summary, react_summary, chat_summary"
    )
    summary_detail_level: str = Field(
        default="medium",
        description="Summary detail: low, medium, high"
    )
    include_traceback: bool = Field(
        default=False,
        description="Include traceback in summaries"
    )
    summary_max_length: int = Field(
        default=2000,
        ge=100,
        description="Maximum summary length in characters"
    )

    # === Memory Configuration (3 parameters) ===
    enable_memory: bool = Field(
        default=True,
        description="Enable long-term memory for evolution"
    )
    memory_type: str = Field(
        default="in_memory",
        description="Memory type: in_memory, redis, database"
    )
    memory_compression: bool = Field(
        default=True,
        description="Enable memory compression for large histories"
    )
    memory_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top solutions to retrieve from memory"
    )

    # === Context Management (3 parameters) ===
    context_window: int = Field(
        default=10000,
        ge=1,
        description="Context window size for PES operations"
    )
    context_compression_threshold: int = Field(
        default=5000,
        ge=1,
        description="Token threshold for context compression"
    )
    use_context_pruning: bool = Field(
        default=True,
        description="Enable intelligent context pruning"
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_aliases(cls, values: Any) -> Any:
        """Map legacy alias fields into current PESConfig fields."""
        if not isinstance(values, dict):
            return values
        data = dict(values)
        if "plan_iterations" in data and "planning_iterations" not in data:
            data["planning_iterations"] = data["plan_iterations"]
        if "max_rounds" in data and "max_refinement_iterations" not in data:
            data["max_refinement_iterations"] = data["max_rounds"]
        if "use_memory" in data and "enable_memory" not in data:
            data["enable_memory"] = data["use_memory"]
        return data

    @property
    def plan_iterations(self) -> int:
        return self.planning_iterations

    @plan_iterations.setter
    def plan_iterations(self, value: int) -> None:
        self.planning_iterations = value

    @property
    def max_rounds(self) -> int:
        return self.max_refinement_iterations

    @max_rounds.setter
    def max_rounds(self, value: int) -> None:
        self.max_refinement_iterations = value

    @property
    def use_memory(self) -> bool:
        return self.enable_memory

    @use_memory.setter
    def use_memory(self, value: bool) -> None:
        self.enable_memory = value


class QDConfig(BaseModel):
    """Quality Diversity (MAP-Elites) Specific Configuration (18 parameters)"""

    @model_validator(mode="before")
    @classmethod
    def _apply_enabled_alias(cls, values: Any) -> Any:
        """Support legacy aliases for enable_map_elites, archive_size, feature_dimensions."""
        if not isinstance(values, dict):
            return values
        data = dict(values)
        if "enabled" in data and "enable_map_elites" not in data:
            data["enable_map_elites"] = data["enabled"]
        if "archive_size" in data and "archive_size_limit" not in data:
            data["archive_size_limit"] = data["archive_size"]
        if "feature_dimensions" in data and "grid_dimensions" not in data:
            data["grid_dimensions"] = data["feature_dimensions"]
        return data

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "enabled":
            name = "enable_map_elites"
        elif name == "archive_size":
            name = "archive_size_limit"
        elif name == "feature_dimensions":
            name = "grid_dimensions"
        super().__setattr__(name, value)

    @property
    def enabled(self) -> bool:
        return self.enable_map_elites

    @property
    def archive_size(self) -> int:
        return self.archive_size_limit if self.archive_size_limit is not None else 1000

    @property
    def feature_dimensions(self) -> List[str]:
        return self.grid_dimensions

    # === Grid Configuration (6 parameters) ===
    enable_map_elites: bool = Field(
        default=True,
        description="Enable MAP-Elites algorithm"
    )
    grid_resolution: int = Field(
        default=10,
        ge=2,
        description="Resolution of MAP-Elites grid (bins per dimension)"
    )
    grid_dimensions: List[str] = Field(
        default_factory=lambda: ["complexity", "diversity"],
        description="Feature dimensions for grid axes"
    )
    adaptive_grid: bool = Field(
        default=False,
        description="Enable adaptive grid resolution based on solution distribution"
    )
    grid_update_interval: int = Field(
        default=100,
        ge=1,
        description="Update adaptive grid every N iterations"
    )

    # === Archive Configuration (5 parameters) ===
    archive_type: str = Field(
        default="map_elites",
        description="Archive type: map_elites, cvt_map_elites, submarine_map_elites"
    )
    archive_size_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum archive size (None = unlimited)"
    )
    archive_elitism: bool = Field(
        default=True,
        description="Use elitism (keep best in each cell)"
    )
    use_novelty: bool = Field(
        default=False,
        description="Use novelty search in addition to quality"
    )
    novelty_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Novelty threshold for novelty search"
    )

    # === Feature Calculation (4 parameters) ===
    feature_extraction_method: str = Field(
        default="auto",
        description="Feature extraction: auto, manual, learned"
    )
    feature_normalization: str = Field(
        default="minmax",
        description="Feature normalization: minmax, standard, robust, none"
    )
    use_feature_learning: bool = Field(
        default=False,
        description="Enable learning of feature representations"
    )
    feature_learning_rate: float = Field(
        default=0.001,
        ge=0.0,
        description="Learning rate for feature learning"
    )

    # === QD-Specific Parameters (3 parameters) ===
    cvt_samples: int = Field(
        default=10000,
        ge=1,
        description="Number of samples for CVT initialization"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for QD variation operators"
    )
    use_niching: bool = Field(
        default=True,
        description="Use niching to maintain diverse solutions"
    )
    niche_radius: float = Field(
        default=0.1,
        ge=0.0,
        description="Radius for niche identification"
    )


class MOConfig(BaseModel):
    """Multi-Objective Optimization Configuration (15 parameters)"""

    # Enable MO mode
    enabled: bool = Field(default=False, description="Enable multi-objective optimization")

    # === Objective Configuration (4 parameters) ===
    objectives: List[str] = Field(
        default_factory=lambda: ["score"],
        description="List of objective names to optimize"
    )
    objective_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for each objective (None = equal weight)"
    )
    optimization_direction: Dict[str, str] = Field(
        default_factory=lambda: {"score": "maximize"},
        description="Direction for each objective: maximize or minimize"
    )
    use_pareto: bool = Field(
        default=True,
        description="Use Pareto dominance for multi-objective selection"
    )

    # === Pareto Front Configuration (4 parameters) ===
    pareto_archive_size: int = Field(
        default=100,
        ge=1,
        description="Maximum size of Pareto front archive"
    )
    pareto_pruning_method: str = Field(
        default="crowding_distance",
        description="Pruning method: crowding_distance, hypervolume, epsilon_indicator"
    )
    crowding_distance_metric: str = Field(
        default="euclidean",
        description="Distance metric for crowding: euclidean, manhattan, cosine"
    )
    use_hypervolume: bool = Field(
        default=False,
        description="Use hypervolume indicator for archive quality"
    )

    # === Selection Configuration (4 parameters) ===
    selection_method: str = Field(
        default="nsga2",
        description="Multi-objective selection: nsga2, nsga3, spea2, moead"
    )
    tournament_size: int = Field(
        default=2,
        ge=2,
        description="Tournament size for selection"
    )
    crossover_rate: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Crossover rate for recombination"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for variation"
    )

    # === Scalarization (3 parameters) ===
    use_scalarization: bool = Field(
        default=False,
        description="Use scalarization for single-objective conversion"
    )
    scalarization_method: str = Field(
        default="weighted_sum",
        description="Scalarization method: weighted_sum, tchebycheff, achievement"
    )
    reference_point: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reference point for scalarization methods"
    )


class AdversarialConfig(BaseModel):
    """Adversarial Evolution Configuration (12 parameters) ==="""

    # Enable adversarial mode
    enabled: bool = Field(default=False, description="Enable adversarial co-evolution")

    # === Adversarial Setup (4 parameters) ===
    enable_adversarial: bool = Field(
        default=False,
        description="Enable adversarial evolution"
    )
    num_adversaries: int = Field(
        default=2,
        ge=2,
        description="Number of adversarial populations"
    )
    adversarial_mode: str = Field(
        default="generator_discriminator",
        description="Mode: generator_discriminator, predator_prey, competitive_cooperative"
    )
    adversarial_rounds: int = Field(
        default=20,
        ge=1,
        description="Number of adversarial rounds per iteration"
    )

    # === Generator/Discriminator (3 parameters) ===
    generator_objective: str = Field(
        default="fool_discriminator",
        description="Generator's objective: fool_discriminator, maximize_fitness, diversity"
    )
    discriminator_objective: str = Field(
        default="detect_fake",
        description="Discriminator's objective: detect_fake, classify_quality, novelty"
    )
    balance_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between generator and discriminator updates"
    )

    # === Coevolution Dynamics (5 parameters) ===
    use_coevolution: bool = Field(
        default=True,
        description="Enable co-evolutionary dynamics"
    )
    coevolution_frequency: int = Field(
        default=5,
        ge=1,
        description="Update adversaries every N iterations"
    )
    fitness_sharing: bool = Field(
        default=True,
        description="Use fitness sharing to maintain diversity"
    )
    fitness_sharing_sigma: float = Field(
        default=0.1,
        ge=0.0,
        description="Sigma for fitness sharing"
    )
    use_arms_race: bool = Field(
        default=False,
        description="Enable arms race dynamics (progressive difficulty)"
    )
    robustness_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum robustness score required to pass adversarial checks"
    )


class OpenEvolveConfig(BaseModel):
    """OpenEvolve-Specific Configuration (48 parameters)"""

    # === Legacy Core Parameters (for backward compatibility) ===
    max_iterations: int = Field(
        default=10000,
        ge=1,
        description="Maximum number of iterations (legacy OpenEvolve config)"
    )
    checkpoint_interval: int = Field(
        default=100,
        ge=1,
        description="Checkpoint interval (legacy OpenEvolve config)"
    )
    random_seed: Optional[int] = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (legacy OpenEvolve config)"
    )

    # === Code Evolution (6 parameters) ===
    diff_based_evolution: bool = Field(
        default=True,
        description="Use diff-based evolution (vs full rewrites)"
    )
    max_code_length: int = Field(
        default=10000,
        ge=100,
        description="Maximum allowed code length in characters"
    )
    language: str = Field(
        default="python",
        description="Programming language for evolution"
    )
    file_suffix: str = Field(
        default=".py",
        description="File suffix for generated programs"
    )
    enable_simplification: bool = Field(
        default=True,
        description="Enable automatic code simplification"
    )
    suggest_simplification_after_chars: int = Field(
        default=500,
        ge=100,
        description="Suggest simplification if code exceeds this length"
    )

    # === Prompt Configuration (8 parameters) ===
    template_dir: Optional[str] = Field(
        default=None,
        description="Directory for custom prompt templates"
    )
    system_message: str = Field(
        default="You are an expert coder helping to improve programs through evolution.",
        description="System message for evolution LLM"
    )
    evaluator_system_message: str = Field(
        default="You are an expert code reviewer.",
        description="System message for evaluator LLM"
    )
    num_top_programs: int = Field(
        default=3,
        ge=0,
        description="Number of top programs to include in prompt"
    )
    num_diverse_programs: int = Field(
        default=2,
        ge=0,
        description="Number of diverse programs to include in prompt"
    )
    use_template_stochasticity: bool = Field(
        default=True,
        description="Use random variations in prompt templates"
    )
    template_variations: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Alternative phrasings for template components"
    )
    include_artifacts: bool = Field(
        default=True,
        description="Include execution artifacts in prompts"
    )

    # === Artifact Handling (5 parameters) ===
    max_artifact_bytes: int = Field(
        default=20 * 1024,
        ge=0,
        description="Maximum artifact size to include in prompt (20KB)"
    )
    artifact_security_filter: bool = Field(
        default=True,
        description="Apply security filtering to artifacts"
    )
    artifact_size_threshold: int = Field(
        default=32 * 1024,
        ge=0,
        description="Size threshold for artifact storage (32KB)"
    )
    cleanup_old_artifacts: bool = Field(
        default=True,
        description="Automatically clean up old artifacts"
    )
    artifact_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days to retain artifacts before cleanup"
    )

    # === Program Labeling (3 parameters) ===
    include_changes_under_chars: int = Field(
        default=100,
        ge=0,
        description="Include change descriptions if under this length"
    )
    concise_implementation_max_lines: int = Field(
        default=10,
        ge=1,
        description="Label as 'concise' if lines <= this"
    )
    comprehensive_implementation_min_lines: int = Field(
        default=50,
        ge=1,
        description="Label as 'comprehensive' if lines >= this"
    )

    # === Early Stopping (4 parameters) ===
    early_stopping_patience: Optional[int] = Field(
        default=None,
        ge=1,
        description="Stop after N iterations without improvement (None = disabled)"
    )
    convergence_threshold: float = Field(
        default=0.001,
        ge=0.0,
        description="Minimum improvement to reset patience counter"
    )
    early_stopping_metric: str = Field(
        default="combined_score",
        description="Metric to track for early stopping"
    )
    target_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Target score to stop evolution (None = continue)"
    )

    # === Meta-Prompting (3 parameters) ===
    use_meta_prompting: bool = Field(
        default=False,
        description="Enable meta-prompting (prompt about prompting)"
    )
    meta_prompt_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for meta-prompt suggestions"
    )
    meta_prompt_interval: int = Field(
        default=10,
        ge=1,
        description="Apply meta-prompting every N iterations"
    )

    # === Evolution Trace (6 parameters) ===
    evolution_trace_enabled: bool = Field(
        default=False,
        description="Enable detailed evolution trace logging"
    )
    evolution_trace_format: str = Field(
        default="jsonl",
        description="Trace format: jsonl, json, hdf5"
    )
    evolution_trace_include_code: bool = Field(
        default=False,
        description="Include full program code in traces"
    )
    evolution_trace_include_prompts: bool = Field(
        default=True,
        description="Include prompts and LLM responses in traces"
    )
    evolution_trace_buffer_size: int = Field(
        default=10,
        ge=1,
        description="Buffer size before writing traces"
    )
    evolution_trace_compress: bool = Field(
        default=False,
        description="Compress trace output"
    )

    # === Advanced Features (13 parameters) ===
    use_embedding: bool = Field(
        default=False,
        description="Use embeddings for semantic search"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model to use"
    )
    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        description="Dimension of embedding vectors"
    )
    enable_novelty_search: bool = Field(
        default=False,
        description="Enable novelty search alongside quality optimization"
    )
    novelty_k_nearest: int = Field(
        default=10,
        ge=1,
        description="K for novelty calculation"
    )
    enable_quality_diversity: bool = Field(
        default=True,
        description="Enable quality-diversity optimization"
    )
    use_crossover: bool = Field(
        default=False,
        description="Enable crossover between solutions"
    )
    crossover_method: str = Field(
        default="single_point",
        description="Crossover method: single_point, two_point, uniform"
    )
    use_mutation: bool = Field(
        default=True,
        description="Enable mutation operations"
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation probability"
    )
    use_selection_pressure: bool = Field(
        default=True,
        description="Apply selection pressure"
    )
    selection_pressure_method: str = Field(
        default="tournament",
        description="Selection method: tournament, roulette, rank"
    )
    tournament_size: int = Field(
        default=3,
        ge=2,
        description="Tournament size for selection"
    )

    def to_unified(self) -> "UnifiedEvolutionConfig":
        """Convert legacy OpenEvolve config to unified config."""
        return UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD if self.random_seed is not None else EvolutionMode.STANDARD,
            max_iterations=self.max_iterations,
            checkpoint_interval=self.checkpoint_interval,
            random_seed=self.random_seed,
            diff_based_evolution=self.diff_based_evolution,
            max_code_length=self.max_code_length,
            language=self.language,
            early_stopping_patience=self.early_stopping_patience,
            convergence_threshold=self.convergence_threshold,
            early_stopping_metric=self.early_stopping_metric,
            qd=QDConfig(enabled=True),
            openevolve=self,
        )


class UnifiedEvolutionConfig(BaseModel):
    """
    Unified Configuration for All Evolutionary Modes

    This is the master configuration that combines:
    - Common parameters (shared by all modes): 29 parameters
    - LLM configuration: 26 parameters
    - Database configuration: 35 parameters
    - Evaluator configuration: 17 parameters
    - PES-specific (LoongFlow): 22 parameters
    - Quality Diversity: 18 parameters
    - Multi-Objective: 15 parameters
    - Adversarial: 12 parameters
    - OpenEvolve-specific: 48 parameters

    Total: 322+ documented parameters
    """

    # === Core Configuration ===
    common: CommonConfig = Field(
        default_factory=CommonConfig,
        description="Common configuration shared by all modes"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM model configuration"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database/memory configuration"
    )
    evaluator: EvaluatorConfig = Field(
        default_factory=EvaluatorConfig,
        description="Evaluator configuration"
    )

    # === Mode Selection ===
    domain: DomainType = Field(
        default=DomainType.GENERAL,
        description="Problem domain (for domain-specific presets)"
    )
    evolution_mode: EvolutionMode = Field(
        default=EvolutionMode.OPENEVOLVE,
        description="Evolution mode: openevolve, pes, qd, mo, adversarial, hybrid"
    )
    enable_modes: List[Union[EvolutionMode, str]] = Field(
        default_factory=lambda: [EvolutionMode.OPENEVOLVE],
        description="List of enabled modes (for hybrid evolution)"
    )

    # === LoongFlow Optional Control ===
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
        description="Require LoongFlow to be available (no fallback)."
    )

    # === Mode-Specific Configurations ===
    pes: Optional[PESConfig] = Field(
        default=None,
        description="PES (Plan-Evolve-Summarize) configuration"
    )
    qd: Optional[QDConfig] = Field(
        default=None,
        description="Quality Diversity (MAP-Elites) configuration"
    )
    mo: Optional[MOConfig] = Field(
        default=None,
        description="Multi-Objective optimization configuration"
    )
    adversarial: Optional[AdversarialConfig] = Field(
        default=None,
        description="Adversarial evolution configuration"
    )
    openevolve: Optional[OpenEvolveConfig] = Field(
        default=None,
        description="OpenEvolve-specific configuration"
    )

    # === Metadata ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the evolution run"
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        """Map legacy top-level fields into nested config sections."""
        if not isinstance(data, dict):
            return data

        data = dict(data)

        # Common overrides
        common_overrides: Dict[str, Any] = {}
        for key in ("max_iterations", "checkpoint_interval", "random_seed", "log_level", "log_dir"):
            if key in data:
                common_overrides[key] = data.pop(key)
        if common_overrides:
            common_section = data.get("common") or {}
            if isinstance(common_section, CommonConfig):
                common_section = common_section.model_dump()
            common_section.update(common_overrides)
            data["common"] = common_section

        # Database overrides
        db_map = {
            "population_size": "population_size",
            "num_islands": "num_islands",
            "migration_interval": "migration_interval",
            "migration_rate": "migration_rate",
            "archive_size": "elite_archive_size",
        }
        db_overrides: Dict[str, Any] = {}
        for key, mapped in db_map.items():
            if key in data:
                db_overrides[mapped] = data.pop(key)
        if db_overrides:
            db_section = data.get("database") or {}
            if isinstance(db_section, DatabaseConfig):
                db_section = db_section.model_dump()
            db_section.update(db_overrides)
            data["database"] = db_section

        # Mutation rate overrides (apply to QD/MO if present)
        if "mutation_rate" in data:
            mutation_rate = data.pop("mutation_rate")

            qd_section = data.get("qd") or {}
            if isinstance(qd_section, QDConfig):
                qd_section = qd_section.model_dump()
            qd_section["mutation_rate"] = mutation_rate
            data["qd"] = qd_section

            mo_section = data.get("mo") or {}
            if isinstance(mo_section, MOConfig):
                mo_section = mo_section.model_dump()
            mo_section["mutation_rate"] = mutation_rate
            data["mo"] = mo_section

        return data

    @field_validator("evolution_mode")
    @classmethod
    def validate_evolution_mode(cls, v: Any) -> EvolutionMode:
        """Validate evolution mode."""
        if isinstance(v, EvolutionMode):
            return v
        if isinstance(v, str):
            try:
                return EvolutionMode(v)
            except ValueError as exc:
                valid_modes = [mode.value for mode in EvolutionMode]
                raise ValueError(
                    f"Invalid evolution_mode '{v}'. Must be one of: {valid_modes}"
                ) from exc
        raise TypeError("evolution_mode must be a string or EvolutionMode")

    @model_validator(mode="after")
    def validate_loongflow_settings(self) -> "UnifiedEvolutionConfig":
        """Validate LoongFlow settings are consistent."""
        if self.require_loongflow and not self.enable_loongflow:
            raise ValueError(
                "require_loongflow=True but enable_loongflow=False is contradictory. "
                "Either set enable_loongflow=True or require_loongflow=False"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "domain": self.domain.value if isinstance(self.domain, DomainType) else self.domain,
            "evolution_mode": self.evolution_mode.value if isinstance(self.evolution_mode, EvolutionMode) else self.evolution_mode,
            "enable_modes": [
                mode.value if isinstance(mode, EvolutionMode) else mode for mode in self.enable_modes
            ],
            "enable_loongflow": self.enable_loongflow,
            "loongflow_fallback_enabled": self.loongflow_fallback_enabled,
            "require_loongflow": self.require_loongflow,
            "common": self.common.model_dump(),
            "llm": self.llm.model_dump(),
            "database": self.database.model_dump(),
            "evaluator": self.evaluator.model_dump(),
            "pes": self.pes.model_dump() if self.pes else None,
            "qd": self.qd.model_dump() if self.qd else None,
            "mo": self.mo.model_dump() if self.mo else None,
            "adversarial": self.adversarial.model_dump() if self.adversarial else None,
            "openevolve": self.openevolve.model_dump() if self.openevolve else None,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Convert configuration to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UnifiedEvolutionConfig":
        """Load configuration from dictionary"""
        # Extract mode-specific configs
        common = CommonConfig(**config_dict.get("common", {}))
        llm = LLMConfig(**config_dict.get("llm", {}))
        database = DatabaseConfig(**config_dict.get("database", {}))
        evaluator = EvaluatorConfig(**config_dict.get("evaluator", {}))

        pes = PESConfig(**config_dict["pes"]) if config_dict.get("pes") else None
        qd = QDConfig(**config_dict["qd"]) if config_dict.get("qd") else None
        mo = MOConfig(**config_dict["mo"]) if config_dict.get("mo") else None
        adversarial = AdversarialConfig(**config_dict["adversarial"]) if config_dict.get("adversarial") else None
        openevolve = OpenEvolveConfig(**config_dict["openevolve"]) if config_dict.get("openevolve") else None

        return cls(
            domain=config_dict.get("domain", DomainType.GENERAL),
            evolution_mode=config_dict.get("evolution_mode", EvolutionMode.OPENEVOLVE),
            enable_modes=config_dict.get("enable_modes", [EvolutionMode.OPENEVOLVE]),
            enable_loongflow=config_dict.get("enable_loongflow", True),
            loongflow_fallback_enabled=config_dict.get("loongflow_fallback_enabled", True),
            require_loongflow=config_dict.get("require_loongflow", False),
            common=common,
            llm=llm,
            database=database,
            evaluator=evaluator,
            pes=pes,
            qd=qd,
            mo=mo,
            adversarial=adversarial,
            openevolve=openevolve,
            metadata=config_dict.get("metadata", {})
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "UnifiedEvolutionConfig":
        """Load configuration from YAML string"""
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, path: Union[str, Path]) -> "UnifiedEvolutionConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            yaml_str = f.read()
        return cls.from_yaml(yaml_str)

    @classmethod
    def from_json(cls, json_str: str) -> "UnifiedEvolutionConfig":
        """Load configuration from JSON string"""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "UnifiedEvolutionConfig":
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            f.write(self.to_json())

    # ------------------------------------------------------------------
    # LoongFlow helper methods
    # ------------------------------------------------------------------

    def is_loongflow_enabled(self) -> bool:
        """Check if LoongFlow is enabled in config."""
        return self.enable_loongflow

    def should_use_loongflow(self) -> bool:
        """
        Determine if LoongFlow should be used based on availability.

        Returns True if LoongFlow is enabled and available.
        Raises RuntimeError if require_loongflow=True but unavailable.
        """
        if not self.enable_loongflow:
            return False

        available = self._check_loongflow_availability()
        if self.require_loongflow:
            if not available:
                raise RuntimeError(
                    "require_loongflow=True but LoongFlow is not available. "
                    "Please install LoongFlow or set require_loongflow=False."
                )
            return True

        if self.loongflow_fallback_enabled and not available:
            logger.warning(
                "LoongFlow is enabled but not available. Falling back to OpenEvolve modes. "
                "Set loongflow_fallback_enabled=False to require LoongFlow."
            )

        return available

    def _check_loongflow_availability(self) -> bool:
        """Check if LoongFlow package is available."""
        try:
            import loongflow  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def openevolve_only(**kwargs) -> "UnifiedEvolutionConfig":
        """Create an OpenEvolve-only configuration (LoongFlow disabled)."""
        return UnifiedEvolutionConfig(
            enable_loongflow=False,
            loongflow_fallback_enabled=False,
            require_loongflow=False,
            **kwargs,
        )

    @staticmethod
    def loongflow_required(**kwargs) -> "UnifiedEvolutionConfig":
        """Create a configuration that strictly requires LoongFlow."""
        return UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            loongflow_fallback_enabled=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Legacy compatibility properties
    # ------------------------------------------------------------------

    @property
    def max_iterations(self) -> int:
        return self.common.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self.common.max_iterations = value

    @property
    def checkpoint_interval(self) -> int:
        return self.common.checkpoint_interval

    @checkpoint_interval.setter
    def checkpoint_interval(self, value: int) -> None:
        self.common.checkpoint_interval = value

    @property
    def random_seed(self) -> Optional[int]:
        return self.common.random_seed

    @random_seed.setter
    def random_seed(self, value: Optional[int]) -> None:
        self.common.random_seed = value

    @property
    def population_size(self) -> int:
        return self.database.population_size

    @population_size.setter
    def population_size(self, value: int) -> None:
        self.database.population_size = value

    @property
    def num_islands(self) -> int:
        return self.database.num_islands

    @num_islands.setter
    def num_islands(self, value: int) -> None:
        self.database.num_islands = value

    @property
    def migration_interval(self) -> int:
        return self.database.migration_interval

    @migration_interval.setter
    def migration_interval(self, value: int) -> None:
        self.database.migration_interval = value

    @property
    def migration_rate(self) -> float:
        return self.database.migration_rate

    @migration_rate.setter
    def migration_rate(self, value: float) -> None:
        self.database.migration_rate = value

    @property
    def archive_size(self) -> int:
        return self.database.elite_archive_size

    @archive_size.setter
    def archive_size(self, value: int) -> None:
        self.database.elite_archive_size = value

    @property
    def mutation_rate(self) -> float:
        if self.qd is not None:
            return self.qd.mutation_rate
        if self.mo is not None:
            return self.mo.mutation_rate
        return 0.0

    @mutation_rate.setter
    def mutation_rate(self, value: float) -> None:
        if self.qd is None:
            self.qd = QDConfig()
        self.qd.mutation_rate = value
        if self.mo is not None:
            self.mo.mutation_rate = value

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }
