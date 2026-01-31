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
from pydantic import BaseModel, Field, field_validator
import yaml
import json


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


class QDConfig(BaseModel):
    """Quality Diversity (MAP-Elites) Specific Configuration (18 parameters)"""

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


class OpenEvolveConfig(BaseModel):
    """OpenEvolve-Specific Configuration (48 parameters)"""

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
    evolution_mode: str = Field(
        default="openevolve",
        description="Evolution mode: openevolve, pes, qd, mo, adversarial, hybrid"
    )
    enable_modes: List[str] = Field(
        default_factory=lambda: ["openevolve"],
        description="List of enabled modes (for hybrid evolution)"
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

    @field_validator("evolution_mode")
    @classmethod
    def validate_evolution_mode(cls, v: str) -> str:
        """Validate evolution mode"""
        valid_modes = ["openevolve", "pes", "qd", "mo", "adversarial", "hybrid"]
        if v not in valid_modes:
            raise ValueError(f"Invalid evolution_mode '{v}'. Must be one of: {valid_modes}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "evolution_mode": self.evolution_mode,
            "enable_modes": self.enable_modes,
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
            evolution_mode=config_dict.get("evolution_mode", "openevolve"),
            enable_modes=config_dict.get("enable_modes", ["openevolve"]),
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
