"""
Parameter Manager - Manages all 211 OpenEvolve parameters
Provides validation, presets, and persistence for OpenEvolve configuration
"""


import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from configuration_manager import config_manager # Import config_manager

class ParameterType(Enum):
    """Parameter data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SELECT = "select"


@dataclass
class Parameter:
    """Definition of a single parameter"""
    name: str
    type: ParameterType
    default: Any
    description: str
    category: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[str]] = None
    required: bool = False
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from parameter validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ParameterSchema:
    """Defines all 211 OpenEvolve parameters"""
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self._initialize_parameters()

    def _load_parameters_from_dict(self, parameters_data: Dict[str, Any]):
        """Load parameters from a dictionary structure."""
        for category, params in parameters_data.items():
            for param_name, param_details in params.items():
                param_type = ParameterType[param_details["type"].upper()]
                self._add_param(
                    name=param_name,
                    param_type=param_type,
                    default=param_details["default"],
                    description=param_details["description"],
                    category=category,
                    **{k: v for k, v in param_details.items() if k not in ["type", "default", "description"]}
                )
    
    def _initialize_parameters(self):
        """Initialize all 211 parameters organized by category"""
        
        # Category 1: Core Evolution Parameters (15)
        self._add_param("evolution_mode", ParameterType.SELECT, "standard", 
                       "Evolution strategy", "core_evolution",
                       options=["standard", "quality_diversity", "multi_objective", "adversarial", "problem_decomposition"])
        self._add_param("max_iterations", ParameterType.INTEGER, 10,
                       "Maximum evolution iterations", "core_evolution", min_value=1, max_value=1000)
        self._add_param("population_size", ParameterType.INTEGER, 20,
                       "Population size per generation", "core_evolution", min_value=1, max_value=1000)
        self._add_param("temperature", ParameterType.FLOAT, 0.7,
                       "LLM sampling temperature", "core_evolution", min_value=0.0, max_value=2.0)
        self._add_param("max_tokens", ParameterType.INTEGER, 2048,
                       "Maximum tokens per LLM call", "core_evolution", min_value=1, max_value=32000)
        self._add_param("top_p", ParameterType.FLOAT, 1.0,
                       "Nucleus sampling parameter", "core_evolution", min_value=0.0, max_value=1.0)
        self._add_param("frequency_penalty", ParameterType.FLOAT, 0.0,
                       "Frequency penalty", "core_evolution", min_value=-2.0, max_value=2.0)
        self._add_param("presence_penalty", ParameterType.FLOAT, 0.0,
                       "Presence penalty", "core_evolution", min_value=-2.0, max_value=2.0)
        self._add_param("seed", ParameterType.INTEGER, None,
                       "Random seed for reproducibility", "core_evolution")
        self._add_param("random_seed", ParameterType.INTEGER, 42,
                       "Alternative random seed", "core_evolution")
        self._add_param("api_timeout", ParameterType.INTEGER, 60,
                       "API request timeout (seconds)", "core_evolution", min_value=1, max_value=600)
        self._add_param("api_retries", ParameterType.INTEGER, 3,
                       "Number of API retry attempts", "core_evolution", min_value=0, max_value=10)
        self._add_param("api_retry_delay", ParameterType.INTEGER, 5,
                       "Delay between retries (seconds)", "core_evolution", min_value=1, max_value=60)
        self._add_param("content_type", ParameterType.STRING, "general",
                       "Type of content being evolved", "core_evolution")
        self._add_param("system_message", ParameterType.STRING, "",
                       "System prompt for LLM", "core_evolution")
        
        # Category 2: Model Configuration (10)
        self._add_param("model_configs", ParameterType.LIST, [],
                       "List of model configurations", "model_config")
        self._add_param("api_key", ParameterType.STRING, "",
                       "API key for LLM provider", "model_config", required=True)
        self._add_param("api_base", ParameterType.STRING, "https://api.openai.com/v1",
                       "Base URL for API", "model_config")
        self._add_param("extra_headers", ParameterType.DICT, {},
                       "Additional HTTP headers", "model_config")
        self._add_param("n", ParameterType.INTEGER, 1,
                       "Number of completions per request", "model_config", min_value=1, max_value=10)
        self._add_param("logit_bias", ParameterType.DICT, {},
                       "Token likelihood modifications", "model_config")
        self._add_param("stop_sequences", ParameterType.LIST, [],
                       "Sequences that stop generation", "model_config")
        self._add_param("logprobs", ParameterType.BOOLEAN, False,
                       "Include log probabilities", "model_config")
        self._add_param("top_logprobs", ParameterType.INTEGER, 0,
                       "Number of top log probs", "model_config", min_value=0, max_value=20)
        self._add_param("response_format", ParameterType.SELECT, "text",
                       "Response format", "model_config", options=["text", "json"])
        
        # Category 3: Quality Diversity (12)
        self._add_param("feature_dimensions", ParameterType.LIST, None,
                       "Feature dimensions for behavior", "quality_diversity")
        self._add_param("feature_bins", ParameterType.INTEGER, 10,
                       "Bins per feature dimension", "quality_diversity", min_value=2, max_value=100)
        self._add_param("archive_size", ParameterType.INTEGER, 100,
                       "Maximum archive size", "quality_diversity", min_value=1, max_value=10000)
        self._add_param("behavior_dimensions", ParameterType.LIST, [],
                       "Specific behavior dimensions", "quality_diversity")
        self._add_param("diversity_metric", ParameterType.SELECT, "edit_distance",
                       "Diversity measurement metric", "quality_diversity",
                       options=["edit_distance", "semantic", "behavioral"])
        self._add_param("diversity_reference_size", ParameterType.INTEGER, 20,
                       "Reference set size for diversity", "quality_diversity", min_value=1, max_value=1000)
        self._add_param("adaptive_feature_dimensions", ParameterType.BOOLEAN, True,
                       "Dynamically adjust features", "quality_diversity")
        self._add_param("double_selection", ParameterType.BOOLEAN, True,
                       "Different programs for performance vs inspiration", "quality_diversity")
        self._add_param("qd_algorithm", ParameterType.SELECT, "MAP-Elites",
                       "QD algorithm to use", "quality_diversity",
                       options=["MAP-Elites", "CVT-MAP-Elites", "CMA-ME"])
        self._add_param("novelty_threshold", ParameterType.FLOAT, 0.1,
                       "Minimum novelty for archive", "quality_diversity", min_value=0.0, max_value=1.0)
        self._add_param("behavior_descriptor_type", ParameterType.SELECT, "hand_crafted",
                       "Type of behavior descriptor", "quality_diversity",
                       options=["hand_crafted", "learned"])
        self._add_param("archive_learning_rate", ParameterType.FLOAT, 0.1,
                       "Archive adaptation rate", "quality_diversity", min_value=0.0, max_value=1.0)
        
        # Category 4: Multi-Objective (10)
        self._add_param("objectives", ParameterType.LIST, None,
                       "List of objectives to optimize", "multi_objective")
        self._add_param("objective_weights", ParameterType.LIST, [],
                       "Weights for each objective", "multi_objective")
        self._add_param("pareto_front_size", ParameterType.INTEGER, 50,
                       "Maximum Pareto front size", "multi_objective", min_value=1, max_value=1000)
        self._add_param("dominance_metric", ParameterType.SELECT, "pareto",
                       "Dominance metric", "multi_objective", options=["pareto", "epsilon"])
        self._add_param("constraint_handling", ParameterType.SELECT, "penalty",
                       "Constraint handling method", "multi_objective",
                       options=["penalty", "repair", "death_penalty"])
        self._add_param("reference_point", ParameterType.LIST, [],
                       "Reference point for hypervolume", "multi_objective")
        self._add_param("crowding_distance", ParameterType.BOOLEAN, True,
                       "Use crowding distance", "multi_objective")
        self._add_param("epsilon_dominance", ParameterType.FLOAT, 0.01,
                       "Epsilon for epsilon-dominance", "multi_objective", min_value=0.0, max_value=1.0)
        self._add_param("decomposition_method", ParameterType.SELECT, "weighted_sum",
                       "Objective decomposition method", "multi_objective",
                       options=["weighted_sum", "tchebycheff", "boundary_intersection"])
        self._add_param("scalarization_function", ParameterType.STRING, "weighted_sum",
                       "Scalarization function", "multi_objective")
        
        # Category 5: Adversarial (12)
        self._add_param("attack_model_config", ParameterType.DICT, None,
                       "Attack model configuration", "adversarial")
        self._add_param("defense_model_config", ParameterType.DICT, None,
                       "Defense model configuration", "adversarial")
        self._add_param("adversarial_rounds", ParameterType.INTEGER, 5,
                       "Number of adversarial rounds", "adversarial", min_value=1, max_value=100)
        self._add_param("attack_strength", ParameterType.FLOAT, 0.5,
                       "Strength of attacks", "adversarial", min_value=0.0, max_value=1.0)
        self._add_param("defense_strategy", ParameterType.SELECT, "reactive",
                       "Defense strategy", "adversarial",
                       options=["reactive", "proactive", "adaptive"])
        self._add_param("coevolutionary_approach", ParameterType.BOOLEAN, False,
                       "Use co-evolution", "adversarial")
        self._add_param("red_team_models", ParameterType.LIST, [],
                       "Red team model IDs", "adversarial")
        self._add_param("blue_team_models", ParameterType.LIST, [],
                       "Blue team model IDs", "adversarial")
        self._add_param("red_team_sample_size", ParameterType.INTEGER, 3,
                       "Red team models to sample", "adversarial", min_value=1, max_value=20)
        self._add_param("blue_team_sample_size", ParameterType.INTEGER, 3,
                       "Blue team models to sample", "adversarial", min_value=1, max_value=20)
        self._add_param("adversarial_temperature", ParameterType.FLOAT, 0.8,
                       "Temperature for adversarial generation", "adversarial", min_value=0.0, max_value=2.0)
        self._add_param("attack_diversity", ParameterType.BOOLEAN, True,
                       "Encourage diverse attacks", "adversarial")
        
        # Category 6: Island Model (10)
        self._add_param("num_islands", ParameterType.INTEGER, 5,
                       "Number of islands", "island_model", min_value=1, max_value=100)
        self._add_param("migration_interval", ParameterType.INTEGER, 10,
                       "Generations between migrations", "island_model", min_value=1, max_value=1000)
        self._add_param("migration_rate", ParameterType.FLOAT, 0.1,
                       "Proportion to migrate", "island_model", min_value=0.0, max_value=1.0)
        self._add_param("migration_topology", ParameterType.SELECT, "ring",
                       "Migration topology", "island_model",
                       options=["ring", "fully_connected", "random", "star"])
        self._add_param("ring_topology", ParameterType.BOOLEAN, True,
                       "Use ring topology", "island_model")
        self._add_param("controlled_gene_flow", ParameterType.BOOLEAN, True,
                       "Control gene flow", "island_model")
        self._add_param("island_diversity_metric", ParameterType.STRING, "edit_distance",
                       "Island diversity metric", "island_model")
        self._add_param("migration_selection", ParameterType.SELECT, "best",
                       "Migrant selection method", "island_model",
                       options=["best", "random", "diverse", "tournament"])
        self._add_param("island_initialization", ParameterType.SELECT, "random",
                       "Island initialization method", "island_model",
                       options=["random", "clustered", "diverse"])
        self._add_param("island_specialization", ParameterType.BOOLEAN, False,
                       "Allow island specialization", "island_model")
        
        # Category 7: Selection & Reproduction (12)
        self._add_param("elite_ratio", ParameterType.FLOAT, 0.1,
                       "Proportion of elites", "selection", min_value=0.0, max_value=1.0)
        self._add_param("exploration_ratio", ParameterType.FLOAT, 0.2,
                       "Proportion for exploration", "selection", min_value=0.0, max_value=1.0)
        self._add_param("exploitation_ratio", ParameterType.FLOAT, 0.7,
                       "Proportion for exploitation", "selection", min_value=0.0, max_value=1.0)
        self._add_param("multi_strategy_sampling", ParameterType.BOOLEAN, True,
                       "Use multiple sampling strategies", "selection")
        self._add_param("selection_pressure", ParameterType.FLOAT, 2.0,
                       "Selection pressure", "selection", min_value=1.0, max_value=10.0)
        self._add_param("tournament_size", ParameterType.INTEGER, 3,
                       "Tournament size", "selection", min_value=2, max_value=20)
        self._add_param("crossover_rate", ParameterType.FLOAT, 0.8,
                       "Crossover rate", "selection", min_value=0.0, max_value=1.0)
        self._add_param("mutation_rate", ParameterType.FLOAT, 0.1,
                       "Mutation rate", "selection", min_value=0.0, max_value=1.0)
        self._add_param("elitism_count", ParameterType.INTEGER, 2,
                       "Number of elites to preserve", "selection", min_value=0, max_value=100)
        self._add_param("selection_method", ParameterType.SELECT, "tournament",
                       "Selection method", "selection",
                       options=["tournament", "roulette", "rank", "stochastic"])
        self._add_param("reproduction_method", ParameterType.SELECT, "both",
                       "Reproduction method", "selection",
                       options=["crossover", "mutation", "both"])
        self._add_param("parent_selection", ParameterType.SELECT, "fitness",
                       "Parent selection method", "selection",
                       options=["fitness", "random", "diverse"])
        
        # Category 8: Evaluation (15)
        self._add_param("cascade_evaluation", ParameterType.BOOLEAN, True,
                       "Use cascade evaluation", "evaluation")
        self._add_param("cascade_thresholds", ParameterType.LIST, [0.5, 0.75, 0.9],
                       "Thresholds for cascade levels", "evaluation")
        self._add_param("parallel_evaluations", ParameterType.INTEGER, 4,
                       "Number of parallel workers", "evaluation", min_value=1, max_value=100)
        self._add_param("evaluator_timeout", ParameterType.INTEGER, 300,
                       "Evaluation timeout (seconds)", "evaluation", min_value=1, max_value=3600)
        self._add_param("max_retries_eval", ParameterType.INTEGER, 3,
                       "Max evaluation retries", "evaluation", min_value=0, max_value=10)
        self._add_param("use_llm_feedback", ParameterType.BOOLEAN, False,
                       "Use LLM-based feedback", "evaluation")
        self._add_param("llm_feedback_weight", ParameterType.FLOAT, 0.1,
                       "Weight for LLM feedback", "evaluation", min_value=0.0, max_value=1.0)
        self._add_param("evaluator_models", ParameterType.LIST, None,
                       "Evaluator model configurations", "evaluation")
        self._add_param("evaluator_system_message", ParameterType.STRING, "",
                       "System prompt for evaluator", "evaluation")
        self._add_param("ensemble_size", ParameterType.INTEGER, 3,
                       "Number of evaluators in ensemble", "evaluation", min_value=1, max_value=20)
        self._add_param("consensus_threshold", ParameterType.FLOAT, 0.7,
                       "Threshold for consensus", "evaluation", min_value=0.0, max_value=1.0)
        self._add_param("evaluation_criteria", ParameterType.LIST, [],
                       "List of evaluation criteria", "evaluation")
        self._add_param("custom_evaluator", ParameterType.STRING, None,
                       "Custom evaluation function", "evaluation")
        self._add_param("evaluation_batch_size", ParameterType.INTEGER, 10,
                       "Batch size for evaluations", "evaluation", min_value=1, max_value=1000)
        self._add_param("cache_evaluations", ParameterType.BOOLEAN, True,
                       "Cache evaluation results", "evaluation")
        
        # Category 9: Prompt Engineering (12 parameters)
        self._add_param("prompt_template", ParameterType.STRING, "default",
                       "Base prompt template", "prompt_engineering")
        self._add_param("system_prompt", ParameterType.STRING, "",
                       "System-level prompt", "prompt_engineering")
        self._add_param("context_length", ParameterType.INTEGER, 2000,
                       "Maximum context length", "prompt_engineering", min_value=100, max_value=8000)
        self._add_param("prompt_optimization", ParameterType.BOOLEAN, True,
                       "Optimize prompts during evolution", "prompt_engineering")
        self._add_param("template_stochasticity", ParameterType.BOOLEAN, True,
                       "Use stochastic templates", "prompt_engineering")
        self._add_param("meta_prompting", ParameterType.BOOLEAN, False,
                       "Use meta-prompting techniques", "prompt_engineering")
        self._add_param("few_shot_examples", ParameterType.INTEGER, 3,
                       "Number of few-shot examples", "prompt_engineering", min_value=0, max_value=20)
        self._add_param("chain_of_thought", ParameterType.BOOLEAN, True,
                       "Use chain-of-thought prompting", "prompt_engineering")
        self._add_param("self_consistency", ParameterType.BOOLEAN, False,
                       "Use self-consistency decoding", "prompt_engineering")
        self._add_param("prompt_ensembling", ParameterType.BOOLEAN, False,
                       "Ensemble multiple prompts", "prompt_engineering")
        self._add_param("dynamic_prompting", ParameterType.BOOLEAN, False,
                       "Dynamically adjust prompts", "prompt_engineering")
        self._add_param("prompt_compression", ParameterType.BOOLEAN, False,
                       "Compress long prompts", "prompt_engineering")
        
        # Category 10: Artifact Management (10 parameters)
        self._add_param("enable_artifacts", ParameterType.BOOLEAN, True,
                       "Enable artifact generation", "artifact_management")
        self._add_param("artifact_types", ParameterType.LIST, ["code", "text"],
                       "Types of artifacts to generate", "artifact_management")
        self._add_param("max_artifact_size", ParameterType.INTEGER, 20480,
                       "Maximum artifact size in bytes", "artifact_management", min_value=1024, max_value=1048576)
        self._add_param("artifact_validation", ParameterType.BOOLEAN, True,
                       "Validate generated artifacts", "artifact_management")
        self._add_param("artifact_compression", ParameterType.BOOLEAN, False,
                       "Compress artifacts", "artifact_management")
        self._add_param("artifact_versioning", ParameterType.BOOLEAN, True,
                       "Version control for artifacts", "artifact_management")
        self._add_param("artifact_metadata", ParameterType.BOOLEAN, True,
                       "Include metadata with artifacts", "artifact_management")
        self._add_param("artifact_cleanup", ParameterType.BOOLEAN, True,
                       "Clean up old artifacts", "artifact_management")
        self._add_param("artifact_storage", ParameterType.SELECT, "memory",
                       "Artifact storage location", "artifact_management",
                       options=["memory", "disk", "cloud"])
        self._add_param("artifact_encryption", ParameterType.BOOLEAN, False,
                       "Encrypt sensitive artifacts", "artifact_management")
        
        # Category 11: Resource Management (10 parameters)
        self._add_param("memory_limit_mb", ParameterType.INTEGER, 4096,
                       "Memory limit in MB", "resource_management", min_value=512, max_value=32768)
        self._add_param("cpu_limit", ParameterType.FLOAT, 0.8,
                       "CPU usage limit (fraction)", "resource_management", min_value=0.1, max_value=1.0)
        self._add_param("max_time", ParameterType.INTEGER, 1800,
                       "Maximum execution time in seconds", "resource_management", min_value=60, max_value=7200)
        self._add_param("disk_limit_mb", ParameterType.INTEGER, 1024,
                       "Disk usage limit in MB", "resource_management", min_value=100, max_value=10240)
        self._add_param("network_limit_mbps", ParameterType.INTEGER, 100,
                       "Network bandwidth limit", "resource_management", min_value=1, max_value=1000)
        self._add_param("api_call_limit", ParameterType.INTEGER, 1000,
                       "Maximum API calls", "resource_management", min_value=10, max_value=10000)
        self._add_param("token_limit", ParameterType.INTEGER, 100000,
                       "Maximum tokens", "resource_management", min_value=1000, max_value=1000000)
        self._add_param("cost_limit_usd", ParameterType.FLOAT, 10.0,
                       "Maximum cost in USD", "resource_management", min_value=0.01, max_value=1000.0)
        self._add_param("resource_monitoring", ParameterType.BOOLEAN, True,
                       "Monitor resource usage", "resource_management")
        self._add_param("auto_scaling", ParameterType.BOOLEAN, False,
                       "Auto-scale resources", "resource_management")
        
        # Category 12: Database & Storage (10 parameters)
        self._add_param("db_path", ParameterType.STRING, "./openevolve.db",
                       "Database file path", "database_storage")
        self._add_param("db_type", ParameterType.SELECT, "sqlite",
                       "Database type", "database_storage",
                       options=["sqlite", "postgresql", "mongodb"])
        self._add_param("connection_string", ParameterType.STRING, "",
                       "Database connection string", "database_storage")
        self._add_param("max_connections", ParameterType.INTEGER, 10,
                       "Maximum database connections", "database_storage", min_value=1, max_value=100)
        self._add_param("connection_timeout", ParameterType.INTEGER, 30,
                       "Connection timeout in seconds", "database_storage", min_value=1, max_value=60)
        self._add_param("query_timeout", ParameterType.INTEGER, 60,
                       "Query timeout in seconds", "database_storage", min_value=1, max_value=300)
        self._add_param("batch_size", ParameterType.INTEGER, 1000,
                       "Batch size for operations", "database_storage", min_value=1, max_value=10000)
        self._add_param("compression", ParameterType.BOOLEAN, True,
                       "Compress stored data", "database_storage")
        self._add_param("encryption", ParameterType.BOOLEAN, False,
                       "Encrypt stored data", "database_storage")
        self._add_param("backup_enabled", ParameterType.BOOLEAN, True,
                       "Enable automatic backups", "database_storage")
        
        # Category 13: Evolution Tracing (12 parameters)
        self._add_param("trace_enabled", ParameterType.BOOLEAN, False,
                       "Enable evolution tracing", "evolution_tracing")
        self._add_param("trace_level", ParameterType.SELECT, "basic",
                       "Level of tracing detail", "evolution_tracing",
                       options=["basic", "detailed", "full"])
        self._add_param("trace_format", ParameterType.SELECT, "json",
                       "Trace output format", "evolution_tracing",
                       options=["json", "csv", "binary"])
        self._add_param("trace_file", ParameterType.STRING, "./trace.log",
                       "Trace output file", "evolution_tracing")
        self._add_param("trace_compression", ParameterType.BOOLEAN, True,
                       "Compress trace files", "evolution_tracing")
        self._add_param("trace_rotation", ParameterType.BOOLEAN, True,
                       "Rotate trace files", "evolution_tracing")
        self._add_param("max_trace_size_mb", ParameterType.INTEGER, 100,
                       "Maximum trace file size", "evolution_tracing", min_value=1, max_value=1024)
        self._add_param("trace_buffer_size", ParameterType.INTEGER, 1000,
                       "Trace buffer size", "evolution_tracing", min_value=100, max_value=10000)
        self._add_param("real_time_tracing", ParameterType.BOOLEAN, False,
                       "Real-time trace streaming", "evolution_tracing")
        self._add_param("trace_sampling", ParameterType.FLOAT, 1.0,
                       "Sampling rate for tracing", "evolution_tracing", min_value=0.01, max_value=1.0)
        self._add_param("include_population", ParameterType.BOOLEAN, False,
                       "Include population in trace", "evolution_tracing")
        self._add_param("include_fitness", ParameterType.BOOLEAN, True,
                       "Include fitness in trace", "evolution_tracing")
        
        # Category 14: Early Stopping (8 parameters)
        self._add_param("early_stopping", ParameterType.BOOLEAN, False,
                       "Enable early stopping", "early_stopping")
        self._add_param("early_stopping_patience", ParameterType.INTEGER, 10,
                       "Patience for early stopping", "early_stopping", min_value=1, max_value=100)
        self._add_param("min_improvement", ParameterType.FLOAT, 0.001,
                       "Minimum improvement threshold", "early_stopping", min_value=0.0, max_value=1.0)
        self._add_param("improvement_window", ParameterType.INTEGER, 5,
                       "Window for improvement calculation", "early_stopping", min_value=1, max_value=50)
        self._add_param("plateau_threshold", ParameterType.INTEGER, 20,
                       "Generations to consider plateau", "early_stopping", min_value=1, max_value=100)
        self._add_param("convergence_check", ParameterType.BOOLEAN, True,
                       "Check for convergence", "early_stopping")
        self._add_param("diversity_threshold", ParameterType.FLOAT, 0.01,
                       "Minimum diversity threshold", "early_stopping", min_value=0.0, max_value=1.0)
        self._add_param("stagnation_limit", ParameterType.INTEGER, 50,
                       "Maximum stagnation generations", "early_stopping", min_value=1, max_value=100)
        
        # Category 15: Distributed Processing (10 parameters)
        self._add_param("distributed", ParameterType.BOOLEAN, False,
                       "Enable distributed processing", "distributed_processing")
        self._add_param("num_workers", ParameterType.INTEGER, 4,
                       "Number of worker processes", "distributed_processing", min_value=1, max_value=100)
        self._add_param("worker_timeout", ParameterType.INTEGER, 120,
                       "Worker timeout in seconds", "distributed_processing", min_value=10, max_value=600)
        self._add_param("load_balancing", ParameterType.SELECT, "round_robin",
                       "Load balancing strategy", "distributed_processing",
                       options=["round_robin", "least_loaded", "random"])
        self._add_param("fault_tolerance", ParameterType.BOOLEAN, True,
                       "Enable fault tolerance", "distributed_processing")
        self._add_param("worker_restart", ParameterType.BOOLEAN, True,
                       "Auto-restart failed workers", "distributed_processing")
        self._add_param("communication_backend", ParameterType.SELECT, "local",
                       "Communication backend", "distributed_processing",
                       options=["local", "redis", "rabbitmq"])
        self._add_param("message_compression", ParameterType.BOOLEAN, True,
                       "Compress messages", "distributed_processing")
        self._add_param("heartbeat_interval", ParameterType.INTEGER, 10,
                       "Heartbeat interval in seconds", "distributed_processing", min_value=1, max_value=60)
        self._add_param("cluster_scaling", ParameterType.BOOLEAN, False,
                       "Auto-scale cluster", "distributed_processing")
        
        # Category 16: Advanced Research (20 parameters)
        self._add_param("novelty_search", ParameterType.BOOLEAN, False,
                       "Enable novelty search", "advanced_research")
        self._add_param("curiosity_driven", ParameterType.BOOLEAN, False,
                       "Curiosity-driven exploration", "advanced_research")
        self._add_param("meta_learning", ParameterType.BOOLEAN, False,
                       "Enable meta-learning", "advanced_research")
        self._add_param("transfer_learning", ParameterType.BOOLEAN, False,
                       "Transfer from previous runs", "advanced_research")
        self._add_param("continual_learning", ParameterType.BOOLEAN, False,
                       "Continual learning mode", "advanced_research")
        self._add_param("few_shot_adaptation", ParameterType.BOOLEAN, False,
                       "Few-shot adaptation", "advanced_research")
        self._add_param("zero_shot_transfer", ParameterType.BOOLEAN, False,
                       "Zero-shot transfer", "advanced_research")
        self._add_param("domain_adaptation", ParameterType.BOOLEAN, False,
                       "Domain adaptation", "advanced_research")
        self._add_param("multi_task_learning", ParameterType.BOOLEAN, False,
                       "Multi-task learning", "advanced_research")
        self._add_param("lifelong_learning", ParameterType.BOOLEAN, False,
                       "Lifelong learning", "advanced_research")
        self._add_param("neural_architecture_search", ParameterType.BOOLEAN, False,
                       "NAS integration", "advanced_research")
        self._add_param("hyperparameter_optimization", ParameterType.BOOLEAN, False,
                       "HPO integration", "advanced_research")
        self._add_param("automated_ml", ParameterType.BOOLEAN, False,
                       "AutoML features", "advanced_research")
        self._add_param("explainable_ai", ParameterType.BOOLEAN, False,
                       "XAI integration", "advanced_research")
        self._add_param("federated_learning", ParameterType.BOOLEAN, False,
                       "Federated learning", "advanced_research")
        self._add_param("differential_privacy", ParameterType.BOOLEAN, False,
                       "Privacy preservation", "advanced_research")
        self._add_param("quantum_computing", ParameterType.BOOLEAN, False,
                       "Quantum computing support", "advanced_research")
        self._add_param("neuromorphic_computing", ParameterType.BOOLEAN, False,
                       "Neuromorphic support", "advanced_research")
        self._add_param("edge_computing", ParameterType.BOOLEAN, False,
                       "Edge deployment", "advanced_research")
        self._add_param("green_ai", ParameterType.BOOLEAN, False,
                       "Energy-efficient AI", "advanced_research")
        
        # Category 17: Custom Requirements (8 parameters)
        self._add_param("custom_fitness", ParameterType.STRING, "",
                       "Custom fitness function code", "custom_requirements")
        self._add_param("custom_operators", ParameterType.LIST, [],
                       "Custom genetic operators", "custom_requirements")
        self._add_param("custom_constraints", ParameterType.LIST, [],
                       "Custom constraint functions", "custom_requirements")
        self._add_param("domain_knowledge", ParameterType.STRING, "",
                       "Domain-specific knowledge", "custom_requirements")
        self._add_param("expert_rules", ParameterType.LIST, [],
                       "Expert-defined rules", "custom_requirements")
        self._add_param("business_logic", ParameterType.STRING, "",
                       "Business logic constraints", "custom_requirements")
        self._add_param("regulatory_compliance", ParameterType.LIST, [],
                       "Compliance requirements", "custom_requirements")
        self._add_param("ethical_guidelines", ParameterType.LIST, [],
                       "Ethical AI guidelines", "custom_requirements")
        
        # Category 18: UI & Visualization (8 parameters)
        self._add_param("enable_visualization", ParameterType.BOOLEAN, True,
                       "Enable visualizations", "ui_visualization")
        self._add_param("plot_frequency", ParameterType.INTEGER, 10,
                       "Plotting frequency", "ui_visualization", min_value=1, max_value=100)
        self._add_param("plot_types", ParameterType.LIST, ["fitness", "diversity"],
                       "Types of plots to generate", "ui_visualization")
        self._add_param("interactive_plots", ParameterType.BOOLEAN, True,
                       "Interactive visualizations", "ui_visualization")
        self._add_param("real_time_updates", ParameterType.BOOLEAN, False,
                       "Real-time plot updates", "ui_visualization")
        self._add_param("export_plots", ParameterType.BOOLEAN, True,
                       "Export plots to files", "ui_visualization")
        self._add_param("plot_format", ParameterType.SELECT, "png",
                       "Plot export format", "ui_visualization",
                       options=["png", "svg", "pdf"])
        self._add_param("dashboard_enabled", ParameterType.BOOLEAN, True,
                       "Enable monitoring dashboard", "ui_visualization")
        
        # Category 19: Experimental (7 parameters)
        self._add_param("experimental_features", ParameterType.BOOLEAN, False,
                       "Enable experimental features", "experimental")
        self._add_param("beta_algorithms", ParameterType.BOOLEAN, False,
                       "Use beta algorithms", "experimental")
        self._add_param("research_mode", ParameterType.BOOLEAN, False,
                       "Research mode settings", "experimental")
        self._add_param("debug_mode", ParameterType.BOOLEAN, False,
                       "Debug mode", "experimental")
        self._add_param("profiling_enabled", ParameterType.BOOLEAN, False,
                       "Performance profiling", "experimental")
        self._add_param("memory_profiling", ParameterType.BOOLEAN, False,
                       "Memory usage profiling", "experimental")
        self._add_param("experimental_logging", ParameterType.BOOLEAN, False,
                       "Experimental logging", "experimental")
        
        # Category 20: Adaptive MDAP (8 parameters)
        self._add_param("enable_adaptive_mdap", ParameterType.BOOLEAN, True,
                       "Enable Adaptive MDAP resource allocation", "adaptive_mdap")
        self._add_param("adaptive_mdap_profile", ParameterType.SELECT, "balanced",
                       "Resource allocation profile", "adaptive_mdap",
                       options=["conservative", "balanced", "aggressive"])
        self._add_param("adaptive_mdap_learning", ParameterType.BOOLEAN, False,
                       "Enable learning from execution history", "adaptive_mdap")
        self._add_param("adaptive_mdap_context_aware", ParameterType.BOOLEAN, False,
                       "Use workflow context for complexity estimation", "adaptive_mdap")
        self._add_param("adaptive_mdap_threshold_1", ParameterType.FLOAT, 0.2,
                       "DIRECT to MDAP_LIGHT threshold", "adaptive_mdap", min_value=0.0, max_value=1.0)
        self._add_param("adaptive_mdap_threshold_2", ParameterType.FLOAT, 0.4,
                       "MDAP_LIGHT to MDAP_MEDIUM threshold", "adaptive_mdap", min_value=0.0, max_value=1.0)
        self._add_param("adaptive_mdap_threshold_3", ParameterType.FLOAT, 0.6,
                       "MDAP_MEDIUM to MAKER_FULL threshold", "adaptive_mdap", min_value=0.0, max_value=1.0)
        self._add_param("adaptive_mdap_threshold_4", ParameterType.FLOAT, 0.8,
                       "MAKER_FULL to MAKER_ULTRA threshold", "adaptive_mdap", min_value=0.0, max_value=1.0)
        
        # Additional core parameters from API reference
        self._add_param("convergence_threshold", ParameterType.FLOAT, 0.001,
                       "Threshold for convergence detection", "core_evolution", min_value=0.0, max_value=1.0)
        self._add_param("fitness_function", ParameterType.STRING, "default",
                       "Fitness evaluation function", "core_evolution")
        self._add_param("elitism", ParameterType.BOOLEAN, True,
                       "Preserve best individuals", "core_evolution")
        self._add_param("diversity_maintenance", ParameterType.BOOLEAN, True,
                       "Maintain population diversity", "core_evolution")
        self._add_param("adaptive_parameters", ParameterType.BOOLEAN, False,
                       "Adapt parameters during evolution", "core_evolution")
        
        # Additional model configuration parameters
        self._add_param("model_id", ParameterType.STRING, "gpt-4",
                       "Primary model identifier", "model_config")
        self._add_param("backup_models", ParameterType.LIST, [],
                       "Fallback model list", "model_config")
        self._add_param("timeout", ParameterType.INTEGER, 30,
                       "Request timeout in seconds", "model_config", min_value=1, max_value=300)
        self._add_param("max_retries", ParameterType.INTEGER, 3,
                       "Maximum retry attempts", "model_config", min_value=0, max_value=10)
        self._add_param("retry_delay", ParameterType.FLOAT, 1.0,
                       "Delay between retries", "model_config", min_value=0.1, max_value=10.0)
        self._add_param("rate_limit", ParameterType.INTEGER, 60,
                       "Requests per minute", "model_config", min_value=1, max_value=1000)
        self._add_param("concurrent_requests", ParameterType.INTEGER, 5,
                       "Concurrent API requests", "model_config", min_value=1, max_value=50)
        self._add_param("model_rotation", ParameterType.BOOLEAN, False,
                       "Rotate between models", "model_config")
        
        # Additional quality diversity parameters
        self._add_param("quality_threshold", ParameterType.FLOAT, 0.0,
                       "Minimum quality for archive", "quality_diversity", min_value=0.0, max_value=1.0)
        self._add_param("diversity_weight", ParameterType.FLOAT, 0.5,
                       "Weight of diversity vs quality", "quality_diversity", min_value=0.0, max_value=1.0)
        self._add_param("behavior_space", ParameterType.STRING, "auto",
                       "Behavior space definition", "quality_diversity")
        self._add_param("distance_metric", ParameterType.SELECT, "euclidean",
                       "Distance calculation method", "quality_diversity",
                       options=["euclidean", "manhattan", "cosine"])
        self._add_param("archive_update_freq", ParameterType.INTEGER, 1,
                       "Archive update frequency", "quality_diversity", min_value=1, max_value=100)
        self._add_param("exploration_bonus", ParameterType.FLOAT, 0.1,
                       "Bonus for exploration", "quality_diversity", min_value=0.0, max_value=2.0)
        self._add_param("pareto_layers", ParameterType.INTEGER, 3,
                       "Number of Pareto layers", "quality_diversity", min_value=1, max_value=10)
        
        # Additional multi-objective parameters
        self._add_param("dominance_type", ParameterType.SELECT, "standard",
                       "Dominance relation type", "multi_objective",
                       options=["standard", "epsilon", "fuzzy"])
        self._add_param("epsilon_values", ParameterType.LIST, [],
                       "Epsilon values for epsilon-dominance", "multi_objective")
        self._add_param("scalarization", ParameterType.SELECT, "weighted_sum",
                       "Scalarization method", "multi_objective",
                       options=["weighted_sum", "tchebycheff", "pbi"])
        self._add_param("constraint_tolerance", ParameterType.FLOAT, 0.01,
                       "Tolerance for constraints", "multi_objective", min_value=0.0, max_value=1.0)
        self._add_param("hypervolume_ref", ParameterType.LIST, [],
                       "Hypervolume reference point", "multi_objective")
        
        # Additional adversarial parameters
        self._add_param("attack_strength", ParameterType.FLOAT, 1.0,
                       "Strength of adversarial attacks", "adversarial", min_value=0.1, max_value=2.0)
        self._add_param("defense_strength", ParameterType.FLOAT, 1.0,
                       "Strength of defense mechanisms", "adversarial", min_value=0.1, max_value=2.0)
        self._add_param("adversarial_budget", ParameterType.INTEGER, 100,
                       "Budget for adversarial operations", "adversarial", min_value=1, max_value=1000)
        self._add_param("attack_types", ParameterType.LIST, [],
                       "Types of attacks to use", "adversarial")
        self._add_param("defense_strategies", ParameterType.LIST, [],
                       "Defense strategies to employ", "adversarial")
        self._add_param("robustness_metric", ParameterType.STRING, "accuracy",
                       "Metric for robustness evaluation", "adversarial")
        self._add_param("perturbation_bound", ParameterType.FLOAT, 0.1,
                       "Maximum perturbation allowed", "adversarial", min_value=0.0, max_value=1.0)
        self._add_param("gradient_masking", ParameterType.BOOLEAN, False,
                       "Use gradient masking", "adversarial")
        self._add_param("ensemble_defense", ParameterType.BOOLEAN, True,
                       "Use ensemble for defense", "adversarial")
        
        # Additional island model parameters
        self._add_param("migration_size", ParameterType.INTEGER, 5,
                       "Number of individuals to migrate", "island_model", min_value=1, max_value=50)
        self._add_param("migration_policy", ParameterType.SELECT, "best",
                       "Migration selection policy", "island_model",
                       options=["best", "random", "diverse"])
        self._add_param("replacement_policy", ParameterType.SELECT, "worst",
                       "Replacement policy", "island_model",
                       options=["worst", "random", "similar"])
        self._add_param("island_sizes", ParameterType.LIST, [],
                       "Custom sizes for each island", "island_model")
        self._add_param("heterogeneous_islands", ParameterType.BOOLEAN, False,
                       "Use different algorithms per island", "island_model")
        self._add_param("synchronous_migration", ParameterType.BOOLEAN, True,
                       "Synchronize migration timing", "island_model")
        self._add_param("adaptive_migration", ParameterType.BOOLEAN, False,
                       "Adapt migration parameters", "island_model")
        
        # Additional selection parameters
        self._add_param("random_ratio", ParameterType.FLOAT, 0.2,
                       "Ratio for random selection", "selection", min_value=0.0, max_value=1.0)
        self._add_param("survivor_selection", ParameterType.SELECT, "generational",
                       "Survivor selection method", "selection",
                       options=["generational", "steady_state"])
        self._add_param("replacement_rate", ParameterType.FLOAT, 1.0,
                       "Population replacement rate", "selection", min_value=0.0, max_value=1.0)
        self._add_param("selection_pressure_decay", ParameterType.FLOAT, 0.0,
                       "Selection pressure decay rate", "selection", min_value=0.0, max_value=1.0)
        self._add_param("diversity_selection", ParameterType.BOOLEAN, False,
                       "Include diversity in selection", "selection")
        self._add_param("age_based_selection", ParameterType.BOOLEAN, False,
                       "Consider individual age", "selection")
        
        # Additional evaluation parameters
        self._add_param("cache_size", ParameterType.INTEGER, 1000,
                       "Maximum cache size", "evaluation", min_value=100, max_value=10000)
        self._add_param("evaluation_noise", ParameterType.FLOAT, 0.0,
                       "Noise level in evaluations", "evaluation", min_value=0.0, max_value=0.5)
        self._add_param("fitness_scaling", ParameterType.SELECT, "linear",
                       "Fitness scaling method", "evaluation",
                       options=["linear", "exponential", "logarithmic"])
        self._add_param("normalization", ParameterType.BOOLEAN, True,
                       "Normalize fitness values", "evaluation")
        self._add_param("multi_criteria_eval", ParameterType.BOOLEAN, False,
                       "Multi-criteria evaluation", "evaluation")
        self._add_param("evaluation_budget", ParameterType.INTEGER, 10000,
                       "Total evaluation budget", "evaluation", min_value=1, max_value=100000)
        self._add_param("incremental_eval", ParameterType.BOOLEAN, False,
                       "Incremental evaluation", "evaluation")
        self._add_param("surrogate_model", ParameterType.BOOLEAN, False,
                       "Use surrogate model", "evaluation")
        self._add_param("active_learning", ParameterType.BOOLEAN, False,
                       "Active learning for evaluation", "evaluation")
        self._add_param("uncertainty_sampling", ParameterType.BOOLEAN, False,
                       "Sample uncertain regions", "evaluation")
        
        # Additional checkpoint parameter
        self._add_param("checkpoint_interval", ParameterType.INTEGER, 10,
                       "Generations between checkpoints", "resource_management", min_value=1, max_value=1000)
        
        # Additional core evolution parameters to reach 211 total
        self._add_param("adaptive_stopping", ParameterType.BOOLEAN, False,
                       "Adaptive stopping criteria", "early_stopping")
        self._add_param("reasoning_effort", ParameterType.SELECT, "medium",
                       "Reasoning effort level", "core_evolution",
                       options=["low", "medium", "high"])
        self._add_param("language", ParameterType.STRING, "python",
                       "Programming language", "core_evolution")
        self._add_param("file_suffix", ParameterType.STRING, ".py",
                       "File extension", "core_evolution")
    
    def _add_param(self, name: str, param_type: ParameterType, default: Any,
                   description: str, category: str, **kwargs):
        """Add a parameter to the schema"""
        self.parameters[name] = Parameter(
            name=name,
            type=param_type,
            default=default,
            description=description,
            category=category,
            **kwargs
        )
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get parameter definition"""
        return self.parameters.get(name)
    
    def get_categories(self) -> List[str]:
        """Get all parameter categories"""
        return list(set(p.category for p in self.parameters.values()))
    
    def get_parameters_by_category(self, category: str) -> List[Parameter]:
        """Get all parameters in a category"""
        return [p for p in self.parameters.values() if p.category == category]


class ParameterValidator:
    """Validates parameter values"""
    
    def __init__(self, schema: ParameterSchema):
        self.schema = schema
    
    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameter configuration"""
        result = ValidationResult(valid=True)
        
        # Check required parameters
        for param in self.schema.parameters.values():
            if param.required and param.name not in params:
                result.errors.append(f"Required parameter '{param.name}' is missing")
                result.valid = False
        
        # Validate each provided parameter
        for name, value in params.items():
            param = self.schema.get_parameter(name)
            if not param:
                result.warnings.append(f"Unknown parameter '{name}'")
                continue
            
            # Type validation
            if not self._validate_type(value, param.type):
                result.errors.append(f"Parameter '{name}' has invalid type")
                result.valid = False
                continue
            
            # Range validation
            if param.min_value is not None and isinstance(value, (int, float)):
                if value < param.min_value:
                    result.errors.append(f"Parameter '{name}' below minimum {param.min_value}")
                    result.valid = False
            
            if param.max_value is not None and isinstance(value, (int, float)):
                if value > param.max_value:
                    result.errors.append(f"Parameter '{name}' above maximum {param.max_value}")
                    result.valid = False
            
            # Options validation
            if param.options and value not in param.options:
                result.errors.append(f"Parameter '{name}' must be one of {param.options}")
                result.valid = False
        
        return result
    
    def _validate_type(self, value: Any, param_type: ParameterType) -> bool:
        """Validate value type"""
        if value is None:
            return True
        
        type_map = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.LIST: list,
            ParameterType.DICT: dict,
            ParameterType.SELECT: str
        }
        
        expected_type = type_map.get(param_type)
        if expected_type:
            return isinstance(value, expected_type)
        return True


class PresetManager:
    """Manages configuration presets"""
    
    def __init__(self):
        self.presets = self._initialize_presets()
    
    def _initialize_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configuration presets"""
        return {
            "fast": {
                "max_iterations": 5,
                "population_size": 10,
                "archive_size": 50,
                "parallel_evaluations": 8,
                "checkpoint_interval": 5
            },
            "balanced": {
                "max_iterations": 10,
                "population_size": 20,
                "archive_size": 100,
                "parallel_evaluations": 4,
                "checkpoint_interval": 10
            },
            "thorough": {
                "max_iterations": 50,
                "population_size": 50,
                "archive_size": 500,
                "parallel_evaluations": 2,
                "checkpoint_interval": 25,
                "cascade_evaluation": True,
                "use_llm_feedback": True
            },
            "research": {
                "max_iterations": 100,
                "population_size": 100,
                "archive_size": 1000,
                "parallel_evaluations": 1,
                "checkpoint_interval": 50,
                "cascade_evaluation": True,
                "use_llm_feedback": True,
                "evolution_trace_enabled": True,
                "double_selection": True,
                "adaptive_feature_dimensions": True
            }
        }
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get preset configuration"""
        return self.presets.get(name)
    
    def list_presets(self) -> List[str]:
        """List available presets"""
        return list(self.presets.keys())


class ParameterPersistence:
    """Handles saving and loading configurations"""
    
    def __init__(self, config_dir: str = ".openevolve"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, name: str, params: Dict[str, Any]):
        """Save configuration to file"""
        filepath = os.path.join(self.config_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        filepath = os.path.join(self.config_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_configs(self) -> List[str]:
        """List saved configurations"""
        if not os.path.exists(self.config_dir):
            return []
        
        configs = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                configs.append(filename[:-5])
        return configs
    
    def delete_config(self, name: str) -> bool:
        """Delete saved configuration"""
        filepath = os.path.join(self.config_dir, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


class ParameterManager:
    """Main parameter management class"""
    
    def __init__(self, config_dir: str = ".openevolve"):
        self.schema = ParameterSchema()
        self.validator = ParameterValidator(self.schema)
        self.preset_manager = PresetManager()
        self.persistence = ParameterPersistence(config_dir)
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get parameter definition"""
        return self.schema.get_parameter(name)
    
    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameters"""
        return self.validator.validate(params)
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameters (returns tuple for backward compatibility)"""
        result = self.validate(params)
        return result.valid, result.errors
    
    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get preset configuration"""
        return self.preset_manager.get_preset(name)
    
    def list_presets(self) -> List[str]:
        """List available presets"""
        return self.preset_manager.list_presets()
    
    def save_config(self, name: str, params: Dict[str, Any]):
        """Save configuration"""
        self.persistence.save_config(name, params)
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load configuration"""
        return self.persistence.load_config(name)
    
    def list_configs(self) -> List[str]:
        """List saved configurations"""
        return self.persistence.list_configs()
    
    def delete_config(self, name: str) -> bool:
        """Delete configuration"""
        return self.persistence.delete_config(name)
    
    def get_categories(self) -> List[str]:
        """Get all parameter categories"""
        return self.schema.get_categories()
    
    def get_parameters_by_category(self, category: str) -> List[Parameter]:
        """Get parameters in a category"""
        return self.schema.get_parameters_by_category(category)
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get all default parameter values"""
        return {name: param.default for name, param in self.schema.parameters.items()}
    
    def merge_with_defaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided params with defaults"""
        defaults = self.get_defaults()
        defaults.update(params)
        return defaults
