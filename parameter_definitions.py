from typing import Any, Dict, List, Optional

DEFAULT_PARAMETER_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "core_evolution": {
        "evolution_mode": {
            "type": "select",
            "default": "standard",
            "description": "Evolution strategy",
            "options": ["standard", "quality_diversity", "multi_objective", "adversarial", "problem_decomposition"]
        },
        "max_iterations": {
            "type": "integer",
            "default": 10,
            "description": "Maximum evolution iterations",
            "min_value": 1,
            "max_value": 1000
        },
        "population_size": {
            "type": "integer",
            "default": 20,
            "description": "Population size per generation",
            "min_value": 1,
            "max_value": 1000
        },
        "temperature": {
            "type": "float",
            "default": 0.7,
            "description": "LLM sampling temperature",
            "min_value": 0.0,
            "max_value": 2.0
        },
        "max_tokens": {
            "type": "integer",
            "default": 2048,
            "description": "Maximum tokens per LLM call",
            "min_value": 1,
            "max_value": 32000
        },
        "top_p": {
            "type": "float",
            "default": 1.0,
            "description": "Nucleus sampling parameter",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "frequency_penalty": {
            "type": "float",
            "default": 0.0,
            "description": "Frequency penalty",
            "min_value": -2.0,
            "max_value": 2.0
        },
        "presence_penalty": {
            "type": "float",
            "default": 0.0,
            "description": "Presence penalty",
            "min_value": -2.0,
            "max_value": 2.0
        },
        "seed": {
            "type": "integer",
            "default": None,
            "description": "Random seed for reproducibility"
        },
        "random_seed": {
            "type": "integer",
            "default": 42,
            "description": "Alternative random seed"
        },
        "api_timeout": {
            "type": "integer",
            "default": 60,
            "description": "API request timeout (seconds)",
            "min_value": 1,
            "max_value": 600
        },
        "api_retries": {
            "type": "integer",
            "default": 3,
            "description": "Number of API retry attempts",
            "min_value": 0,
            "max_value": 10
        },
        "api_retry_delay": {
            "type": "integer",
            "default": 5,
            "description": "Delay between retries (seconds)",
            "min_value": 1,
            "max_value": 60
        },
        "content_type": {
            "type": "string",
            "default": "general",
            "description": "Type of content being evolved"
        },
        "system_message": {
            "type": "string",
            "default": "",
            "description": "System prompt for LLM"
        },
        "convergence_threshold": {
            "type": "float",
            "default": 0.001,
            "description": "Threshold for convergence detection",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "fitness_function": {
            "type": "string",
            "default": "default",
            "description": "Fitness evaluation function"
        },
        "elitism": {
            "type": "boolean",
            "default": True,
            "description": "Preserve best individuals"
        },
        "diversity_maintenance": {
            "type": "boolean",
            "default": True,
            "description": "Maintain population diversity"
        },
        "adaptive_parameters": {
            "type": "boolean",
            "default": False,
            "description": "Adapt parameters during evolution"
        },
        "reasoning_effort": {
            "type": "select",
            "default": "medium",
            "description": "Reasoning effort level",
            "options": ["low", "medium", "high"]
        },
        "language": {
            "type": "string",
            "default": "python",
            "description": "Programming language"
        },
        "file_suffix": {
            "type": "string",
            "default": ".py",
            "description": "File extension"
        }
    },
    "model_config": {
        "model_configs": {
            "type": "list",
            "default": [],
            "description": "List of model configurations"
        },
        "api_key": {
            "type": "string",
            "default": "",
            "description": "API key for LLM provider",
            "required": True
        },
        "api_base": {
            "type": "string",
            "default": "https://api.openai.com/v1",
            "description": "Base URL for API"
        },
        "extra_headers": {
            "type": "dict",
            "default": {},
            "description": "Additional HTTP headers"
        },
        "n": {
            "type": "integer",
            "default": 1,
            "description": "Number of completions per request",
            "min_value": 1,
            "max_value": 10
        },
        "logit_bias": {
            "type": "dict",
            "default": {},
            "description": "Token likelihood modifications"
        },
        "stop_sequences": {
            "type": "list",
            "default": [],
            "description": "Sequences that stop generation"
        },
        "logprobs": {
            "type": "boolean",
            "default": False,
            "description": "Include log probabilities"
        },
        "top_logprobs": {
            "type": "integer",
            "default": 0,
            "description": "Number of top log probs",
            "min_value": 0,
            "max_value": 20
        },
        "response_format": {
            "type": "select",
            "default": "text",
            "description": "Response format",
            "options": ["text", "json"]
        },
        "model_id": {
            "type": "string",
            "default": "gpt-4",
            "description": "Primary model identifier"
        },
        "backup_models": {
            "type": "list",
            "default": [],
            "description": "Fallback model list"
        },
        "timeout": {
            "type": "integer",
            "default": 30,
            "description": "Request timeout in seconds",
            "min_value": 1,
            "max_value": 300
        },
        "max_retries": {
            "type": "integer",
            "default": 3,
            "description": "Maximum retry attempts",
            "min_value": 0,
            "max_value": 10
        },
        "retry_delay": {
            "type": "float",
            "default": 1.0,
            "description": "Delay between retries",
            "min_value": 0.1,
            "max_value": 10.0
        },
        "rate_limit": {
            "type": "integer",
            "default": 60,
            "description": "Requests per minute",
            "min_value": 1,
            "max_value": 1000
        },
        "concurrent_requests": {
            "type": "integer",
            "default": 5,
            "description": "Concurrent API requests",
            "min_value": 1,
            "max_value": 50
        },
        "model_rotation": {
            "type": "boolean",
            "default": False,
            "description": "Rotate between models"
        }
    },
    "quality_diversity": {
        "feature_dimensions": {
            "type": "list",
            "default": None,
            "description": "Feature dimensions for behavior"
        },
        "feature_bins": {
            "type": "integer",
            "default": 10,
            "description": "Bins per feature dimension",
            "min_value": 2,
            "max_value": 100
        },
        "archive_size": {
            "type": "integer",
            "default": 100,
            "description": "Maximum archive size",
            "min_value": 1,
            "max_value": 10000
        },
        "behavior_dimensions": {
            "type": "list",
            "default": [],
            "description": "Specific behavior dimensions"
        },
        "diversity_metric": {
            "type": "select",
            "default": "edit_distance",
            "description": "Diversity measurement metric",
            "options": ["edit_distance", "semantic", "behavioral"]
        },
        "diversity_reference_size": {
            "type": "integer",
            "default": 20,
            "description": "Reference set size for diversity",
            "min_value": 1,
            "max_value": 1000
        },
        "adaptive_feature_dimensions": {
            "type": "boolean",
            "default": True,
            "description": "Dynamically adjust features"
        },
        "double_selection": {
            "type": "boolean",
            "default": True,
            "description": "Different programs for performance vs inspiration"
        },
        "qd_algorithm": {
            "type": "select",
            "default": "MAP-Elites",
            "description": "QD algorithm to use",
            "options": ["MAP-Elites", "CVT-MAP-Elites", "CMA-ME"]
        },
        "novelty_threshold": {
            "type": "float",
            "default": 0.1,
            "description": "Minimum novelty for archive",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "behavior_descriptor_type": {
            "type": "select",
            "default": "hand_crafted",
            "description": "Type of behavior descriptor",
            "options": ["hand_crafted", "learned"]
        },
        "archive_learning_rate": {
            "type": "float",
            "default": 0.1,
            "description": "Archive adaptation rate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "quality_threshold": {
            "type": "float",
            "default": 0.0,
            "description": "Minimum quality for archive",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "diversity_weight": {
            "type": "float",
            "default": 0.5,
            "description": "Weight of diversity vs quality",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "behavior_space": {
            "type": "string",
            "default": "auto",
            "description": "Behavior space definition"
        },
        "distance_metric": {
            "type": "select",
            "default": "euclidean",
            "description": "Distance calculation method",
            "options": ["euclidean", "manhattan", "cosine"]
        },
        "archive_update_freq": {
            "type": "integer",
            "default": 1,
            "description": "Archive update frequency",
            "min_value": 1,
            "max_value": 100
        },
        "exploration_bonus": {
            "type": "float",
            "default": 0.1,
            "description": "Bonus for exploration",
            "min_value": 0.0,
            "max_value": 2.0
        },
        "pareto_layers": {
            "type": "integer",
            "default": 3,
            "description": "Number of Pareto layers",
            "min_value": 1,
            "max_value": 10
        }
    },
    "multi_objective": {
        "objectives": {
            "type": "list",
            "default": None,
            "description": "List of objectives to optimize"
        },
        "objective_weights": {
            "type": "list",
            "default": [],
            "description": "Weights for each objective"
        },
        "pareto_front_size": {
            "type": "integer",
            "default": 50,
            "description": "Maximum Pareto front size",
            "min_value": 1,
            "max_value": 1000
        },
        "dominance_metric": {
            "type": "select",
            "default": "pareto",
            "description": "Dominance metric",
            "options": ["pareto", "epsilon"]
        },
        "constraint_handling": {
            "type": "select",
            "default": "penalty",
            "description": "Constraint handling method",
            "options": ["penalty", "repair", "death_penalty"]
        },
        "reference_point": {
            "type": "list",
            "default": [],
            "description": "Reference point for hypervolume"
        },
        "crowding_distance": {
            "type": "boolean",
            "default": True,
            "description": "Use crowding distance"
        },
        "epsilon_dominance": {
            "type": "float",
            "default": 0.01,
            "description": "Epsilon for epsilon-dominance",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "decomposition_method": {
            "type": "select",
            "default": "weighted_sum",
            "description": "Objective decomposition method",
            "options": ["weighted_sum", "tchebycheff", "boundary_intersection"]
        },
        "scalarization_function": {
            "type": "string",
            "default": "weighted_sum",
            "description": "Scalarization function"
        },
        "dominance_type": {
            "type": "select",
            "default": "standard",
            "description": "Dominance relation type",
            "options": ["standard", "epsilon", "fuzzy"]
        },
        "epsilon_values": {
            "type": "list",
            "default": [],
            "description": "Epsilon values for epsilon-dominance"
        },
        "scalarization": {
            "type": "select",
            "default": "weighted_sum",
            "description": "Scalarization method",
            "options": ["weighted_sum", "tchebycheff", "pbi"]
        },
        "constraint_tolerance": {
            "type": "float",
            "default": 0.01,
            "description": "Tolerance for constraints",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "hypervolume_ref": {
            "type": "list",
            "default": [],
            "description": "Hypervolume reference point"
        }
    },
    "adversarial": {
        "attack_model_config": {
            "type": "dict",
            "default": None,
            "description": "Attack model configuration"
        },
        "defense_model_config": {
            "type": "dict",
            "default": None,
            "description": "Defense model configuration"
        },
        "adversarial_rounds": {
            "type": "integer",
            "default": 5,
            "description": "Number of adversarial rounds",
            "min_value": 1,
            "max_value": 100
        },
        "attack_strength": {
            "type": "float",
            "default": 0.5,
            "description": "Strength of attacks",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "defense_strategy": {
            "type": "select",
            "default": "reactive",
            "description": "Defense strategy",
            "options": ["reactive", "proactive", "adaptive"]
        },
        "coevolutionary_approach": {
            "type": "boolean",
            "default": False,
            "description": "Use co-evolution"
        },
        "red_team_models": {
            "type": "list",
            "default": [],
            "description": "Red team model IDs"
        },
        "blue_team_models": {
            "type": "list",
            "default": [],
            "description": "Blue team model IDs"
        },
        "red_team_sample_size": {
            "type": "integer",
            "default": 3,
            "description": "Red team models to sample",
            "min_value": 1,
            "max_value": 20
        },
        "blue_team_sample_size": {
            "type": "integer",
            "default": 3,
            "description": "Blue team models to sample",
            "min_value": 1,
            "max_value": 20
        },
        "adversarial_temperature": {
            "type": "float",
            "default": 0.8,
            "description": "Temperature for adversarial generation",
            "min_value": 0.0,
            "max_value": 2.0
        },
        "attack_diversity": {
            "type": "boolean",
            "default": True,
            "description": "Encourage diverse attacks"
        },
        "defense_strength": {
            "type": "float",
            "default": 1.0,
            "description": "Strength of defense mechanisms",
            "min_value": 0.1,
            "max_value": 2.0
        },
        "adversarial_budget": {
            "type": "integer",
            "default": 100,
            "description": "Budget for adversarial operations",
            "min_value": 1,
            "max_value": 1000
        },
        "attack_types": {
            "type": "list",
            "default": [],
            "description": "Types of attacks to use"
        },
        "defense_strategies": {
            "type": "list",
            "default": [],
            "description": "Defense strategies to employ"
        },
        "robustness_metric": {
            "type": "string",
            "default": "accuracy",
            "description": "Metric for robustness evaluation"
        },
        "perturbation_bound": {
            "type": "float",
            "default": 0.1,
            "description": "Maximum perturbation allowed",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "gradient_masking": {
            "type": "boolean",
            "default": False,
            "description": "Use gradient masking"
        },
        "ensemble_defense": {
            "type": "boolean",
            "default": True,
            "description": "Use ensemble for defense"
        }
    },
    "island_model": {
        "num_islands": {
            "type": "integer",
            "default": 5,
            "description": "Number of islands",
            "min_value": 1,
            "max_value": 100
        },
        "migration_interval": {
            "type": "integer",
            "default": 10,
            "description": "Generations between migrations",
            "min_value": 1,
            "max_value": 1000
        },
        "migration_rate": {
            "type": "float",
            "default": 0.1,
            "description": "Proportion to migrate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "migration_topology": {
            "type": "select",
            "default": "ring",
            "description": "Migration topology",
            "options": ["ring", "fully_connected", "random", "star"]
        },
        "ring_topology": {
            "type": "boolean",
            "default": True,
            "description": "Use ring topology"
        },
        "controlled_gene_flow": {
            "type": "boolean",
            "default": True,
            "description": "Control gene flow"
        },
        "island_diversity_metric": {
            "type": "string",
            "default": "edit_distance",
            "description": "Island diversity metric"
        },
        "migration_selection": {
            "type": "select",
            "default": "best",
            "description": "Migrant selection method",
            "options": ["best", "random", "diverse", "tournament"]
        },
        "island_initialization": {
            "type": "select",
            "default": "random",
            "description": "Island initialization method",
            "options": ["random", "clustered", "diverse"]
        },
        "island_specialization": {
            "type": "boolean",
            "default": False,
            "description": "Allow island specialization"
        },
        "migration_size": {
            "type": "integer",
            "default": 5,
            "description": "Number of individuals to migrate",
            "min_value": 1,
            "max_value": 50
        },
        "migration_policy": {
            "type": "select",
            "default": "best",
            "description": "Migration selection policy",
            "options": ["best", "random", "diverse"]
        },
        "replacement_policy": {
            "type": "select",
            "default": "worst",
            "description": "Replacement policy",
            "options": ["worst", "random", "similar"]
        },
        "island_sizes": {
            "type": "list",
            "default": [],
            "description": "Custom sizes for each island"
        },
        "heterogeneous_islands": {
            "type": "boolean",
            "default": False,
            "description": "Use different algorithms per island"
        },
        "synchronous_migration": {
            "type": "boolean",
            "default": True,
            "description": "Synchronize migration timing"
        },
        "adaptive_migration": {
            "type": "boolean",
            "default": False,
            "description": "Adapt migration parameters"
        }
    },
    "selection": {
        "elite_ratio": {
            "type": "float",
            "default": 0.1,
            "description": "Proportion of elites",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "exploration_ratio": {
            "type": "float",
            "default": 0.2,
            "description": "Proportion for exploration",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "exploitation_ratio": {
            "type": "float",
            "default": 0.7,
            "description": "Proportion for exploitation",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "multi_strategy_sampling": {
            "type": "boolean",
            "default": True,
            "description": "Use multiple sampling strategies"
        },
        "selection_pressure": {
            "type": "float",
            "default": 2.0,
            "description": "Selection pressure",
            "min_value": 1.0,
            "max_value": 10.0
        },
        "tournament_size": {
            "type": "integer",
            "default": 3,
            "description": "Tournament size",
            "min_value": 2,
            "max_value": 20
        },
        "crossover_rate": {
            "type": "float",
            "default": 0.8,
            "description": "Crossover rate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "mutation_rate": {
            "type": "float",
            "default": 0.1,
            "description": "Mutation rate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "elitism_count": {
            "type": "integer",
            "default": 2,
            "description": "Number of elites to preserve",
            "min_value": 0,
            "max_value": 100
        },
        "selection_method": {
            "type": "select",
            "default": "tournament",
            "description": "Selection method",
            "options": ["tournament", "roulette", "rank", "stochastic"]
        },
        "reproduction_method": {
            "type": "select",
            "default": "both",
            "description": "Reproduction method",
            "options": ["crossover", "mutation", "both"]
        },
        "parent_selection": {
            "type": "select",
            "default": "fitness",
            "description": "Parent selection method",
            "options": ["fitness", "random", "diverse"]
        },
        "random_ratio": {
            "type": "float",
            "default": 0.2,
            "description": "Ratio for random selection",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "survivor_selection": {
            "type": "select",
            "default": "generational",
            "description": "Survivor selection method",
            "options": ["generational", "steady_state"]
        },
        "replacement_rate": {
            "type": "float",
            "default": 1.0,
            "description": "Population replacement rate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "selection_pressure_decay": {
            "type": "float",
            "default": 0.0,
            "description": "Selection pressure decay rate",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "diversity_selection": {
            "type": "boolean",
            "default": False,
            "description": "Include diversity in selection"
        },
        "age_based_selection": {
            "type": "boolean",
            "default": False,
            "description": "Consider individual age"
        }
    },
    "evaluation": {
        "cascade_evaluation": {
            "type": "boolean",
            "default": True,
            "description": "Use cascade evaluation"
        },
        "cascade_thresholds": {
            "type": "list",
            "default": [0.5, 0.75, 0.9],
            "description": "Thresholds for cascade levels"
        },
        "parallel_evaluations": {
            "type": "integer",
            "default": 4,
            "description": "Number of parallel workers",
            "min_value": 1,
            "max_value": 100
        },
        "evaluator_timeout": {
            "type": "integer",
            "default": 300,
            "description": "Evaluation timeout (seconds)",
            "min_value": 1,
            "max_value": 3600
        },
        "max_retries_eval": {
            "type": "integer",
            "default": 3,
            "description": "Max evaluation retries",
            "min_value": 0,
            "max_value": 10
        },
        "use_llm_feedback": {
            "type": "boolean",
            "default": False,
            "description": "Use LLM-based feedback"
        },
        "llm_feedback_weight": {
            "type": "float",
            "default": 0.1,
            "description": "Weight for LLM feedback",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "evaluator_models": {
            "type": "list",
            "default": None,
            "description": "Evaluator model configurations"
        },
        "evaluator_system_message": {
            "type": "string",
            "default": "",
            "description": "System prompt for evaluator"
        },
        "ensemble_size": {
            "type": "integer",
            "default": 3,
            "description": "Number of evaluators in ensemble",
            "min_value": 1,
            "max_value": 20
        },
        "consensus_threshold": {
            "type": "float",
            "default": 0.7,
            "description": "Threshold for consensus",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "evaluation_criteria": {
            "type": "list",
            "default": [],
            "description": "List of evaluation criteria"
        },
        "custom_evaluator": {
            "type": "string",
            "default": None,
            "description": "Custom evaluation function"
        },
        "evaluation_batch_size": {
            "type": "integer",
            "default": 10,
            "description": "Batch size for evaluations",
            "min_value": 1,
            "max_value": 1000
        },
        "cache_evaluations": {
            "type": "boolean",
            "default": True,
            "description": "Cache evaluation results"
        },
        "cache_size": {
            "type": "integer",
            "default": 1000,
            "description": "Maximum cache size",
            "min_value": 100,
            "max_value": 10000
        },
        "evaluation_noise": {
            "type": "float",
            "default": 0.0,
            "description": "Noise level in evaluations",
            "min_value": 0.0,
            "max_value": 0.5
        },
        "fitness_scaling": {
            "type": "select",
            "default": "linear",
            "description": "Fitness scaling method",
            "options": ["linear", "exponential", "logarithmic"]
        },
        "normalization": {
            "type": "boolean",
            "default": True,
            "description": "Normalize fitness values"
        },
        "multi_criteria_eval": {
            "type": "boolean",
            "default": False,
            "description": "Multi-criteria evaluation"
        },
        "evaluation_budget": {
            "type": "integer",
            "default": 10000,
            "description": "Total evaluation budget",
            "min_value": 1,
            "max_value": 100000
        },
        "incremental_eval": {
            "type": "boolean",
            "default": False,
            "description": "Incremental evaluation"
        },
        "surrogate_model": {
            "type": "boolean",
            "default": False,
            "description": "Use surrogate model"
        },
        "active_learning": {
            "type": "boolean",
            "default": False,
            "description": "Active learning for evaluation"
        },
        "uncertainty_sampling": {
            "type": "boolean",
            "default": False,
            "description": "Sample uncertain regions"
        }
    },
    "prompt_engineering": {
        "prompt_template": {
            "type": "string",
            "default": "default",
            "description": "Base prompt template"
        },
        "system_prompt": {
            "type": "string",
            "default": "",
            "description": "System-level prompt"
        },
        "context_length": {
            "type": "integer",
            "default": 2000,
            "description": "Maximum context length",
            "min_value": 100,
            "max_value": 8000
        },
        "prompt_optimization": {
            "type": "boolean",
            "default": True,
            "description": "Optimize prompts during evolution"
        },
        "template_stochasticity": {
            "type": "boolean",
            "default": True,
            "description": "Use stochastic templates"
        },
        "meta_prompting": {
            "type": "boolean",
            "default": False,
            "description": "Use meta-prompting techniques"
        },
        "few_shot_examples": {
            "type": "integer",
            "default": 3,
            "description": "Number of few-shot examples",
            "min_value": 0,
            "max_value": 20
        },
        "chain_of_thought": {
            "type": "boolean",
            "default": True,
            "description": "Use chain-of-thought prompting"
        },
        "self_consistency": {
            "type": "boolean",
            "default": False,
            "description": "Use self-consistency decoding"
        },
        "prompt_ensembling": {
            "type": "boolean",
            "default": False,
            "description": "Ensemble multiple prompts"
        },
        "dynamic_prompting": {
            "type": "boolean",
            "default": False,
            "description": "Dynamically adjust prompts"
        },
        "prompt_compression": {
            "type": "boolean",
            "default": False,
            "description": "Compress long prompts"
        }
    },
    "artifact_management": {
        "enable_artifacts": {
            "type": "boolean",
            "default": True,
            "description": "Enable artifact generation"
        },
        "artifact_types": {
            "type": "list",
            "default": ["code", "text"],
            "description": "Types of artifacts to generate"
        },
        "max_artifact_size": {
            "type": "integer",
            "default": 20480,
            "description": "Maximum artifact size in bytes",
            "min_value": 1024,
            "max_value": 1048576
        },
        "artifact_validation": {
            "type": "boolean",
            "default": True,
            "description": "Validate generated artifacts"
        },
        "artifact_compression": {
            "type": "boolean",
            "default": False,
            "description": "Compress artifacts"
        },
        "artifact_versioning": {
            "type": "boolean",
            "default": True,
            "description": "Version control for artifacts"
        },
        "artifact_metadata": {
            "type": "boolean",
            "default": True,
            "description": "Include metadata with artifacts"
        },
        "artifact_cleanup": {
            "type": "boolean",
            "default": True,
            "description": "Clean up old artifacts"
        },
        "artifact_storage": {
            "type": "select",
            "default": "memory",
            "description": "Artifact storage location",
            "options": ["memory", "disk", "cloud"]
        },
        "artifact_encryption": {
            "type": "boolean",
            "default": False,
            "description": "Encrypt sensitive artifacts"
        }
    },
    "resource_management": {
        "memory_limit_mb": {
            "type": "integer",
            "default": 4096,
            "description": "Memory limit in MB",
            "min_value": 512,
            "max_value": 32768
        },
        "cpu_limit": {
            "type": "float",
            "default": 0.8,
            "description": "CPU usage limit (fraction)",
            "min_value": 0.1,
            "max_value": 1.0
        },
        "max_time": {
            "type": "integer",
            "default": 1800,
            "description": "Maximum execution time in seconds",
            "min_value": 60,
            "max_value": 7200
        },
        "disk_limit_mb": {
            "type": "integer",
            "default": 1024,
            "description": "Disk usage limit in MB",
            "min_value": 100,
            "max_value": 10240
        },
        "network_limit_mbps": {
            "type": "integer",
            "default": 100,
            "description": "Network bandwidth limit",
            "min_value": 1,
            "max_value": 1000
        },
        "api_call_limit": {
            "type": "integer",
            "default": 1000,
            "description": "Maximum API calls",
            "min_value": 10,
            "max_value": 10000
        },
        "token_limit": {
            "type": "integer",
            "default": 100000,
            "description": "Maximum tokens",
            "min_value": 1000,
            "max_value": 1000000
        },
        "cost_limit_usd": {
            "type": "float",
            "default": 10.0,
            "description": "Maximum cost in USD",
            "min_value": 0.01,
            "max_value": 1000.0
        },
        "resource_monitoring": {
            "type": "boolean",
            "default": True,
            "description": "Monitor resource usage"
        },
        "auto_scaling": {
            "type": "boolean",
            "default": False,
            "description": "Auto-scale resources"
        },
        "checkpoint_interval": {
            "type": "integer",
            "default": 10,
            "description": "Generations between checkpoints",
            "min_value": 1,
            "max_value": 1000
        }
    },
    "database_storage": {
        "db_path": {
            "type": "string",
            "default": "./openevolve.db",
            "description": "Database file path"
        },
        "db_type": {
            "type": "select",
            "default": "sqlite",
            "description": "Database type",
            "options": ["sqlite", "postgresql", "mongodb"]
        },
        "connection_string": {
            "type": "string",
            "default": "",
            "description": "Database connection string"
        },
        "max_connections": {
            "type": "integer",
            "default": 10,
            "description": "Maximum database connections",
            "min_value": 1,
            "max_value": 100
        },
        "connection_timeout": {
            "type": "integer",
            "default": 30,
            "description": "Connection timeout in seconds",
            "min_value": 1,
            "max_value": 60
        },
        "query_timeout": {
            "type": "integer",
            "default": 60,
            "description": "Query timeout in seconds",
            "min_value": 1,
            "max_value": 300
        },
        "batch_size": {
            "type": "integer",
            "default": 1000,
            "description": "Batch size for operations",
            "min_value": 1,
            "max_value": 10000
        },
        "compression": {
            "type": "boolean",
            "default": True,
            "description": "Compress stored data"
        },
        "encryption": {
            "type": "boolean",
            "default": False,
            "description": "Encrypt stored data"
        },
        "backup_enabled": {
            "type": "boolean",
            "default": True,
            "description": "Enable automatic backups"
        }
    },
    "evolution_tracing": {
        "trace_enabled": {
            "type": "boolean",
            "default": False,
            "description": "Enable evolution tracing"
        },
        "trace_level": {
            "type": "select",
            "default": "basic",
            "description": "Level of tracing detail",
            "options": ["basic", "detailed", "full"]
        },
        "trace_format": {
            "type": "select",
            "default": "json",
            "description": "Trace output format",
            "options": ["json", "csv", "binary"]
        },
        "trace_file": {
            "type": "string",
            "default": "./trace.log",
            "description": "Trace output file"
        },
        "trace_compression": {
            "type": "boolean",
            "default": True,
            "description": "Compress trace files"
        },
        "trace_rotation": {
            "type": "boolean",
            "default": True,
            "description": "Rotate trace files"
        },
        "max_trace_size_mb": {
            "type": "integer",
            "default": 100,
            "description": "Maximum trace file size",
            "min_value": 1,
            "max_value": 1024
        },
        "trace_buffer_size": {
            "type": "integer",
            "default": 1000,
            "description": "Trace buffer size",
            "min_value": 100,
            "max_value": 10000
        },
        "real_time_tracing": {
            "type": "boolean",
            "default": False,
            "description": "Real-time trace streaming"
        },
        "trace_sampling": {
            "type": "float",
            "default": 1.0,
            "description": "Sampling rate for tracing",
            "min_value": 0.01,
            "max_value": 1.0
        },
        "include_population": {
            "type": "boolean",
            "default": False,
            "description": "Include population in trace"
        },
        "include_fitness": {
            "type": "boolean",
            "default": True,
            "description": "Include fitness in trace"
        }
    },
    "early_stopping": {
        "early_stopping": {
            "type": "boolean",
            "default": False,
            "description": "Enable early stopping"
        },
        "early_stopping_patience": {
            "type": "integer",
            "default": 10,
            "description": "Patience for early stopping",
            "min_value": 1,
            "max_value": 100
        },
        "min_improvement": {
            "type": "float",
            "default": 0.001,
            "description": "Minimum improvement threshold",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "improvement_window": {
            "type": "integer",
            "default": 5,
            "description": "Window for improvement calculation",
            "min_value": 1,
            "max_value": 50
        },
        "plateau_threshold": {
            "type": "integer",
            "default": 20,
            "description": "Generations to consider plateau",
            "min_value": 1,
            "max_value": 100
        },
        "convergence_check": {
            "type": "boolean",
            "default": True,
            "description": "Check for convergence"
        },
        "diversity_threshold": {
            "type": "float",
            "default": 0.01,
            "description": "Minimum diversity threshold",
            "min_value": 0.0,
            "max_value": 1.0
        },
        "stagnation_limit": {
            "type": "integer",
            "default": 50,
            "description": "Maximum stagnation generations",
            "min_value": 1,
            "max_value": 100
        },
        "adaptive_stopping": {
            "type": "boolean",
            "default": False,
            "description": "Adaptive stopping criteria"
        }
    },
    "distributed_processing": {
        "distributed": {
            "type": "boolean",
            "default": False,
            "description": "Enable distributed processing"
        },
        "num_workers": {
            "type": "integer",
            "default": 4,
            "description": "Number of worker processes",
            "min_value": 1,
            "max_value": 100
        },
        "worker_timeout": {
            "type": "integer",
            "default": 120,
            "description": "Worker timeout in seconds",
            "min_value": 10,
            "max_value": 600
        },
        "load_balancing": {
            "type": "select",
            "default": "round_robin",
            "description": "Load balancing strategy",
            "options": ["round_robin", "least_loaded", "random"]
        },
        "fault_tolerance": {
            "type": "boolean",
            "default": True,
            "description": "Enable fault tolerance"
        },
        "worker_restart": {
            "type": "boolean",
            "default": True,
            "description": "Auto-restart failed workers"
        },
        "communication_backend": {
            "type": "select",
            "default": "local",
            "description": "Communication backend",
            "options": ["local", "redis", "rabbitmq"]
        },
        "message_compression": {
            "type": "boolean",
            "default": True,
            "description": "Compress messages"
        },
        "heartbeat_interval": {
            "type": "integer",
            "default": 10,
            "description": "Heartbeat interval in seconds",
            "min_value": 1,
            "max_value": 60
        },
        "cluster_scaling": {
            "type": "boolean",
            "default": False,
            "description": "Auto-scale cluster"
        }
    },
    "advanced_research": {
        "novelty_search": {
            "type": "boolean",
            "default": False,
            "description": "Enable novelty search"
        },
        "curiosity_driven": {
            "type": "boolean",
            "default": False,
            "description": "Curiosity-driven exploration"
        },
        "meta_learning": {
            "type": "boolean",
            "default": False,
            "description": "Enable meta-learning"
        },
        "transfer_learning": {
            "type": "boolean",
            "default": False,
            "description": "Transfer from previous runs"
        },
        "continual_learning": {
            "type": "boolean",
            "default": False,
            "description": "Continual learning mode"
        },
        "few_shot_adaptation": {
            "type": "boolean",
            "default": False,
            "description": "Few-shot adaptation"
        },
        "zero_shot_transfer": {
            "type": "boolean",
            "default": False,
            "description": "Zero-shot transfer"
        },
        "domain_adaptation": {
            "type": "boolean",
            "default": False,
            "description": "Domain adaptation"
        },
        "multi_task_learning": {
            "type": "boolean",
            "default": False,
            "description": "Multi-task learning"
        },
        "lifelong_learning": {
            "type": "boolean",
            "default": False,
            "description": "Lifelong learning"
        },
        "neural_architecture_search": {
            "type": "boolean",
            "default": False,
            "description": "NAS integration"
        },
        "hyperparameter_optimization": {
            "type": "boolean",
            "default": False,
            "description": "HPO integration"
        },
        "automated_ml": {
            "type": "boolean",
            "default": False,
            "description": "AutoML features"
        },
        "explainable_ai": {
            "type": "boolean",
            "default": False,
            "description": "XAI integration"
        },
        "federated_learning": {
            "type": "boolean",
            "default": False,
            "description": "Federated learning"
        },
        "differential_privacy": {
            "type": "boolean",
            "default": False,
            "description": "Privacy preservation"
        },
        "quantum_computing": {
            "type": "boolean",
            "default": False,
            "description": "Quantum computing support"
        },
        "neuromorphic_computing": {
            "type": "boolean",
            "default": False,
            "description": "Neuromorphic support"
        },
        "edge_computing": {
            "type": "boolean",
            "default": False,
            "description": "Edge deployment"
        },
        "green_ai": {
            "type": "boolean",
            "default": False,
            "description": "Energy-efficient AI"
        }
    },
    "custom_requirements": {
        "custom_fitness": {
            "type": "string",
            "default": "",
            "description": "Custom fitness function code"
        },
        "custom_operators": {
            "type": "list",
            "default": [],
            "description": "Custom genetic operators"
        },
        "custom_constraints": {
            "type": "list",
            "default": [],
            "description": "Custom constraint functions"
        },
        "domain_knowledge": {
            "type": "string",
            "default": "",
            "description": "Domain-specific knowledge"
        },
        "expert_rules": {
            "type": "list",
            "default": [],
            "description": "Expert-defined rules"
        },
        "business_logic": {
            "type": "string",
            "default": "",
            "description": "Business logic constraints"
        },
        "regulatory_compliance": {
            "type": "list",
            "default": [],
            "description": "Compliance requirements"
        },
        "ethical_guidelines": {
            "type": "list",
            "default": [],
            "description": "Ethical AI guidelines"
        }
    },
    "ui_visualization": {
        "enable_visualization": {
            "type": "boolean",
            "default": True,
            "description": "Enable visualizations"
        },
        "plot_frequency": {
            "type": "integer",
            "default": 10,
            "description": "Plotting frequency",
            "min_value": 1,
            "max_value": 100
        },
        "plot_types": {
            "type": "list",
            "default": ["fitness", "diversity"],
            "description": "Types of plots to generate"
        },
        "interactive_plots": {
            "type": "boolean",
            "default": True,
            "description": "Interactive visualizations"
        },
        "real_time_updates": {
            "type": "boolean",
            "default": False,
            "description": "Real-time plot updates"
        },
        "export_plots": {
            "type": "boolean",
            "default": True,
            "description": "Export plots to files"
        },
        "plot_format": {
            "type": "select",
            "default": "png",
            "description": "Plot export format",
            "options": ["png", "svg", "pdf"]
        },
        "dashboard_enabled": {
            "type": "boolean",
            "default": True,
            "description": "Enable monitoring dashboard"
        }
    },
    "experimental": {
        "experimental_features": {
            "type": "boolean",
            "default": False,
            "description": "Enable experimental features"
        },
        "beta_algorithms": {
            "type": "boolean",
            "default": False,
            "description": "Use beta algorithms"
        },
        "research_mode": {
            "type": "boolean",
            "default": False,
            "description": "Research mode settings"
        },
        "debug_mode": {
            "type": "boolean",
            "default": False,
            "description": "Debug mode"
        },
        "profiling_enabled": {
            "type": "boolean",
            "default": False,
            "description": "Performance profiling"
        },
        "memory_profiling": {
            "type": "boolean",
            "default": False,
            "description": "Memory usage profiling"
        },
        "experimental_logging": {
            "type": "boolean",
            "default": False,
            "description": "Experimental logging"
        }
    }
}
