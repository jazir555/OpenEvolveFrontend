"""
Environment Variable Mappings

Maps all 102+ configuration parameters to their environment variable names and types.
"""

from typing import Dict, Tuple, Type, Any

# =============================================================================
# ENVIRONMENT VARIABLE MAPPINGS
# =============================================================================
# Format: 'parameter_name': ('EVOLVE_PARAMETER_NAME', type)
#
# Usage:
#   export EVOLVE_MAX_ITERATIONS=100
#   export EVOLVE_ENABLE_PLANNING=true
#   export EVOLVE_TEMPERATURE=0.7
# =============================================================================

ENV_MAPPINGS: Dict[str, Tuple[str, Type]] = {
    # =========================================================================
    # CORE EVOLUTION PARAMETERS
    # =========================================================================
    'max_iterations': ('EVOLVE_MAX_ITERATIONS', int),
    'population_size': ('EVOLVE_POPULATION_SIZE', int),
    'generations': ('EVOLVE_GENERATIONS', int),
    'evolution_mode': ('EVOLVE_MODE', str),
    'domain': ('EVOLVE_DOMAIN', str),
    'seed': ('EVOLVE_SEED', int),

    # =========================================================================
    # LLM PARAMETERS
    # =========================================================================
    'model_id': ('EVOLVE_MODEL', str),
    'model_name': ('EVOLVE_MODEL_NAME', str),
    'temperature': ('EVOLVE_TEMPERATURE', float),
    'max_tokens': ('EVOLVE_MAX_TOKENS', int),
    'top_p': ('EVOLVE_TOP_P', float),
    'top_k': ('EVOLVE_TOP_K', int),
    'frequency_penalty': ('EVOLVE_FREQUENCY_PENALTY', float),
    'presence_penalty': ('EVOLVE_PRESENCE_PENALTY', float),

    # API Configuration
    'api_key': ('EVOLVE_API_KEY', str),
    'api_base': ('EVOLVE_API_BASE', str),
    'api_version': ('EVOLVE_API_VERSION', str),
    'api_type': ('EVOLVE_API_TYPE', str),  # 'openai', 'azure', etc.

    # =========================================================================
    # PES (PRIESTLEY, EMERT, SMITH) PARAMETERS
    # =========================================================================
    'enable_planning': ('EVOLVE_ENABLE_PLANNING', bool),
    'enable_memory': ('EVOLVE_ENABLE_MEMORY', bool),
    'plan_temperature': ('EVOLVE_PLAN_TEMPERATURE', float),
    'plan_max_tokens': ('EVOLVE_PLAN_MAX_TOKENS', int),
    'memory_type': ('EVOLVE_MEMORY_TYPE', str),  # 'episodic', 'semantic', 'working'

    # Planning Parameters
    'planner_model': ('EVOLVE_PLANNER_MODEL', str),
    'planner_temperature': ('EVOLVE_PLANNER_TEMPERATURE', float),
    'planner_max_retries': ('EVOLVE_PLANNER_MAX_RETRIES', int),
    'planner_timeout': ('EVOLVE_PLANNER_TIMEOUT', int),

    # Memory Parameters
    'memory_capacity': ('EVOLVE_MEMORY_CAPACITY', int),
    'memory_retention': ('EVOLVE_MEMORY_RETENTION', float),
    'memory_decay': ('EVOLVE_MEMORY_DECAY', float),

    # =========================================================================
    # QUALITY DIVERSITY (QD) PARAMETERS
    # =========================================================================
    'qd_enabled': ('EVOLVE_QD_ENABLED', bool),
    'qd_algorithm': ('EVOLVE_QD_ALGORITHM', str),  # 'map_elites', 'cvt_map_elites', etc.
    'qd_grid_resolution': ('EVOLVE_QD_GRID_RESOLUTION', int),
    'qd_archive_size': ('EVOLVE_QD_ARCHIVE_SIZE', int),
    'qd_feature_dimensions': ('EVOLVE_QD_FEATURE_DIMENSIONS', int),
    'qd_novelty_threshold': ('EVOLVE_QD_NOVELTY_THRESHOLD', float),

    # =========================================================================
    # GENETIC ALGORITHM PARAMETERS
    # =========================================================================
    'mutation_rate': ('EVOLVE_MUTATION_RATE', float),
    'crossover_rate': ('EVOLVE_CROSSOVER_RATE', float),
    'elitism_count': ('EVOLVE_ELITISM_COUNT', int),
    'tournament_size': ('EVOLVE_TOURNAMENT_SIZE', int),
    'selection_method': ('EVOLVE_SELECTION_METHOD', str),  # 'tournament', 'roulette', 'rank'

    # =========================================================================
    # ADVERSARIAL PARAMETERS
    # =========================================================================
    'adversarial_rounds': ('EVOLVE_ADVERSARIAL_ROUNDS', int),
    'attack_strength': ('EVOLVE_ATTACK_STRENGTH', float),
    'defense_strategy': ('EVOLVE_DEFENSE_STRATEGY', str),  # 'reactive', 'proactive', 'hybrid'
    'adversarial_mode': ('EVOLVE_ADVERSARIAL_MODE', str),  # 'attack', 'defend', 'both'

    # =========================================================================
    # GAUNTLET PARAMETERS
    # =========================================================================
    'enable_gauntlet': ('EVOLVE_ENABLE_GAUNTLET', bool),
    'gauntlet_rounds': ('EVOLVE_GAUNTLET_ROUNDS', int),
    'gauntlet_timeout': ('EVOLVE_GAUNTLET_TIMEOUT', int),
    'gauntlet_strictness': ('EVOLVE_GAUNTLET_STRICTNESS', float),  # 0.0 to 1.0

    # =========================================================================
    # LOGGING PARAMETERS
    # =========================================================================
    'log_level': ('EVOLVE_LOG_LEVEL', str),  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_file': ('EVOLVE_LOG_FILE', str),
    'log_format': ('EVOLVE_LOG_FORMAT', str),  # 'json', 'text'
    'verbose': ('EVOLVE_VERBOSE', bool),
    'debug': ('EVOLVE_DEBUG', bool),

    # =========================================================================
    # OUTPUT PARAMETERS
    # =========================================================================
    'output_dir': ('EVOLVE_OUTPUT_DIR', str),
    'save_intermediate_results': ('EVOLVE_SAVE_INTERMEDIATE', bool),
    'save_final_results': ('EVOLVE_SAVE_FINAL', bool),
    'save_frequency': ('EVOLVE_SAVE_FREQUENCY', int),
    'result_format': ('EVOLVE_RESULT_FORMAT', str),  # 'json', 'yaml', 'csv'

    # =========================================================================
    # PERFORMANCE PARAMETERS
    # =========================================================================
    'parallel_workers': ('EVOLVE_PARALLEL_WORKERS', int),
    'batch_size': ('EVOLVE_BATCH_SIZE', int),
    'cache_enabled': ('EVOLVE_CACHE_ENABLED', bool),
    'cache_size': ('EVOLVE_CACHE_SIZE', int),
    'timeout': ('EVOLVE_TIMEOUT', int),
    'max_retries': ('EVOLVE_MAX_RETRIES', int),
    'retry_delay': ('EVOLVE_RETRY_DELAY', float),

    # =========================================================================
    # VALIDATION PARAMETERS
    # =========================================================================
    'validate_outputs': ('EVOLVE_VALIDATE_OUTPUTS', bool),
    'validation_frequency': ('EVOLVE_VALIDATION_FREQUENCY', int),
    'validation_strictness': ('EVOLVE_VALIDATION_STRICTNESS', str),  # 'strict', 'moderate', 'lenient'

    # =========================================================================
    # STOPPING CONDITIONS
    # =========================================================================
    'early_stopping': ('EVOLVE_EARLY_STOPPING', bool),
    'early_stopping_patience': ('EVOLVE_EARLY_STOPPING_PATIENCE', int),
    'early_stopping_threshold': ('EVOLVE_EARLY_STOPPING_THRESHOLD', float),
    'min_improvement': ('EVOLVE_MIN_IMPROVEMENT', float),
    'target_score': ('EVOLVE_TARGET_SCORE', float),

    # =========================================================================
    # DIVERSITY PARAMETERS
    # =========================================================================
    'diversity_weight': ('EVOLVE_DIVERSITY_WEIGHT', float),
    'novelty_weight': ('EVOLVE_NEAT_NOVELTY_WEIGHT', float),
    'diversity_threshold': ('EVOLVE_DIVERSITY_THRESHOLD', float),
    'min_diversity': ('EVOLVE_MIN_DIVERSITY', float),

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    'max_complexity': ('EVOLVE_MAX_COMPLEXITY', int),
    'max_depth': ('EVOLVE_MAX_DEPTH', int),
    'max_length': ('EVOLVE_MAX_LENGTH', int),
    'complexity_penalty': ('EVOLVE_COMPLEXITY_PENALTY', float),

    # =========================================================================
    # ADVANCED PARAMETERS
    # =========================================================================
    'adaptive_mutation': ('EVOLVE_ADAPTIVE_MUTATION', bool),
    'learning_rate': ('EVOLVE_LEARNING_RATE', float),
    'momentum': ('EVOLVE_MOMENTUM', float),
    'decay_rate': ('EVOLVE_DECAY_RATE', float),
    'exploration_factor': ('EVOLVE_EXPLORATION_FACTOR', float),

    # =========================================================================
    # DOMAIN-SPECIFIC PARAMETERS
    # =========================================================================
    # Engineering Domain
    'engineering_optimizer_type': ('EVOLVE_ENGINEERING_OPTIMIZER', str),
    'engineering_constraints': ('EVOLVE_ENGINEERING_CONSTRAINTS', str),

    # Finance Domain
    'finance_trading_mode': ('EVOLVE_FINANCE_MODE', str),
    'finance_risk_tolerance': ('EVOLVE_FINANCE_RISK', float),

    # Pharma Domain
    'pharma_target_type': ('EVOLVE_PHARMA_TARGET', str),
    'pharma_molecular_constraints': ('EVOLVE_PHARMA_CONSTRAINTS', str),

    # Science Domain
    'science_experiment_type': ('EVOLVE_SCIENCE_TYPE', str),
    'science_hypothesis_space': ('EVOLVE_SCIENCE_HYPOTHESIS', str),

    # Trading Domain
    'trading_strategy': ('EVOLVE_TRADING_STRATEGY', str),
    'trading_timeframe': ('EVOLVE_TRADING_TIMEFRAME', str),

    # Web Design Domain
    'web_design_framework': ('EVOLVE_WEB_FRAMEWORK', str),
    'web_design_responsive': ('EVOLVE_WEB_RESPONSIVE', bool),

    # =========================================================================
    # INTEGRATION PARAMETERS
    # =========================================================================
    'enable_ray': ('EVOLVE_ENABLE_RAY', bool),
    'ray_address': ('EVOLVE_RAY_ADDRESS', str),
    'enable_mlflow': ('EVOLVE_ENABLE_MLFLOW', bool),
    'mlflow_tracking_uri': ('EVOLVE_MLFLOW_URI', str),
    'enable_wandb': ('EVOLVE_ENABLE_WANDB', bool),
    'wandb_project': ('EVOLVE_WANDB_PROJECT', str),

    # =========================================================================
    # SECURITY PARAMETERS
    # =========================================================================
    'encrypt_outputs': ('EVOLVE_ENCRYPT_OUTPUTS', bool),
    'secure_mode': ('EVOLVE_SECURE_MODE', bool),
    'allowed_domains': ('EVOLVE_ALLOWED_DOMAINS', list),  # Comma-separated list
    'blocked_domains': ('EVOLVE_BLOCKED_DOMAINS', list),

    # =========================================================================
    # MISC PARAMETERS
    # =========================================================================
    'config_file': ('EVOLVE_CONFIG', str),
    'profile': ('EVOLVE_PROFILE', str),
    'workspace': ('EVOLVE_WORKSPACE', str),
    'experiment_name': ('EVOLVE_EXPERIMENT', str),
    'tags': ('EVOLVE_TAGS', list),  # Comma-separated list
}


# =============================================================================
# PARAMETER RANGES (for validation)
# =============================================================================
# Defines valid ranges for numeric parameters
# =============================================================================

ENV_RANGES: Dict[str, Tuple[Any, Any]] = {
    # Core Evolution
    'max_iterations': (1, 10000),
    'population_size': (1, 10000),
    'generations': (1, 1000),
    'seed': (0, 2**32 - 1),

    # LLM Parameters
    'temperature': (0.0, 2.0),
    'max_tokens': (1, 128000),
    'top_p': (0.0, 1.0),
    'top_k': (1, 100),
    'frequency_penalty': (-2.0, 2.0),
    'presence_penalty': (-2.0, 2.0),

    # PES Parameters
    'plan_temperature': (0.0, 2.0),
    'plan_max_tokens': (1, 32000),
    'memory_capacity': (1, 10000),
    'memory_retention': (0.0, 1.0),
    'memory_decay': (0.0, 1.0),

    # QD Parameters
    'qd_grid_resolution': (2, 1000),
    'qd_archive_size': (10, 100000),
    'qd_feature_dimensions': (1, 20),
    'qd_novelty_threshold': (0.0, 1.0),

    # Genetic Algorithm
    'mutation_rate': (0.0, 1.0),
    'crossover_rate': (0.0, 1.0),
    'elitism_count': (0, 1000),
    'tournament_size': (1, 1000),

    # Adversarial
    'adversarial_rounds': (1, 1000),
    'attack_strength': (0.0, 1.0),

    # Gauntlet
    'gauntlet_rounds': (1, 1000),
    'gauntlet_timeout': (1, 3600),
    'gauntlet_strictness': (0.0, 1.0),

    # Performance
    'parallel_workers': (1, 1000),
    'batch_size': (1, 10000),
    'cache_size': (1, 100000),
    'timeout': (1, 86400),  # 1 second to 1 day
    'max_retries': (0, 100),
    'retry_delay': (0.0, 60.0),

    # Stopping
    'early_stopping_patience': (1, 1000),
    'early_stopping_threshold': (0.0, 1.0),
    'min_improvement': (0.0, 1.0),

    # Diversity
    'diversity_weight': (0.0, 1.0),
    'novelty_weight': (0.0, 1.0),
    'diversity_threshold': (0.0, 1.0),
    'min_diversity': (0.0, 1.0),

    # Constraints
    'max_complexity': (1, 10000),
    'max_depth': (1, 1000),
    'max_length': (1, 100000),
    'complexity_penalty': (0.0, 1.0),

    # Advanced
    'learning_rate': (0.0, 1.0),
    'momentum': (0.0, 1.0),
    'decay_rate': (0.0, 1.0),
    'exploration_factor': (0.0, 1.0),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def env_name_to_config(env_var: str) -> str:
    """
    Convert environment variable name to config parameter name.

    Example: EVOLVE_MAX_ITERATIONS → max_iterations

    Args:
        env_var: Environment variable name

    Returns:
        Configuration parameter name
    """
    # Remove prefix if present
    if env_var.startswith('EVOLVE_'):
        env_var = env_var[7:]  # Remove 'EVOLVE_'

    return env_var.lower()


def config_to_env_name(param_name: str) -> str:
    """
    Convert config parameter name to environment variable name.

    Example: max_iterations → EVOLVE_MAX_ITERATIONS

    Args:
        param_name: Configuration parameter name

    Returns:
        Environment variable name
    """
    return f'EVOLVE_{param_name.upper()}'


def get_env_var_for_param(param_name: str) -> str:
    """
    Get environment variable name for a parameter.

    Args:
        param_name: Configuration parameter name

    Returns:
        Environment variable name

    Raises:
        KeyError: If parameter not in mappings
    """
    if param_name not in ENV_MAPPINGS:
        raise KeyError(f"Parameter '{param_name}' not in environment mappings")
    return ENV_MAPPINGS[param_name][0]


def get_param_type(param_name: str) -> Type:
    """
    Get type for a parameter.

    Args:
        param_name: Configuration parameter name

    Returns:
        Parameter type

    Raises:
        KeyError: If parameter not in mappings
    """
    if param_name not in ENV_MAPPINGS:
        raise KeyError(f"Parameter '{param_name}' not in environment mappings")
    return ENV_MAPPINGS[param_name][1]


def is_valid_param(param_name: str) -> bool:
    """
    Check if parameter is in mappings.

    Args:
        param_name: Configuration parameter name

    Returns:
        True if parameter exists in mappings
    """
    return param_name in ENV_MAPPINGS


def list_all_params() -> list:
    """
    Get list of all parameter names.

    Returns:
        List of parameter names
    """
    return sorted(ENV_MAPPINGS.keys())


def count_mappings() -> int:
    """
    Get count of all parameter mappings.

    Returns:
        Number of mapped parameters
    """
    return len(ENV_MAPPINGS)
