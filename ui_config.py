"""
Configuration constants for UI components.
"""

from typing import Dict, Any

# ============================================================================
# UI Configuration
# ============================================================================

UI_CONFIG: Dict[str, Any] = {
    "analytics": {
        "default_time_range": "7d",
        "max_data_points": 1000,
        "refresh_interval": 60,
        "chart_height": 400,
        "chart_colors": {
            "primary": "#1f77b4",
            "success": "#2ca02c",
            "warning": "#ff7f0e",
            "error": "#d62728",
        }
    },
    "knowledge_base": {
        "page_size": 20,
        "max_search_results": 100,
        "search_timeout": 2,
        "graph_layout": "spring",
    },
    "monitoring": {
        "update_interval": 2,
        "alert_threshold": 0.85,
        "max_alerts": 50,
        "log_lines": 100,
    },
    "dependency_viz": {
        "node_size": 30,
        "edge_width": 2,
        "critical_path_color": "#ff7f0e",
        "circular_dep_color": "#d62728",
        "layout_algorithm": "spring",
    },
    "batch_operations": {
        "max_selection": 100,
        "confirmation_required": True,
        "rollback_timeout": 5,
    },
    "templates": {
        "storage_path": "workflow_templates",
        "max_templates": 50,
        "version": "1.0",
    },
    "auto_approval": {
        "max_rules": 20,
        "audit_log_size": 1000,
        "default_enabled": False,
    }
}


# ============================================================================
# Feature Flags
# ============================================================================

FEATURE_FLAGS: Dict[str, bool] = {
    "analytics_dashboard": True,
    "knowledge_base": True,
    "dependency_viz": True,
    "auto_approval_ui": True,
    "batch_operations": True,
    "enhanced_monitoring": True,
    "workflow_templates": True,
}


# ============================================================================
# Color Schemes
# ============================================================================

STATUS_COLORS: Dict[str, str] = {
    "pending": "gray",
    "in_progress": "orange",
    "completed": "green",
    "failed": "red",
    "paused": "blue",
    "terminated": "red",
}

SEVERITY_COLORS: Dict[str, str] = {
    "info": "blue",
    "warning": "orange",
    "error": "red",
}


# ============================================================================
# Chart Configuration
# ============================================================================

CHART_CONFIG: Dict[str, Any] = {
    "plotly": {
        "config": {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        "layout": {
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
            "hovermode": "closest",
        }
    }
}


# ============================================================================
# OpenEvolve Integration Configuration
# ============================================================================

OPENEVOLVE_CONFIG: Dict[str, Any] = {
    "default_model": "gpt-4",
    "default_evolution_mode": "standard",
    "default_temperature": 0.7,
    "default_max_iterations": 10,
    "metrics_refresh_interval": 5,
    "cost_per_1k_tokens": {
        "gpt-4": 0.03,
        "gpt-3.5-turbo": 0.002,
        "claude-3": 0.015,
    }
}


# ============================================================================
# Time Range Options
# ============================================================================

TIME_RANGE_OPTIONS: Dict[str, str] = {
    "Last Hour": "1h",
    "Last 24 Hours": "24h",
    "Last 7 Days": "7d",
    "Last 30 Days": "30d",
    "Last 90 Days": "90d",
    "All Time": "all",
}


# ============================================================================
# Export Configuration
# ============================================================================

EXPORT_CONFIG: Dict[str, Any] = {
    "formats": ["csv", "json", "xlsx"],
    "max_export_rows": 10000,
    "default_format": "csv",
}


# OpenEvolve Comprehensive Parameter Configuration (211 parameters across 19 categories)
OPENEVOLVE_PARAMS: Dict[str, Dict[str, Any]] = {
    # Category 1: Core Evolution Parameters
    "core_evolution": {
        "evolution_mode": {
            "type": "select",
            "options": ["standard", "quality_diversity", "multi_objective", "adversarial", "coevolution"],
            "default": "standard",
            "description": "Primary evolution strategy to use"
        },
        "max_iterations": {
            "type": "int",
            "min": 1,
            "max": 1000,
            "default": 20,
            "description": "Maximum number of evolution iterations"
        },
        "population_size": {
            "type": "int",
            "min": 2,
            "max": 1000,
            "default": 30,
            "description": "Number of individuals in the population"
        },
        "temperature": {
            "type": "float",
            "min": 0.0,
            "max": 2.0,
            "default": 0.7,
            "description": "LLM temperature for generation"
        },
        "max_tokens": {
            "type": "int",
            "min": 1,
            "max": 32768,
            "default": 2048,
            "description": "Maximum tokens per generation"
        },
        "seed": {
            "type": "int",
            "min": 0,
            "max": 2147483647,
            "default": None,
            "description": "Random seed for reproducibility"
        },
    },
    
    # Category 2: Selection Parameters
    "selection": {
        "elite_ratio": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.1,
            "description": "Ratio of top individuals to preserve"
        },
        "exploration_ratio": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.4,
            "description": "Ratio for exploration vs exploitation"
        },
        "exploitation_ratio": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Ratio for exploitation"
        },
        "tournament_size": {
            "type": "int",
            "min": 2,
            "max": 20,
            "default": 3,
            "description": "Size of tournament selection"
        },
        "selection_pressure": {
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 2.0,
            "description": "Selection pressure coefficient"
        },
    },
    
    # Category 3: Quality Diversity Parameters
    "quality_diversity": {
        "enable_quality_diversity": {
            "type": "bool",
            "default": False,
            "description": "Enable MAP-Elites quality diversity"
        },
        "feature_dimensions": {
            "type": "list",
            "default": ["complexity", "novelty"],
            "description": "Behavior dimensions for archive"
        },
        "feature_bins": {
            "type": "int",
            "min": 2,
            "max": 100,
            "default": 10,
            "description": "Number of bins per dimension"
        },
        "archive_size": {
            "type": "int",
            "min": 10,
            "max": 10000,
            "default": 100,
            "description": "Maximum archive size"
        },
        "diversity_metric": {
            "type": "select",
            "options": ["edit_distance", "semantic", "structural", "behavioral"],
            "default": "edit_distance",
            "description": "Metric for measuring diversity"
        },
        "novelty_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.1,
            "description": "Minimum novelty for archive inclusion"
        },
    },
    
    # Category 4: Multi-Objective Parameters
    "multi_objective": {
        "enable_multi_objective": {
            "type": "bool",
            "default": False,
            "description": "Enable multi-objective optimization"
        },
        "objectives": {
            "type": "list",
            "default": ["quality", "efficiency"],
            "description": "List of objectives to optimize"
        },
        "pareto_front_size": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "default": 50,
            "description": "Maximum Pareto front size"
        },
        "objective_weights": {
            "type": "dict",
            "default": {},
            "description": "Weights for each objective"
        },
        "crowding_distance_weight": {
            "type": "float",
            "min": 0.0,
            "max": 10.0,
            "default": 1.0,
            "description": "Weight for crowding distance"
        },
    },
    
    # Category 5: Evaluation Parameters
    "evaluation": {
        "enable_cascade_evaluation": {
            "type": "bool",
            "default": True,
            "description": "Enable cascade evaluation"
        },
        "cascade_thresholds": {
            "type": "list",
            "default": [0.5, 0.75, 0.9],
            "description": "Thresholds for cascade levels"
        },
        "parallel_evaluations": {
            "type": "int",
            "min": 1,
            "max": 32,
            "default": 4,
            "description": "Number of parallel evaluations"
        },
        "evaluation_timeout": {
            "type": "int",
            "min": 1,
            "max": 3600,
            "default": 300,
            "description": "Evaluation timeout in seconds"
        },
        "max_retries": {
            "type": "int",
            "min": 0,
            "max": 10,
            "default": 3,
            "description": "Maximum evaluation retries"
        },
        "ensemble_size": {
            "type": "int",
            "min": 1,
            "max": 10,
            "default": 3,
            "description": "Number of evaluators in ensemble"
        },
        "consensus_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.7,
            "description": "Consensus threshold for ensemble"
        },
    },
    
    # Category 6: Island Model Parameters
    "island_model": {
        "enable_island_model": {
            "type": "bool",
            "default": False,
            "description": "Enable island model evolution"
        },
        "num_islands": {
            "type": "int",
            "min": 2,
            "max": 20,
            "default": 4,
            "description": "Number of islands"
        },
        "migration_interval": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 10,
            "description": "Iterations between migrations"
        },
        "migration_size": {
            "type": "int",
            "min": 1,
            "max": 50,
            "default": 5,
            "description": "Number of individuals to migrate"
        },
        "migration_topology": {
            "type": "select",
            "options": ["ring", "star", "fully_connected", "random"],
            "default": "ring",
            "description": "Migration topology"
        },
    },
    
    # Category 7: Artifact Management Parameters
    "artifacts": {
        "enable_artifacts": {
            "type": "bool",
            "default": True,
            "description": "Enable artifact management"
        },
        "artifact_types": {
            "type": "list",
            "default": ["code", "documentation", "tests"],
            "description": "Types of artifacts to manage"
        },
        "max_artifacts_per_run": {
            "type": "int",
            "min": 1,
            "max": 1000,
            "default": 100,
            "description": "Maximum artifacts per run"
        },
        "artifact_retention_days": {
            "type": "int",
            "min": 1,
            "max": 365,
            "default": 30,
            "description": "Days to retain artifacts"
        },
    },
    
    # Category 8: Checkpoint Parameters
    "checkpointing": {
        "checkpoint_interval": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 10,
            "description": "Iterations between checkpoints"
        },
        "max_checkpoints": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 10,
            "description": "Maximum checkpoints to keep"
        },
        "checkpoint_compression": {
            "type": "bool",
            "default": True,
            "description": "Compress checkpoint files"
        },
    },
    
    # Category 9: Prompt Engineering Parameters
    "prompt_engineering": {
        "enable_meta_prompting": {
            "type": "bool",
            "default": False,
            "description": "Enable meta-prompting"
        },
        "enable_template_stochasticity": {
            "type": "bool",
            "default": False,
            "description": "Enable template stochasticity"
        },
        "prompt_mutation_rate": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.1,
            "description": "Rate of prompt mutations"
        },
        "system_prompt_template": {
            "type": "text",
            "default": "",
            "description": "System prompt template"
        },
    },
    
    # Category 10: Resource Management Parameters
    "resources": {
        "max_cost_usd": {
            "type": "float",
            "min": 0.0,
            "max": 10000.0,
            "default": 100.0,
            "description": "Maximum cost in USD"
        },
        "max_api_calls": {
            "type": "int",
            "min": 1,
            "max": 100000,
            "default": 1000,
            "description": "Maximum API calls"
        },
        "max_execution_time": {
            "type": "int",
            "min": 1,
            "max": 86400,
            "default": 3600,
            "description": "Maximum execution time in seconds"
        },
        "memory_limit_mb": {
            "type": "int",
            "min": 100,
            "max": 32000,
            "default": 4096,
            "description": "Memory limit in MB"
        },
    },
    
    # Category 11: Distributed Processing Parameters
    "distributed": {
        "enable_distributed": {
            "type": "bool",
            "default": False,
            "description": "Enable distributed processing"
        },
        "num_workers": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 4,
            "description": "Number of worker processes"
        },
        "communication_backend": {
            "type": "select",
            "options": ["local", "redis", "rabbitmq", "kafka"],
            "default": "local",
            "description": "Communication backend"
        },
        "worker_timeout": {
            "type": "int",
            "min": 1,
            "max": 3600,
            "default": 300,
            "description": "Worker timeout in seconds"
        },
    },
    
    # Category 12: Logging and Monitoring Parameters
    "logging": {
        "log_level": {
            "type": "select",
            "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
            "default": "INFO",
            "description": "Logging level"
        },
        "log_to_file": {
            "type": "bool",
            "default": True,
            "description": "Log to file"
        },
        "log_file_path": {
            "type": "text",
            "default": "openevolve.log",
            "description": "Log file path"
        },
        "metrics_collection_interval": {
            "type": "int",
            "min": 1,
            "max": 3600,
            "default": 60,
            "description": "Metrics collection interval in seconds"
        },
    },
    
    # Category 13: Adversarial Testing Parameters
    "adversarial": {
        "enable_adversarial": {
            "type": "bool",
            "default": False,
            "description": "Enable adversarial testing"
        },
        "adversarial_rounds": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 5,
            "description": "Number of adversarial rounds"
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
        "critique_depth": {
            "type": "int",
            "min": 1,
            "max": 10,
            "default": 5,
            "description": "Depth of critique analysis"
        },
    },
    
    # Category 14: Mutation Parameters
    "mutation": {
        "mutation_rate": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.1,
            "description": "Probability of mutation"
        },
        "mutation_strength": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Strength of mutations"
        },
        "adaptive_mutation": {
            "type": "bool",
            "default": True,
            "description": "Enable adaptive mutation rates"
        },
    },
    
    # Category 15: Crossover Parameters
    "crossover": {
        "crossover_rate": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.7,
            "description": "Probability of crossover"
        },
        "crossover_type": {
            "type": "select",
            "options": ["single_point", "two_point", "uniform", "semantic"],
            "default": "uniform",
            "description": "Type of crossover operation"
        },
    },
    
    # Category 16: Termination Criteria Parameters
    "termination": {
        "fitness_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.95,
            "description": "Fitness threshold for early termination"
        },
        "stagnation_generations": {
            "type": "int",
            "min": 1,
            "max": 100,
            "default": 20,
            "description": "Generations without improvement before termination"
        },
        "enable_early_stopping": {
            "type": "bool",
            "default": True,
            "description": "Enable early stopping"
        },
    },
    
    # Category 17: Caching Parameters
    "caching": {
        "enable_caching": {
            "type": "bool",
            "default": True,
            "description": "Enable result caching"
        },
        "cache_size_mb": {
            "type": "int",
            "min": 10,
            "max": 10000,
            "default": 1000,
            "description": "Cache size in MB"
        },
        "cache_ttl_seconds": {
            "type": "int",
            "min": 60,
            "max": 86400,
            "default": 3600,
            "description": "Cache TTL in seconds"
        },
    },
    
    # Category 18: Visualization Parameters
    "visualization": {
        "enable_live_visualization": {
            "type": "bool",
            "default": True,
            "description": "Enable live visualization"
        },
        "visualization_update_interval": {
            "type": "int",
            "min": 1,
            "max": 60,
            "default": 5,
            "description": "Visualization update interval in seconds"
        },
        "plot_style": {
            "type": "select",
            "options": ["default", "dark", "seaborn", "ggplot"],
            "default": "default",
            "description": "Plot style"
        },
    },
    
    # Category 19: Advanced Features Parameters
    "advanced": {
        "enable_coevolution": {
            "type": "bool",
            "default": False,
            "description": "Enable coevolution"
        },
        "enable_speciation": {
            "type": "bool",
            "default": False,
            "description": "Enable speciation"
        },
        "speciation_threshold": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.3,
            "description": "Threshold for speciation"
        },
        "enable_niching": {
            "type": "bool",
            "default": False,
            "description": "Enable niching"
        },
        "niche_radius": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.1,
            "description": "Radius for niching"
        },
    },
}

OPENEVOLVE_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "max_iterations": 5,
        "population_size": 10,
        "temperature": 0.5,
        "elite_ratio": 0.2,
        "exploration_ratio": 0.3,
        "exploitation_ratio": 0.5,
        "checkpoint_interval": 5,
        "enable_artifacts": False,
        "enable_cascade_evaluation": False,
        "parallel_evaluations": 2,
    },
    "balanced": {
        "max_iterations": 20,
        "population_size": 30,
        "temperature": 0.7,
        "elite_ratio": 0.15,
        "exploration_ratio": 0.35,
        "exploitation_ratio": 0.5,
        "checkpoint_interval": 10,
        "enable_artifacts": True,
        "enable_cascade_evaluation": True,
        "cascade_thresholds": [0.5, 0.75, 0.9],
        "parallel_evaluations": 4,
    },
    "thorough": {
        "max_iterations": 50,
        "population_size": 50,
        "temperature": 0.8,
        "elite_ratio": 0.1,
        "exploration_ratio": 0.4,
        "exploitation_ratio": 0.5,
        "checkpoint_interval": 10,
        "enable_artifacts": True,
        "enable_cascade_evaluation": True,
        "cascade_thresholds": [0.6, 0.8, 0.95],
        "parallel_evaluations": 8,
        "enable_quality_diversity": True,
        "feature_dimensions": ["complexity", "novelty", "quality"],
        "feature_bins": 10,
    },
    "research": {
        "max_iterations": 100,
        "population_size": 100,
        "temperature": 0.9,
        "elite_ratio": 0.05,
        "exploration_ratio": 0.5,
        "exploitation_ratio": 0.45,
        "checkpoint_interval": 5,
        "enable_artifacts": True,
        "enable_cascade_evaluation": True,
        "cascade_thresholds": [0.5, 0.7, 0.85, 0.95],
        "parallel_evaluations": 16,
        "enable_quality_diversity": True,
        "feature_dimensions": ["complexity", "novelty", "quality", "diversity"],
        "feature_bins": 20,
        "enable_island_model": True,
        "num_islands": 4,
        "migration_interval": 10,
        "migration_size": 5,
        "enable_meta_prompting": True,
        "enable_template_stochasticity": True,
    },
}
