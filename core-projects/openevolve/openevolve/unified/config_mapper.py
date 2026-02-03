"""
Configuration Mapper

Converts unified configuration to mode-specific configurations for:
- OpenEvolve format
- LoongFlow PES format
- Quality Diversity format
- Multi-Objective format
- Adversarial format
"""

from typing import Any, Dict, Optional
from .config import (
    UnifiedEvolutionConfig,
    CommonConfig,
    LLMConfig,
    DatabaseConfig,
    EvaluatorConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig,
)


class ConfigMapper:
    """
    Maps unified configuration to mode-specific configurations

    This class provides bidirectional conversion between:
    - Unified config (single source of truth)
    - Mode-specific configs (for compatibility with existing systems)
    """

    @staticmethod
    def to_openevolve_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to OpenEvolve's config format

        Returns a dictionary compatible with openevolve.config.Config
        """
        config = {}

        # General settings
        config["max_iterations"] = unified.common.max_iterations
        config["checkpoint_interval"] = unified.common.checkpoint_interval
        config["log_level"] = unified.common.log_level
        config["log_dir"] = unified.common.log_dir
        config["random_seed"] = unified.common.random_seed

        # LLM configuration
        llm_config = {}
        llm_config["api_base"] = unified.llm.default_api_base
        llm_config["api_key"] = unified.llm.default_api_key
        llm_config["temperature"] = unified.llm.default_temperature
        llm_config["timeout"] = unified.llm.models[0].timeout if unified.llm.models else 60
        llm_config["retries"] = unified.llm.models[0].retries if unified.llm.models else 3
        llm_config["retry_delay"] = unified.llm.models[0].retry_delay if unified.llm.models else 5

        # Convert models to OpenEvolve format
        llm_config["models"] = []
        for model in unified.llm.models:
            model_dict = {
                "name": model.name,
                "weight": model.weight,
                "api_base": model.api_base,
                "api_key": model.api_key,
                "temperature": model.temperature,
                "top_p": model.top_p,
                "max_tokens": model.max_tokens,
                "timeout": model.timeout,
                "retries": model.retries,
                "retry_delay": model.retry_delay,
                "random_seed": unified.common.random_seed,
                "reasoning_effort": model.reasoning_effort,
            }
            llm_config["models"].append(model_dict)

        # Evaluator models
        llm_config["evaluator_models"] = []
        if unified.llm.evaluator_models:
            for model in unified.llm.evaluator_models:
                model_dict = {
                    "name": model.name,
                    "weight": model.weight,
                    "api_base": model.api_base,
                    "api_key": model.api_key,
                    "temperature": model.temperature,
                    "top_p": model.top_p,
                    "max_tokens": model.max_tokens,
                    "timeout": model.timeout,
                    "retries": model.retries,
                    "retry_delay": model.retry_delay,
                    "random_seed": unified.common.random_seed,
                    "reasoning_effort": model.reasoning_effort,
                }
                llm_config["evaluator_models"].append(model_dict)

        config["llm"] = llm_config

        # Prompt configuration
        if unified.openevolve:
            prompt_config = {
                "template_dir": unified.openevolve.template_dir,
                "system_message": unified.openevolve.system_message,
                "evaluator_system_message": unified.openevolve.evaluator_system_message,
                "num_top_programs": unified.openevolve.num_top_programs,
                "num_diverse_programs": unified.openevolve.num_diverse_programs,
                "use_template_stochasticity": unified.openevolve.use_template_stochasticity,
                "template_variations": unified.openevolve.template_variations,
                "include_artifacts": unified.openevolve.include_artifacts,
                "max_artifact_bytes": unified.openevolve.max_artifact_bytes,
                "artifact_security_filter": unified.openevolve.artifact_security_filter,
                "suggest_simplification_after_chars": unified.openevolve.suggest_simplification_after_chars,
                "include_changes_under_chars": unified.openevolve.include_changes_under_chars,
                "concise_implementation_max_lines": unified.openevolve.concise_implementation_max_lines,
                "comprehensive_implementation_min_lines": unified.openevolve.comprehensive_implementation_min_lines,
            }
            config["prompt"] = prompt_config

        # Database configuration
        db_config = {
            "db_path": unified.database.db_path,
            "in_memory": unified.database.storage_type == "in_memory",
            "log_prompts": unified.database.log_prompts,
            "population_size": unified.database.population_size,
            "archive_size": unified.database.elite_archive_size,
            "num_islands": unified.database.num_islands,
            "elite_selection_ratio": unified.database.elite_selection_ratio,
            "exploration_ratio": unified.database.exploration_rate,
            "exploitation_ratio": unified.database.exploitation_ratio,
            "diversity_metric": unified.database.diversity_metric,
            "feature_dimensions": unified.database.feature_dimensions,
            "feature_bins": unified.database.feature_bins,
            "diversity_reference_size": unified.database.diversity_reference_size,
            "migration_interval": unified.database.migration_interval,
            "migration_rate": unified.database.migration_rate,
            "random_seed": unified.common.random_seed,
            "artifacts_base_path": None,
            "artifact_size_threshold": unified.evaluator.max_artifact_storage if unified.evaluator else 100 * 1024 * 1024,
            "cleanup_old_artifacts": False,
            "artifact_retention_days": 30,
        }
        config["database"] = db_config

        # Evaluator configuration
        eval_config = {
            "timeout": unified.evaluator.timeout,
            "max_retries": unified.evaluator.max_retries,
            "memory_limit_mb": unified.evaluator.memory_limit_mb,
            "cpu_limit": unified.evaluator.cpu_limit,
            "cascade_evaluation": unified.evaluator.cascade_evaluation,
            "cascade_thresholds": unified.evaluator.cascade_thresholds,
            "parallel_evaluations": unified.evaluator.parallel_evaluations,
            "distributed": unified.evaluator.distributed,
            "use_llm_feedback": unified.evaluator.use_llm_feedback,
            "llm_feedback_weight": unified.evaluator.llm_feedback_weight,
            "enable_artifacts": unified.evaluator.enable_artifacts,
            "max_artifact_storage": unified.evaluator.max_artifact_storage,
        }
        config["evaluator"] = eval_config

        # Evolution trace configuration
        if unified.openevolve:
            trace_config = {
                "enabled": unified.openevolve.evolution_trace_enabled,
                "format": unified.openevolve.evolution_trace_format,
                "include_code": unified.openevolve.evolution_trace_include_code,
                "include_prompts": unified.openevolve.evolution_trace_include_prompts,
                "output_path": None,
                "buffer_size": unified.openevolve.evolution_trace_buffer_size,
                "compress": unified.openevolve.evolution_trace_compress,
            }
            config["evolution_trace"] = trace_config

        # Evolution settings
        if unified.openevolve:
            config["diff_based_evolution"] = unified.openevolve.diff_based_evolution
            config["max_code_length"] = unified.openevolve.max_code_length
            config["language"] = unified.openevolve.language
            config["file_suffix"] = unified.openevolve.file_suffix

        # Early stopping
        if unified.openevolve:
            config["early_stopping_patience"] = unified.openevolve.early_stopping_patience
            config["convergence_threshold"] = unified.openevolve.convergence_threshold
            config["early_stopping_metric"] = unified.openevolve.early_stopping_metric

        return config

    @staticmethod
    def to_pes_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to LoongFlow PES config format

        Returns a dictionary compatible with loongflow.framework.pes.context.config
        """
        config = {}

        # Workspace path
        config["workspace_path"] = unified.common.workspace_path

        # Logger configuration
        logger_config = {
            "level": unified.common.log_level,
            "console_logging": unified.common.log_to_console,
            "file_logging": unified.common.log_to_file,
            "log_path": unified.common.log_dir,
            "filename": "evolux.log",
            "rotation": unified.common.log_rotation,
            "backup_count": unified.common.log_backup_count,
        }
        config["logger"] = logger_config

        # LLM configuration
        if unified.llm.models:
            primary_model = unified.llm.models[0]
            llm_config = {
                "model": primary_model.name,
                "url": primary_model.api_base,
                "api_key": primary_model.api_key or unified.llm.default_api_key,
                "model_provider": primary_model.model_provider,
                "temperature": primary_model.temperature or unified.llm.default_temperature,
                "context_length": primary_model.context_length,
                "max_tokens": primary_model.max_tokens,
                "top_p": primary_model.top_p,
                "timeout": primary_model.timeout or unified.common.timeout,
                "completion_token_price": 0.0,
                "prompt_token_price": 0.0,
            }
            config["llm_config"] = llm_config

        # Database configuration
        db_config = {
            "storage_type": unified.database.storage_type,
            "redis_url": unified.database.redis_url,
            "num_islands": unified.database.num_islands,
            "population_size": unified.database.population_size,
            "elite_archive_size": unified.database.elite_archive_size,
            "use_sampling_weight": unified.database.use_sampling_weight,
            "sampling_weight_power": unified.database.sampling_weight_power,
            "exploration_rate": unified.database.exploration_rate,
            "migration_interval": unified.database.migration_interval,
            "migration_rate": unified.database.migration_rate,
            "boltzmann_temperature": unified.database.boltzmann_temperature,
            "feature_bins": unified.database.feature_bins,
            "feature_dimensions": unified.database.feature_dimensions,
            "feature_scaling_method": unified.database.feature_scaling_method,
            "checkpoint_interval": unified.database.checkpoint_interval,
            "output_path": unified.database.output_path,
        }
        config["database"] = db_config

        # Evaluator configuration
        eval_config = {
            "llm_config": None,  # Will inherit from global
            "evaluate_code": unified.evaluator.evaluate_code,
            "workspace_path": unified.evaluator.workspace_path,
            "timeout": unified.evaluator.timeout,
            "evolve_target": unified.evaluator.evolve_target,
            "agent": {},
        }
        config["evaluator"] = eval_config

        # Evolve configuration
        evolve_config = {
            "task_name": unified.common.task_name,
            "task": unified.common.task_description or "Evolution task",
            "initial_code": "",
            "initial_score": None,
            "initial_evaluation": "",
            "workspace_path": unified.common.workspace_path,
            "database": db_config,
            "evaluator": eval_config,
            "max_iterations": unified.common.max_iterations,
            "target_score": 1.0,
            "concurrency": unified.common.concurrency,
            "planner_name": "evolve_planner",
            "executor_name": "evolve_executor",
            "summary_name": "evolve_summary",
            "metadata": unified.metadata,
        }
        config["evolve"] = evolve_config

        # Planners
        if unified.pes:
            config["planners"] = {
                "evolve_planner": {
                    "type": unified.pes.planner_type,
                    "llm_config": None,  # Will inherit from global
                },
            }

        # Executors
        if unified.pes:
            config["executors"] = {
                "evolve_executor": {
                    "type": unified.pes.executor_type,
                    "llm_config": None,  # Will inherit from global
                },
            }

        # Summarizers
        if unified.pes:
            config["summarizers"] = {
                "evolve_summary": {
                    "type": unified.pes.summary_type,
                    "llm_config": None,  # Will inherit from global
                },
            }

        return config

    @staticmethod
    def to_qd_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to Quality Diversity config format

        Returns optimized config for QD/MAP-Elites evolution
        """
        config = ConfigMapper.to_openevolve_config(unified)

        # Override with QD-specific settings
        if unified.qd:
            # Feature dimensions
            config["database"]["feature_dimensions"] = unified.qd.grid_dimensions
            config["database"]["feature_bins"] = unified.qd.grid_resolution

            # QD-specific settings
            config["qd"] = {
                "enable_map_elites": unified.qd.enable_map_elites,
                "adaptive_grid": unified.qd.adaptive_grid,
                "grid_update_interval": unified.qd.grid_update_interval,
                "archive_type": unified.qd.archive_type,
                "archive_size_limit": unified.qd.archive_size_limit,
                "archive_elitism": unified.qd.archive_elitism,
                "use_novelty": unified.qd.use_novelty,
                "novelty_threshold": unified.qd.novelty_threshold,
                "feature_extraction_method": unified.qd.feature_extraction_method,
                "feature_normalization": unified.qd.feature_normalization,
                "use_feature_learning": unified.qd.use_feature_learning,
                "feature_learning_rate": unified.qd.feature_learning_rate,
                "cvt_samples": unified.qd.cvt_samples,
                "use_niching": unified.qd.use_niching,
                "niche_radius": unified.qd.niche_radius,
            }

        return config

    @staticmethod
    def to_mo_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to Multi-Objective config format

        Returns optimized config for MO optimization
        """
        config = ConfigMapper.to_openevolve_config(unified)

        # Override with MO-specific settings
        if unified.mo:
            config["mo"] = {
                "objectives": unified.mo.objectives,
                "objective_weights": unified.mo.objective_weights,
                "optimization_direction": unified.mo.optimization_direction,
                "use_pareto": unified.mo.use_pareto,
                "pareto_archive_size": unified.mo.pareto_archive_size,
                "pareto_pruning_method": unified.mo.pareto_pruning_method,
                "crowding_distance_metric": unified.mo.crowding_distance_metric,
                "use_hypervolume": unified.mo.use_hypervolume,
                "selection_method": unified.mo.selection_method,
                "tournament_size": unified.mo.tournament_size,
                "crossover_rate": unified.mo.crossover_rate,
                "mutation_rate": unified.mo.mutation_rate,
                "use_scalarization": unified.mo.use_scalarization,
                "scalarization_method": unified.mo.scalarization_method,
                "reference_point": unified.mo.reference_point,
            }

            # For MO, modify database to track multiple objectives
            config["database"]["feature_dimensions"] = unified.mo.objectives

        return config

    @staticmethod
    def to_adversarial_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to Adversarial Evolution config format

        Returns optimized config for adversarial evolution
        """
        config = ConfigMapper.to_openevolve_config(unified)

        # Override with adversarial-specific settings
        if unified.adversarial:
            config["adversarial"] = {
                "enable_adversarial": unified.adversarial.enable_adversarial,
                "num_adversaries": unified.adversarial.num_adversaries,
                "adversarial_mode": unified.adversarial.adversarial_mode,
                "adversarial_rounds": unified.adversarial.adversarial_rounds,
                "generator_objective": unified.adversarial.generator_objective,
                "discriminator_objective": unified.adversarial.discriminator_objective,
                "balance_factor": unified.adversarial.balance_factor,
                "use_coevolution": unified.adversarial.use_coevolution,
                "coevolution_frequency": unified.adversarial.coevolution_frequency,
                "fitness_sharing": unified.adversarial.fitness_sharing,
                "fitness_sharing_sigma": unified.adversarial.fitness_sharing_sigma,
                "use_arms_race": unified.adversarial.use_arms_race,
            }

            # For adversarial, need multiple populations
            config["database"]["num_islands"] = unified.adversarial.num_adversaries

        return config

    @staticmethod
    def from_openevolve_config(openevolve_config: Dict[str, Any]) -> UnifiedEvolutionConfig:
        """
        Convert OpenEvolve config to unified config

        Args:
            openevolve_config: Dictionary from openevolve.config.Config.to_dict()

        Returns:
            UnifiedEvolutionConfig instance
        """
        # Extract common config
        common = CommonConfig(
            max_iterations=openevolve_config.get("max_iterations", 100),
            random_seed=openevolve_config.get("random_seed", 42),
            checkpoint_interval=openevolve_config.get("checkpoint_interval", 50),
            log_level=openevolve_config.get("log_level", "INFO"),
            log_dir=openevolve_config.get("log_dir"),
        )

        # Extract LLM config
        llm_dict = openevolve_config.get("llm", {})
        models = []
        if "models" in llm_dict:
            for m in llm_dict["models"]:
                models.append(LLMModelConfig(**m))

        llm = LLMConfig(
            models=models,
            default_api_base=llm_dict.get("api_base", "https://api.openai.com/v1"),
            default_api_key=llm_dict.get("api_key"),
            default_temperature=llm_dict.get("temperature", 0.7),
        )

        # Extract database config
        db_dict = openevolve_config.get("database", {})
        database = DatabaseConfig(
            storage_type="in_memory" if db_dict.get("in_memory", True) else "file",
            db_path=db_dict.get("db_path"),
            population_size=db_dict.get("population_size", 1000),
            elite_archive_size=db_dict.get("archive_size", 100),
            num_islands=db_dict.get("num_islands", 5),
            migration_interval=db_dict.get("migration_interval", 50),
            migration_rate=db_dict.get("migration_rate", 0.1),
            feature_dimensions=db_dict.get("feature_dimensions", ["complexity", "diversity"]),
            feature_bins=db_dict.get("feature_bins", 10),
        )

        # Extract evaluator config
        eval_dict = openevolve_config.get("evaluator", {})
        evaluator = EvaluatorConfig(
            timeout=eval_dict.get("timeout", 300),
            max_retries=eval_dict.get("max_retries", 3),
            cascade_evaluation=eval_dict.get("cascade_evaluation", True),
            cascade_thresholds=eval_dict.get("cascade_thresholds", [0.5, 0.75, 0.9]),
            parallel_evaluations=eval_dict.get("parallel_evaluations", 4),
        )

        # Extract OpenEvolve-specific config
        prompt_dict = openevolve_config.get("prompt", {})
        openevolve = OpenEvolveConfig(
            system_message=prompt_dict.get("system_message", "You are an expert coder."),
            num_top_programs=prompt_dict.get("num_top_programs", 3),
            num_diverse_programs=prompt_dict.get("num_diverse_programs", 2),
            diff_based_evolution=openevolve_config.get("diff_based_evolution", True),
            max_code_length=openevolve_config.get("max_code_length", 10000),
        )

        return UnifiedEvolutionConfig(
            evolution_mode="openevolve",
            common=common,
            llm=llm,
            database=database,
            evaluator=evaluator,
            openevolve=openevolve,
        )
