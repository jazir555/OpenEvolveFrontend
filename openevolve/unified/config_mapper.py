"""
Config Mapper
Converts unified configuration to mode-specific formats

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional
from .config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig
)


class ConfigMapper:
    """
    Map unified configuration to mode-specific formats

    Handles bidirectional conversion between:
    - Unified config
    - LoongFlow PES format
    - OpenEvolve format
    - QD mode format
    - MO mode format
    - Adversarial mode format
    """

    @staticmethod
    def to_pes_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to LoongFlow PES format

        Returns:
            Dictionary compatible with LoongFlow PES agent
        """
        mode = unified.evolution_mode

        # Core parameters
        pes_config = {
            # Task configuration
            "task": {
                "max_iterations": unified.max_iterations,
                "time_limit_seconds": unified.time_limit_seconds,
                "target_fitness": unified.target_fitness,
                "domain": unified.domain.value,
            },

            # PES-specific
            "evolve": {
                "enable_planning": unified.pes.enable_planning,
                "enable_memory": unified.database.enable_memory,
                "early_stopping": unified.evaluator.early_stopping,
                "early_stopping_patience": unified.evaluator.early_stopping_patience,
                "early_stopping_threshold": unified.evaluator.early_stopping_threshold,
            },

            # LLM configuration
            "llm": {
                "planner_models": [
                    {
                        "name": m.name,
                        "weight": m.weight,
                        "api_base": m.api_base,
                        "temperature": m.temperature or unified.llm.plan_temperature,
                        "max_tokens": m.max_tokens or unified.llm.max_tokens,
                    }
                    for m in unified.llm.planner_models or unified.llm.models
                ],
                "executor_models": [
                    {
                        "name": m.name,
                        "weight": m.weight,
                        "api_base": m.api_base,
                        "temperature": m.temperature or unified.llm.temperature,
                        "max_tokens": m.max_tokens or unified.llm.max_tokens,
                    }
                    for m in unified.llm.models
                ],
                "summary_models": [
                    {
                        "name": m.name,
                        "weight": m.weight,
                        "api_base": m.api_base,
                        "temperature": m.temperature or unified.llm.summary_temperature,
                        "max_tokens": m.max_tokens or unified.llm.max_tokens,
                    }
                    for m in unified.llm.summary_models or unified.llm.models
                ],
                "timeout": unified.llm.timeout,
                "retries": unified.llm.retries,
                "retry_delay": unified.llm.retry_delay,
            },

            # Database / Memory
            "database": {
                "num_islands": unified.database.num_islands,
                "population_size": unified.database.population_size,
                "archive_size": unified.database.archive_size,
                "exploration_rate": unified.database.exploration_rate,
                "adaptive_exploration": unified.database.adaptive_exploration,
                "memory_path": unified.database.memory_path,
                "log_prompts": unified.database.log_prompts,
            },

            # Executor
            "executor": {
                "max_rounds": unified.pes.max_rounds,
                "parallel_candidates": unified.pes.parallel_candidates,
                "max_plans": unified.pes.max_plans,
            },

            # Summary
            "summary": {
                "enable_summary": unified.pes.enable_summary,
                "summary_iterations": unified.pes.summary_iterations,
            },

            # Planner
            "planner": {
                "use_memory": unified.pes.use_memory,
                "memory_top_k": unified.pes.memory_top_k,
            },
        }

        return pes_config

    @staticmethod
    def to_openevolve_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to OpenEvolve format

        Returns:
            Dictionary compatible with OpenEvolve controller
        """
        mode = unified.evolution_mode

        # Core parameters
        oe_config = {
            # General settings
            "max_iterations": unified.max_iterations,
            "checkpoint_interval": unified.checkpoint_interval,
            "random_seed": unified.random_seed,

            # Evolution settings
            "diff_based_evolution": unified.diff_based_evolution,
            "max_code_length": unified.max_code_length,
            "language": unified.language,

            # Early stopping
            "early_stopping_patience": unified.early_stopping_patience,
            "convergence_threshold": unified.convergence_threshold,
            "early_stopping_metric": unified.early_stopping_metric,
        }

        # Database configuration
        oe_config["database"] = {
            "population_size": unified.database.population_size,
            "archive_size": unified.database.archive_size,
            "num_islands": unified.database.num_islands,

            # Selection
            "elite_selection_ratio": unified.database.elite_selection_ratio,
            "exploration_ratio": unified.database.exploration_ratio,
            "exploitation_ratio": unified.database.exploitation_ratio,

            # MAP-Elites
            "feature_dimensions": unified.database.feature_dimensions,
            "feature_bins": unified.database.feature_bins,

            # Migration
            "migration_interval": unified.database.migration_interval,
            "migration_rate": unified.database.migration_rate,
            "migration_topology": unified.database.migration_topology,

            # Diversity
            "diversity_metric": unified.database.diversity_metric,
            "diversity_reference_size": unified.database.diversity_reference_size,

            # Logging
            "log_prompts": unified.database.log_prompts,
        }

        # LLM configuration
        oe_config["llm"] = {
            "models": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "api_base": m.api_base,
                    "api_key": m.api_key,
                    "temperature": m.temperature or unified.llm.temperature,
                    "max_tokens": m.max_tokens or unified.llm.max_tokens,
                }
                for m in unified.llm.models
            ],
            "evaluator_models": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "api_base": m.api_base,
                }
                for m in unified.llm.evaluator_models
            ],
            "temperature": unified.llm.temperature,
            "top_p": unified.llm.top_p,
            "max_tokens": unified.llm.max_tokens,
            "timeout": unified.llm.timeout,
            "retries": unified.llm.retries,
            "retry_delay": unified.llm.retry_delay,
            "random_seed": unified.llm.random_seed,
            "reasoning_effort": unified.llm.reasoning_effort,
        }

        # Evaluator configuration
        oe_config["evaluator"] = {
            "timeout": unified.evaluator.timeout,
            "max_retries": unified.evaluator.max_retries,
            "cascade_evaluation": unified.evaluator.cascade_evaluation,
            "cascade_thresholds": unified.evaluator.cascade_thresholds,
            "parallel_evaluations": unified.evaluator.parallel_evaluations,
            "use_llm_feedback": unified.evaluator.use_llm_feedback,
            "llm_feedback_weight": unified.evaluator.llm_feedback_weight,
            "enable_artifacts": unified.evaluator.enable_artifacts,
            "max_artifact_storage": unified.evaluator.max_artifact_storage,
        }

        # Mode-specific
        if mode == EvolutionMode.QD or unified.qd.enabled:
            oe_config["evolution_mode"] = "qd"
            oe_config["qd"] = {
                "grid_resolution": unified.qd.grid_resolution,
                "feature_dimensions": unified.qd.feature_dimensions or unified.database.feature_dimensions,
                "archive_size": unified.qd.archive_size,
                "use_cvt_map_elites": unified.qd.use_cvt_map_elites,
                "cvt_samples": unified.qd.cvt_samples,
            }
        elif mode == EvolutionMode.MO or unified.mo.enabled:
            oe_config["evolution_mode"] = "mo"
            oe_config["mo"] = {
                "objectives": unified.mo.objectives,
                "objective_weights": unified.mo.objective_weights,
                "algorithm": unified.mo.algorithm,
                "pareto_size": unified.mo.pareto_size,
                "use_constraint_domination": unified.mo.use_constraint_domination,
            }
        elif mode == EvolutionMode.ADVERSARIAL or unified.adversarial.enabled:
            oe_config["evolution_mode"] = "adversarial"
            oe_config["adversarial"] = {
                "adversarial_rounds": unified.adversarial.adversarial_rounds,
                "red_team_models": unified.adversarial.red_team_models,
                "blue_team_models": unified.adversarial.blue_team_models,
                "robustness_threshold": unified.adversarial.robustness_threshold,
            }
        else:
            oe_config["evolution_mode"] = "standard"

        return oe_config

    @staticmethod
    def to_qd_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to QD (Quality-Diversity) format

        Returns:
            Dictionary with QD-specific parameters
        """
        return {
            "evolution_mode": "qd",
            "grid_resolution": unified.qd.grid_resolution,
            "feature_dimensions": unified.qd.feature_dimensions or unified.database.feature_dimensions,
            "archive_size": unified.qd.archive_size,
            "use_cvt_map_elites": unified.qd.use_cvt_map_elites,
            "cvt_samples": unified.qd.cvt_samples,
            # Database parameters
            "num_islands": unified.database.num_islands,
            "migration_interval": unified.database.migration_interval,
            "migration_rate": unified.database.migration_rate,
        }

    @staticmethod
    def to_mo_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to MO (Multi-Objective) format

        Returns:
            Dictionary with MO-specific parameters
        """
        return {
            "evolution_mode": "mo",
            "objectives": unified.mo.objectives,
            "objective_weights": unified.mo.objective_weights,
            "algorithm": unified.mo.algorithm,
            "pareto_size": unified.mo.pareto_size,
            "use_constraint_domination": unified.mo.use_constraint_domination,
        }

    @staticmethod
    def to_adversarial_config(unified: UnifiedEvolutionConfig) -> Dict[str, Any]:
        """
        Convert unified config to Adversarial format

        Returns:
            Dictionary with adversarial-specific parameters
        """
        return {
            "evolution_mode": "adversarial",
            "adversarial_rounds": unified.adversarial.adversarial_rounds,
            "red_team_models": unified.adversarial.red_team_models,
            "blue_team_models": unified.adversarial.blue_team_models,
            "robustness_threshold": unified.adversarial.robustness_threshold,
        }

    @staticmethod
    def from_openevolve_dict(oe_config: Dict[str, Any]) -> UnifiedEvolutionConfig:
        """
        Convert OpenEvolve config dict to unified config

        Args:
            oe_config: OpenEvolve configuration dictionary

        Returns:
            UnifiedEvolutionConfig instance
        """
        # Extract general settings
        general = {
            "max_iterations": oe_config.get("max_iterations", 10000),
            "checkpoint_interval": oe_config.get("checkpoint_interval", 100),
            "random_seed": oe_config.get("random_seed", 42),
            "diff_based_evolution": oe_config.get("diff_based_evolution", True),
            "max_code_length": oe_config.get("max_code_length", 10000),
            "language": oe_config.get("language"),
            "early_stopping_patience": oe_config.get("early_stopping_patience"),
            "convergence_threshold": oe_config.get("convergence_threshold", 0.001),
            "early_stopping_metric": oe_config.get("early_stopping_metric", "combined_score"),
        }

        # Extract database config
        db = oe_config.get("database", {})
        database = {
            "population_size": db.get("population_size", 1000),
            "archive_size": db.get("archive_size", 100),
            "num_islands": db.get("num_islands", 5),
            "elite_selection_ratio": db.get("elite_selection_ratio", 0.1),
            "exploration_ratio": db.get("exploration_ratio", 0.2),
            "exploitation_ratio": db.get("exploitation_ratio", 0.7),
            "feature_dimensions": db.get("feature_dimensions", ["complexity", "diversity"]),
            "feature_bins": db.get("feature_bins", 10),
            "migration_interval": db.get("migration_interval", 50),
            "migration_rate": db.get("migration_rate", 0.1),
            "log_prompts": db.get("log_prompts", True),
        }

        # Extract LLM config
        llm_cfg = oe_config.get("llm", {})
        llm = {
            "temperature": llm_cfg.get("temperature", 0.7),
            "top_p": llm_cfg.get("top_p", 0.95),
            "max_tokens": llm_cfg.get("max_tokens", 4096),
            "timeout": llm_cfg.get("timeout", 60),
            "retries": llm_cfg.get("retries", 3),
            "random_seed": llm_cfg.get("random_seed", 42),
        }

        # Extract evaluator config
        eval_cfg = oe_config.get("evaluator", {})
        evaluator = {
            "timeout": eval_cfg.get("timeout", 300),
            "max_retries": eval_cfg.get("max_retries", 3),
            "cascade_evaluation": eval_cfg.get("cascade_evaluation", True),
            "cascade_thresholds": eval_cfg.get("cascade_thresholds", [0.5, 0.75, 0.9]),
            "parallel_evaluations": eval_cfg.get("parallel_evaluations", 4),
        }

        # Determine mode
        evolution_mode = oe_config.get("evolution_mode", "standard")
        if evolution_mode == "qd":
            qd_cfg = oe_config.get("qd", {})
            qd = {"enabled": True, "grid_resolution": qd_cfg.get("grid_resolution", 10)}
        else:
            qd = {"enabled": False}

        # Create unified config
        return UnifiedEvolutionConfig(
            **general,
            database=database,
            llm=llm,
            evaluator=evaluator,
            qd=qd,
        )

    @staticmethod
    def from_pes_dict(pes_config: Dict[str, Any]) -> UnifiedEvolutionConfig:
        """
        Convert LoongFlow PES config dict to unified config

        Args:
            pes_config: LoongFlow PES configuration dictionary

        Returns:
            UnifiedEvolutionConfig instance
        """
        # Extract task config
        task = pes_config.get("task", {})
        general = {
            "max_iterations": task.get("max_iterations", 10000),
            "time_limit_seconds": task.get("time_limit_seconds"),
            "target_fitness": task.get("target_fitness"),
            "domain": task.get("domain", "general"),
        }

        # Extract evolve config
        evolve = pes_config.get("evolve", {})
        pes = {
            "enabled": True,
            "enable_planning": evolve.get("enable_planning", True),
            "max_rounds": evolve.get("max_rounds", 3),
        }

        # Extract database config
        db = pes_config.get("database", {})
        database = {
            "num_islands": db.get("num_islands", 5),
            "population_size": db.get("population_size", 1000),
            "enable_memory": evolve.get("enable_memory", True),
            "exploration_rate": db.get("exploration_rate", 0.2),
            "adaptive_exploration": db.get("adaptive_exploration", True),
        }

        # Extract LLM config
        llm_cfg = pes_config.get("llm", {})
        llm = {
            "temperature": llm_cfg.get("temperature", 0.7),
            "timeout": llm_cfg.get("timeout", 60),
            "retries": llm_cfg.get("retries", 3),
        }

        # Extract evaluator config
        executor = pes_config.get("executor", {})
        evaluator = {
            "early_stopping": evolve.get("early_stopping", True),
            "early_stopping_patience": evolve.get("early_stopping_patience", 5),
            "early_stopping_threshold": evolve.get("early_stopping_threshold", 0.01),
        }

        # Create unified config
        return UnifiedEvolutionConfig(
            **general,
            evolution_mode=EvolutionMode.PES,
            pes=pes,
            database=database,
            llm=llm,
            evaluator=evaluator,
        )
