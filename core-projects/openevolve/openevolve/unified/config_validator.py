"""
Configuration Validator

Validates unified configuration and detects conflicts, incompatibilities,
and resource constraints.
"""

from typing import Any, Dict, List, Optional, Tuple
from .config import (
    UnifiedEvolutionConfig,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    OpenEvolveConfig,
)


class ConfigValidator:
    """
    Validates configuration and detects conflicts

    Provides comprehensive validation including:
    - Mode compatibility checks
    - Parameter conflicts
    - Resource constraints
    - Feature dimension validity
    - LLM model compatibility
    """

    def __init__(self, config: UnifiedEvolutionConfig):
        """
        Initialize validator with configuration

        Args:
            config: UnifiedEvolutionConfig to validate
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[List[str], List[str]]:
        """
        Run all validation checks

        Returns:
            Tuple of (errors, warnings)
            - errors: Critical issues that prevent execution
            - warnings: Non-critical issues that should be reviewed
        """
        self.errors = []
        self.warnings = []

        # Run all validation checks
        self._check_mode_compatibility()
        self._check_parameter_conflicts()
        self._check_resource_constraints()
        self._check_feature_dimensions()
        self._check_llm_configuration()
        self._check_database_configuration()
        self._check_evaluator_configuration()
        self._check_mode_specific_configs()

        return self.errors, self.warnings

    def _check_mode_compatibility(self) -> None:
        """Check if selected modes are compatible with each other"""
        mode = self.config.evolution_mode
        enabled_modes = self.config.enable_modes

        # Check if evolution_mode is in enabled_modes
        if mode not in enabled_modes:
            self.errors.append(
                f"Evolution mode '{mode}' is not in enabled_modes: {enabled_modes}"
            )

        # Check for incompatible mode combinations
        if "adversarial" in enabled_modes and "mo" in enabled_modes:
            self.warnings.append(
                "Adversarial and Multi-Objective modes may conflict. "
                "Consider using hybrid mode with careful configuration."
            )

        if "pes" in enabled_modes and self.config.qd is None:
            self.warnings.append(
                "PES mode is enabled but no QD config provided. "
                "PES works best with Quality Diversity optimization."
            )

    def _check_parameter_conflicts(self) -> None:
        """Check for conflicting parameter values"""
        db = self.config.database
        common = self.config.common

        # Check selection ratios sum to 1.0
        total_ratio = db.elite_selection_ratio + db.exploration_rate + db.exploitation_ratio
        if abs(total_ratio - 1.0) > 0.01:
            self.warnings.append(
                f"Selection ratios sum to {total_ratio:.2f}, expected 1.0. "
                f"elite={db.elite_selection_ratio}, exploration={db.exploration_rate}, "
                f"exploitation={db.exploitation_ratio}"
            )

        # Check migration parameters
        if db.migration_interval >= common.max_iterations:
            self.warnings.append(
                f"Migration interval ({db.migration_interval}) >= max_iterations "
                f"({common.max_iterations}). Migration will never occur."
            )

        # Check concurrency vs population
        if common.concurrency > db.population_size:
            self.warnings.append(
                f"Concurrency ({common.concurrency}) > population_size "
                f"({db.population_size}). Reducing concurrency to population size."
            )

        # Check if checkpoint_interval makes sense
        if common.checkpoint_interval > common.max_iterations:
            self.warnings.append(
                f"Checkpoint interval ({common.checkpoint_interval}) > max_iterations "
                f"({common.max_iterations}). Only one checkpoint will be saved at the end."
            )

    def _check_resource_constraints(self) -> None:
        """Check if resource constraints are reasonable"""
        db = self.config.database
        evaluator = self.config.evaluator

        # Check population size vs archive size
        if db.elite_archive_size >= db.population_size:
            self.warnings.append(
                f"Elite archive size ({db.elite_archive_size}) >= population size "
                f"({db.population_size}). Archive may contain entire population."
            )

        # Check if population size is too small for number of islands
        if db.population_size < db.num_islands * 10:
            self.warnings.append(
                f"Population size ({db.population_size}) is small for {db.num_islands} islands. "
                f"Recommend at least {db.num_islands * 10} population size."
            )

        # Check timeout constraints
        if evaluator.timeout < 10:
            self.warnings.append(
                f"Evaluator timeout ({evaluator.timeout}s) is very short. "
                f"Consider increasing to at least 60s for complex evaluations."
            )

        # Calculate max total evaluation time
        max_total_time = (
            evaluator.timeout
            * evaluator.max_retries
            * self.config.common.max_iterations
            * self.config.common.concurrency
        )
        if max_total_time > 86400:  # 24 hours
            self.warnings.append(
                f"Maximum total evaluation time is {max_total_time / 3600:.1f} hours. "
                f"Consider reducing timeout, max_iterations, or concurrency."
            )

        # Check memory limits
        if evaluator.memory_limit_mb and evaluator.memory_limit_mb < 100:
            self.warnings.append(
                f"Memory limit ({evaluator.memory_limit_mb}MB) is very low. "
                f"May cause out-of-memory errors."
            )

    def _check_feature_dimensions(self) -> None:
        """Check if feature dimensions are valid"""
        db = self.config.database

        # Check if feature_dimensions is empty
        if not db.feature_dimensions:
            self.errors.append(
                "feature_dimensions cannot be empty. "
                "At least one feature dimension must be specified."
            )

        # Check if feature_bins is valid
        if isinstance(db.feature_bins, dict):
            for dim in db.feature_dimensions:
                if dim not in db.feature_bins:
                    self.errors.append(
                        f"Feature dimension '{dim}' not found in feature_bins dict. "
                        f"Dimensions: {db.feature_dimensions}, Bins: {list(db.feature_bins.keys())}"
                    )
        elif db.feature_bins < 2:
            self.errors.append(
                f"feature_bins must be >= 2, got {db.feature_bins}"
            )

        # Check for duplicate dimensions
        if len(db.feature_dimensions) != len(set(db.feature_dimensions)):
            self.errors.append(
                f"Duplicate feature dimensions found: {db.feature_dimensions}"
            )

        # Check QD-specific settings
        if self.config.qd:
            qd = self.config.qd
            if qd.grid_resolution < 2:
                self.errors.append(
                    f"QD grid_resolution must be >= 2, got {qd.grid_resolution}"
                )

            if qd.grid_resolution > 100:
                self.warnings.append(
                    f"QD grid_resolution ({qd.grid_resolution}) is very high. "
                    f"This may require {qd.grid_resolution ** len(qd.grid_dimensions)} cells. "
                    f"Consider reducing to <= 50."
                )

            # Check if grid dimensions match database feature dimensions
            if set(qd.grid_dimensions) != set(db.feature_dimensions):
                self.warnings.append(
                    f"QD grid_dimensions {qd.grid_dimensions} don't match "
                    f"database feature_dimensions {db.feature_dimensions}"
                )

        # Check MO-specific settings
        if self.config.mo:
            mo = self.config.mo
            if not mo.objectives:
                self.errors.append(
                    "MO objectives cannot be empty. At least one objective must be specified."
                )

            # Check if objective weights match objectives
            if mo.objective_weights:
                if set(mo.objective_weights.keys()) != set(mo.objectives):
                    self.warnings.append(
                        f"MO objective_weights don't match objectives. "
                        f"Objectives: {mo.objectives}, Weights: {list(mo.objective_weights.keys())}"
                    )

            # Check if optimization directions match objectives
            if set(mo.optimization_direction.keys()) != set(mo.objectives):
                self.errors.append(
                    f"MO optimization_direction must match objectives. "
                    f"Objectives: {mo.objectives}, Directions: {list(mo.optimization_direction.keys())}"
                )

    def _check_llm_configuration(self) -> None:
        """Check if LLM configuration is valid"""
        llm = self.config.llm

        # Check if models list is empty
        if not llm.models:
            self.errors.append(
                "LLM models list cannot be empty. At least one model must be specified."
            )

        # Check model weights
        total_weight = sum(m.weight for m in llm.models)
        if total_weight == 0:
            self.errors.append(
                "Total model weight is 0. At least one model must have weight > 0."
            )

        # Check if evaluator_models is configured
        if not llm.evaluator_models:
            self.warnings.append(
                "No evaluator_models specified. Will use evolution models for evaluation."
            )
        else:
            eval_weight = sum(m.weight for m in llm.evaluator_models)
            if eval_weight == 0:
                self.errors.append(
                    "Total evaluator model weight is 0. At least one model must have weight > 0."
                )

        # Check if API key is available
        has_api_key = (
            llm.default_api_key
            or any(m.api_key for m in llm.models)
            or any(m.api_key for m in llm.evaluator_models)
        )
        if not has_api_key:
            self.warnings.append(
                "No API key configured. Will rely on environment variables (OPENAI_API_KEY, etc.)."
            )

        # Check model context lengths
        for model in llm.models:
            if model.max_tokens > model.context_length:
                self.errors.append(
                    f"Model '{model.name}' has max_tokens ({model.max_tokens}) > "
                    f"context_length ({model.context_length})."
                )

    def _check_database_configuration(self) -> None:
        """Check if database configuration is valid"""
        db = self.config.database

        # Check storage type
        valid_storage_types = ["in_memory", "redis", "file", "database"]
        if db.storage_type not in valid_storage_types:
            self.errors.append(
                f"Invalid storage_type '{db.storage_type}'. "
                f"Must be one of: {valid_storage_types}"
            )

        # Check Redis configuration
        if db.storage_type == "redis":
            if not db.redis_url or db.redis_url == "redis://localhost:6379/0":
                self.warnings.append(
                    "Using Redis storage with default URL. Ensure Redis is running locally."
                )

        # Check file storage configuration
        if db.storage_type == "file" and not db.db_path:
            self.errors.append(
                "File storage selected but db_path not specified."
            )

        # Check if checkpoint interval is reasonable
        if db.checkpoint_interval > self.config.common.max_iterations:
            self.warnings.append(
                f"Database checkpoint_interval ({db.checkpoint_interval}) > "
                f"max_iterations ({self.config.common.max_iterations}). "
                f"Only one checkpoint will be saved at the end."
            )

    def _check_evaluator_configuration(self) -> None:
        """Check if evaluator configuration is valid"""
        evaluator = self.config.evaluator

        # Check if evaluate_code is provided
        if not evaluator.evaluate_code:
            self.warnings.append(
                "No evaluate_code provided. Evolution may not work properly."
            )

        # Check cascade thresholds
        if evaluator.cascade_evaluation:
            if not evaluator.cascade_thresholds:
                self.errors.append(
                    "Cascade evaluation enabled but no cascade_thresholds provided."
                )
            else:
                # Check if thresholds are sorted
                if evaluator.cascade_thresholds != sorted(evaluator.cascade_thresholds):
                    self.warnings.append(
                        "Cascade thresholds should be in ascending order."
                    )

                # Check if thresholds are in valid range
                for threshold in evaluator.cascade_thresholds:
                    if not (0.0 <= threshold <= 1.0):
                        self.errors.append(
                            f"Cascade threshold {threshold} is not in [0.0, 1.0] range."
                        )

        # Check distributed evaluation
        if evaluator.distributed and evaluator.parallel_evaluations < 2:
            self.warnings.append(
                "Distributed evaluation enabled but parallel_evaluations < 2. "
                "Distributed mode requires multiple workers."
            )

        # Check LLM feedback
        if evaluator.use_llm_feedback and not self.config.llm.evaluator_models:
            self.warnings.append(
                "LLM feedback enabled but no evaluator_models configured. "
                "Feedback will use evolution models."
            )

    def _check_mode_specific_configs(self) -> None:
        """Check mode-specific configurations"""
        # Check PES config
        if self.config.pes:
            pes = self.config.pes

            # Check if planning is enabled
            if pes.enable_planning and not pes.planner_type:
                self.errors.append(
                    "PES planning enabled but planner_type not specified."
                )

            # Check executor configuration
            if not pes.executor_type:
                self.errors.append(
                    "PES executor_type not specified."
                )

            # Check if code execution is enabled
            if not pes.enable_code_execution:
                self.warnings.append(
                    "PES code execution disabled. Evolution may be limited."
                )

        # Check Adversarial config
        if self.config.adversarial:
            adv = self.config.adversarial

            if adv.enable_adversarial:
                if adv.num_adversaries < 2:
                    self.errors.append(
                        f"Adversarial mode requires at least 2 adversaries, got {adv.num_adversaries}"
                    )

                if adv.balance_factor <= 0 or adv.balance_factor >= 1:
                    self.warnings.append(
                        f"Adversarial balance_factor {adv.balance_factor} is extreme. "
                        f"Consider values closer to 0.5 for balanced evolution."
                    )

        # Check OpenEvolve config
        if self.config.openevolve:
            oe = self.config.openevolve

            # Check code length settings
            if oe.max_code_length < oe.suggest_simplification_after_chars:
                self.warnings.append(
                    f"max_code_length ({oe.max_code_length}) < "
                    f"suggest_simplification_after_chars ({oe.suggest_simplification_after_chars}). "
                    f"Simplification suggestions may never trigger."
                )

            # Check early stopping
            if oe.early_stopping_patience and oe.early_stopping_patience > self.config.common.max_iterations:
                self.warnings.append(
                    f"early_stopping_patience ({oe.early_stopping_patience}) >= "
                    f"max_iterations ({self.config.common.max_iterations}). "
                    f"Early stopping will never trigger."
                )

    def is_valid(self) -> bool:
        """
        Quick check if configuration is valid

        Returns:
            True if no errors found, False otherwise
        """
        errors, _ = self.validate()
        return len(errors) == 0

    def get_validation_report(self) -> str:
        """
        Get detailed validation report

        Returns:
            Formatted string with all errors and warnings
        """
        errors, warnings = self.validate()

        lines = []
        lines.append("=" * 80)
        lines.append("CONFIGURATION VALIDATION REPORT")
        lines.append("=" * 80)

        if not errors and not warnings:
            lines.append("[OK] Configuration is valid! No issues found.")
        else:
            if errors:
                lines.append(f"\n[FAIL] ERRORS ({len(errors)}):")
                lines.append("-" * 80)
                for i, error in enumerate(errors, 1):
                    lines.append(f"{i}. {error}")

            if warnings:
                lines.append(f"\n[WARN]  WARNINGS ({len(warnings)}):")
                lines.append("-" * 80)
                for i, warning in enumerate(warnings, 1):
                    lines.append(f"{i}. {warning}")

        lines.append("=" * 80)

        return "\n".join(lines)


def validate_config(config: UnifiedEvolutionConfig) -> Tuple[List[str], List[str]]:
    """
    Convenience function to validate a configuration

    Args:
        config: UnifiedEvolutionConfig to validate

    Returns:
        Tuple of (errors, warnings)
    """
    validator = ConfigValidator(config)
    return validator.validate()


def is_valid_config(config: UnifiedEvolutionConfig) -> bool:
    """
    Convenience function to check if configuration is valid

    Args:
        config: UnifiedEvolutionConfig to check

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator(config)
    return validator.is_valid()
