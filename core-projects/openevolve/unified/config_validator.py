"""
Config Validator
Validates unified configuration and detects conflicts

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import List, Dict, Any, Optional, Tuple
from .config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig
)


class ValidationError:
    """Represents a single validation error"""

    def __init__(self, category: str, message: str, severity: str = "error"):
        self.category = category
        self.message = message
        self.severity = severity  # "error", "warning", "info"

    def __str__(self):
        return f"[{self.severity.upper()}] {self.category}: {self.message}"


class ConfigValidator:
    """
    Validate configuration and detect conflicts

    Checks:
    - Mode compatibility
    - Parameter conflicts
    - Resource constraints
    - Domain-specific validation
    - LLM configuration sanity
    """

    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate(self) -> Tuple[List[ValidationError], List[ValidationError]]:
        """
        Run full validation

        Returns:
            Tuple of (errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Determine mode
        mode = self._determine_mode()

        # Check mode-specific validations
        self._check_mode_compatibility(mode)
        self._check_parameter_conflicts(mode)
        self._check_resource_constraints()
        self._check_llm_configuration()
        self._check_database_configuration()
        self._check_evaluator_configuration()

        # Domain-specific checks
        self._check_domain_validation()

        return self.errors, self.warnings

    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)"""
        errors, warnings = self.validate()
        return len(errors) == 0

    def _determine_mode(self) -> EvolutionMode:
        """Determine which mode will be used"""
        if self.config.evolution_mode != EvolutionMode.AUTO:
            return self.config.evolution_mode

        # Auto-select based on enabled configs
        if self.config.pes.enabled:
            return EvolutionMode.PES
        elif self.config.qd.enabled:
            return EvolutionMode.QD
        elif self.config.mo.enabled:
            return EvolutionMode.MO
        elif self.config.adversarial.enabled:
            return EvolutionMode.ADVERSARIAL
        else:
            return EvolutionMode.STANDARD

    def _check_mode_compatibility(self, mode: EvolutionMode):
        """Check if mode is compatible with enabled configs"""

        # PES mode checks
        if mode == EvolutionMode.PES:
            if not self.config.pes.enable_planning:
                self.warnings.append(ValidationError(
                    "mode",
                    "PES mode enabled but planning phase is disabled"
                ))

            if not self.config.database.enable_memory:
                self.warnings.append(ValidationError(
                    "mode",
                    "PES works best with memory enabled"
                ))

        # QD mode checks
        if mode == EvolutionMode.QD:
            if len(self.config.database.feature_dimensions) < 1:
                self.errors.append(ValidationError(
                    "mode",
                    "QD mode requires at least one feature dimension"
                ))

            if self.config.database.num_islands < 2:
                self.warnings.append(ValidationError(
                    "mode",
                    "QD mode benefits from multiple islands (num_islands >= 2)"
                ))

        # MO mode checks
        if mode == EvolutionMode.MO:
            if not self.config.mo.objectives or len(self.config.mo.objectives) < 2:
                self.errors.append(ValidationError(
                    "mode",
                    "Multi-objective mode requires at least 2 objectives"
                ))

        # Adversarial mode checks
        if mode == EvolutionMode.ADVERSARIAL:
            if not self.config.adversarial.red_team_models:
                self.errors.append(ValidationError(
                    "mode",
                    "Adversarial mode requires red_team_models"
                ))

    def _check_parameter_conflicts(self, mode: EvolutionMode):
        """Check for conflicting parameters"""

        # Check selection ratios
        total_ratio = (
            self.config.database.elite_selection_ratio +
            self.config.database.exploration_ratio +
            self.config.database.exploitation_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            self.warnings.append(ValidationError(
                "database",
                f"Selection ratios sum to {total_ratio:.2f} (expected 1.0)"
            ))

        # Check PES-specific conflicts
        if mode != EvolutionMode.PES:
            if self.config.pes.enabled:
                self.warnings.append(ValidationError(
                    "mode",
                    "PES config is enabled but mode is not PES"
                ))

        # Check QD-specific conflicts
        if mode != EvolutionMode.QD:
            if self.config.qd.enabled:
                self.warnings.append(ValidationError(
                    "mode",
                    "QD config is enabled but mode is not QD"
                ))

        # Check MO-specific conflicts
        if mode != EvolutionMode.MO:
            if self.config.mo.enabled:
                self.warnings.append(ValidationError(
                    "mode",
                    "MO config is enabled but mode is not MO"
                ))

        # Check adversarial-specific conflicts
        if mode != EvolutionMode.ADVERSARIAL:
            if self.config.adversarial.enabled:
                self.warnings.append(ValidationError(
                    "mode",
                    "Adversarial config is enabled but mode is not adversarial"
                ))

        # Check conflicting evolution modes
        enabled_modes = sum([
            self.config.pes.enabled,
            self.config.qd.enabled,
            self.config.mo.enabled,
            self.config.adversarial.enabled,
        ])
        if enabled_modes > 1:
            self.errors.append(ValidationError(
                "mode",
                f"Multiple evolution modes enabled ({enabled_modes}). Only one should be enabled."
            ))

    def _check_resource_constraints(self):
        """Check resource constraints"""

        # Check iteration count
        if self.config.max_iterations > 100000:
            self.warnings.append(ValidationError(
                "resources",
                f"max_iterations ({self.config.max_iterations}) is very high and may take a long time"
            ))

        # Check population size
        if self.config.database.population_size > 10000:
            self.warnings.append(ValidationError(
                "resources",
                f"population_size ({self.config.database.population_size}) may use excessive memory"
            ))

        # Check num_islands vs population_size
        if self.config.database.num_islands > self.config.database.population_size:
            self.errors.append(ValidationError(
                "resources",
                f"num_islands ({self.config.database.num_islands}) > population_size ({self.config.database.population_size})"
            ))

        # Check archive size
        if self.config.database.archive_size > self.config.database.population_size:
            self.warnings.append(ValidationError(
                "resources",
                f"archive_size ({self.config.database.archive_size}) > population_size ({self.config.database.population_size})"
            ))

        # Check checkpoint frequency
        if self.config.checkpoint_interval > self.config.max_iterations:
            self.warnings.append(ValidationError(
                "resources",
                "checkpoint_interval > max_iterations (no checkpoints will be saved)"
            ))

    def _check_llm_configuration(self):
        """Check LLM configuration sanity"""

        # Check if models are configured
        if not self.config.llm.models and not self.config.llm.planner_models:
            self.errors.append(ValidationError(
                "llm",
                "No LLM models configured. At least one model is required."
            ))

        # Check PES LLM requirements
        if self.config.evolution_mode == EvolutionMode.PES:
            if not self.config.llm.planner_models:
                self.warnings.append(ValidationError(
                    "llm",
                    "PES mode benefits from explicit planner_models configuration"
                ))

        # Check temperature ranges
        if self.config.llm.temperature < 0.0 or self.config.llm.temperature > 2.0:
            self.errors.append(ValidationError(
                "llm",
                f"temperature ({self.config.llm.temperature}) out of range [0.0, 2.0]"
            ))

        if self.config.llm.plan_temperature < 0.0 or self.config.llm.plan_temperature > 2.0:
            self.errors.append(ValidationError(
                "llm",
                f"plan_temperature ({self.config.llm.plan_temperature}) out of range [0.0, 2.0]"
            ))

        # Check ensemble weights
        for model in self.config.llm.models:
            if model.weight < 0.0:
                self.errors.append(ValidationError(
                    "llm",
                    f"Model {model.name} has negative weight: {model.weight}"
                ))

        # Check timeout
        if self.config.llm.timeout < 1:
            self.warnings.append(ValidationError(
                "llm",
                f"LLM timeout ({self.config.llm.timeout}s) is very short"
            ))

        if self.config.llm.timeout > 600:
            self.warnings.append(ValidationError(
                "llm",
                f"LLM timeout ({self.config.llm.timeout}s) is very long"
            ))

    def _check_database_configuration(self):
        """Check database configuration"""

        # Check feature dimensions
        if not self.config.database.feature_dimensions:
            self.errors.append(ValidationError(
                "database",
                "No feature_dimensions specified"
            ))

        # Check feature_bins
        if isinstance(self.config.database.feature_bins, int):
            if self.config.database.feature_bins < 2:
                self.errors.append(ValidationError(
                    "database",
                    f"feature_bins ({self.config.database.feature_bins}) must be >= 2"
                ))
            if self.config.database.feature_bins > 100:
                self.warnings.append(ValidationError(
                    "database",
                    f"feature_bins ({self.config.database.feature_bins}) is very high, may cause sparse grids"
                ))

        # Check migration parameters
        if self.config.database.migration_interval < 1:
            self.errors.append(ValidationError(
                "database",
                f"migration_interval ({self.config.database.migration_interval}) must be >= 1"
            ))

        if self.config.database.migration_rate < 0.0 or self.config.database.migration_rate > 1.0:
            self.errors.append(ValidationError(
                "database",
                f"migration_rate ({self.config.database.migration_rate}) out of range [0.0, 1.0]"
            ))

        # Check num_islands
        if self.config.database.num_islands > 20:
            self.warnings.append(ValidationError(
                "database",
                f"num_islands ({self.config.database.num_islands}) is high, may slow convergence"
            ))

    def _check_evaluator_configuration(self):
        """Check evaluator configuration"""

        # Check timeout
        if self.config.evaluator.timeout < 1:
            self.errors.append(ValidationError(
                "evaluator",
                f"timeout ({self.config.evaluator.timeout}s) must be >= 1"
            ))

        # Check cascade thresholds
        if self.config.evaluator.cascade_evaluation:
            thresholds = self.config.evaluator.cascade_thresholds
            if not thresholds:
                self.errors.append(ValidationError(
                    "evaluator",
                    "cascade_evaluation enabled but no cascade_thresholds specified"
                ))
            else:
                # Check monotonically increasing
                for i in range(len(thresholds) - 1):
                    if thresholds[i] >= thresholds[i + 1]:
                        self.errors.append(ValidationError(
                            "evaluator",
                            f"cascade_thresholds must be monotonically increasing: {thresholds}"
                        ))

        # Check parallel evaluations
        if self.config.evaluator.parallel_evaluations > 100:
            self.warnings.append(ValidationError(
                "evaluator",
                f"parallel_evaluations ({self.config.evaluator.parallel_evaluations}) is very high"
            ))

    def _check_domain_validation(self):
        """Domain-specific validation"""

        domain = self.config.domain.value

        # Finance domain checks
        if domain == "finance":
            if not self.config.mo.enabled:
                self.warnings.append(ValidationError(
                    "domain",
                    "Finance problems often benefit from multi-objective optimization (return vs risk)"
                ))

        # Trading domain checks
        if domain == "trading":
            if self.config.evaluator.cascade_evaluation:
                self.warnings.append(ValidationError(
                    "domain",
                    "Trading strategies should use full backtests (cascade_evaluation may hide overfitting)"
                ))

        # Science domain checks
        if domain == "science":
            if self.config.evolution_mode == EvolutionMode.PES:
                pass  # PES is good for science
            else:
                self.warnings.append(ValidationError(
                    "domain",
                    "Scientific experiments may benefit from PES mode"
                ))

        # Engineering domain checks
        if domain == "engineering":
            if self.config.evaluator.timeout < 300:
                self.warnings.append(ValidationError(
                    "domain",
                    "Engineering simulations (FEA/CFD) often require longer timeouts"
                ))

        # Math domain checks
        if domain == "math":
            if self.config.evolution_mode == EvolutionMode.QD:
                self.warnings.append(ValidationError(
                    "domain",
                    "Math problems may not benefit from QD mode (single objective usually)"
                ))

        # ML domain checks
        if domain == "ml":
            if self.config.evolution_mode == EvolutionMode.PES:
                pass  # PES is good for ML
            else:
                self.warnings.append(ValidationError(
                    "domain",
                    "ML hyperparameter tuning may benefit from PES mode"
                ))


def validate_config(config: UnifiedEvolutionConfig) -> Tuple[List[ValidationError], List[ValidationError]]:
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
        config: UnifiedEvolutionConfig to validate

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator(config)
    return validator.is_valid()
