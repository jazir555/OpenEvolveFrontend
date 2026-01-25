"""
ROMA Reliability Adapter Configuration
======================================

Configuration settings for the ROMA reliability adapter.

This configuration extends the base reliability configuration with
ROMA-specific settings.

Author: OpenEvolve Team
Version: 1.0.0
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class RomaAdapterConfig:
    """
    Configuration for ROMA Reliability Adapter.

    Attributes:
        enabled: Enable the ROMA reliability adapter
        lmql_enabled: Enable LMQL constraint layer
        guardrails_enabled: Enable Guardrails validation layer
        max_depth_default: Default maximum decomposition depth
        execution_mode_default: Default execution mode ("recursive" or "event_driven")
        enable_checkpoints_default: Default checkpointing enabled state
        constraint_defaults: Default constraint values
        validation_defaults: Default validation settings
    """

    # Layer enablement
    enabled: bool = True
    lmql_enabled: bool = True
    guardrails_enabled: bool = True

    # ROMA defaults
    max_depth_default: int = 3
    execution_mode_default: str = "recursive"  # "recursive" or "event_driven"
    enable_checkpoints_default: bool = True

    # LMQL constraint defaults
    constraint_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 3,
        "max_subtasks": 10,
        "subtask_token_limit": 500,
        "require_json": False
    })

    # Guardrails validation defaults
    validation_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "input_validators": ["roma_length", "toxic_language"],
        "output_validators": ["json_structure", "roma_depth"],
        "on_fail": "fix_reask"
    })

    # Fallback behavior
    fallback_on_error: bool = True
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> 'RomaAdapterConfig':
        """
        Load configuration from environment variables.

        Environment Variables:
            ROMA_ADAPTER_ENABLED: Enable/disable adapter (default: true)
            ROMA_LMQL_ENABLED: Enable LMQL layer (default: true)
            ROMA_GUARDRAILS_ENABLED: Enable Guardrails layer (default: true)
            ROMA_MAX_DEPTH: Default max depth (default: 3)
            ROMA_EXECUTION_MODE: Default execution mode (default: recursive)
            ROMA_CHECKPOINTS: Enable checkpoints (default: true)
            ROMA_FALLBACK: Enable fallback on error (default: true)
            ROMA_MAX_RETRIES: Maximum retry attempts (default: 3)

        Returns:
            RomaAdapterConfig instance
        """
        return cls(
            enabled=os.getenv("ROMA_ADAPTER_ENABLED", "true").lower() == "true",
            lmql_enabled=os.getenv("ROMA_LMQL_ENABLED", "true").lower() == "true",
            guardrails_enabled=os.getenv("ROMA_GUARDRAILS_ENABLED", "true").lower() == "true",
            max_depth_default=int(os.getenv("ROMA_MAX_DEPTH", "3")),
            execution_mode_default=os.getenv("ROMA_EXECUTION_MODE", "recursive"),
            enable_checkpoints_default=os.getenv("ROMA_CHECKPOINTS", "true").lower() == "true",
            fallback_on_error=os.getenv("ROMA_FALLBACK", "true").lower() == "true",
            max_retries=int(os.getenv("ROMA_MAX_RETRIES", "3"))
        )

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate max_depth
        if self.max_depth_default < 1 or self.max_depth_default > 10:
            errors.append(f"max_depth_default must be 1-10, got: {self.max_depth_default}")

        # Validate execution_mode
        if self.execution_mode_default not in ["recursive", "event_driven"]:
            errors.append(f"execution_mode must be 'recursive' or 'event_driven', got: {self.execution_mode_default}")

        # Validate max_retries
        if self.max_retries < 0 or self.max_retries > 10:
            errors.append(f"max_retries must be 0-10, got: {self.max_retries}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "enabled": self.enabled,
            "lmql_enabled": self.lmql_enabled,
            "guardrails_enabled": self.guardrails_enabled,
            "max_depth_default": self.max_depth_default,
            "execution_mode_default": self.execution_mode_default,
            "enable_checkpoints_default": self.enable_checkpoints_default,
            "constraint_defaults": self.constraint_defaults,
            "validation_defaults": self.validation_defaults,
            "fallback_on_error": self.fallback_on_error,
            "max_retries": self.max_retries
        }


# Global configuration instance
_global_config: Optional[RomaAdapterConfig] = None


def get_config() -> RomaAdapterConfig:
    """
    Get global ROMA adapter configuration.

    Returns:
        RomaAdapterConfig instance (creates from env if not exists)
    """
    global _global_config
    if _global_config is None:
        _global_config = RomaAdapterConfig.from_env()

        # Validate configuration
        errors = _global_config.validate()
        if errors:
            raise ValueError(f"Invalid ROMA adapter configuration: {errors}")

    return _global_config


def set_config(config: RomaAdapterConfig) -> None:
    """
    Set global ROMA adapter configuration.

    Args:
        config: Configuration to set
    """
    global _global_config

    # Validate before setting
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid ROMA adapter configuration: {errors}")

    _global_config = config


def reset_config() -> None:
    """Reset global configuration to None (will reload from env on next get)."""
    global _global_config
    _global_config = None


# =============================================================================
# CONSTRAINT BUILDER
# =============================================================================

class RomaConstraintBuilder:
    """
    Builder for creating ROMA-specific LMQL constraints.

    Provides a fluent API for building constraint dictionaries.
    """

    def __init__(self):
        self.constraints: Dict[str, Any] = {}

    def with_max_depth(self, depth: int) -> 'RomaConstraintBuilder':
        """Set maximum decomposition depth."""
        self.constraints["max_depth"] = depth
        return self

    def with_max_subtasks(self, count: int) -> 'RomaConstraintBuilder':
        """Set maximum number of subtasks."""
        self.constraints["max_subtasks"] = count
        return self

    def with_subtask_token_limit(self, limit: int) -> 'RomaConstraintBuilder':
        """Set maximum tokens per subtask description."""
        self.constraints["subtask_token_limit"] = limit
        return self

    def require_json(self, required: bool = True) -> 'RomaConstraintBuilder':
        """Require JSON output format."""
        self.constraints["require_json"] = required
        return self

    def with_custom_constraint(self, constraint: Any) -> 'RomaConstraintBuilder':
        """Add a custom constraint object."""
        if "custom_constraints" not in self.constraints:
            self.constraints["custom_constraints"] = []
        self.constraints["custom_constraints"].append(constraint)
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the constraint dictionary.

        Returns:
            Constraint configuration dictionary
        """
        return self.constraints


def create_constraints() -> RomaConstraintBuilder:
    """
    Create a new constraint builder.

    Returns:
        RomaConstraintBuilder instance

    Example:
        constraints = create_constraints() \\
            .with_max_depth(3) \\
            .with_max_subtasks(10) \\
            .require_json() \\
            .build()
    """
    return RomaConstraintBuilder()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RomaAdapterConfig",
    "get_config",
    "set_config",
    "reset_config",
    "RomaConstraintBuilder",
    "create_constraints"
]
