"""
RESE Pipeline Configuration

Following CLAUDE.md - Law of Configuration Explicitness:
All configuration values must come from environment variables.
Crashes immediately if required configuration is missing.
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """
    Configuration for RESE pipeline.

    All values are loaded from environment variables.
    Crashes immediately if required vars are missing (Law of Configuration Explicitness).
    """

    # Phase timeouts (milliseconds)
    phase_i_timeout_ms: int
    phase_ii_timeout_ms: int
    phase_iii_timeout_ms: int
    phase_iv_timeout_ms: int
    pipeline_timeout_ms: int

    # Retry configuration
    max_retries: int
    retry_initial_delay_ms: int
    retry_max_delay_ms: int
    retry_backoff_multiplier: float

    # Circuit breaker configuration
    circuit_breaker_threshold: int
    circuit_breaker_timeout_ms: int
    circuit_breaker_half_open_attempts: int

    # Dead Letter Queue configuration
    dlq_max_size: int
    dlq_persist_path: Optional[str]

    # Event bus configuration
    event_bus_max_events: int
    event_bus_persist_events: bool
    event_bus_persist_path: Optional[str]

    # Phase-specific configuration
    enable_phase_i: bool
    enable_phase_ii: bool
    enable_phase_iii: bool
    enable_phase_iv: bool

    # DEE configuration
    dee_exploration_depth: int
    dee_mcts_iterations: int
    dee_convergence_threshold: float

    # LLTL configuration
    lltl_encoding_dim: int
    lltl_timeout_ms: int

    # SCE configuration
    sce_contradiction_detection: bool
    sce_formal_verification: bool

    # Logging configuration
    log_level: str
    log_format: str  # "json" or "text"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """
        Load configuration from environment variables.

        Crashes immediately if required vars are missing (Law of Configuration Explicitness).
        """
        required_vars = {
            "PIPELINE_TIMEOUT_MS": "pipeline_timeout_ms",
            "PHASE_I_TIMEOUT_MS": "phase_i_timeout_ms",
            "PHASE_II_TIMEOUT_MS": "phase_ii_timeout_ms",
            "PHASE_III_TIMEOUT_MS": "phase_iii_timeout_ms",
            "PHASE_IV_TIMEOUT_MS": "phase_iv_timeout_ms",
            "MAX_RETRIES": "max_retries",
            "RETRY_INITIAL_DELAY_MS": "retry_initial_delay_ms",
            "RETRY_MAX_DELAY_MS": "retry_max_delay_ms",
        }

        config = {}
        errors = []

        # Load required variables
        for env_var, field_name in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                errors.append(f"Missing required environment variable: {env_var}")
            else:
                try:
                    config[field_name] = int(value)
                except ValueError:
                    errors.append(f"Invalid value for {env_var}: {value} (expected integer)")

        # If errors, crash immediately
        if errors:
            print("FATAL: Configuration validation failed")
            print("\nErrors:")
            for error in errors:
                print(f"  - {error}")
            print("\nRequired environment variables:")
            for env_var in required_vars.keys():
                print(f"  - {env_var}")
            sys.exit(1)

        # Load optional variables with defaults
        config.update({
            "retry_backoff_multiplier": float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0")),
            "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            "circuit_breaker_timeout_ms": int(os.getenv("CIRCUIT_BREAKER_TIMEOUT_MS", "60000")),
            "circuit_breaker_half_open_attempts": int(os.getenv("CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS", "3")),
            "dlq_max_size": int(os.getenv("DLQ_MAX_SIZE", "1000")),
            "dlq_persist_path": os.getenv("DLQ_PERSIST_PATH"),
            "event_bus_max_events": int(os.getenv("EVENT_BUS_MAX_EVENTS", "10000")),
            "event_bus_persist_events": os.getenv("EVENT_BUS_PERSIST_EVENTS", "true").lower() == "true",
            "event_bus_persist_path": os.getenv("EVENT_BUS_PERSIST_PATH"),
            "enable_phase_i": os.getenv("ENABLE_PHASE_I", "true").lower() == "true",
            "enable_phase_ii": os.getenv("ENABLE_PHASE_II", "true").lower() == "true",
            "enable_phase_iii": os.getenv("ENABLE_PHASE_III", "true").lower() == "true",
            "enable_phase_iv": os.getenv("ENABLE_PHASE_IV", "true").lower() == "true",
            "dee_exploration_depth": int(os.getenv("DEE_EXPLORATION_DEPTH", "10")),
            "dee_mcts_iterations": int(os.getenv("DEE_MCTS_ITERATIONS", "1000")),
            "dee_convergence_threshold": float(os.getenv("DEE_CONVERGENCE_THRESHOLD", "0.001")),
            "lltl_encoding_dim": int(os.getenv("LLTL_ENCODING_DIM", "128")),
            "lltl_timeout_ms": int(os.getenv("LLTL_TIMEOUT_MS", "3000")),
            "sce_contradiction_detection": os.getenv("SCE_CONTRADICTION_DETECTION", "true").lower() == "true",
            "sce_formal_verification": os.getenv("SCE_FORMAL_VERIFICATION", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "json"),
        })

        return cls(**config)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "phase_timeouts": {
                "phase_i_ms": self.phase_i_timeout_ms,
                "phase_ii_ms": self.phase_ii_timeout_ms,
                "phase_iii_ms": self.phase_iii_timeout_ms,
                "phase_iv_ms": self.phase_iv_timeout_ms,
                "pipeline_ms": self.pipeline_timeout_ms,
            },
            "retry": {
                "max_retries": self.max_retries,
                "initial_delay_ms": self.retry_initial_delay_ms,
                "max_delay_ms": self.retry_max_delay_ms,
                "backoff_multiplier": self.retry_backoff_multiplier,
            },
            "circuit_breaker": {
                "threshold": self.circuit_breaker_threshold,
                "timeout_ms": self.circuit_breaker_timeout_ms,
                "half_open_attempts": self.circuit_breaker_half_open_attempts,
            },
            "dlq": {
                "max_size": self.dlq_max_size,
                "persist_path": self.dlq_persist_path,
            },
            "event_bus": {
                "max_events": self.event_bus_max_events,
                "persist_events": self.event_bus_persist_events,
                "persist_path": self.event_bus_persist_path,
            },
            "phases_enabled": {
                "phase_i": self.enable_phase_i,
                "phase_ii": self.enable_phase_ii,
                "phase_iii": self.enable_phase_iii,
                "phase_iv": self.enable_phase_iv,
            },
            "dee": {
                "exploration_depth": self.dee_exploration_depth,
                "mcts_iterations": self.dee_mcts_iterations,
                "convergence_threshold": self.dee_convergence_threshold,
            },
            "lltl": {
                "encoding_dim": self.lltl_encoding_dim,
                "timeout_ms": self.lltl_timeout_ms,
            },
            "sce": {
                "contradiction_detection": self.sce_contradiction_detection,
                "formal_verification": self.sce_formal_verification,
            },
            "logging": {
                "level": self.log_level,
                "format": self.log_format,
            },
        }


def validate_config(config: PipelineConfig) -> bool:
    """
    Validate configuration values.

    Args:
        config: Pipeline configuration

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    errors = []

    # Validate timeouts are positive
    if config.phase_i_timeout_ms <= 0:
        errors.append("PHASE_I_TIMEOUT_MS must be positive")
    if config.phase_ii_timeout_ms <= 0:
        errors.append("PHASE_II_TIMEOUT_MS must be positive")
    if config.phase_iii_timeout_ms <= 0:
        errors.append("PHASE_III_TIMEOUT_MS must be positive")
    if config.phase_iv_timeout_ms <= 0:
        errors.append("PHASE_IV_TIMEOUT_MS must be positive")
    if config.pipeline_timeout_ms <= 0:
        errors.append("PIPELINE_TIMEOUT_MS must be positive")

    # Validate retry configuration
    if config.max_retries < 0:
        errors.append("MAX_RETRIES must be non-negative")
    if config.retry_initial_delay_ms <= 0:
        errors.append("RETRY_INITIAL_DELAY_MS must be positive")
    if config.retry_max_delay_ms < config.retry_initial_delay_ms:
        errors.append("RETRY_MAX_DELAY_MS must be >= RETRY_INITIAL_DELAY_MS")
    if config.retry_backoff_multiplier <= 1.0:
        errors.append("RETRY_BACKOFF_MULTIPLIER must be > 1.0")

    # Validate circuit breaker
    if config.circuit_breaker_threshold <= 0:
        errors.append("CIRCUIT_BREAKER_THRESHOLD must be positive")

    # Validate DEE configuration
    if config.dee_exploration_depth <= 0:
        errors.append("DEE_EXPLORATION_DEPTH must be positive")
    if config.dee_mcts_iterations <= 0:
        errors.append("DEE_MCTS_ITERATIONS must be positive")
    if not (0.0 < config.dee_convergence_threshold < 1.0):
        errors.append("DEE_CONVERGENCE_THRESHOLD must be between 0 and 1")

    # Validate LLTL configuration
    if config.lltl_encoding_dim <= 0:
        errors.append("LLTL_ENCODING_DIM must be positive")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return True
