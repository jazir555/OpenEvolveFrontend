"""
RESE Framework - Configuration Loader

This module loads and validates configuration following the "Law of Configuration
Explicitness". The application will CRASH at startup if configuration is invalid.
"""

import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from glue.adapters.rese_integration.config_validator import (
    ConfigValidator,
    ValidationError,
)


class ConfigurationError(Exception):
    """Fatal configuration error - application cannot start."""
    pass


class RESEConfig:
    """
    Loaded and validated configuration.

    This class provides type-safe access to all configuration variables.
    All values are guaranteed to be valid after initialization.
    """

    def __init__(self, env_dict: Optional[Dict[str, str]] = None):
        """
        Load and validate configuration.

        Args:
            env_dict: Environment variables dict (defaults to os.environ)

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self._raw = env_dict or dict(os.environ)

        # Validate configuration
        validator = ConfigValidator(self._raw)
        try:
            validator.validate_all()
        except ValidationError as e:
            raise ConfigurationError(str(e))

        # Cache converted values
        self._cache: Dict[str, Any] = {}

    def _get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get string value."""
        if key in self._cache:
            return self._cache[key]

        value = self._raw.get(key)
        if value is None:
            if default is not None:
                return default
            raise ConfigurationError(f"Missing required variable: {key}")

        self._cache[key] = value
        return value

    def _get_int(self, key: str, default: Optional[int] = None) -> int:
        """Get integer value."""
        if key in self._cache:
            return self._cache[key]

        value = self._raw.get(key)
        if value is None:
            if default is not None:
                return default
            raise ConfigurationError(f"Missing required variable: {key}")

        try:
            int_value = int(value)
            self._cache[key] = int_value
            return int_value
        except ValueError:
            raise ConfigurationError(f"Invalid integer value for {key}: {value}")

    def _get_float(self, key: str, default: Optional[float] = None) -> float:
        """Get float value."""
        if key in self._cache:
            return self._cache[key]

        value = self._raw.get(key)
        if value is None:
            if default is not None:
                return default
            raise ConfigurationError(f"Missing required variable: {key}")

        try:
            float_value = float(value)
            self._cache[key] = float_value
            return float_value
        except ValueError:
            raise ConfigurationError(f"Invalid float value for {key}: {value}")

    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value."""
        if key in self._cache:
            return self._cache[key]

        value = self._raw.get(key, "false").lower()
        bool_value = value in ["true", "1", "yes"]
        self._cache[key] = bool_value
        return bool_value

    # =========================================================================
    # General Configuration
    # =========================================================================

    @property
    def env(self) -> str:
        """Environment mode (development/staging/production)."""
        return self._get_str("RESE_ENV")

    @property
    def log_level(self) -> str:
        """Log level (DEBUG/INFO/WARN/ERROR/CRITICAL)."""
        return self._get_str("RESE_LOG_LEVEL")

    @property
    def correlation_id(self) -> str:
        """Correlation ID for request tracing."""
        return self._get_str("RESE_CORRELATION_ID")

    # =========================================================================
    # Phase I Configuration
    # =========================================================================

    @property
    def phase1_timeout_ms(self) -> int:
        """Phase I timeout in milliseconds."""
        return self._get_int("PHASE1_TIMEOUT_MS")

    @property
    def phase1_max_assumptions(self) -> int:
        """Maximum assumptions to extract."""
        return self._get_int("PHASE1_MAX_ASSUMPTIONS")

    @property
    def phase1_min_assumption_confidence(self) -> float:
        """Minimum confidence for assumptions (0.0 to 1.0)."""
        return self._get_float("PHASE1_MIN_ASSUMPTION_CONFIDENCE")

    @property
    def phase1_circuit_breaker_threshold(self) -> int:
        """Circuit breaker threshold."""
        return self._get_int("PHASE1_CIRCUIT_BREAKER_THRESHOLD")

    @property
    def phase1_enable_tacit_mining(self) -> bool:
        """Enable tacit assumption mining."""
        return self._get_bool("PHASE1_ENABLE_TACIT_MINING")

    @property
    def phase1_enable_red_team(self) -> bool:
        """Enable red team mode."""
        return self._get_bool("PHASE1_ENABLE_RED_TEAM")

    @property
    def phase1_enable_lean4_integration(self) -> bool:
        """Enable Lean4 integration."""
        return self._get_bool("PHASE1_ENABLE_LEAN4_INTEGRATION")

    @property
    def lean4_exec_path(self) -> Optional[str]:
        """Path to Lean4 executable."""
        return self._get_str("LEAN4_EXEC_PATH", default=None)

    # =========================================================================
    # Phase II Configuration
    # =========================================================================

    @property
    def phase2_timeout_ms(self) -> int:
        """Phase II timeout in milliseconds."""
        return self._get_int("PHASE2_TIMEOUT_MS")

    @property
    def phase2_imech_threshold(self) -> float:
        """Isomorphism threshold (0.0 to 1.0)."""
        return self._get_float("PHASE2_IMECH_THRESHOLD")

    @property
    def phase2_max_target_domains(self) -> int:
        """Maximum target domains."""
        return self._get_int("PHASE2_MAX_TARGET_DOMAINS")

    @property
    def phase2_pattern_threshold(self) -> float:
        """Pattern matching threshold (0.0 to 1.0)."""
        return self._get_float("PHASE2_PATTERN_THRESHOLD")

    @property
    def phase2_max_mappings(self) -> int:
        """Maximum isomorphic mappings."""
        return self._get_int("PHASE2_MAX_MAPPINGS")

    @property
    def phase2_enable_constraint_inversion(self) -> bool:
        """Enable constraint inversion."""
        return self._get_bool("PHASE2_ENABLE_CONSTRAINT_INVERSION")

    @property
    def phase2_search_depth(self) -> int:
        """Search depth."""
        return self._get_int("PHASE2_SEARCH_DEPTH")

    # =========================================================================
    # Phase III Configuration
    # =========================================================================

    @property
    def phase3_timeout_ms(self) -> int:
        """Phase III timeout in milliseconds."""
        return self._get_int("PHASE3_TIMEOUT_MS")

    @property
    def phase3_iterations(self) -> int:
        """MCTS iterations."""
        return self._get_int("PHASE3_ITERATIONS")

    @property
    def phase3_ucb1_c(self) -> float:
        """UCB1 exploration constant."""
        return self._get_float("PHASE3_UCB1_C")

    @property
    def phase3_convergence_threshold(self) -> float:
        """Convergence threshold (0.0 to 1.0)."""
        return self._get_float("PHASE3_CONVERGENCE_THRESHOLD")

    @property
    def phase3_aci_window(self) -> int:
        """ACI window size."""
        return self._get_int("PHASE3_ACI_WINDOW")

    @property
    def phase3_sig_threshold(self) -> float:
        """Significance threshold (0.0 to 1.0)."""
        return self._get_float("PHASE3_SIG_THRESHOLD")

    @property
    def phase3_parallel_workers(self) -> int:
        """Parallel workers."""
        return self._get_int("PHASE3_PARALLEL_WORKERS")

    # =========================================================================
    # Phase IV Configuration
    # =========================================================================

    @property
    def phase4_timeout_ms(self) -> int:
        """Phase IV timeout in milliseconds."""
        return self._get_int("PHASE4_TIMEOUT_MS")

    @property
    def phase4_beam_width(self) -> int:
        """Beam width."""
        return self._get_int("PHASE4_BEAM_WIDTH")

    @property
    def phase4_validation_level(self) -> int:
        """Validation level (0-3)."""
        return self._get_int("PHASE4_VALIDATION_LEVEL")

    @property
    def phase4_integration_strategy(self) -> str:
        """Integration strategy (conservative/balanced/aggressive)."""
        return self._get_str("PHASE4_INTEGRATION_STRATEGY")

    @property
    def phase4_min_confidence_threshold(self) -> float:
        """Minimum confidence threshold (0.0 to 1.0)."""
        return self._get_float("PHASE4_MIN_CONFIDENCE_THRESHOLD")

    # =========================================================================
    # LLTL Configuration
    # =========================================================================

    @property
    def lltl_encoding_dim(self) -> int:
        """LLTL encoding dimension."""
        return self._get_int("LLTL_ENCODING_DIM")

    @property
    def lltl_default_loss_type(self) -> str:
        """Default loss type."""
        return self._get_str("LLTL_DEFAULT_LOSS_TYPE")

    @property
    def lltl_contradiction_threshold(self) -> float:
        """Contradiction threshold (0.0 to 1.0)."""
        return self._get_float("LLTL_CONTRADICTION_THRESHOLD")

    @property
    def lltl_timeout_ms(self) -> int:
        """LLTL timeout in milliseconds."""
        return self._get_int("LLTL_TIMEOUT_MS")

    # =========================================================================
    # External Services
    # =========================================================================

    @property
    def openai_api_key(self) -> str:
        """OpenAI API key."""
        return self._get_str("OPENAI_API_KEY")

    @property
    def openai_model(self) -> str:
        """OpenAI model."""
        return self._get_str("OPENAI_MODEL")

    @property
    def redis_url(self) -> str:
        """Redis URL."""
        return self._get_str("REDIS_URL")

    @property
    def redis_key_ttl(self) -> int:
        """Redis key TTL in seconds."""
        return self._get_int("REDIS_KEY_TTL")

    # =========================================================================
    # Telemetry
    # =========================================================================

    @property
    def enable_metrics(self) -> bool:
        """Enable metrics."""
        return self._get_bool("ENABLE_METRICS")

    @property
    def metrics_port(self) -> Optional[int]:
        """Metrics port."""
        return self._get_int("METRICS_PORT", default=None)

    @property
    def enable_tracing(self) -> bool:
        """Enable tracing."""
        return self._get_bool("ENABLE_TRACING")

    @property
    def jaeger_endpoint(self) -> Optional[str]:
        """Jaeger endpoint."""
        return self._get_str("JAEGER_ENDPOINT", default=None)

    # =========================================================================
    # Failure Handling
    # =========================================================================

    @property
    def enable_circuit_breakers(self) -> bool:
        """Enable circuit breakers."""
        return self._get_bool("ENABLE_CIRCUIT_BREAKERS")

    @property
    def circuit_breaker_reset_timeout_ms(self) -> int:
        """Circuit breaker reset timeout."""
        return self._get_int("CIRCUIT_BREAKER_RESET_TIMEOUT_MS")

    @property
    def enable_retry(self) -> bool:
        """Enable retry."""
        return self._get_bool("ENABLE_RETRY")

    @property
    def max_retry_attempts(self) -> int:
        """Maximum retry attempts."""
        return self._get_int("MAX_RETRY_ATTEMPTS")

    @property
    def retry_base_delay_ms(self) -> int:
        """Retry base delay."""
        return self._get_int("RETRY_BASE_DELAY_MS")

    @property
    def enable_dlq(self) -> bool:
        """Enable dead letter queue."""
        return self._get_bool("ENABLE_DLQ")

    @property
    def dlq_name(self) -> str:
        """Dead letter queue name."""
        return self._get_str("DLQ_NAME", default="rese-failures")

    # =========================================================================
    # Advanced
    # =========================================================================

    @property
    def deterministic_mode(self) -> bool:
        """Deterministic mode."""
        return self._get_bool("DETERMINISTIC_MODE", default=False)

    @property
    def random_seed(self) -> Optional[int]:
        """Random seed."""
        return self._get_int("RANDOM_SEED", default=None)

    @property
    def enable_profiling(self) -> bool:
        """Enable profiling."""
        return self._get_bool("ENABLE_PROFILING", default=False)

    @property
    def max_memory_mb(self) -> int:
        """Maximum memory usage."""
        return self._get_int("MAX_MEMORY_MB")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary of all configuration values
        """
        return {
            "general": {
                "env": self.env,
                "log_level": self.log_level,
                "correlation_id": self.correlation_id,
            },
            "phase1": {
                "timeout_ms": self.phase1_timeout_ms,
                "max_assumptions": self.phase1_max_assumptions,
                "min_confidence": self.phase1_min_assumption_confidence,
                "circuit_breaker_threshold": self.phase1_circuit_breaker_threshold,
                "enable_tacit_mining": self.phase1_enable_tacit_mining,
                "enable_red_team": self.phase1_enable_red_team,
                "enable_lean4": self.phase1_enable_lean4_integration,
            },
            "phase2": {
                "timeout_ms": self.phase2_timeout_ms,
                "imech_threshold": self.phase2_imech_threshold,
                "max_domains": self.phase2_max_target_domains,
                "pattern_threshold": self.phase2_pattern_threshold,
                "max_mappings": self.phase2_max_mappings,
                "constraint_inversion": self.phase2_enable_constraint_inversion,
                "search_depth": self.phase2_search_depth,
            },
            "phase3": {
                "timeout_ms": self.phase3_timeout_ms,
                "iterations": self.phase3_iterations,
                "ucb1_c": self.phase3_ucb1_c,
                "convergence_threshold": self.phase3_convergence_threshold,
                "aci_window": self.phase3_aci_window,
                "sig_threshold": self.phase3_sig_threshold,
                "parallel_workers": self.phase3_parallel_workers,
            },
            "phase4": {
                "timeout_ms": self.phase4_timeout_ms,
                "beam_width": self.phase4_beam_width,
                "validation_level": self.phase4_validation_level,
                "integration_strategy": self.phase4_integration_strategy,
                "min_confidence": self.phase4_min_confidence_threshold,
            },
            "lltl": {
                "encoding_dim": self.lltl_encoding_dim,
                "loss_type": self.lltl_default_loss_type,
                "contradiction_threshold": self.lltl_contradiction_threshold,
                "timeout_ms": self.lltl_timeout_ms,
            },
        }

    def __repr__(self) -> str:
        """String representation (hides sensitive values)."""
        return f"RESEConfig(env={self.env}, phases=[I, II, III, IV])"


# Singleton instance
_config: Optional[RESEConfig] = None


def load_config(env_file: Optional[str] = None) -> RESEConfig:
    """
    Load and validate configuration.

    This function should be called once at application startup.

    Args:
        env_file: Optional path to .env file

    Returns:
        Validated configuration object

    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config

    if _config is not None:
        return _config

    # Load .env file if specified
    env_dict = dict(os.environ)
    if env_file:
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_dict[key.strip()] = value.strip()
        except FileNotFoundError:
            raise ConfigurationError(f".env file not found: {env_file}")

    # Load and validate configuration
    _config = RESEConfig(env_dict)
    return _config


def get_config() -> RESEConfig:
    """
    Get the loaded configuration.

    Returns:
        Configuration object

    Raises:
        ConfigurationError: If configuration has not been loaded
    """
    if _config is None:
        raise ConfigurationError(
            "Configuration not loaded. Call load_config() first."
        )
    return _config
