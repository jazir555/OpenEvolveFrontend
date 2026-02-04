"""
RESE LLTL Adapter

Adapter for the Logic-to-Loss Translation Layer.
Provides a clean interface for translating symbolic constraints to differentiable loss functions.

Following CLAUDE.md principles:
- Law of the Air Gap: No imports from core-projects
- Law of Runtime Truth: Verify before using
- Law of Idempotency: Cache translations
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect and handle failures
- Structured Logging: JSON logs with correlation_id
- Timeout: All operations timeout

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import uuid

# Add glue/lib to path for imports
glue_lib_path = Path(__file__).parent.parent.parent.parent / "lib"
sys.path.insert(0, str(glue_lib_path))

try:
    from rese_lltl import (
        LogicToLossTranslator,
        SymbolicConstraintEncoder,
        LossFunctionComposer,
        DITOOptimizer,
        EncodingConfig,
        LossConfig,
        DITOConfig,
        StructuredLogger
    )
    LLTL_AVAILABLE = True
except ImportError as e:
    LLTL_AVAILABLE = False
    LLTL_IMPORT_ERROR = str(e)


# Configure structured logging
class AdapterLogger:
    """Structured logger for LLTL adapter."""

    def __init__(self):
        self.logger = logging.getLogger("lltl_adapter")
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "lltl_adapter", "message": "%(message)s"}'
        ))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, level: str, msg: str, **kwargs):
        """Log structured message."""
        log_data = {
            "correlation_id": kwargs.get("correlation_id"),
            "operation": kwargs.get("operation"),
            "constraint_count": kwargs.get("constraint_count"),
            "duration_ms": kwargs.get("duration_ms"),
            "success": kwargs.get("success"),
            "error": kwargs.get("error"),
            "message": msg
        }
        log_data = {k: v for k, v in log_data.items() if v is not None}
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))


logger = AdapterLogger()


class LLTLAdapter:
    """
    Adapter for Logic-to-Loss Translation Layer.

    Provides a simplified interface for constraint translation while maintaining
    all CLAUDE.md compliance requirements.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLTL adapter.

        Args:
            config: Optional configuration dict (overrides env vars)

        Raises:
            RuntimeError: If LLTL module is not available
        """
        if not LLTL_AVAILABLE:
            raise RuntimeError(f"LLTL module not available: {LLTL_IMPORT_ERROR}")

        # Load configuration from environment (Law of Configuration Explicitness)
        self.config = self._load_config(config)
        self._validate_config()

        # Initialize translator
        self.translator = LogicToLossTranslator(
            encoding_config=EncodingConfig(**self.config["encoding"]),
            loss_config=LossConfig(**self.config["loss"]),
            dito_config=DITOConfig(**self.config["dito"])
        )

        logger.log("INFO", "LLTL adapter initialized",
                  operation="initialize",
                  success=True)

    def _load_config(self, override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        CLAUDE.md: Law of Configuration Explicitness - all config must be explicit.
        """
        config = {
            "encoding": {
                "encoding_dim": int(os.getenv("LLTL_ENCODING_DIM", "128")),
                "use_positional": os.getenv("LLTL_USE_POSITIONAL", "true").lower() == "true",
                "use_type_embedding": os.getenv("LLTL_USE_TYPE_EMBEDDING", "true").lower() == "true",
                "use_category_embedding": os.getenv("LLTL_USE_CATEGORY_EMBEDDING", "true").lower() == "true",
                "max_sequence_length": int(os.getenv("LLTL_MAX_SEQUENCE_LENGTH", "512")),
                "cache_size": int(os.getenv("LLTL_CACHE_SIZE", "1000"))
            },
            "loss": {
                "default_type": os.getenv("LLTL_DEFAULT_LOSS_TYPE", "mse"),
                "combination_strategy": os.getenv("LLTL_COMBINATION_STRATEGY", "weighted_sum"),
                "normalize_weights": os.getenv("LLTL_NORMALIZE_WEIGHTS", "true").lower() == "true",
                "gradient_clip_value": float(os.getenv("LLTL_GRADIENT_CLIP", "0")) or None,
                "learning_rate": float(os.getenv("LLTL_LEARNING_RATE", "0.001"))
            },
            "dito": {
                "enable_rtree": os.getenv("LLTL_ENABLE_RTREE", "false").lower() == "true",
                "enable_lsh": os.getenv("LLTL_ENABLE_LSH", "false").lower() == "true",
                "enable_hag": os.getenv("LLTL_ENABLE_HAG", "false").lower() == "true",
                "contradiction_threshold": float(os.getenv("LLTL_CONTRADICTION_THRESHOLD", "0.8")),
                "max_contradictions": int(os.getenv("LLTL_MAX_CONTRADICTIONS", "1000")),
                "cache_size": int(os.getenv("LLTL_DITO_CACHE_SIZE", "1000"))
            },
            "timeout_ms": int(os.getenv("LLTL_TIMEOUT_MS", "3000"))
        }

        # Apply overrides
        if override_config:
            for section, values in override_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values

        return config

    def _validate_config(self):
        """
        Validate configuration.

        CLAUDE.md: Crash immediately if required configuration is invalid.
        """
        errors = []

        # Validate encoding dimension
        if self.config["encoding"]["encoding_dim"] <= 0:
            errors.append("ENCODING_DIM must be positive")

        # Validate timeout
        if self.config["timeout_ms"] <= 0:
            errors.append("TIMEOUT_MS must be positive")

        # Validate learning rate
        if self.config["loss"]["learning_rate"] <= 0:
            errors.append("LEARNING_RATE must be positive")

        if errors:
            error_msg = f"Configuration validation failed: {', '.join(errors)}"
            logger.log("ERROR", error_msg, operation="validate_config", success=False)
            raise RuntimeError(error_msg)

    def translate_constraints(
        self,
        constraints: List[Any],
        timeout_ms: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Translate symbolic constraints to loss functions.

        This is the main entry point for the LLTL adapter.

        Args:
            constraints: List of symbolic constraint objects
            timeout_ms: Operation timeout (overrides config)
            correlation_id: For tracing

        Returns:
            Tuple of (result_dict, error_message)

        Example:
            >>> adapter = LLTLAdapter()
            >>> constraints = [SymbolicConstraint(...)]
            >>> result, error = adapter.translate_constraints(constraints)
            >>> if error:
            ...     print(f"Error: {error}")
            ... else:
            ...     print(f"Translated {result['loss_functions']} constraints")
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())
        timeout_ms = timeout_ms or self.config["timeout_ms"]

        try:
            logger.log("INFO", f"Translating {len(constraints)} constraints",
                      correlation_id=correlation_id,
                      operation="translate_constraints",
                      constraint_count=len(constraints))

            # Translate using core translator
            result, error = self.translator.translate(
                constraints=constraints,
                timeout_ms=timeout_ms,
                correlation_id=correlation_id
            )

            if error:
                logger.log("ERROR", f"Translation failed: {error}",
                          correlation_id=correlation_id,
                          operation="translate_constraints",
                          success=False,
                          error=error,
                          duration_ms=(time.time() - start_time) * 1000)
                return None, error

            duration_ms = (time.time() - start_time) * 1000
            logger.log("INFO", f"Translation completed successfully",
                      correlation_id=correlation_id,
                      operation="translate_constraints",
                      success=True,
                      constraint_count=len(constraints),
                      duration_ms=duration_ms)

            return result, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Translation error: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="translate_constraints",
                      success=False,
                      error=error_msg,
                      duration_ms=duration_ms)
            return None, error_msg

    def encode_single(
        self,
        constraint: Any,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Encode a single constraint.

        Useful for testing and debugging.

        Args:
            constraint: Single constraint object
            correlation_id: For tracing

        Returns:
            Tuple of (encoded_dict, error_message)
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            encoded, error = self.translator.encoder.encode(
                constraint=constraint,
                correlation_id=correlation_id
            )

            if error:
                logger.log("ERROR", f"Encoding failed: {error}",
                          correlation_id=correlation_id,
                          operation="encode_single",
                          success=False)
                return None, error

            logger.log("INFO", "Constraint encoded",
                      correlation_id=correlation_id,
                      operation="encode_single",
                      success=True)

            return encoded, None

        except Exception as e:
            error_msg = f"Encoding error: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="encode_single",
                      success=False)
            return None, error_msg

    def detect_contradictions(
        self,
        constraints: List[Any],
        correlation_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Detect contradictions between constraints.

        Uses DITO (naive O(n²) implementation).

        Args:
            constraints: List of constraints to check
            correlation_id: For tracing

        Returns:
            Tuple of (contradictions_list, error_message)
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            contradictions, error = self.translator.dito.detect_contradictions(
                constraints=constraints,
                correlation_id=correlation_id
            )

            if error:
                logger.log("WARNING", f"Contradiction detection had errors: {error}",
                          correlation_id=correlation_id,
                          operation="detect_contradictions")
            else:
                logger.log("INFO", f"Detected {len(contradictions)} contradictions",
                          correlation_id=correlation_id,
                          operation="detect_contradictions",
                          success=True)

            return contradictions, error

        except Exception as e:
            error_msg = f"Contradiction detection error: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="detect_contradictions",
                      success=False)
            return [], error_msg

    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter and translator statistics.

        Useful for monitoring and debugging.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "adapter_config": self.config,
            "translator_stats": self.translator.get_stats(),
            "available": LLTL_AVAILABLE
        }

        return stats

    def health_check(self) -> Tuple[bool, str]:
        """
        Perform health check.

        CLAUDE.md: Verify before using (Law of Runtime Truth).

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check if translator is available
            if not hasattr(self, 'translator'):
                return False, "Translator not initialized"

            # Check encoder
            if not hasattr(self.translator, 'encoder'):
                return False, "Encoder not available"

            # Check composer
            if not hasattr(self.translator, 'composer'):
                return False, "Composer not available"

            # Check DITO
            if not hasattr(self.translator, 'dito'):
                return False, "DITO not available"

            return True, "All components healthy"

        except Exception as e:
            return False, f"Health check failed: {str(e)}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_adapter(config: Optional[Dict[str, Any]] = None) -> LLTLAdapter:
    """
    Factory function to create LLTL adapter.

    Args:
        config: Optional configuration

    Returns:
        LLTLAdapter instance

    Raises:
        RuntimeError: If LLTL module is not available
    """
    return LLTLAdapter(config)


def is_available() -> bool:
    """Check if LLTL module is available."""
    return LLTL_AVAILABLE


def get_import_error() -> Optional[str]:
    """Get import error if LLTL is not available."""
    if LLTL_AVAILABLE:
        return None
    return LLTL_IMPORT_ERROR


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    "LLTLAdapter",
    "create_adapter",
    "is_available",
    "get_import_error",
]
