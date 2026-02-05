"""
RESE LLTL Confidence Tracker

Tracks and manages confidence thresholds for DEE -> SCE translations.

Following CLAUDE.md principles:
- Law of Idempotency: Same input produces same threshold
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON logs with correlation_id
- Law of UTC: All timestamps in UTC

Author: RESE Team
Created: 2026-02-04
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid


# Configure structured logging
class ConfidenceLogger:
    """Structured logger for confidence tracking."""

    def __init__(self):
        self.logger = logging.getLogger("confidence_tracker")
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "confidence_tracker", "message": "%(message)s"}'
        ))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, level: str, msg: str, **kwargs):
        """Log structured message."""
        log_data = {
            "correlation_id": kwargs.get("correlation_id"),
            "operation": kwargs.get("operation"),
            "confidence": kwargs.get("confidence"),
            "threshold": kwargs.get("threshold"),
            "proposition_id": kwargs.get("proposition_id"),
            "message": msg
        }
        log_data = {k: v for k, v in log_data.items() if v is not None}
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))


logger = ConfidenceLogger()


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"  # >= 0.95
    HIGH = "high"            # >= 0.80
    MODERATE = "moderate"    # >= 0.60
    LOW = "low"              # < 0.60


@dataclass
class ConfidenceThreshold:
    """
    A confidence threshold with metadata.

    Attributes:
        threshold: Minimum confidence required to accept proposition
        level: Confidence level category
        significance_level: Statistical significance (alpha)
        derived_at: UTC ISO-8601 timestamp when threshold was derived
        derivation_method: How threshold was calculated
        correlation_id: For tracing
    """
    threshold: float
    level: ConfidenceLevel
    significance_level: float
    derived_at: str
    derivation_method: str
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "threshold": self.threshold,
            "level": self.level.value,
            "significance_level": self.significance_level,
            "derived_at": self.derived_at,
            "derivation_method": self.derivation_method,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


@dataclass
class ConfidenceHistory:
    """
    History of confidence calculations for auditability.

    Tracks all confidence threshold calculations for complete audit trail.
    """
    history_id: str
    proposition_id: str
    input_confidence: float
    calculated_threshold: ConfidenceThreshold
    timestamp: str  # UTC ISO-8601
    correlation_id: str


class ConfidenceTracker:
    """
    Tracks and manages confidence thresholds for DEE -> SCE translations.

    From RESE Technical Manual §2.2:
    "DEE -> SCE (Auditability): The DEE's statistical results are converted
    into auditable Formal Propositional Commitments by assigning explicit
    Confidence Thresholds that the SCE can integrate into its logic graph
    for contradiction detection."

    Features:
    - Calculate confidence thresholds from statistical confidence
    - Track threshold history for auditability
    - Configurable threshold calculation strategies
    - Idempotent: Same input produces same threshold
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize confidence tracker.

        Args:
            config: Optional configuration dict (overrides env vars)
        """
        self.config = self._load_config(config)
        self._validate_config()

        # Threshold history for auditability
        self.threshold_history: List[ConfidenceHistory] = []

        # Threshold cache for idempotency
        self._threshold_cache: Dict[str, ConfidenceThreshold] = {}

        logger.log("INFO", "Confidence tracker initialized",
                  operation="initialize",
                  significance_level=self.config["significance_level"])

    def _load_config(self, override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        CLAUDE.md: Law of Configuration Explicitness.
        """
        config = {
            "significance_level": float(os.getenv("LLTL_SIGNIFICANCE_LEVEL", "0.05")),
            "default_threshold": float(os.getenv("LLTL_CONFIDENCE_THRESHOLD_DEFAULT", "0.75")),
            "very_high_threshold": float(os.getenv("LLTL_VERY_HIGH_THRESHOLD", "0.90")),
            "high_threshold": float(os.getenv("LLTL_HIGH_THRESHOLD", "0.75")),
            "moderate_threshold": float(os.getenv("LLTL_MODERATE_THRESHOLD", "0.60")),
            "low_threshold": float(os.getenv("LLTL_LOW_THRESHOLD", "0.50")),
            "calculation_strategy": os.getenv("LLTL_THRESHOLD_STRATEGY", "tiered"),
            "enable_history": os.getenv("LLTL_ENABLE_THRESHOLD_HISTORY", "true").lower() == "true",
            "max_history_size": int(os.getenv("LLTL_MAX_THRESHOLD_HISTORY", "10000"))
        }

        # Apply overrides
        if override_config:
            config.update(override_config)

        return config

    def _validate_config(self):
        """
        Validate configuration.

        CLAUDE.md: Crash immediately if config is invalid.
        """
        errors = []

        # Validate significance level
        if not (0 < self.config["significance_level"] < 1):
            errors.append("SIGNIFICANCE_LEVEL must be between 0 and 1")

        # Validate thresholds
        if not (0 <= self.config["very_high_threshold"] <= 1):
            errors.append("VERY_HIGH_THRESHOLD must be between 0 and 1")
        if not (0 <= self.config["high_threshold"] <= 1):
            errors.append("HIGH_THRESHOLD must be between 0 and 1")
        if not (0 <= self.config["moderate_threshold"] <= 1):
            errors.append("MODERATE_THRESHOLD must be between 0 and 1")
        if not (0 <= self.config["low_threshold"] <= 1):
            errors.append("LOW_THRESHOLD must be between 0 and 1")

        # Validate max history size
        if self.config["max_history_size"] <= 0:
            errors.append("MAX_HISTORY_SIZE must be positive")

        if errors:
            error_msg = f"Configuration validation failed: {', '.join(errors)}"
            logger.log("ERROR", error_msg, operation="validate_config")
            raise RuntimeError(error_msg)

    def calculate_threshold(
        self,
        confidence: float,
        derivation_method: str = "tiered",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConfidenceThreshold:
        """
        Calculate confidence threshold from statistical confidence.

        This implements the core DEE -> SCE confidence threshold assignment.

        High confidence = lower threshold (more certain)
        Low confidence = higher threshold (more skeptical)

        Args:
            confidence: Statistical confidence (0-1)
            derivation_method: How threshold was derived (e.g., "mcts_validation")
            correlation_id: For tracing
            metadata: Additional metadata

        Returns:
            ConfidenceThreshold object

        Raises:
            ValueError: If confidence is not in [0, 1]
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        # Validate input
        if not (0 <= confidence <= 1):
            logger.log("ERROR", f"Invalid confidence value: {confidence}",
                      correlation_id=correlation_id,
                      operation="calculate_threshold")
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        # Check cache (Law of Idempotency)
        cache_key = self._generate_cache_key(confidence, derivation_method)
        if cache_key in self._threshold_cache:
            cached = self._threshold_cache[cache_key]
            logger.log("DEBUG", "Using cached threshold",
                      correlation_id=correlation_id,
                      operation="calculate_threshold",
                      confidence=confidence,
                      threshold=cached.threshold)
            return cached

        # Calculate threshold based on strategy
        if self.config["calculation_strategy"] == "tiered":
            threshold, level = self._tiered_calculation(confidence)
        elif self.config["calculation_strategy"] == "linear":
            threshold, level = self._linear_calculation(confidence)
        elif self.config["calculation_strategy"] == "adaptive":
            threshold, level = self._adaptive_calculation(confidence)
        else:
            logger.log("WARNING", f"Unknown strategy: {self.config['calculation_strategy']}, using tiered",
                      correlation_id=correlation_id,
                      operation="calculate_threshold")
            threshold, level = self._tiered_calculation(confidence)

        # Create threshold object
        confidence_threshold = ConfidenceThreshold(
            threshold=threshold,
            level=level,
            significance_level=self.config["significance_level"],
            derived_at=datetime.now(timezone.utc).isoformat(),
            derivation_method=derivation_method,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )

        # Cache result
        self._threshold_cache[cache_key] = confidence_threshold

        logger.log("INFO", f"Calculated confidence threshold: {threshold:.3f} (level: {level.value})",
                  correlation_id=correlation_id,
                  operation="calculate_threshold",
                  confidence=confidence,
                  threshold_val=threshold,
                  conf_level=level.value)

        return confidence_threshold

    def _tiered_calculation(self, confidence: float) -> Tuple[float, ConfidenceLevel]:
        """
        Tiered threshold calculation strategy.

        Maps confidence ranges to fixed threshold values.

        Args:
            confidence: Statistical confidence (0-1)

        Returns:
            Tuple of (threshold, confidence_level)
        """
        if confidence >= 0.95:
            # Very high confidence - high threshold
            return self.config["very_high_threshold"], ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            # High confidence - moderate threshold
            return self.config["high_threshold"], ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            # Moderate confidence - conservative threshold
            return self.config["moderate_threshold"], ConfidenceLevel.MODERATE
        else:
            # Low confidence - very conservative threshold
            return self.config["low_threshold"], ConfidenceLevel.LOW

    def _linear_calculation(self, confidence: float) -> Tuple[float, ConfidenceLevel]:
        """
        Linear threshold calculation strategy.

        Maps confidence linearly to threshold range.

        Args:
            confidence: Statistical confidence (0-1)

        Returns:
            Tuple of (threshold, confidence_level)
        """
        # Linear interpolation from [0, 1] to [low_threshold, very_high_threshold]
        low = self.config["low_threshold"]
        high = self.config["very_high_threshold"]
        threshold = low + (high - low) * confidence

        # Determine level
        if confidence >= 0.95:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            level = ConfidenceLevel.MODERATE
        else:
            level = ConfidenceLevel.LOW

        return threshold, level

    def _adaptive_calculation(self, confidence: float) -> Tuple[float, ConfidenceLevel]:
        """
        Adaptive threshold calculation strategy.

        Uses historical performance to adapt thresholds.

        Args:
            confidence: Statistical confidence (0-1)

        Returns:
            Tuple of (threshold, confidence_level)
        """
        # For now, fall back to tiered
        # In production, this would analyze historical accuracy
        # and adjust thresholds based on past performance
        return self._tiered_calculation(confidence)

    def track_threshold(
        self,
        proposition_id: str,
        input_confidence: float,
        threshold: ConfidenceThreshold,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Track threshold calculation in history.

        Provides complete audit trail for DEE -> SCE translations.

        Args:
            proposition_id: ID of proposition
            input_confidence: Input statistical confidence
            threshold: Calculated threshold
            correlation_id: For tracing

        Returns:
            History ID

        Raises:
            RuntimeError: If history tracking is disabled
        """
        if not self.config["enable_history"]:
            logger.log("WARNING", "Threshold history tracking is disabled",
                      correlation_id=correlation_id,
                      operation="track_threshold")
            raise RuntimeError("Threshold history tracking is disabled")

        correlation_id = correlation_id or str(uuid.uuid4())

        # Create history entry
        history = ConfidenceHistory(
            history_id=str(uuid.uuid4()),
            proposition_id=proposition_id,
            input_confidence=input_confidence,
            calculated_threshold=threshold,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id
        )

        # Add to history
        self.threshold_history.append(history)

        # Prune if exceeds max size
        if len(self.threshold_history) > self.config["max_history_size"]:
            # Remove oldest entries
            excess = len(self.threshold_history) - self.config["max_history_size"]
            self.threshold_history = self.threshold_history[excess:]
            logger.log("INFO", f"Pruned {excess} history entries",
                      correlation_id=correlation_id,
                      operation="track_threshold")

        logger.log("DEBUG", f"Tracked threshold for proposition {proposition_id}",
                  correlation_id=correlation_id,
                  operation="track_threshold",
                  proposition_id=proposition_id,
                  history_size=len(self.threshold_history))

        return history.history_id

    def get_history(
        self,
        proposition_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ConfidenceHistory]:
        """
        Get threshold calculation history.

        Args:
            proposition_id: Filter by proposition ID (None = all)
            limit: Maximum number of entries to return

        Returns:
            List of ConfidenceHistory objects
        """
        history = self.threshold_history

        # Filter by proposition ID
        if proposition_id:
            history = [h for h in history if h.proposition_id == proposition_id]

        # Limit and sort by most recent first
        history = sorted(history, key=lambda h: h.timestamp, reverse=True)[:limit]

        return history

    def get_stats(self) -> Dict[str, Any]:
        """
        Get confidence tracker statistics.

        Returns:
            Dictionary with statistics
        """
        # Calculate distribution
        level_counts = {level.value: 0 for level in ConfidenceLevel}
        for history in self.threshold_history:
            level_counts[history.calculated_threshold.level.value] += 1

        return {
            "config": {
                "significance_level": self.config["significance_level"],
                "calculation_strategy": self.config["calculation_strategy"],
                "enable_history": self.config["enable_history"],
                "max_history_size": self.config["max_history_size"]
            },
            "history": {
                "total_entries": len(self.threshold_history),
                "cache_size": len(self._threshold_cache),
                "level_distribution": level_counts
            },
            "thresholds": {
                "very_high": self.config["very_high_threshold"],
                "high": self.config["high_threshold"],
                "moderate": self.config["moderate_threshold"],
                "low": self.config["low_threshold"]
            }
        }

    def clear_history(self) -> int:
        """
        Clear threshold history.

        Useful for testing and isolation.

        Returns:
            Number of entries cleared
        """
        count = len(self.threshold_history)
        self.threshold_history.clear()
        logger.log("INFO", f"Cleared {count} threshold history entries",
                  operation="clear_history")
        return count

    def _generate_cache_key(self, confidence: float, derivation_method: str) -> str:
        """
        Generate cache key for threshold calculation.

        Args:
            confidence: Statistical confidence
            derivation_method: Derivation method

        Returns:
            Cache key string
        """
        # Round confidence to 3 decimal places for cache key
        # to avoid floating point precision issues
        rounded_conf = round(confidence, 3)
        return f"{rounded_conf:.3f}:{derivation_method}"
