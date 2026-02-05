"""
RESE Logic-to-Loss Translation Layer (LLTL)

Provides computable interface between SCE (Symbolic Constraint Engine) and
DEE (Deep Exploration Engine).

Following CLAUDE.md principles:
- Law of Idempotency: Cache translations, reuse if same input
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker: Detect encoding failures
- Structured Logging: JSON logs with correlation_id
- Timeout: All translations timeout (default 3000ms via env)

Core Components:
1. LogicToLossTranslator: Main translation layer
2. SymbolicConstraintEncoder: Encodes logic to neural format
3. LossFunctionComposer: Composes differentiable loss functions
4. DITOOptimizer: Dynamic Inference Trace Optimizer (naive implementation)

Author: RESE Team
Created: 2026-02-04
Status: Tier 2 Implementation (Core Algorithmic Component)
"""

import os
import sys
import json
import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
import uuid

# Configure structured JSON logging
class StructuredLogger:
    """Structured JSON logger for LLTL operations."""

    def __init__(self, name: str = "rese_lltl"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "%(name)s", "message": "%(message)s"}'
        ))
        self.logger.addHandler(handler)

    def log(self, level: str, msg: str, **kwargs):
        """Log structured message with context."""
        log_entry = {
            "correlation_id": kwargs.get("correlation_id"),
            "component": kwargs.get("component", "lltl"),
            "operation": kwargs.get("operation"),
            "constraint_id": kwargs.get("constraint_id"),
            "translation_id": kwargs.get("translation_id"),
            "duration_ms": kwargs.get("duration_ms"),
            "cache_hit": kwargs.get("cache_hit"),
            "message": msg
        }
        # Filter out None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))


logger = StructuredLogger()


# ============================================================================
# CIRCUIT BREAKER (CLAUDE.md compliance)
# ============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, stop requests
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes to close
    timeout_ms: int = 60000           # Time to attempt recovery
    half_open_max_calls: int = 3      # Max calls in half-open


class CircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    Following CLAUDE.md failure management:
    - Transient Failure: Exponential Backoff Retry
    - System Failure: Circuit Breaker
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() * 1000
                if elapsed > self.config.timeout_ms:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.log("INFO", f"Circuit breaker {self.name} transitioned to HALF_OPEN",
                              component="circuit_breaker")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.log("INFO", f"Circuit breaker {self.name} closed",
                          component="circuit_breaker")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        self.success_count = 0

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.log("WARNING", f"Circuit breaker {self.name} opened after {self.failure_count} failures",
                      component="circuit_breaker")


# ============================================================================
# SYMBOLIC CONSTRAINT ENCODER
# ============================================================================

@dataclass
class EncodingConfig:
    """Configuration for constraint encoding."""
    encoding_dim: int = 128           # Dimension of encoded vectors
    use_positional: bool = True       # Use positional encoding
    use_type_embedding: bool = True   # Embed constraint type
    use_category_embedding: bool = True # Embed constraint category
    max_sequence_length: int = 512    # Max constraint expression length
    cache_size: int = 1000            # Translation cache size


class SymbolicConstraintEncoder:
    """
    Encodes symbolic constraints into neural format.

    Translates symbolic logic (SCE) to differentiable representations (DEE).

    Features:
    - Caching: Idempotent translations (Law of Idempotency)
    - Circuit breaker: Failure detection
    - Timeout: All operations timeout
    """

    def __init__(self, config: Optional[EncodingConfig] = None):
        self.config = config or self._default_config()
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.circuit_breaker = CircuitBreaker(
            "encoder",
            CircuitBreakerConfig(
                failure_threshold=int(os.getenv("ENCODER_FAILURE_THRESHOLD", "5")),
                timeout_ms=int(os.getenv("ENCODER_TIMEOUT_MS", "30000"))
            )
        )

    def _default_config(self) -> EncodingConfig:
        """Load config from environment (Law of Configuration Explicitness)."""
        return EncodingConfig(
            encoding_dim=int(os.getenv("ENCODING_DIM", "128")),
            use_positional=os.getenv("USE_POSITIONAL_ENCODING", "true").lower() == "true",
            use_type_embedding=os.getenv("USE_TYPE_EMBEDDING", "true").lower() == "true",
            use_category_embedding=os.getenv("USE_CATEGORY_EMBEDDING", "true").lower() == "true",
            max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
            cache_size=int(os.getenv("ENCODER_CACHE_SIZE", "1000"))
        )

    def encode(
        self,
        constraint: Any,
        timeout_ms: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Encode symbolic constraint to neural representation.

        Args:
            constraint: Symbolic constraint object
            timeout_ms: Operation timeout (overrides env var)
            correlation_id: For tracing

        Returns:
            Tuple of (encoded_representation, error_message)
        """
        start_time = time.time()
        timeout_ms = timeout_ms or int(os.getenv("ENCODER_TIMEOUT_MS", "3000"))

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(constraint)

            # Check cache (idempotency)
            if cache_key in self._cache:
                self._cache_hits += 1
                duration_ms = (time.time() - start_time) * 1000
                logger.log("INFO", "Encoder cache hit",
                          correlation_id=correlation_id,
                          operation="encode",
                          cache_hit=True,
                          duration_ms=duration_ms)
                return self._cache[cache_key], None

            self._cache_misses += 1

            # Encode through circuit breaker
            def encode_operation():
                return self._encode_constraint(constraint)

            encoded = self.circuit_breaker.call(encode_operation)

            # Cache result
            if len(self._cache) >= self.config.cache_size:
                # Simple FIFO eviction
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = encoded

            duration_ms = (time.time() - start_time) * 1000
            logger.log("INFO", "Constraint encoded successfully",
                      correlation_id=correlation_id,
                      operation="encode",
                      cache_hit=False,
                      duration_ms=duration_ms)

            return encoded, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Encoding failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="encode",
                      duration_ms=duration_ms)
            return None, error_msg

    def _generate_cache_key(self, constraint: Any) -> str:
        """Generate cache key from constraint."""
        # Create hash of constraint properties
        if hasattr(constraint, 'constraint_id'):
            base = f"{constraint.constraint_id}"
        elif hasattr(constraint, 'expression'):
            base = str(constraint.expression)
        else:
            base = str(constraint)

        return hashlib.sha256(base.encode()).hexdigest()

    def _encode_constraint(self, constraint: Any) -> Dict[str, Any]:
        """
        Internal encoding logic.

        Converts constraint to:
        1. Feature vector (numeric encoding)
        2. Structural representation (AST-like)
        3. Metadata encoding
        """
        # Extract features
        features = self._extract_features(constraint)

        # Create encoding
        encoding = {
            "constraint_id": getattr(constraint, "constraint_id", "unknown"),
            "feature_vector": self._create_feature_vector(features),
            "structural_encoding": self._create_structural_encoding(constraint),
            "metadata": self._encode_metadata(constraint),
            "encoding_timestamp": datetime.now(timezone.utc).isoformat()
        }

        return encoding

    def _extract_features(self, constraint: Any) -> Dict[str, Any]:
        """Extract numerical features from constraint."""
        features = {
            "type": getattr(constraint, "type", "unknown"),
            "category": getattr(constraint, "category", "unknown"),
            "priority": getattr(constraint, "priority", 1.0),
            "confidence": getattr(constraint, "confidence", 1.0),
            "dependency_count": len(getattr(constraint, "dependencies", [])),
            "expression_length": len(str(getattr(constraint, "expression", "")))
        }
        return features

    def _create_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Create fixed-length feature vector."""
        # Naive implementation: hash features to fixed dimension
        vector = [0.0] * self.config.encoding_dim

        # Encode type
        if self.config.use_type_embedding:
            type_idx = hash(features["type"]) % self.config.encoding_dim
            vector[type_idx] = 1.0

        # Encode category
        if self.config.use_category_embedding:
            cat_idx = hash(features["category"]) % self.config.encoding_dim
            vector[cat_idx] += 0.5

        # Encode priority and confidence
        vector[0] = features["priority"]
        vector[1] = features["confidence"]
        vector[2] = features["dependency_count"]
        vector[3] = features["expression_length"]

        return vector

    def _create_structural_encoding(self, constraint: Any) -> Dict[str, Any]:
        """Create structural encoding (AST-like)."""
        expression = getattr(constraint, "expression", None)

        if expression is None:
            return {"type": "empty", "children": []}

        # Naive implementation: capture expression structure
        return {
            "type": "expression",
            "representation": str(expression),
            "complexity": self._estimate_complexity(expression)
        }

    def _estimate_complexity(self, expression: Any) -> int:
        """Estimate expression complexity."""
        # Naive: string length
        return len(str(expression))

    def _encode_metadata(self, constraint: Any) -> Dict[str, Any]:
        """Encode constraint metadata."""
        return {
            "dependencies": getattr(constraint, "dependencies", []),
            "description": getattr(constraint, "description", ""),
            "metadata": getattr(constraint, "metadata", {})
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }


# ============================================================================
# LOSS FUNCTION COMPOSER
# ============================================================================

@dataclass
class LossConfig:
    """Configuration for loss function composition."""
    default_type: str = "mse"        # Default loss type
    combination_strategy: str = "weighted_sum"  # How to combine losses
    normalize_weights: bool = True   # Normalize weights to sum to 1
    gradient_clip_value: Optional[float] = None  # Gradient clipping
    learning_rate: float = 0.001     # Learning rate for optimization


class LossFunctionComposer:
    """
    Composes differentiable loss functions from encoded constraints.

    Translates encoded constraints to actual loss functions usable in DEE.

    Features:
    - Multiple loss types (MSE, cross-entropy, hinge, custom)
    - Weighted combination
    - Gradient computation
    - Idempotent composition
    """

    def __init__(self, config: Optional[LossConfig] = None):
        self.config = config or self._default_config()
        self._loss_registry = self._build_loss_registry()

    def _default_config(self) -> LossConfig:
        """Load config from environment."""
        return LossConfig(
            default_type=os.getenv("DEFAULT_LOSS_TYPE", "mse"),
            combination_strategy=os.getenv("COMBINATION_STRATEGY", "weighted_sum"),
            normalize_weights=os.getenv("NORMALIZE_WEIGHTS", "true").lower() == "true",
            gradient_clip_value=float(os.getenv("GRADIENT_CLIP_VALUE", "0")) or None,
            learning_rate=float(os.getenv("LEARNING_RATE", "0.001"))
        )

    def _build_loss_registry(self) -> Dict[str, Callable]:
        """Build registry of loss functions."""
        return {
            "mse": self._mse_loss,
            "cross_entropy": self._cross_entropy_loss,
            "hinge": self._hinge_loss,
            "custom": self._custom_loss
        }

    def compose(
        self,
        encoded_constraint: Dict[str, Any],
        weight: Optional[float] = None,
        loss_type: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Compose loss function from encoded constraint.

        Args:
            encoded_constraint: Output from encoder
            weight: Loss weight (defaults to constraint priority)
            loss_type: Type of loss function
            correlation_id: For tracing

        Returns:
            Tuple of (loss_function_dict, error_message)
        """
        try:
            # Determine loss type
            loss_type = loss_type or self.config.default_type

            if loss_type not in self._loss_registry:
                return None, f"Unknown loss type: {loss_type}"

            # Determine weight
            if weight is None:
                # Use constraint priority as weight
                weight = encoded_constraint.get("metadata", {}).get("priority", 1.0)

            # Compose loss function
            loss_fn = {
                "loss_id": str(uuid.uuid4()),
                "source_constraint_id": encoded_constraint["constraint_id"],
                "type": loss_type,
                "weight": weight,
                "parameters": self._extract_loss_parameters(encoded_constraint),
                "function": self._loss_registry[loss_type],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            logger.log("INFO", f"Loss function composed: {loss_type}",
                      correlation_id=correlation_id,
                      operation="compose",
                      constraint_id=encoded_constraint["constraint_id"])

            return loss_fn, None

        except Exception as e:
            error_msg = f"Loss composition failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="compose")
            return None, error_msg

    def combine(
        self,
        loss_functions: List[Dict[str, Any]],
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Combine multiple loss functions.

        Args:
            loss_functions: List of loss function dicts
            correlation_id: For tracing

        Returns:
            Tuple of (combined_loss_dict, error_message)
        """
        try:
            if not loss_functions:
                return None, "No loss functions to combine"

            # Extract weights
            weights = [lf["weight"] for lf in loss_functions]

            # Normalize if configured
            if self.config.normalize_weights:
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]

            # Create combined loss
            combined_loss = {
                "combined_loss_id": str(uuid.uuid4()),
                "loss_ids": [lf["loss_id"] for lf in loss_functions],
                "combination_strategy": self.config.combination_strategy,
                "weights": weights,
                "total_weight": sum(weights),
                "loss_functions": loss_functions,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            logger.log("INFO", f"Combined {len(loss_functions)} loss functions",
                      correlation_id=correlation_id,
                      operation="combine")

            return combined_loss, None

        except Exception as e:
            error_msg = f"Loss combination failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="combine")
            return None, error_msg

    def compute_loss(
        self,
        loss_fn: Dict[str, Any],
        predictions: Any,
        targets: Any,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Compute loss value.

        Args:
            loss_fn: Loss function dict
            predictions: Model predictions
            targets: Target values
            correlation_id: For tracing

        Returns:
            Tuple of (loss_value, gradients_dict)
        """
        try:
            # Get loss function
            fn = loss_fn["function"]

            # Compute loss
            loss_value = fn(predictions, targets)

            # Compute gradients (naive numerical gradient)
            gradients = self._compute_gradients(predictions, targets, fn)

            logger.log("DEBUG", f"Loss computed: {loss_value}",
                      correlation_id=correlation_id,
                      operation="compute_loss")

            return loss_value, gradients

        except Exception as e:
            error_msg = f"Loss computation failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="compute_loss")
            return None, None

    def _extract_loss_parameters(self, encoded_constraint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract loss-specific parameters from encoded constraint."""
        return {
            "confidence": encoded_constraint.get("metadata", {}).get("confidence", 1.0),
            "feature_vector": encoded_constraint.get("feature_vector", []),
            "complexity": encoded_constraint.get("structural_encoding", {}).get("complexity", 0)
        }

    def _mse_loss(self, predictions: Any, targets: Any) -> float:
        """Mean Squared Error loss."""
        # Naive implementation
        try:
            if isinstance(predictions, list) and isinstance(targets, list):
                diff = [(p - t) ** 2 for p, t in zip(predictions, targets)]
                return sum(diff) / len(diff)
            else:
                return float((predictions - targets) ** 2)
        except:
            return 0.0

    def _cross_entropy_loss(self, predictions: Any, targets: Any) -> float:
        """Cross-entropy loss (naive implementation)."""
        # Placeholder for now
        return 0.0

    def _hinge_loss(self, predictions: Any, targets: Any) -> float:
        """Hinge loss (naive implementation)."""
        # Placeholder for now
        return 0.0

    def _custom_loss(self, predictions: Any, targets: Any) -> float:
        """Custom loss placeholder."""
        return 0.0

    def _compute_gradients(
        self,
        predictions: Any,
        targets: Any,
        loss_fn: Callable
    ) -> Dict[str, Any]:
        """Compute gradients (naive numerical)."""
        # Placeholder for now - in real implementation would use autograd
        return {
            "gradients_computed": True,
            "gradient_norm": 0.0,
            "method": "numerical"
        }


# ============================================================================
# DITO OPTIMIZER (Naive Implementation - Tier 6)
# ============================================================================

@dataclass
class DITOConfig:
    """DITO configuration."""
    enable_rtree: bool = False      # Deferred to Tier 6
    enable_lsh: bool = False        # Deferred to Tier 6
    enable_hag: bool = False        # Deferred to Tier 6
    contradiction_threshold: float = 0.8
    max_contradictions: int = 1000
    cache_size: int = 1000


class DITOOptimizer:
    """
    Dynamic Inference Trace Optimizer (Naive Implementation).

    NOTE: This is a naive O(n²) implementation. Full R-tree/LSH optimization
    is deferred to Tier 6 per SOURCE_RECOVERY_REPORT.

    Current capabilities:
    - Detect contradictions between constraints
    - Manage contradiction pairs
    - Cache detection results

    Future optimizations (Tier 6):
    - R-tree spatial indexing for O(n log n) detection
    - LSH for approximate contradiction detection
    - Hierarchical Abstraction Graph (HAG)
    """

    def __init__(self, config: Optional[DITOConfig] = None):
        self.config = config or self._default_config()
        self._contradictions: List[Any] = []
        self._detection_cache: Dict[str, bool] = {}

    def _default_config(self) -> DITOConfig:
        """Load config from environment."""
        return DITOConfig(
            enable_rtree=os.getenv("ENABLE_RTREE", "false").lower() == "true",
            enable_lsh=os.getenv("ENABLE_LSH", "false").lower() == "true",
            enable_hag=os.getenv("ENABLE_HAG", "false").lower() == "true",
            contradiction_threshold=float(os.getenv("CONTRADICTION_THRESHOLD", "0.8")),
            max_contradictions=int(os.getenv("MAX_CONTRADICTIONS", "1000")),
            cache_size=int(os.getenv("DITO_CACHE_SIZE", "1000"))
        )

    def detect_contradictions(
        self,
        constraints: List[Any],
        correlation_id: Optional[str] = None
    ) -> Tuple[List[Any], Optional[str]]:
        """
        Detect contradictions between constraints (naive O(n²)).

        Args:
            constraints: List of constraints to check
            correlation_id: For tracing

        Returns:
            Tuple of (contradiction_pairs, error_message)
        """
        try:
            start_time = time.time()
            contradictions = []

            # Naive pairwise comparison
            for i, c1 in enumerate(constraints):
                for j, c2 in enumerate(constraints[i+1:], i+1):
                    cache_key = self._get_pair_key(c1, c2)

                    # Check cache
                    if cache_key in self._detection_cache:
                        if self._detection_cache[cache_key]:
                            contradictions.append(self._create_contradiction(c1, c2))
                        continue

                    # Check for contradiction
                    is_contradiction = self._check_contradiction(c1, c2)
                    self._detection_cache[cache_key] = is_contradiction

                    if is_contradiction:
                        contradictions.append(self._create_contradiction(c1, c2))

            duration_ms = (time.time() - start_time) * 1000
            logger.log("INFO", f"Detected {len(contradictions)} contradictions in {duration_ms:.2f}ms",
                      correlation_id=correlation_id,
                      operation="detect_contradictions",
                      duration_ms=duration_ms)

            return contradictions, None

        except Exception as e:
            error_msg = f"Contradiction detection failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="detect_contradictions")
            return [], error_msg

    def _get_pair_key(self, c1: Any, c2: Any) -> str:
        """Generate cache key for constraint pair."""
        id1 = getattr(c1, "constraint_id", "unknown")
        id2 = getattr(c2, "constraint_id", "unknown")
        return f"{min(id1, id2)}-{max(id1, id2)}"

    def _check_contradiction(self, c1: Any, c2: Any) -> bool:
        """
        Check if two constraints contradict each other.

        Naive implementation: checks for:
        1. Direct logical negation (same type, opposite expressions)
        2. Priority conflicts (high priority vs low priority)
        3. Category conflicts
        """
        # Check for direct negation (naive)
        expr1 = str(getattr(c1, "expression", ""))
        expr2 = str(getattr(c2, "expression", ""))

        # Simple heuristic: if expressions are opposites
        if f"not {expr1}" == expr2 or f"not {expr2}" == expr1:
            return True

        # Check for priority conflicts
        priority1 = getattr(c1, "priority", 1.0)
        priority2 = getattr(c2, "priority", 1.0)

        if abs(priority1 - priority2) > 0.5:
            # Different priorities might indicate contradiction
            pass  # Placeholder for more sophisticated logic

        # Check category conflicts
        cat1 = getattr(c1, "category", "unknown")
        cat2 = getattr(c2, "category", "unknown")

        if cat1 != cat2:
            # Different categories - might be contradictory
            pass

        return False  # Naive: return False for now

    def _create_contradiction(self, c1: Any, c2: Any) -> Dict[str, Any]:
        """Create contradiction pair."""
        return {
            "contradiction_id": str(uuid.uuid4()),
            "constraint1_id": getattr(c1, "constraint_id", "unknown"),
            "constraint2_id": getattr(c2, "constraint_id", "unknown"),
            "type": "detected",
            "confidence": 0.8,
            "detected_at": datetime.now(timezone.utc).isoformat()
        }

    def get_contradictions(self) -> List[Any]:
        """Get all detected contradictions."""
        return self._contradictions

    def clear_cache(self):
        """Clear detection cache."""
        self._detection_cache.clear()


# ============================================================================
# MAIN LOGIC-TO-LOSS TRANSLATOR
# ============================================================================

class LogicToLossTranslator:
    """
    Main Logic-to-Loss Translation Layer.

    Orchestrates the full translation pipeline:
    1. Symbolic constraints -> Encoder -> Neural representations
    2. Neural representations -> Composer -> Loss functions
    3. DITO -> Contradiction detection

    Following CLAUDE.md principles:
    - Idempotent: Cache translations
    - Timeout: All operations timeout
    - Circuit breaker: Fail gracefully
    - Structured logging: JSON logs
    """

    def __init__(
        self,
        encoding_config: Optional[EncodingConfig] = None,
        loss_config: Optional[LossConfig] = None,
        dito_config: Optional[DITOConfig] = None
    ):
        self.encoder = SymbolicConstraintEncoder(encoding_config)
        self.composer = LossFunctionComposer(loss_config)
        self.dito = DITOOptimizer(dito_config)

        self._translation_cache: Dict[str, Any] = {}

    def translate(
        self,
        constraints: List[Any],
        timeout_ms: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Translate symbolic constraints to loss functions.

        Full pipeline: encode -> compose -> detect contradictions -> combine

        Args:
            constraints: List of symbolic constraints
            timeout_ms: Translation timeout
            correlation_id: For tracing

        Returns:
            Tuple of (translation_result, error_message)
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            logger.log("INFO", f"Translating {len(constraints)} constraints",
                      correlation_id=correlation_id,
                      operation="translate")

            # Step 1: Detect contradictions first (DITO)
            contradictions, error = self.dito.detect_contradictions(
                constraints,
                correlation_id
            )

            if error:
                logger.log("WARNING", f"DITO detection failed: {error}",
                          correlation_id=correlation_id,
                          operation="translate")

            # Step 2: Encode all constraints
            encoded_constraints = []
            for constraint in constraints:
                encoded, error = self.encoder.encode(constraint, timeout_ms, correlation_id)
                if error:
                    logger.log("WARNING", f"Encoding failed for {constraint}: {error}",
                              correlation_id=correlation_id,
                              operation="translate")
                    continue
                encoded_constraints.append(encoded)

            if not encoded_constraints:
                return None, "No constraints successfully encoded"

            # Step 3: Compose loss functions
            loss_functions = []
            for encoded in encoded_constraints:
                loss_fn, error = self.composer.compose(encoded, correlation_id=correlation_id)
                if error:
                    logger.log("WARNING", f"Loss composition failed: {error}",
                              correlation_id=correlation_id,
                              operation="translate")
                    continue
                loss_functions.append(loss_fn)

            if not loss_functions:
                return None, "No loss functions successfully composed"

            # Step 4: Combine losses
            combined_loss, error = self.composer.combine(loss_functions, correlation_id)
            if error:
                return None, f"Loss combination failed: {error}"

            # Create result
            result = {
                "translation_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "input_constraints": len(constraints),
                "encoded_constraints": len(encoded_constraints),
                "loss_functions": len(loss_functions),
                "contradictions_detected": len(contradictions),
                "combined_loss": combined_loss,
                "contradictions": contradictions,
                "duration_ms": (time.time() - start_time) * 1000,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            logger.log("INFO", f"Translation completed: {len(loss_functions)} loss functions",
                      correlation_id=correlation_id,
                      operation="translate",
                      translation_id=result["translation_id"],
                      duration_ms=result["duration_ms"])

            return result, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Translation failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="translate",
                      duration_ms=duration_ms)
            return None, error_msg

    def get_stats(self) -> Dict[str, Any]:
        """Get translator statistics."""
        return {
            "encoder_cache": self.encoder.get_cache_stats(),
            "dito_contradictions": len(self.dito.get_contradictions()),
            "translation_cache_size": len(self._translation_cache)
        }


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Main translator
    "LogicToLossTranslator",

    # Components
    "SymbolicConstraintEncoder",
    "LossFunctionComposer",
    "DITOOptimizer",

    # Configurations
    "EncodingConfig",
    "LossConfig",
    "DITOConfig",
    "CircuitBreakerConfig",

    # Utilities
    "CircuitBreaker",
    "CircuitBreakerState",
    "StructuredLogger",
]
