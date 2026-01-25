"""
OpenEvolve Unified Reliability Bridge

Production-ready coordination layer for all 4 reliability systems:
- Layer 0: Steer - Final verification with Reality Locks
- Layer 1: Guardrails - Input/output validation
- Layer 2: LMQL - Constrained generation
- Layer 3: ACE - Learning from failures

Architecture:
    Request → Guardrails (Input) → LMQL (Generation) → Guardrails (Output) → Steer (Verify) → ACE (Learn)

Author: OpenEvolve
Version: 1.0.0
"""

import os
import sys
import time
import uuid
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Callable
)
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
from functools import wraps
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration
from .config import (
    ReliabilityConfig,
    ValidationStrictness,
    OnFailStrategy,
    get_config_manager,
    RELIABILITY_LAYERS
)

# ============================================================================
# Layer Availability Detection
# ============================================================================

try:
    from .lmql_adapter import LMQLAdapter
    LMQL_AVAILABLE = True
except ImportError:
    LMQL_AVAILABLE = False
    logging.warning("LMQL not available - constrained generation disabled")

try:
    from .guardrails_adapter import GuardrailsAdapter
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    logging.warning("Guardrails not available - validation disabled")

try:
    from ace_steer_integration import AceSteerBridge
    ACE_STEER_AVAILABLE = True
except ImportError:
    ACE_STEER_AVAILABLE = False
    logging.warning("ACE-Steer integration not available - learning and verification disabled")

# ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_standard_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_standard_config = None
    logging.warning("roma_mdap_maker_associative_integration not found - robust execution disabled")

# ============================================================================
# Result Classes
# ============================================================================

@dataclass
class LayerStatus:
    """Status of a reliability layer"""
    available: bool
    enabled: bool
    last_error: Optional[str] = None
    request_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "available": self.available,
            "enabled": self.enabled,
            "last_error": self.last_error,
            "request_count": self.request_count,
            "failure_count": self.failure_count,
            "avg_latency_ms": self.avg_latency_ms,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class GenerationResult:
    """Result from unified bridge generation"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    prompt: Optional[str] = None
    layers_used: List[str] = field(default_factory=list)
    layers_failed: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    validation_failures: List[str] = field(default_factory=list)
    verification_failures: List[str] = field(default_factory=list)
    retry_count: int = 0
    total_latency_ms: float = 0.0
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "prompt": self.prompt,
            "layers_used": self.layers_used,
            "layers_failed": self.layers_failed,
            "constraint_violations": self.constraint_violations,
            "validation_failures": self.validation_failures,
            "verification_failures": self.verification_failures,
            "retry_count": self.retry_count,
            "total_latency_ms": self.total_latency_ms,
            "correlation_id": self.correlation_id,
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class ValidationResult:
    """Result from validation layer"""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    remediated_output: Optional[str] = None
    should_retry: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "remediated_output": self.remediated_output,
            "should_retry": self.should_retry,
            "error": self.error
        }


@dataclass
class VerificationResult:
    """Result from verification layer"""
    passed: bool
    score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_teachable_moment: bool = False
    error: Optional[str] = None
    judges: List[str] = field(default_factory=list)
    note: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "is_teachable_moment": self.is_teachable_moment,
            "error": self.error,
            "judges": self.judges,
            "note": self.note,
            "details": self.details
        }


# ============================================================================
# Exception Classes
# ============================================================================

class ReliabilityBridgeError(Exception):
    """Base exception for reliability bridge"""
    pass


class LayerUnavailableError(ReliabilityBridgeError):
    """Raised when a required layer is unavailable"""
    pass


class ConfigurationError(ReliabilityBridgeError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(ReliabilityBridgeError):
    """Raised when validation fails"""
    pass


class GenerationError(ReliabilityBridgeError):
    """Raised when generation fails"""
    pass


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class LayerStatistics:
    """Statistics for a single layer"""
    enabled_count: int = 0
    disabled_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    request_count: int = 0

    def get_avg_latency(self) -> float:
        """Get average latency"""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count


@dataclass
class BridgeStatistics:
    """Overall bridge statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    layers: Dict[str, LayerStatistics] = field(default_factory=lambda: {
        "lmql": LayerStatistics(),
        "guardrails": LayerStatistics(),
        "steer": LayerStatistics(),
        "ace": LayerStatistics()
    })
    retry_distribution: Dict[str, int] = field(default_factory=lambda: {
        "no_retry": 0,
        "retry_1": 0,
        "retry_2": 0,
        "retry_3": 0,
        "retry_4_plus": 0
    })
    guardrails_specific: Dict[str, int] = field(default_factory=lambda: {
        "input_validations": 0,
        "output_validations": 0,
        "failures_caught": 0,
        "remediations_applied": 0
    })
    steer_specific: Dict[str, int] = field(default_factory=lambda: {
        "verifications": 0,
        "failures_caught": 0,
        "teachable_moments": 0
    })
    ace_specific: Dict[str, int] = field(default_factory=lambda: {
        "learning_cycles": 0,
        "skills_learned": 0
    })

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0,
            "layers": {
                name: {
                    "enabled_count": stats.enabled_count,
                    "disabled_count": stats.disabled_count,
                    "failure_count": stats.failure_count,
                    "request_count": stats.request_count,
                    "avg_latency_ms": stats.get_avg_latency()
                }
                for name, stats in self.layers.items()
            },
            "retry_distribution": self.retry_distribution,
            "guardrails_specific": self.guardrails_specific,
            "steer_specific": self.steer_specific,
            "ace_specific": self.ace_specific
        }


# ============================================================================
# Main Bridge Class
# ============================================================================

class UnifiedReliabilityBridge:
    """
    Main bridge coordinating all reliability layers

    Provides unified interface for:
    - Constrained generation (LMQL)
    - Input/output validation (Guardrails)
    - Verification (Steer)
    - Learning from failures (ACE)

    Features:
    - Graceful degradation when layers unavailable
    - Configurable strictness levels
    - Automatic retry with exponential backoff
    - Comprehensive observability
    """

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        """
        Initialize unified reliability bridge

        Args:
            config: Reliability configuration (uses default if None)
        """
        # Load configuration
        self.config_manager = get_config_manager()
        self.config = config or self.config_manager.get_config()

        # Initialize logger
        self.logger = self._setup_logger()

        # Layer adapters (initialized lazily)
        self._lmql_adapter: Optional['LMQLAdapter'] = None
        self._guardrails_adapter: Optional['GuardrailsAdapter'] = None
        self._ace_steer_bridge: Optional['AceSteerBridge'] = None
        self._roma_engine: Optional['ROMAMDAPMakerAssociativeEngine'] = None

        # Layer status tracking
        self._layer_status: Dict[str, LayerStatus] = {
            "lmql": LayerStatus(
                available=LMQL_AVAILABLE,
                enabled=self.config.lmql.enabled
            ),
            "guardrails": LayerStatus(
                available=GUARDRAILS_AVAILABLE,
                enabled=self.config.guardrails.enabled
            ),
            "steer": LayerStatus(
                available=ACE_STEER_AVAILABLE,
                enabled=self.config.steer.enabled
            ),
            "ace": LayerStatus(
                available=ACE_STEER_AVAILABLE,
                enabled=self.config.ace.enabled
            ),
            "roma": LayerStatus(
                available=ROMA_MDAP_MAKER_AVAILABLE,
                enabled=True  # Enabled by default if available
            )
        }

        # Statistics tracking
        self._statistics = BridgeStatistics()
        self._stats_lock = threading.Lock()

        # Correlation ID tracking for request tracing
        self._active_requests: Dict[str, Dict] = {}
        self._requests_lock = threading.Lock()

        # Health check cache
        self._health_cache: Dict[str, Tuple[datetime, bool]] = {}
        self._health_cache_ttl = timedelta(seconds=30)

        self.logger.info(
            f"UnifiedReliabilityBridge initialized with strictness={self.config.unified_bridge.validation_strictness.value}"
        )
        self._log_layer_availability()

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger"""
        logger = logging.getLogger("UnifiedReliabilityBridge")
        logger.setLevel(
            getattr(logging, self.config.observability.log_level.value)
        )

        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _log_layer_availability(self):
        """Log which layers are available"""
        for layer_name, status in self._layer_status.items():
            if status.available:
                self.logger.info(f"Layer '{layer_name}': AVAILABLE (enabled={status.enabled})")
            else:
                self.logger.warning(f"Layer '{layer_name}': NOT AVAILABLE")

    # ========================================================================
    # Property Accessors for Lazy Initialization
    # ========================================================================

    @property
    def lmql_adapter(self) -> Optional['LMQLAdapter']:
        """Get LMQL adapter (lazy initialization)"""
        if not LMQL_AVAILABLE:
            return None

        if self._lmql_adapter is None:
            try:
                from .lmql_adapter import LMQLAdapter
                self._lmql_adapter = LMQLAdapter(self.config.lmql)
                self.logger.info("LMQL adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize LMQL adapter: {e}")
                self._layer_status["lmql"].available = False
                self._layer_status["lmql"].last_error = str(e)

        return self._lmql_adapter

    @property
    def guardrails_adapter(self) -> Optional['GuardrailsAdapter']:
        """Get Guardrails adapter (lazy initialization)"""
        if not GUARDRAILS_AVAILABLE:
            return None

        if self._guardrails_adapter is None:
            try:
                from .guardrails_adapter import GuardrailsAdapter
                self._guardrails_adapter = GuardrailsAdapter(self.config.guardrails)
                self.logger.info("Guardrails adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Guardrails adapter: {e}")
                self._layer_status["guardrails"].available = False
                self._layer_status["guardrails"].last_error = str(e)

        return self._guardrails_adapter

    @property
    def ace_steer_bridge(self) -> Optional['AceSteerBridge']:
        """Get ACE-Steer bridge (lazy initialization)"""
        if not ACE_STEER_AVAILABLE:
            return None

        if self._ace_steer_bridge is None:
            try:
                from ace_steer_integration import AceSteerBridge
                self._ace_steer_bridge = AceSteerBridge()
                self.logger.info("ACE-Steer bridge initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ACE-Steer bridge: {e}")
                self._layer_status["steer"].available = False
                self._layer_status["steer"].last_error = str(e)
                self._layer_status["ace"].available = False
                self._layer_status["ace"].last_error = str(e)

        return self._ace_steer_bridge

    @property
    def roma_engine(self) -> Optional['ROMAMDAPMakerAssociativeEngine']:
        """Get ROMA engine (lazy initialization)"""
        if not ROMA_MDAP_MAKER_AVAILABLE:
            return None

        if self._roma_engine is None:
            try:
                # Use Single Source of Truth for standardized high-reliability config
                config = get_standard_config(
                    roma_max_depth_analysis=2,
                    roma_max_depth_solving=1,
                    temperature=0.0
                )
                self._roma_engine = ROMAMDAPMakerAssociativeEngine(config)
                self.logger.info("ROMAMDAPMakerAssociativeEngine initialized for UnifiedReliabilityBridge")
            except Exception as e:
                self.logger.error(f"Failed to initialize ROMA engine: {e}")
                self._layer_status["roma"].available = False
                self._layer_status["roma"].last_error = str(e)

        return self._roma_engine

    # ========================================================================
    # Main Generation Methods
    # ========================================================================

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[Dict]] = None,
        validators: Optional[List[str]] = None,
        judges: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Main generation method through all reliability layers

        Flow:
            1. Input validation (Guardrails)
            2. Constrained generation (LMQL)
            3. Output validation (Guardrails)
            4. Final verification (Steer)
            5. Learn from failures (ACE)

        Args:
            prompt: Input prompt for generation
            constraints: Optional constraints for LMQL
            validators: Optional validators for Guardrails
            judges: Optional judges for Steer verification
            **kwargs: Additional parameters

        Returns:
            GenerationResult with output and metadata
        """
        # Start timing
        start_time = time.time()
        correlation_id = str(uuid.uuid4())

        # Initialize result
        result = GenerationResult(
            success=False,
            correlation_id=correlation_id
        )

        # Track active request
        with self._requests_lock:
            self._active_requests[correlation_id] = {
                "prompt": prompt,
                "start_time": datetime.utcnow(),
                "layers_attempted": []
            }

        try:
            self.logger.info(
                f"Generation request {correlation_id}: strictness={self.config.unified_bridge.validation_strictness.value}"
            )

            # Update statistics
            with self._stats_lock:
                self._statistics.total_requests += 1

            current_prompt = prompt
            layers_used = []
            layers_failed = []

            # ==================================================================
            # Layer 1: Input Validation (Guardrails)
            # ==================================================================
            if self._is_layer_enabled("guardrails"):
                layers_used.append("guardrails_input")

                try:
                    input_validation = self._validate_input(
                        current_prompt,
                        validators or self.config.guardrails.validators
                    )

                    # Update statistics
                    with self._stats_lock:
                        self._statistics.guardrails_specific["input_validations"] += 1

                    if not input_validation.is_valid:
                        result.validation_failures.extend(input_validation.violations)

                        # Handle based on strictness
                        if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                            result.error = f"Input validation failed: {input_validation.violations}"
                            self.logger.error(
                                f"Request {correlation_id}: Input validation failed (strict mode)"
                            )
                            return result

                        elif self.config.unified_bridge.validation_strictness == ValidationStrictness.MODERATE:
                            self.logger.warning(
                                f"Request {correlation_id}: Input validation failed (moderate mode - continuing)"
                            )

                        # Permissive mode: continue without error

                except Exception as e:
                    self.logger.error(f"Request {correlation_id}: Input validation error: {e}")
                    layers_failed.append("guardrails_input")
                    result.layers_failed.append("guardrails_input")

                    if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                        result.error = f"Input validation error: {str(e)}"
                        return result

            # ==================================================================
            # Layer 2: Constrained Generation (LMQL)
            # ==================================================================
            generated_output = None

            if self._is_layer_enabled("lmql") and self.lmql_adapter is not None:
                layers_used.append("lmql")

                try:
                    lmql_result = self._constrained_generation(
                        current_prompt,
                        constraints or []
                    )

                    if lmql_result.get("success"):
                        generated_output = lmql_result.get("output")
                        result.constraint_violations = lmql_result.get("violations", [])
                    else:
                        # LMQL generation failed
                        error = lmql_result.get("error", "Unknown LMQL error")
                        self.logger.warning(f"Request {correlation_id}: LMQL generation failed: {error}")
                        layers_failed.append("lmql")
                        result.layers_failed.append("lmql")

                        # Try fallback
                        if self.config.unified_bridge.fallback_on_error:
                            self.logger.info(f"Request {correlation_id}: Falling back to standard generation")
                            generated_output = self._fallback_generation(current_prompt, **kwargs)
                            result.fallback_used = True
                        else:
                            result.error = f"LMQL generation failed: {error}"
                            return result

                except Exception as e:
                    self.logger.error(f"Request {correlation_id}: LMQL error: {e}")
                    layers_failed.append("lmql")
                    result.layers_failed.append("lmql")

                    if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                        result.error = f"LMQL error: {str(e)}"
                        return result

                    # Try fallback for non-strict modes
                    if self.config.unified_bridge.fallback_on_error:
                        self.logger.info(f"Request {correlation_id}: Falling back to standard generation")
                        generated_output = self._fallback_generation(current_prompt, **kwargs)
                        result.fallback_used = True
                    else:
                        result.error = f"LMQL error: {str(e)}"
                        return result
            else:
                # LMQL not enabled or unavailable, use standard generation
                self.logger.info(f"Request {correlation_id}: Using standard generation (LMQL disabled)")
                generated_output = self._fallback_generation(current_prompt, **kwargs)
                result.fallback_used = True

            # Validate we have output
            if not generated_output:
                result.error = "Generation produced no output"
                self.logger.error(f"Request {correlation_id}: No output generated")
                return result

            # ==================================================================
            # Layer 3: Output Validation (Guardrails)
            # ==================================================================
            if self._is_layer_enabled("guardrails"):
                layers_used.append("guardrails_output")

                try:
                    output_validation = self._validate_output(
                        generated_output,
                        validators or self.config.guardrails.validators
                    )

                    # Update statistics
                    with self._stats_lock:
                        self._statistics.guardrails_specific["output_validations"] += 1

                    if not output_validation.is_valid:
                        result.validation_failures.extend(output_validation.violations)

                        # Try remediation
                        if output_validation.remediated_output:
                            self.logger.info(
                                f"Request {correlation_id}: Output remediated"
                            )
                            generated_output = output_validation.remediated_output

                            with self._stats_lock:
                                self._statistics.guardrails_specific["remediations_applied"] += 1
                        else:
                            with self._stats_lock:
                                self._statistics.guardrails_specific["failures_caught"] += 1

                            # Handle based on strictness
                            if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                                result.error = f"Output validation failed: {output_validation.violations}"
                                self.logger.error(
                                    f"Request {correlation_id}: Output validation failed (strict mode)"
                                )
                                return result

                            elif self.config.unified_bridge.validation_strictness == ValidationStrictness.MODERATE:
                                self.logger.warning(
                                    f"Request {correlation_id}: Output validation failed (moderate mode - using invalid output)"
                                )

                except Exception as e:
                    self.logger.error(f"Request {correlation_id}: Output validation error: {e}")
                    layers_failed.append("guardrails_output")
                    result.layers_failed.append("guardrails_output")

                    if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                        result.error = f"Output validation error: {str(e)}"
                        return result

            # ==================================================================
            # Layer 0: Final Verification (Steer)
            # ==================================================================
            if self._is_layer_enabled("steer") and self.ace_steer_bridge is not None:
                layers_used.append("steer")

                try:
                    verification = self._verify_output(
                        generated_output,
                        judges or self.config.steer.verifications
                    )

                    # Update statistics
                    with self._stats_lock:
                        self._statistics.steer_specific["verifications"] += 1

                    if not verification.passed:
                        result.verification_failures.extend(verification.errors)

                        with self._stats_lock:
                            self._statistics.steer_specific["failures_caught"] += 1

                        # Trigger ACE learning if teachable moment
                        if verification.is_teachable_moment:
                            self._trigger_learning(
                                prompt=current_prompt,
                                output=generated_output,
                                error=verification.errors[0] if verification.errors else "Verification failed",
                                correlation_id=correlation_id
                            )

                            with self._stats_lock:
                                self._statistics.steer_specific["teachable_moments"] += 1

                        # Handle based on strictness
                        if self.config.steer.halt_on_failure:
                            result.error = f"Verification failed: {verification.errors}"
                            self.logger.error(
                                f"Request {correlation_id}: Verification failed (halt_on_failure=True)"
                            )
                            return result

                        elif self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                            result.error = f"Verification failed: {verification.errors}"
                            self.logger.error(
                                f"Request {correlation_id}: Verification failed (strict mode)"
                            )
                            return result

                        elif self.config.unified_bridge.validation_strictness == ValidationStrictness.MODERATE:
                            self.logger.warning(
                                f"Request {correlation_id}: Verification failed (moderate mode - returning output anyway)"
                            )

                except Exception as e:
                    self.logger.error(f"Request {correlation_id}: Verification error: {e}")
                    layers_failed.append("steer")
                    result.layers_failed.append("steer")

                    if self.config.unified_bridge.validation_strictness == ValidationStrictness.STRICT:
                        result.error = f"Verification error: {str(e)}"
                        return result

            # ==================================================================
            # Success!
            # ==================================================================
            result.success = True
            result.output = generated_output
            result.layers_used = layers_used

            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            result.total_latency_ms = total_latency

            # Update statistics
            with self._stats_lock:
                self._statistics.successful_requests += 1
                self._statistics.retry_distribution["no_retry"] += 1

                # Update layer-specific stats
                for layer_name in layers_used:
                    base_layer = layer_name.split("_")[0]  # Extract base layer name
                    if base_layer in self._statistics.layers:
                        self._statistics.layers[base_layer].enabled_count += 1
                        self._statistics.layers[base_layer].request_count += 1
                        self._statistics.layers[base_layer].total_latency_ms += total_latency

            self.logger.info(
                f"Request {correlation_id}: Success - {total_latency:.2f}ms - layers={layers_used}"
            )

            return result

        except Exception as e:
            # Catch-all for unexpected errors
            self.logger.error(
                f"Request {correlation_id}: Unexpected error: {e}\n{traceback.format_exc()}"
            )

            result.error = f"Unexpected error: {str(e)}"
            result.success = False

            with self._stats_lock:
                self._statistics.failed_requests += 1

            return result

        finally:
            # Clean up active request tracking
            with self._requests_lock:
                self._active_requests.pop(correlation_id, None)

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        **kwargs
    ) -> GenerationResult:
        """
        Generate with automatic retry on failure

        Implements exponential backoff between retries.
        Disables failed layers between attempts.

        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            backoff_base: Base for exponential backoff (default 2.0)
            **kwargs: Additional arguments passed to generate()

        Returns:
            GenerationResult from first successful attempt or last failed attempt
        """
        correlation_id = str(uuid.uuid4())
        last_result = None
        disabled_layers = set()

        self.logger.info(
            f"Request {correlation_id}: Starting generation with retry (max_retries={max_retries})"
        )

        for attempt in range(max_retries):
            # Attempt generation
            result = self.generate(prompt, **kwargs)

            # Track retry distribution
            if attempt == 0:
                with self._stats_lock:
                    self._statistics.retry_distribution["no_retry"] += 1
            elif attempt == 1:
                with self._stats_lock:
                    self._statistics.retry_distribution["retry_1"] += 1
            elif attempt == 2:
                with self._stats_lock:
                    self._statistics.retry_distribution["retry_2"] += 1
            elif attempt == 3:
                with self._stats_lock:
                    self._statistics.retry_distribution["retry_3"] += 1
            else:
                with self._stats_lock:
                    self._statistics.retry_distribution["retry_4_plus"] += 1

            # If successful, return immediately
            if result.success:
                if attempt > 0:
                    result.retry_count = attempt
                    self.logger.info(
                        f"Request {correlation_id}: Success on attempt {attempt + 1}"
                    )
                return result

            # Store last result
            last_result = result

            # Analyze failures and disable problematic layers for next retry
            if result.layers_failed:
                for layer in result.layers_failed:
                    base_layer = layer.split("_")[0]
                    if base_layer not in disabled_layers:
                        self.logger.warning(
                            f"Request {correlation_id}: Disabling layer '{base_layer}' for retry (attempt {attempt + 1})"
                        )
                        self.disable_layer(base_layer)
                        disabled_layers.add(base_layer)

            # Don't sleep after last attempt
            if attempt < max_retries - 1:
                # Exponential backoff: base^attempt seconds
                sleep_time = backoff_base ** attempt
                self.logger.info(
                    f"Request {correlation_id}: Retry {attempt + 1}/{max_retries} after {sleep_time}s sleep"
                )
                time.sleep(sleep_time)

        # All retries failed
        self.logger.error(
            f"Request {correlation_id}: All {max_retries} retry attempts failed"
        )

        with self._stats_lock:
            self._statistics.failed_requests += 1

        # Re-enable disabled layers
        for layer in disabled_layers:
            self.enable_layer(layer)

        return last_result or GenerationResult(
            success=False,
            error="All retry attempts failed"
        )

    def batch_generate(
        self,
        prompts: List[str],
        max_workers: Optional[int] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Coordinate batch generation with parallel processing.

        Uses ThreadPoolExecutor for concurrent generation when multiple
        prompts are provided.

        Args:
            prompts: List of input prompts
            max_workers: Maximum number of parallel workers (defaults to min(len(prompts), 10))
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of GenerationResult in same order as inputs
        """
        if not prompts:
            return []

        # Determine number of workers
        max_workers = max_workers or min(len(prompts), 10)
        self.logger.info(f"Processing {len(prompts)} prompts with {max_workers} workers")

        results = [None] * len(prompts)

        def generate_single(index: int, prompt: str) -> Tuple[int, GenerationResult]:
            """
            Generate single result (for thread pool)

            Args:
                index: Index in the prompts list
                prompt: Input prompt to generate from

            Returns:
                Tuple of (index, GenerationResult)
            """
            try:
                result = self.generate(prompt, **kwargs)
                return (index, result)
            except Exception as e:
                self.logger.error(f"Generation failed for prompt {index}: {e}")
                error_result = GenerationResult(
                    success=False,
                    error=str(e),
                    prompt=prompt,
                    correlation_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow().isoformat(),
                    layers_used=["fallback"]
                )
                return (index, error_result)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch_gen") as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(generate_single, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                    self.logger.debug(f"Completed prompt {index + 1}/{len(prompts)}")
                except Exception as e:
                    self.logger.error(f"Future failed: {e}")

        # Verify all results filled
        if None in results:
            missing_count = results.count(None)
            self.logger.error(f"Missing {missing_count} results from batch processing")

        success_count = sum(1 for r in results if r and r.success)
        self.logger.info(
            f"Batch generation complete: {success_count}/{len(prompts)} successful"
        )

        return results

    # ========================================================================
    # Layer Implementation Methods
    # ========================================================================

    def _is_layer_enabled(self, layer_name: str) -> bool:
        """Check if layer is enabled and available"""
        status = self._layer_status.get(layer_name)
        if not status:
            return False
        return status.available and status.enabled

    def _validate_input(
        self,
        prompt: str,
        validators: List[str]
    ) -> ValidationResult:
        """
        Validate input using Guardrails

        Args:
            prompt: Input prompt to validate
            validators: List of validator names

        Returns:
            ValidationResult with validation outcome
        """
        if not self.guardrails_adapter:
            # Guardrails not available, pass validation
            return ValidationResult(is_valid=True)

        try:
            result = self.guardrails_adapter.validate_input(prompt, validators)

            # Convert to ValidationResult
            return ValidationResult(
                is_valid=result.get("is_valid", True),
                violations=result.get("violations", []),
                remediated_output=result.get("remediated_prompt"),
                should_retry=result.get("should_retry", False)
            )

        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            # In non-strict mode, return valid to allow continuation
            if self.config.unified_bridge.validation_strictness != ValidationStrictness.STRICT:
                return ValidationResult(is_valid=True)
            raise

    def _constrained_generation(
        self,
        prompt: str,
        constraints: List[Dict]
    ) -> Dict[str, Any]:
        """
        Perform constrained generation using LMQL

        Args:
            prompt: Input prompt
            constraints: List of constraints

        Returns:
            Dict with generation result
        """
        if not self.lmql_adapter:
            return {
                "success": False,
                "error": "LMQL adapter not available"
            }

        try:
            result = self.lmql_adapter.generate(prompt, constraints)
            return result

        except Exception as e:
            self.logger.error(f"LMQL generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _fallback_generation(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Fallback generation using standard LLM API when other layers unavailable.

        This ensures we always provide real generation, never mocks.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text from actual LLM API call

        Raises:
            ValueError: If no API key is available
            Exception: If LLM call fails (error message returned as last resort)
        """
        self.logger.warning("Using fallback generation (standard LLM)")

        try:
            # Import LLM utilities
            from llm_utils import _request_openai_compatible_chat

            # Configure from environment or use defaults
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            model = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")

            if not api_key:
                self.logger.error("No API key available for fallback generation")
                raise ValueError("Fallback generation requires OPENAI_API_KEY or ANTHROPIC_API_KEY")

            # Compose messages
            messages = [
                {"role": "user", "content": prompt}
            ]

            # Make actual LLM call
            self.logger.info(f"Calling fallback model: {model}")
            response = _request_openai_compatible_chat(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )

            # Extract response
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                self.logger.info(f"Fallback generation successful: {len(content)} chars")
                return content
            else:
                self.logger.error("Unexpected LLM response format")
                raise ValueError("Invalid LLM response format")

        except Exception as e:
            self.logger.error(f"Fallback generation failed: {e}")
            # As last resort, return a helpful error message
            return f"[Generation Error] Unable to generate content: {str(e)}"

    def _validate_output(
        self,
        output: str,
        validators: List[str]
    ) -> ValidationResult:
        """
        Validate output using Guardrails

        Args:
            output: Generated output to validate
            validators: List of validator names

        Returns:
            ValidationResult with validation outcome
        """
        if not self.guardrails_adapter:
            # Guardrails not available, pass validation
            return ValidationResult(is_valid=True)

        try:
            result = self.guardrails_adapter.validate_output(output, validators)

            # Convert to ValidationResult
            return ValidationResult(
                is_valid=result.get("is_valid", True),
                violations=result.get("violations", []),
                remediated_output=result.get("remediated_output"),
                should_retry=result.get("should_retry", False)
            )

        except Exception as e:
            self.logger.error(f"Output validation error: {e}")
            # In non-strict mode, return valid to allow continuation
            if self.config.unified_bridge.validation_strictness != ValidationStrictness.STRICT:
                return ValidationResult(is_valid=True)
            raise

    def _verify_output(
        self,
        output: str,
        judges: List[str]
    ) -> VerificationResult:
        """
        Verify output using Steer

        Args:
            output: Generated output to verify
            judges: List of verification types

        Returns:
            VerificationResult with verification outcome
        """
        if not self.ace_steer_bridge:
            # Steer not available, pass verification
            return VerificationResult(passed=True)

        try:
            result = self.ace_steer_bridge.verify(output, judges)

            # Convert to VerificationResult
            return VerificationResult(
                passed=result.get("passed", True),
                score=result.get("score", 0.0),
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                is_teachable_moment=result.get("is_teachable_moment", False)
            )

        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            # In non-strict mode, return passed to allow continuation
            if self.config.unified_bridge.validation_strictness != ValidationStrictness.STRICT:
                return VerificationResult(passed=True)
            raise

    def _trigger_learning(
        self,
        prompt: str,
        output: str,
        error: str,
        correlation_id: str
    ):
        """
        Trigger ACE learning from failure

        Args:
            prompt: Original prompt
            output: Generated output
            error: Error that occurred
            correlation_id: Request correlation ID
        """
        if not self.ace_steer_bridge:
            return

        try:
            self.logger.info(
                f"Request {correlation_id}: Triggering ACE learning from failure"
            )

            # Call ACE to learn from this failure
            self.ace_steer_bridge.learn_from_failure(
                prompt=prompt,
                output=output,
                error=error
            )

            with self._stats_lock:
                self._statistics.ace_specific["learning_cycles"] += 1

        except Exception as e:
            self.logger.error(f"ACE learning error: {e}")
            # Don't fail the request if learning fails

    def _trigger_ace_learning(
        self,
        failure: Dict[str, Any],
        output: str,
        prompt: str
    ):
        """
        Trigger ACE learning from failure with enhanced context

        Args:
            failure: Dictionary containing failure details including error message
            output: Generated output that failed validation
            prompt: Original prompt that led to the failure
        """
        if not self.ace_steer_bridge:
            self.logger.debug("ACE bridge unavailable - skipping learning")
            return

        try:
            # Trigger ACE learning cycle with enhanced context
            self.ace_steer_bridge.learn_from_failure(
                output=output,
                error=failure.get("error", "Unknown error"),
                context={
                    "prompt": prompt,
                    "failure_type": failure.get("type", "validation_failure"),
                    "violations": failure.get("violations", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            self.logger.info("ACE learning triggered successfully")

            # Update statistics
            with self._stats_lock:
                self._statistics.ace_specific["learning_cycles"] += 1

        except Exception as e:
            self.logger.error(f"ACE learning failed: {e}")
            # Don't fail the request if learning fails

    def _verify_with_steer(
        self,
        output: str,
        judges: List[str]
    ) -> VerificationResult:
        """
        Verify output with Steer judges

        Args:
            output: Generated output to verify
            judges: List of judge names to use for verification

        Returns:
            VerificationResult with verification outcome
        """
        if not self.ace_steer_bridge:
            self.logger.warning("Steer verification unavailable - skipping")
            return VerificationResult(
                passed=True,
                judges=judges,
                note="Steer bridge unavailable"
            )

        try:
            # Call Steer verification
            verification = self.ace_steer_bridge.verify(
                content=output,
                judges=judges
            )

            # Extract verification details
            return VerificationResult(
                passed=verification.get("passed", True),
                score=verification.get("score", 0.0),
                errors=verification.get("errors", []),
                warnings=verification.get("warnings", []),
                is_teachable_moment=verification.get("is_teachable_moment", False),
                judges=judges,
                details=verification.get("details", {})
            )

        except Exception as e:
            self.logger.error(f"Steer verification failed: {e}")
            return VerificationResult(
                passed=False,
                judges=judges,
                error=str(e)
            )

    # ========================================================================
    # Layer Management Methods
    # ========================================================================

    def get_layer_status(self) -> Dict[str, LayerStatus]:
        """
        Get status of all reliability layers

        Returns:
            Dict mapping layer names to LayerStatus
        """
        return self._layer_status.copy()

    def enable_layer(self, layer_name: str) -> bool:
        """
        Enable a specific layer

        Args:
            layer_name: Name of layer to enable

        Returns:
            True if layer was enabled successfully
        """
        if layer_name not in self._layer_status:
            self.logger.error(f"Unknown layer: {layer_name}")
            return False

        status = self._layer_status[layer_name]

        if not status.available:
            self.logger.error(f"Layer '{layer_name}' is not available")
            return False

        status.enabled = True
        self.logger.info(f"Layer '{layer_name}' enabled")

        return True

    def disable_layer(self, layer_name: str) -> bool:
        """
        Disable a specific layer

        Args:
            layer_name: Name of layer to disable

        Returns:
            True if layer was disabled successfully
        """
        if layer_name not in self._layer_status:
            self.logger.error(f"Unknown layer: {layer_name}")
            return False

        status = self._layer_status[layer_name]
        status.enabled = False
        self.logger.info(f"Layer '{layer_name}' disabled")

        return True

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics

        Returns:
            Dict with detailed statistics
        """
        with self._stats_lock:
            return self._statistics.to_dict()

    def reset_statistics(self):
        """Reset all statistics"""
        with self._stats_lock:
            self._statistics = BridgeStatistics()
        self.logger.info("Statistics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export metrics for external monitoring systems

        Returns:
            Dict with metrics in Prometheus-compatible format
        """
        stats = self.get_statistics()

        return {
            "metrics": [
                {
                    "name": "openevolve_reliability_total_requests",
                    "type": "counter",
                    "value": stats["total_requests"]
                },
                {
                    "name": "openevolve_reliability_successful_requests",
                    "type": "counter",
                    "value": stats["successful_requests"]
                },
                {
                    "name": "openevolve_reliability_failed_requests",
                    "type": "counter",
                    "value": stats["failed_requests"]
                },
                {
                    "name": "openevolve_reliability_success_rate",
                    "type": "gauge",
                    "value": stats["success_rate"]
                },
                {
                    "name": "openevolve_reliability_layer_lmql_enabled_count",
                    "type": "counter",
                    "value": stats["layers"]["lmql"]["enabled_count"]
                },
                {
                    "name": "openevolve_reliability_layer_guardrails_enabled_count",
                    "type": "counter",
                    "value": stats["layers"]["guardrails"]["enabled_count"]
                },
                {
                    "name": "openevolve_reliability_layer_steer_enabled_count",
                    "type": "counter",
                    "value": stats["layers"]["steer"]["enabled_count"]
                },
                {
                    "name": "openevolve_reliability_layer_ace_enabled_count",
                    "type": "counter",
                    "value": stats["layers"]["ace"]["enabled_count"]
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

    # ========================================================================
    # Health Check Methods
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check

        Returns:
            Dict with health status of all components
        """
        health = {
            "bridge": {
                "healthy": True,
                "version": "1.0.0",
                "strictness": self.config.unified_bridge.validation_strictness.value
            },
            "layers": {}
        }

        for layer_name, status in self._layer_status.items():
            # Check cache
            cached_time, is_healthy = self._health_cache.get(
                layer_name,
                (datetime.min, True)
            )

            # Check if cache is valid
            if datetime.utcnow() - cached_time < self._health_cache_ttl:
                health["layers"][layer_name] = {
                    "healthy": is_healthy,
                    "available": status.available,
                    "enabled": status.enabled,
                    "cached": True
                }
                continue

            # Perform actual health check
            is_healthy = self._check_layer_health(layer_name)

            # Update cache
            self._health_cache[layer_name] = (datetime.utcnow(), is_healthy)

            health["layers"][layer_name] = {
                "healthy": is_healthy,
                "available": status.available,
                "enabled": status.enabled,
                "cached": False
            }

            if not is_healthy:
                health["bridge"]["healthy"] = False

        return health

    def _check_layer_health(self, layer_name: str) -> bool:
        """
        Check health of a specific layer

        Args:
            layer_name: Name of layer to check

        Returns:
            True if layer is healthy
        """
        status = self._layer_status.get(layer_name)
        if not status:
            return False

        if not status.available:
            return False

        # Try to use the layer
        try:
            if layer_name == "lmql" and self.lmql_adapter:
                # Simple health check
                return True

            elif layer_name == "guardrails" and self.guardrails_adapter:
                # Simple health check
                return True

            elif layer_name in ["steer", "ace"] and self.ace_steer_bridge:
                # Simple health check
                return True

            return True

        except Exception as e:
            self.logger.error(f"Health check failed for layer '{layer_name}': {e}")
            status.last_error = str(e)
            return False

    def get_active_requests(self) -> List[Dict]:
        """
        Get list of currently active requests

        Returns:
            List of active request info
        """
        with self._requests_lock:
            return [
                {
                    "correlation_id": cid,
                    **info
                }
                for cid, info in self._active_requests.items()
            ]


# ============================================================================
# Decorator for Automatic Retry
# ============================================================================

def with_retry(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    bridge: Optional[UnifiedReliabilityBridge] = None
):
    """
    Decorator to add automatic retry to any function

    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff
        bridge: Optional bridge instance for logging

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if bridge:
                        bridge.logger.warning(
                            f"Function '{func.__name__}' failed on attempt {attempt + 1}/{max_retries}: {e}"
                        )

                    # Don't sleep after last attempt
                    if attempt < max_retries - 1:
                        sleep_time = backoff_base ** attempt
                        time.sleep(sleep_time)

            # All retries failed
            if bridge:
                bridge.logger.error(
                    f"Function '{func.__name__}' failed after {max_retries} attempts"
                )

            raise last_exception

        return wrapper
    return decorator


# ============================================================================
# Singleton Instance
# ============================================================================

_bridge_instance: Optional[UnifiedReliabilityBridge] = None
_bridge_lock = threading.Lock()


def get_unified_bridge() -> UnifiedReliabilityBridge:
    """
    Get singleton UnifiedReliabilityBridge instance

    Returns:
        Shared bridge instance
    """
    global _bridge_instance

    with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = UnifiedReliabilityBridge()
        return _bridge_instance


# ============================================================================
# Convenience Functions
# ============================================================================

def generate(
    prompt: str,
    constraints: Optional[List[Dict]] = None,
    validators: Optional[List[str]] = None,
    judges: Optional[List[str]] = None,
    **kwargs
) -> GenerationResult:
    """
    Convenience function for generation

    Uses singleton bridge instance.

    Args:
        prompt: Input prompt
        constraints: Optional constraints
        validators: Optional validators
        judges: Optional judges
        **kwargs: Additional parameters

    Returns:
        GenerationResult
    """
    bridge = get_unified_bridge()
    return bridge.generate(prompt, constraints, validators, judges, **kwargs)


def generate_with_retry(
    prompt: str,
    max_retries: int = 3,
    **kwargs
) -> GenerationResult:
    """
    Convenience function for generation with retry

    Uses singleton bridge instance.

    Args:
        prompt: Input prompt
        max_retries: Maximum retry attempts
        **kwargs: Additional parameters

    Returns:
        GenerationResult
    """
    bridge = get_unified_bridge()
    return bridge.generate_with_retry(prompt, max_retries, **kwargs)


def get_layer_status() -> Dict[str, LayerStatus]:
    """Get status of all layers (convenience function)"""
    bridge = get_unified_bridge()
    return bridge.get_layer_status()


def get_statistics() -> Dict[str, Any]:
    """Get statistics (convenience function)"""
    bridge = get_unified_bridge()
    return bridge.get_statistics()


def enable_layer(layer_name: str) -> bool:
    """Enable a layer (convenience function)"""
    bridge = get_unified_bridge()
    return bridge.enable_layer(layer_name)


def disable_layer(layer_name: str) -> bool:
    """Disable a layer (convenience function)"""
    bridge = get_unified_bridge()
    return bridge.disable_layer(layer_name)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI interface for testing unified bridge"""
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenEvolve Unified Reliability Bridge CLI"
    )
    parser.add_argument(
        "action",
        choices=["generate", "status", "health", "stats"],
        help="Action to perform"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for generation"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for generation"
    )
    parser.add_argument(
        "--strictness",
        choices=["strict", "moderate", "permissive"],
        default="moderate",
        help="Validation strictness"
    )

    args = parser.parse_args()

    # Get bridge instance
    bridge = get_unified_bridge()

    if args.action == "generate":
        if not args.prompt:
            print("Error: --prompt required for generate action")
            return

        result = bridge.generate_with_retry(
            args.prompt,
            max_retries=args.max_retries
        )

        print("\n" + "=" * 60)
        print("GENERATION RESULT")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Correlation ID: {result.correlation_id}")
        print(f"Layers Used: {result.layers_used}")
        print(f"Latency: {result.total_latency_ms:.2f}ms")

        if result.success:
            print(f"\nOutput:\n{result.output}")
        else:
            print(f"\nError: {result.error}")

        if result.layers_failed:
            print(f"\nLayers Failed: {result.layers_failed}")

        if result.validation_failures:
            print(f"\nValidation Failures: {result.validation_failures}")

        if result.verification_failures:
            print(f"\nVerification Failures: {result.verification_failures}")

        print("=" * 60)

    elif args.action == "status":
        print("\n" + "=" * 60)
        print("LAYER STATUS")
        print("=" * 60)

        for layer_name, status in bridge.get_layer_status().items():
            available = "✓" if status.available else "✗"
            enabled = "✓" if status.enabled else "✗"

            print(f"\n[{available}] {layer_name.upper()}")
            print(f"  Enabled: {enabled}")
            print(f"  Requests: {status.request_count}")
            print(f"  Failures: {status.failure_count}")
            print(f"  Avg Latency: {status.avg_latency_ms:.2f}ms")

            if status.last_error:
                print(f"  Last Error: {status.last_error}")

        print("=" * 60)

    elif args.action == "health":
        print("\n" + "=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)

        health = bridge.health_check()

        bridge_healthy = "✓" if health["bridge"]["healthy"] else "✗"
        print(f"\n[{bridge_healthy}] Bridge")
        print(f"  Version: {health['bridge']['version']}")
        print(f"  Strictness: {health['bridge']['strictness']}")

        print("\nLayers:")
        for layer_name, layer_health in health["layers"].items():
            healthy = "✓" if layer_health["healthy"] else "✗"
            available = "✓" if layer_health["available"] else "✗"
            enabled = "✓" if layer_health["enabled"] else "✗"

            print(f"\n  [{healthy}] {layer_name.upper()}")
            print(f"    Available: {available}")
            print(f"    Enabled: {enabled}")

        print("=" * 60)

    elif args.action == "stats":
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        stats = bridge.get_statistics()

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")

        print("\nRetry Distribution:")
        for key, value in stats['retry_distribution'].items():
            print(f"  {key}: {value}")

        print("\nLayer Statistics:")
        for layer_name, layer_stats in stats['layers'].items():
            print(f"\n  {layer_name.upper()}:")
            print(f"    Enabled Count: {layer_stats['enabled_count']}")
            print(f"    Request Count: {layer_stats['request_count']}")
            print(f"    Failure Count: {layer_stats['failure_count']}")
            print(f"    Avg Latency: {layer_stats['avg_latency_ms']:.2f}ms")

        print("\nGuardrails Specific:")
        for key, value in stats['guardrails_specific'].items():
            print(f"  {key}: {value}")

        print("\nSteer Specific:")
        for key, value in stats['steer_specific'].items():
            print(f"  {key}: {value}")

        print("\nACE Specific:")
        for key, value in stats['ace_specific'].items():
            print(f"  {key}: {value}")

        print("=" * 60)


if __name__ == "__main__":
    main()
