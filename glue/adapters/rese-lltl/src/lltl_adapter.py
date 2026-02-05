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
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
import uuid

# Add glue/lib to path for imports
glue_lib_path = Path(__file__).parent.parent.parent.parent / "lib"
sys.path.insert(0, str(glue_lib_path))

# Add parent directory for z3prover_integration
parent_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(parent_path))

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

# Try to import Z3ProverIntegration for contradiction detection
Z3_AVAILABLE = False
Z3_IMPORT_ERROR = None
try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Config,
        Z3ResultStatus,
        Z3Variable,
        Z3Constraint,
        Z3ConstraintType,
        is_z3_available
    )
    Z3_AVAILABLE = is_z3_available()
except ImportError as e:
    Z3_IMPORT_ERROR = str(e)
    Z3_AVAILABLE = False


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


# ============================================================================
# FORMAL COMMITMENT (DEE → SCE)
# ============================================================================

@dataclass
class FormalCommitment:
    """
    A formal propositional commitment for SCE integration

    Represents a statistical result as a formal logical proposition
    that can be integrated into the SCE logic graph for contradiction
    detection and auditability.

    From RESE Technical Manual §2.2:
    "DEE → SCE (Auditability): The DEE's statistical results are converted
    into auditable Formal Propositional Commitments by assigning explicit
    Confidence Thresholds that the SCE can integrate into its logic graph
    for contradiction detection."
    """
    proposition_id: str
    statement: str  # Formal logical statement
    confidence_threshold: float  # 0-1, minimum confidence to accept
    statistical_evidence: Dict[str, float]  # p-value, confidence interval, etc.
    source_hypothesis: str  # ID of hypothesis
    derivation_method: str  # How this was derived (e.g., "mcts_validation")
    timestamp: str  # UTC ISO-8601
    correlation_id: str
    lean4_theorem: Optional[str] = None  # Lean 4 formalization (future)

    def to_sce_constraint(self) -> Dict[str, Any]:
        """
        Convert to SCE constraint format

        Returns constraint dict that SCE can integrate into logic graph
        """
        return {
            "constraint_id": self.proposition_id,
            "formal_statement": self.statement,
            "confidence": self.confidence_threshold,
            "evidence": self.statistical_evidence,
            "type": "statistical_commitment"
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "proposition_id": self.proposition_id,
            "statement": self.statement,
            "confidence_threshold": self.confidence_threshold,
            "statistical_evidence": self.statistical_evidence,
            "source_hypothesis": self.source_hypothesis,
            "derivation_method": self.derivation_method,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "lean4_theorem": self.lean4_theorem
        }


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

        # Initialize auditability components for DEE → SCE
        self.committed_propositions: Dict[str, FormalCommitment] = {}
        self.auditability_enabled = os.getenv("LLTL_AUDITABILITY_ENABLED", "true").lower() == "true"
        self.default_confidence_threshold = float(os.getenv("LLTL_CONFIDENCE_THRESHOLD_DEFAULT", "0.75"))
        self.significance_level = float(os.getenv("LLTL_SIGNIFICANCE_LEVEL", "0.05"))
        self.audit_timeout_ms = int(os.getenv("LLTL_AUDIT_TIMEOUT_MS", "5000"))

        # Initialize Z3 for contradiction detection (Priority 5 MEDIUM integration)
        self.z3_enabled = os.getenv("RESE_Z3_LLTL_ENABLED", "true").lower() == "true"
        self.z3_timeout_ms = int(os.getenv("Z3_TIMEOUT", "5000"))
        self.z3_solver = None

        if self.z3_enabled and Z3_AVAILABLE:
            try:
                z3_config = Z3Config(
                    timeout=self.z3_timeout_ms / 1000.0,  # Convert to seconds
                    unsat_core=True,  # Enable unsat core for contradiction extraction
                    proof_generation=False  # Don't need full proofs for contradiction detection
                )
                self.z3_solver = Z3SolverEngine(config=z3_config)
                logger.log("INFO", "Z3 contradiction detection initialized",
                          operation="initialize",
                          z3_available=True,
                          timeout_ms=self.z3_timeout_ms)
            except Exception as e:
                logger.log("WARNING", f"Z3 initialization failed, falling back to naive method: {e}",
                          operation="initialize")
                self.z3_solver = None
                self.z3_enabled = False
        elif self.z3_enabled and not Z3_AVAILABLE:
            logger.log("WARNING", "Z3 enabled but not available, falling back to naive method",
                      operation="initialize",
                      z3_available=False,
                      z3_import_error=Z3_IMPORT_ERROR)
            self.z3_enabled = False

        logger.log("INFO", "LLTL adapter initialized",
                  operation="initialize",
                  success=True,
                  z3_enabled=self.z3_enabled)

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

        Uses Z3 for efficient contradiction detection (O(n log n) vs O(n²) naive).
        Falls back to naive DITO if Z3 is not available.

        From RESE Technical Manual §2.2:
        Use Z3 for efficient contradiction detection.

        Args:
            constraints: List of constraints to check (FormalCommitment or compatible)
            correlation_id: For tracing

        Returns:
            Tuple of (contradictions_list, error_message)
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            # Filter to FormalCommitment objects
            formal_commitments = [c for c in constraints if isinstance(c, FormalCommitment)]

            if not formal_commitments:
                # No formal commitments, use existing DITO
                contradictions, error = self.translator.dito.detect_contradictions(
                    constraints=constraints,
                    correlation_id=correlation_id
                )
                return contradictions, error

            # Use Z3 if enabled and available
            if self.z3_enabled and self.z3_solver is not None:
                contradictions = self._detect_contradictions_z3(
                    formal_commitments=formal_commitments,
                    correlation_id=correlation_id
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.log("INFO", f"Z3 detected {len(contradictions)} contradictions",
                          correlation_id=correlation_id,
                          operation="detect_contradictions_z3",
                          success=True,
                          num_commitments=len(formal_commitments),
                          duration_ms=duration_ms,
                          solver="z3")

                return contradictions, None
            else:
                # Fallback to naive method
                contradictions = self._detect_contradictions_naive(
                    formal_commitments=formal_commitments,
                    correlation_id=correlation_id
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.log("INFO", f"Naive method detected {len(contradictions)} contradictions",
                          correlation_id=correlation_id,
                          operation="detect_contradictions_naive",
                          success=True,
                          num_commitments=len(formal_commitments),
                          duration_ms=duration_ms,
                          solver="naive")

                return contradictions, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Contradiction detection error: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="detect_contradictions",
                      success=False,
                      duration_ms=duration_ms)
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

    # ==========================================================================
    # DEE → SCE: AUDITABILITY COMPONENT
    # ==========================================================================

    def statistical_to_formal(
        self,
        statistical_result: Dict[str, Union[float, List, Dict, str]],
        source_hypothesis: str,
        derivation_method: str,
        correlation_id: Optional[str] = None
    ) -> Tuple[Optional[FormalCommitment], Optional[str]]:
        """
        Convert DEE statistical result to Formal Propositional Commitment

        This is the missing DEE → SCE component specified in §2.2

        From RESE Technical Manual §2.2:
        "DEE → SCE (Auditability): The DEE's statistical results are converted
        into auditable Formal Propositional Commitments by assigning explicit
        Confidence Thresholds that the SCE can integrate into its logic graph
        for contradiction detection."

        Args:
            statistical_result: Dict with:
                - hypothesis_statement: str
                - confidence: float (0-1)
                - p_value: float
                - confidence_interval: Tuple[float, float]
                - expected_value: float
                - validation_metric: str (e.g., "aci_score")
                - evidence: List[Dict]

            source_hypothesis: ID of hypothesis
            derivation_method: How result was derived (e.g., "mcts_validation")
            correlation_id: For tracking

        Returns:
            Tuple of (FormalCommitment, error_message)
        """
        start_time = time.time()
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            # Validate required fields
            required_fields = ['hypothesis_statement', 'confidence']
            missing_fields = [f for f in required_fields if f not in statistical_result]
            if missing_fields:
                return None, f"Missing required fields: {missing_fields}"

            # 1. Extract statistical evidence
            confidence = float(statistical_result.get('confidence', 0.0))
            p_value = float(statistical_result.get('p_value', 1.0))
            confidence_interval = statistical_result.get('confidence_interval', (0.0, 1.0))

            # Handle confidence_interval as tuple or list
            if isinstance(confidence_interval, (list, tuple)) and len(confidence_interval) >= 2:
                ci_lower, ci_upper = float(confidence_interval[0]), float(confidence_interval[1])
            else:
                ci_lower, ci_upper = 0.0, 1.0

            # 2. Determine confidence threshold
            # High confidence = lower threshold (more certain)
            # Low confidence = higher threshold (more skeptical)
            confidence_threshold = self._calculate_confidence_threshold(confidence)

            # 3. Construct formal logical statement
            hypothesis_stmt = str(statistical_result['hypothesis_statement'])

            formal_statement = self._construct_formal_statement(
                hypothesis=hypothesis_stmt,
                confidence=confidence,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper)
            )

            # 4. Create formal commitment
            commitment = FormalCommitment(
                proposition_id=str(uuid.uuid4()),
                statement=formal_statement,
                confidence_threshold=confidence_threshold,
                statistical_evidence={
                    'confidence': confidence,
                    'p_value': p_value,
                    'confidence_interval_lower': ci_lower,
                    'confidence_interval_upper': ci_upper,
                    'expected_value': float(statistical_result.get('expected_value', 0.0))
                },
                source_hypothesis=source_hypothesis,
                derivation_method=derivation_method,
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id
            )

            # 5. Store for auditability (Law of Idempotency)
            self.committed_propositions[commitment.proposition_id] = commitment

            duration_ms = (time.time() - start_time) * 1000
            logger.log("INFO", "Statistical result converted to formal commitment",
                      correlation_id=correlation_id,
                      operation="statistical_to_formal",
                      proposition_id=commitment.proposition_id,
                      confidence=confidence,
                      threshold=confidence_threshold,
                      duration_ms=duration_ms)

            return commitment, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Statistical to formal conversion failed: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="statistical_to_formal",
                      duration_ms=duration_ms)
            return None, error_msg

    def _calculate_confidence_threshold(self, confidence: float) -> float:
        """
        Calculate confidence threshold from statistical confidence

        High confidence = lower threshold (more certain)
        Low confidence = higher threshold (more skeptical)

        Args:
            confidence: Statistical confidence (0-1)

        Returns:
            float: Confidence threshold
        """
        if confidence >= 0.95:
            # Very high confidence - accept with high threshold
            return 0.90
        elif confidence >= 0.80:
            # High confidence - moderate threshold
            return 0.75
        elif confidence >= 0.60:
            # Moderate confidence - conservative threshold
            return 0.60
        else:
            # Low confidence - very conservative threshold
            return 0.50

    def _construct_formal_statement(
        self,
        hypothesis: str,
        confidence: float,
        p_value: float,
        confidence_interval: Tuple[float, float]
    ) -> str:
        """
        Construct formal logical statement from statistical evidence

        Format: "H ∧ (confidence ≥ T) ∧ (p ≤ α) → Accept(H)"

        Args:
            hypothesis: Hypothesis statement
            confidence: Statistical confidence
            p_value: Statistical significance
            confidence_interval: (lower, upper) bounds

        Returns:
            str: Formal logical statement
        """
        # Significance level
        α = self.significance_level

        # Truncate hypothesis for readability
        hypothesis_short = hypothesis[:50] + "..." if len(hypothesis) > 50 else hypothesis

        # Construct statement
        statement = (
            f"({hypothesis}) ∧ "
            f"(confidence ≥ {confidence:.3f}) ∧ "
            f"(p_value ≤ {α:.3f}) ∧ "
            f"(CI ∈ [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]) "
            f"→ Accept({hypothesis_short})"
        )

        return statement

    def integrate_into_sce(
        self,
        commitment: FormalCommitment,
        sce_engine: Any,
        correlation_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Integrate formal commitment into SCE logic graph

        This enables auditability and contradiction detection

        Args:
            commitment: Formal commitment to integrate
            sce_engine: SymbolicConstraintEngine instance (or compatible)
            correlation_id: For tracking

        Returns:
            Tuple of (success, error_message)
        """
        start_time = time.time()
        correlation_id = correlation_id or commitment.correlation_id

        try:
            if not self.auditability_enabled:
                logger.log("INFO", "SCE integration disabled (LLTL_AUDITABILITY_ENABLED=false)",
                          correlation_id=correlation_id,
                          operation="integrate_into_sce")
                return True, None

            # 1. Convert to SCE constraint format
            sce_constraint = commitment.to_sce_constraint()

            # 2. Add to SCE logic graph
            # Check if sce_engine has add_constraint method
            if hasattr(sce_engine, 'add_constraint'):
                # Import SCE Constraint if available
                try:
                    # Use importlib for module with hyphens
                    import importlib
                    import importlib.util
                    import sys
                    from pathlib import Path

                    # Build path to SCE bridge
                    sce_bridge_path = Path(__file__).parent.parent.parent / "rese-sce" / "src" / "sce_bridge.py"

                    if sce_bridge_path.exists():
                        spec = importlib.util.spec_from_file_location("sce_bridge", sce_bridge_path)
                        sce_bridge_module = importlib.util.module_from_spec(spec)
                        sys.modules['sce_bridge'] = sce_bridge_module
                        spec.loader.exec_module(sce_bridge_module)

                        Constraint = sce_bridge_module.Constraint
                        ConstraintType = sce_bridge_module.ConstraintType
                        ConstraintCategory = sce_bridge_module.ConstraintCategory

                        constraint = Constraint(
                            constraint_id=sce_constraint['constraint_id'],
                            type=ConstraintType.SOFT,  # Statistical commitments are soft constraints
                            category=ConstraintCategory.SOFT_STATISTICAL,
                            description=sce_constraint['formal_statement'],
                            expression=sce_constraint['formal_statement']
                        )

                        # Use async add_constraint
                        import asyncio
                        if hasattr(sce_engine, 'add_constraint'):
                            try:
                                # Try running in event loop if one exists
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # Create task for async execution
                                    asyncio.create_task(sce_engine.add_constraint(constraint, correlation_id))
                                else:
                                    # Run synchronously
                                    loop.run_until_complete(sce_engine.add_constraint(constraint, correlation_id))
                            except RuntimeError:
                                # No event loop, create new one
                                asyncio.run(sce_engine.add_constraint(constraint, correlation_id))
                    else:
                        raise ImportError("SCE bridge module not found")

                except (ImportError, AttributeError) as import_error:
                    # Fallback: Try direct call without SCE types
                    logger.log("WARNING", f"SCE types not available, using direct integration: {import_error}",
                              correlation_id=correlation_id,
                              operation="integrate_into_sce")
                    if callable(getattr(sce_engine, 'add_constraint', None)):
                        try:
                            # Try direct call with dict
                            sce_engine.add_constraint(
                                constraint_id=sce_constraint['constraint_id'],
                                formal_statement=sce_constraint['formal_statement'],
                                confidence=sce_constraint['confidence']
                            )
                        except TypeError:
                            # Try with constraint object as dict
                            sce_engine.add_constraint(sce_constraint, correlation_id)
            else:
                logger.log("WARNING", "SCE engine does not have add_constraint method",
                          correlation_id=correlation_id,
                          operation="integrate_into_sce")
                return False, "SCE engine missing add_constraint method"

            # 3. Check for contradictions with existing commitments
            if hasattr(sce_engine, 'detect_contradictions'):
                try:
                    # Try async detection
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule for async execution
                            asyncio.create_task(sce_engine.detect_contradictions(correlation_id))
                        else:
                            # Run synchronously
                            contradiction_result = loop.run_until_complete(
                                sce_engine.detect_contradictions(correlation_id)
                            )
                            if hasattr(contradiction_result, 'contradictions'):
                                contradictions = contradiction_result.contradictions
                                if contradictions and len(contradictions) > 0:
                                    logger.log("WARNING", "Formal commitment contradicts existing knowledge",
                                              correlation_id=correlation_id,
                                              operation="integrate_into_sce",
                                              proposition_id=commitment.proposition_id,
                                              contradictions=len(contradictions))
                                    # Don't fail integration, but log warning
                    except RuntimeError:
                        # No event loop
                        contradiction_result = asyncio.run(
                            sce_engine.detect_contradictions(correlation_id)
                        )
                        if hasattr(contradiction_result, 'contradictions'):
                            contradictions = contradiction_result.contradictions
                            if contradictions and len(contradictions) > 0:
                                logger.log("WARNING", "Formal commitment contradicts existing knowledge",
                                          correlation_id=correlation_id,
                                          operation="integrate_into_sce",
                                          proposition_id=commitment.proposition_id,
                                          contradictions=len(contradictions))
                except Exception as e:
                    logger.log("WARNING", f"Contradiction detection failed: {str(e)}",
                              correlation_id=correlation_id,
                              operation="integrate_into_sce")

            duration_ms = (time.time() - start_time) * 1000
            logger.log("INFO", "Formal commitment integrated into SCE",
                      correlation_id=correlation_id,
                      operation="integrate_into_sce",
                      proposition_id=commitment.proposition_id,
                      duration_ms=duration_ms)

            return True, None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Failed to integrate commitment into SCE: {str(e)}"
            logger.log("ERROR", error_msg,
                      correlation_id=correlation_id,
                      operation="integrate_into_sce",
                      proposition_id=commitment.proposition_id,
                      duration_ms=duration_ms)
            return False, error_msg

    def get_audit_trail(self) -> List[FormalCommitment]:
        """
        Get all formal commitments for audit

        Returns complete audit trail of DEE → SCE translations

        Returns:
            List of all FormalCommitment objects
        """
        return list(self.committed_propositions.values())

    def get_commitment(self, proposition_id: str) -> Optional[FormalCommitment]:
        """
        Get a specific commitment by ID

        Args:
            proposition_id: ID of commitment to retrieve

        Returns:
            FormalCommitment or None if not found
        """
        return self.committed_propositions.get(proposition_id)

    def clear_audit_trail(self) -> int:
        """
        Clear audit trail

        Useful for testing and isolation

        Returns:
            Number of commitments cleared
        """
        count = len(self.committed_propositions)
        self.committed_propositions.clear()
        logger.log("INFO", f"Audit trail cleared ({count} commitments)",
                  operation="clear_audit_trail")
        return count


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
    "FormalCommitment",
    "create_adapter",
    "is_available",
    "get_import_error",
]
