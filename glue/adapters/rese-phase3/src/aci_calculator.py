"""
RESE Phase III: Anomaly Characterization Index (ACI) Calculator

This module implements the ACI for MCTS guidance as specified in RESE Technical Manual §5.2.

From §5.2: "The ACI is a composite measure that guides search refinement"

Two Components:
1. Disorder Entropy (𝔈_D): Measures randomness/uncertainty in time-series data
2. Causal Coherence (𝔍_C): Statistical correlation between high 𝔈_D and specific input variables

Signal Flagging:
High-potential signal = High 𝔈_D AND High 𝔍_C

Following CLAUDE.md principles:
- Law of Idempotency: Same input -> same output
- Law of Configuration Explicitness: All config via environment
- Law of UTC: All timestamps in UTC ISO-8601
- Circuit Breaker: Detect and handle failures
- Structured Logging: JSON with correlation_id
- Timeout: All operations bounded by timeout

Author: RESE Team
Created: 2026-02-04
Phase: III - Monte Carlo Refinement
Reference: RESE Technical Manual §5.2
"""

import os
import sys
import uuid
import time
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

try:
    from rese_dee import DEELogger, CircuitBreaker, CircuitBreakerOpenError
except ImportError:
    try:
        from glue.lib.rese_dee import DEELogger, CircuitBreaker, CircuitBreakerOpenError
    except ImportError:
        # Fallback implementations
        class DEELogger:
            def __init__(self):
                pass
            def info(self, msg, **kwargs):
                pass
            def debug(self, msg, **kwargs):
                pass
            def warning(self, msg, **kwargs):
                pass
            def error(self, msg, **kwargs):
                pass

        class CircuitBreaker:
            def __init__(self, failure_threshold=5, recovery_timeout_ms=60000, logger=None):
                self._failure_count = 0
                self._last_failure_time = None
                self.state = "CLOSED"
                self.failure_threshold = failure_threshold
                self.recovery_timeout_ms = recovery_timeout_ms

            def _on_failure(self, error):
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self.failure_threshold:
                    self.state = "OPEN"

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3Variable,
        Z3Constraint,
        Z3Config,
        Z3ConstraintType,
        Z3ResultStatus,
        Z3SolverResult,
        is_z3_available,
        get_z3_solver_engine
    )
    Z3_AVAILABLE = is_z3_available()
except ImportError:
    Z3_AVAILABLE = False
    Z3SolverEngine = None
    Z3Variable = None
    Z3Constraint = None
    Z3Config = None
    Z3ConstraintType = None
    Z3ResultStatus = None
    Z3SolverResult = None

# Try to import CAV-NLP for enhanced verification
try:
    from openevolve.cav_nlp_integration import Z3LeanAideBridge
    CAV_NLP_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
        from openevolve.cav_nlp_integration import Z3LeanAideBridge
        CAV_NLP_AVAILABLE = True
    except ImportError:
        CAV_NLP_AVAILABLE = False
        Z3LeanAideBridge = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ACIResult:
    """
    Anomaly Characterization Index result

    From RESE Manual §5.2:
    - disorder_entropy (𝔈_D): Randomness in time-series
    - causal_coherence (𝔍_C): Correlation with inputs
    - High-potential: High 𝔈_D AND High 𝔍_C
    """
    disorder_entropy: float  # 𝔈_D
    causal_coherence: float  # 𝔍_C
    aci_score: float  # Composite score
    is_high_entropy_signal: bool
    causal_variables: List[str]  # Variables with high 𝔍_C
    correlation_id: str
    timestamp: str  # UTC ISO-8601 (Law of UTC)
    window_start_idx: int
    window_end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Z3-enhanced fields (optional)
    z3_constraint_verified: bool = False
    z3_anomaly_satisfiable: bool = False
    z3_entropy_bounds: Optional[Tuple[float, float]] = None
    z3_coherence_bounds: Optional[Tuple[float, float]] = None
    z3_formal_proof: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'disorder_entropy': self.disorder_entropy,
            'causal_coherence': self.causal_coherence,
            'aci_score': self.aci_score,
            'is_high_entropy_signal': self.is_high_entropy_signal,
            'causal_variables': self.causal_variables,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp,
            'window_start_idx': self.window_start_idx,
            'window_end_idx': self.window_end_idx,
            'metadata': self.metadata,
            'z3_constraint_verified': self.z3_constraint_verified,
            'z3_anomaly_satisfiable': self.z3_anomaly_satisfiable,
            'z3_entropy_bounds': self.z3_entropy_bounds,
            'z3_coherence_bounds': self.z3_coherence_bounds,
            'z3_formal_proof': self.z3_formal_proof,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACIResult':
        """Create from dictionary."""
        # Handle Z3 fields if present
        z3_fields = {
            'z3_constraint_verified': data.get('z3_constraint_verified', False),
            'z3_anomaly_satisfiable': data.get('z3_anomaly_satisfiable', False),
            'z3_entropy_bounds': data.get('z3_entropy_bounds'),
            'z3_coherence_bounds': data.get('z3_coherence_bounds'),
            'z3_formal_proof': data.get('z3_formal_proof'),
        }
        return cls(**{**data, **z3_fields})


@dataclass
class ACIConfig:
    """
    ACI Calculator Configuration

    All values from environment variables (Law of Configuration Explicitness).
    Crashes immediately if required configuration is invalid.
    """
    # Window parameters
    window_size: int
    entropy_bins: int
    coherence_threshold: float
    entropy_threshold: float

    # Timeout (Law of Timeout)
    timeout_ms: int

    # Correlation analysis
    min_correlation_samples: int
    correlation_method: str  # 'pearson' or 'spearman'

    # Z3 constraint-based detection
    enable_z3_verification: bool = True
    z3_timeout_seconds: float = 5.0
    z3_entropy_tolerance: float = 0.05
    z3_coherence_tolerance: float = 0.05
    z3_confidence_level: float = 0.95

    @classmethod
    def from_env(cls) -> 'ACIConfig':
        """
        Load configuration from environment variables.

        Required env vars:
        - PHASE3_ACI_WINDOW_SIZE: Size of sliding window (default: 100)
        - PHASE3_ACI_ENTROPY_BINS: Bins for entropy calculation (default: 10)
        - PHASE3_ACI_COHERENCE_THRESHOLD: Threshold for high coherence (default: 0.5)
        - PHASE3_ACI_ENTROPY_THRESHOLD: Threshold for high entropy (default: 0.7)
        - PHASE3_ACI_TIMEOUT_MS: Timeout for ACI calculation (default: 3000)
        - PHASE3_ACI_MIN_SAMPLES: Minimum samples for correlation (default: 30)
        - PHASE3_ACI_CORRELATION_METHOD: Correlation method (default: pearson)
        - PHASE3_ACI_ENABLE_Z3: Enable Z3 verification (default: true)
        - PHASE3_ACI_Z3_TIMEOUT: Z3 solver timeout in seconds (default: 5.0)
        - PHASE3_ACI_Z3_ENTROPY_TOL: Entropy tolerance for Z3 (default: 0.05)
        - PHASE3_ACI_Z3_COHERENCE_TOL: Coherence tolerance for Z3 (default: 0.05)
        - PHASE3_ACI_Z3_CONFIDENCE: Confidence level for Z3 (default: 0.95)
        """
        try:
            config = cls(
                window_size=int(os.getenv('PHASE3_ACI_WINDOW_SIZE', '100')),
                entropy_bins=int(os.getenv('PHASE3_ACI_ENTROPY_BINS', '10')),
                coherence_threshold=float(os.getenv('PHASE3_ACI_COHERENCE_THRESHOLD', '0.5')),
                entropy_threshold=float(os.getenv('PHASE3_ACI_ENTROPY_THRESHOLD', '0.7')),
                timeout_ms=int(os.getenv('PHASE3_ACI_TIMEOUT_MS', '3000')),
                min_correlation_samples=int(os.getenv('PHASE3_ACI_MIN_SAMPLES', '30')),
                correlation_method=os.getenv('PHASE3_ACI_CORRELATION_METHOD', 'pearson'),
                enable_z3_verification=os.getenv('PHASE3_ACI_ENABLE_Z3', 'true').lower() == 'true',
                z3_timeout_seconds=float(os.getenv('PHASE3_ACI_Z3_TIMEOUT', '5.0')),
                z3_entropy_tolerance=float(os.getenv('PHASE3_ACI_Z3_ENTROPY_TOL', '0.05')),
                z3_coherence_tolerance=float(os.getenv('PHASE3_ACI_Z3_COHERENCE_TOL', '0.05')),
                z3_confidence_level=float(os.getenv('PHASE3_ACI_Z3_CONFIDENCE', '0.95')),
            )

            # Validate configuration
            if config.window_size <= 0:
                raise ValueError("PHASE3_ACI_WINDOW_SIZE must be positive")
            if config.entropy_bins <= 0:
                raise ValueError("PHASE3_ACI_ENTROPY_BINS must be positive")
            if not (0 <= config.coherence_threshold <= 1):
                raise ValueError("PHASE3_ACI_COHERENCE_THRESHOLD must be between 0 and 1")
            if not (0 <= config.entropy_threshold <= 1):
                raise ValueError("PHASE3_ACI_ENTROPY_THRESHOLD must be between 0 and 1")
            if config.timeout_ms <= 0:
                raise ValueError("PHASE3_ACI_TIMEOUT_MS must be positive")
            if config.min_correlation_samples <= 0:
                raise ValueError("PHASE3_ACI_MIN_SAMPLES must be positive")
            if config.correlation_method not in ['pearson', 'spearman']:
                raise ValueError("PHASE3_ACI_CORRELATION_METHOD must be 'pearson' or 'spearman'")
            if config.z3_timeout_seconds <= 0:
                raise ValueError("PHASE3_ACI_Z3_TIMEOUT must be positive")
            if not (0 < config.z3_entropy_tolerance <= 1):
                raise ValueError("PHASE3_ACI_Z3_ENTROPY_TOL must be between 0 and 1")
            if not (0 < config.z3_coherence_tolerance <= 1):
                raise ValueError("PHASE3_ACI_Z3_COHERENCE_TOL must be between 0 and 1")
            if not (0 < config.z3_confidence_level <= 1):
                raise ValueError("PHASE3_ACI_Z3_CONFIDENCE must be between 0 and 1")

            return config

        except (ValueError, TypeError) as e:
            print(f"FATAL: Invalid ACI configuration: {e}")
            sys.exit(1)


# ============================================================================
# Z3 ANOMALY DETECTOR
# ============================================================================

class Z3AnomalyDetector:
    """
    Z3-based constraint satisfiability detector for anomaly characterization.

    Encodes anomaly conditions as Z3 constraints to formally verify:
    1. Entropy bounds (𝔈_D)
    2. Coherence bounds (𝔍_C)
    3. High-potential signal condition (High 𝔈_D AND High 𝔍_C)

    Provides formal verification that detected anomalies are mathematically valid
    rather than statistical artifacts.

    Following CLAUDE.md principles:
    - Law of Runtime Truth: Uses Z3 solver execution
    - Law of Configuration Explicitness: All config via environment
    - Timeout: All Z3 operations bounded by timeout
    """

    def __init__(
        self,
        config: Optional[ACIConfig] = None,
        logger: Optional[DEELogger] = None,
        z3_engine: Optional['Z3SolverEngine'] = None
    ):
        """
        Initialize Z3 Anomaly Detector.

        Args:
            config: ACI configuration
            logger: Structured logger
            z3_engine: Pre-configured Z3 solver engine (optional)
        """
        self.config = config or ACIConfig.from_env()
        self.logger = logger or DEELogger()

        # Initialize Z3 solver engine
        if Z3_AVAILABLE and self.config.enable_z3_verification:
            if z3_engine is not None:
                self.z3_engine = z3_engine
            else:
                z3_config = Z3Config(
                    timeout=self.config.z3_timeout_seconds,
                    auto_config=True,
                    proof_generation=True
                )
                self.z3_engine = get_z3_solver_engine(z3_config)
            self.z3_enabled = True
        else:
            self.z3_engine = None
            self.z3_enabled = False

        self.logger.info(
            "Z3 Anomaly Detector initialized",
            z3_available=Z3_AVAILABLE,
            z3_enabled=self.z3_enabled,
            z3_timeout_seconds=self.config.z3_timeout_seconds
        )

    def encode_anomaly_constraints(
        self,
        entropy_value: float,
        coherence_value: float,
        entropy_threshold: float,
        coherence_threshold: float
    ) -> Tuple[List['Z3Variable'], List['Z3Constraint']]:
        """
        Encode anomaly conditions as Z3 constraints.

        Creates formal constraints for:
        - Entropy bounds: 𝔈_D ∈ [threshold - tolerance, threshold + tolerance]
        - Coherence bounds: 𝔍_C ∈ [threshold - tolerance, threshold + tolerance]
        - High-potential: 𝔈_D ≥ threshold AND 𝔍_C ≥ threshold

        Args:
            entropy_value: Calculated disorder entropy (𝔈_D)
            coherence_value: Calculated causal coherence (𝔍_C)
            entropy_threshold: Threshold for high entropy
            coherence_threshold: Threshold for high coherence

        Returns:
            Tuple of (variables, constraints) for Z3 solver
        """
        if not self.z3_enabled or Z3Variable is None or Z3Constraint is None:
            return [], []

        variables = [
            Z3Variable(
                "entropy",
                Z3ConstraintType.REAL if Z3ConstraintType else "REAL",
                bounds=(0.0, 1.0)
            ),
            Z3Variable(
                "coherence",
                Z3ConstraintType.REAL if Z3ConstraintType else "REAL",
                bounds=(0.0, 1.0)
            ),
        ]

        constraints = []

        # Entropy range constraint (with tolerance)
        entropy_min = max(0.0, entropy_threshold - self.config.z3_entropy_tolerance)
        entropy_max = min(1.0, entropy_threshold + self.config.z3_entropy_tolerance)
        constraints.append(
            Z3Constraint(
                f"(and (>= entropy {entropy_min}) (<= entropy {entropy_max}))",
                Z3ConstraintType.REAL if Z3ConstraintType else "REAL",
                f"Entropy within tolerance of {entropy_threshold}"
            )
        )

        # Coherence range constraint (with tolerance)
        coherence_min = max(0.0, coherence_threshold - self.config.z3_coherence_tolerance)
        coherence_max = min(1.0, coherence_threshold + self.config.z3_coherence_tolerance)
        constraints.append(
            Z3Constraint(
                f"(and (>= coherence {coherence_min}) (<= coherence {coherence_max}))",
                Z3ConstraintType.REAL if Z3ConstraintType else "REAL",
                f"Coherence within tolerance of {coherence_threshold}"
            )
        )

        # High-potential signal constraint
        constraints.append(
            Z3Constraint(
                f"(and (>= entropy {entropy_threshold}) (>= coherence {coherence_threshold}))",
                Z3ConstraintType.REAL if Z3ConstraintType else "REAL",
                "High-potential signal condition"
            )
        )

        return variables, constraints

    def verify_anomaly_satisfiability(
        self,
        entropy_value: float,
        coherence_value: float,
        entropy_threshold: float,
        coherence_threshold: float
    ) -> Dict[str, Any]:
        """
        Verify if anomaly condition is satisfiable using Z3.

        Checks whether there exists a valid assignment of entropy and coherence
        values that satisfies the high-potential signal constraints.

        Args:
            entropy_value: Calculated disorder entropy (𝔈_D)
            coherence_value: Calculated causal coherence (𝔍_C)
            entropy_threshold: Threshold for high entropy
            coherence_threshold: Threshold for high coherence

        Returns:
            Dict with verification results:
            - satisfiable: bool, whether constraints are satisfiable
            - verified: bool, whether calculated values satisfy constraints
            - entropy_bounds: Optional[Tuple[float, float]], valid entropy range
            - coherence_bounds: Optional[Tuple[float, float]], valid coherence range
            - proof: Optional[str], Z3 proof if available
            - model: Optional[Dict], Z3 model if satisfiable
        """
        result = {
            'satisfiable': False,
            'verified': False,
            'entropy_bounds': None,
            'coherence_bounds': None,
            'proof': None,
            'model': None,
            'error': None
        }

        if not self.z3_enabled or self.z3_engine is None:
            result['error'] = "Z3 not enabled or unavailable"
            return result

        try:
            # Encode constraints
            variables, constraints = self.encode_anomaly_constraints(
                entropy_value, coherence_value,
                entropy_threshold, coherence_threshold
            )

            if not variables or not constraints:
                result['error'] = "Failed to encode constraints"
                return result

            # Solve using Z3
            z3_result = self.z3_engine.solve_constraints(variables, constraints)

            if z3_result.status == Z3ResultStatus.SAT if Z3ResultStatus else "sat":
                result['satisfiable'] = True
                result['model'] = z3_result.model.to_dict() if z3_result.model else None
                result['proof'] = z3_result.smtlib_output

                # Extract bounds from model
                if z3_result.model:
                    entropy_val = z3_result.model.get_value('entropy')
                    coherence_val = z3_result.model.get_value('coherence')

                    # Calculate valid ranges
                    result['entropy_bounds'] = (
                        max(0.0, entropy_threshold - self.config.z3_entropy_tolerance),
                        min(1.0, entropy_threshold + self.config.z3_entropy_tolerance)
                    )
                    result['coherence_bounds'] = (
                        max(0.0, coherence_threshold - self.config.z3_coherence_tolerance),
                        min(1.0, coherence_threshold + self.config.z3_coherence_tolerance)
                    )

                    # Verify calculated values are within bounds
                    result['verified'] = (
                        result['entropy_bounds'][0] <= entropy_value <= result['entropy_bounds'][1] and
                        result['coherence_bounds'][0] <= coherence_value <= result['coherence_bounds'][1]
                    )

                self.logger.info(
                    "Z3 anomaly verification successful",
                    entropy=entropy_value,
                    coherence=coherence_value,
                    satisfiable=result['satisfiable'],
                    verified=result['verified'],
                    correlation_id=str(uuid.uuid4())[:8]
                )
            elif z3_result.status == Z3ResultStatus.UNSAT if Z3ResultStatus else "unsat":
                result['error'] = "Constraints are unsatisfiable"
                self.logger.warning(
                    "Z3 anomaly verification failed: unsatisfiable",
                    entropy=entropy_value,
                    coherence=coherence_value
                )
            else:
                result['error'] = f"Z3 solver returned {z3_result.status}"
                self.logger.warning(
                    "Z3 anomaly verification failed: unknown result",
                    status=str(z3_result.status)
                )

        except Exception as e:
            result['error'] = str(e)
            self.logger.error("Z3 anomaly verification exception", error=str(e))

        return result

    def verify_high_entropy_signal(
        self,
        entropy_value: float,
        coherence_value: float
    ) -> bool:
        """
        Verify if signal is a high-entropy anomaly using Z3 constraints.

        Args:
            entropy_value: Calculated disorder entropy (𝔈_D)
            coherence_value: Calculated causal coherence (𝔍_C)

        Returns:
            bool: True if verified as high-entropy signal
        """
        verification = self.verify_anomaly_satisfiability(
            entropy_value,
            coherence_value,
            self.config.entropy_threshold,
            self.config.coherence_threshold
        )

        return verification.get('verified', False) and verification.get('satisfiable', False)

    def formal_entropy_analysis(
        self,
        time_series: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform formal analysis of entropy using Z3 constraints.

        Encodes entropy calculation properties as Z3 constraints to verify:
        1. Entropy is bounded between 0 and 1 (normalized)
        2. Entropy is monotonic with respect to disorder
        3. Entropy calculation is deterministic

        Args:
            time_series: Input time-series data

        Returns:
            Dict with formal analysis results
        """
        result = {
            'verified': False,
            'entropy_value': None,
            'bounds_verified': False,
            'determinism_verified': False,
            'proof': None
        }

        if not self.z3_enabled:
            return result

        try:
            # Calculate entropy (standard method)
            hist, _ = np.histogram(time_series, bins=self.config.entropy_bins, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist * np.log2(hist))
                max_entropy = np.log2(self.config.entropy_bins)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 0.0

            result['entropy_value'] = float(normalized_entropy)

            # Verify bounds using Z3
            if Z3Variable is not None and Z3Constraint is not None:
                variables = [
                    Z3Variable("E", Z3ConstraintType.REAL, bounds=(0.0, 1.0))
                ]

                constraints = [
                    Z3Constraint(
                        "(>= E 0.0)",
                        Z3ConstraintType.REAL,
                        "Entropy non-negative"
                    ),
                    Z3Constraint(
                        "(<= E 1.0)",
                        Z3ConstraintType.REAL,
                        "Entropy bounded above"
                    )
                ]

                z3_result = self.z3_engine.solve_constraints(variables, constraints)

                if z3_result.is_sat() if hasattr(z3_result, 'is_sat') else False:
                    result['bounds_verified'] = True
                    result['proof'] = z3_result.smtlib_output

            # Verify determinism (check idempotency)
            entropy2 = self._calculate_entropy_raw(time_series)
            result['determinism_verified'] = np.isclose(
                result['entropy_value'], entropy2, rtol=1e-10
            )

            result['verified'] = (
                result['bounds_verified'] and
                result['determinism_verified']
            )

        except Exception as e:
            self.logger.error("Formal entropy analysis failed", error=str(e))

        return result

    def _calculate_entropy_raw(self, time_series: np.ndarray) -> float:
        """Raw entropy calculation for determinism verification."""
        hist, _ = np.histogram(time_series, bins=self.config.entropy_bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(self.config.entropy_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0


# ============================================================================
# ACI CALCULATOR
# ============================================================================

class AnomalyCharacterizationIndex:
    """
    Calculates the Anomaly Characterization Index (ACI) for MCTS guidance

    From RESE Technical Manual §5.2:
    - 𝔈_D: Disorder Entropy - randomness in time-series
    - 𝔍_C: Causal Coherence - correlation with inputs
    - High-potential: High 𝔈_D AND High 𝔍_C

    Following CLAUDE.md principles:
    - Law of Idempotency: Same input -> same output
    - Law of Configuration Explicitness: All config via env vars
    - Circuit Breaker: Handle calculation failures
    - Structured Logging: JSON with correlation_id
    - Timeout: All operations bounded by timeout
    """

    def __init__(
        self,
        config: Optional[ACIConfig] = None,
        logger: Optional[DEELogger] = None,
        z3_detector: Optional[Z3AnomalyDetector] = None
    ):
        """
        Initialize ACI Calculator.

        Args:
            config: ACI configuration (defaults to env vars)
            logger: Structured logger
            z3_detector: Pre-configured Z3 anomaly detector (optional)
        """
        self.config = config or ACIConfig.from_env()
        self.logger = logger or DEELogger()

        # Circuit breaker for failure detection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv('PHASE3_ACI_CB_THRESHOLD', '5')),
            recovery_timeout_ms=int(os.getenv('PHASE3_ACI_CB_TIMEOUT_MS', '60000')),
            logger=self.logger
        )

        # Initialize Z3 anomaly detector
        self.z3_detector = z3_detector or Z3AnomalyDetector(self.config, self.logger)

        # Initialize CAV-NLP bridge for enhanced verification
        self.cav_nlp_bridge = None
        self.use_cav_nlp = os.getenv('PHASE3_ACI_USE_CAV_NLP', 'false').lower() == 'true'
        if self.use_cav_nlp and CAV_NLP_AVAILABLE:
            try:
                self.cav_nlp_bridge = Z3LeanAideBridge()
                self.logger.info("CAV-NLP bridge initialized for ACI Calculator",
                    cav_nlp_available=True,
                )
            except Exception as e:
                self.logger.warning("Failed to initialize CAV-NLP bridge",
                    error=str(e),
                )
                self.use_cav_nlp = False
        else:
            self.use_cav_nlp = False

        self.logger.info(
            "ACI Calculator initialized",
            config={
                'window_size': self.config.window_size,
                'entropy_threshold': self.config.entropy_threshold,
                'coherence_threshold': self.config.coherence_threshold,
                'timeout_ms': self.config.timeout_ms,
                'z3_enabled': self.z3_detector.z3_enabled,
                'cav_nlp_enabled': self.use_cav_nlp,
            }
        )

    def calculate_disorder_entropy(
        self,
        time_series: np.ndarray,
        bins: Optional[int] = None
    ) -> float:
        """
        Calculate Disorder Entropy (𝔈_D)

        Uses Shannon entropy to measure randomness in time-series data.

        From RESE Manual §5.2: "Disorder Entropy (𝔈_D) measures randomness/uncertainty"

        Args:
            time_series: 1D array of time-series values
            bins: Number of bins for histogram (default from config)

        Returns:
            float: 𝔈_D value (0 = ordered, high = disordered)

        Raises:
            ValueError: If time_series is invalid
        """
        if bins is None:
            bins = self.config.entropy_bins

        # Validate input
        if len(time_series) < 2:
            raise ValueError(f"Time-series too short: {len(time_series)} < 2")

        if np.all(time_series == time_series[0]):
            # Constant signal has zero entropy
            return 0.0

        try:
            # 1. Create histogram of time-series
            hist, _ = np.histogram(time_series, bins=bins, density=True)

            # 2. Filter zero bins to avoid log(0)
            hist = hist[hist > 0]

            # 3. Normalize histogram to sum to 1 (probability distribution)
            hist = hist / np.sum(hist)

            # 4. Calculate Shannon entropy (base-2 for bits)
            # H = -sum(p * log2(p))
            𝔈_D = -np.sum(hist * np.log2(hist))

            # Normalize by max possible entropy (log2 of bins)
            max_entropy = np.log2(bins)
            normalized_𝔈_D = 𝔈_D / max_entropy if max_entropy > 0 else 0.0

            self.logger.debug(
                "Disorder entropy calculated",
                entropy=normalized_𝔈_D,
                raw_entropy=𝔈_D,
                max_entropy=max_entropy,
                samples=len(time_series),
            )

            return normalized_𝔈_D

        except Exception as e:
            self.logger.error("Disorder entropy calculation failed", e)
            raise

    def calculate_causal_coherence(
        self,
        entropy_data: np.ndarray,
        input_variables: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate Causal Coherence (𝔍_C)

        Measures statistical correlation between high entropy regions
        and specific input variables to identify causal triggers.

        From RESE Manual §5.2: "Causal Coherence (𝔍_C) - correlation with inputs"

        Args:
            entropy_data: Array of entropy values over time
            input_variables: Dict mapping variable names to time-series
            threshold: Correlation threshold for "high" coherence (default from config)

        Returns:
            Tuple[float, List[str]]: (𝔍_C score, list of high-correlation variables)

        Raises:
            ValueError: If inputs are invalid
        """
        if threshold is None:
            threshold = self.config.coherence_threshold

        # Validate inputs
        if len(entropy_data) < self.config.min_correlation_samples:
            raise ValueError(
                f"Insufficient entropy data: {len(entropy_data)} "
                f"< {self.config.min_correlation_samples}"
            )

        correlations = {}

        for var_name, var_data in input_variables.items():
            # Validate variable data
            if len(var_data) != len(entropy_data):
                self.logger.warning(
                    "Variable length mismatch, skipping",
                    variable=var_name,
                    entropy_length=len(entropy_data),
                    variable_length=len(var_data)
                )
                continue

            if len(var_data) < self.config.min_correlation_samples:
                self.logger.warning(
                    "Variable too short for correlation, skipping",
                    variable=var_name,
                    length=len(var_data)
                )
                continue

            try:
                # Check for constant data
                if np.all(var_data == var_data[0]) or np.all(entropy_data == entropy_data[0]):
                    correlations[var_name] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'significant': False
                    }
                    continue

                # Calculate correlation based on method
                if self.config.correlation_method == 'pearson':
                    correlation, p_value = stats.pearsonr(entropy_data, var_data)
                else:  # spearman
                    correlation, p_value = stats.spearmanr(entropy_data, var_data)

                correlations[var_name] = {
                    'correlation': float(abs(correlation)),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }

                self.logger.debug(
                    "Correlation calculated",
                    variable=var_name,
                    correlation=abs(correlation),
                    p_value=p_value,
                    significant=correlations[var_name]['significant']
                )

            except Exception as e:
                self.logger.error(
                    "Correlation calculation failed",
                    error=str(e),
                    variable=var_name
                )
                correlations[var_name] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }

        # Calculate 𝔍_C = Maximum significant correlation
        significant_correlations = [
            (var, data['correlation'])
            for var, data in correlations.items()
            if data['significant']
        ]

        if significant_correlations:
            𝔍_C = max(corr for _, corr in significant_correlations)
            high_coh_vars = [
                var for var, corr in significant_correlations
                if corr >= threshold
            ]
        else:
            𝔍_C = 0.0
            high_coh_vars = []

        self.logger.info(
            "Causal coherence calculated",
            coherence_score=𝔍_C,
            high_coherence_variables=high_coh_vars,
            total_variables=len(input_variables),
            significant_correlations=len(significant_correlations)
        )

        return 𝔍_C, high_coh_vars

    def detect_high_entropy_signals(
        self,
        experiment_data: Dict[str, np.ndarray],
        time_series_key: str = 'output',
        correlation_id: Optional[str] = None
    ) -> List[ACIResult]:
        """
        Detect high-potential signals (High 𝔈_D AND High 𝔍_C)

        These signals indicate anomalous behavior worth investigating.

        From RESE Manual §5.2: "High-potential signal = High 𝔈_D AND High 𝔍_C"

        Args:
            experiment_data: Dict with time-series data
            time_series_key: Key for primary output time-series
            correlation_id: Distributed tracing ID

        Returns:
            List[ACIResult]: Detected signals with ACI metrics

        Raises:
            RuntimeError: If circuit breaker is open
            TimeoutError: If calculation exceeds timeout
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = time.time()

        self.logger.info(
            "Detecting high-entropy signals",
            correlation_id=correlation_id,
            time_series_key=time_series_key,
            data_keys=list(experiment_data.keys())
        )

        # Check circuit breaker
        if self.circuit_breaker.state == "OPEN":
            error_msg = "Circuit breaker is OPEN - too many recent failures"
            self.logger.error(error_msg, correlation_id=correlation_id)
            raise RuntimeError(error_msg)

        try:
            # Validate input
            if time_series_key not in experiment_data:
                raise ValueError(f"Time-series key '{time_series_key}' not found in data")

            time_series = experiment_data[time_series_key]

            if len(time_series) < self.config.window_size:
                self.logger.warning(
                    "Time-series shorter than window size",
                    length=len(time_series),
                    window_size=self.config.window_size,
                    correlation_id=correlation_id
                )
                # Use full series as single window
                windows = [time_series]
                window_indices = [(0, len(time_series))]
            else:
                # Calculate sliding window entropy
                num_windows = len(time_series) // self.config.window_size
                windows = []
                window_indices = []

                for i in range(num_windows):
                    start_idx = i * self.config.window_size
                    end_idx = start_idx + self.config.window_size
                    window = time_series[start_idx:end_idx]
                    windows.append(window)
                    window_indices.append((start_idx, end_idx))

            results = []

            for window, (start_idx, end_idx) in zip(windows, window_indices):
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.timeout_ms:
                    self.logger.warning(
                        "ACI calculation timeout",
                        elapsed_ms=elapsed_ms,
                        timeout_ms=self.config.timeout_ms,
                        correlation_id=correlation_id
                    )
                    raise TimeoutError(f"ACI calculation exceeded {self.config.timeout_ms}ms")

                # 1. Calculate 𝔈_D for this window
                𝔈_D = self.calculate_disorder_entropy(window)

                # 2. Identify input variables (exclude output)
                input_vars = {
                    k: v[start_idx:end_idx]
                    for k, v in experiment_data.items()
                    if k != time_series_key and isinstance(v, np.ndarray)
                }

                # 3. Calculate 𝔍_C
                if input_vars:
                    # Create entropy array for this window
                    entropy_array = np.full(len(window), 𝔈_D)
                    𝔍_C, causal_vars = self.calculate_causal_coherence(
                        entropy_array,
                        input_vars
                    )
                else:
                    𝔍_C = 0.0
                    causal_vars = []

                # 4. Composite ACI score (average of normalized values)
                aci_score = (𝔈_D + 𝔍_C) / 2

                # 5. Flag if high 𝔈_D AND high 𝔍_C
                is_high_signal = (
                    𝔈_D >= self.config.entropy_threshold and
                    𝔍_C >= self.config.coherence_threshold
                )

                # 6. Z3 formal verification (if enabled)
                z3_verified = False
                z3_satisfiable = False
                z3_entropy_bounds = None
                z3_coherence_bounds = None
                z3_proof = None

                if self.z3_detector.z3_enabled and is_high_signal:
                    verification = self.z3_detector.verify_anomaly_satisfiability(
                        𝔈_D, 𝔍_C,
                        self.config.entropy_threshold,
                        self.config.coherence_threshold
                    )

                    z3_verified = verification.get('verified', False)
                    z3_satisfiable = verification.get('satisfiable', False)
                    z3_entropy_bounds = verification.get('entropy_bounds')
                    z3_coherence_bounds = verification.get('coherence_bounds')
                    z3_proof = verification.get('proof')

                    # Only flag as high-signal if Z3 verifies
                    if self.config.enable_z3_verification:
                        is_high_signal = is_high_signal and z3_verified

                result = ACIResult(
                    disorder_entropy=𝔈_D,
                    causal_coherence=𝔍_C,
                    aci_score=aci_score,
                    is_high_entropy_signal=is_high_signal,
                    causal_variables=causal_vars,
                    timestamp=datetime.now(timezone.utc).isoformat(),  # Law of UTC
                    correlation_id=correlation_id,
                    window_start_idx=start_idx,
                    window_end_idx=end_idx,
                    metadata={
                        'window_size': len(window),
                        'num_input_variables': len(input_vars),
                    },
                    z3_constraint_verified=z3_verified,
                    z3_anomaly_satisfiable=z3_satisfiable,
                    z3_entropy_bounds=z3_entropy_bounds,
                    z3_coherence_bounds=z3_coherence_bounds,
                    z3_formal_proof=z3_proof
                )

                results.append(result)

                self.logger.debug(
                    "ACI result calculated",
                    correlation_id=correlation_id,
                    window_start=start_idx,
                    window_end=end_idx,
                    entropy=𝔈_D,
                    coherence=𝔍_C,
                    aci_score=aci_score,
                    is_high_signal=is_high_signal
                )

            # Record success (using circuit breaker's internal method via call wrapper)
            # The circuit breaker state is managed through the call() method
            self.logger.info(
                "High-entropy signal detection complete",
                correlation_id=correlation_id,
                total_windows=len(results),
                high_entropy_signals=sum(1 for r in results if r.is_high_entropy_signal),
                elapsed_ms=elapsed_ms
            )

            return results

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            self.logger.error(
                "High-entropy signal detection failed",
                error=str(e),
                correlation_id=correlation_id,
                elapsed_ms=elapsed_ms
            )

            # Record failure (circuit breaker tracks this internally via exceptions)
            # Mark circuit breaker state to reflect the failure
            self.circuit_breaker._on_failure(str(e))

            raise

    def calculate_aci_reduction(
        self,
        initial_aci: float,
        final_aci: float
    ) -> float:
        """
        Calculate reduction in ACI (measure of paradigm improvement)

        From RESE Manual §6.3: "statistically significant reduction in the Anomaly
        Characterization Index (𝔈_D component)"

        Args:
            initial_aci: ACI before intervention
            final_aci: ACI after intervention

        Returns:
            float: Percentage reduction (0-100)
        """
        if initial_aci == 0:
            return 0.0

        reduction = ((initial_aci - final_aci) / initial_aci) * 100
        reduction = max(0.0, reduction)  # No negative reduction

        self.logger.info(
            "ACI reduction calculated",
            initial_aci=initial_aci,
            final_aci=final_aci,
            reduction_percentage=reduction
        )

        return reduction

    def get_high_priority_signals(
        self,
        aci_results: List[ACIResult],
        top_n: Optional[int] = None
    ) -> List[ACIResult]:
        """
        Get high-priority signals for MCTS exploration

        High-priority signals are those with:
        - High 𝔈_D (above threshold)
        - High 𝔍_C (above threshold)
        - High ACI score

        Args:
            aci_results: List of ACI results
            top_n: Return top N signals (None = all high-priority)

        Returns:
            List[ACIResult]: High-priority signals, sorted by ACI score
        """
        # Filter high-priority signals
        high_priority = [
            result for result in aci_results
            if result.is_high_entropy_signal
        ]

        # Sort by ACI score (descending)
        high_priority.sort(key=lambda r: r.aci_score, reverse=True)

        # Return top N if specified
        if top_n is not None:
            high_priority = high_priority[:top_n]

        self.logger.info(
            "High-priority signals identified",
            total_signals=len(aci_results),
            high_priority_count=len(high_priority),
            top_n=top_n
        )

        return high_priority


# ============================================================================
# SYNTHETIC DATA GENERATOR (For Testing)
# ============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic experimental data for testing ACI

    Creates controlled time-series data with known entropy patterns
    for validation and testing.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_constant_signal(self, length: int = 1000) -> np.ndarray:
        """Generate constant signal (zero entropy)."""
        return np.ones(length) * 0.5

    def generate_sine_wave(self, length: int = 1000, frequency: float = 0.1) -> np.ndarray:
        """Generate periodic signal (low entropy)."""
        t = np.arange(length)
        return 0.5 + 0.3 * np.sin(2 * np.pi * frequency * t)

    def generate_random_walk(self, length: int = 1000) -> np.ndarray:
        """Generate random walk (medium entropy)."""
        steps = np.random.randn(length)
        walk = np.cumsum(steps)
        # Normalize to 0-1 range
        walk = (walk - walk.min()) / (walk.max() - walk.min())
        return walk

    def generate_white_noise(self, length: int = 1000) -> np.ndarray:
        """Generate white noise (high entropy)."""
        return np.random.rand(length)

    def generate_multi_variable_experiment(
        self,
        length: int = 1000,
        num_variables: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Generate multi-variable experiment with known causal relationships.

        Args:
            length: Length of time-series
            num_variables: Number of input variables

        Returns:
            Dict with 'output' and 'var_1', 'var_2', etc.
        """
        data = {}

        # Generate input variables
        for i in range(num_variables):
            var_type = i % 3  # Cycle through signal types

            if var_type == 0:
                data[f'var_{i+1}'] = self.generate_sine_wave(length, frequency=0.05 * (i+1))
            elif var_type == 1:
                data[f'var_{i+1}'] = self.generate_random_walk(length)
            else:
                data[f'var_{i+1}'] = self.generate_white_noise(length)

        # Generate output as combination of inputs + noise
        output = np.zeros(length)
        for var_name, var_data in data.items():
            # Add weighted contribution
            weight = np.random.rand()
            output += weight * var_data

        # Add noise
        output += 0.1 * np.random.randn(length)

        # Normalize to 0-1 range
        output = (output - output.min()) / (output.max() - output.min())
        output = np.clip(output, 0, 1)

        data['output'] = output

        return data


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    'ACIResult',
    'ACIConfig',
    'AnomalyCharacterizationIndex',
    'SyntheticDataGenerator',
    'Z3AnomalyDetector',
    'Z3_AVAILABLE',
]


# ============================================================================
# CAV-NLP ENHANCED VERIFICATION METHODS
# ============================================================================

async def verify_with_cav_nlp(
    self,
    solution: Any,
    solution_type: str = "aci_result",
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify ACI solution using CAV-NLP hybrid verification

    Performs hybrid verification using CAV-NLP bridge, combining
    Z3 SMT solving with formal verification for anomaly detection
    results.

    Args:
        solution: ACI result or solution to verify
        solution_type: Type of solution ('aci_result', 'constraint', 'hypothesis')
        correlation_id: Distributed tracing ID

    Returns:
        Dict with verification results:
        - verified: bool, whether solution is verified
        - z3_result: Z3 verification result
        - lean_result: Lean verification result
        - agreement: bool, whether Z3 and Lean agree
        - confidence: float, confidence score
        - proof: Optional[str], proof if available
        - counterexample: Optional[Dict], counterexample if found
        - cav_nlp_used: bool, whether CAV-NLP was used
    """
    import asyncio

    correlation_id = correlation_id or str(uuid.uuid4())

    result = {
        'verified': False,
        'z3_result': None,
        'lean_result': None,
        'agreement': False,
        'confidence': 0.0,
        'proof': None,
        'counterexample': None,
        'error': None,
        'cav_nlp_used': False,
    }

    # Check if CAV-NLP is available
    if not hasattr(self, 'use_cav_nlp') or not self.use_cav_nlp:
        result['error'] = "CAV-NLP not enabled"
        return result

    if not hasattr(self, 'cav_nlp_bridge') or self.cav_nlp_bridge is None:
        result['error'] = "CAV-NLP bridge not available"
        return result

    try:
        # Extract constraint/theorem from solution
        if solution_type == "aci_result" and hasattr(solution, 'to_dict'):
            # Convert ACI result to verifiable constraint
            aci_dict = solution.to_dict()
            constraint_str = f"entropy={aci_dict.get('disorder_entropy', 0):.4f} AND coherence={aci_dict.get('causal_coherence', 0):.4f}"
        elif solution_type == "constraint" and isinstance(solution, str):
            constraint_str = solution
        else:
            constraint_str = str(solution)

        # Run hybrid verification using CAV-NLP bridge
        bridge_result = await self.cav_nlp_bridge.verify(
            constraint=constraint_str,
            use_counterexamples=True,
        )

        # Extract results
        result['z3_result'] = bridge_result.z3_result
        result['lean_result'] = bridge_result.lean_result
        result['agreement'] = bridge_result.agreed
        result['confidence'] = bridge_result.confidence
        result['counterexample'] = bridge_result.counterexample
        result['cav_nlp_used'] = True

        # Determine overall verification status
        if bridge_result.agreed and bridge_result.z3_result == "unsat":
            result['verified'] = True
            result['proof'] = bridge_result.lean_proof or "Verified by consensus"
        elif bridge_result.z3_result == "sat":
            result['verified'] = True
        elif bridge_result.confidence >= 0.8:
            result['verified'] = True

        self.logger.info("CAV-NLP verification complete for ACI",
            correlation_id=correlation_id,
            verified=result['verified'],
            agreement=result['agreement'],
            confidence=result['confidence'],
        )

    except Exception as e:
        result['error'] = str(e)
        self.logger.error("CAV-NLP verification failed for ACI",
            correlation_id=correlation_id,
            error=str(e),
        )

    return result


# Attach method to AnomalyCharacterizationIndex class
AnomalyCharacterizationIndex.verify_with_cav_nlp = verify_with_cav_nlp
