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

try:
    from rese_dee import DEELogger, CircuitBreaker, CircuitBreakerOpenError
except ImportError:
    from glue.lib.rese_dee import DEELogger, CircuitBreaker, CircuitBreakerOpenError


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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACIResult':
        """Create from dictionary."""
        return cls(**data)


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

            return config

        except (ValueError, TypeError) as e:
            print(f"FATAL: Invalid ACI configuration: {e}")
            sys.exit(1)


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
        logger: Optional[DEELogger] = None
    ):
        """
        Initialize ACI Calculator.

        Args:
            config: ACI configuration (defaults to env vars)
            logger: Structured logger
        """
        self.config = config or ACIConfig.from_env()
        self.logger = logger or DEELogger()

        # Circuit breaker for failure detection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv('PHASE3_ACI_CB_THRESHOLD', '5')),
            recovery_timeout_ms=int(os.getenv('PHASE3_ACI_CB_TIMEOUT_MS', '60000')),
            logger=self.logger
        )

        self.logger.info(
            "ACI Calculator initialized",
            config={
                'window_size': self.config.window_size,
                'entropy_threshold': self.config.entropy_threshold,
                'coherence_threshold': self.config.coherence_threshold,
                'timeout_ms': self.config.timeout_ms,
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
                    }
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
]
