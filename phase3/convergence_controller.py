"""
Convergence Controller for RESE Phase III (Monte Carlo Refinement)

Implements adaptive convergence control for Monte Carlo Tree Search with:
- Multiple convergence detectors (ACI, variance, gradient, solution stability)
- Dynamic N_max estimation and adjustment
- Adaptive stopping strategies
- Integration with Γ₁ (ACI calculation) and Stage 9 (E2E validation)

Author: Agent D3 (N_max Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
Dependencies:
    - rese.gamma1.core.aci_calculator (Γ₁ - ACI calculation)
    - rese.phase3.mcts_search (MCTS search)
    - rese.phase3.statistical_validator (Statistical validation)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import math
import time

# Try to import Γ₁ ACI calculator
try:
    from gamma1.core.aci_calculator import ACICalculator, ACIResult
    from gamma1.core.csp_models import CSPInstance
except ImportError:
    ACICalculator = None
    ACIResult = None
    CSPInstance = None

# Try to import MCTS module
try:
    from phase3.mcts_search import MCTSNode, MCTSState, MCTSSearch
except ImportError:
    MCTSNode = None
    MCTSState = None
    MCTSSearch = None

# Try to import statistical validator
try:
    from phase3.statistical_validator import StatisticalValidator, ConvergenceResult
except ImportError:
    StatisticalValidator = None
    ConvergenceResult = None


class StoppingStrategy(Enum):
    """Strategies for combining stopping criteria"""
    ANY = "any"  # Stop if any criterion met (fast)
    ALL = "all"  # Stop if all criteria met (thorough)
    MAJORITY = "majority"  # Stop if majority met (balanced)
    WEIGHTED = "weighted"  # Weighted combination


@dataclass
class ConvergenceConfig:
    """
    Configuration for convergence control

    Attributes:
        # Detection methods
        use_aci_stability: Enable ACI stability detector
        use_solution_stability: Enable solution stability detector
        use_variance: Enable variance-based detector
        use_gradient: Enable gradient-based detector
        use_gelman_rubin: Enable Gelman-Rubin detector (expensive, requires multiple chains)

        # Thresholds
        variance_threshold: Convergence threshold for variance detector
        gradient_threshold: Convergence threshold for gradient detector
        aci_variance_threshold: Convergence threshold for ACI stability
        r_hat_threshold: Convergence threshold for Gelman-Rubin (typically 1.1 or 1.05)

        # Window sizes
        convergence_window: Window size for moving window detectors
        stability_window: Window size for stability detectors
        aci_window: Window size for ACI trajectory

        # N_max estimation
        base_n_max: Base number of iterations
        min_n_max: Minimum allowed iterations
        max_n_max: Maximum allowed iterations
        aci_weight_n_max: Weight for ACI in N_max estimation (vs structural)
        use_dynamic_adjustment: Enable dynamic N_max adjustment

        # Early stopping
        enable_early_stopping: Enable early stopping for low ACI or no progress
        low_aci_threshold: ACI threshold for early stopping
        no_improvement_iterations: Iterations without improvement before early stop
        diminishing_returns_threshold: Improvement rate threshold for diminishing returns

        # Stopping strategy
        stopping_strategy: How to combine multiple stopping criteria
        min_iterations_before_stop: Minimum iterations before allowing stop
        detector_weights: Custom weights for each detector (for WEIGHTED strategy)

        # Integration
        aci_computation_interval: Compute ACI every N iterations
        check_interval: Check convergence every N iterations
        adjust_interval: Adjust N_max every N iterations
        report_to_stage9: Enable reporting to Stage 9

        # Logging
        verbose: Enable verbose logging
    """
    # Detection methods
    use_aci_stability: bool = True
    use_solution_stability: bool = True
    use_variance: bool = True
    use_gradient: bool = True
    use_gelman_rubin: bool = False  # Expensive, requires multiple chains

    # Thresholds
    variance_threshold: float = 0.001
    gradient_threshold: float = 0.001
    aci_variance_threshold: float = 0.01
    r_hat_threshold: float = 1.1

    # Window sizes
    convergence_window: int = 20
    stability_window: int = 50
    aci_window: int = 30

    # N_max estimation
    base_n_max: int = 1000
    min_n_max: int = 100
    max_n_max: int = 10000
    aci_weight_n_max: float = 0.7  # Weight for ACI vs structure
    use_dynamic_adjustment: bool = True

    # Early stopping
    enable_early_stopping: bool = True
    low_aci_threshold: float = 0.3
    no_improvement_iterations: int = 100
    diminishing_returns_threshold: float = 0.001

    # Stopping strategy
    stopping_strategy: str = 'MAJORITY'  # ANY, ALL, MAJORITY, WEIGHTED
    min_iterations_before_stop: int = 50
    detector_weights: Dict[str, float] = None  # Will be initialized in __post_init__

    # Integration
    aci_computation_interval: int = 10  # Compute ACI every N iterations
    check_interval: int = 20  # Check convergence every N iterations
    adjust_interval: int = 50  # Adjust N_max every N iterations
    report_to_stage9: bool = True

    # Logging
    verbose: bool = False

    def __post_init__(self):
        """Initialize detector_weights if not provided"""
        if self.detector_weights is None:
            # Default equal weights for all detectors
            self.detector_weights = {
                'ACIStabilityDetector': 1.0,
                'SolutionStabilityDetector': 1.0,
                'VarianceDetector': 1.0,
                'GradientDetector': 1.0,
                'GelmanRubinDetector': 1.0
            }


@dataclass
class SearchState:
    """
    Current state of MCTS search for convergence monitoring.

    Attributes:
        iteration: Current iteration number
        value_history: History of best values over iterations
        current_value: Current best value
        aci_history: History of ACI scores
        current_aci: Current ACI score
        best_solution: Best solution found so far
        last_improvement_iteration: Iteration when last improvement occurred
        start_time: Search start time
        elapsed_time: Elapsed time since start
        n_max: Current maximum iterations
    """
    iteration: int = 0
    value_history: List[float] = field(default_factory=list)
    current_value: float = 0.0
    aci_history: List[float] = field(default_factory=list)
    current_aci: float = 0.5
    best_solution: Any = None
    last_improvement_iteration: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    n_max: int = 1000

    def update(self, iteration: int, value: float, aci: float = None):
        """Update search state with new iteration"""
        self.iteration = iteration

        # Check if this is an improvement BEFORE appending
        is_improvement = not self.value_history or value > max(self.value_history)

        self.value_history.append(value)
        self.current_value = value

        if aci is not None:
            self.aci_history.append(aci)
            self.current_aci = aci

        # Update last improvement
        if is_improvement:
            self.last_improvement_iteration = iteration
            self.best_solution = value  # In practice, would store actual solution

        # Update elapsed time
        if self.start_time > 0:
            self.elapsed_time = time.time() - self.start_time


class ConvergenceDetector(ABC):
    """Base class for convergence detectors"""

    def __init__(self, config: ConvergenceConfig):
        self.config = config

    @abstractmethod
    def detect(self, search_state: SearchState) -> ConvergenceResult:
        """
        Detect convergence.

        Args:
            search_state: Current search state

        Returns:
            ConvergenceResult with convergence status and details
        """
        pass


class ACIStabilityDetector(ConvergenceDetector):
    """
    Detect convergence based on ACI stabilization.

    Converged if: Variance of ACI in window < threshold
    """

    def detect(self, search_state: SearchState) -> ConvergenceResult:
        if len(search_state.aci_history) < self.config.aci_window:
            return ConvergenceResult(
                converged=False,
                method=type(self),
                iteration=search_state.iteration,
                confidence=0.0,
                details={'reason': 'Insufficient ACI history'}
            )

        # Get recent ACI values
        recent_aci = search_state.aci_history[-self.config.aci_window:]

        # Calculate variance
        aci_variance = np.var(recent_aci)

        # Check convergence
        converged = aci_variance < self.config.aci_variance_threshold

        # Confidence based on how far below threshold
        if converged:
            confidence = min(1.0, self.config.aci_variance_threshold / (aci_variance + 1e-10))
        else:
            confidence = max(0.0, 1.0 - aci_variance / self.config.aci_variance_threshold)

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=confidence,
            details={
                'aci_variance': aci_variance,
                'threshold': self.config.aci_variance_threshold,
                'recent_aci_mean': np.mean(recent_aci)
            }
        )


class SolutionStabilityDetector(ConvergenceDetector):
    """
    Detect convergence based on solution stability.

    Converged if: No improvement for K iterations
    """

    def detect(self, search_state: SearchState) -> ConvergenceResult:
        iterations_since_improvement = search_state.iteration - search_state.last_improvement_iteration

        # Check if enough iterations since last improvement
        converged = iterations_since_improvement >= self.config.no_improvement_iterations

        # Confidence based on how far beyond threshold
        if converged:
            confidence = min(1.0, iterations_since_improvement / self.config.no_improvement_iterations)
        else:
            confidence = iterations_since_improvement / self.config.no_improvement_iterations

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=confidence,
            details={
                'iterations_since_improvement': iterations_since_improvement,
                'threshold': self.config.no_improvement_iterations,
                'last_improvement': search_state.last_improvement_iteration
            }
        )


class VarianceDetector(ConvergenceDetector):
    """
    Detect convergence based on moving window variance.

    Converged if: Variance in window < threshold
    """

    def detect(self, search_state: SearchState) -> ConvergenceResult:
        if len(search_state.value_history) < self.config.convergence_window:
            return ConvergenceResult(
                converged=False,
                method=type(self),
                iteration=search_state.iteration,
                confidence=0.0,
                details={'reason': 'Insufficient value history'}
            )

        # Get recent values
        recent_values = search_state.value_history[-self.config.convergence_window:]

        # Calculate variance
        variance = np.var(recent_values)

        # Check convergence
        converged = variance < self.config.variance_threshold

        # Confidence based on how far below threshold
        if converged:
            confidence = min(1.0, self.config.variance_threshold / (variance + 1e-10))
        else:
            confidence = max(0.0, 1.0 - variance / self.config.variance_threshold)

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=confidence,
            details={
                'variance': variance,
                'threshold': self.config.variance_threshold,
                'recent_mean': np.mean(recent_values)
            }
        )


class GradientDetector(ConvergenceDetector):
    """
    Detect convergence based on gradient (rate of improvement).

    Converged if: Average absolute gradient in window < threshold
    """

    def detect(self, search_state: SearchState) -> ConvergenceResult:
        if len(search_state.value_history) < self.config.convergence_window + 1:
            return ConvergenceResult(
                converged=False,
                method=type(self),
                iteration=search_state.iteration,
                confidence=0.0,
                details={'reason': 'Insufficient value history'}
            )

        # Get recent values
        recent_values = search_state.value_history[-(self.config.convergence_window + 1):]

        # Calculate gradients
        gradients = [abs(recent_values[i+1] - recent_values[i])
                    for i in range(len(recent_values) - 1)]

        # Average gradient
        avg_gradient = np.mean(gradients)

        # Check convergence
        converged = avg_gradient < self.config.gradient_threshold

        # Confidence based on how far below threshold
        if converged:
            confidence = min(1.0, self.config.gradient_threshold / (avg_gradient + 1e-10))
        else:
            confidence = max(0.0, 1.0 - avg_gradient / self.config.gradient_threshold)

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=confidence,
            details={
                'gradient': avg_gradient,
                'threshold': self.config.gradient_threshold,
                'max_gradient': max(gradients)
            }
        )


class GelmanRubinDetector(ConvergenceDetector):
    """
    Detect convergence using Gelman-Rubin R-hat statistic.

    Converged if: R-hat < threshold (typically 1.1 or 1.05)

    Note: Requires multiple chains (parallel MCTS).
    """

    def __init__(self, config: ConvergenceConfig):
        super().__init__(config)
        self.chain_histories: List[List[float]] = []

    def add_chain(self, chain_values: List[float]):
        """Add a chain's value history"""
        self.chain_histories.append(chain_values)

    def clear_chains(self):
        """Clear all chain histories"""
        self.chain_histories = []

    def detect(self, search_state: SearchState) -> ConvergenceResult:
        # Need at least 2 chains
        if len(self.chain_histories) < 2:
            return ConvergenceResult(
                converged=False,
                method=type(self),
                iteration=search_state.iteration,
                confidence=0.0,
                details={'reason': 'Insufficient chains (need ≥2)'}
            )

        # Calculate R-hat
        try:
            r_hat = self._calculate_r_hat()
        except Exception as e:
            return ConvergenceResult(
                converged=False,
                method=type(self),
                iteration=search_state.iteration,
                confidence=0.0,
                details={'reason': f'Calculation error: {e}'}
            )

        # Check convergence
        converged = r_hat < self.config.r_hat_threshold

        # Confidence based on how far below threshold
        if converged:
            confidence = min(1.0, (self.config.r_hat_threshold - 1.0) / (r_hat - 1.0 + 1e-10))
        else:
            confidence = max(0.0, 1.0 - (r_hat - 1.0) / (self.config.r_hat_threshold - 1.0))

        return ConvergenceResult(
            converged=converged,
            method=type(self),
            iteration=search_state.iteration,
            confidence=confidence,
            details={
                'r_hat': r_hat,
                'threshold': self.config.r_hat_threshold,
                'num_chains': len(self.chain_histories)
            }
        )

    def _calculate_r_hat(self) -> float:
        """Calculate Gelman-Rubin R-hat statistic"""
        # Get chain means
        chain_means = [np.mean(chain) for chain in self.chain_histories]
        chain_vars = [np.var(chain, ddof=1) for chain in self.chain_histories]

        m = len(self.chain_histories)  # Number of chains
        n = len(self.chain_histories[0])  # Chain length (assume equal)

        # Between-chain variance
        overall_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean(chain_vars)

        # Estimated variance
        var_hat = (n - 1) / n * W + B / n

        # R-hat
        r_hat = np.sqrt(var_hat / W)

        return r_hat


class NMaxEstimator:
    """
    Estimate and adjust N_max (maximum iterations).

    Combines ACI-based and structural complexity estimation.
    """

    def __init__(self, config: ConvergenceConfig, aci_calculator: ACICalculator = None):
        self.config = config
        self.aci_calculator = aci_calculator

    def estimate_initial(self,
                        csp: CSPInstance = None,
                        aci_result: ACIResult = None,
                        problem_size: int = 0) -> int:
        """
        Estimate initial N_max.

        Args:
            csp: CSP instance (optional)
            aci_result: ACI result (optional)
            problem_size: Problem size metric (e.g., number of variables)

        Returns:
            Estimated N_max
        """
        # ACI-based estimate
        if aci_result is not None:
            aci_estimate = self._aci_based_estimate(aci_result.ACI)
        else:
            aci_estimate = self.config.base_n_max

        # Structural estimate
        if csp is not None:
            structural_estimate = self._structural_estimate(csp)
        elif problem_size > 0:
            # Use problem size as proxy
            structural_estimate = self._size_based_estimate(problem_size)
        else:
            structural_estimate = self.config.base_n_max

        # Combine estimates
        combined = (self.config.aci_weight_n_max * aci_estimate +
                   (1 - self.config.aci_weight_n_max) * structural_estimate)

        # Apply bounds
        n_max = int(np.clip(combined,
                           self.config.min_n_max,
                           self.config.max_n_max))

        return n_max

    def _aci_based_estimate(self, aci_score: float) -> int:
        """Estimate N_max based on ACI score"""
        # High ACI → Low N_max (easy problem)
        # Low ACI → High N_max (hard problem)

        if aci_score > 0.8:
            # Highly tractable
            return int(self.config.base_n_max * 0.2)
        elif aci_score > 0.6:
            # Tractable
            return int(self.config.base_n_max * 0.5)
        elif aci_score > 0.4:
            # Challenging
            return self.config.base_n_max
        elif aci_score > 0.2:
            # Highly intractable
            return int(self.config.base_n_max * 2.0)
        else:
            # Provably intractable
            return int(self.config.base_n_max * 5.0)

    def _structural_estimate(self, csp: CSPInstance) -> int:
        """Estimate N_max based on CSP structure"""
        n = csp.num_variables()
        m = csp.num_constraints()

        # Average domain size
        domain_sizes = [v.domain_size() for v in csp.variables]
        avg_domain = np.mean(domain_sizes) if domain_sizes else 2

        # Constraint density
        if n > 1:
            max_constraints = n * (n - 1) / 2
            density = m / max_constraints
        else:
            density = 0.0

        # Complexity score
        complexity = n * np.log(avg_domain + 1) * (1 + density)

        # Scale to N_max
        n_max = int(self.config.base_n_max * complexity / 100)

        return n_max

    def _size_based_estimate(self, problem_size: int) -> int:
        """Estimate N_max based on problem size"""
        # Simple scaling
        scale = problem_size / 100  # Normalize to 100 variables
        n_max = int(self.config.base_n_max * max(0.1, min(10, scale)))

        return n_max

    def adjust_dynamic(self, search_state: SearchState) -> Tuple[int, str]:
        """
        Dynamically adjust N_max based on search progress.

        Args:
            search_state: Current search state

        Returns:
            (adjusted_n_max, reason)
        """
        current_n_max = search_state.n_max
        iteration = search_state.iteration

        # Must have some history
        if len(search_state.value_history) < self.config.convergence_window:
            return current_n_max, "Insufficient history"

        # Calculate improvement rate
        recent_values = search_state.value_history[-self.config.convergence_window:]
        improvement_rate = (recent_values[-1] - recent_values[0]) / (abs(recent_values[0]) + 1e-10)

        # ACI trajectory
        if len(search_state.aci_history) >= self.config.aci_window:
            recent_aci = search_state.aci_history[-self.config.aci_window:]
            aci_trend = (recent_aci[-1] - recent_aci[0])  # Positive = improving
        else:
            aci_trend = 0.0

        # Adjustment factor
        adjustment = 1.0

        if improvement_rate > 0.01:  # Good improvement
            if aci_trend > 0:
                adjustment = 0.8  # Can reduce (improving + better ACI)
                reason = "Good improvement, ACI improving"
            else:
                adjustment = 0.9  # Slight reduction
                reason = "Good improvement"
        elif improvement_rate < -0.001:  # Degrading
            if aci_trend < 0:
                adjustment = 1.5  # Increase significantly
                reason = "Degrading, ACI worsening"
            else:
                adjustment = 1.2  # Moderate increase
                reason = "Degrading"
        else:  # Stable
            if aci_trend > 0:
                adjustment = 0.9  # Reduce slightly (ACI improving)
                reason = "Stable, ACI improving"
            else:
                adjustment = 1.0  # Maintain
                reason = "Stable"

        # Apply adjustment
        new_n_max = int(current_n_max * adjustment)

        # Ensure minimum remaining iterations
        remaining = new_n_max - iteration
        if remaining < self.config.min_iterations_before_stop:
            new_n_max = iteration + self.config.min_iterations_before_stop
            reason += " (enforced minimum remaining)"

        # Cap at max
        new_n_max = min(new_n_max, self.config.max_n_max)

        return new_n_max, reason


class EarlyStoppingRule:
    """
    Early stopping for low ACI or no progress.
    """

    def __init__(self, config: ConvergenceConfig):
        self.config = config

    def should_stop_early(self, search_state: SearchState) -> Tuple[bool, str]:
        """
        Check if search should stop early.

        Args:
            search_state: Current search state

        Returns:
            (should_stop, reason)
        """
        if not self.config.enable_early_stopping:
            return False, "Early stopping disabled"

        # Must have minimum iterations
        if search_state.iteration < self.config.min_iterations_before_stop:
            return False, f"Minimum iterations not reached ({search_state.iteration}/{self.config.min_iterations_before_stop})"

        # Check 1: Low ACI + no improvement
        if search_state.current_aci < self.config.low_aci_threshold:
            iterations_since_improvement = search_state.iteration - search_state.last_improvement_iteration
            if iterations_since_improvement >= self.config.no_improvement_iterations:
                return True, (f"Early stopping: Low ACI ({search_state.current_aci:.3f} < {self.config.low_aci_threshold}) "
                            f"and no improvement for {iterations_since_improvement} iterations")

        # Check 2: Diminishing returns
        if len(search_state.value_history) >= self.config.convergence_window:
            recent_values = search_state.value_history[-self.config.convergence_window:]
            improvement_rate = abs(recent_values[-1] - recent_values[0]) / (abs(recent_values[0]) + 1e-10)

            if improvement_rate < self.config.diminishing_returns_threshold:
                return True, (f"Early stopping: Diminishing returns "
                            f"(improvement rate {improvement_rate:.6f} < {self.config.diminishing_returns_threshold})")

        return False, "No early stopping condition met"


class ConvergenceController:
    """
    Main convergence control interface.

    Coordinates all detectors, makes stopping decisions,
    and adjusts N_max dynamically.
    """

    def __init__(self, config: ConvergenceConfig = None,
                aci_calculator: ACICalculator = None,
                stage9_reporter: Any = None):
        """
        Initialize convergence controller.

        Args:
            config: Convergence configuration
            aci_calculator: Γ₁ ACI calculator (optional)
            stage9_reporter: Stage 9 reporter for E2E validation (optional)
        """
        self.config = config or ConvergenceConfig()
        self.aci_calculator = aci_calculator
        self.stage9_reporter = stage9_reporter

        # Initialize detectors
        self.detectors = []

        if self.config.use_aci_stability:
            self.detectors.append(ACIStabilityDetector(self.config))

        if self.config.use_solution_stability:
            self.detectors.append(SolutionStabilityDetector(self.config))

        if self.config.use_variance:
            self.detectors.append(VarianceDetector(self.config))

        if self.config.use_gradient:
            self.detectors.append(GradientDetector(self.config))

        if self.config.use_gelman_rubin:
            self.gelman_rubin_detector = GelmanRubinDetector(self.config)
            self.detectors.append(self.gelman_rubin_detector)
        else:
            self.gelman_rubin_detector = None

        # N_max estimator
        self.n_max_estimator = NMaxEstimator(self.config, self.aci_calculator)

        # Early stopping
        self.early_stopping = EarlyStoppingRule(self.config)

        # Statistics
        self.total_checks = 0
        self.total_adjustments = 0

    def get_n_max(self,
                 csp: CSPInstance = None,
                 aci_result: ACIResult = None,
                 problem_size: int = 0) -> int:
        """
        Get initial N_max estimate.

        Args:
            csp: CSP instance (optional)
            aci_result: ACI result (optional)
            problem_size: Problem size (number of variables)

        Returns:
            Estimated N_max
        """
        n_max = self.n_max_estimator.estimate_initial(csp, aci_result, problem_size)

        if self.config.verbose:
            print(f"[ConvergenceController] Initial N_max: {n_max}")

        return n_max

    def should_stop(self, search_state: SearchState) -> Tuple[bool, str]:
        """
        Check if search should stop.

        Args:
            search_state: Current search state

        Returns:
            (should_stop, reason)
        """
        self.total_checks += 1

        # Check 1: Minimum iterations
        if search_state.iteration < self.config.min_iterations_before_stop:
            return False, f"Minimum iterations not reached ({search_state.iteration}/{self.config.min_iterations_before_stop})"

        # Check 2: Maximum iterations
        if search_state.iteration >= search_state.n_max:
            return True, f"Maximum iterations reached ({search_state.iteration}/{search_state.n_max})"

        # Check 3: Early stopping
        should_stop_early, early_reason = self.early_stopping.should_stop_early(search_state)
        if should_stop_early:
            return True, early_reason

        # Check 4: Regular convergence detectors
        # Only check at intervals
        if search_state.iteration % self.config.check_interval != 0:
            return False, "Waiting for next check interval"

        # Run all detectors
        detector_results = []
        for detector in self.detectors:
            try:
                result = detector.detect(search_state)
                detector_results.append(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"[ConvergenceController] Detector {type(detector).__name__} failed: {e}")

        # Combine results based on strategy
        should_stop, reason = self._combine_results(detector_results)

        # Report to Stage 9 if enabled
        if self.stage9_reporter and self.config.report_to_stage9:
            try:
                self.stage9_reporter.report_convergence_check(
                    iteration=search_state.iteration,
                    detector_results=detector_results,
                    decision=should_stop
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"[ConvergenceController] Failed to report to Stage 9: {e}")

        return should_stop, reason

    def adjust_n_max(self, search_state: SearchState) -> Tuple[int, str]:
        """
        Dynamically adjust N_max based on progress.

        Args:
            search_state: Current search state

        Returns:
            (new_n_max, reason)
        """
        if not self.config.use_dynamic_adjustment:
            return search_state.n_max, "Dynamic adjustment disabled"

        # Only adjust at intervals
        if search_state.iteration % self.config.adjust_interval != 0:
            return search_state.n_max, "Waiting for next adjustment interval"

        new_n_max, reason = self.n_max_estimator.adjust_dynamic(search_state)

        self.total_adjustments += 1

        if self.config.verbose and new_n_max != search_state.n_max:
            print(f"[ConvergenceController] Adjusting N_max: {search_state.n_max} -> {new_n_max} ({reason})")

        # Report to Stage 9 if enabled
        if self.stage9_reporter and self.config.report_to_stage9:
            try:
                self.stage9_reporter.report_n_max_adjustment(
                    iteration=search_state.iteration,
                    old_n_max=search_state.n_max,
                    new_n_max=new_n_max,
                    reason=reason
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"[ConvergenceController] Failed to report to Stage 9: {e}")

        return new_n_max, reason

    def _combine_results(self, detector_results: List[ConvergenceResult]) -> Tuple[bool, str]:
        """
        Combine multiple detector results based on stopping strategy.

        Args:
            detector_results: Results from all detectors

        Returns:
            (should_stop, reason)
        """
        if not detector_results:
            return False, "No detector results"

        strategy = self.config.stopping_strategy.upper()

        if strategy == 'ANY':
            # Stop if any detector converged
            converged_results = [r for r in detector_results if r.converged]
            if converged_results:
                detector_names = [type(r.method).__name__ for r in converged_results]
                return True, f"Converged (ANY): {', '.join(detector_names)}"
            else:
                return False, "No detectors converged (ANY strategy)"

        elif strategy == 'ALL':
            # Stop if all detectors converged
            all_converged = all(r.converged for r in detector_results)
            if all_converged:
                return True, "Converged (ALL): All detectors agree"
            else:
                converged_count = sum(1 for r in detector_results if r.converged)
                return False, f"Not converged (ALL): {converged_count}/{len(detector_results)} converged"

        elif strategy == 'MAJORITY':
            # Stop if majority converged
            converged_count = sum(1 for r in detector_results if r.converged)
            if converged_count > len(detector_results) / 2:
                return True, f"Converged (MAJORITY): {converged_count}/{len(detector_results)} converged"
            else:
                return False, f"Not converged (MAJORITY): {converged_count}/{len(detector_results)} converged"

        elif strategy == 'WEIGHTED':
            # Weighted combination using custom detector weights from config
            weights = []
            for result in detector_results:
                detector_name = type(result.method).__name__
                weight = self.config.detector_weights.get(detector_name, 1.0)
                weights.append(weight)

            total_weight = sum(weights)

            weighted_sum = sum(w * r.confidence if r.converged else 0
                             for w, r in zip(weights, detector_results))

            if weighted_sum / total_weight > 0.5:
                return True, f"Converged (WEIGHTED): Score = {weighted_sum/total_weight:.2f}"
            else:
                return False, f"Not converged (WEIGHTED): Score = {weighted_sum/total_weight:.2f}"

        else:
            # Default to MAJORITY
            return self._combine_results_with_strategy(detector_results, 'MAJORITY')

    def get_statistics(self) -> Dict:
        """Get controller statistics"""
        return {
            'total_checks': self.total_checks,
            'total_adjustments': self.total_adjustments,
            'num_detectors': len(self.detectors),
            'config': self.config
        }


class Stage9Reporter:
    """
    Reporter for Stage 9 (E2E Convergence Validation).

    Placeholder implementation - actual integration depends on Stage 9 interface.
    """

    def __init__(self, stage9_validator=None):
        self.stage9 = stage9_validator
        self.reports = []

    def report_convergence_check(self, iteration: int, detector_results: List, decision: bool):
        """Report convergence check to Stage 9"""
        report = {
            'type': 'convergence_check',
            'iteration': iteration,
            'detector_results': detector_results,
            'decision': decision
        }
        self.reports.append(report)

        if self.stage9:
            # Send to actual Stage 9 validator
            pass

    def report_n_max_adjustment(self, iteration: int, old_n_max: int, new_n_max: int, reason: str):
        """Report N_max adjustment to Stage 9"""
        report = {
            'type': 'n_max_adjustment',
            'iteration': iteration,
            'old_n_max': old_n_max,
            'new_n_max': new_n_max,
            'reason': reason
        }
        self.reports.append(report)

        if self.stage9:
            # Send to actual Stage 9 validator
            pass

    def get_reports(self) -> List[Dict]:
        """Get all reports"""
        return self.reports


# Convenience functions
def create_convergence_controller(
    use_aci: bool = True,
    use_dynamic_adjustment: bool = True,
    verbose: bool = False
) -> ConvergenceController:
    """
    Convenience function to create convergence controller with sensible defaults.

    Args:
        use_aci: Enable ACI-based features
        use_dynamic_adjustment: Enable dynamic N_max adjustment
        verbose: Enable verbose logging

    Returns:
        ConvergenceController instance
    """
    config = ConvergenceConfig(
        use_aci_stability=use_aci,
        use_dynamic_adjustment=use_dynamic_adjustment,
        verbose=verbose
    )

    return ConvergenceController(config)


# Example usage (for testing)
if __name__ == "__main__":
    print("Convergence Controller - Ready")
    print("=" * 70)

    # Create controller
    controller = create_convergence_controller(verbose=True)

    # Simulate search
    print("\nSimulating MCTS search...")
    print("-" * 70)

    search_state = SearchState(
        start_time=time.time(),
        n_max=controller.get_n_max(problem_size=50)
    )

    # Simulate converging search
    for i in range(1, 201):
        # Simulate value with noise and decreasing improvement
        improvement = 0.01 * np.exp(-i / 50)
        noise = np.random.normal(0, 0.001)
        value = 0.5 + (1 - np.exp(-i / 50)) + noise

        # Simulate ACI (slightly improving)
        aci = 0.5 + 0.1 * (1 - np.exp(-i / 100))

        # Update search state
        search_state.update(i, value, aci)

        # Check convergence
        should_stop, reason = controller.should_stop(search_state)

        # Adjust N_max
        if i % 50 == 0:
            new_n_max, adj_reason = controller.adjust_n_max(search_state)
            search_state.n_max = new_n_max

        # Log progress
        if i % 20 == 0:
            print(f"Iteration {i}: value={value:.4f}, ACI={aci:.3f}, N_max={search_state.n_max}")

        # Stop if converged
        if should_stop:
            print(f"\nStopping at iteration {i}: {reason}")
            break

    # Statistics
    print("\n" + "=" * 70)
    print("Search Statistics")
    print("=" * 70)
    print(f"Final iteration: {search_state.iteration}")
    print(f"Final value: {search_state.current_value:.4f}")
    print(f"Final ACI: {search_state.current_aci:.3f}")
    print(f"Elapsed time: {search_state.elapsed_time:.2f}s")
    print(f"\nController statistics: {controller.get_statistics()}")

    print("\n" + "=" * 70)
    print("Convergence Controller - Test Complete")
