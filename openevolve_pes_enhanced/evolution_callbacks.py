"""Evolution callbacks for monitoring and controlling PES iterations.

This module provides a callback system that allows monitoring and intervention
during the evolution process. Since AgnosticPESEngine runs as a black box,
these callbacks are designed to work with MonitoredAgnosticPES which wraps
the engine to inject monitoring between iterations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class EvolutionState(Enum):
    """State of the evolution process."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class IterationMetrics:
    """Metrics collected at each iteration."""
    iteration: int
    total_iterations: int
    
    # Fitness metrics
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    
    # Population metrics
    population_size: int = 0
    diversity_score: float = 0.0
    
    # Test results (for AgnosticPES)
    tests_passed: int = 0
    tests_total: int = 0
    failing_tests: List[str] = field(default_factory=list)
    
    # Cost metrics
    cost_this_iteration: float = 0.0
    total_cost: float = 0.0
    tokens_used: int = 0
    
    # Code evolution
    current_code: str = ""
    fixes_applied_this_iteration: List[str] = field(default_factory=list)
    total_fixes_applied: List[str] = field(default_factory=list)
    
    # Timing
    iteration_duration_ms: int = 0
    total_duration_ms: int = 0
    
    # Additional custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "worst_fitness": self.worst_fitness,
            "population_size": self.population_size,
            "diversity_score": self.diversity_score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "failing_tests": self.failing_tests,
            "cost_this_iteration": self.cost_this_iteration,
            "total_cost": self.total_cost,
            "tokens_used": self.tokens_used,
            "iteration_duration_ms": self.iteration_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "fixes_applied": len(self.fixes_applied_this_iteration),
            "total_fixes": len(self.total_fixes_applied),
            "progress_pct": (self.iteration / self.total_iterations * 100) if self.total_iterations > 0 else 0,
            **self.custom_metrics
        }


@dataclass
class EvolutionContext:
    """Context passed to callbacks containing evolution state."""
    state: EvolutionState
    current_iteration: int
    max_iterations: int
    start_time_ms: int = 0
    
    # Historical data
    iteration_history: List[IterationMetrics] = field(default_factory=list)
    
    # Stopping control
    stop_requested: bool = False
    stop_reason: Optional[str] = None
    
    # Problem info
    problem_type: str = "general"
    language: str = "python"
    
    def get_improvement_rate(self, window: int = 3) -> float:
        """Calculate improvement rate over recent iterations."""
        if len(self.iteration_history) < window + 1:
            return 0.0
        
        recent = self.iteration_history[-window:]
        older = self.iteration_history[-(window+1):-1]
        
        recent_avg = sum(m.best_fitness for m in recent) / len(recent)
        older_avg = sum(m.best_fitness for m in older) / len(older)
        
        return recent_avg - older_avg
    
    def has_converged(self, threshold: float = 0.001, window: int = 3) -> bool:
        """Check if evolution has converged."""
        if len(self.iteration_history) < window:
            return False
        
        recent = self.iteration_history[-window:]
        fitness_values = [m.best_fitness for m in recent]
        
        if max(fitness_values) >= 1.0:
            return True
        
        improvement = max(fitness_values) - min(fitness_values)
        return improvement < threshold


class EvolutionCallback(ABC):
    """Abstract base class for evolution callbacks.
    
    Callbacks can monitor evolution progress and optionally request stopping
    when certain conditions are met (budget exceeded, convergence detected, etc.)
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.enabled = True
    
    async def on_evolution_start(
        self,
        context: EvolutionContext,
        initial_code: str,
        tests: List[Dict]
    ) -> None:
        """Called when evolution starts.
        
        Args:
            context: Evolution context
            initial_code: Starting code
            tests: Test cases being used
        """
        pass
    
    @abstractmethod
    async def on_iteration_start(
        self,
        iteration: int,
        context: EvolutionContext
    ) -> None:
        """Called at the start of each iteration.
        
        Args:
            iteration: Current iteration number (0-indexed)
            context: Evolution context
        """
        raise NotImplementedError
    
    @abstractmethod
    async def on_iteration_end(
        self,
        iteration: int,
        metrics: IterationMetrics,
        context: EvolutionContext
    ) -> None:
        """Called at the end of each iteration with metrics.
        
        Args:
            iteration: Current iteration number (0-indexed)
            metrics: Metrics collected during this iteration
            context: Evolution context
        """
        raise NotImplementedError
    
    async def should_stop(
        self,
        context: EvolutionContext,
        metrics: Optional[IterationMetrics] = None
    ) -> Tuple[bool, str]:
        """Determine if evolution should stop.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        return False, ""
    
    async def on_evolution_end(
        self,
        context: EvolutionContext,
        final_metrics: Optional[IterationMetrics] = None,
        result: Optional[Any] = None
    ) -> None:
        """Called when evolution ends.
        
        Args:
            context: Evolution context
            final_metrics: Metrics from final iteration
            result: Final evolution result
        """
        pass
    
    def disable(self) -> None:
        """Disable this callback."""
        self.enabled = False
    
    def enable(self) -> None:
        """Enable this callback."""
        self.enabled = True


class BudgetAwareCallback(EvolutionCallback):
    """Callback that monitors and enforces budget constraints.
    
    Tracks costs across iterations and stops evolution when budget limits
    are reached.
    """
    
    def __init__(
        self,
        max_cost_usd: float = 10.0,
        max_tokens: Optional[int] = None,
        max_time_seconds: Optional[float] = None,
        name: Optional[str] = None
    ):
        super().__init__(name or "BudgetAwareCallback")
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.max_time_ms = int(max_time_seconds * 1000) if max_time_seconds else None
        
        # Tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.start_time_ms = 0
        self.cost_history: List[Dict] = []
    
    async def on_evolution_start(
        self,
        context: EvolutionContext,
        initial_code: str,
        tests: List[Dict]
    ) -> None:
        """Initialize budget tracking."""
        import time
        self.start_time_ms = int(time.time() * 1000)
        self.total_cost = 0.0
        self.total_tokens = 0
        self.cost_history = []
        logger.info(f"[Budget] Initialized with max_cost=${self.max_cost_usd:.2f}")
    
    async def on_iteration_start(
        self,
        iteration: int,
        context: EvolutionContext
    ) -> None:
        """Called at iteration start."""
        pass
    
    async def on_iteration_end(
        self,
        iteration: int,
        metrics: IterationMetrics,
        context: EvolutionContext
    ) -> None:
        """Record cost from metrics."""
        self.total_cost += metrics.cost_this_iteration
        self.total_tokens += metrics.tokens_used
        
        self.cost_history.append({
            "iteration": iteration,
            "cost": metrics.cost_this_iteration,
            "total_cost": self.total_cost,
            "tokens": metrics.tokens_used
        })
        
        # Log at certain thresholds
        budget_pct = self.total_cost / self.max_cost_usd * 100
        if budget_pct in [50, 75, 90, 95]:
            logger.warning(f"[Budget] {budget_pct:.0f}% of budget used (${self.total_cost:.2f}/${self.max_cost_usd:.2f})")
    
    async def should_stop(
        self,
        context: EvolutionContext,
        metrics: Optional[IterationMetrics] = None
    ) -> Tuple[bool, str]:
        """Check if budget exceeded."""
        # Check cost budget
        if self.total_cost >= self.max_cost_usd:
            return True, f"Budget exceeded: ${self.total_cost:.2f} >= ${self.max_cost_usd:.2f}"
        
        # Check token budget
        if self.max_tokens and self.total_tokens >= self.max_tokens:
            return True, f"Token limit exceeded: {self.total_tokens} >= {self.max_tokens}"
        
        # Check time budget
        if self.max_time_ms and self.start_time_ms:
            import time
            elapsed = int(time.time() * 1000) - self.start_time_ms
            if elapsed >= self.max_time_ms:
                return True, f"Time limit exceeded: {elapsed/1000:.1f}s >= {self.max_time_ms/1000:.1f}s"
        
        return False, ""
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        import time
        
        elapsed_ms = 0
        if self.start_time_ms:
            elapsed_ms = int(time.time() * 1000) - self.start_time_ms
        
        return {
            "total_cost_usd": self.total_cost,
            "max_cost_usd": self.max_cost_usd,
            "budget_used_pct": (self.total_cost / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "elapsed_ms": elapsed_ms,
            "max_time_ms": self.max_time_ms,
            "iterations_recorded": len(self.cost_history)
        }


class MonitoringCallback(EvolutionCallback):
    """Callback that monitors evolution progress and detects convergence.
    
    Uses convergence detection and early stopping logic to determine
    when evolution should terminate early.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_improvement: float = 0.01,
        convergence_threshold: float = 0.95,
        name: Optional[str] = None
    ):
        super().__init__(name or "MonitoringCallback")
        self.patience = patience
        self.min_improvement = min_improvement
        self.convergence_threshold = convergence_threshold
        
        # State tracking
        self.best_fitness = 0.0
        self.iterations_without_improvement = 0
        self.fitness_history: List[float] = []
        self.improvement_history: List[float] = []
    
    async def on_evolution_start(
        self,
        context: EvolutionContext,
        initial_code: str,
        tests: List[Dict]
    ) -> None:
        """Reset monitoring state."""
        self.best_fitness = 0.0
        self.iterations_without_improvement = 0
        self.fitness_history = []
        self.improvement_history = []
        logger.info(f"[Monitor] Started with patience={self.patience}, min_improvement={self.min_improvement}")
    
    async def on_iteration_start(
        self,
        iteration: int,
        context: EvolutionContext
    ) -> None:
        """Called at iteration start."""
        pass
    
    async def on_iteration_end(
        self,
        iteration: int,
        metrics: IterationMetrics,
        context: EvolutionContext
    ) -> None:
        """Update execution monitor with current state."""
        current_fitness = metrics.best_fitness
        self.fitness_history.append(current_fitness)
        
        # Calculate improvement
        improvement = current_fitness - self.best_fitness
        self.improvement_history.append(improvement)
        
        if improvement > self.min_improvement:
            self.best_fitness = current_fitness
            self.iterations_without_improvement = 0
            logger.debug(f"[Monitor] Iteration {iteration}: New best fitness {current_fitness:.3f} (+{improvement:.3f})")
        else:
            self.iterations_without_improvement += 1
            logger.debug(f"[Monitor] Iteration {iteration}: No significant improvement ({self.iterations_without_improvement}/{self.patience})")
    
    async def should_stop(
        self,
        context: EvolutionContext,
        metrics: Optional[IterationMetrics] = None
    ) -> Tuple[bool, str]:
        """Check if should stop based on convergence criteria."""
        if not metrics:
            return False, ""
        
        # Check 1: Perfect fitness achieved
        if metrics.best_fitness >= 1.0:
            return True, f"Perfect fitness achieved: {metrics.best_fitness:.3f}"
        
        # Check 2: Convergence threshold reached
        if metrics.best_fitness >= self.convergence_threshold:
            return True, f"Convergence threshold reached: {metrics.best_fitness:.3f} >= {self.convergence_threshold}"
        
        # Check 3: No improvement for patience iterations
        if self.iterations_without_improvement >= self.patience:
            return True, (
                f"No improvement for {self.iterations_without_improvement} iterations "
                f"(patience={self.patience})"
            )
        
        # Check 4: Plateau detection
        if len(self.improvement_history) >= self.patience:
            recent_improvements = self.improvement_history[-self.patience:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            if avg_improvement < self.min_improvement / 10:
                return True, f"Plateau detected: avg improvement {avg_improvement:.6f}"
        
        return False, ""
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "best_fitness": self.best_fitness,
            "iterations_without_improvement": self.iterations_without_improvement,
            "patience": self.patience,
            "convergence_threshold": self.convergence_threshold,
            "fitness_history": self.fitness_history,
            "is_converged": self.iterations_without_improvement >= self.patience
        }


class LoggingCallback(EvolutionCallback):
    """Callback that logs evolution progress.
    
    Useful for debugging and monitoring evolution runs.
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        log_every_n_iterations: int = 1,
        name: Optional[str] = None
    ):
        super().__init__(name or "LoggingCallback")
        self.log_level = log_level
        self.log_every_n = log_every_n_iterations
    
    async def on_evolution_start(
        self,
        context: EvolutionContext,
        initial_code: str,
        tests: List[Dict]
    ) -> None:
        """Log evolution start."""
        logger.log(self.log_level, f"[Evolution] Starting evolution with {len(tests)} tests")
        logger.log(self.log_level, f"[Evolution] Max iterations: {context.max_iterations}")
    
    async def on_iteration_start(
        self,
        iteration: int,
        context: EvolutionContext
    ) -> None:
        """Log iteration start."""
        if iteration % self.log_every_n == 0:
            logger.log(self.log_level, f"[Evolution] Starting iteration {iteration + 1}/{context.max_iterations}")
    
    async def on_iteration_end(
        self,
        iteration: int,
        metrics: IterationMetrics,
        context: EvolutionContext
    ) -> None:
        """Log iteration results."""
        if iteration % self.log_every_n == 0:
            logger.log(
                self.log_level,
                f"[Evolution] Iteration {iteration + 1}: "
                f"fitness={metrics.best_fitness:.3f}, "
                f"tests={metrics.tests_passed}/{metrics.tests_total}, "
                f"cost=${metrics.total_cost:.4f}"
            )
    
    async def on_evolution_end(
        self,
        context: EvolutionContext,
        final_metrics: Optional[IterationMetrics] = None,
        result: Optional[Any] = None
    ) -> None:
        """Log evolution end."""
        if final_metrics:
            logger.log(
                self.log_level,
                f"[Evolution] Completed after {final_metrics.iteration + 1} iterations: "
                f"final_fitness={final_metrics.best_fitness:.3f}"
            )
        else:
            logger.log(self.log_level, f"[Evolution] Completed")


class CompositeCallback(EvolutionCallback):
    """Combines multiple callbacks into one.
    
    Useful for running multiple callbacks together.
    Stops if any callback requests stopping.
    """
    
    def __init__(
        self,
        callbacks: List[EvolutionCallback],
        name: Optional[str] = None
    ):
        super().__init__(name or "CompositeCallback")
        self.callbacks = callbacks
    
    async def on_evolution_start(
        self,
        context: EvolutionContext,
        initial_code: str,
        tests: List[Dict]
    ) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    await callback.on_evolution_start(context, initial_code, tests)
                except Exception as e:
                    logger.error(f"Callback {callback.name} failed in on_evolution_start: {e}")
    
    async def on_iteration_start(
        self,
        iteration: int,
        context: EvolutionContext
    ) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    await callback.on_iteration_start(iteration, context)
                except Exception as e:
                    logger.error(f"Callback {callback.name} failed in on_iteration_start: {e}")
    
    async def on_iteration_end(
        self,
        iteration: int,
        metrics: IterationMetrics,
        context: EvolutionContext
    ) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    await callback.on_iteration_end(iteration, metrics, context)
                except Exception as e:
                    logger.error(f"Callback {callback.name} failed in on_iteration_end: {e}")
    
    async def should_stop(
        self,
        context: EvolutionContext,
        metrics: Optional[IterationMetrics] = None
    ) -> Tuple[bool, str]:
        """Return True if any callback wants to stop."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    should_stop, reason = await callback.should_stop(context, metrics)
                    if should_stop:
                        return True, f"{callback.name}: {reason}"
                except Exception as e:
                    logger.error(f"Callback {callback.name} failed in should_stop: {e}")
        return False, ""
    
    async def on_evolution_end(
        self,
        context: EvolutionContext,
        final_metrics: Optional[IterationMetrics] = None,
        result: Optional[Any] = None
    ) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    await callback.on_evolution_end(context, final_metrics, result)
                except Exception as e:
                    logger.error(f"Callback {callback.name} failed in on_evolution_end: {e}")
    
    def add_callback(self, callback: EvolutionCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback_name: str) -> bool:
        """Remove a callback by name."""
        for i, cb in enumerate(self.callbacks):
            if cb.name == callback_name:
                self.callbacks.pop(i)
                return True
        return False


# Convenience factory functions

def create_budget_callback(
    max_cost_usd: float = 10.0,
    max_tokens: Optional[int] = None,
    max_time_seconds: Optional[float] = None
) -> BudgetAwareCallback:
    """Create a budget-aware callback.
    
    Args:
        max_cost_usd: Maximum cost in USD
        max_tokens: Maximum tokens (optional)
        max_time_seconds: Maximum time in seconds (optional)
    
    Returns:
        BudgetAwareCallback instance
    """
    return BudgetAwareCallback(
        max_cost_usd=max_cost_usd,
        max_tokens=max_tokens,
        max_time_seconds=max_time_seconds
    )


def create_monitoring_callback(
    patience: int = 3,
    min_improvement: float = 0.01,
    convergence_threshold: float = 0.95
) -> MonitoringCallback:
    """Create a monitoring callback.
    
    Args:
        patience: Iterations without improvement before stopping
        min_improvement: Minimum improvement to reset patience
        convergence_threshold: Fitness threshold for convergence
    
    Returns:
        MonitoringCallback instance
    """
    return MonitoringCallback(
        patience=patience,
        min_improvement=min_improvement,
        convergence_threshold=convergence_threshold
    )


def create_logging_callback(
    log_level: int = logging.INFO,
    log_every_n_iterations: int = 1
) -> LoggingCallback:
    """Create a logging callback.
    
    Args:
        log_level: Logging level
        log_every_n_iterations: Log every N iterations
    
    Returns:
        LoggingCallback instance
    """
    return LoggingCallback(
        log_level=log_level,
        log_every_n_iterations=log_every_n_iterations
    )


def create_standard_callbacks(
    max_cost_usd: float = 10.0,
    patience: int = 3,
    enable_logging: bool = True
) -> CompositeCallback:
    """Create standard set of callbacks for typical use.
    
    Args:
        max_cost_usd: Maximum cost budget
        patience: Early stopping patience
        enable_logging: Whether to include logging
    
    Returns:
        CompositeCallback with standard callbacks
    """
    callbacks: List[EvolutionCallback] = [
        BudgetAwareCallback(max_cost_usd=max_cost_usd),
        MonitoringCallback(patience=patience)
    ]
    
    if enable_logging:
        callbacks.append(LoggingCallback())
    
    return CompositeCallback(callbacks, name="StandardCallbacks")
