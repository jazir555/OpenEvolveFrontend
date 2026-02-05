"""Execution monitoring and early stopping - addresses OpenEvolve gaps.

This module provides convergence detection and early stopping that can
wrap around the existing AgnosticPESEngine without modifying it.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceStatus:
    """Current convergence status."""
    
    is_converged: bool = False
    convergence_score: float = 0.0
    plateau_detected: bool = False
    diversity_decreasing: bool = False
    
    fitness_trend: str = "stable"  # improving, stable, declining
    improvement_rate: float = 0.0
    generations_without_improvement: int = 0
    
    reason: str = ""


@dataclass
class ExecutionSnapshot:
    """Snapshot of execution state at a point in time."""
    
    iteration: int
    best_fitness: float
    avg_fitness: float
    diversity_score: float
    timestamp_ms: int
    
    # Optional detailed metrics
    population_stats: Optional[Dict[str, Any]] = None
    evaluation_count: int = 0


class ConvergenceDetector:
    """Detects convergence - extracted from LoongFlow PES.
    
    OpenEvolve had early stopping disabled by default and only basic
    patience-based stopping. This adds multi-factor convergence detection.
    """
    
    def __init__(
        self,
        fitness_threshold: float = 0.95,
        plateau_threshold: float = 0.001,
        diversity_threshold: float = 0.1,
        window_size: int = 10
    ):
        self.fitness_threshold = fitness_threshold
        self.plateau_threshold = plateau_threshold
        self.diversity_threshold = diversity_threshold
        self.window_size = window_size
        
        self.history: deque = deque(maxlen=window_size)
        self.best_fitness_history: deque = deque(maxlen=window_size)
    
    def update(self, snapshot: ExecutionSnapshot):
        """Update with new execution snapshot."""
        self.history.append(snapshot)
        self.best_fitness_history.append(snapshot.best_fitness)
    
    def check_convergence(self) -> ConvergenceStatus:
        """Check for convergence based on multiple factors.
        
        Returns ConvergenceStatus with detailed information.
        """
        status = ConvergenceStatus()
        
        if len(self.history) < 3:
            status.reason = "Insufficient history"
            return status
        
        # Check 1: Fitness threshold reached
        current_best = self.best_fitness_history[-1]
        if current_best >= self.fitness_threshold:
            status.is_converged = True
            status.reason = f"Fitness threshold reached: {current_best:.3f} >= {self.fitness_threshold}"
            return status
        
        # Check 2: Plateau detection
        if len(self.best_fitness_history) >= self.window_size:
            recent_best = list(self.best_fitness_history)[-self.window_size:]
            improvement = recent_best[-1] - recent_best[0]
            
            if abs(improvement) < self.plateau_threshold:
                status.plateau_detected = True
                status.improvement_rate = improvement / len(recent_best) if len(recent_best) > 0 else 0
                
                if improvement <= 0:
                    status.fitness_trend = "stable" if improvement == 0 else "declining"
                    status.is_converged = True
                    status.reason = f"Plateau detected: improvement={improvement:.6f}"
                    return status
        
        # Check 3: Diversity loss
        if len(self.history) >= 3:
            recent_diversity = [s.diversity_score for s in list(self.history)[-3:]]
            if all(d < self.diversity_threshold for d in recent_diversity):
                status.diversity_decreasing = True
                # Don't necessarily stop on diversity loss alone
                status.reason = "Diversity low but continuing"
        
        # Check 4: Improvement trend
        if len(self.best_fitness_history) >= 5:
            older = list(self.best_fitness_history)[-5]
            newer = list(self.best_fitness_history)[-1]
            improvement = newer - older
            
            status.improvement_rate = improvement / 5
            if improvement > 0.01:
                status.fitness_trend = "improving"
            elif improvement > -0.01:
                status.fitness_trend = "stable"
            else:
                status.fitness_trend = "declining"
        
        status.reason = f"Trend: {status.fitness_trend}, rate: {status.improvement_rate:.6f}"
        return status
    
    def get_convergence_score(self) -> float:
        """Get a 0-1 convergence score (1 = fully converged)."""
        if len(self.history) < 3:
            return 0.0
        
        scores = []
        
        # Fitness proximity to threshold
        current_best = self.best_fitness_history[-1]
        fitness_score = min(1.0, current_best / self.fitness_threshold)
        scores.append(fitness_score)
        
        # Plateau score (lower improvement = higher plateau score)
        if len(self.best_fitness_history) >= self.window_size:
            recent = list(self.best_fitness_history)[-self.window_size:]
            improvement = abs(recent[-1] - recent[0])
            plateau_score = 1.0 - min(1.0, improvement / self.plateau_threshold)
            scores.append(plateau_score)
        
        return statistics.mean(scores) if scores else 0.0


class EarlyStoppingController:
    """Controls early stopping - extracted from LoongFlow.
    
    Wraps around existing evolution without modification.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_improvement: float = 0.01,
        improvement_window: int = 10,
        max_evaluations: int = 10000,
        max_duration_ms: int = 300000
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.improvement_window = improvement_window
        self.max_evaluations = max_evaluations
        self.max_duration_ms = max_duration_ms
        
        self.best_fitness = 0.0
        self.iterations_without_improvement = 0
        self.total_evaluations = 0
        self.start_time_ms = None
        self.stopped = False
        self.stop_reason = None
        
        # Convergence detector
        self.convergence_detector = ConvergenceDetector()
    
    def start(self):
        """Start monitoring."""
        import time
        self.start_time_ms = int(time.time() * 1000)
        self.stopped = False
        self.stop_reason = None
        logger.info("Early stopping controller started")
    
    def check_should_stop(
        self,
        iteration: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float,
        evaluations_this_iteration: int = 0
    ) -> tuple[bool, str]:
        """Check if evolution should stop.
        
        Returns:
            (should_stop, reason)
        """
        if self.stopped:
            return True, self.stop_reason or "Already stopped"
        
        import time
        current_time = int(time.time() * 1000)
        self.total_evaluations += evaluations_this_iteration
        
        # Update convergence detector
        snapshot = ExecutionSnapshot(
            iteration=iteration,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            diversity_score=diversity,
            timestamp_ms=current_time,
            evaluation_count=self.total_evaluations
        )
        self.convergence_detector.update(snapshot)
        
        # Check 1: Max evaluations
        if self.total_evaluations >= self.max_evaluations:
            self.stopped = True
            self.stop_reason = f"Max evaluations reached: {self.total_evaluations}"
            return True, self.stop_reason
        
        # Check 2: Max duration
        if self.start_time_ms and (current_time - self.start_time_ms) >= self.max_duration_ms:
            self.stopped = True
            duration_sec = (current_time - self.start_time_ms) / 1000
            self.stop_reason = f"Max duration reached: {duration_sec:.0f}s"
            return True, self.stop_reason
        
        # Check 3: Convergence
        conv_status = self.convergence_detector.check_convergence()
        if conv_status.is_converged:
            self.stopped = True
            self.stop_reason = f"Converged: {conv_status.reason}"
            return True, self.stop_reason
        
        # Check 4: No improvement (patience)
        if best_fitness > self.best_fitness + self.min_improvement:
            self.best_fitness = best_fitness
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
            
            if self.iterations_without_improvement >= self.patience:
                self.stopped = True
                self.stop_reason = (
                    f"No improvement for {self.iterations_without_improvement} iterations "
                    f"(patience={self.patience})"
                )
                return True, self.stop_reason
        
        return False, f"Continue: fitness={best_fitness:.3f}, evals={self.total_evaluations}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        import time
        current_time = int(time.time() * 1000)
        duration = (current_time - self.start_time_ms) / 1000 if self.start_time_ms else 0
        
        return {
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "best_fitness": self.best_fitness,
            "iterations_without_improvement": self.iterations_without_improvement,
            "total_evaluations": self.total_evaluations,
            "duration_seconds": duration,
            "convergence_score": self.convergence_detector.get_convergence_score(),
        }


class ExecutionMonitor:
    """Monitors execution and provides callbacks for external systems.
    
    This wraps around existing evolution without modifying it.
    """
    
    def __init__(
        self,
        early_stopping: Optional[EarlyStoppingController] = None,
        budget_tracker=None,
        on_iteration_callback: Optional[Callable] = None,
        on_stop_callback: Optional[Callable] = None
    ):
        self.early_stopping = early_stopping
        self.budget_tracker = budget_tracker
        self.on_iteration = on_iteration_callback
        self.on_stop = on_stop_callback
        
        self.iteration_history: List[ExecutionSnapshot] = []
        self.start_time_ms = None
    
    def start(self):
        """Start monitoring."""
        import time
        self.start_time_ms = int(time.time() * 1000)
        if self.early_stopping:
            self.early_stopping.start()
        logger.info("Execution monitor started")
    
    def record_iteration(
        self,
        iteration: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float,
        evaluations: int = 0,
        extra_metrics: Optional[Dict] = None
    ) -> tuple[bool, str]:
        """Record an iteration and check stopping conditions.
        
        Returns:
            (should_continue, status_message)
        """
        import time
        
        # Create snapshot
        snapshot = ExecutionSnapshot(
            iteration=iteration,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            diversity_score=diversity,
            timestamp_ms=int(time.time() * 1000),
            evaluation_count=evaluations,
            population_stats=extra_metrics
        )
        self.iteration_history.append(snapshot)
        
        # Check early stopping
        if self.early_stopping:
            should_stop, reason = self.early_stopping.check_should_stop(
                iteration, best_fitness, avg_fitness, diversity, evaluations
            )
            if should_stop:
                if self.on_stop:
                    self.on_stop(reason, self.get_summary())
                return False, reason
        
        # Check budget
        if self.budget_tracker:
            should_continue, budget_reason = self.budget_tracker.should_continue()
            if not should_continue:
                if self.on_stop:
                    self.on_stop(budget_reason, self.get_summary())
                return False, budget_reason
        
        # Callback
        if self.on_iteration:
            self.on_iteration(snapshot)
        
        return True, f"Iteration {iteration}: fitness={best_fitness:.3f}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        if not self.iteration_history:
            return {}
        
        fitness_values = [s.best_fitness for s in self.iteration_history]
        diversity_values = [s.diversity_score for s in self.iteration_history]
        
        return {
            "total_iterations": len(self.iteration_history),
            "best_fitness": max(fitness_values) if fitness_values else 0.0,
            "final_fitness": fitness_values[-1] if fitness_values else 0.0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0.0,
            "fitness_improvement": (fitness_values[-1] - fitness_values[0]) if len(fitness_values) > 1 else 0.0,
            "avg_diversity": sum(diversity_values) / len(diversity_values) if diversity_values else 0.0,
            "early_stopping_status": self.early_stopping.get_status() if self.early_stopping else None,
        }
