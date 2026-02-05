"""Budget enforcement for PES Enhanced - stops evolution when budget exceeded.

This module provides the missing budget enforcement that the original implementation
lacked. It integrates with the existing BudgetTracker to monitor costs and gracefully
stop evolution when thresholds are exceeded.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetCheckResult:
    """Result of a budget check."""
    
    can_continue: bool
    reason: str
    status: str  # ok, warning, critical, exceeded
    percent_used: float


class BudgetEnforcer:
    """Enforces budget limits during evolution execution.
    
    This class provides the missing enforcement layer that checks budget
    during evolution and stops execution when thresholds are exceeded.
    
    Usage:
        enforcer = BudgetEnforcer(budget_tracker, execution_monitor)
        can_continue, reason = enforcer.check_budget()
        if not can_continue:
            # Stop evolution
    """
    
    def __init__(
        self,
        budget_tracker,
        execution_monitor=None,
        warning_threshold: float = 0.70,
        critical_threshold: float = 0.90
    ):
        """Initialize budget enforcer.
        
        Args:
            budget_tracker: BudgetTracker instance to monitor
            execution_monitor: Optional ExecutionMonitor for coordinated stopping
            warning_threshold: Percentage at which to warn (default 70%)
            critical_threshold: Percentage at which to stop (default 90%)
        """
        self.budget_tracker = budget_tracker
        self.execution_monitor = execution_monitor
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._stop_requested = False
        self._stop_reason: Optional[str] = None
        self._warning_logged = False
        
        logger.info(f"BudgetEnforcer initialized (warn={warning_threshold:.0%}, "
                   f"critical={critical_threshold:.0%})")
    
    def check_budget(self) -> Tuple[bool, str]:
        """Check if evolution should continue based on budget.
        
        Returns:
            Tuple of (can_continue, reason)
            - can_continue: True if budget allows continuation
            - reason: Explanation of decision
        """
        # Check for manual stop request
        if self._stop_requested:
            return False, f"Stop requested: {self._stop_reason or 'No reason given'}"
        
        # If no budget tracker, allow continuation
        if not self.budget_tracker:
            return True, "No budget tracking enabled"
        
        # Get current status from budget tracker
        status = self.budget_tracker.get_status()
        
        # Determine which metric is most critical
        max_pct = max(
            status.cost_pct_used,
            status.tokens_pct_used,
            status.time_pct_used
        )
        
        # Check exceeded first (100%+) - MUST STOP
        if status.status == "exceeded":
            reason = self._build_stop_reason(status, max_pct, exceeded=True)
            logger.warning(f"Budget EXCEEDED - stopping: {reason}")
            return False, reason
        
        # Check critical threshold (90%+) - MUST STOP
        if max_pct >= self.critical_threshold:
            reason = self._build_stop_reason(status, max_pct, critical=True)
            logger.warning(f"Budget CRITICAL - stopping: {reason}")
            return False, reason
        
        # Check warning threshold (70%+) - log but continue
        if max_pct >= self.warning_threshold and not self._warning_logged:
            self._warning_logged = True
            reason = self._build_warning_reason(status, max_pct)
            logger.warning(f"Budget WARNING: {reason}")
        
        # Budget OK
        return True, f"Budget OK: {max_pct:.1%} used"
    
    def request_stop(self, reason: str):
        """Manually request evolution to stop.
        
        Args:
            reason: Why evolution should stop
        """
        self._stop_requested = True
        self._stop_reason = reason
        logger.info(f"Budget stop requested: {reason}")
        
        # Also notify execution monitor if available
        if self.execution_monitor and hasattr(self.execution_monitor, 'early_stopping'):
            if self.execution_monitor.early_stopping:
                self.execution_monitor.early_stopping.stopped = True
                self.execution_monitor.early_stopping.stop_reason = f"Budget: {reason}"
    
    def get_status(self) -> BudgetCheckResult:
        """Get detailed budget status.
        
        Returns:
            BudgetCheckResult with full status information
        """
        if not self.budget_tracker:
            return BudgetCheckResult(
                can_continue=True,
                reason="No budget tracking",
                status="ok",
                percent_used=0.0
            )
        
        status = self.budget_tracker.get_status()
        max_pct = max(
            status.cost_pct_used,
            status.tokens_pct_used,
            status.time_pct_used
        )
        
        can_continue, reason = self.check_budget()
        
        return BudgetCheckResult(
            can_continue=can_continue,
            reason=reason,
            status=status.status,
            percent_used=max_pct
        )
    
    def _build_stop_reason(self, status, max_pct: float, critical: bool = False, exceeded: bool = False) -> str:
        """Build a detailed stop reason string."""
        if exceeded:
            level = "EXCEEDED"
            threshold = 1.0
        elif critical:
            level = "CRITICAL"
            threshold = self.critical_threshold
        else:
            level = "exceeded"
            threshold = max_pct
        
        components = []
        if status.cost_pct_used >= threshold:
            components.append(f"cost {status.cost_pct_used:.1%}")
        if status.tokens_pct_used >= threshold:
            components.append(f"tokens {status.tokens_pct_used:.1%}")
        if status.time_pct_used >= threshold:
            components.append(f"time {status.time_pct_used:.1%}")
        
        if not components:
            components.append(f"budget {max_pct:.1%}")
        
        return f"Budget {level}: {', '.join(components)}"
    
    def _build_warning_reason(self, status, max_pct: float) -> str:
        """Build a warning reason string."""
        components = []
        if status.cost_pct_used >= self.warning_threshold:
            components.append(f"cost {status.cost_pct_used:.1%} (${status.cost_used_usd:.2f})")
        if status.tokens_pct_used >= self.warning_threshold:
            components.append(f"tokens {status.tokens_pct_used:.1%} ({status.tokens_used})")
        if status.time_pct_used >= self.warning_threshold:
            components.append(f"time {status.time_pct_used:.1%} ({status.time_used_ms/1000:.0f}s)")
        
        if not components:
            components.append(f"budget {max_pct:.1%}")
        
        return f"Approaching limit: {', '.join(components)}"
    
    def create_callback(self) -> Callable[[], Tuple[bool, str]]:
        """Create a callback function for use by evolution engines.
        
        Returns:
            Callable that returns (can_continue, reason)
        """
        return self.check_budget


class BudgetEnforcedResult:
    """Wraps evolution result with budget enforcement metadata."""
    
    def __init__(
        self,
        original_result: Any,
        stopped_early: bool,
        stop_reason: Optional[str],
        final_budget_status: Optional[BudgetCheckResult],
        iterations_completed: int
    ):
        self.original_result = original_result
        self.stopped_early = stopped_early
        self.stop_reason = stop_reason
        self.final_budget_status = final_budget_status
        self.iterations_completed = iterations_completed
        
        # Copy common attributes from original result
        if original_result:
            self.success = getattr(original_result, 'success', True)
            self.best_fitness = getattr(original_result, 'best_fitness', 0.0)
            self.total_evaluations = getattr(original_result, 'total_evaluations', 0)
            self.code = getattr(original_result, 'code', None)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "success": getattr(self, 'success', True),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "iterations_completed": self.iterations_completed,
            "best_fitness": getattr(self, 'best_fitness', 0.0),
            "total_evaluations": getattr(self, 'total_evaluations', 0),
            "budget_status": {
                "can_continue": self.final_budget_status.can_continue if self.final_budget_status else None,
                "status": self.final_budget_status.status if self.final_budget_status else None,
                "percent_used": self.final_budget_status.percent_used if self.final_budget_status else None,
            } if self.final_budget_status else None
        }
