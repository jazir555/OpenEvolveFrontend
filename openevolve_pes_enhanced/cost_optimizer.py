"""Cost optimization components - extracted from LoongFlow PES.

This module provides cost-aware optimization that can wrap around
the existing openevolve_agnostic_pes without modifying it.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics tracking - from LoongFlow."""
    
    total_evaluations: int = 0
    baseline_evaluations: int = 0
    evaluations_saved: int = 0
    efficiency_gain: float = 0.0  # 0.0 to 1.0 (60% = 0.6)
    
    time_saved_ms: int = 0
    cost_saved_usd: float = 0.0
    
    iterations_to_best: int = 0
    convergence_rate: float = 0.0
    
    def calculate_efficiency_gain(self) -> float:
        """Calculate efficiency gain percentage."""
        if self.baseline_evaluations > 0:
            self.efficiency_gain = (
                (self.baseline_evaluations - self.total_evaluations) 
                / self.baseline_evaluations
            )
        return self.efficiency_gain


@dataclass  
class BudgetStatus:
    """Current budget status."""
    
    cost_used_usd: float = 0.0
    cost_remaining_usd: float = 0.0
    cost_pct_used: float = 0.0
    
    tokens_used: int = 0
    tokens_remaining: int = 0
    tokens_pct_used: float = 0.0
    
    time_used_ms: int = 0
    time_remaining_ms: int = 0
    time_pct_used: float = 0.0
    
    status: str = "ok"  # ok, warning, critical, exceeded
    should_stop: bool = False


class BudgetTracker:
    """Tracks budget usage throughout evolution - from LoongFlow."""
    
    def __init__(
        self,
        max_cost_usd: float = 10.0,
        max_tokens: int = 100000,
        max_time_ms: int = 1800000,
        warning_threshold: float = 0.70,
        critical_threshold: float = 0.90
    ):
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.max_time_ms = max_time_ms
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.start_time = time.time() * 1000
        self.tokens_used = 0
        self.cost_used = 0.0
        
        # Token pricing (can be customized)
        self.prompt_token_price = 0.00001
        self.completion_token_price = 0.00003
    
    def record_tokens(self, prompt_tokens: int, completion_tokens: int):
        """Record token usage."""
        self.tokens_used += prompt_tokens + completion_tokens
        
        prompt_cost = (prompt_tokens / 1000) * self.prompt_token_price
        completion_cost = (completion_tokens / 1000) * self.completion_token_price
        self.cost_used += prompt_cost + completion_cost
        
        logger.debug(f"Tokens: +{prompt_tokens + completion_tokens}, "
                    f"Cost: +${prompt_cost + completion_cost:.6f}")
    
    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        time_used = (time.time() * 1000) - self.start_time
        
        cost_pct = self.cost_used / self.max_cost_usd if self.max_cost_usd > 0 else 0
        tokens_pct = self.tokens_used / self.max_tokens if self.max_tokens > 0 else 0
        time_pct = time_used / self.max_time_ms if self.max_time_ms > 0 else 0
        
        # Determine status
        max_pct = max(cost_pct, tokens_pct, time_pct)
        if max_pct >= 1.0:
            status = "exceeded"
            should_stop = True
        elif max_pct >= self.critical_threshold:
            status = "critical"
            should_stop = True
        elif max_pct >= self.warning_threshold:
            status = "warning"
            should_stop = False
        else:
            status = "ok"
            should_stop = False
        
        return BudgetStatus(
            cost_used_usd=self.cost_used,
            cost_remaining_usd=max(0, self.max_cost_usd - self.cost_used),
            cost_pct_used=cost_pct,
            tokens_used=self.tokens_used,
            tokens_remaining=max(0, self.max_tokens - self.tokens_used),
            tokens_pct_used=tokens_pct,
            time_used_ms=int(time_used),
            time_remaining_ms=max(0, int(self.max_time_ms - time_used)),
            time_pct_used=time_pct,
            status=status,
            should_stop=should_stop,
        )
    
    def estimate_remaining_evaluations(self, cost_per_eval: float = 0.001) -> int:
        """Estimate how many more evaluations we can afford."""
        status = self.get_status()
        if status.cost_remaining_usd <= 0:
            return 0
        return int(status.cost_remaining_usd / cost_per_eval)


class CostOptimizer:
    """Main cost optimizer - wraps around existing PES."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.budget_tracker = None
        self.efficiency_metrics = EfficiencyMetrics()
        
    def initialize_budget(
        self,
        max_cost_usd: float = 10.0,
        max_tokens: int = 100000,
        max_time_ms: int = 1800000
    ):
        """Initialize budget tracking for a run."""
        self.budget_tracker = BudgetTracker(
            max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
            max_time_ms=max_time_ms
        )
        logger.info(f"Budget initialized: ${max_cost_usd:.2f}, "
                   f"{max_tokens} tokens, {max_time_ms/1000:.0f}s")
    
    def should_continue(self) -> Tuple[bool, str]:
        """Check if we should continue evolution based on budget.
        
        Returns:
            (should_continue, reason)
        """
        if not self.budget_tracker:
            return True, "No budget tracking"
        
        status = self.budget_tracker.get_status()
        
        if status.should_stop:
            return False, f"Budget {status.status}: cost={status.cost_pct_used:.1%}, " \
                         f"tokens={status.tokens_pct_used:.1%}, time={status.time_pct_used:.1%}"
        
        return True, f"Budget ok: cost={status.cost_pct_used:.1%}"
    
    def adapt_parameters_for_budget(
        self,
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt evolution parameters based on remaining budget.
        
        This is where LoongFlow's cost awareness comes in - we dynamically
        adjust parameters when budget is tight.
        """
        if not self.budget_tracker:
            return current_params
        
        status = self.budget_tracker.get_status()
        adapted_params = current_params.copy()
        
        # Critical budget - reduce everything
        if status.status == "critical":
            adapted_params["max_iterations"] = min(
                adapted_params.get("max_iterations", 100),
                10  # Drastically reduce
            )
            adapted_params["population_size"] = min(
                adapted_params.get("population_size", 50),
                5  # Minimal population
            )
            logger.warning("Critical budget - drastically reducing parameters")
        
        # Warning budget - moderate reduction
        elif status.status == "warning":
            adapted_params["max_iterations"] = int(
                adapted_params.get("max_iterations", 100) * 0.7
            )
            adapted_params["population_size"] = int(
                adapted_params.get("population_size", 50) * 0.8
            )
            logger.info("Warning budget - reducing parameters by 20-30%")
        
        return adapted_params
    
    def calculate_efficiency(
        self,
        actual_evaluations: int,
        baseline_multiplier: float = 2.5
    ) -> EfficiencyMetrics:
        """Calculate efficiency metrics vs baseline."""
        self.efficiency_metrics.total_evaluations = actual_evaluations
        self.efficiency_metrics.baseline_evaluations = int(
            actual_evaluations * baseline_multiplier
        )
        self.efficiency_metrics.evaluations_saved = (
            self.efficiency_metrics.baseline_evaluations - actual_evaluations
        )
        self.efficiency_metrics.calculate_efficiency_gain()
        
        logger.info(f"Efficiency: {self.efficiency_metrics.efficiency_gain:.1%} gain, "
                   f"{self.efficiency_metrics.evaluations_saved} evals saved")
        
        return self.efficiency_metrics


class CostAwarePlanner:
    """Plans evolution with cost awareness - extracted from LoongFlow."""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def estimate_cost(
        self,
        iterations: int,
        population_size: int,
        avg_tokens_per_eval: int = 500
    ) -> Dict[str, float]:
        """Estimate cost for an evolution run.
        
        This is the LoongFlow-style cost estimation that was missing
        from OpenEvolve.
        """
        total_evals = iterations * population_size
        total_tokens = total_evals * avg_tokens_per_eval
        
        # Assume 70% prompt, 30% completion split
        prompt_tokens = int(total_tokens * 0.7)
        completion_tokens = int(total_tokens * 0.3)
        
        prompt_cost = (prompt_tokens / 1000) * 0.00001
        completion_cost = (completion_tokens / 1000) * 0.00003
        
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_cost_usd": prompt_cost,
            "completion_cost_usd": completion_cost,
            "total_cost_usd": prompt_cost + completion_cost,
            "total_evaluations": total_evals,
        }
    
    def recommend_strategy_for_budget(
        self,
        max_cost_usd: float,
        problem_complexity: str = "medium"
    ) -> Dict[str, Any]:
        """Recommend evolution strategy based on budget constraints.
        
        This addresses the gap where OpenEvolve had 272 parameters
        but no guidance on how to set them based on budget.
        """
        complexity_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "very_high": 4.0
        }
        
        multiplier = complexity_multipliers.get(problem_complexity, 1.0)
        
        # Budget tiers
        if max_cost_usd < 1.0:
            # Very tight budget - minimal approach
            return {
                "strategy": "minimal",
                "iterations": int(10 * multiplier),
                "population_size": max(5, int(5 * multiplier)),
                "early_stopping": True,
                "use_cheap_model": True,
            }
        elif max_cost_usd < 5.0:
            # Moderate budget
            return {
                "strategy": "standard",
                "iterations": int(50 * multiplier),
                "population_size": max(10, int(20 * multiplier)),
                "early_stopping": True,
                "use_cheap_model": False,
            }
        else:
            # Generous budget
            return {
                "strategy": "thorough",
                "iterations": int(100 * multiplier),
                "population_size": max(20, int(50 * multiplier)),
                "early_stopping": True,
                "use_cheap_model": False,
            }
