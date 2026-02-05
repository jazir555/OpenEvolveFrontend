"""
Budget monitoring system for cost-aware evolution.

Tracks resource consumption and provides early warnings
when budgets are approaching limits.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import logging


logger = logging.getLogger(__name__)


class BudgetAlertLevel(Enum):
    """Alert levels for budget monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    tokens_input: int = 0
    tokens_output: int = 0
    api_calls: int = 0
    verification_operations: int = 0
    compute_time_seconds: float = 0.0
    
    def add(self, other: "CostBreakdown"):
        """Add another cost breakdown."""
        self.tokens_input += other.tokens_input
        self.tokens_output += other.tokens_output
        self.api_calls += other.api_calls
        self.verification_operations += other.verification_operations
        self.compute_time_seconds += other.compute_time_seconds


@dataclass
class CostSummary:
    """Summary of costs consumed."""
    total_cost: float = 0.0  # USD
    tokens: int = 0
    api_calls: int = 0
    time_seconds: float = 0.0
    
    def add(self, cost: CostBreakdown, token_cost_rate: float = 0.02):
        """Add a cost breakdown to summary."""
        self.tokens += cost.tokens_input + cost.tokens_output
        self.api_calls += cost.api_calls
        self.time_seconds += cost.compute_time_seconds
        
        # Estimate cost
        token_cost = (cost.tokens_input / 1000 * 0.01) + (cost.tokens_output / 1000 * 0.03)
        api_cost = cost.api_calls * 0.001
        self.total_cost += token_cost + api_cost


@dataclass
class BudgetAlert:
    """Budget alert notification."""
    level: BudgetAlertLevel
    message: str
    budget_type: str
    consumed_percentage: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostBudget:
    """Budget for a specific activity."""
    max_cost: float
    max_tokens: int
    max_api_calls: int
    max_time_seconds: float


@dataclass
class BudgetAllocation:
    """Budget allocation across activities."""
    planning_budget: CostBudget
    evolution_budget: CostBudget
    verification_budget: CostBudget
    contingency_reserve: float
    
    @property
    def total_budget(self) -> float:
        """Calculate total budget."""
        return (
            self.planning_budget.max_cost +
            self.evolution_budget.max_cost +
            self.verification_budget.max_cost
        ) * (1 + self.contingency_reserve)
    
    @property
    def token_budget(self) -> int:
        """Calculate total token budget."""
        return (
            self.planning_budget.max_tokens +
            self.evolution_budget.max_tokens +
            self.verification_budget.max_tokens
        )
    
    @property
    def api_call_budget(self) -> int:
        """Calculate total API call budget."""
        return (
            self.planning_budget.max_api_calls +
            self.evolution_budget.max_api_calls +
            self.verification_budget.max_api_calls
        )
    
    @property
    def time_budget_seconds(self) -> float:
        """Calculate total time budget."""
        return (
            self.planning_budget.max_time_seconds +
            self.evolution_budget.max_time_seconds +
            self.verification_budget.max_time_seconds
        )


@dataclass
class OptimizationSuggestion:
    """Suggestion for cost optimization."""
    type: str
    description: str
    estimated_savings: float
    impact: str  # "low", "medium", "high"


class BudgetMonitor:
    """
    Monitors budget consumption during evolution.
    
    Provides:
    - Real-time cost tracking
    - Threshold alerts
    - Optimization suggestions
    - Early stopping triggers
    """
    
    # Alert thresholds
    WARNING_THRESHOLD = 0.70  # 70%
    CRITICAL_THRESHOLD = 0.90  # 90%
    
    def __init__(self, allocation: BudgetAllocation):
        self.allocation = allocation
        self.consumed = CostSummary()
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.alerts: List[BudgetAlert] = []
        self.warnings_triggered = set()
    
    def record_spending(self, cost: CostBreakdown):
        """Record a cost occurrence."""
        self.consumed.add(cost)
        
        # Check thresholds
        self._check_thresholds()
    
    def _check_thresholds(self):
        """Check if any budget thresholds are crossed."""
        current_time = time.time()
        
        # Check cost budget
        cost_ratio = self.consumed.total_cost / self.allocation.total_budget
        self._maybe_alert("cost", cost_ratio)
        
        # Check token budget
        token_ratio = self.consumed.tokens / self.allocation.token_budget if self.allocation.token_budget > 0 else 0
        self._maybe_alert("tokens", token_ratio)
        
        # Check time budget
        elapsed = current_time - self.start_time
        time_ratio = elapsed / self.allocation.time_budget_seconds if self.allocation.time_budget_seconds > 0 else 0
        self._maybe_alert("time", time_ratio)
        
        self.last_check_time = current_time
    
    def _maybe_alert(self, budget_type: str, ratio: float):
        """Generate alert if threshold crossed."""
        alert_key = f"{budget_type}_{int(ratio * 10)}"  # Bucket by 10%
        
        if alert_key in self.warnings_triggered:
            return
        
        if ratio >= self.CRITICAL_THRESHOLD:
            alert = BudgetAlert(
                level=BudgetAlertLevel.CRITICAL,
                message=f"{budget_type.capitalize()} budget at {ratio:.1%}! Consider stopping.",
                budget_type=budget_type,
                consumed_percentage=ratio
            )
            self.alerts.append(alert)
            self.warnings_triggered.add(alert_key)
            logger.warning(alert.message)
            
        elif ratio >= self.WARNING_THRESHOLD:
            alert = BudgetAlert(
                level=BudgetAlertLevel.WARNING,
                message=f"{budget_type.capitalize()} budget at {ratio:.1%}. Monitor closely.",
                budget_type=budget_type,
                consumed_percentage=ratio
            )
            self.alerts.append(alert)
            self.warnings_triggered.add(alert_key)
            logger.info(alert.message)
    
    def can_continue(self) -> bool:
        """Check if evolution can continue within budget."""
        # Check cost budget
        if self.consumed.total_cost >= self.allocation.total_budget:
            logger.warning("Cost budget exhausted")
            return False
        
        # Check token budget
        if self.allocation.token_budget > 0 and self.consumed.tokens >= self.allocation.token_budget:
            logger.warning("Token budget exhausted")
            return False
        
        # Check time budget
        elapsed = time.time() - self.start_time
        if self.allocation.time_budget_seconds > 0 and elapsed >= self.allocation.time_budget_seconds:
            logger.warning("Time budget exhausted")
            return False
        
        # Check API budget
        if self.allocation.api_call_budget > 0 and self.consumed.api_calls >= self.allocation.api_call_budget:
            logger.warning("API call budget exhausted")
            return False
        
        return True
    
    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining budget for each category."""
        elapsed = time.time() - self.start_time
        
        return {
            "cost_usd": max(0, self.allocation.total_budget - self.consumed.total_cost),
            "tokens": max(0, self.allocation.token_budget - self.consumed.tokens),
            "api_calls": max(0, self.allocation.api_call_budget - self.consumed.api_calls),
            "time_seconds": max(0, self.allocation.time_budget_seconds - elapsed),
        }
    
    def get_consumption_rate(self) -> Dict[str, float]:
        """Get consumption rate per second."""
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            elapsed = 1
        
        return {
            "cost_per_second": self.consumed.total_cost / elapsed,
            "tokens_per_second": self.consumed.tokens / elapsed,
            "api_calls_per_second": self.consumed.api_calls / elapsed,
        }
    
    def project_final_cost(self) -> Dict[str, float]:
        """Project final cost assuming current rate continues."""
        rate = self.get_consumption_rate()
        remaining_time = self.get_remaining_budget()["time_seconds"]
        
        return {
            "projected_cost": self.consumed.total_cost + (rate["cost_per_second"] * remaining_time),
            "projected_tokens": self.consumed.tokens + (rate["tokens_per_second"] * remaining_time),
            "projected_api_calls": self.consumed.api_calls + (rate["api_calls_per_second"] * remaining_time),
        }
    
    def suggest_optimizations(self) -> List[OptimizationSuggestion]:
        """Suggest optimizations based on spending patterns."""
        suggestions = []
        
        rate = self.get_consumption_rate()
        remaining = self.get_remaining_budget()
        projection = self.project_final_cost()
        
        # If burning through tokens too fast
        if projection["projected_tokens"] > self.allocation.token_budget * 0.9:
            suggestions.append(OptimizationSuggestion(
                type="reduce_population",
                description="Reduce population size by 20% to save tokens",
                estimated_savings=self.consumed.total_cost * 0.15,
                impact="medium"
            ))
            suggestions.append(OptimizationSuggestion(
                type="early_stopping",
                description="Enable aggressive early stopping",
                estimated_savings=self.consumed.total_cost * 0.20,
                impact="high"
            ))
        
        # If API calls are expensive
        if rate["api_calls_per_second"] > 1:
            suggestions.append(OptimizationSuggestion(
                type="batch_requests",
                description="Batch multiple evaluations into single API call",
                estimated_savings=self.consumed.total_cost * 0.10,
                impact="medium"
            ))
        
        # If time is running out
        if remaining["time_seconds"] < self.allocation.time_budget_seconds * 0.2:
            suggestions.append(OptimizationSuggestion(
                type="reduce_iterations",
                description="Reduce max iterations to fit time budget",
                estimated_savings=0,  # Time savings, not cost
                impact="high"
            ))
        
        return suggestions
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete budget summary."""
        return {
            "consumed": {
                "cost_usd": round(self.consumed.total_cost, 4),
                "tokens": self.consumed.tokens,
                "api_calls": self.consumed.api_calls,
                "time_seconds": round(time.time() - self.start_time, 2),
            },
            "allocated": {
                "cost_usd": round(self.allocation.total_budget, 4),
                "tokens": self.allocation.token_budget,
                "api_calls": self.allocation.api_call_budget,
                "time_seconds": round(self.allocation.time_budget_seconds, 2),
            },
            "remaining": self.get_remaining_budget(),
            "consumption_rate": self.get_consumption_rate(),
            "projections": self.project_final_cost(),
            "alerts": [
                {"level": a.level.value, "message": a.message, "type": a.budget_type}
                for a in self.alerts
            ],
            "suggestions": [
                {"type": s.type, "description": s.description, "savings": s.estimated_savings}
                for s in self.suggest_optimizations()
            ]
        }
