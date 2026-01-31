"""
Data models for financial evolution bridge.

Provides canonical schemas for financial strategies, backtesting results,
and crisis-aware evaluation metrics.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class StrategyType(str, Enum):
    """Types of financial trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    GROWTH = "growth"
    FACTOR_COMBINATION = "factor_combination"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    CUSTOM = "custom"


class CrisisType(str, Enum):
    """Types of market crises"""
    DOTCOM = "dotcom"
    GFC = "gfc"  # Global Financial Crisis
    COVID = "covid"
    INFLATION = "inflation"
    CUSTOM = "custom"


class Strategy(BaseModel):
    """Financial trading strategy definition"""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    strategy_type: StrategyType = Field(..., description="Type of strategy")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters (lookback windows, thresholds, etc.)"
    )
    description: str = Field(..., description="Human-readable description")
    entry_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions for entering positions"
    )
    exit_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions for exiting positions"
    )
    risk_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk management constraints"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp (UTC)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DelistingEvent(BaseModel):
    """Record of a security delisting"""
    ticker: str = Field(..., description="Security ticker")
    delisting_date: datetime = Field(..., description="Delisting timestamp (UTC)")
    reason: str = Field(..., description="Reason for delisting")
    last_price: float = Field(..., description="Last trading price")
    recovery_rate: Optional[float] = Field(
        default=None,
        description="Recovery rate if applicable"
    )
    impact: float = Field(
        ...,
        description="Impact on portfolio (negative = loss)"
    )


class BacktestResult(BaseModel):
    """Results from backtesting a strategy"""
    strategy_id: str = Field(..., description="Strategy identifier")
    returns: List[float] = Field(..., description="Time series of returns")
    drawdowns: List[float] = Field(..., description="Time series of drawdowns")
    delistings: List[DelistingEvent] = Field(
        default_factory=list,
        description="Delisting events encountered"
    )
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(
        default=None,
        description="Sortino ratio"
    )
    max_drawdown: float = Field(..., description="Maximum drawdown")
    final_wealth: float = Field(..., description="Final wealth ratio")
    volatility: float = Field(..., description="Return volatility")
    total_trades: int = Field(default=0, description="Total number of trades")
    win_rate: Optional[float] = Field(
        default=None,
        description="Percentage of winning trades"
    )
    start_date: datetime = Field(..., description="Backtest start (UTC)")
    end_date: datetime = Field(..., description="Backtest end (UTC)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FitnessScore(BaseModel):
    """Fitness score with components"""
    base_score: float = Field(..., description="Base fitness from returns")
    learned_boost: float = Field(
        default=0.0,
        description="Boost from LoongFlow-learned heuristics"
    )
    total_score: float = Field(..., description="Total fitness score")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Component scores (sharpe, max_dd, crisis_survival, etc.)"
    )

    def __lt__(self, other):
        """Enable sorting (higher score is better)"""
        return self.total_score < other.total_score


class CrisisLesson(BaseModel):
    """Lesson learned from crisis evolution"""
    lesson_id: str = Field(..., description="Unique lesson identifier")
    crisis: CrisisType = Field(..., description="Crisis type")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    successful: bool = Field(..., description="Whether strategy was successful")
    lesson: str = Field(..., description="Learned lesson (natural language)")
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Importance of features during crisis"
    )
    boost_amount: float = Field(
        default=0.0,
        description="Fitness boost for similar conditions"
    )
    conditions_met: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market conditions when lesson applies"
    )
    learned_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When lesson was learned (UTC)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def condition_matches(
        self,
        returns: List[float],
        drawdowns: List[float],
        current_volatility: float
    ) -> bool:
        """
        Check if lesson applies to current conditions.

        Args:
            returns: Recent returns
            drawdowns: Recent drawdowns
            current_volatility: Current market volatility

        Returns:
            True if lesson conditions are met
        """
        # Check volatility condition
        if "volatility_threshold" in self.conditions_met:
            vol_threshold = self.conditions_met["volatility_threshold"]
            if current_volatility < vol_threshold:
                return False

        # Check drawdown condition
        if "max_drawdown_threshold" in self.conditions_met:
            dd_threshold = self.conditions_met["max_drawdown_threshold"]
            current_max_dd = max(drawdowns) if drawdowns else 0
            if current_max_dd < dd_threshold:
                return False

        # Check trend condition
        if "trend_requirement" in self.conditions_met:
            trend_req = self.conditions_met["trend_requirement"]
            recent_returns = returns[-10:] if len(returns) >= 10 else returns
            avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0

            if trend_req == "positive" and avg_return <= 0:
                return False
            elif trend_req == "negative" and avg_return >= 0:
                return False

        return True


class StrategyFailure(BaseModel):
    """Record of strategy failure for learning"""
    strategy_id: str = Field(..., description="Failed strategy ID")
    failure_type: str = Field(..., description="Type of failure")
    crisis_context: Optional[CrisisType] = Field(
        default=None,
        description="Crisis context if applicable"
    )
    failure_reason: str = Field(..., description="Why it failed")
    metrics_at_failure: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics when failure occurred"
    )
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When failure occurred (UTC)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketConditions(BaseModel):
    """Current market conditions"""
    volatility: float = Field(..., description="Market volatility")
    trend: str = Field(..., description="Market trend (up/down/sideways)")
    resembles_crisis: Optional[CrisisType] = Field(
        default=None,
        description="Crisis type if conditions resemble a crisis"
    )
    vix: Optional[float] = Field(default=None, description="VIX level if available")
    credit_spread: Optional[float] = Field(
        default=None,
        description="Credit spread indicator"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Observation timestamp (UTC)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GenerationPlan(BaseModel):
    """Plan for strategy generation (LoongFlow PLAN phase)"""
    iteration: int = Field(..., description="Current iteration number")
    hypotheses: List[str] = Field(
        ...,
        description="Generated hypotheses for strategies"
    )
    parameter_ranges: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Parameter ranges to explore"
    )
    estimated_cost: float = Field(..., description="Estimated LLM cost")
    rationale: str = Field(..., description="Rationale for plan")


class GenerationExecution(BaseModel):
    """Execution results (OpenEvolve EXECUTE phase)"""
    all_results: List[Tuple[Strategy, BacktestResult, FitnessScore]] = Field(
        ...,
        description="All strategies with results"
    )
    best_strategy: Optional[Tuple[Strategy, BacktestResult, FitnessScore]] = Field(
        default=None,
        description="Best performing strategy"
    )
    worst_strategies: List[Tuple[Strategy, BacktestResult, FitnessScore]] = Field(
        default_factory=list,
        description="Worst performing strategies (for learning)"
    )
    average_fitness: float = Field(..., description="Average fitness score")
    total_executed: int = Field(..., description="Total strategies executed")


class GenerationSummary(BaseModel):
    """Summary from generation (LoongFlow SUMMARIZE phase)"""
    lessons: List[CrisisLesson] = Field(
        ...,
        description="Lessons learned from this generation"
    )
    converged: bool = Field(..., description="Whether evolution has converged")
    total_cost: float = Field(..., description="Actual LLM cost incurred")
    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights extracted"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )


class FinancialEvolutionResult(BaseModel):
    """Complete results from financial evolution"""
    best_strategies: List[Strategy] = Field(
        ...,
        description="Best strategies found (ranked)"
    )
    lessons_learned: List[CrisisLesson] = Field(
        ...,
        description="All lessons learned"
    )
    iterations_completed: int = Field(..., description="Total iterations run")
    final_cost: float = Field(..., description="Total LLM cost incurred")
    converged: bool = Field(..., description="Whether convergence achieved")
    execution_time_seconds: float = Field(
        ...,
        description="Total execution time"
    )
    best_fitness: float = Field(..., description="Best fitness achieved")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SurvivalConstraints(BaseModel):
    """Constraints for strategy survival"""
    max_drawdown: float = Field(
        default=0.30,
        description="Maximum acceptable drawdown (30% = 0.30)"
    )
    min_equity_final: float = Field(
        default=1.0,
        description="Minimum final wealth ratio (1.0 = break even)"
    )
    delisting_penalty: float = Field(
        default=-1000.0,
        description="Penalty per delisting event"
    )
    min_sharpe_ratio: Optional[float] = Field(
        default=None,
        description="Minimum Sharpe ratio required"
    )
    crisis_survival_required: bool = Field(
        default=True,
        description="Must survive all crisis periods"
    )


class EvolutionObjective(BaseModel):
    """Objective for financial evolution"""
    universe: str = Field(
        ...,
        description="Asset universe (e.g., 'survivorship_free_equities_2000_2026')"
    )
    crisis_periods: List[Tuple[str, str, CrisisType]] = Field(
        ...,
        description="Crisis periods (start, end, type)"
    )
    survival_constraints: SurvivalConstraints = Field(
        default_factory=SurvivalConstraints,
        description="Survival constraints"
    )
    target_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Target metrics to optimize"
    )


class EvolutionBudget(BaseModel):
    """Budget constraints for evolution"""
    iterations: int = Field(default=500, description="Maximum generations")
    cost_cap: float = Field(default=500, description="Maximum LLM cost ($)")
    strategies_per_iteration: int = Field(
        default=50,
        description="Strategies to generate per iteration"
    )
    time_limit_seconds: Optional[float] = Field(
        default=None,
        description="Maximum execution time"
    )
