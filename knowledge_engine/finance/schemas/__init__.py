"""Knowledge Engine Finance Schemas.

Canonical data models for the financial evolution bridge:
strategy definitions, backtest results, crisis-aware fitness scores,
market conditions, crisis lessons, and the evolution control structures
(objectives / budgets / results).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class StrategyType(str, Enum):
    """Types of financial trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    GROWTH = "growth"
    FACTOR_COMBINATION = "factor_combination"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    CUSTOM = "custom"


class CrisisType(str, Enum):
    """Types of market crises used for crisis-aware evaluation."""
    DOTCOM = "dotcom"
    GFC = "gfc"  # Global Financial Crisis
    COVID = "covid"
    INFLATION = "inflation"
    CUSTOM = "custom"


@dataclass
class Strategy:
    """A financial trading strategy definition."""
    strategy_id: str
    strategy_type: StrategyType
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    entry_conditions: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)
    risk_constraints: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DelistingEvent:
    """Record of a security delisting encountered during backtesting."""
    ticker: str
    delisting_date: datetime
    reason: str
    last_price: float
    recovery_rate: Optional[float] = None
    impact: float = 0.0


@dataclass
class BacktestResult:
    """Results from backtesting a strategy (survivorship-aware)."""
    strategy_id: str
    returns: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    delistings: List[DelistingEvent] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    sortino_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    final_wealth: float = 1.0
    volatility: float = 0.0
    total_trades: int = 0
    win_rate: Optional[float] = None
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FitnessScore:
    """A crisis-aware fitness score with transparent components."""
    base_score: float = 0.0
    learned_boost: float = 0.0
    total_score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Current market conditions used to retrieve relevant lessons."""
    volatility: float = 0.2
    resembles_crisis: Optional[CrisisType] = None
    trend: str = "neutral"  # "up" | "down" | "neutral"
    regime: str = "normal"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrisisLesson:
    """A lesson learned about a strategy type during a crisis."""
    crisis: CrisisType
    strategy_type: StrategyType
    successful: bool
    lesson: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    boost_amount: float = 0.0
    conditions_met: Dict[str, Any] = field(default_factory=dict)
    lesson_id: str = field(default_factory=lambda: "lesson_" + uuid.uuid4().hex[:12])
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyFailure:
    """A recorded strategy failure used for learning."""
    failure_type: str
    strategy_type: StrategyType
    detail: str = ""
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvolutionObjective:
    """What the evolution run is trying to achieve."""
    universe: str = "survivorship_free_equities"
    crisis_periods: List[Tuple[str, str, CrisisType]] = field(default_factory=list)
    survival_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionBudget:
    """Computational / cost budget for an evolution run."""
    iterations: int = 50
    cost_cap: float = 100.0
    strategies_per_iteration: int = 20


@dataclass
class EvolutionResult:
    """Outcome of an evolution run."""
    best_strategies: List[Strategy] = field(default_factory=list)
    lessons_learned: List[CrisisLesson] = field(default_factory=list)
    final_cost: float = 0.0
    generations: int = 0
    total_strategies: int = 0
    best_score: float = 0.0


# ---------------------------------------------------------------------------
# Legacy schemas retained for backwards compatibility.
# ---------------------------------------------------------------------------
@dataclass
class FinancialConfig:
    """Financial configuration."""
    risk_tolerance: float = 0.5
    return_target: float = 0.1
    constraints: Dict[str, Any] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


@dataclass
class Portfolio:
    """Portfolio."""
    assets: List[str] = None
    weights: List[float] = None

    def __post_init__(self):
        if self.assets is None:
            self.assets = []
        if self.weights is None:
            self.weights = []
