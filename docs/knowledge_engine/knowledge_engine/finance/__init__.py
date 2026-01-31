"""
Financial Evolution Bridge - LoongFlow x OpenEvolve

Combines LoongFlow's PES (Plan-Execute-Summarize) with OpenEvolve's
crisis-aware backtesting to evolve financial strategies that survive
extinction events.

Core Components:
- FinancialEvolutionAgent: Main orchestrator
- CrisisAwareFitness: Fitness function with crisis-aware scoring
- SurvivorshipBacktester: Backtester avoiding survivorship bias
- FinancialEvolutionMemory: Hybrid memory for strategy evolution
"""

from .financial_evolution_agent import (
    FinancialEvolutionAgent,
    FinancialEvolutionResult,
    GenerationPlan,
    GenerationExecution,
    GenerationSummary
)

from .financial_memory import (
    FinancialEvolutionMemory,
    CrisisLesson,
    StrategyFailure,
    MarketConditions
)

from .crisis_aware_fitness import (
    CrisisAwareFitness,
    FitnessScore
)

from .survivorship_backtester import (
    SurvivorshipBacktester,
    BacktestResult
)

from .schemas import (
    Strategy,
    DelistingEvent
)

__all__ = [
    # Main Agent
    "FinancialEvolutionAgent",
    "FinancialEvolutionResult",

    # Generation Types
    "GenerationPlan",
    "GenerationExecution",
    "GenerationSummary",

    # Memory
    "FinancialEvolutionMemory",
    "CrisisLesson",
    "StrategyFailure",
    "MarketConditions",

    # Fitness
    "CrisisAwareFitness",
    "FitnessScore",

    # Backtesting
    "SurvivorshipBacktester",
    "BacktestResult",

    # Schemas
    "Strategy",
    "DelistingEvent"
]

__version__ = "1.0.0"
__author__ = "OpenEvolve Finance Team"
