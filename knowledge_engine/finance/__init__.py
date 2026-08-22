"""Knowledge Engine Finance module.

Production bridge between high-level planning and crisis-aware financial
strategy evolution. See ``README.md`` for the documented API.
"""
from __future__ import annotations

from .schemas import (
    BacktestResult,
    CrisisLesson,
    CrisisType,
    DelistingEvent,
    EvolutionBudget,
    EvolutionObjective,
    EvolutionResult,
    FitnessScore,
    MarketConditions,
    Strategy,
    StrategyFailure,
    StrategyType,
)
from .financial_memory import FinancialMemory
from .survivorship_backtester import SurvivorshipBacktester
from .crisis_aware_fitness import CrisisAwareFitness
from .financial_evolution_agent import FinancialEvolutionAgent

# Legacy aliases retained for backwards compatibility.
from .schemas import FinancialConfig, Portfolio


class FinancialEvolutionEngine:
    """Legacy facade (thin wrapper over :class:`FinancialEvolutionAgent`)."""

    def __init__(self, **kwargs):
        self.agent = FinancialEvolutionAgent(**kwargs)

    async def evolve(self, objective=None, budget=None):
        return await self.agent.evolve_strategies(objective, budget)


class FinancialOptimizer:
    """Legacy optimizer stub."""


__all__ = [
    "StrategyType",
    "CrisisType",
    "Strategy",
    "DelistingEvent",
    "BacktestResult",
    "FitnessScore",
    "MarketConditions",
    "CrisisLesson",
    "StrategyFailure",
    "EvolutionObjective",
    "EvolutionBudget",
    "EvolutionResult",
    "FinancialMemory",
    "SurvivorshipBacktester",
    "CrisisAwareFitness",
    "FinancialEvolutionAgent",
    "FinancialEvolutionEngine",
    "FinancialOptimizer",
    "FinancialConfig",
    "Portfolio",
]
