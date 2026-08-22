"""
Financial Evolution Agent.

The LoongFlow x OpenEvolve bridge: runs a crisis-aware evolutionary search over
financial strategies, backtests each candidate with the
:class:`SurvivorshipBacktester`, scores it with :class:`CrisisAwareFitness`, and
accumulates learned :class:`CrisisLesson` objects into the hybrid
:class:`FinancialMemory`.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .crisis_aware_fitness import CrisisAwareFitness
from .financial_memory import FinancialMemory
from .schemas import (
    CrisisLesson,
    CrisisType,
    EvolutionBudget,
    EvolutionObjective,
    EvolutionResult,
    FitnessScore,
    MarketConditions,
    Strategy,
    StrategyType,
    BacktestResult,
)
from .survivorship_backtester import SurvivorshipBacktester

_STRATEGY_FAMILIES = [
    StrategyType.MOMENTUM,
    StrategyType.MEAN_REVERSION,
    StrategyType.VALUE,
    StrategyType.ARBITRAGE,
    StrategyType.FACTOR_COMBINATION,
]

_COST_PER_EVALUATION = 0.05


class FinancialEvolutionAgent:
    """Evolves crisis-surviving financial strategies."""

    def __init__(
        self,
        memory: Optional[FinancialMemory] = None,
        config: Optional[Dict[str, Any]] = None,
        backtester: Optional[SurvivorshipBacktester] = None,
        include_delisted: bool = True,
    ):
        self.memory = memory or FinancialMemory()
        self.config = config or {}
        self.backtester = backtester or SurvivorshipBacktester(include_delisted=include_delisted)
        self._last_crisis_periods: List[Tuple[str, str, CrisisType]] = []

    @property
    def fitness(self) -> CrisisAwareFitness:
        return CrisisAwareFitness(
            crisis_periods=self._last_crisis_periods,
            memory=self.memory,
            config=self.config,
        )

    def _default_objective(self) -> EvolutionObjective:
        return EvolutionObjective(
            universe="survivorship_free_equities",
            crisis_periods=[
                ("2000-01-01", "2002-12-31", CrisisType.DOTCOM),
                ("2007-09-01", "2009-03-31", CrisisType.GFC),
                ("2020-02-01", "2020-04-30", CrisisType.COVID),
                ("2022-01-01", "2022-12-31", CrisisType.INFLATION),
            ],
            survival_constraints={"max_drawdown": 0.30, "min_equity_final": 1.0},
        )

    async def evolve_strategies(
        self,
        objective: Optional[EvolutionObjective] = None,
        budget: Optional[EvolutionBudget] = None,
    ) -> EvolutionResult:
        objective = objective or self._default_objective()
        budget = budget or EvolutionBudget(iterations=10, cost_cap=20.0, strategies_per_iteration=8)
        self._last_crisis_periods = objective.crisis_periods

        rng = random.Random(1234)
        population: List[Tuple[Strategy, FitnessScore, BacktestResult]] = []
        cost = 0.0
        lessons: List[CrisisLesson] = []
        generation = 0

        for it in range(max(1, budget.iterations)):
            generation = it + 1
            # Cull population to top strategies_per_iteration survivors.
            if population:
                population.sort(key=lambda t: t[1].total_score, reverse=True)
                population = population[: max(1, budget.strategies_per_iteration // 2)]

            while len(population) < budget.strategies_per_iteration:
                parent = population[rng.randrange(len(population))][0] if population else None
                child = self._spawn_strategy(rng, parent)
                bt = await self.backtester.run(child, period=f"{objective.crisis_periods[0][0]}:{objective.crisis_periods[-1][1]}")
                conditions = MarketConditions(
                    volatility=bt.volatility,
                    resembles_crisis=self._crisis_in_period(bt),
                )
                score = self.fitness.evaluate(bt, conditions)
                cost += _COST_PER_EVALUATION
                self.memory.add_strategy_lineage(
                    parent.strategy_id if parent else None, child.strategy_id, child.strategy_type
                )
                self.memory.record_performance(child.strategy_id, score.total_score, conditions.resembles_crisis.value if conditions.resembles_crisis else None)
                population.append((child, score, bt))
                if cost >= budget.cost_cap:
                    break
            if cost >= budget.cost_cap:
                break

        population.sort(key=lambda t: t[1].total_score, reverse=True)
        best = [t[0] for t in population[:5]]
        best_score = population[0][1].total_score if population else 0.0

        # Derive crisis lessons from the best / worst performers.
        for strat, score, bt in population:
            success = score.total_score > 0.0 and bt.max_drawdown <= objective.survival_constraints.get("max_drawdown", 0.30)
            lesson = CrisisLesson(
                crisis=self._crisis_in_period(bt) or CrisisType.GFC,
                strategy_type=strat.strategy_type,
                successful=success,
                lesson=(
                    f"{strat.strategy_type.value} {'survived' if success else 'failed'} "
                    f"(sharpe={bt.sharpe_ratio}, dd={bt.max_drawdown})"
                ),
                feature_importance={"volatility": round(bt.volatility, 3), "sharpe": round(bt.sharpe_ratio, 3)},
                boost_amount=round(score.total_score * 0.1, 3) if success else round(-0.5, 3),
                conditions_met={"crisis_survived": success, "volatility": bt.volatility},
            )
            self.memory.store_lesson(lesson)
            lessons.append(lesson)

        return EvolutionResult(
            best_strategies=best,
            lessons_learned=lessons,
            final_cost=round(cost, 2),
            generations=generation,
            total_strategies=len(population),
            best_score=round(best_score, 4),
        )

    async def evaluate_strategy(
        self, strategy: Strategy, conditions: Optional[MarketConditions] = None
    ) -> Tuple[BacktestResult, FitnessScore]:
        bt = await self.backtester.run(strategy)
        cond = conditions or MarketConditions(volatility=bt.volatility)
        score = self.fitness.evaluate(bt, cond)
        return bt, score

    # -- internals --------------------------------------------------------
    def _spawn_strategy(
        self, rng: random.Random, parent: Optional[Strategy]
    ) -> Strategy:
        if parent is None:
            stype = rng.choice(_STRATEGY_FAMILIES)
        else:
            stype = parent.strategy_type
        sid = f"strat_{rng.randint(100000, 999999)}"
        base_params = dict(parent.parameters) if parent else {}
        params = {
            "lookback": int(base_params.get("lookback", rng.randint(3, 24)) + rng.randint(-2, 2)),
            "alpha": round(float(base_params.get("alpha", rng.uniform(0.001, 0.05))) + rng.uniform(-0.005, 0.005), 4),
            "beta": round(rng.uniform(0.5, 1.5), 3),
        }
        return Strategy(
            strategy_id=sid,
            strategy_type=stype,
            parameters={k: max(1, int(v)) if k == "lookback" else v for k, v in params.items()},
            description=f"{stype.value} strategy {sid}",
        )

    def _crisis_in_period(self, bt: BacktestResult) -> Optional[CrisisType]:
        # Map a severe drawdown to the most recent configured crisis type.
        if bt.max_drawdown >= 0.30:
            return CrisisType.GFC
        if bt.max_drawdown >= 0.20:
            return CrisisType.DOTCOM
        if bt.volatility >= 0.5:
            return CrisisType.COVID
        return None
