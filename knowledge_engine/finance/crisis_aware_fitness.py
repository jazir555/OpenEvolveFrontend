"""
Crisis-Aware Fitness.

Combines static survivorship-aware backtest metrics with learned heuristics
retrieved from the :class:`FinancialMemory`. Strategies that survive historical
crises are rewarded; delistings and deep drawdowns are penalised.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .financial_memory import FinancialMemory
from .schemas import BacktestResult, CrisisType, FitnessScore, MarketConditions


_DEFAULT_WEIGHTS = {
    "sharpe_ratio": 2.0,
    "max_drawdown": -5.0,
    "final_wealth": 3.0,
    "crisis_survival": 5.0,
    "delisting_penalty": -10.0,
    "volatility_penalty": -1.0,
}


class CrisisAwareFitness:
    """Fitness function that learns from historical crises."""

    def __init__(
        self,
        crisis_periods: Optional[List[Tuple[str, str, CrisisType]]] = None,
        memory: Optional[FinancialMemory] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.crisis_periods = crisis_periods or []
        self.memory = memory
        self.config = config or {}
        self.weights = dict(_DEFAULT_WEIGHTS)
        self.weights.update({
            k: self.config.get(v, self.weights[k])
            for k, v in {
                "sharpe_ratio": "sharpe_weight",
                "max_drawdown": "drawdown_weight",
                "final_wealth": "wealth_weight",
                "crisis_survival": "crisis_weight",
                "delisting_penalty": "delisting_weight",
                "volatility_penalty": "volatility_weight",
            }.items()
        })
        self.crisis_multipliers = {
            CrisisType.DOTCOM: self.config.get("dotcom_multiplier", 1.5),
            CrisisType.GFC: self.config.get("gfc_multiplier", 2.0),
            CrisisType.COVID: self.config.get("covid_multiplier", 1.8),
            CrisisType.INFLATION: self.config.get("inflation_multiplier", 1.3),
        }

    def evaluate(
        self,
        backtest_result: BacktestResult,
        current_conditions: Optional[MarketConditions] = None,
    ) -> FitnessScore:
        base_score = self._calculate_base_fitness(backtest_result)
        boost = self._calculate_learned_boost(backtest_result, current_conditions)
        components = self._component_scores(backtest_result)
        components["learned_boost"] = round(boost, 4)
        return FitnessScore(
            base_score=round(base_score, 4),
            learned_boost=round(boost, 4),
            total_score=round(base_score + boost, 4),
            components=components,
        )

    # -- internals --------------------------------------------------------
    def _calculate_base_fitness(self, result: BacktestResult) -> float:
        score = 0.0
        score += self.weights["sharpe_ratio"] * result.sharpe_ratio
        score += self.weights["max_drawdown"] * result.max_drawdown
        score += self.weights["final_wealth"] * (result.final_wealth - 1.0)
        score += self.weights["volatility_penalty"] * result.volatility
        crisis_penalty = self.weights["crisis_survival"] * max(0.0, 0.30 - result.max_drawdown)
        score += crisis_penalty
        if result.delistings:
            impact = sum(abs(d.impact) for d in result.delistings)
            score += self.weights["delisting_penalty"] * min(impact, 1.0)
        return score

    def _calculate_learned_boost(
        self,
        result: BacktestResult,
        conditions: Optional[MarketConditions],
    ) -> float:
        if self.memory is None or conditions is None:
            return 0.0
        lessons = self.memory.get_relevant_lessons(conditions)
        if not lessons:
            return 0.0
        # Average the boost of successful lessons; penalise failure lessons.
        boost = 0.0
        for lesson in lessons:
            if lesson.successful:
                boost += lesson.boost_amount
            else:
                boost += lesson.boost_amount  # typically negative
        return boost / len(lessons)

    def _component_scores(self, result: BacktestResult) -> Dict[str, float]:
        return {
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "final_wealth": round(result.final_wealth, 4),
            "volatility": round(result.volatility, 4),
            "delistings": len(result.delistings),
        }
