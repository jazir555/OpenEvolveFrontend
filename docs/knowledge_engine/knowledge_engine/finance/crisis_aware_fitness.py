"""
CrisisAwareFitness - Enhanced fitness with LoongFlow-learned heuristics

Combines:
- Static rules (survivorship-aware backtesting metrics)
- Dynamic heuristics (LoongFlow-learned from historical crises)
- Crisis-specific evaluations (performance in different crisis types)
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from datetime import datetime

from .schemas import (
    FitnessScore,
    CrisisLesson,
    MarketConditions,
    CrisisType,
    BacktestResult
)
from .financial_memory import FinancialEvolutionMemory


class CrisisAwareFitness:
    """
    Fitness function that learns from historical crises.

    Combines base metrics (Sharpe, drawdown, etc.) with LoongFlow-learned
    heuristics to score strategies based on crisis-survival capability.
    """

    def __init__(
        self,
        crisis_periods: List[Tuple[str, str, CrisisType]],
        memory: FinancialEvolutionMemory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize crisis-aware fitness.

        Args:
            crisis_periods: List of (start_date, end_date, crisis_type)
            memory: FinancialEvolutionMemory for learned lessons
            config: Optional configuration
        """
        self.crisis_periods = crisis_periods
        self.memory = memory
        self.config = config or {}

        # Fitness weights
        self.weights = {
            "sharpe_ratio": self.config.get("sharpe_weight", 2.0),
            "max_drawdown": self.config.get("drawdown_weight", -5.0),
            "final_wealth": self.config.get("wealth_weight", 3.0),
            "crisis_survival": self.config.get("crisis_weight", 5.0),
            "delisting_penalty": self.config.get("delisting_weight", -10.0),
            "volatility_penalty": self.config.get("volatility_weight", -1.0),
            "consistency": self.config.get("consistency_weight", 1.0)
        }

        # Crisis-specific adjustments
        self.crisis_multipliers = {
            CrisisType.DOTCOM: self.config.get("dotcom_multiplier", 1.5),
            CrisisType.GFC: self.config.get("gfc_multiplier", 2.0),
            CrisisType.COVID: self.config.get("covid_multiplier", 1.8),
            CrisisType.INFLATION: self.config.get("inflation_multiplier", 1.3)
        }

    def evaluate(
        self,
        backtest_result: BacktestResult,
        current_conditions: Optional[MarketConditions] = None
    ) -> FitnessScore:
        """
        Evaluate strategy with crisis-aware fitness.

        Args:
            backtest_result: Backtest results
            current_conditions: Current market conditions (optional)

        Returns:
            FitnessScore with base, boost, and total
        """
        # Calculate base fitness from static rules
        base_score = self._calculate_base_fitness(backtest_result)

        # Calculate learned boost from LoongFlow lessons
        boost = self._calculate_learned_boost(
            backtest_result,
            current_conditions
        )

        # Total score
        total_score = base_score + boost

        # Component scores for transparency
        components = self._get_component_scores(backtest_result)

        return FitnessScore(
            base_score=base_score,
            learned_boost=boost,
            total_score=total_score,
            components=components
        )

    def _calculate_base_fitness(self, result: BacktestResult) -> float:
        """
        Calculate base fitness from static metrics.

        Args:
            result: Backtest result

        Returns:
            Base fitness score
        """
        score = 0.0

        # Sharpe ratio (risk-adjusted returns)
        sharpe = result.sharpe_ratio
        score += self.weights["sharpe_ratio"] * sharpe

        # Max drawdown (penalty for large losses)
        max_dd = result.max_drawdown
        score += self.weights["max_drawdown"] * max_dd

        # Final wealth (absolute performance)
        final_wealth = result.final_wealth
        score += self.weights["final_wealth"] * final_wealth

        # Delisting penalty (survivorship bias awareness)
        delisting_count = len(result.delistings)
        score += self.weights["delisting_penalty"] * delisting_count

        # Volatility penalty (prefer stable strategies)
        if result.volatility > 0:
            vol_penalty = min(result.volatility, 1.0)
            score += self.weights["volatility_penalty"] * vol_penalty

        # Consistency bonus (returns consistency)
        if len(result.returns) > 1:
            returns_std = np.std(result.returns)
            consistency_bonus = 1.0 / (1.0 + returns_std)
            score += self.weights["consistency"] * consistency_bonus

        # Crisis survival bonus
        crisis_score = self._calculate_crisis_survival_score(result)
        score += self.weights["crisis_survival"] * crisis_score

        return score

    def _calculate_crisis_survival_score(self, result: BacktestResult) -> float:
        """
        Calculate crisis-specific survival score.

        Args:
            result: Backtest result

        Returns:
            Crisis survival score (0-1)
        """
        if not result.returns:
            return 0.0

        # Get crisis period indices
        crisis_scores = []

        for start_str, end_str, crisis_type in self.crisis_periods:
            try:
                start_date = datetime.fromisoformat(start_str)
                end_date = datetime.fromisoformat(end_str)

                # Calculate returns during crisis
                crisis_returns = self._extract_period_returns(
                    result,
                    start_date,
                    end_date
                )

                if crisis_returns:
                    # Calculate performance during crisis
                    avg_return = np.mean(crisis_returns)
                    max_dd_crisis = self._calculate_max_drawdown(crisis_returns)

                    # Score based on crisis severity multiplier
                    multiplier = self.crisis_multipliers.get(crisis_type, 1.0)

                    # Positive returns in crisis = good
                    # Negative returns in crisis = bad (scaled by multiplier)
                    crisis_score = (avg_return * multiplier) - (max_dd_crisis * multiplier)
                    crisis_scores.append(max(crisis_score, -1.0))

            except (ValueError, KeyError):
                continue

        # Return average crisis score
        if not crisis_scores:
            return 0.0

        return np.mean(crisis_scores)

    def _calculate_learned_boost(
        self,
        result: BacktestResult,
        current_conditions: Optional[MarketConditions]
    ) -> float:
        """
        Apply LoongFlow-learned heuristics.

        Args:
            result: Backtest result
            current_conditions: Current market conditions

        Returns:
            Learned boost to fitness
        """
        boost = 0.0

        if not current_conditions:
            return boost

        # Get relevant lessons
        lessons = self.memory.get_relevant_lessons(current_conditions)

        # Calculate current conditions
        current_volatility = result.volatility
        current_max_dd = result.max_drawdown
        recent_returns = result.returns[-10:] if len(result.returns) >= 10 else result.returns

        # Apply matching lessons
        for lesson in lessons:
            if lesson.condition_matches(recent_returns, result.drawdowns, current_volatility):
                boost += lesson.boost_amount

        # Feature importance boost
        for feature, importance in self._extract_feature_importance(result).items():
            avg_importance = self.memory.get_average_feature_importance(
                feature,
                current_conditions.resembles_crisis
            )
            if avg_importance > 0.7:  # High importance feature
                boost += 0.1 * importance

        return boost

    def _get_component_scores(self, result: BacktestResult) -> Dict[str, float]:
        """
        Get individual component scores for transparency.

        Args:
            result: Backtest result

        Returns:
            Dictionary of component scores
        """
        return {
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "final_wealth": result.final_wealth,
            "delisting_count": len(result.delistings),
            "volatility": result.volatility,
            "crisis_survival": self._calculate_crisis_survival_score(result),
            "consistency": 1.0 / (1.0 + np.std(result.returns)) if result.returns else 0.0,
            "sortino_ratio": result.sortino_ratio or 0.0,
            "win_rate": result.win_rate or 0.0
        }

    def _extract_period_returns(
        self,
        result: BacktestResult,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """
        Extract returns for a specific period.

        Args:
            result: Backtest result
            start_date: Period start
            end_date: Period end

        Returns:
            List of returns in period
        """
        # This is a simplified implementation
        # In production, use actual date-indexed returns
        total_days = (end_date - start_date).days
        if total_days <= 0:
            return []

        # Estimate period returns (simplified)
        start_idx = int(len(result.returns) * 0.3)  # Example: 30% through
        end_idx = int(len(result.returns) * 0.5)    # Example: 50% through

        if start_idx >= len(result.returns) or end_idx > len(result.returns):
            return []

        return result.returns[start_idx:end_idx]

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: List of returns

        Returns:
            Maximum drawdown
        """
        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _extract_feature_importance(
        self,
        result: BacktestResult
    ) -> Dict[str, float]:
        """
        Extract feature importance from backtest result.

        Args:
            result: Backtest result

        Returns:
            Dictionary of feature -> importance
        """
        # Simplified feature extraction
        # In production, analyze strategy parameters and performance

        features = {
            "sharpe_ratio": min(result.sharpe_ratio / 2.0, 1.0),
            "max_drawdown": 1.0 - min(result.max_drawdown, 1.0),
            "final_wealth": min(result.final_wealth - 1.0, 1.0),
            "volatility": 1.0 - min(result.volatility, 1.0)
        }

        return {k: max(0.0, v) for k, v in features.items()}

    def update_lesson_from_result(
        self,
        result: BacktestResult,
        crisis_type: CrisisType,
        successful: bool
    ) -> CrisisLesson:
        """
        Create a lesson from backtest result.

        Args:
            result: Backtest result
            crisis_type: Type of crisis
            successful: Whether strategy was successful

        Returns:
            CrisisLesson to store
        """
        # Calculate feature importance
        feature_importance = self._extract_feature_importance(result)

        # Determine boost amount
        if successful:
            boost = result.final_wealth * 0.5  # Boost based on performance
        else:
            boost = -abs(result.max_drawdown) * 0.5  # Penalty for failure

        # Create conditions
        conditions_met = {
            "volatility_threshold": result.volatility * 0.9,
            "max_drawdown_threshold": result.max_drawdown * 1.1,
            "trend_requirement": "positive" if result.final_wealth > 1.0 else "negative"
        }

        # Generate lesson text
        lesson_text = self._generate_lesson_text(result, crisis_type, successful)

        return CrisisLesson(
            lesson_id=f"{result.strategy_id}_{crisis_type}_{datetime.utcnow().isoformat()}",
            crisis=crisis_type,
            strategy_type=self._infer_strategy_type(result),
            successful=successful,
            lesson=lesson_text,
            feature_importance=feature_importance,
            boost_amount=boost,
            conditions_met=conditions_met
        )

    def _infer_strategy_type(self, result: BacktestResult) -> str:
        """
        Infer strategy type from backtest result.

        Args:
            result: Backtest result

        Returns:
            Strategy type string
        """
        # Simplified inference based on metrics
        if result.volatility > 0.3:
            return "momentum"
        elif result.final_wealth > 1.2 and result.max_drawdown < 0.2:
            return "value"
        else:
            return "factor_combination"

    def _generate_lesson_text(
        self,
        result: BacktestResult,
        crisis_type: CrisisType,
        successful: bool
    ) -> str:
        """
        Generate natural language lesson from result.

        Args:
            result: Backtest result
            crisis_type: Type of crisis
            successful: Whether successful

        Returns:
            Natural language lesson
        """
        if successful:
            return (
                f"Strategy survived {crisis_type} crisis with "
                f"{result.final_wealth:.2f}x final wealth and "
                f"{result.max_drawdown:.1%} max drawdown. "
                f"Sharpe ratio: {result.sharpe_ratio:.2f}. "
                f"Key success factors: low volatility, consistent returns."
            )
        else:
            return (
                f"Strategy failed during {crisis_type} crisis. "
                f"Suffered {result.max_drawdown:.1%} drawdown and ended with "
                f"{result.final_wealth:.2f}x wealth. "
                f"Failure factors: excessive risk, poor crisis adaptation."
            )
