"""
Adaptive Strategy Selector for Decomposition Engine

This module provides adaptive weight calculation for strategy selection,
adjusting base weights based on historical performance data.

Key Features:
- Performance-based weight adjustment
- Learning from past outcomes
- Trend analysis for strategy improvement
- Confidence-based adaptation
- Domain and problem-type specific learning
"""

import logging
from typing import Dict, Optional
from strategy_performance_tracker import StrategyPerformanceTracker

logger = logging.getLogger(__name__)


class AdaptiveWeightCalculator:
    """
    Calculates strategy weights based on historical performance.

    This class adapts base algorithmic weights by incorporating
    learned performance data from past decompositions.

    Adaptation Formula:
        adaptive_weight = base_weight * performance_multiplier

    Where performance_multiplier considers:
        - Historical quality scores (1.2x for excellent, 0.8x for poor)
        - Success rate in this domain/problem_type
        - Recent performance trend (improving/stable/declining)
        - Confidence in the data (sample size)
    """

    def __init__(self,
                 performance_tracker: StrategyPerformanceTracker,
                 learning_rate: float = 0.5,
                 min_confidence_threshold: float = 0.3):
        """
        Initialize with performance tracker.

        Args:
            performance_tracker: StrategyPerformanceTracker instance
            learning_rate: How much to trust performance data (0.0 to 1.0)
                          - 0.0: Use only base weights (no learning)
                          - 0.5: Balance base and learned (default)
                          - 1.0: Use only learned weights
            min_confidence_threshold: Minimum confidence to apply learning
                                    (0.0 to 1.0, default 0.3)
        """
        self.performance_tracker = performance_tracker
        self.learning_rate = learning_rate
        self.min_confidence_threshold = min_confidence_threshold
        logger.info(f"AdaptiveWeightCalculator initialized with learning_rate={learning_rate}")

    def calculate_adaptive_weights(self,
                                  base_weights: Dict[str, float],
                                  problem_type: Optional[str] = None,
                                  domain: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights adjusting base weights by performance.

        Args:
            base_weights: Dictionary of strategy name -> base weight (0.0 to 1.0)
            problem_type: Optional problem type for filtering
            domain: Optional domain for filtering

        Returns:
            Dictionary of strategy name -> adaptive weight (0.0 to 1.0)

        Examples:
            >>> base = {'semantic': 0.7, 'dependency': 0.6}
            >>> adaptive = calculator.calculate_adaptive_weights(
            ...     base, problem_type='algorithm_design', domain='software_engineering'
            ... )
            >>> print(adaptive)
            {'semantic': 0.84, 'dependency': 0.54}  # semantic boosted by performance
        """
        logger.info(f"Calculating adaptive weights for {len(base_weights)} strategies")

        adaptive_weights = {}
        adjustments = {}  # Track adjustments for transparency

        for strategy, base_weight in base_weights.items():
            # Get historical performance
            perf = self.performance_tracker.get_strategy_performance(
                strategy,
                problem_type=problem_type,
                domain=domain
            )

            # Calculate performance multiplier
            multiplier = self._calculate_multiplier(perf)

            # Apply learning rate (blend base and learned)
            # If learning_rate is 0.5, we do: base * 0.5 + (base * multiplier) * 0.5
            # This gives: base * (1 + (multiplier - 1) * learning_rate)
            adjustment_factor = 1 + (multiplier - 1) * self.learning_rate

            # Apply confidence-based gating
            if perf["confidence"] < self.min_confidence_threshold:
                # Not enough data, reduce adjustment
                adjustment_factor = 1 + (adjustment_factor - 1) * (perf["confidence"] / self.min_confidence_threshold)

            adaptive_weight = base_weight * adjustment_factor

            # Clamp to valid range
            adaptive_weight = max(0.0, min(1.0, adaptive_weight))

            adaptive_weights[strategy] = adaptive_weight
            adjustments[strategy] = {
                "base_weight": base_weight,
                "multiplier": multiplier,
                "adjustment_factor": adjustment_factor,
                "confidence": perf["confidence"],
                "usage_count": perf["usage_count"]
            }

            logger.debug(f"Strategy '{strategy}': "
                        f"base={base_weight:.3f} -> adaptive={adaptive_weight:.3f} "
                        f"(mult={multiplier:.2f}, adj={adjustment_factor:.2f})")

        # Log rankings
        sorted_strategies = sorted(adaptive_weights.items(), key=lambda x: x[1], reverse=True)
        logger.info("Adaptive weights (sorted):")
        for strategy, weight in sorted_strategies:
            logger.info(f"  {strategy}: {weight:.3f} "
                       f"(base={adjustments[strategy]['base_weight']:.3f}, "
                       f"usage={adjustments[strategy]['usage_count']})")

        return adaptive_weights

    def _calculate_multiplier(self, performance: Dict[str, float]) -> float:
        """
        Calculate performance multiplier based on historical data.

        Args:
            performance: Performance dictionary from tracker

        Returns:
            Multiplier (0.5 to 2.0)
            - >1.0: Better than expected (boost weight)
            - 1.0: Expected performance (no change)
            - <1.0: Worse than expected (reduce weight)

        Formula:
            multiplier = quality_mult * success_mult * trend_mult

        Where:
            quality_mult = avg_quality_score (0.7 -> 0.7x)
            success_mult = 1.0 + (success_rate - 0.5) (0.8 -> 0.8 + 0.3 = 1.3x)
            trend_mult = based on trend (improving=1.2, stable=1.0, declining=0.8)
        """
        # Check if we have enough data
        if performance["usage_count"] == 0:
            # No data yet, neutral multiplier
            return 1.0

        # Quality-based multiplier (direct scaling)
        # If quality is 0.8, multiplier is 0.8
        quality_mult = performance["avg_quality_score"]

        # Success rate multiplier
        # Success rate of 0.5 is neutral, higher is better
        # Maps 0.0 -> 0.5, 0.5 -> 1.0, 1.0 -> 1.5
        success_mult = 1.0 + (performance["success_rate"] - 0.5)

        # Trend multiplier
        trend_mult = self._get_trend_multiplier(performance["trend"])

        # Combine multipliers
        combined = quality_mult * success_mult * trend_mult

        # Clamp to reasonable range [0.5, 2.0]
        multiplier = max(0.5, min(2.0, combined))

        logger.debug(f"Multiplier calculation: "
                    f"quality={quality_mult:.2f} * "
                    f"success={success_mult:.2f} * "
                    f"trend={trend_mult:.2f} = {multiplier:.2f}")

        return multiplier

    def _get_trend_multiplier(self, trend: str) -> float:
        """
        Get multiplier based on performance trend.

        Args:
            trend: 'improving', 'stable', 'declining', or 'unknown'

        Returns:
            Multiplier value
        """
        trend_multipliers = {
            "improving": 1.2,   # Boost strategies that are improving
            "stable": 1.0,      # Neutral for stable performance
            "declining": 0.8,   # Reduce strategies that are declining
            "unknown": 1.0      # Neutral when trend unknown
        }

        return trend_multipliers.get(trend, 1.0)

    def get_strategy_recommendations(self,
                                    base_weights: Dict[str, float],
                                    problem_type: Optional[str] = None,
                                    domain: Optional[str] = None,
                                    top_n: int = 3) -> list:
        """
        Get top recommended strategies with explanations.

        Args:
            base_weights: Base weights for strategies
            problem_type: Optional problem type
            domain: Optional domain
            top_n: Number of top strategies to return

        Returns:
            List of (strategy, adaptive_weight, explanation) tuples
        """
        adaptive_weights = self.calculate_adaptive_weights(
            base_weights, problem_type, domain
        )

        recommendations = []
        sorted_strategies = sorted(adaptive_weights.items(),
                                 key=lambda x: x[1],
                                 reverse=True)

        for strategy, weight in sorted_strategies[:top_n]:
            perf = self.performance_tracker.get_strategy_performance(
                strategy, problem_type, domain
            )

            explanation = self._generate_explanation(
                strategy, weight, perf, base_weights[strategy]
            )

            recommendations.append((strategy, weight, explanation))

        return recommendations

    def _generate_explanation(self,
                            strategy: str,
                            adaptive_weight: float,
                            performance: Dict,
                            base_weight: float) -> str:
        """Generate human-readable explanation for strategy recommendation."""
        parts = []

        # Weight change
        change = adaptive_weight - base_weight
        if change > 0.1:
            parts.append(f"↑ boosted by {change:.2f}")
        elif change < -0.1:
            parts.append(f"↓ reduced by {abs(change):.2f}")
        else:
            parts.append("neutral adjustment")

        # Performance data
        if performance["usage_count"] > 0:
            parts.append(f"quality: {performance['avg_quality_score']:.2f}")
            parts.append(f"success: {performance['success_rate']:.0%}")

            if performance["trend"] == "improving":
                parts.append("trend: improving ↑")
            elif performance["trend"] == "declining":
                parts.append("trend: declining ↓")

            if performance["confidence"] >= 0.7:
                parts.append("high confidence")
            elif performance["confidence"] < 0.3:
                parts.append("low confidence")
        else:
            parts.append("no performance data yet")

        return f"{strategy}: {', '.join(parts)}"

    def calculate_learning_progress(self) -> Dict:
        """
        Calculate overall learning progress of the system.

        Returns:
            Dictionary with learning metrics
        """
        stats = self.performance_tracker.get_statistics_summary()

        # Calculate average confidence across all strategies
        strategies = self.performance_tracker.get_all_strategies()
        confidences = []

        for strategy in strategies:
            perf = self.performance_tracker.get_strategy_performance(strategy)
            confidences.append(perf["confidence"])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine learning stage
        if avg_confidence < 0.3:
            stage = "early_learning"
            description = "Gathering initial data"
        elif avg_confidence < 0.7:
            stage = "intermediate_learning"
            description = "Building confidence in patterns"
        else:
            stage = "mature_learning"
            description = "High confidence in learned patterns"

        return {
            "learning_stage": stage,
            "stage_description": description,
            "average_confidence": avg_confidence,
            "total_decompositions": stats["total_decompositions"],
            "strategies_tracked": stats["total_strategies"],
            "learning_rate": self.learning_rate
        }

    def adjust_learning_rate(self, new_rate: float):
        """
        Adjust the learning rate.

        Args:
            new_rate: New learning rate (0.0 to 1.0)
        """
        old_rate = self.learning_rate
        self.learning_rate = max(0.0, min(1.0, new_rate))
        logger.info(f"Learning rate adjusted: {old_rate:.2f} -> {self.learning_rate:.2f}")

    def get_performance_summary(self) -> Dict:
        """
        Get summary of strategy performance across all strategies.

        Returns:
            Dictionary with performance summary
        """
        summary = {
            "learning_rate": self.learning_rate,
            "min_confidence_threshold": self.min_confidence_threshold,
            "strategies": {}
        }

        strategies = self.performance_tracker.get_all_strategies()

        for strategy in strategies:
            perf = self.performance_tracker.get_strategy_performance(strategy)
            summary["strategies"][strategy] = {
                "avg_quality": perf["avg_quality_score"],
                "success_rate": perf["success_rate"],
                "usage_count": perf["usage_count"],
                "trend": perf["trend"],
                "confidence": perf["confidence"]
            }

        return summary

    def reset_learning(self):
        """
        Reset all learned performance data.

        This will clear all historical performance data.
        Use with caution - this removes all learning progress.
        """
        logger.warning("Resetting all learned performance data")
        strategies = self.performance_tracker.get_all_strategies()

        for strategy in strategies:
            self.performance_tracker.reset_strategy_data(strategy)

        logger.info("Learning reset complete")
