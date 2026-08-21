"""
Adaptive Strategy Selector for Decomposition Engine

This module provides adaptive weight calculation and strategy selection
based on historical performance data for the decomposition engine.
"""
from __future__ import annotations


import logging
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

# **ACTUAL INTEGRATION**: Knowledge engine and alerting for strategy optimization
try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformanceData:
    """Performance data for a single strategy."""
    strategy_name: str
    success_count: int = 0
    failure_count: int = 0
    average_quality: float = 0.0
    last_used: Optional[datetime] = None
    total_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """Calculate confidence based on attempt count (0-1)."""
        # More attempts = higher confidence (up to a point)
        if self.total_attempts < 5:
            return self.total_attempts / 5.0
        return min(1.0, 5.0 / self.total_attempts)


class StrategyPerformanceTracker:
    """
    Tracks historical performance of decomposition strategies.

    Maintains performance metrics for each strategy and provides
    data for adaptive weight calculation.
    """

    def __init__(self):
        """Initialize the performance tracker."""
        self.strategies: Dict[str, StrategyPerformanceData] = {}
        self.history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.StrategyPerformanceTracker")

    def record_attempt(
        self,
        strategy_name: str,
        success: bool,
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a strategy execution attempt.

        Args:
            strategy_name: Name of the strategy used
            success: Whether the decomposition succeeded
            quality_score: Quality score (0-100)
            metadata: Additional metadata about the attempt
        """
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = StrategyPerformanceData(
                strategy_name=strategy_name
            )

        data = self.strategies[strategy_name]

        if success:
            data.success_count += 1
        else:
            data.failure_count += 1

        # Update rolling average quality safely (first attempt included).
        previous_attempts = data.total_attempts
        new_attempt_count = previous_attempts + 1
        data.average_quality = (
            (data.average_quality * previous_attempts) + quality_score
        ) / new_attempt_count

        data.last_used = datetime.now()
        data.total_attempts = new_attempt_count

        # Add to history
        self.history.append({
            'strategy': strategy_name,
            'success': success,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })

        # Limit history size
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        self.logger.debug(
            f"Recorded {strategy_name} attempt: "
            f"success={success}, quality={quality_score:.1f}"
        )

        # **ACTUAL INTEGRATION**: Periodically store to knowledge (every 10 attempts)
        if data.total_attempts % 10 == 0:
            self.store_performance_to_knowledge()

        # **ACTUAL INTEGRATION**: Check for degradation and alert
        if data.total_attempts >= 10:
            self.check_strategy_degradation()

    def get_strategy_data(
        self,
        strategy_name: str
    ) -> Optional[StrategyPerformanceData]:
        """Get performance data for a strategy."""
        return self.strategies.get(strategy_name)

    def get_all_strategies(self) -> Dict[str, StrategyPerformanceData]:
        """Get all strategy performance data."""
        return self.strategies.copy()

    def get_best_strategy(
        self,
        min_attempts: int = 3
    ) -> Optional[str]:
        """
        Get the best performing strategy.

        Args:
            min_attempts: Minimum number of attempts required

        Returns:
            Best strategy name or None if insufficient data
        """
        best_strategy = None
        best_score = -1.0

        for name, data in self.strategies.items():
            if data.total_attempts >= min_attempts:
                # Score combines success rate and quality
                score = (
                    data.success_rate * 0.6 +
                    (data.average_quality / 100) * 0.4
                )

                if score > best_score:
                    best_score = score
                    best_strategy = name

        return best_strategy

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Knowledge and alerting for strategy optimization
    # =========================================================================

    def store_performance_to_knowledge(self) -> bool:
        """
        **ACTUAL INTEGRATION**: Store strategy performance to knowledge engine.

        Enables learning from strategy effectiveness across sessions.
        """
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            # Create artifact from performance data
            artifact = KnowledgeArtifact(
                artifact_id=f"strategy_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="strategy_performance",
                source_component="adaptive_strategy_selector",
                title="Strategy Performance Summary",
                content={
                    "strategies": {
                        name: {
                            "success_rate": data.success_rate,
                            "average_quality": data.average_quality,
                            "total_attempts": data.total_attempts,
                            "confidence": data.confidence
                        }
                        for name, data in self.strategies.items()
                    },
                    "best_strategy": self.get_best_strategy(),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "total_strategies": len(self.strategies),
                    "total_recorded": len(self.history)
                },
                tags=["strategy", "performance", "adaptive"]
            )

            # Store in knowledge engine
            knowledge_engine.store_artifact(artifact)
            self.logger.debug(f"Stored strategy performance to knowledge: {artifact.artifact_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store performance to knowledge: {e}")
            return False

    def check_strategy_degradation(self) -> None:
        """
        **ACTUAL INTEGRATION**: Check for strategy degradation and trigger alerts.

        Alerts when:
        - Strategy success rate drops below threshold
        - Strategy quality degrades significantly
        """
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            for name, data in self.strategies.items():
                # Check for degradation (needs at least 10 attempts)
                if data.total_attempts >= 10:
                    # Alert on low success rate
                    if data.success_rate < 0.5:
                        alert_manager.create_alert(
                            title=f"Strategy Degradation: {name}",
                            description=f"Strategy '{name}' has low success rate: {data.success_rate:.1%}. "
                                       f"Consider alternative strategies.",
                            severity=AlertSeverity.MEDIUM.value,
                            source="adaptive_strategy_selector",
                            component="strategy_tracker",
                            metadata={
                                "strategy": name,
                                "success_rate": data.success_rate,
                                "average_quality": data.average_quality,
                                "total_attempts": data.total_attempts
                            }
                        )

                    # Alert on quality degradation
                    if data.average_quality < 50 and data.total_attempts >= 5:
                        alert_manager.create_alert(
                            title=f"Low Quality Strategy: {name}",
                            description=f"Strategy '{name}' producing low quality results: {data.average_quality:.1f}/100",
                            severity=AlertSeverity.LOW.value,
                            source="adaptive_strategy_selector",
                            component="strategy_tracker",
                            metadata={
                                "strategy": name,
                                "average_quality": data.average_quality,
                                "total_attempts": data.total_attempts
                            }
                        )

        except Exception as e:
            self.logger.error(f"Failed to check strategy degradation: {e}")

    def query_knowledge_for_recommendations(
        self,
        problem_type: Optional[str] = None
    ) -> List[str]:
        """
        **ACTUAL INTEGRATION**: Query knowledge engine for strategy recommendations.

        Returns:
            List of recommended strategy names
        """
        if not KNOWLEDGE_AVAILABLE:
            return []

        try:
            knowledge_engine = get_knowledge_engine()

            # Query for strategy performance patterns
            query = f"strategy recommendations {problem_type or 'general'}"
            results = knowledge_engine.search_knowledge(
                query=query,
                query_type='semantic',
                filters={'artifact_type': 'strategy_performance'},
                limit=5
            )

            recommendations = []
            if results.get('results'):
                for result in results['results']:
                    content = result.get('content', {})
                    if 'best_strategy' in content:
                        recommendations.append(content['best_strategy'])

            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to query knowledge for recommendations: {e}")
            return []


class AdaptiveWeightCalculator:
    """
    Calculates adaptive weights for strategy selection based on performance data.

    Uses historical performance to adjust base algorithmic weights.
    """

    def __init__(self, tracker: StrategyPerformanceTracker):
        """
        Initialize the weight calculator.

        Args:
            tracker: Performance tracker with historical data
        """
        self.tracker = tracker
        self.logger = logging.getLogger(f"{__name__}.AdaptiveWeightCalculator")

    def calculate_adaptive_adjustment(
        self,
        strategy: str,
        problem_type: Optional[str] = None
    ) -> float:
        """
        Calculate adaptive weight adjustment factor for a strategy.

        Returns:
            Adjustment factor (0.5-2.0, where 1.0 means no adjustment)
        """
        data = self.tracker.get_strategy_data(strategy)

        if not data or data.total_attempts < 3:
            return 1.0  # Not enough data, no adjustment

        # Calculate adjustment based on success rate and quality
        success_factor = data.success_rate
        quality_factor = data.average_quality / 100.0

        # Combine factors (success is more important)
        adjustment = (
            success_factor * 0.7 +
            quality_factor * 0.3
        )

        # Map to adjustment range
        # Low performance -> 0.5 (reduce weight)
        # Medium performance -> 1.0 (no change)
        # High performance -> 2.0 (increase weight)

        if adjustment < 0.4:
            return 0.5
        elif adjustment > 0.8:
            return 2.0
        else:
            # Linear interpolation between 0.4-0.8
            return 0.5 + (adjustment - 0.4) / 0.4

    def get_performance_summary(
        self
    ) -> Dict[str, Any]:
        """
        Get a summary of all strategy performance.

        Returns:
            Dict with strategy summaries
        """
        summary = {}

        for name, data in self.tracker.get_all_strategies().items():
            summary[name] = {
                'total_attempts': data.total_attempts,
                'success_count': data.success_count,
                'failure_count': data.failure_count,
                'success_rate': data.success_rate,
                'average_quality': data.average_quality,
                'confidence': data.confidence,
                'last_used': data.last_used.isoformat() if data.last_used else None
            }

        return summary


def create_performance_tracker() -> StrategyPerformanceTracker:
    """Factory function to create a performance tracker."""
    return StrategyPerformanceTracker()


def create_adaptive_calculator(
    tracker: StrategyPerformanceTracker
) -> AdaptiveWeightCalculator:
    """Factory function to create an adaptive weight calculator."""
    return AdaptiveWeightCalculator(tracker)


# Default instances for backward compatibility
_default_tracker: Optional[StrategyPerformanceTracker] = None
_default_calculator: Optional[AdaptiveWeightCalculator] = None


def get_performance_tracker() -> StrategyPerformanceTracker:
    """Get or create the default performance tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = create_performance_tracker()
    return _default_tracker


def get_adaptive_calculator() -> AdaptiveWeightCalculator:
    """Get or create the default adaptive calculator."""
    global _default_calculator
    if _default_calculator is None:
        tracker = get_performance_tracker()
        _default_calculator = create_adaptive_calculator(tracker)
    return _default_calculator
