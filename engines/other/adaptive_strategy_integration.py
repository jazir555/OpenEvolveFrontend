"""
Adaptive Strategy Selection Integration

Connects the adaptive_strategy_selector to decomposition workflows
and other components, enabling dynamic strategy optimization based
on performance tracking.
"""
from __future__ import annotations


import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import adaptive strategy selector
try:
    from adaptive_strategy_selector import (
        AdaptiveStrategySelector,
        StrategyPerformanceData,
        StrategyPerformanceTracker,
        AdaptiveWeightCalculator,
        select_decomposition_strategy,
    )
    ADAPTIVE_STRATEGY_AVAILABLE = True
except ImportError:
    ADAPTIVE_STRATEGY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available strategy types."""
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class AdaptiveIntegrationManager:
    """
    Manages adaptive strategy selection across components.

    Connects performance tracking with strategy selection to
    automatically optimize system behavior.
    """

    def __init__(self):
        """Initialize adaptive integration manager."""
        self.selector: Optional[AdaptiveStrategySelector] = None
        self.tracker: Optional[StrategyPerformanceTracker] = None
        self.component_strategies: Dict[str, str] = {}
        self.performance_history: List[Dict[str, Any]] = []

        if ADAPTIVE_STRATEGY_AVAILABLE:
            try:
                self.selector = AdaptiveStrategySelector()
                self.tracker = StrategyPerformanceTracker()
                logger.info("Adaptive strategy integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive strategy: {e}")

    def record_performance(
        self,
        component: str,
        strategy: str,
        execution_time: float,
        success: bool,
        quality_score: Optional[float] = None,
        resource_usage: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record performance data for a strategy.

        Args:
            component: Component name
            strategy: Strategy used
            execution_time: Execution time in seconds
            success: Whether operation succeeded
            quality_score: Optional quality score (0-1)
            resource_usage: Optional resource usage metrics
            metadata: Optional additional metadata

        Returns:
            True if recorded successfully
        """
        if not self.tracker:
            return False

        try:
            performance_data = StrategyPerformanceData(
                strategy_name=strategy,
                component=component,
                execution_time=execution_time,
                success=success,
                quality_score=quality_score or (1.0 if success else 0.0),
                timestamp=datetime.now(),
                resource_usage=resource_usage or {},
                metadata=metadata or {}
            )

            self.tracker.record_performance(performance_data)

            # Add to history
            self.performance_history.append({
                'component': component,
                'strategy': strategy,
                'execution_time': execution_time,
                'success': success,
                'quality_score': quality_score,
                'timestamp': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            logger.error(f"Failed to record performance: {e}")
            return False

    def select_strategy(
        self,
        component: str,
        strategy_type: StrategyType,
        problem_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Select optimal strategy for a component.

        Args:
            component: Component name
            strategy_type: Type of strategy
            problem_context: Optional problem context

        Returns:
            Selected strategy name or None
        """
        if not self.selector:
            return None

        try:
            # Get performance data for this component
            component_history = [
                p for p in self.performance_history
                if p['component'] == component
            ]

            # Select strategy based on performance
            if component_history:
                # Use adaptive selection
                strategy = select_decomposition_strategy(
                    problem_context=problem_context or {},
                    performance_history=component_history
                )
            else:
                # Use default strategy
                strategy = self._get_default_strategy(component, strategy_type)

            # Cache selected strategy
            self.component_strategies[component] = strategy

            return strategy

        except Exception as e:
            logger.error(f"Failed to select strategy: {e}")
            return self._get_default_strategy(component, strategy_type)

    def _get_default_strategy(self, component: str, strategy_type: StrategyType) -> str:
        """Get default strategy for component and type."""
        defaults = {
            StrategyType.DECOMPOSITION: "hierarchical_decomposition",
            StrategyType.VERIFICATION: "z3_formal_verification",
            StrategyType.OPTIMIZATION: "genetic_algorithm",
            StrategyType.VALIDATION: "constraint_satisfaction",
        }
        return defaults.get(strategy_type, "default_strategy")

    def get_recommended_strategies(
        self,
        component: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get recommended strategies for a component.

        Args:
            component: Component name
            limit: Maximum number of recommendations

        Returns:
            List of recommended strategies with scores
        """
        if not self.selector:
            return []

        try:
            # Get component performance history
            component_history = [
                p for p in self.performance_history
                if p['component'] == component
            ]

            if not component_history:
                return []

            # Calculate strategy performance
            strategy_performance = {}
            for record in component_history:
                strategy = record['strategy']
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'count': 0,
                        'successes': 0,
                        'total_time': 0,
                        'total_quality': 0,
                    }

                stats = strategy_performance[strategy]
                stats['count'] += 1
                if record['success']:
                    stats['successes'] += 1
                stats['total_time'] += record['execution_time']
                if record['quality_score']:
                    stats['total_quality'] += record['quality_score']

            # Calculate scores
            recommendations = []
            for strategy, stats in strategy_performance.items():
                success_rate = stats['successes'] / stats['count']
                avg_time = stats['total_time'] / stats['count']
                avg_quality = stats['total_quality'] / stats['count'] if stats['total_quality'] > 0 else 0

                # Combined score (higher is better)
                score = (success_rate * 0.5 + avg_quality * 0.3 + (1 / (avg_time + 1)) * 0.2)

                recommendations.append({
                    'strategy': strategy,
                    'score': score,
                    'success_rate': success_rate,
                    'avg_time': avg_time,
                    'avg_quality': avg_quality,
                    'usage_count': stats['count'],
                })

            # Sort by score and return top N
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []

    def optimize_component_strategies(self) -> Dict[str, str]:
        """
        Optimize strategy selection for all components.

        Returns:
            Dictionary of component -> optimal strategy
        """
        optimized = {}

        try:
            # Get unique components
            components = set(p['component'] for p in self.performance_history)

            for component in components:
                # Get best strategy for this component
                recommendations = self.get_recommended_strategies(component, limit=1)
                if recommendations:
                    optimized[component] = recommendations[0]['strategy']

            return optimized

        except Exception as e:
            logger.error(f"Failed to optimize strategies: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all components."""
        summary = {
            'total_records': len(self.performance_history),
            'components': {},
            'overall_success_rate': 0,
        }

        if not self.performance_history:
            return summary

        # Calculate overall success rate
        successful = sum(1 for p in self.performance_history if p['success'])
        summary['overall_success_rate'] = successful / len(self.performance_history)

        # Calculate per-component stats
        components = set(p['component'] for p in self.performance_history)
        for component in components:
            component_records = [
                p for p in self.performance_history
                if p['component'] == component
            ]

            component_successful = sum(1 for p in component_records if p['success'])
            component_total = len(component_records)

            summary['components'][component] = {
                'total_operations': component_total,
                'successful': component_successful,
                'success_rate': component_successful / component_total if component_total > 0 else 0,
                'avg_execution_time': sum(
                    p['execution_time'] for p in component_records
                ) / component_total if component_total > 0 else 0,
            }

        return summary


# Global instance
_adaptive_manager: Optional[AdaptiveIntegrationManager] = None


def get_adaptive_manager() -> AdaptiveIntegrationManager:
    """Get or create the adaptive integration manager singleton."""
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveIntegrationManager()
    return _adaptive_manager


def adaptive_strategy_decorator(component: str, strategy_type: StrategyType):
    """
    Decorator for applying adaptive strategy selection.

    Automatically records performance and optimizes strategy selection.

    Args:
        component: Component name
        strategy_type: Type of strategy

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_adaptive_manager()
            start_time = datetime.now()

            # Select optimal strategy
            strategy = manager.select_strategy(
                component=component,
                strategy_type=strategy_type,
                problem_context=kwargs.get('context')
            )

            # Add strategy to kwargs
            kwargs['strategy'] = strategy

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record successful performance
                execution_time = (datetime.now() - start_time).total_seconds()
                manager.record_performance(
                    component=component,
                    strategy=strategy or 'default',
                    execution_time=execution_time,
                    success=True,
                    quality_score=kwargs.get('quality_score'),
                    resource_usage=kwargs.get('resource_usage'),
                )

                return result

            except Exception as e:
                # Record failed performance
                execution_time = (datetime.now() - start_time).total_seconds()
                manager.record_performance(
                    component=component,
                    strategy=strategy or 'default',
                    execution_time=execution_time,
                    success=False,
                    metadata={'error': str(e)}
                )
                raise

        return wrapper
    return decorator


__all__ = [
    'StrategyType',
    'AdaptiveIntegrationManager',
    'get_adaptive_manager',
    'adaptive_strategy_decorator',
]
