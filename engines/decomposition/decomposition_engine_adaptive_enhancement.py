"""
Adaptive Strategy Selection Enhancement for Decomposition Engine

This module provides the integration of adaptive strategy selection
into the DecompositionEngine, adding v3 strategy selection with
learning capabilities and feedback loops.
"""
from __future__ import annotations


import logging
from typing import Dict, Optional, List
from datetime import datetime

from strategy_performance_tracker import StrategyPerformanceTracker
from adaptive_strategy_selector import AdaptiveWeightCalculator

# Import weight calculation functions from decomposition_engine
from decomposition_engine import (
    calculate_functional_weight,
    calculate_temporal_weight,
    calculate_risk_weight,
    calculate_value_weight,
    calculate_technical_weight
)

logger = logging.getLogger(__name__)


def select_decomposition_strategy_v3(
    problem,
    analyzed_context=None,
    performance_tracker: Optional[StrategyPerformanceTracker] = None,
    adaptive_calculator: Optional[AdaptiveWeightCalculator] = None,
    use_adaptive_selection: bool = True
) -> tuple:
    """
    Select decomposition strategy using adaptive weights based on historical performance.

    This is V3 - adaptive version that learns from past outcomes.
    It combines fast algorithmic weight calculation with learned performance data.

    Algorithm:
    1. Calculate base weights using algorithmic functions (fast, deterministic)
    2. Adjust weights based on historical performance (adaptive learning)
    3. Find strategy with highest adjusted weight
    4. If max weight > 0.6, use that strategy
    5. Otherwise, use hybrid combining top strategies

    Args:
        problem: The problem to select a strategy for
        analyzed_context: Optional pre-analyzed context (uses problem if not provided)
        performance_tracker: Optional StrategyPerformanceTracker for learning
        adaptive_calculator: Optional AdaptiveWeightCalculator for weight adjustment
        use_adaptive_selection: If True, use adaptive weights (default: True)

    Returns:
        Tuple of (strategy_name, selection_metadata)
        - strategy_name: Selected strategy name
        - selection_metadata: Dict with selection details including:
            * base_weights: Original algorithmic weights
            * adaptive_weights: Performance-adjusted weights (if adaptive)
            * performance_adjustment: Details of adjustments made
            * selection_reason: Why this strategy was chosen

    Examples:
        >>> strategy, metadata = select_decomposition_strategy_v3(problem)
        >>> print(strategy)
        'semantic'

        >>> strategy, metadata = select_decomposition_strategy_v3(problem, use_adaptive=False)
        >>> print(metadata['selection_reason'])
        'Base weights showed strong preference for semantic'
    """
    logger.info(f"Selecting decomposition strategy (v3) for problem: {problem.id}")

    # Use problem if analyzed_context not provided
    if analyzed_context is None:
        analyzed_context = problem

    # Step 1: Calculate base weights (algorithmic - fast)
    base_weights = {
        'functional': calculate_functional_weight(analyzed_context),
        'temporal': calculate_temporal_weight(analyzed_context),
        'risk_based': calculate_risk_weight(analyzed_context),
        'value_based': calculate_value_weight(analyzed_context),
        'technical': calculate_technical_weight(analyzed_context)
    }

    logger.info("Base strategy weights:")
    for strategy, weight in sorted(base_weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {strategy}: {weight:.3f}")

    # Step 2: Adjust weights based on performance (adaptive)
    adaptive_weights = base_weights.copy()
    performance_adjustment = None

    if use_adaptive_selection and adaptive_calculator and performance_tracker:
        try:
            # Get problem classification
            problem_type = getattr(analyzed_context, 'problem_type', None)
            if problem_type and hasattr(problem_type, 'value'):
                problem_type_str = problem_type.value
            else:
                problem_type_str = 'unknown'

            # Get domain
            domain_context = getattr(analyzed_context, 'domain_context', None)
            if domain_context and hasattr(domain_context, 'domain'):
                domain_str = domain_context.domain
            else:
                domain_str = 'general'

            # Calculate adaptive weights
            adaptive_weights = adaptive_calculator.calculate_adaptive_weights(
                base_weights,
                problem_type=problem_type_str,
                domain=domain_str
            )

            # Log adjustment details
            performance_adjustment = {
                'problem_type': problem_type_str,
                'domain': domain_str,
                'learning_rate': adaptive_calculator.learning_rate,
                'adjustments': {}
            }

            for strategy in base_weights:
                base = base_weights[strategy]
                adaptive = adaptive_weights[strategy]
                change = adaptive - base
                performance_adjustment['adjustments'][strategy] = {
                    'base': base,
                    'adaptive': adaptive,
                    'change': change,
                    'change_percent': (change / base * 100) if base > 0 else 0
                }

            logger.info("Adaptive weights applied (performance-adjusted):")
            for strategy, weight in sorted(adaptive_weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {strategy}: {weight:.3f}")

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"Adaptive weight calculation failed: {e}. Using base weights.", exc_info=True)
            adaptive_weights = base_weights
            performance_adjustment = None
    else:
        if not use_adaptive_selection:
            logger.info("Adaptive selection disabled, using base weights")
        elif not adaptive_calculator:
            logger.info("No adaptive calculator provided, using base weights")
        elif not performance_tracker:
            logger.info("No performance tracker provided, using base weights")

    # Step 3: Find strategy with highest weight
    max_weight_strategy = max(adaptive_weights, key=adaptive_weights.get)
    max_weight = adaptive_weights[max_weight_strategy]

    # Step 4: Apply threshold logic
    if max_weight > 0.6:
        # Strong preference - use single strategy
        strategy_mapping = {
            'functional': 'semantic',
            'temporal': 'semantic',
            'risk_based': 'complexity',
            'value_based': 'semantic',
            'technical': 'dependency'
        }

        selected_strategy = strategy_mapping.get(max_weight_strategy, 'hybrid')

        selection_reason = f"Strong preference for {max_weight_strategy} (weight: {max_weight:.3f} > 0.6)"
        if performance_adjustment:
            original_weight = base_weights[max_weight_strategy]
            if abs(max_weight - original_weight) > 0.1:
                selection_reason += f", adjusted from {original_weight:.3f} by learning"

        logger.info(f"Selected single strategy: {selected_strategy} ({selection_reason})")

    else:
        # No strong preference - use hybrid approach
        sorted_weights = sorted(adaptive_weights.items(), key=lambda x: x[1], reverse=True)

        # Get top 2-3 strategies with weight > 0.3
        top_strategies = [strategy for strategy, weight in sorted_weights[:3] if weight > 0.3]

        if len(top_strategies) < 2:
            selected_strategy = 'hybrid'
            selection_reason = 'No strong secondary strategy found'
        else:
            # Map to internal strategy names
            strategy_mapping = {
                'functional': 'semantic',
                'temporal': 'semantic',
                'risk_based': 'complexity',
                'value_based': 'semantic',
                'technical': 'dependency'
            }

            mapped_strategies = [strategy_mapping.get(s, 'semantic') for s in top_strategies]
            # Remove duplicates while preserving order
            unique_strategies = list(dict.fromkeys(mapped_strategies))

            if len(unique_strategies) == 1:
                selected_strategy = unique_strategies[0]
                selection_reason = f"All top strategies map to {selected_strategy}"
            else:
                selected_strategy = 'hybrid'
                selection_reason = f"Multiple competitive strategies: {', '.join(top_strategies[:2])}"

        logger.info(f"Selected strategy: {selected_strategy} ({selection_reason})")

    # Prepare metadata
    selection_metadata = {
        'version': 'v3_adaptive' if use_adaptive_selection else 'v3_base',
        'selected_strategy': selected_strategy,
        'original_winner': max_weight_strategy,
        'base_weights': base_weights,
        'final_weights': adaptive_weights,
        'performance_adjustment': performance_adjustment,
        'selection_reason': selection_reason,
        'threshold_met': max_weight > 0.6,
        'max_weight': max_weight,
        'timestamp': datetime.now().isoformat()
    }

    return selected_strategy, selection_metadata


def record_decomposition_outcome(
    performance_tracker: StrategyPerformanceTracker,
    strategy: str,
    problem,
    quality_score: float,
    time_to_complete: Optional[float] = None,
    user_satisfaction: Optional[float] = None
):
    """
    Record the outcome of a decomposition for learning.

    Args:
        performance_tracker: StrategyPerformanceTracker instance
        strategy: Strategy that was used
        problem: Problem that was decomposed
        quality_score: Overall quality score (0.0 to 1.0)
        time_to_complete: Optional completion time in seconds
        user_satisfaction: Optional user satisfaction score (0.0 to 1.0)
    """
    try:
        # Get problem classification
        problem_type = getattr(problem, 'problem_type', None)
        if problem_type and hasattr(problem_type, 'value'):
            problem_type_str = problem_type.value
        else:
            problem_type_str = 'unknown'

        # Get domain
        domain_context = getattr(problem, 'domain_context', None)
        if domain_context and hasattr(domain_context, 'domain'):
            domain_str = domain_context.domain
        else:
            domain_str = 'general'

        # Map strategy name back to weight category
        strategy_reverse_mapping = {
            'semantic': 'functional',  # Semantic was likely chosen from functional
            'dependency': 'technical',  # Dependency from technical
            'complexity': 'risk_based',  # Complexity from risk_based
            'hybrid': 'functional'  # Hybrid defaults to functional
        }

        weight_strategy = strategy_reverse_mapping.get(strategy, 'functional')

        # Record outcome
        performance_tracker.record_strategy_outcome(
            strategy=weight_strategy,
            problem_type=problem_type_str,
            domain=domain_str,
            quality_score=quality_score,
            user_satisfaction=user_satisfaction,
            time_to_complete=time_to_complete
        )

        logger.info(f"Recorded decomposition outcome: strategy={strategy}, "
                   f"quality={quality_score:.2f}, type={problem_type_str}, domain={domain_str}")

    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Failed to record decomposition outcome: {e}", exc_info=True)


class AdaptiveDecompositionEngineMixin:
    """
    Mixin class to add adaptive strategy selection to DecompositionEngine.

    This mixin provides:
    - Adaptive strategy selection (v3)
    - Performance tracking
    - Feedback loop integration
    - Learning progress monitoring

    Usage:
        class DecompositionEngine(AdaptiveDecompositionEngineMixin):
            ...

    Or patch existing DecompositionEngine:
        DecompositionEngine.__bases__ += (AdaptiveDecompositionEngineMixin,)
    """

    def __init__(self, *args, **kwargs):
        """Initialize adaptive components."""
        # Extract adaptive-specific parameters
        self.use_adaptive_selection = kwargs.pop('use_adaptive_selection', True)
        performance_storage_path = kwargs.pop('performance_storage_path', 'strategy_performance.json')
        learning_rate = kwargs.pop('learning_rate', 0.5)

        # Initialize performance tracker
        self.performance_tracker = StrategyPerformanceTracker(storage_path=performance_storage_path)

        # Initialize adaptive calculator
        self.adaptive_calculator = AdaptiveWeightCalculator(
            performance_tracker=self.performance_tracker,
            learning_rate=learning_rate
        )

        # Log initialization
        logger.info(f"AdaptiveDecompositionEngineMixin initialized: "
                   f"use_adaptive={self.use_adaptive_selection}, "
                   f"learning_rate={learning_rate}")

        # Call parent init if needed (when used as mixin)
        super().__init__(*args, **kwargs)

    def select_strategy_adaptive(self, problem) -> tuple:
        """
        Select strategy using adaptive learning.

        Args:
            problem: Problem to analyze

        Returns:
            Tuple of (strategy_name, selection_metadata)
        """
        return select_decomposition_strategy_v3(
            problem=problem,
            performance_tracker=self.performance_tracker,
            adaptive_calculator=self.adaptive_calculator,
            use_adaptive_selection=self.use_adaptive_selection
        )

    def record_outcome(self, strategy, problem, quality_score, time_to_complete=None):
        """Record decomposition outcome for learning."""
        record_decomposition_outcome(
            performance_tracker=self.performance_tracker,
            strategy=strategy,
            problem=problem,
            quality_score=quality_score,
            time_to_complete=time_to_complete
        )

    def get_learning_progress(self) -> Dict:
        """Get learning system progress."""
        return self.adaptive_calculator.calculate_learning_progress()

    def get_performance_summary(self) -> Dict:
        """Get performance summary across all strategies."""
        return self.adaptive_calculator.get_performance_summary()

    def export_performance_report(self, output_path: str = "performance_report.json"):
        """Export detailed performance report."""
        self.performance_tracker.export_performance_report(output_path)


# Standalone helper function for non-class usage
def setup_adaptive_selection(
    decomposition_engine,
    use_adaptive_selection: bool = True,
    performance_storage_path: str = "strategy_performance.json",
    learning_rate: float = 0.5
):
    """
    Add adaptive selection capabilities to an existing DecompositionEngine instance.

    Args:
        decomposition_engine: Existing DecompositionEngine instance
        use_adaptive_selection: Whether to use adaptive selection
        performance_storage_path: Path for performance data storage
        learning_rate: Learning rate for adaptive calculator

    Returns:
        The modified decomposition_engine with adaptive capabilities
    """
    # Add attributes
    decomposition_engine.use_adaptive_selection = use_adaptive_selection
    decomposition_engine.performance_tracker = StrategyPerformanceTracker(
        storage_path=performance_storage_path
    )
    decomposition_engine.adaptive_calculator = AdaptiveWeightCalculator(
        performance_tracker=decomposition_engine.performance_tracker,
        learning_rate=learning_rate
    )

    # Add methods
    decomposition_engine.select_strategy_adaptive = lambda problem: select_decomposition_strategy_v3(
        problem=problem,
        performance_tracker=decomposition_engine.performance_tracker,
        adaptive_calculator=decomposition_engine.adaptive_calculator,
        use_adaptive_selection=use_adaptive_selection
    )

    decomposition_engine.record_outcome = lambda strategy, problem, quality_score, time_to_complete=None: \
        record_decomposition_outcome(
            performance_tracker=decomposition_engine.performance_tracker,
            strategy=strategy,
            problem=problem,
            quality_score=quality_score,
            time_to_complete=time_to_complete
        )

    decomposition_engine.get_learning_progress = lambda: \
        decomposition_engine.adaptive_calculator.calculate_learning_progress()

    decomposition_engine.get_performance_summary = lambda: \
        decomposition_engine.adaptive_calculator.get_performance_summary()

    decomposition_engine.export_performance_report = lambda output_path="performance_report.json": \
        decomposition_engine.performance_tracker.export_performance_report(output_path)

    logger.info("Adaptive selection capabilities added to DecompositionEngine")

    return decomposition_engine
