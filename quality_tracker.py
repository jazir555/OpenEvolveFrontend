"""
Quality Trend Tracker for Decomposition Assessment

Tracks quality assessment trends over time and provides insights for continuous improvement.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQualityScores:
    """
    Enhanced quality assessment scores for decomposition evaluation.

    Provides multi-dimensional quality assessment with detailed breakdowns and recommendations.
    """
    # Overall scores
    overall_score: float
    meets_thresholds: bool

    # Dimension scores
    completeness_score: float
    consistency_score: float
    feasibility_score: float
    dependency_score: float
    balance_score: float

    # Detailed assessments for each dimension
    completeness_details: Dict[str, Any]
    consistency_details: Dict[str, Any]
    feasibility_details: Dict[str, Any]
    dependency_details: Dict[str, Any]
    balance_details: Dict[str, Any]

    # Improvement guidance
    improvement_recommendations: List[str]
    critical_issues: List[str]

    # Validation and tracking
    validation_checkpoints: List[str]
    timestamp: datetime


class QualityTracker:
    """
    Track quality assessment trends over time.

    Provides functionality for:
    - Recording quality assessments
    - Analyzing trends over time periods
    - Identifying consistently low-scoring dimensions
    - Generating improvement insights
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize quality tracker.

        Args:
            storage_path: Optional path to JSON file for persistence
        """
        self.storage_path = storage_path
        self.assessments: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.dimension_history: Dict[str, List[float]] = defaultdict(list)

        # Load from storage if provided
        if storage_path:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """
        Load assessments from persistent storage.

        Raises:
            OSError: If file cannot be read due to permission or I/O errors
            json.JSONDecodeError: If storage file contains invalid JSON
        """
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assessments = defaultdict(dict, data.get('assessments', {}))
                self.dimension_history = defaultdict(list, data.get('dimension_history', {}))
                logger.info(f"Loaded {len(self.assessments)} assessments from {self.storage_path}")
        except FileNotFoundError:
            logger.info(f"No existing storage found at {self.storage_path}, starting fresh")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in storage file {self.storage_path}: {e}")
            raise
        except (OSError, IOError) as e:
            logger.error(f"Failed to read storage file {self.storage_path}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Storage file {self.storage_path} has invalid structure: {e}")
            raise

    def _save_to_storage(self) -> None:
        """
        Save assessments to persistent storage.

        Raises:
            OSError: If file cannot be written due to permission or I/O errors
            TypeError: If data contains non-serializable objects
        """
        if not self.storage_path:
            return

        try:
            # Ensure directory exists
            storage_dir = os.path.dirname(self.storage_path)
            if storage_dir and not os.path.exists(storage_dir):
                os.makedirs(storage_dir, exist_ok=True)

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'assessments': dict(self.assessments),
                    'dimension_history': dict(self.dimension_history)
                }, f, indent=2, default=str)
            logger.debug(f"Saved assessments to {self.storage_path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to write to storage file {self.storage_path}: {e}")
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize assessment data: {e}")
            raise

    def record_assessment(self,
                         plan_id: str,
                         scores: EnhancedQualityScores,
                         problem_type: Optional[str] = None,
                         strategy: Optional[str] = None) -> None:
        """
        Record assessment for tracking.

        Args:
            plan_id: Unique identifier for the decomposition plan
            scores: EnhancedQualityScores object containing assessment results
            problem_type: Optional problem type for categorization
            strategy: Optional strategy used for decomposition

        Raises:
            ValueError: If plan_id is empty or scores contain invalid values
            AttributeError: If required score attributes are missing
        """
        if not plan_id or not plan_id.strip():
            raise ValueError("plan_id cannot be empty")

        if not isinstance(scores, EnhancedQualityScores):
            raise ValueError(f"scores must be EnhancedQualityScores, got {type(scores)}")

        # Validate score values
        if not 0 <= scores.overall_score <= 1:
            raise ValueError(f"overall_score must be between 0 and 1, got {scores.overall_score}")

        timestamp = datetime.now()

        # Store assessment
        try:
            self.assessments[plan_id] = {
                'timestamp': timestamp.isoformat(),
                'overall_score': float(scores.overall_score),
                'meets_thresholds': bool(scores.meets_thresholds),
                'completeness': float(scores.completeness_score),
                'consistency': float(scores.consistency_score),
                'feasibility': float(scores.feasibility_score),
                'dependency': float(scores.dependency_score),
                'balance': float(scores.balance_score),
                'problem_type': problem_type,
                'strategy': strategy,
                'critical_issues': list(scores.critical_issues) if scores.critical_issues else [],
                'recommendations_count': len(scores.improvement_recommendations)
            }
        except (AttributeError, TypeError) as e:
            logger.error(f"Failed to extract scores from EnhancedQualityScores: {e}")
            raise

        # Update dimension history
        for dimension in ['completeness', 'consistency', 'feasibility', 'dependency', 'balance']:
            try:
                score = getattr(scores, f'{dimension}_score')
                self.dimension_history[dimension].append({
                    'score': float(score),
                    'timestamp': timestamp.isoformat(),
                    'plan_id': plan_id
                })
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to extract {dimension}_score: {e}")

        # Save to storage
        try:
            self._save_to_storage()
        except (OSError, IOError, TypeError, ValueError) as e:
            logger.error(f"Failed to persist assessment to storage: {e}")
            # Don't raise - assessment is still recorded in memory

        logger.info(f"Recorded assessment for plan {plan_id}: overall={scores.overall_score:.3f}")

    def get_trends(self, time_period: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """
        Get quality trends over time period.

        Args:
            time_period: Time period to analyze (default: 30 days)

        Returns:
            Dictionary containing trend analysis with the following structure:
            {
                'period_days': int,
                'total_assessments': int,
                'overall_stats': {'avg': float, 'min': float, 'max': float, 'count': int},
                'dimension_stats': {dimension: stats_dict},
                'threshold_met_rate': float,
                'overall_trend': {  # optional, if enough data
                    'change': float,
                    'direction': str,  # 'improving', 'declining', 'stable'
                    'percent_change': float
                }
            }

        Raises:
            ValueError: If time_period is invalid
        """
        if not isinstance(time_period, timedelta) or time_period.total_seconds() <= 0:
            raise ValueError("time_period must be a positive timedelta")

        try:
            cutoff_time = datetime.now() - time_period

            # Filter assessments within time period
            recent_assessments = []
            for plan_id, assessment in self.assessments.items():
                try:
                    assessment_time = datetime.fromisoformat(assessment['timestamp'])
                    if assessment_time >= cutoff_time:
                        recent_assessments.append({
                            'plan_id': plan_id,
                            **assessment
                        })
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid assessment {plan_id}: {e}")
                    continue

            if not recent_assessments:
                return {
                    'period_days': time_period.days,
                    'total_assessments': 0,
                    'message': 'No assessments in time period'
                }

            # Calculate statistics
            overall_scores = [a['overall_score'] for a in recent_assessments]
            dimension_scores = {
                'completeness': [a['completeness'] for a in recent_assessments],
                'consistency': [a['consistency'] for a in recent_assessments],
                'feasibility': [a['feasibility'] for a in recent_assessments],
                'dependency': [a['dependency'] for a in recent_assessments],
                'balance': [a['balance'] for a in recent_assessments]
            }

            def calculate_stats(scores: List[float]) -> Dict[str, Any]:
                """Calculate statistics for a list of scores."""
                if not scores:
                    return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
                return {
                    'avg': round(sum(scores) / len(scores), 3),
                    'min': round(min(scores), 3),
                    'max': round(max(scores), 3),
                    'count': len(scores)
                }

            trends = {
                'period_days': time_period.days,
                'total_assessments': len(recent_assessments),
                'overall_stats': calculate_stats(overall_scores),
                'dimension_stats': {
                    dim: calculate_stats(scores)
                    for dim, scores in dimension_scores.items()
                },
                'threshold_met_rate': sum(1 for a in recent_assessments if a['meets_thresholds']) / len(recent_assessments)
            }

            # Calculate trends (improvement/decline)
            if len(recent_assessments) >= 2:
                first_half = recent_assessments[:len(recent_assessments) // 2]
                second_half = recent_assessments[len(recent_assessments) // 2:]

                first_avg = sum(a['overall_score'] for a in first_half) / len(first_half)
                second_avg = sum(a['overall_score'] for a in second_half) / len(second_half)

                trends['overall_trend'] = {
                    'change': round(second_avg - first_avg, 3),
                    'direction': 'improving' if second_avg > first_avg else 'declining' if second_avg < first_avg else 'stable',
                    'percent_change': round(((second_avg - first_avg) / first_avg) * 100, 1) if first_avg > 0 else 0.0
                }

            return trends

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Failed to calculate trends: {e}")
            raise

    def identify_improvement_areas(self, min_assessments: int = 5, threshold: float = 0.7) -> List[str]:
        """
        Identify dimensions that consistently score low.

        Args:
            min_assessments: Minimum number of assessments before declaring a trend
            threshold: Quality threshold below which dimensions need improvement (default: 0.7)

        Returns:
            List of dimension names that need improvement, ordered by priority (lowest scores first)

        Raises:
            ValueError: If min_assessments or threshold are invalid
        """
        if min_assessments < 1:
            raise ValueError("min_assessments must be at least 1")

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        if len(self.assessments) < min_assessments:
            logger.debug(f"Insufficient assessments ({len(self.assessments)}) to identify improvement areas (need {min_assessments})")
            return []

        try:
            # Calculate average scores for each dimension
            dimension_avgs = {}
            for dimension in ['completeness', 'consistency', 'feasibility', 'dependency', 'balance']:
                try:
                    scores = [assessment[dimension] for assessment in self.assessments.values()]
                    if scores:
                        dimension_avgs[dimension] = sum(scores) / len(scores)
                except KeyError as e:
                    logger.warning(f"Dimension {dimension} not found in assessments: {e}")
                    continue

            # Identify dimensions below threshold
            low_dimensions = [
                (dim, avg)
                for dim, avg in dimension_avgs.items()
                if avg < threshold
            ]

            # Sort by score (lowest first)
            low_dimensions.sort(key=lambda x: x[1])

            # Generate improvement area descriptions
            improvement_areas = []
            dimension_names = {
                'completeness': 'Completeness (coverage of all problem aspects)',
                'consistency': 'Consistency (alignment and coherence)',
                'feasibility': 'Feasibility (resource and time constraints)',
                'dependency': 'Dependency Validity (execution flow)',
                'balance': 'Balance (complexity and effort distribution)'
            }

            for dim, avg in low_dimensions:
                improvement_areas.append(f"{dimension_names[dim]}: avg {avg:.3f}")

            logger.info(f"Identified {len(improvement_areas)} improvement areas")
            return improvement_areas

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Failed to identify improvement areas: {e}")
            raise

    def get_best_strategies(self, min_usage: int = 1) -> List[Dict[str, Any]]:
        """
        Analyze which strategies produce the best quality scores.

        Args:
            min_usage: Minimum number of times a strategy must be used to be included

        Returns:
            List of strategies with their performance metrics, sorted by average score:
            [
                {
                    'strategy': str,
                    'avg_score': float,
                    'usage_count': int,
                    'threshold_met_rate': float
                },
                ...
            ]

        Raises:
            ValueError: If min_usage is invalid
        """
        if min_usage < 1:
            raise ValueError("min_usage must be at least 1")

        try:
            strategy_performance = defaultdict(lambda: {
                'count': 0,
                'total_score': 0.0,
                'meets_thresholds': 0
            })

            for assessment in self.assessments.values():
                try:
                    strategy = assessment.get('strategy', 'unknown')
                    if not strategy:
                        strategy = 'unknown'

                    strategy_performance[strategy]['count'] += 1
                    strategy_performance[strategy]['total_score'] += assessment['overall_score']
                    if assessment['meets_thresholds']:
                        strategy_performance[strategy]['meets_thresholds'] += 1
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping invalid assessment in strategy analysis: {e}")
                    continue

            # Calculate averages
            results = []
            for strategy, data in strategy_performance.items():
                try:
                    if data['count'] >= min_usage:
                        results.append({
                            'strategy': strategy,
                            'avg_score': round(data['total_score'] / data['count'], 3),
                            'usage_count': data['count'],
                            'threshold_met_rate': round(data['meets_thresholds'] / data['count'], 3)
                        })
                except ZeroDivisionError:
                    # Should not happen due to count check, but handle gracefully
                    logger.warning(f"ZeroDivisionError for strategy {strategy}")
                    continue

            # Sort by average score
            results.sort(key=lambda x: x['avg_score'], reverse=True)

            logger.info(f"Analyzed {len(results)} strategies with min_usage={min_usage}")
            return results

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Failed to analyze best strategies: {e}")
            raise

    def get_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights from quality tracking.

        Returns:
            Dictionary containing actionable insights with the following structure:
            {
                'summary': {
                    'total_assessments': int,
                    'overall_average': float,
                    'trend_direction': str  # 'improving', 'declining', 'stable', 'unknown'
                },
                'improvement_areas': List[str],
                'recommended_strategies': List[Dict],
                'threshold_met_rate': float,
                'action_items': List[str]
            }
        """
        try:
            trends = self.get_trends()
            improvement_areas = self.identify_improvement_areas()
            best_strategies = self.get_best_strategies()

            insights = {
                'summary': {
                    'total_assessments': len(self.assessments),
                    'overall_average': trends.get('overall_stats', {}).get('avg', 0.0),
                    'trend_direction': trends.get('overall_trend', {}).get('direction', 'unknown')
                },
                'improvement_areas': improvement_areas,
                'recommended_strategies': best_strategies[:3] if best_strategies else [],
                'threshold_met_rate': trends.get('threshold_met_rate', 0.0),
                'action_items': []
            }

            # Generate action items based on insights
            if improvement_areas:
                insights['action_items'].append(
                    f"Focus on improving: {', '.join(improvement_areas[:2])}"
                )

            if best_strategies and len(best_strategies) > 1:
                top_strategy = best_strategies[0]['strategy']
                insights['action_items'].append(
                    f"Consider using '{top_strategy}' strategy more often (avg score: {best_strategies[0]['avg_score']:.3f})"
                )

            if trends.get('overall_trend', {}).get('direction') == 'declining':
                insights['action_items'].append(
                    "Quality scores are declining - review recent decomposition changes"
                )

            threshold_rate = trends.get('threshold_met_rate', 0.0)
            if threshold_rate < 0.8:
                insights['action_items'].append(
                    f"Only {threshold_rate:.0%} of decompositions meet quality threshold - review assessment criteria"
                )

            logger.info(f"Generated {len(insights['action_items'])} action items from insights")
            return insights

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to generate insights: {e}")
            raise

    def get_dimension_history(self, dimension: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get historical scores for a specific dimension.

        Args:
            dimension: One of 'completeness', 'consistency', 'feasibility', 'dependency', 'balance'
            limit: Maximum number of records to return (must be positive)

        Returns:
            List of historical score records sorted by timestamp (most recent first)

        Raises:
            ValueError: If dimension is invalid or limit is invalid
        """
        valid_dimensions = ['completeness', 'consistency', 'feasibility', 'dependency', 'balance']

        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension '{dimension}'. Must be one of {valid_dimensions}")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        if dimension not in self.dimension_history:
            logger.debug(f"No history found for dimension: {dimension}")
            return []

        try:
            history = self.dimension_history[dimension]
            # Return most recent first
            sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
            return sorted_history[:limit]

        except (KeyError, TypeError) as e:
            logger.error(f"Failed to retrieve dimension history for {dimension}: {e}")
            raise

    def clear_old_assessments(self, days_to_keep: int = 90) -> int:
        """
        Remove assessments older than specified days.

        Args:
            days_to_keep: Number of days of history to retain (must be positive)

        Returns:
            Number of assessments that were removed

        Raises:
            ValueError: If days_to_keep is invalid
        """
        if days_to_keep < 1:
            raise ValueError("days_to_keep must be at least 1")

        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)

            to_remove = []
            for plan_id, assessment in self.assessments.items():
                try:
                    assessment_time = datetime.fromisoformat(assessment['timestamp'])
                    if assessment_time < cutoff_time:
                        to_remove.append(plan_id)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid assessment {plan_id} during cleanup: {e}")
                    continue

            for plan_id in to_remove:
                del self.assessments[plan_id]

            # Clean up dimension history
            for dimension in self.dimension_history:
                try:
                    self.dimension_history[dimension] = [
                        entry for entry in self.dimension_history[dimension]
                        if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
                    ]
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to clean dimension history for {dimension}: {e}")
                    continue

            if to_remove:
                self._save_to_storage()
                logger.info(f"Cleared {len(to_remove)} old assessments (kept last {days_to_keep} days)")

            return len(to_remove)

        except (KeyError, ValueError, OSError) as e:
            logger.error(f"Failed to clear old assessments: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about quality tracking.

        Returns:
            Dictionary containing various statistics with the following structure:
            {
                'total_assessments': int,
                'time_span_days': int,
                'overall_stats': {
                    'avg': float,
                    'min': float,
                    'max': float,
                    'stddev': float
                },
                'threshold_met_count': int,
                'threshold_met_rate': float
            }
            Or if no assessments: {'total_assessments': 0, 'message': str}
        """
        try:
            if not self.assessments:
                return {
                    'total_assessments': 0,
                    'message': 'No assessments recorded yet'
                }

            overall_scores = []
            timestamps = []

            for assessment in self.assessments.values():
                try:
                    overall_scores.append(assessment['overall_score'])
                    timestamps.append(datetime.fromisoformat(assessment['timestamp']))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid assessment in statistics: {e}")
                    continue

            if not overall_scores:
                return {
                    'total_assessments': 0,
                    'message': 'No valid assessments found'
                }

            # Calculate time range
            time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else timedelta(0)

            # Calculate statistics
            avg_score = sum(overall_scores) / len(overall_scores)
            variance = sum((x - avg_score) ** 2 for x in overall_scores) / len(overall_scores)
            stddev = variance ** 0.5 if len(overall_scores) > 1 else 0.0

            threshold_met_count = sum(
                1 for a in self.assessments.values()
                if a.get('meets_thresholds', False)
            )

            return {
                'total_assessments': len(self.assessments),
                'time_span_days': time_range.days,
                'overall_stats': {
                    'avg': round(avg_score, 3),
                    'min': round(min(overall_scores), 3),
                    'max': round(max(overall_scores), 3),
                    'stddev': round(stddev, 3)
                },
                'threshold_met_count': threshold_met_count,
                'threshold_met_rate': round(threshold_met_count / len(self.assessments), 3)
            }

        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Failed to calculate statistics: {e}")
            raise


def create_mock_quality_scores(
    overall_score: float = 0.75,
    completeness: float = 0.8,
    consistency: float = 0.75,
    feasibility: float = 0.7,
    dependency: float = 0.85,
    balance: float = 0.7
) -> EnhancedQualityScores:
    """
    Create a mock EnhancedQualityScores object for testing.

    Args:
        overall_score: Overall quality score (0-1)
        completeness: Completeness score (0-1)
        consistency: Consistency score (0-1)
        feasibility: Feasibility score (0-1)
        dependency: Dependency score (0-1)
        balance: Balance score (0-1)

    Returns:
        EnhancedQualityScores object with provided values
    """
    return EnhancedQualityScores(
        overall_score=overall_score,
        meets_thresholds=overall_score >= 0.7,
        completeness_score=completeness,
        consistency_score=consistency,
        feasibility_score=feasibility,
        dependency_score=dependency,
        balance_score=balance,
        completeness_details={'score': completeness, 'issues': []},
        consistency_details={'score': consistency, 'issues': []},
        feasibility_details={'score': feasibility, 'issues': []},
        dependency_details={'score': dependency, 'issues': []},
        balance_details={'score': balance, 'issues': []},
        improvement_recommendations=[],
        critical_issues=[],
        validation_checkpoints=[],
        timestamp=datetime.now()
    )


# Usage Examples
if __name__ == "__main__":
    import tempfile
    import os

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example 1: Basic usage with in-memory storage
    print("=== Example 1: Basic Quality Tracking ===")
    tracker = QualityTracker()

    # Record some assessments
    for i in range(10):
        scores = create_mock_quality_scores(
            overall_score=0.65 + (i * 0.03),  # Improving over time
            completeness=0.7 + (i * 0.02),
            consistency=0.75 + (i * 0.01),
            feasibility=0.6 + (i * 0.03),
            dependency=0.8,
            balance=0.7
        )
        tracker.record_assessment(
            plan_id=f"plan_{i}",
            scores=scores,
            problem_type="algorithm_design",
            strategy="hierarchical"
        )

    # Get trends
    trends = tracker.get_trends()
    print(f"Trends over last 30 days:")
    print(f"  Total assessments: {trends['total_assessments']}")
    print(f"  Average score: {trends['overall_stats']['avg']:.3f}")
    if 'overall_trend' in trends:
        print(f"  Trend direction: {trends['overall_trend']['direction']}")
        print(f"  Change: {trends['overall_trend']['change']:.3f}")

    # Example 2: Identify improvement areas
    print("\n=== Example 2: Improvement Areas ===")
    improvement_areas = tracker.identify_improvement_areas()
    if improvement_areas:
        print("Areas needing improvement:")
        for area in improvement_areas:
            print(f"  - {area}")
    else:
        print("All dimensions meet quality threshold!")

    # Example 3: Best strategies
    print("\n=== Example 3: Strategy Analysis ===")
    # Add some assessments with different strategies
    for i in range(5):
        scores = create_mock_quality_scores(overall_score=0.85)
        tracker.record_assessment(
            plan_id=f"plan_ml_{i}",
            scores=scores,
            strategy="machine_learning"
        )

    strategies = tracker.get_best_strategies()
    print("Strategy performance:")
    for strat in strategies:
        print(f"  {strat['strategy']}: avg={strat['avg_score']:.3f}, "
              f"usage={strat['usage_count']}, "
              f"threshold_rate={strat['threshold_met_rate']:.3f}")

    # Example 4: Get insights
    print("\n=== Example 4: Comprehensive Insights ===")
    insights = tracker.get_insights()
    print(f"Total assessments: {insights['summary']['total_assessments']}")
    print(f"Overall average: {insights['summary']['overall_average']:.3f}")
    print(f"Trend: {insights['summary']['trend_direction']}")
    print(f"Threshold met rate: {insights['threshold_met_rate']:.1%}")
    print("\nAction items:")
    for item in insights['action_items']:
        print(f"  - {item}")

    # Example 5: Persistent storage
    print("\n=== Example 5: Persistent Storage ===")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        storage_path = f.name

    try:
        # Create tracker with storage
        persistent_tracker = QualityTracker(storage_path=storage_path)

        # Record an assessment
        scores = create_mock_quality_scores(overall_score=0.82)
        persistent_tracker.record_assessment(
            plan_id="persistent_plan_1",
            scores=scores,
            problem_type="data_processing",
            strategy="parallel"
        )

        # Create new tracker instance to verify persistence
        new_tracker = QualityTracker(storage_path=storage_path)
        stats = new_tracker.get_statistics()
        print(f"Loaded {stats['total_assessments']} assessments from storage")

    finally:
        # Clean up
        if os.path.exists(storage_path):
            os.remove(storage_path)
            print("Cleaned up temporary storage")

    # Example 6: Dimension history
    print("\n=== Example 6: Dimension History ===")
    completeness_history = tracker.get_dimension_history('completeness', limit=5)
    print("Recent completeness scores:")
    for entry in completeness_history:
        print(f"  {entry['timestamp']}: {entry['score']:.3f} (plan: {entry['plan_id']})")

    # Example 7: Statistics
    print("\n=== Example 7: Overall Statistics ===")
    stats = tracker.get_statistics()
    print(f"Total assessments: {stats['total_assessments']}")
    print(f"Time span: {stats['time_span_days']} days")
    print(f"Score statistics:")
    print(f"  Average: {stats['overall_stats']['avg']:.3f}")
    print(f"  Min: {stats['overall_stats']['min']:.3f}")
    print(f"  Max: {stats['overall_stats']['max']:.3f}")
    print(f"  StdDev: {stats['overall_stats']['stddev']:.3f}")
    print(f"Threshold met: {stats['threshold_met_count']}/{stats['total_assessments']} "
          f"({stats['threshold_met_rate']:.1%})")

    print("\n=== All Examples Complete ===")
