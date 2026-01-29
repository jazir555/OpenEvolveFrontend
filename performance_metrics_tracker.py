"""
Performance Metrics Tracking System for Sovereign Decomposition

This module implements comprehensive performance tracking at multiple levels:
- Individual sub-problem performance
- Strategy performance
- Team performance
- Domain performance
- Overall system performance

TRACKING CAPABILITIES:
- Decomposition metrics
- Solution generation metrics
- Validation metrics
- Trend analysis
- Performance reporting
"""

import json
import logging
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from sovereign_data_models import (
    DecompositionPlan, SolutionAttempt, ProblemDefinition,
    SubProblem, ValidationResult, ComplexityScore,
    DecompositionStrategy, generate_id
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformanceMetrics:
    """Performance metrics for a decomposition strategy."""
    strategy: str
    domain: str
    problem_type: str

    # Usage
    total_uses: int = 0
    usage_frequency: float = 0.0  # uses per day

    # Quality
    avg_quality_score: float = 0.0
    min_quality_score: float = 0.0
    max_quality_score: float = 0.0
    quality_std_dev: float = 0.0

    # Success
    success_rate: float = 0.0  # % passing validation
    avg_revision_count: float = 0.0

    # Performance
    avg_decomposition_time: float = 0.0
    avg_solution_time: float = 0.0

    # Trends
    quality_trend: str = "stable"  # "improving", "stable", "declining"
    usage_trend: str = "stable"

    # Time
    first_used: str = ""
    last_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformanceMetrics':
        return cls(**data)


@dataclass
class TeamPerformanceMetrics:
    """Performance metrics for a team."""
    team_id: str
    team_name: str

    # Assignments
    total_assignments: int = 0
    assignments_by_role: Dict[str, int] = field(default_factory=dict)

    # Performance
    avg_quality_score: float = 0.0
    success_rate: float = 0.0
    avg_completion_time: float = 0.0

    # By problem type
    performance_by_problem_type: Dict[str, float] = field(default_factory=dict)
    performance_by_domain: Dict[str, float] = field(default_factory=dict)

    # Trends
    recent_performance: List[float] = field(default_factory=list)
    trend: str = "stable"  # "improving", "stable", "declining"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamPerformanceMetrics':
        return cls(**data)


@dataclass
class DomainPerformanceMetrics:
    """Performance metrics for a domain."""
    domain: str

    # Volume
    total_problems: int = 0
    problems_per_month: float = 0.0

    # Quality
    avg_quality_score: float = 0.0
    success_rate: float = 0.0

    # Common patterns
    common_strategies: List[str] = field(default_factory=list)
    avg_sub_problem_count: float = 0.0
    avg_complexity: float = 0.0

    # Performance
    avg_decomposition_time: float = 0.0
    avg_solution_time: float = 0.0

    # Trends
    quality_trend: str = "stable"
    volume_trend: str = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainPerformanceMetrics':
        return cls(**data)


@dataclass
class OverallPerformanceMetrics:
    """Overall system performance metrics."""

    # Volume
    total_problems_processed: int = 0
    total_sub_problems: int = 0

    # Quality
    overall_quality_score: float = 0.0
    overall_success_rate: float = 0.0

    # Performance
    avg_decomposition_time: float = 0.0
    avg_solution_time: float = 0.0
    avg_validation_time: float = 0.0
    total_cycle_time: float = 0.0

    # Breakdown by domain
    domain_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Breakdown by strategy
    strategy_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Trends
    quality_trend: str = "stable"
    throughput_trend: str = "stable"

    # Time period
    period_start: str = ""
    period_end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OverallPerformanceMetrics':
        return cls(**data)


@dataclass
class TrendAnalysis:
    """Trend analysis for a metric."""
    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1, how strong is the trend
    confidence: float  # 0-1

    # Statistics
    mean: float
    std_dev: float
    min_value: float
    max_value: float

    # Prediction
    predicted_next_value: float
    prediction_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrendAnalysis':
        return cls(**data)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    generated_at: str
    time_period: str

    # Overall metrics
    overall_metrics: OverallPerformanceMetrics

    # Strategy breakdowns
    strategy_metrics: Dict[str, StrategyPerformanceMetrics]

    # Team breakdowns
    team_metrics: Dict[str, TeamPerformanceMetrics]

    # Domain breakdowns
    domain_metrics: Dict[str, DomainPerformanceMetrics]

    # Improvement areas
    improvement_areas: List[str]

    # Trend analyses
    trend_analyses: Dict[str, TrendAnalysis]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at,
            'time_period': self.time_period,
            'overall_metrics': self.overall_metrics.to_dict(),
            'strategy_metrics': {k: v.to_dict() for k, v in self.strategy_metrics.items()},
            'team_metrics': {k: v.to_dict() for k, v in self.team_metrics.items()},
            'domain_metrics': {k: v.to_dict() for k, v in self.domain_metrics.items()},
            'improvement_areas': self.improvement_areas,
            'trend_analyses': {k: v.to_dict() for k, v in self.trend_analyses.items()}
        }


class PerformanceMetricsTracker:
    """
    Comprehensive performance tracking system.

    Tracks metrics at multiple levels:
    - Individual sub-problem performance
    - Strategy performance
    - Team performance
    - Domain performance
    - Overall system performance
    """

    def __init__(self, storage_path: str = "performance_metrics.json"):
        """
        Initialize with persistent storage.

        Args:
            storage_path: Path to JSON file for metrics storage
        """
        self.storage_path = Path(storage_path)
        self.metrics: Dict[str, Any] = {
            'decomposition_metrics': [],
            'solution_metrics': [],
            'validation_metrics': [],
            'strategy_metrics': {},
            'team_metrics': {},
            'domain_metrics': {},
            'overall_metrics': {}
        }
        self._load_metrics()
        logger.info("PerformanceMetricsTracker initialized")

    def _load_metrics(self):
        """Load metrics from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {self.storage_path}")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.error(f"Failed to load metrics: {e}", exc_info=True)
                self.metrics = {
                    'decomposition_metrics': [],
                    'solution_metrics': [],
                    'validation_metrics': [],
                    'strategy_metrics': {},
                    'team_metrics': {},
                    'domain_metrics': {},
                    'overall_metrics': {}
                }

    def _save_metrics(self):
        """Save metrics to persistent storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved metrics to {self.storage_path}")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save metrics: {e}", exc_info=True)

    def record_decomposition_metrics(
        self,
        plan: DecompositionPlan,
        problem: ProblemDefinition,
        decomposition_time: float
    ):
        """
        Record metrics for a decomposition.

        Args:
            plan: The decomposition plan created
            problem: The problem being decomposed
            decomposition_time: Time taken to decompose (seconds)
        """
        try:
            domain = problem.domain_context.domain
            problem_type = problem.problem_type.value if hasattr(problem.problem_type, 'value') else str(problem.problem_type)
            strategy = plan.strategy.value if hasattr(plan.strategy, 'value') else str(plan.strategy)

            # Record decomposition metrics
            decomp_metric = {
                'problem_id': problem.id,
                'plan_id': plan.id,
                'domain': domain,
                'problem_type': problem_type,
                'strategy': strategy,
                'decomposition_time': decomposition_time,
                'num_sub_problems': len(plan.sub_problems),
                'timestamp': datetime.now().isoformat()
            }

            self.metrics['decomposition_metrics'].append(decomp_metric)

            # Update strategy metrics
            strategy_key = f"{strategy}_{domain}_{problem_type}"
            if strategy_key not in self.metrics['strategy_metrics']:
                self.metrics['strategy_metrics'][strategy_key] = {
                    'strategy': strategy,
                    'domain': domain,
                    'problem_type': problem_type,
                    'total_uses': 0,
                    'decomposition_times': [],
                    'quality_scores': [],
                    'success_count': 0,
                    'first_used': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat()
                }

            strat_metrics = self.metrics['strategy_metrics'][strategy_key]
            strat_metrics['total_uses'] += 1
            strat_metrics['decomposition_times'].append(decomposition_time)
            strat_metrics['last_used'] = datetime.now().isoformat()

            # Update domain metrics
            if domain not in self.metrics['domain_metrics']:
                self.metrics['domain_metrics'][domain] = {
                    'domain': domain,
                    'total_problems': 0,
                    'sub_problem_counts': [],
                    'decomposition_times': [],
                    'quality_scores': [],
                    'strategies_used': defaultdict(int)
                }

            dom_metrics = self.metrics['domain_metrics'][domain]
            dom_metrics['total_problems'] += 1
            dom_metrics['sub_problem_counts'].append(len(plan.sub_problems))
            dom_metrics['decomposition_times'].append(decomposition_time)
            dom_metrics['strategies_used'][strategy] += 1

            # Update overall metrics
            if 'total_problems_processed' not in self.metrics['overall_metrics']:
                self.metrics['overall_metrics']['total_problems_processed'] = 0
            self.metrics['overall_metrics']['total_problems_processed'] += 1

            self._save_metrics()
            logger.debug(f"Recorded decomposition metrics for {problem.id}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to record decomposition metrics: {e}", exc_info=True)

    def record_solution_metrics(
        self,
        sub_problem_id: str,
        solution: SolutionAttempt,
        validation: Optional[ValidationResult],
        generation_time: float
    ):
        """
        Record metrics for a solution.

        Args:
            sub_problem_id: ID of the sub-problem
            solution: The solution attempt
            validation: Validation results (optional)
            generation_time: Time taken to generate solution (seconds)
        """
        try:
            # Record solution metrics
            solution_metric = {
                'sub_problem_id': sub_problem_id,
                'solution_id': solution.id,
                'team_id': solution.team_id,
                'approach': solution.approach,
                'confidence_score': solution.confidence_score,
                'generation_time': generation_time,
                'passed_validation': validation.passed if validation and hasattr(validation, 'passed') else None,
                'validation_score': validation.score if validation else None,
                'timestamp': datetime.now().isoformat()
            }

            self.metrics['solution_metrics'].append(solution_metric)

            # Update team metrics
            team_id = solution.team_id
            if team_id not in self.metrics['team_metrics']:
                self.metrics['team_metrics'][team_id] = {
                    'team_id': team_id,
                    'total_assignments': 0,
                    'quality_scores': [],
                    'completion_times': [],
                    'success_count': 0
                }

            team_metrics = self.metrics['team_metrics'][team_id]
            team_metrics['total_assignments'] += 1
            team_metrics['quality_scores'].append(solution.confidence_score)
            team_metrics['completion_times'].append(generation_time)
            if validation and hasattr(validation, 'passed') and validation.passed:
                team_metrics['success_count'] += 1

            self._save_metrics()
            logger.debug(f"Recorded solution metrics for {sub_problem_id}")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to record solution metrics: {e}", exc_info=True)

    def record_validation_metrics(
        self,
        validation: ValidationResult,
        validation_time: float
    ):
        """
        Record metrics for validation.

        Args:
            validation: The validation result
            validation_time: Time taken to validate (seconds)
        """
        try:
            # Record validation metrics
            validation_metric = {
                'validator': validation.validator,
                'passed': validation.passed,
                'score': validation.score,
                'validation_time': validation_time,
                'timestamp': datetime.now().isoformat()
            }

            self.metrics['validation_metrics'].append(validation_metric)
            self._save_metrics()
            logger.debug(f"Recorded validation metrics")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to record validation metrics: {e}", exc_info=True)

    def get_strategy_performance(
        self,
        strategy: str,
        domain: str = None,
        problem_type: str = None
    ) -> Optional[StrategyPerformanceMetrics]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy: The strategy name
            domain: Optional domain filter
            problem_type: Optional problem type filter

        Returns:
            StrategyPerformanceMetrics or None
        """
        try:
            # Find matching strategy metrics
            matching_keys = []
            for key in self.metrics['strategy_metrics'].keys():
                parts = key.split('_')
                if len(parts) >= 3:
                    key_strategy = parts[0]
                    key_domain = parts[1]
                    key_problem_type = '_'.join(parts[2:])

                    if key_strategy == strategy:
                        if (domain is None or key_domain == domain) and \
                           (problem_type is None or key_problem_type == problem_type):
                            matching_keys.append(key)

            if not matching_keys:
                return None

            # Aggregate metrics from all matching keys
            total_uses = 0
            decomp_times = []
            quality_scores = []
            first_used = None
            last_used = None

            for key in matching_keys:
                data = self.metrics['strategy_metrics'][key]
                total_uses += data['total_uses']
                decomp_times.extend(data['decomposition_times'])
                quality_scores.extend(data['quality_scores'])

                if not first_used or data['first_used'] < first_used:
                    first_used = data['first_used']
                if not last_used or data['last_used'] > last_used:
                    last_used = data['last_used']

            # Calculate metrics
            avg_decomp_time = statistics.mean(decomp_times) if decomp_times else 0.0
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
            success_rate = 0.0  # Would need more detailed tracking

            # Calculate quality trend
            quality_trend = "stable"
            if len(quality_scores) >= 5:
                recent = quality_scores[-5:]
                earlier = quality_scores[:-5] if len(quality_scores) > 5 else quality_scores[:5]
                if statistics.mean(recent) > statistics.mean(earlier) + 0.1:
                    quality_trend = "improving"
                elif statistics.mean(recent) < statistics.mean(earlier) - 0.1:
                    quality_trend = "declining"

            return StrategyPerformanceMetrics(
                strategy=strategy,
                domain=domain or 'all',
                problem_type=problem_type or 'all',
                total_uses=total_uses,
                avg_decomposition_time=avg_decomp_time,
                avg_quality_score=avg_quality,
                min_quality_score=min(quality_scores) if quality_scores else 0.0,
                max_quality_score=max(quality_scores) if quality_scores else 0.0,
                quality_std_dev=statistics.stdev(quality_scores) if len(quality_scores) >= 2 else 0.0,
                success_rate=success_rate,
                quality_trend=quality_trend,
                first_used=first_used or '',
                last_used=last_used or ''
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to get strategy performance: {e}", exc_info=True)
            return None

    def get_team_performance(
        self,
        team_id: str
    ) -> Optional[TeamPerformanceMetrics]:
        """
        Get performance metrics for a team.

        Args:
            team_id: The team ID

        Returns:
            TeamPerformanceMetrics or None
        """
        try:
            if team_id not in self.metrics['team_metrics']:
                return None

            data = self.metrics['team_metrics'][team_id]
            quality_scores = data['quality_scores']
            completion_times = data['completion_times']

            # Calculate trend
            trend = "stable"
            if len(quality_scores) >= 5:
                recent = quality_scores[-5:]
                earlier = quality_scores[:-5] if len(quality_scores) > 5 else quality_scores[:5]
                if statistics.mean(recent) > statistics.mean(earlier) + 0.1:
                    trend = "improving"
                elif statistics.mean(recent) < statistics.mean(earlier) - 0.1:
                    trend = "declining"

            return TeamPerformanceMetrics(
                team_id=team_id,
                team_name=team_id,  # Would need to store actual name
                total_assignments=data['total_assignments'],
                avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
                success_rate=data['success_count'] / data['total_assignments'] if data['total_assignments'] > 0 else 0.0,
                avg_completion_time=statistics.mean(completion_times) if completion_times else 0.0,
                recent_performance=quality_scores[-10:] if len(quality_scores) > 10 else quality_scores,
                trend=trend
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to get team performance: {e}", exc_info=True)
            return None

    def get_domain_performance(
        self,
        domain: str
    ) -> Optional[DomainPerformanceMetrics]:
        """
        Get performance metrics for a domain.

        Args:
            domain: The domain name

        Returns:
            DomainPerformanceMetrics or None
        """
        try:
            if domain not in self.metrics['domain_metrics']:
                return None

            data = self.metrics['domain_metrics'][domain]

            # Get most common strategies
            strategies = data['strategies_used']
            if isinstance(strategies, dict):
                common_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)[:3]
                common_strategies = [s[0] for s in common_strategies]
            else:
                common_strategies = []

            return DomainPerformanceMetrics(
                domain=domain,
                total_problems=data['total_problems'],
                avg_quality_score=statistics.mean(data['quality_scores']) if data['quality_scores'] else 0.0,
                success_rate=0.0,  # Would need more detailed tracking
                common_strategies=common_strategies,
                avg_sub_problem_count=statistics.mean(data['sub_problem_counts']) if data['sub_problem_counts'] else 0.0,
                avg_decomposition_time=statistics.mean(data['decomposition_times']) if data['decomposition_times'] else 0.0
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to get domain performance: {e}", exc_info=True)
            return None

    def get_overall_performance(self) -> OverallPerformanceMetrics:
        """Get overall system performance metrics."""
        try:
            overall = self.metrics['overall_metrics']

            # Calculate averages from stored metrics
            decomp_times = [m['decomposition_time'] for m in self.metrics['decomposition_metrics']]
            solution_times = [m['generation_time'] for m in self.metrics['solution_metrics']]

            return OverallPerformanceMetrics(
                total_problems_processed=overall.get('total_problems_processed', 0),
                total_sub_problems=len(self.metrics['solution_metrics']),
                overall_quality_score=0.0,  # Would need aggregation
                overall_success_rate=0.0,  # Would need aggregation
                avg_decomposition_time=statistics.mean(decomp_times) if decomp_times else 0.0,
                avg_solution_time=statistics.mean(solution_times) if solution_times else 0.0,
                domain_breakdown={k: self.get_domain_performance(k).to_dict()
                                 for k in self.metrics['domain_metrics'].keys()},
                strategy_breakdown={k: v for k, v in self.metrics['strategy_metrics'].items()}
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to get overall performance: {e}", exc_info=True)
            return OverallPerformanceMetrics()

    def generate_performance_report(
        self,
        time_period: str = "all"
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            time_period: Time period for report ("all", "week", "month", "year")

        Returns:
            PerformanceReport
        """
        try:
            # Filter metrics by time period
            cutoff_time = None
            if time_period == "week":
                cutoff_time = datetime.now() - timedelta(weeks=1)
            elif time_period == "month":
                cutoff_time = datetime.now() - timedelta(days=30)
            elif time_period == "year":
                cutoff_time = datetime.now() - timedelta(days=365)

            # Generate metrics
            overall_metrics = self.get_overall_performance()

            # Get strategy metrics
            strategy_metrics = {}
            for key in self.metrics['strategy_metrics'].keys():
                parts = key.split('_')
                if len(parts) >= 3:
                    strategy = parts[0]
                    metrics = self.get_strategy_performance(strategy)
                    if metrics:
                        strategy_metrics[key] = metrics

            # Get team metrics
            team_metrics = {}
            for team_id in self.metrics['team_metrics'].keys():
                metrics = self.get_team_performance(team_id)
                if metrics:
                    team_metrics[team_id] = metrics

            # Get domain metrics
            domain_metrics = {}
            for domain in self.metrics['domain_metrics'].keys():
                metrics = self.get_domain_performance(domain)
                if metrics:
                    domain_metrics[domain] = metrics

            # Identify improvement areas
            improvement_areas = self.identify_improvement_areas()

            # Generate trend analyses
            trend_analyses = {}
            if overall_metrics.total_problems_processed > 0:
                trend_analyses['quality'] = self.calculate_trends('quality_score')
                trend_analyses['decomposition_time'] = self.calculate_trends('decomposition_time')

            return PerformanceReport(
                report_id=generate_id("perf_report"),
                generated_at=datetime.now().isoformat(),
                time_period=time_period,
                overall_metrics=overall_metrics,
                strategy_metrics=strategy_metrics,
                team_metrics=team_metrics,
                domain_metrics=domain_metrics,
                improvement_areas=improvement_areas,
                trend_analyses=trend_analyses
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to generate performance report: {e}", exc_info=True)
            return PerformanceReport(
                report_id=generate_id("perf_report"),
                generated_at=datetime.now().isoformat(),
                time_period=time_period,
                overall_metrics=OverallPerformanceMetrics(),
                strategy_metrics={},
                team_metrics={},
                domain_metrics={},
                improvement_areas=[],
                trend_analyses={}
            )

    def identify_improvement_areas(self) -> List[str]:
        """
        Identify areas needing improvement.

        Returns:
            List of improvement recommendations
        """
        improvements = []

        try:
            overall = self.get_overall_performance()

            # Check overall quality
            if overall.overall_quality_score < 0.7:
                improvements.append("Overall solution quality is below target. Consider reviewing decomposition strategies.")

            # Check decomposition time
            if overall.avg_decomposition_time > 30.0:
                improvements.append("Decomposition time is high. Consider optimization or caching.")

            # Check for struggling strategies
            for key, data in self.metrics['strategy_metrics'].items():
                if data['quality_scores']:
                    avg_quality = statistics.mean(data['quality_scores'])
                    if avg_quality < 0.6:
                        strategy = data['strategy']
                        improvements.append(f"Strategy '{strategy}' shows low quality. Consider reviewing or replacing.")

            # Check for struggling teams
            for team_id, data in self.metrics['team_metrics'].items():
                if data['quality_scores']:
                    avg_quality = statistics.mean(data['quality_scores'])
                    if avg_quality < 0.6:
                        improvements.append(f"Team '{team_id}' shows low performance. Consider additional training or support.")

            # Check domain-specific issues
            for domain, data in self.metrics['domain_metrics'].items():
                if data['quality_scores']:
                    avg_quality = statistics.mean(data['quality_scores'])
                    if avg_quality < 0.6:
                        improvements.append(f"Domain '{domain}' shows lower quality. Consider domain-specific optimizations.")

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to identify improvement areas: {e}", exc_info=True)

        return improvements

    def calculate_trends(
        self,
        metric_name: str,
        window_size: int = 10
    ) -> TrendAnalysis:
        """
        Calculate trends for a metric over time.

        Args:
            metric_name: Name of metric to analyze
            window_size: Size of moving window for trend calculation

        Returns:
            TrendAnalysis
        """
        try:
            # Extract metric values over time
            values = []
            if metric_name == 'quality_score':
                values = [m['confidence_score'] for m in self.metrics['solution_metrics']
                         if 'confidence_score' in m]
            elif metric_name == 'decomposition_time':
                values = [m['decomposition_time'] for m in self.metrics['decomposition_metrics']
                         if 'decomposition_time' in m]

            if len(values) < 3:
                return TrendAnalysis(
                    metric_name=metric_name,
                    current_value=values[-1] if values else 0.0,
                    trend_direction="stable",
                    trend_strength=0.0,
                    confidence=0.0,
                    mean=0.0,
                    std_dev=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    predicted_next_value=0.0,
                    prediction_confidence=0.0
                )

            # Calculate statistics
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) >= 2 else 0.0
            min_val = min(values)
            max_val = max(values)
            current = values[-1]

            # Determine trend direction
            if len(values) >= window_size:
                recent = values[-window_size:]
                earlier = values[-(window_size * 2):-window_size] if len(values) >= window_size * 2 else values[:-window_size]

                if len(earlier) > 0:
                    recent_avg = statistics.mean(recent)
                    earlier_avg = statistics.mean(earlier)
                    diff = recent_avg - earlier_avg
                    threshold = std_dev * 0.5

                    if diff > threshold:
                        direction = "increasing"
                    elif diff < -threshold:
                        direction = "decreasing"
                    else:
                        direction = "stable"

                    # Calculate trend strength
                    strength = min(abs(diff) / (std_dev + 0.001), 1.0)
                else:
                    direction = "stable"
                    strength = 0.0
            else:
                direction = "stable"
                strength = 0.0

            # Simple prediction (linear extrapolation)
            predicted = current + (current - values[-2]) if len(values) >= 2 else current
            prediction_confidence = strength if len(values) >= window_size else 0.0

            return TrendAnalysis(
                metric_name=metric_name,
                current_value=current,
                trend_direction=direction,
                trend_strength=strength,
                confidence=min(len(values) / 50.0, 1.0),  # More data = more confidence
                mean=mean,
                std_dev=std_dev,
                min_value=min_val,
                max_value=max_val,
                predicted_next_value=predicted,
                prediction_confidence=prediction_confidence
            )

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to calculate trends: {e}", exc_info=True)
            return TrendAnalysis(
                metric_name=metric_name,
                current_value=0.0,
                trend_direction="stable",
                trend_strength=0.0,
                confidence=0.0,
                mean=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                predicted_next_value=0.0,
                prediction_confidence=0.0
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'total_decompositions': len(self.metrics['decomposition_metrics']),
            'total_solutions': len(self.metrics['solution_metrics']),
            'total_validations': len(self.metrics['validation_metrics']),
            'strategies_tracked': len(self.metrics['strategy_metrics']),
            'teams_tracked': len(self.metrics['team_metrics']),
            'domains_tracked': len(self.metrics['domain_metrics']),
            'last_updated': datetime.now().isoformat()
        }
