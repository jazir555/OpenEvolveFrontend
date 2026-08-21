"""
Blue Team Performance Tracking and Analytics for OpenEvolve
Comprehensive system for tracking, analyzing, and reporting Blue Team performance.

This module provides:
- Individual team member performance tracking
- Solution success rate monitoring
- Time-to-solve metrics
- Quality improvement scoring
- Patch effectiveness analysis
- Team-level analytics
- Workload distribution tracking
- Bottleneck identification
- Predictive performance modeling
- Automated performance alerts
- Performance-based team selection
"""
from __future__ import annotations



import os
import json
import csv
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    SUCCESS_RATE = "success_rate"
    TIME_TO_SOLVE = "time_to_solve"
    QUALITY_SCORE = "quality_score"
    PATCH_EFFECTIVENESS = "patch_effectiveness"
    RELIABILITY = "reliability"
    THROUGHPUT = "throughput"
    CONSISTENCY = "consistency"


class AlertLevel(Enum):
    """Severity levels for performance alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SpecializationType(Enum):
    """Types of Blue Team specializations"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


@dataclass
class PerformanceMetric:
    """A single performance metric measurement"""
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    team_member_id: Optional[str] = None
    task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'team_member_id': self.team_member_id,
            'task_id': self.task_id
        }


@dataclass
class TaskPerformanceRecord:
    """Record of a single task's performance"""
    task_id: str
    team_member_id: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    quality_score: float
    time_to_solve: Optional[float]
    specializations: List[SpecializationType]
    difficulty_level: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'team_member_id': self.team_member_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'success': self.success,
            'quality_score': self.quality_score,
            'time_to_solve': self.time_to_solve,
            'specializations': [s.value for s in self.specializations],
            'difficulty_level': self.difficulty_level,
            'context': self.context
        }


@dataclass
class PerformanceAlert:
    """A performance alert"""
    alert_id: str
    level: AlertLevel
    metric_type: PerformanceMetricType
    message: str
    timestamp: datetime
    team_member_id: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'metric_type': self.metric_type.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'team_member_id': self.team_member_id,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
            'recommendations': self.recommendations
        }


class PerformanceMetrics:
    """
    Core performance metrics tracking for Blue Team operations.

    Tracks:
    - Solution success rate
    - Average time to solve
    - Quality improvement scores
    - Patch effectiveness
    - Team member reliability
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize performance metrics tracking.

        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), 'data', 'performance_metrics'
        )
        os.makedirs(self.storage_path, exist_ok=True)

        self.metrics: List[PerformanceMetric] = []
        self.task_records: List[TaskPerformanceRecord] = []
        self.lock = threading.Lock()

        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from storage"""
        try:
            metrics_file = os.path.join(self.storage_path, 'metrics.jsonl')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        metric = PerformanceMetric(
                            metric_type=PerformanceMetricType(data['metric_type']),
                            value=data['value'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            context=data.get('context', {}),
                            team_member_id=data.get('team_member_id'),
                            task_id=data.get('task_id')
                        )
                        self.metrics.append(metric)

            tasks_file = os.path.join(self.storage_path, 'tasks.jsonl')
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        record = TaskPerformanceRecord(
                            task_id=data['task_id'],
                            team_member_id=data['team_member_id'],
                            start_time=datetime.fromisoformat(data['start_time']),
                            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
                            success=data['success'],
                            quality_score=data['quality_score'],
                            time_to_solve=data.get('time_to_solve'),
                            specializations=[SpecializationType(s) for s in data.get('specializations', [])],
                            difficulty_level=data.get('difficulty_level', 0.5),
                            context=data.get('context', {})
                        )
                        self.task_records.append(record)

            logger.info(f"Loaded {len(self.metrics)} metrics and {len(self.task_records)} task records")

        except (IOError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error loading metrics: {e}")

    def record_metric(
        self,
        metric_type: PerformanceMetricType,
        value: float,
        team_member_id: Optional[str] = None,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetric:
        """
        Record a performance metric.

        Args:
            metric_type: Type of metric
            value: Metric value
            team_member_id: Optional team member ID
            task_id: Optional task ID
            context: Additional context

        Returns:
            The recorded metric
        """
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            team_member_id=team_member_id,
            task_id=task_id
        )

        with self.lock:
            self.metrics.append(metric)
            self._persist_metric(metric)

        return metric

    def _persist_metric(self, metric: PerformanceMetric):
        """Persist metric to storage"""
        try:
            metrics_file = os.path.join(self.storage_path, 'metrics.jsonl')
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metric.to_dict()) + '\n')
        except (IOError, TypeError) as e:
            logger.error(f"Error persisting metric: {e}")

    def start_task_tracking(
        self,
        task_id: str,
        team_member_id: str,
        specializations: List[SpecializationType],
        difficulty_level: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskPerformanceRecord:
        """
        Start tracking a task.

        Args:
            task_id: Task identifier
            team_member_id: Team member ID
            specializations: List of specializations involved
            difficulty_level: Task difficulty (0-1)
            context: Additional context

        Returns:
            The task record
        """
        record = TaskPerformanceRecord(
            task_id=task_id,
            team_member_id=team_member_id,
            start_time=datetime.now(),
            end_time=None,
            success=False,
            quality_score=0.0,
            time_to_solve=None,
            specializations=specializations,
            difficulty_level=difficulty_level,
            context=context or {}
        )

        with self.lock:
            self.task_records.append(record)
            self._persist_task_record(record)

        return record

    def complete_task_tracking(
        self,
        task_id: str,
        success: bool,
        quality_score: float
    ) -> Optional[TaskPerformanceRecord]:
        """
        Complete task tracking.

        Args:
            task_id: Task identifier
            success: Whether task was successful
            quality_score: Quality score (0-100)

        Returns:
            The updated task record, or None if not found
        """
        with self.lock:
            for record in self.task_records:
                if record.task_id == task_id and record.end_time is None:
                    record.end_time = datetime.now()
                    record.success = success
                    record.quality_score = quality_score
                    record.time_to_solve = (
                        record.end_time - record.start_time
                    ).total_seconds()

                    # Record derived metrics
                    self.record_metric(
                        PerformanceMetricType.TIME_TO_SOLVE,
                        record.time_to_solve,
                        team_member_id=record.team_member_id,
                        task_id=task_id
                    )

                    self.record_metric(
                        PerformanceMetricType.SUCCESS_RATE,
                        1.0 if success else 0.0,
                        team_member_id=record.team_member_id,
                        task_id=task_id
                    )

                    self.record_metric(
                        PerformanceMetricType.QUALITY_SCORE,
                        quality_score,
                        team_member_id=record.team_member_id,
                        task_id=task_id
                    )

                    self._persist_task_record(record)
                    return record

        return None

    def _persist_task_record(self, record: TaskPerformanceRecord):
        """Persist task record to storage"""
        try:
            tasks_file = os.path.join(self.storage_path, 'tasks.jsonl')
            # Append only if it's a new record or completed
            if record.end_time is None:
                with open(tasks_file, 'a') as f:
                    f.write(json.dumps(record.to_dict()) + '\n')
            else:
                # For completed records, we need to update the file
                # In production, this would use a proper database
                pass
        except (IOError, TypeError) as e:
            logger.error(f"Error persisting task record: {e}")

    def get_success_rate(
        self,
        team_member_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> float:
        """
        Calculate success rate.

        Args:
            team_member_id: Optional team member filter
            time_window: Optional time window

        Returns:
            Success rate (0-1)
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        relevant_records = [
            r for r in self.task_records
            if r.end_time is not None
            and (team_member_id is None or r.team_member_id == team_member_id)
            and (cutoff_time is None or r.end_time >= cutoff_time)
        ]

        if not relevant_records:
            return 0.0

        return sum(1 for r in relevant_records if r.success) / len(relevant_records)

    def get_average_time_to_solve(
        self,
        team_member_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> float:
        """
        Calculate average time to solve.

        Args:
            team_member_id: Optional team member filter
            time_window: Optional time window

        Returns:
            Average time in seconds
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        times = [
            r.time_to_solve
            for r in self.task_records
            if r.time_to_solve is not None
            and (team_member_id is None or r.team_member_id == team_member_id)
            and (cutoff_time is None or r.end_time >= cutoff_time)
        ]

        if not times:
            return 0.0

        return statistics.mean(times)

    def get_average_quality_score(
        self,
        team_member_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> float:
        """
        Calculate average quality score.

        Args:
            team_member_id: Optional team member filter
            time_window: Optional time window

        Returns:
            Average quality score (0-100)
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        scores = [
            r.quality_score
            for r in self.task_records
            if r.end_time is not None
            and (team_member_id is None or r.team_member_id == team_member_id)
            and (cutoff_time is None or r.end_time >= cutoff_time)
        ]

        if not scores:
            return 0.0

        return statistics.mean(scores)


class TeamMemberPerformance:
    """
    Track individual team member performance.

    Features:
    - Historical performance tracking
    - Specialization effectiveness
    - Strengths and weaknesses analysis
    - Performance trends
    - Reliability scoring
    """

    def __init__(self, team_member_id: str, metrics: PerformanceMetrics):
        """
        Initialize team member performance tracking.

        Args:
            team_member_id: Team member identifier
            metrics: Performance metrics instance
        """
        self.team_member_id = team_member_id
        self.metrics = metrics
        self.specialization_scores: Dict[SpecializationType, List[float]] = defaultdict(list)
        self.performance_history: List[Dict[str, Any]] = []

    def update_specialization_score(
        self,
        specialization: SpecializationType,
        score: float
    ):
        """
        Update specialization performance score.

        Args:
            specialization: Specialization type
            score: Performance score (0-100)
        """
        self.specialization_scores[specialization].append(score)

    def get_specialization_effectiveness(
        self,
        specialization: Optional[SpecializationType] = None
    ) -> Dict[str, float]:
        """
        Get specialization effectiveness scores.

        Args:
            specialization: Optional specific specialization

        Returns:
            Dictionary of specialization to effectiveness score
        """
        if specialization:
            scores = self.specialization_scores.get(specialization, [])
            return {
                specialization.value: statistics.mean(scores) if scores else 0.0
            }

        return {
            spec.value: statistics.mean(scores) if scores else 0.0
            for spec, scores in self.specialization_scores.items()
        }

    def get_performance_trend(
        self,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance trend over recent tasks.

        Args:
            window_size: Number of recent tasks to analyze

        Returns:
            Trend analysis dictionary
        """
        # Get recent task records for this member
        recent_records = [
            r for r in self.metrics.task_records
            if r.team_member_id == self.team_member_id
            and r.end_time is not None
        ][-window_size:]

        if len(recent_records) < 2:
            return {
                'trend': 'insufficient_data',
                'improvement_rate': 0.0,
                'current_performance': 0.0
            }

        # Calculate trend in quality scores
        quality_scores = [r.quality_score for r in recent_records]
        first_half = quality_scores[:len(quality_scores)//2]
        second_half = quality_scores[len(quality_scores)//2:]

        improvement_rate = (
            statistics.mean(second_half) - statistics.mean(first_half)
        ) / (statistics.mean(first_half) + 1e-6)

        current_performance = statistics.mean(quality_scores[-5:])

        return {
            'trend': 'improving' if improvement_rate > 0.05 else 'declining' if improvement_rate < -0.05 else 'stable',
            'improvement_rate': improvement_rate,
            'current_performance': current_performance,
            'sample_size': len(recent_records)
        }

    def get_strengths_and_weaknesses(
        self,
        min_samples: int = 3
    ) -> Tuple[List[SpecializationType], List[SpecializationType]]:
        """
        Identify team member's strengths and weaknesses.

        Args:
            min_samples: Minimum samples required for analysis

        Returns:
            Tuple of (strengths, weaknesses) lists
        """
        effectiveness = self.get_specialization_effectiveness()

        # Filter specializations with sufficient data
        valid_specializations = {
            SpecializationType(k): v
            for k, v in effectiveness.items()
            if len(self.specialization_scores.get(SpecializationType(k), [])) >= min_samples
        }

        if not valid_specializations:
            return [], []

        # Calculate overall median
        scores = list(valid_specializations.values())
        median_score = statistics.median(scores)

        # Strengths: above median
        strengths = [
            spec for spec, score in valid_specializations.items()
            if score > median_score + 10
        ]

        # Weaknesses: below median
        weaknesses = [
            spec for spec, score in valid_specializations.items()
            if score < median_score - 10
        ]

        return strengths, weaknesses

    def calculate_reliability_score(self) -> float:
        """
        Calculate overall reliability score.

        Returns:
            Reliability score (0-100)
        """
        # Get success rate
        success_rate = self.metrics.get_success_rate(team_member_id=self.team_member_id)

        # Get consistency (low variance in quality scores)
        quality_scores = [
            r.quality_score
            for r in self.metrics.task_records
            if r.team_member_id == self.team_member_id
            and r.end_time is not None
        ]

        if len(quality_scores) < 2:
            consistency_score = 50.0
        else:
            std_dev = statistics.stdev(quality_scores)
            # Lower standard deviation = higher consistency
            consistency_score = max(0, 100 - std_dev)

        # Get average quality
        avg_quality = self.metrics.get_average_quality_score(team_member_id=self.team_member_id)

        # Combined reliability score
        reliability = (
            success_rate * 40 +
            (avg_quality / 100) * 40 +
            consistency_score * 20
        )

        return min(100, max(0, reliability))


class PerformanceAnalytics:
    """
    Team-level performance analytics.

    Features:
    - Workload distribution analysis
    - Bottleneck identification
    - Performance optimization recommendations
    - Team composition analysis
    - Predictive modeling
    """

    def __init__(self, metrics: PerformanceMetrics):
        """
        Initialize performance analytics.

        Args:
            metrics: Performance metrics instance
        """
        self.metrics = metrics
        self.team_members: Dict[str, TeamMemberPerformance] = {}

    def register_team_member(self, team_member_id: str) -> TeamMemberPerformance:
        """
        Register a team member for tracking.

        Args:
            team_member_id: Team member identifier

        Returns:
            TeamMemberPerformance instance
        """
        if team_member_id not in self.team_members:
            self.team_members[team_member_id] = TeamMemberPerformance(
                team_member_id, self.metrics
            )

        return self.team_members[team_member_id]

    def analyze_workload_distribution(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze workload distribution across team members.

        Args:
            time_window: Optional time window to analyze

        Returns:
            Workload distribution analysis
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        # Count tasks per team member
        task_counts = defaultdict(int)
        total_time = defaultdict(float)

        for record in self.metrics.task_records:
            if record.end_time is not None:
                if cutoff_time is None or record.start_time >= cutoff_time:
                    task_counts[record.team_member_id] += 1
                    if record.time_to_solve:
                        total_time[record.team_member_id] += record.time_to_solve

        if not task_counts:
            return {
                'total_tasks': 0,
                'distribution': {},
                'imbalance_score': 0.0,
                'recommendation': 'No tasks to analyze'
            }

        # Calculate balance
        counts = list(task_counts.values())
        mean_tasks = statistics.mean(counts)
        std_tasks = statistics.stdev(counts) if len(counts) > 1 else 0

        imbalance_score = (std_tasks / (mean_tasks + 1e-6)) * 100

        distribution = {
            member_id: {
                'task_count': count,
                'total_time': total_time.get(member_id, 0),
                'avg_time_per_task': total_time.get(member_id, 0) / count if count > 0 else 0
            }
            for member_id, count in task_counts.items()
        }

        return {
            'total_tasks': sum(counts),
            'distribution': distribution,
            'imbalance_score': imbalance_score,
            'recommendation': self._get_workload_recommendation(imbalance_score)
        }

    def _get_workload_recommendation(self, imbalance_score: float) -> str:
        """Get workload distribution recommendation"""
        if imbalance_score < 20:
            return "Workload is well balanced"
        elif imbalance_score < 50:
            return "Moderate imbalance - consider redistributing some tasks"
        else:
            return "Significant imbalance - urgently redistribute workload"

    def identify_bottlenecks(
        self,
        min_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            min_samples: Minimum samples required for analysis

        Returns:
            List of bottleneck findings
        """
        bottlenecks = []

        # Analyze each team member
        for member_id, member in self.team_members.items():
            records = [
                r for r in self.metrics.task_records
                if r.team_member_id == member_id
                and r.end_time is not None
            ]

            if len(records) < min_samples:
                continue

            # Check for slow solve times
            avg_time = statistics.mean([r.time_to_solve for r in records if r.time_to_solve])
            team_avg_time = self.metrics.get_average_time_to_solve()

            if avg_time > team_avg_time * 1.5:
                bottlenecks.append({
                    'type': 'slow_solving',
                    'team_member_id': member_id,
                    'severity': 'high',
                    'description': f'Solve time ({avg_time:.1f}s) is 50% above team average',
                    'recommendation': 'Consider additional training or task reassignment'
                })

            # Check for low success rate
            success_rate = self.metrics.get_success_rate(team_member_id=member_id)
            if success_rate < 0.7:
                bottlenecks.append({
                    'type': 'low_success_rate',
                    'team_member_id': member_id,
                    'severity': 'critical',
                    'description': f'Success rate ({success_rate*100:.1f}%) is below 70%',
                    'recommendation': 'Review task difficulty and provide additional support'
                })

            # Check for low quality
            avg_quality = self.metrics.get_average_quality_score(team_member_id=member_id)
            if avg_quality < 60:
                bottlenecks.append({
                    'type': 'low_quality',
                    'team_member_id': member_id,
                    'severity': 'high',
                    'description': f'Average quality score ({avg_quality:.1f}) is below 60',
                    'recommendation': 'Review work quality and provide feedback'
                })

        return bottlenecks

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate performance optimization recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Workload imbalance
        workload_analysis = self.analyze_workload_distribution()
        if workload_analysis['imbalance_score'] > 30:
            recommendations.append({
                'category': 'workload_distribution',
                'priority': 'high',
                'recommendation': 'Redistribute tasks to balance workload',
                'expected_impact': 'Improved team efficiency and reduced burnout risk'
            })

        # Bottleneck resolution
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            recommendations.append({
                'category': 'bottleneck_resolution',
                'priority': 'critical',
                'recommendation': f'Address {len(bottlenecks)} identified bottlenecks',
                'expected_impact': 'Improved overall team performance',
                'bottlenecks': bottlenecks
            })

        # Specialization optimization
        for member_id, member in self.team_members.items():
            strengths, weaknesses = member.get_strengths_and_weaknesses()
            if weaknesses:
                recommendations.append({
                    'category': 'skill_development',
                    'priority': 'medium',
                    'team_member_id': member_id,
                    'recommendation': f'Provide training in weak areas: {[w.value for w in weaknesses]}',
                    'expected_impact': 'More balanced team capabilities'
                })

        return recommendations

    def predict_performance(
        self,
        team_member_id: str,
        task_specializations: List[SpecializationType],
        difficulty_level: float
    ) -> Dict[str, Any]:
        """
        Predict performance for a given task.

        Args:
            team_member_id: Team member ID
            task_specializations: Required specializations
            difficulty_level: Task difficulty (0-1)

        Returns:
            Performance prediction
        """
        if team_member_id not in self.team_members:
            return {
                'success_probability': 0.5,
                'expected_quality': 50.0,
                'expected_time': 300.0,
                'confidence': 'low'
            }

        member = self.team_members[team_member_id]

        # Get specialization effectiveness
        effectiveness = member.get_specialization_effectiveness()

        # Calculate task match score
        match_scores = []
        for spec in task_specializations:
            if spec.value in effectiveness:
                match_scores.append(effectiveness[spec.value] / 100)

        task_match = statistics.mean(match_scores) if match_scores else 0.5

        # Adjust for difficulty
        difficulty_adjustment = 1 - (difficulty_level * 0.3)

        # Predict success
        success_probability = min(0.95, max(0.05, task_match * difficulty_adjustment))

        # Predict quality
        base_quality = member.metrics.get_average_quality_score(team_member_id=team_member_id)
        expected_quality = base_quality * task_match * difficulty_adjustment

        # Predict time
        base_time = member.metrics.get_average_time_to_solve(team_member_id=team_member_id)
        difficulty_factor = 1 + (difficulty_level * 2)
        expected_time = base_time * difficulty_factor / (task_match + 0.1)

        return {
            'success_probability': success_probability,
            'expected_quality': min(100, max(0, expected_quality)),
            'expected_time': max(0, expected_time),
            'confidence': 'high' if len(member.metrics.task_records) > 10 else 'medium'
        }


class PerformanceReporter:
    """
    Generate comprehensive performance reports.

    Features:
    - Generate performance reports
    - Export metrics (JSON, CSV, HTML)
    - Trend analysis
    - Comparison reports
    """

    def __init__(self, metrics: PerformanceMetrics, analytics: PerformanceAnalytics):
        """
        Initialize performance reporter.

        Args:
            metrics: Performance metrics instance
            analytics: Performance analytics instance
        """
        self.metrics = metrics
        self.analytics = analytics

    def generate_team_report(
        self,
        time_window: Optional[timedelta] = None,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive team performance report.

        Args:
            time_window: Optional time window for analysis
            include_predictions: Whether to include predictions

        Returns:
            Team performance report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': time_window.days if time_window else None,
            'summary': self._generate_summary(time_window),
            'team_members': self._generate_team_member_reports(time_window),
            'workload_analysis': self.analytics.analyze_workload_distribution(time_window),
            'bottlenecks': self.analytics.identify_bottlenecks(),
            'recommendations': self.analytics.get_optimization_recommendations()
        }

        if include_predictions:
            report['predictions'] = self._generate_predictions()

        return report

    def _generate_summary(self, time_window: Optional[timedelta]) -> Dict[str, Any]:
        """Generate report summary"""
        return {
            'total_tasks': len([
                r for r in self.metrics.task_records
                if r.end_time is not None
                and (time_window is None or r.end_time >= datetime.now() - time_window)
            ]),
            'overall_success_rate': self.metrics.get_success_rate(time_window=time_window),
            'average_time_to_solve': self.metrics.get_average_time_to_solve(time_window=time_window),
            'average_quality_score': self.metrics.get_average_quality_score(time_window=time_window),
            'active_team_members': len(self.analytics.team_members)
        }

    def _generate_team_member_reports(
        self,
        time_window: Optional[timedelta]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate individual team member reports"""
        reports = {}

        for member_id, member in self.analytics.team_members.items():
            reports[member_id] = {
                'tasks_completed': len([
                    r for r in self.metrics.task_records
                    if r.team_member_id == member_id
                    and r.end_time is not None
                    and (time_window is None or r.end_time >= datetime.now() - time_window)
                ]),
                'success_rate': self.metrics.get_success_rate(
                    team_member_id=member_id,
                    time_window=time_window
                ),
                'average_quality': self.metrics.get_average_quality_score(
                    team_member_id=member_id,
                    time_window=time_window
                ),
                'average_time': self.metrics.get_average_time_to_solve(
                    team_member_id=member_id,
                    time_window=time_window
                ),
                'reliability_score': member.calculate_reliability_score(),
                'specialization_effectiveness': member.get_specialization_effectiveness(),
                'performance_trend': member.get_performance_trend()
            }

        return reports

    def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate performance predictions"""
        predictions = []

        # For each team member, predict performance on different task types
        for member_id in self.analytics.team_members.keys():
            for difficulty in [0.3, 0.5, 0.7, 0.9]:
                for specialization in list(SpecializationType)[:3]:  # Sample specializations
                    prediction = self.analytics.predict_performance(
                        member_id,
                        [specialization],
                        difficulty
                    )

                    predictions.append({
                        'team_member_id': member_id,
                        'specialization': specialization.value,
                        'difficulty_level': difficulty,
                        'prediction': prediction
                    })

        return predictions

    def export_json(
        self,
        report: Dict[str, Any],
        output_path: str
    ):
        """
        Export report to JSON.

        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report exported to {output_path}")

    def export_csv(
        self,
        report: Dict[str, Any],
        output_path: str
    ):
        """
        Export task performance data to CSV.

        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Task ID', 'Team Member ID', 'Start Time', 'End Time',
                'Success', 'Quality Score', 'Time to Solve',
                'Specializations', 'Difficulty Level'
            ])

            # Write data
            for record in self.metrics.task_records:
                if record.end_time:
                    writer.writerow([
                        record.task_id,
                        record.team_member_id,
                        record.start_time.isoformat(),
                        record.end_time.isoformat(),
                        record.success,
                        record.quality_score,
                        record.time_to_solve,
                        ','.join([s.value for s in record.specializations]),
                        record.difficulty_level
                    ])

        logger.info(f"CSV report exported to {output_path}")

    def export_html(
        self,
        report: Dict[str, Any],
        output_path: str
    ):
        """
        Export report to HTML.

        Args:
            report: Report dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Blue Team Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; margin-top: 30px; }
                .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }
                .metric-label { font-weight: bold; color: #7f8c8d; }
                .metric-value { font-size: 24px; color: #2c3e50; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #3498db; color: white; }
                tr:hover { background-color: #f5f5f5; }
                .recommendation { background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
                .bottleneck { background: #f8d7da; padding: 10px; margin: 10px 0; border-left: 4px solid #dc3545; }
            </style>
        </head>
        <body>
            <h1>Blue Team Performance Report</h1>
            <p>Generated: {generated_at}</p>

            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-label">Total Tasks</div>
                    <div class="metric-value">{total_tasks}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{success_rate:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Quality</div>
                    <div class="metric-value">{avg_quality:.1f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Time</div>
                    <div class="metric-value">{avg_time:.1f}s</div>
                </div>
            </div>

            <h2>Team Member Performance</h2>
            <table>
                <tr>
                    <th>Team Member</th>
                    <th>Tasks</th>
                    <th>Success Rate</th>
                    <th>Avg Quality</th>
                    <th>Reliability</th>
                </tr>
                {team_rows}
            </table>

            <h2>Recommendations</h2>
            {recommendations}

            <h2>Bottlenecks</h2>
            {bottlenecks}
        </body>
        </html>
        """

        # Format data
        summary = report['summary']
        team_members = report['team_members']

        team_rows = ""
        for member_id, data in team_members.items():
            team_rows += f"""
                <tr>
                    <td>{member_id}</td>
                    <td>{data['tasks_completed']}</td>
                    <td>{data['success_rate']*100:.1f}%</td>
                    <td>{data['average_quality']:.1f}</td>
                    <td>{data['reliability_score']:.1f}</td>
                </tr>
            """

        recommendations_html = ""
        for rec in report['recommendations']:
            recommendations_html += f"""
                <div class="recommendation">
                    <strong>{rec['category'].replace('_', ' ').title()}</strong> (Priority: {rec['priority']})<br>
                    {rec['recommendation']}
                </div>
            """

        bottlenecks_html = ""
        for bottleneck in report['bottlenecks']:
            bottlenecks_html += f"""
                <div class="bottleneck">
                    <strong>{bottleneck['type'].replace('_', ' ').title()}</strong> - {bottleneck['team_member_id']}<br>
                    Severity: {bottleneck['severity']}<br>
                    {bottleneck['description']}<br>
                    <em>Recommendation: {bottleneck['recommendation']}</em>
                </div>
            """

        # Fill template
        html_content = html.format(
            generated_at=report['generated_at'],
            total_tasks=summary['total_tasks'],
            success_rate=summary['overall_success_rate'] * 100,
            avg_quality=summary['average_quality_score'],
            avg_time=summary['average_time_to_solve'],
            team_rows=team_rows,
            recommendations=recommendations_html,
            bottlenecks=bottlenecks_html
        )

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report exported to {output_path}")

    def generate_comparison_report(
        self,
        team_member_ids: List[str],
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison report between team members.

        Args:
            team_member_ids: List of team member IDs to compare
            time_window: Optional time window

        Returns:
            Comparison report
        """
        comparison = {
            'generated_at': datetime.now().isoformat(),
            'team_members': team_member_ids,
            'metrics': {}
        }

        # Compare each metric
        metrics_to_compare = [
            ('success_rate', 'Success Rate'),
            ('average_quality', 'Average Quality'),
            ('average_time', 'Average Time to Solve')
        ]

        for metric_key, metric_name in metrics_to_compare:
            comparison['metrics'][metric_key] = {}

            for member_id in team_member_ids:
                if metric_key == 'success_rate':
                    value = self.metrics.get_success_rate(
                        team_member_id=member_id,
                        time_window=time_window
                    )
                elif metric_key == 'average_quality':
                    value = self.metrics.get_average_quality_score(
                        team_member_id=member_id,
                        time_window=time_window
                    )
                else:  # average_time
                    value = self.metrics.get_average_time_to_solve(
                        team_member_id=member_id,
                        time_window=time_window
                    )

                comparison['metrics'][metric_key][member_id] = value

        # Determine rankings
        comparison['rankings'] = {}
        for metric_key in comparison['metrics']:
            values = {
                k: v for k, v in comparison['metrics'][metric_key].items()
                if v > 0  # Exclude members with no data
            }

            if values:
                # Sort (descending for success/quality, ascending for time)
                reverse = metric_key != 'average_time'
                sorted_members = sorted(
                    values.items(),
                    key=lambda x: x[1],
                    reverse=reverse
                )

                comparison['rankings'][metric_key] = [
                    {'member_id': m, 'value': v, 'rank': i+1}
                    for i, (m, v) in enumerate(sorted_members)
                ]

        return comparison


class PerformanceAlertManager:
    """
    Manage performance alerts.

    Features:
    - Real-time monitoring
    - Threshold-based alerting
    - Alert recommendations
    - Alert history
    """

    def __init__(self, metrics: PerformanceMetrics):
        """
        Initialize alert manager.

        Args:
            metrics: Performance metrics instance
        """
        self.metrics = metrics
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[PerformanceMetricType, Tuple[float, float]] = {
            PerformanceMetricType.SUCCESS_RATE: (0.7, 0.5),  # Warning, Critical
            PerformanceMetricType.TIME_TO_SOLVE: (600, 1200),  # Warning, Critical (seconds)
            PerformanceMetricType.QUALITY_SCORE: (60, 40),  # Warning, Critical
            PerformanceMetricType.RELIABILITY: (60, 40),  # Warning, Critical
        }
        self.alert_handlers: List[Callable] = []

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """
        Add alert handler callback.

        Args:
            handler: Callback function
        """
        self.alert_handlers.append(handler)

    def check_thresholds(
        self,
        team_member_id: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """
        Check if any thresholds are breached.

        Args:
            team_member_id: Optional team member to check

        Returns:
            List of new alerts
        """
        new_alerts = []

        # Check success rate
        success_rate = self.metrics.get_success_rate(team_member_id=team_member_id)
        warning_threshold, critical_threshold = self.thresholds[PerformanceMetricType.SUCCESS_RATE]

        if success_rate < critical_threshold:
            alert = self._create_alert(
                AlertLevel.CRITICAL,
                PerformanceMetricType.SUCCESS_RATE,
                f"Success rate ({success_rate*100:.1f}%) is below critical threshold ({critical_threshold*100:.1f}%)",
                team_member_id,
                critical_threshold,
                success_rate
            )
            new_alerts.append(alert)
        elif success_rate < warning_threshold:
            alert = self._create_alert(
                AlertLevel.WARNING,
                PerformanceMetricType.SUCCESS_RATE,
                f"Success rate ({success_rate*100:.1f}%) is below warning threshold ({warning_threshold*100:.1f}%)",
                team_member_id,
                warning_threshold,
                success_rate
            )
            new_alerts.append(alert)

        # Check quality score
        quality_score = self.metrics.get_average_quality_score(team_member_id=team_member_id)
        warning_threshold, critical_threshold = self.thresholds[PerformanceMetricType.QUALITY_SCORE]

        if quality_score < critical_threshold:
            alert = self._create_alert(
                AlertLevel.CRITICAL,
                PerformanceMetricType.QUALITY_SCORE,
                f"Quality score ({quality_score:.1f}) is below critical threshold ({critical_threshold:.1f})",
                team_member_id,
                critical_threshold,
                quality_score
            )
            new_alerts.append(alert)
        elif quality_score < warning_threshold:
            alert = self._create_alert(
                AlertLevel.WARNING,
                PerformanceMetricType.QUALITY_SCORE,
                f"Quality score ({quality_score:.1f}) is below warning threshold ({warning_threshold:.1f})",
                team_member_id,
                warning_threshold,
                quality_score
            )
            new_alerts.append(alert)

        # Store alerts
        self.alerts.extend(new_alerts)

        # Trigger handlers
        for alert in new_alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except (RuntimeError, TypeError, ValueError) as e:
                    logger.error(f"Error in alert handler: {e}")

        return new_alerts

    def _create_alert(
        self,
        level: AlertLevel,
        metric_type: PerformanceMetricType,
        message: str,
        team_member_id: Optional[str],
        threshold_value: float,
        actual_value: float
    ) -> PerformanceAlert:
        """Create a performance alert"""
        alert_id = f"{metric_type.value}_{team_member_id}_{int(time.time())}"

        recommendations = self._generate_recommendations(metric_type, level, actual_value)

        return PerformanceAlert(
            alert_id=alert_id,
            level=level,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now(),
            team_member_id=team_member_id,
            threshold_value=threshold_value,
            actual_value=actual_value,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        metric_type: PerformanceMetricType,
        level: AlertLevel,
        actual_value: float
    ) -> List[str]:
        """Generate alert recommendations"""
        recommendations = []

        if metric_type == PerformanceMetricType.SUCCESS_RATE:
            recommendations = [
                "Review recent failures to identify patterns",
                "Consider providing additional training or resources",
                "Evaluate task difficulty alignment"
            ]
        elif metric_type == PerformanceMetricType.QUALITY_SCORE:
            recommendations = [
                "Review quality assessment criteria",
                "Provide detailed feedback on recent work",
                "Consider pairing with high-performing team members"
            ]
        elif metric_type == PerformanceMetricType.TIME_TO_SOLVE:
            recommendations = [
                "Analyze time-consuming steps in the workflow",
                "Check for available tools or automation opportunities",
                "Review task complexity estimates"
            ]

        return recommendations

    def get_alert_history(
        self,
        team_member_id: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        time_window: Optional[timedelta] = None
    ) -> List[PerformanceAlert]:
        """
        Get alert history.

        Args:
            team_member_id: Optional team member filter
            level: Optional level filter
            time_window: Optional time window

        Returns:
            List of alerts
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        filtered_alerts = [
            alert for alert in self.alerts
            if (team_member_id is None or alert.team_member_id == team_member_id)
            and (level is None or alert.level == level)
            and (cutoff_time is None or alert.timestamp >= cutoff_time)
        ]

        return filtered_alerts


@contextmanager
def track_blue_team_performance(
    tracker: 'BlueTeamPerformanceTracker',
    task_id: str,
    team_member_id: str,
    specializations: List[SpecializationType],
    difficulty_level: float = 0.5
):
    """
    Context manager for tracking Blue Team task performance.

    Args:
        tracker: Performance tracker instance
        task_id: Task identifier
        team_member_id: Team member ID
        specializations: List of specializations
        difficulty_level: Task difficulty

    Yields:
        Task performance record
    """
    # Start tracking
    record = tracker.start_task(
        task_id=task_id,
        team_member_id=team_member_id,
        specializations=specializations,
        difficulty_level=difficulty_level
    )

    try:
        yield record

        # Task completed successfully (unless explicitly failed)
        success = True
        quality_score = 75.0  # Default quality

        tracker.complete_task(
            task_id=task_id,
            success=success,
            quality_score=quality_score
        )

    except (RuntimeError, ValueError, TypeError, IOError) as e:
        # Task failed
        tracker.complete_task(task_id=task_id, success=False, quality_score=0.0)
        raise


class BlueTeamPerformanceTracker:
    """
    Main entry point for Blue Team performance tracking.

    Integrates all performance tracking components:
    - Metrics tracking
    - Team member analysis
    - Analytics
    - Reporting
    - Alert management
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize performance tracker.

        Args:
            storage_path: Path for storing performance data
        """
        self.metrics = PerformanceMetrics(storage_path)
        self.analytics = PerformanceAnalytics(self.metrics)
        self.reporter = PerformanceReporter(self.metrics, self.analytics)
        self.alert_manager = PerformanceAlertManager(self.metrics)

        logger.info("Blue Team Performance Tracker initialized")

    def register_team_member(self, team_member_id: str) -> TeamMemberPerformance:
        """
        Register a team member for tracking.

        Args:
            team_member_id: Team member identifier

        Returns:
            TeamMemberPerformance instance
        """
        return self.analytics.register_team_member(team_member_id)

    def start_task(
        self,
        task_id: str,
        team_member_id: str,
        specializations: List[SpecializationType],
        difficulty_level: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskPerformanceRecord:
        """
        Start tracking a task.

        Args:
            task_id: Task identifier
            team_member_id: Team member ID
            specializations: Required specializations
            difficulty_level: Task difficulty (0-1)
            context: Additional context

        Returns:
            Task performance record
        """
        # Ensure team member is registered
        if team_member_id not in self.analytics.team_members:
            self.register_team_member(team_member_id)

        # Start tracking
        record = self.metrics.start_task_tracking(
            task_id=task_id,
            team_member_id=team_member_id,
            specializations=specializations,
            difficulty_level=difficulty_level,
            context=context
        )

        return record

    def complete_task(
        self,
        task_id: str,
        success: bool,
        quality_score: float
    ) -> Optional[TaskPerformanceRecord]:
        """
        Complete task tracking.

        Args:
            task_id: Task identifier
            success: Whether task was successful
            quality_score: Quality score (0-100)

        Returns:
            Updated task record, or None if not found
        """
        record = self.metrics.complete_task_tracking(task_id, success, quality_score)

        if record:
            # Update specialization scores
            member = self.analytics.team_members.get(record.team_member_id)
            if member:
                for spec in record.specializations:
                    member.update_specialization_score(spec, quality_score)

            # Check for alerts
            self.alert_manager.check_thresholds(team_member_id=record.team_member_id)

        return record

    def get_team_member_performance(
        self,
        team_member_id: str
    ) -> Optional[TeamMemberPerformance]:
        """
        Get team member performance data.

        Args:
            team_member_id: Team member ID

        Returns:
            TeamMemberPerformance instance, or None if not found
        """
        return self.analytics.team_members.get(team_member_id)

    def generate_report(
        self,
        time_window_days: Optional[int] = None,
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate performance report.

        Args:
            time_window_days: Optional time window in days
            format: Report format ('json', 'csv', 'html', 'dict')
            output_path: Optional output file path

        Returns:
            Report dictionary
        """
        time_window = timedelta(days=time_window_days) if time_window_days else None

        report = self.reporter.generate_team_report(
            time_window=time_window,
            include_predictions=True
        )

        if output_path:
            if format == 'json':
                self.reporter.export_json(report, output_path)
            elif format == 'csv':
                self.reporter.export_csv(report, output_path)
            elif format == 'html':
                self.reporter.export_html(report, output_path)

        return report

    def get_optimal_team_member(
        self,
        required_specializations: List[SpecializationType],
        difficulty_level: float,
        exclude_members: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select optimal team member for a task.

        Args:
            required_specializations: Required specializations
            difficulty_level: Task difficulty (0-1)
            exclude_members: Optional list of members to exclude

        Returns:
            Best team member ID, or None if no members available
        """
        exclude_members = exclude_members or []

        predictions = []
        for member_id in self.analytics.team_members.keys():
            if member_id in exclude_members:
                continue

            prediction = self.analytics.predict_performance(
                member_id,
                required_specializations,
                difficulty_level
            )

            predictions.append({
                'member_id': member_id,
                'score': (
                    prediction['success_probability'] * 0.5 +
                    (prediction['expected_quality'] / 100) * 0.3 +
                    (1 / (prediction['expected_time'] + 1)) * 0.2
                ),
                'prediction': prediction
            })

        if not predictions:
            return None

        # Sort by score
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[0]['member_id']

    def check_performance_alerts(
        self,
        team_member_id: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """
        Check for performance alerts.

        Args:
            team_member_id: Optional team member to check

        Returns:
            List of new alerts
        """
        return self.alert_manager.check_thresholds(team_member_id)

    def get_workload_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get workload distribution recommendations.

        Returns:
            List of recommendations
        """
        return self.analytics.get_optimization_recommendations()


# Convenience functions for common operations
def create_performance_tracker(storage_path: Optional[str] = None) -> BlueTeamPerformanceTracker:
    """
    Create a Blue Team performance tracker.

    Args:
        storage_path: Optional storage path

    Returns:
        BlueTeamPerformanceTracker instance
    """
    return BlueTeamPerformanceTracker(storage_path)


def quick_performance_report(
    tracker: BlueTeamPerformanceTracker,
    days: int = 7
) -> str:
    """
    Generate a quick performance report summary.

    Args:
        tracker: Performance tracker
        days: Number of days to analyze

    Returns:
        Report summary string
    """
    report = tracker.generate_report(time_window_days=days)

    summary = report['summary']
    lines = [
        f"Blue Team Performance Report ({days} days)",
        "=" * 50,
        f"Total Tasks: {summary['total_tasks']}",
        f"Success Rate: {summary['overall_success_rate']*100:.1f}%",
        f"Avg Quality: {summary['average_quality_score']:.1f}",
        f"Avg Time: {summary['average_time_to_solve']:.1f}s",
        f"Active Members: {summary['active_team_members']}",
    ]

    if report['recommendations']:
        lines.append("\nRecommendations:")
        for rec in report['recommendations'][:3]:
            lines.append(f"  - {rec['recommendation']}")

    return "\n".join(lines)


# =============================================================================
# TEST COMPATIBILITY CLASS
# =============================================================================

class PerformanceTracker:
    """Wrapper class for test compatibility."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.metrics = []
    
    def track_metric(self, name: str, value: float):
        """Track a metric."""
        self.metrics.append({'name': name, 'value': value, 'timestamp': time.time()})
    
    def get_metrics(self) -> list:
        """Get all metrics."""
        return self.metrics
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = [m['value'] for m in self.metrics if m['name'] == metric_name]
        return sum(values) / len(values) if values else 0.0
