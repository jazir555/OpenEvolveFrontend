"""
Evaluator Analytics Module for OpenEvolve

This module provides comprehensive analytics and metrics for the Evaluator Team,
including individual performance tracking, team-level analytics, bias detection,
and quality trend analysis.

Author: OpenEvolve
Date: 2025-01-04
"""
from __future__ import annotations


from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
from collections import defaultdict
import numpy as np
from scipy import stats

class EvaluationStage(Enum):
    """Evaluation stages in the workflow"""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    PLANNING = "planning"
    SOLUTION_GENERATION = "solution_generation"
    TESTING = "testing"
    VALIDATION = "validation"
    FINAL_REVIEW = "final_review"

class BiasType(Enum):
    """Types of evaluator biases"""
    LENIENCY = "leniency"  # Consistently higher scores
    SEVERITY = "severity"  # Consistently lower scores
    CENTRAL_TENDENCY = "central_tendency"  # Clustering around middle
    HALO_EFFECT = "halo_effect"  # Overall impression bias
    RECENCY = "recency"  # Recent evaluations influence current
    CONFIRMATION = "confirmation"  # Confirming preexisting beliefs
    TEMPORAL = "temporal"  # Time-based patterns
    SUBJECT_MATTER = "subject_matter"  # Domain-specific bias


@dataclass
class EvaluationRecord:
    """Record of a single evaluation"""
    evaluator_id: str
    evaluation_id: str
    stage: EvaluationStage
    timestamp: datetime
    score: float
    confidence: float
    time_taken: float  # seconds
    criteria_scores: Dict[str, float]
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "evaluator_id": self.evaluator_id,
            "evaluation_id": self.evaluation_id,
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "confidence": self.confidence,
            "time_taken": self.time_taken,
            "criteria_scores": self.criteria_scores,
            "feedback": self.feedback,
            "metadata": self.metadata
        }


@dataclass
class EvaluatorMetrics:
    """Comprehensive metrics for an evaluator"""
    evaluator_id: str
    total_evaluations: int = 0
    average_score: float = 0.0
    average_confidence: float = 0.0
    average_time: float = 0.0
    accuracy: float = 0.0
    consistency_score: float = 0.0
    reliability_score: float = 0.0
    bias_scores: Dict[str, float] = field(default_factory=dict)
    stage_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    time_trends: List[float] = field(default_factory=list)
    score_trends: List[float] = field(default_factory=list)
    last_evaluation: Optional[datetime] = None
    evaluation_frequency: float = 0.0  # evaluations per day
    # Ensemble-specific metrics
    ensemble_selection_count: int = 0  # Times selected by ensemble
    ensemble_weight: float = 1.0  # Current weight in ensemble
    ensemble_utilization: float = 0.0  # Utilization rate when available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "evaluator_id": self.evaluator_id,
            "total_evaluations": self.total_evaluations,
            "average_score": self.average_score,
            "average_confidence": self.average_confidence,
            "average_time": self.average_time,
            "accuracy": self.accuracy,
            "consistency_score": self.consistency_score,
            "reliability_score": self.reliability_score,
            "bias_scores": self.bias_scores,
            "stage_performance": self.stage_performance,
            "time_trends": self.time_trends,
            "score_trends": self.score_trends,
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "evaluation_frequency": self.evaluation_frequency,
            "ensemble_selection_count": self.ensemble_selection_count,
            "ensemble_weight": self.ensemble_weight,
            "ensemble_utilization": self.ensemble_utilization
        }


@dataclass
class BiasProfile:
    """Bias profile for an evaluator"""
    evaluator_id: str
    bias_type: BiasType
    severity: float  # 0-1 scale
    confidence: float
    description: str
    affected_stages: List[str]
    mitigation_suggestions: List[str]
    first_detected: datetime
    last_updated: datetime
    trend: str  # "increasing", "decreasing", "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "evaluator_id": self.evaluator_id,
            "bias_type": self.bias_type.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "affected_stages": self.affected_stages,
            "mitigation_suggestions": self.mitigation_suggestions,
            "first_detected": self.first_detected.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "trend": self.trend
        }


class BiasDetector:
    """
    Detects and analyzes evaluator biases using statistical methods
    and machine learning techniques.
    """

    def __init__(self, confidence_threshold: float = 0.05):
        """
        Initialize bias detector

        Args:
            confidence_threshold: Statistical significance threshold
        """
        self.confidence_threshold = confidence_threshold
        self.bias_history: Dict[str, List[BiasProfile]] = defaultdict(list)

    def detect_leniency_bias(
        self,
        evaluator_scores: List[float],
        team_scores: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect leniency bias (consistently higher scores)

        Args:
            evaluator_scores: Scores from this evaluator
            team_scores: Scores from all evaluators

        Returns:
            (has_bias, severity, description)
        """
        if len(evaluator_scores) < 5:
            return False, 0.0, "Insufficient data"

        evaluator_mean = np.mean(evaluator_scores)
        team_mean = np.mean(team_scores)

        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(evaluator_scores, team_mean)

        has_bias = p_value < self.confidence_threshold and evaluator_mean > team_mean
        severity = abs(evaluator_mean - team_mean) / max(np.std(team_scores), 0.1)
        description = f"Evaluator mean ({evaluator_mean:.2f}) significantly higher than team mean ({team_mean:.2f})"

        return has_bias, min(severity, 1.0), description

    def detect_severity_bias(
        self,
        evaluator_scores: List[float],
        team_scores: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect severity bias (consistently lower scores)

        Args:
            evaluator_scores: Scores from this evaluator
            team_scores: Scores from all evaluators

        Returns:
            (has_bias, severity, description)
        """
        if len(evaluator_scores) < 5:
            return False, 0.0, "Insufficient data"

        evaluator_mean = np.mean(evaluator_scores)
        team_mean = np.mean(team_scores)

        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(evaluator_scores, team_mean)

        has_bias = p_value < self.confidence_threshold and evaluator_mean < team_mean
        severity = abs(evaluator_mean - team_mean) / max(np.std(team_scores), 0.1)
        description = f"Evaluator mean ({evaluator_mean:.2f}) significantly lower than team mean ({team_mean:.2f})"

        return has_bias, min(severity, 1.0), description

    def detect_central_tendency_bias(
        self,
        evaluator_scores: List[float],
        scale_range: Tuple[float, float] = (0.0, 10.0)
    ) -> Tuple[bool, float, str]:
        """
        Detect central tendency bias (clustering around middle)

        Args:
            evaluator_scores: Scores from this evaluator
            scale_range: Min and max of scoring scale

        Returns:
            (has_bias, severity, description)
        """
        if len(evaluator_scores) < 5:
            return False, 0.0, "Insufficient data"

        scale_midpoint = (scale_range[0] + scale_range[1]) / 2
        deviations = [abs(score - scale_midpoint) for score in evaluator_scores]
        mean_deviation = np.mean(deviations)
        max_deviation = scale_range[1] - scale_midpoint

        # Calculate proportion of scores near midpoint
        near_midpoint = sum(1 for score in evaluator_scores
                           if abs(score - scale_midpoint) < (max_deviation * 0.2))
        proportion_near_mid = near_midpoint / len(evaluator_scores)

        has_bias = proportion_near_mid > 0.6
        severity = proportion_near_mid
        description = f"{proportion_near_mid*100:.1f}% of scores cluster around midpoint ({scale_midpoint:.1f})"

        return has_bias, severity, description

    def detect_temporal_bias(
        self,
        records: List[EvaluationRecord]
    ) -> Tuple[bool, float, str]:
        """
        Detect temporal bias (time-based patterns in scoring)

        Args:
            records: Evaluation records sorted by timestamp

        Returns:
            (has_bias, severity, description)
        """
        if len(records) < 10:
            return False, 0.0, "Insufficient data"

        # Split records into halves and compare means
        mid_point = len(records) // 2
        first_half = [r.score for r in records[:mid_point]]
        second_half = [r.score for r in records[mid_point:]]

        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(first_half, second_half)

        has_bias = p_value < self.confidence_threshold
        severity = abs(second_mean - first_mean) / max(np.std(first_half + second_half), 0.1)
        direction = "increased" if second_mean > first_mean else "decreased"
        description = f"Scores {direction} from {first_mean:.2f} to {second_mean:.2f} over time"

        return has_bias, min(severity, 1.0), description

    def generate_bias_profile(
        self,
        evaluator_id: str,
        records: List[EvaluationRecord],
        all_records: List[EvaluationRecord]
    ) -> List[BiasProfile]:
        """
        Generate comprehensive bias profile for an evaluator

        Args:
            evaluator_id: Evaluator ID
            records: Records for this evaluator
            all_records: All team records for comparison

        Returns:
            List of detected biases
        """
        evaluator_scores = [r.score for r in records]
        team_scores = [r.score for r in all_records]

        profiles = []
        now = datetime.now()

        # Check leniency bias
        has_bias, severity, desc = self.detect_leniency_bias(evaluator_scores, team_scores)
        if has_bias:
            profiles.append(BiasProfile(
                evaluator_id=evaluator_id,
                bias_type=BiasType.LENIENCY,
                severity=severity,
                confidence=0.95,
                description=desc,
                affected_stages=[r.stage.value for r in records],
                mitigation_suggestions=[
                    "Review scoring criteria and calibration",
                    "Participate in normalization sessions",
                    "Compare scores with team averages regularly"
                ],
                first_detected=now,
                last_updated=now,
                trend="stable"
            ))

        # Check severity bias
        has_bias, severity, desc = self.detect_severity_bias(evaluator_scores, team_scores)
        if has_bias:
            profiles.append(BiasProfile(
                evaluator_id=evaluator_id,
                bias_type=BiasType.SEVERITY,
                severity=severity,
                confidence=0.95,
                description=desc,
                affected_stages=[r.stage.value for r in records],
                mitigation_suggestions=[
                    "Review scoring criteria for strictness",
                    "Consider re-calibration training",
                    "Compare with peer evaluations"
                ],
                first_detected=now,
                last_updated=now,
                trend="stable"
            ))

        # Check central tendency bias
        has_bias, severity, desc = self.detect_central_tendency_bias(evaluator_scores)
        if has_bias:
            profiles.append(BiasProfile(
                evaluator_id=evaluator_id,
                bias_type=BiasType.CENTRAL_TENDENCY,
                severity=severity,
                confidence=0.90,
                description=desc,
                affected_stages=[r.stage.value for r in records],
                mitigation_suggestions=[
                    "Use full range of scoring scale",
                    "Practice extreme case scoring",
                    "Review score distribution regularly"
                ],
                first_detected=now,
                last_updated=now,
                trend="stable"
            ))

        # Check temporal bias
        has_bias, severity, desc = self.detect_temporal_bias(records)
        if has_bias:
            profiles.append(BiasProfile(
                evaluator_id=evaluator_id,
                bias_type=BiasType.TEMPORAL,
                severity=severity,
                confidence=0.90,
                description=desc,
                affected_stages=[r.stage.value for r in records],
                mitigation_suggestions=[
                    "Take breaks during long evaluation sessions",
                    "Review previous evaluations for consistency",
                    "Consider time-of-day effects"
                ],
                first_detected=now,
                last_updated=now,
                trend="monitoring"
            ))

        # Store in history
        self.bias_history[evaluator_id].extend(profiles)

        return profiles


class EvaluatorAnalytics:
    """
    Comprehensive analytics for evaluator performance tracking and analysis.
    """

    def __init__(self, knowledge_base=None):
        """
        Initialize evaluator analytics

        Args:
            knowledge_base: Optional knowledge base for persistence
        """
        self.knowledge_base = knowledge_base
        self.evaluation_records: Dict[str, List[EvaluationRecord]] = defaultdict(list)
        self.evaluator_metrics: Dict[str, EvaluatorMetrics] = {}
        self.bias_detector = BiasDetector()
        self.team_metrics: Dict[str, Any] = {}

    def add_evaluation_record(self, record: EvaluationRecord) -> None:
        """
        Add an evaluation record to analytics

        Args:
            record: Evaluation record to add
        """
        self.evaluation_records[record.evaluator_id].append(record)
        self._update_evaluator_metrics(record.evaluator_id)
        self._update_team_metrics()

        # Persist if knowledge base available
        if self.knowledge_base:
            self._persist_record(record)

    def get_evaluator_metrics(self, evaluator_id: str) -> Optional[EvaluatorMetrics]:
        """
        Get metrics for a specific evaluator

        Args:
            evaluator_id: Evaluator ID

        Returns:
            Evaluator metrics or None if not found
        """
        return self.evaluator_metrics.get(evaluator_id)

    def get_team_metrics(self) -> Dict[str, Any]:
        """
        Get team-level metrics

        Returns:
            Team metrics dictionary
        """
        return self.team_metrics

    def calculate_consistency_score(
        self,
        records: List[EvaluationRecord]
    ) -> float:
        """
        Calculate consistency score for an evaluator

        Consistency is measured by the standard deviation of scores
        adjusted for the range of criteria

        Args:
            records: Evaluation records

        Returns:
            Consistency score (0-1)
        """
        if len(records) < 2:
            return 1.0

        scores = [r.score for r in records]
        std_dev = np.std(scores)

        # Lower std_dev = higher consistency
        # Normalize assuming reasonable range of scores
        consistency = 1.0 / (1.0 + std_dev / 2.0)

        return float(consistency)

    def calculate_reliability_score(
        self,
        evaluator_id: str,
        records: List[EvaluationRecord]
    ) -> float:
        """
        Calculate reliability score (inter-rater reliability)

        Compares evaluator scores with team consensus

        Args:
            evaluator_id: Evaluator ID
            records: Evaluation records

        Returns:
            Reliability score (0-1)
        """
        if len(records) < 3:
            return 1.0

        # Get all evaluations for same items from other evaluators
        deviations = []
        for record in records:
            others = [r.score for r in self.evaluation_records.values()
                     for r in r if r.evaluation_id == record.evaluation_id
                     and r.evaluator_id != evaluator_id]
            if others:
                team_avg = np.mean(others)
                deviations.append(abs(record.score - team_avg))

        if not deviations:
            return 1.0

        mean_deviation = np.mean(deviations)
        reliability = 1.0 / (1.0 + mean_deviation / 2.0)

        return float(reliability)

    def analyze_performance_trends(
        self,
        evaluator_id: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance trends for an evaluator

        Args:
            evaluator_id: Evaluator ID
            window_size: Size of rolling window

        Returns:
            Trend analysis dictionary
        """
        if evaluator_id not in self.evaluation_records:
            return {"error": "Evaluator not found"}

        records = sorted(
            self.evaluation_records[evaluator_id],
            key=lambda x: x.timestamp
        )

        if len(records) < window_size:
            return {"error": "Insufficient data for trend analysis"}

        scores = [r.score for r in records]
        times = [r.time_taken for r in records]

        # Calculate moving averages
        score_ma = self._moving_average(scores, window_size)
        time_ma = self._moving_average(times, window_size)

        # Calculate trends (linear regression slope)
        x = list(range(len(scores)))
        score_slope, _, _, _, _ = stats.linregress(x, scores)
        time_slope, _, _, _, _ = stats.linregress(x, times)

        return {
            "score_trend": "improving" if score_slope > 0 else "declining" if score_slope < 0 else "stable",
            "score_slope": float(score_slope),
            "score_moving_average": score_ma,
            "time_trend": "faster" if time_slope < 0 else "slower" if time_slope > 0 else "stable",
            "time_slope": float(time_slope),
            "time_moving_average": time_ma,
            "window_size": window_size
        }

    def compare_evaluators(
        self,
        evaluator_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple evaluators across metrics

        Args:
            evaluator_ids: List of evaluator IDs to compare

        Returns:
            Comparison dictionary
        """
        comparison = {
            "evaluators": evaluator_ids,
            "metrics_comparison": {},
            "ranking": {}
        }

        metrics_dict = {}
        for eval_id in evaluator_ids:
            metrics = self.get_evaluator_metrics(eval_id)
            if metrics:
                metrics_dict[eval_id] = metrics

        # Compare across dimensions
        dimensions = [
            "average_score", "accuracy", "consistency_score",
            "reliability_score", "evaluation_frequency"
        ]

        for dimension in dimensions:
            values = {
                eval_id: getattr(metrics, dimension)
                for eval_id, metrics in metrics_dict.items()
            }
            comparison["metrics_comparison"][dimension] = values
            comparison["ranking"][dimension] = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=True
            )

        return comparison

    def detect_biases(
        self,
        evaluator_id: str
    ) -> List[BiasProfile]:
        """
        Detect biases for an evaluator

        Args:
            evaluator_id: Evaluator ID

        Returns:
            List of bias profiles
        """
        if evaluator_id not in self.evaluation_records:
            return []

        records = self.evaluation_records[evaluator_id]
        all_records = []
        for eval_records in self.evaluation_records.values():
            all_records.extend(eval_records)

        return self.bias_detector.generate_bias_profile(
            evaluator_id,
            records,
            all_records
        )

    def get_top_performers(
        self,
        metric: str = "accuracy",
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top performers by metric

        Args:
            metric: Metric to rank by
            top_n: Number of top performers to return

        Returns:
            List of (evaluator_id, metric_value) tuples
        """
        performers = []

        for eval_id, metrics in self.evaluator_metrics.items():
            value = getattr(metrics, metric, 0.0)
            performers.append((eval_id, value))

        return sorted(performers, key=lambda x: x[1], reverse=True)[:top_n]

    def get_stage_performance(
        self,
        stage: EvaluationStage
    ) -> Dict[str, Any]:
        """
        Get team performance for a specific stage

        Args:
            stage: Evaluation stage

        Returns:
            Stage performance dictionary
        """
        stage_records = []
        for records in self.evaluation_records.values():
            stage_records.extend([r for r in records if r.stage == stage])

        if not stage_records:
            return {"error": f"No data for stage {stage.value}"}

        scores = [r.score for r in stage_records]
        times = [r.time_taken for r in stage_records]
        confidences = [r.confidence for r in stage_records]

        return {
            "stage": stage.value,
            "total_evaluations": len(stage_records),
            "average_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "average_time": float(np.mean(times)),
            "average_confidence": float(np.mean(confidences)),
            "evaluator_count": len(set(r.evaluator_id for r in stage_records))
        }

    def generate_quality_report(
        self,
        evaluator_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for an evaluator

        Args:
            evaluator_id: Evaluator ID

        Returns:
            Quality report dictionary
        """
        metrics = self.get_evaluator_metrics(evaluator_id)
        if not metrics:
            return {"error": "Evaluator not found"}

        biases = self.detect_biases(evaluator_id)
        trends = self.analyze_performance_trends(evaluator_id)

        # Calculate overall quality score
        quality_components = [
            metrics.accuracy * 0.3,
            metrics.consistency_score * 0.2,
            metrics.reliability_score * 0.2,
            metrics.average_confidence * 0.1,
            (1.0 - np.mean([b.severity for b in biases]) if biases else 1.0) * 0.2
        ]

        overall_quality = float(np.sum(quality_components))

        return {
            "evaluator_id": evaluator_id,
            "overall_quality_score": overall_quality,
            "metrics": metrics.to_dict(),
            "biases": [b.to_dict() for b in biases],
            "trends": trends,
            "quality_components": {
                "accuracy_contribution": quality_components[0],
                "consistency_contribution": quality_components[1],
                "reliability_contribution": quality_components[2],
                "confidence_contribution": quality_components[3],
                "bias_contribution": quality_components[4]
            },
            "generated_at": datetime.now().isoformat()
        }

    def _update_evaluator_metrics(self, evaluator_id: str) -> None:
        """
        Update metrics for a specific evaluator

        Args:
            evaluator_id: Evaluator ID
        """
        records = self.evaluation_records[evaluator_id]

        if not records:
            return

        scores = [r.score for r in records]
        confidences = [r.confidence for r in records]
        times = [r.time_taken for r in records]

        metrics = EvaluatorMetrics(
            evaluator_id=evaluator_id,
            total_evaluations=len(records),
            average_score=float(np.mean(scores)),
            average_confidence=float(np.mean(confidences)),
            average_time=float(np.mean(times)),
            consistency_score=self.calculate_consistency_score(records),
            reliability_score=self.calculate_reliability_score(evaluator_id, records),
            last_evaluation=max(r.timestamp for r in records)
        )

        # Calculate stage-specific metrics
        stage_data = defaultdict(list)
        for record in records:
            stage_data[record.stage.value].append(record)

        stage_performance = {}
        for stage, stage_records in stage_data.items():
            stage_scores = [r.score for r in stage_records]
            stage_performance[stage] = {
                "count": len(stage_records),
                "average_score": float(np.mean(stage_scores)),
                "std_dev": float(np.std(stage_scores))
            }

        metrics.stage_performance = stage_performance

        # Calculate time trends
        sorted_records = sorted(records, key=lambda x: x.timestamp)
        metrics.time_trends = [r.time_taken for r in sorted_records[-20:]]
        metrics.score_trends = [r.score for r in sorted_records[-20:]]

        # Calculate evaluation frequency
        if len(records) > 1:
            time_span = (metrics.last_evaluation -
                        min(r.timestamp for r in records)).total_seconds()
            metrics.evaluation_frequency = len(records) / max(time_span / 86400, 1)  # per day

        self.evaluator_metrics[evaluator_id] = metrics

    def _update_team_metrics(self) -> None:
        """
        Update team-level metrics
        """
        all_records = []
        for records in self.evaluation_records.values():
            all_records.extend(records)

        if not all_records:
            return

        scores = [r.score for r in all_records]
        evaluators = set(r.evaluator_id for r in all_records)

        self.team_metrics = {
            "total_evaluations": len(all_records),
            "unique_evaluators": len(evaluators),
            "average_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "total_evaluation_time": float(sum(r.time_taken for r in all_records)),
            "average_evaluation_time": float(np.mean([r.time_taken for r in all_records])),
            "stages_represented": list(set(r.stage.value for r in all_records)),
            "last_updated": datetime.now().isoformat()
        }

    def _moving_average(
        self,
        data: List[float],
        window: int
    ) -> List[float]:
        """
        Calculate moving average

        Args:
            data: Data points
            window: Window size

        Returns:
            List of moving averages
        """
        if len(data) < window:
            return []

        return [
            float(np.mean(data[i:i+window]))
            for i in range(len(data) - window + 1)
        ]

    def _persist_record(self, record: EvaluationRecord) -> None:
        """
        Persist evaluation record to knowledge base

        Args:
            record: Record to persist
        """
        if self.knowledge_base:
            self.knowledge_base.store(
                key=f"evaluation_record_{record.evaluation_id}",
                value=record.to_dict(),
                category="evaluator_analytics"
            )

    def export_analytics(self) -> Dict[str, Any]:
        """
        Export all analytics data

        Returns:
            Analytics export dictionary
        """
        return {
            "evaluator_metrics": {
                eval_id: metrics.to_dict()
                for eval_id, metrics in self.evaluator_metrics.items()
            },
            "team_metrics": self.team_metrics,
            "bias_profiles": {
                eval_id: [b.to_dict() for b in profiles]
                for eval_id, profiles in self.bias_detector.bias_history.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }

    def load_analytics(self, data: Dict[str, Any]) -> None:
        """
        Load analytics data from export

        Args:
            data: Analytics export dictionary
        """
        # Load evaluator metrics
        for eval_id, metrics_dict in data.get("evaluator_metrics", {}).items():
            metrics = EvaluatorMetrics(
                evaluator_id=metrics_dict["evaluator_id"],
                total_evaluations=metrics_dict["total_evaluations"],
                average_score=metrics_dict["average_score"],
                average_confidence=metrics_dict["average_confidence"],
                average_time=metrics_dict["average_time"],
                accuracy=metrics_dict["accuracy"],
                consistency_score=metrics_dict["consistency_score"],
                reliability_score=metrics_dict["reliability_score"],
                bias_scores=metrics_dict.get("bias_scores", {}),
                stage_performance=metrics_dict.get("stage_performance", {}),
                time_trends=metrics_dict.get("time_trends", []),
                score_trends=metrics_dict.get("score_trends", []),
                evaluation_frequency=metrics_dict.get("evaluation_frequency", 0.0)
            )
            self.evaluator_metrics[eval_id] = metrics

        self.team_metrics = data.get("team_metrics", {})
