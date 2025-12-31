"""
Metrics Analyzer

Analyzes collected metrics and generates comprehensive reports.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    MetricSet,
    MetricCategory,
    MetricType,
    EvaluationMetricsCollector
)

logger = logging.getLogger(__name__)


@dataclass
class CategoryScore:
    """Score for a metric category"""
    category: MetricCategory
    score: float
    weight: float = 1.0
    metric_count: int = 0
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "category": self.category.value,
            "score": self.score,
            "weight": self.weight,
            "metric_count": self.metric_count,
            "issues": self.issues,
            "strengths": self.strengths
        }


@dataclass
class AnalysisReport:
    """Comprehensive analysis report"""
    artifact_id: str
    overall_score: float
    category_scores: List[CategoryScore]
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "overall_score": self.overall_score,
            "category_scores": [cs.to_dict() for cs in self.category_scores],
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
            "critical_issues": self.critical_issues,
            "strengths": self.strengths,
            "metadata": self.metadata
        }


class MetricsAnalyzer:
    """
    Analyzes metrics and generates comprehensive reports.

    Usage:
        analyzer = MetricsAnalyzer(metrics_collector)

        # Analyze artifact metrics
        report = await analyzer.analyze_artifact("art_123")

        # Compare artifacts
        comparison = await analyzer.compare_artifacts("art_123", "art_456")
    """

    # Default weights for each category
    DEFAULT_CATEGORY_WEIGHTS = {
        MetricCategory.QUALITY: 1.0,
        MetricCategory.PERFORMANCE: 0.8,
        MetricCategory.RELIABILITY: 1.0,
        MetricCategory.SECURITY: 1.2,
        MetricCategory.COMPLETENESS: 1.0,
        MetricCategory.EFFICIENCY: 0.9,
        MetricCategory.MAINTAINABILITY: 0.7,
        MetricCategory.SCALABILITY: 0.8
    }

    # Thresholds for scoring
    SCORE_THRESHOLDS = {
        "excellent": 0.9,
        "good": 0.75,
        "satisfactory": 0.6,
        "poor": 0.4
    }

    def __init__(
        self,
        metrics_collector: EvaluationMetricsCollector,
        category_weights: Optional[Dict[MetricCategory, float]] = None
    ):
        """
        Initialize metrics analyzer.

        Args:
            metrics_collector: Metrics collector to analyze
            category_weights: Optional custom category weights
        """
        self.metrics_collector = metrics_collector
        self.category_weights = category_weights or self.DEFAULT_CATEGORY_WEIGHTS

        logger.info("MetricsAnalyzer initialized")

    async def analyze_artifact(
        self,
        artifact_id: str,
        include_recommendations: bool = True
    ) -> Optional[AnalysisReport]:
        """
        Analyze metrics for an artifact and generate report.

        Args:
            artifact_id: Artifact to analyze
            include_recommendations: Whether to generate recommendations

        Returns:
            Analysis report or None if artifact not found
        """
        metric_set = await self.metrics_collector.get_metrics(artifact_id)

        if not metric_set:
            logger.warning(f"No metrics found for artifact {artifact_id}")
            return None

        # Analyze each category
        category_scores = []

        for category in MetricCategory:
            category_metrics = metric_set.get_metrics_by_category(category)

            if category_metrics:
                score = self._calculate_category_score(category_metrics)
                issues, strengths = self._analyze_category_metrics(
                    category_metrics,
                    score
                )

                category_scores.append(CategoryScore(
                    category=category,
                    score=score,
                    weight=self.category_weights[category],
                    metric_count=len(category_metrics),
                    issues=issues,
                    strengths=strengths
                ))

        # Calculate overall weighted score
        overall_score = self._calculate_overall_score(category_scores)

        # Generate recommendations
        recommendations = []
        critical_issues = []
        all_strengths = []

        if include_recommendations:
            recommendations, critical_issues = self._generate_recommendations(
                category_scores
            )

            all_strengths = [
                strength
                for cs in category_scores
                for strength in cs.strengths
            ]

        report = AnalysisReport(
            artifact_id=artifact_id,
            overall_score=overall_score,
            category_scores=category_scores,
            recommendations=recommendations,
            critical_issues=critical_issues,
            strengths=all_strengths,
            metadata={
                "artifact_type": metric_set.artifact_type,
                "sub_problem_id": metric_set.sub_problem_id,
                "workflow_stage": metric_set.workflow_stage,
                "metric_count": len(metric_set.metrics)
            }
        )

        logger.info(
            f"Generated analysis report for {artifact_id}: "
            f"overall_score={overall_score:.2f}"
        )

        return report

    def _calculate_category_score(
        self,
        metrics: List
    ) -> float:
        """Calculate score for a metric category"""
        if not metrics:
            return 0.0

        # Normalize values to 0-1 range
        normalized_values = []

        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                # Use min/max if available
                if metric.min_value is not None and metric.max_value is not None:
                    range_size = metric.max_value - metric.min_value
                    if range_size > 0:
                        normalized = (metric.value - metric.min_value) / range_size
                    else:
                        normalized = 1.0 if metric.value >= metric.max_value else 0.0
                else:
                    # Assume percentage (0-100) or use sigmoid
                    if metric.value <= 1.0:
                        normalized = metric.value
                    elif metric.value <= 100.0:
                        normalized = metric.value / 100.0
                    else:
                        # Use sigmoid for unbounded values
                        import math
                        normalized = 1 / (1 + math.exp(-0.01 * (metric.value - 50)))

                normalized_values.append(normalized)

        if not normalized_values:
            return 0.0

        return sum(normalized_values) / len(normalized_values)

    def _analyze_category_metrics(
        self,
        metrics: List,
        score: float
    ) -> tuple[List[str], List[str]]:
        """Analyze metrics and identify issues and strengths"""
        issues = []
        strengths = []

        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                # Check if value is normalized
                if metric.min_value is not None and metric.max_value is not None:
                    normalized = (metric.value - metric.min_value) / (metric.max_value - metric.min_value)
                elif metric.value <= 1.0:
                    normalized = metric.value
                elif metric.value <= 100.0:
                    normalized = metric.value / 100.0
                else:
                    continue

                # Identify issues (below threshold)
                if normalized < self.SCORE_THRESHOLDS["poor"]:
                    issues.append(
                        f"Low {metric.metric_type.value}: {metric.value} "
                        f"(threshold: {self.SCORE_THRESHOLDS['poor'] * 100:.0f}%)"
                    )

                # Identify strengths (above excellent threshold)
                if normalized >= self.SCORE_THRESHOLDS["excellent"]:
                    strengths.append(
                        f"Strong {metric.metric_type.value}: {metric.value}"
                    )

        return issues, strengths

    def _calculate_overall_score(
        self,
        category_scores: List[CategoryScore]
    ) -> float:
        """Calculate overall weighted score"""
        if not category_scores:
            return 0.0

        total_weight = sum(cs.weight for cs in category_scores)
        weighted_sum = sum(cs.score * cs.weight for cs in category_scores)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(
        self,
        category_scores: List[CategoryScore]
    ) -> tuple[List[str], List[str]]:
        """Generate recommendations and identify critical issues"""
        recommendations = []
        critical_issues = []

        for cs in category_scores:
            # Critical issues (very low scores)
            if cs.score < self.SCORE_THRESHOLDS["poor"]:
                critical_issues.append(
                    f"Critical: {cs.category.value} score is {cs.score:.2f} "
                    f"(below {self.SCORE_THRESHOLDS['poor'] * 100:.0f}% threshold)"
                )

            # Recommendations for improvement
            if cs.score < self.SCORE_THRESHOLDS["good"]:
                recommendations.append(
                    f"Improve {cs.category.value}: "
                    f"current score {cs.score:.2f}, target {self.SCORE_THRESHOLDS['good']:.2f}"
                )

        # Add general recommendations
        if not critical_issues and len(recommendations) == 0:
            recommendations.append("All metrics are within acceptable ranges")

        return recommendations, critical_issues

    async def compare_artifacts(
        self,
        artifact_id_1: str,
        artifact_id_2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare two artifacts' metrics.

        Args:
            artifact_id_1: First artifact
            artifact_id_2: Second artifact

        Returns:
            Comparison dict or None if artifacts not found
        """
        report_1 = await self.analyze_artifact(artifact_id_1)
        report_2 = await self.analyze_artifact(artifact_id_2)

        if not report_1 or not report_2:
            return None

        # Calculate overall difference
        score_diff = report_2.overall_score - report_1.overall_score

        # Compare category scores
        category_comparison = []

        for cs1 in report_1.category_scores:
            cs2 = next(
                (c for c in report_2.category_scores if c.category == cs1.category),
                None
            )

            if cs2:
                category_comparison.append({
                    "category": cs1.category.value,
                    "artifact_1_score": cs1.score,
                    "artifact_2_score": cs2.score,
                    "difference": cs2.score - cs1.score,
                    "improved": cs2.score > cs1.score
                })

        return {
            "artifact_1": artifact_id_1,
            "artifact_2": artifact_id_2,
            "overall_score_diff": score_diff,
            "artifact_1_improved": score_diff < 0,
            "category_comparison": category_comparison,
            "summary": self._generate_comparison_summary(
                report_1,
                report_2,
                score_diff
            )
        }

    def _generate_comparison_summary(
        self,
        report_1: AnalysisReport,
        report_2: AnalysisReport,
        score_diff: float
    ) -> str:
        """Generate human-readable comparison summary"""
        if abs(score_diff) < 0.05:
            return f"Both artifacts perform similarly (score diff: {score_diff:.2f})"
        elif score_diff > 0:
            return (
                f"{report_2.artifact_id} outperforms {report_1.artifact_id} "
                f"by {score_diff:.2f} points"
            )
        else:
            return (
                f"{report_1.artifact_id} outperforms {report_2.artifact_id} "
                f"by {-score_diff:.2f} points"
            )

    async def analyze_subproblem(
        self,
        sub_problem_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze all artifacts for a sub-problem.

        Args:
            sub_problem_id: Sub-problem to analyze

        Returns:
            Aggregated analysis
        """
        metric_sets = await self.metrics_collector.get_metrics_by_subproblem(
            sub_problem_id
        )

        if not metric_sets:
            return None

        # Analyze each artifact
        reports = []

        for ms in metric_sets:
            report = await self.analyze_artifact(ms.artifact_id)
            if report:
                reports.append(report)

        if not reports:
            return None

        # Calculate aggregate scores
        avg_score = sum(r.overall_score for r in reports) / len(reports)

        # Find best and worst performing
        best = max(reports, key=lambda r: r.overall_score)
        worst = min(reports, key=lambda r: r.overall_score)

        return {
            "sub_problem_id": sub_problem_id,
            "artifact_count": len(reports),
            "average_score": avg_score,
            "best_artifact": {
                "id": best.artifact_id,
                "score": best.overall_score
            },
            "worst_artifact": {
                "id": worst.artifact_id,
                "score": worst.overall_score
            },
            "score_range": best.overall_score - worst.overall_score,
            "recommendations": self._generate_subproblem_recommendations(reports)
        }

    def _generate_subproblem_recommendations(
        self,
        reports: List[AnalysisReport]
    ) -> List[str]:
        """Generate recommendations for sub-problem improvement"""
        recommendations = []

        # Find common issues across artifacts
        all_issues = {}

        for report in reports:
            for cs in report.category_scores:
                if cs.score < self.SCORE_THRESHOLDS["satisfactory"]:
                    if cs.category not in all_issues:
                        all_issues[cs.category] = 0
                    all_issues[cs.category] += 1

        # Generate recommendations for problematic categories
        for category, count in all_issues.items():
            if count > len(reports) / 2:  # More than half of artifacts
                recommendations.append(
                    f"Multiple artifacts need improvement in {category.value} "
                    f"({count}/{len(reports)} artifacts below threshold)"
                )

        return recommendations
