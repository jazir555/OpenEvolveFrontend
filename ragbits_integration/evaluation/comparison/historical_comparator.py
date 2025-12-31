"""
Historical Comparator

Compares current solutions with historical solutions to identify
patterns, improvements, and regressions.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    EvaluationMetricsCollector,
    MetricSet,
    MetricCategory
)
from ragbits_integration.evaluation.metrics.metrics_analyzer import (
    MetricsAnalyzer,
    AnalysisReport
)
from ragbits_integration.evaluation.gauntlets.enhanced_gauntlet import (
    MultiDimensionalScore
)

logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of comparisons"""
    CURRENT_VS_HISTORICAL = "current_vs_historical"
    TREND_ANALYSIS = "trend_analysis"
    PEER_COMPARISON = "peer_comparison"
    BASELINE_COMPARISON = "baseline_comparison"


@dataclass
class ComparisonMetric:
    """A comparison metric"""
    metric_name: str
    current_value: float
    historical_value: float
    difference: float
    percent_change: float
    trend: str  # "improving", "declining", "stable"
    significance: str  # "significant", "moderate", "minimal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "historical_value": self.historical_value,
            "difference": self.difference,
            "percent_change": self.percent_change,
            "trend": self.trend,
            "significance": self.significance
        }


@dataclass
class ComparisonInsight:
    """An insight from comparison"""
    insight_type: str  # "improvement", "regression", "anomaly", "pattern"
    description: str
    metrics: List[str]
    confidence: float  # 0-1
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "insight_type": self.insight_type,
            "description": self.description,
            "metrics": self.metrics,
            "confidence": self.confidence,
            "recommendation": self.recommendation
        }


@dataclass
class ComparisonReport:
    """Historical comparison report"""
    artifact_id: str
    comparison_type: ComparisonType
    current_score: float
    historical_scores: List[float]
    metrics: List[ComparisonMetric]
    insights: List[ComparisonInsight]
    percentile_rank: Optional[float] = None
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "comparison_type": self.comparison_type.value,
            "current_score": self.current_score,
            "historical_scores": self.historical_scores,
            "metrics": [m.to_dict() for m in self.metrics],
            "insights": [i.to_dict() for i in self.insights],
            "percentile_rank": self.percentile_rank,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """Generate text summary"""
        if not self.historical_scores:
            return "No historical data available for comparison"

        avg_historical = sum(self.historical_scores) / len(self.historical_scores)
        diff = self.current_score - avg_historical

        if diff > 0.5:
            return f"Significantly above historical average (+{diff:.2f})"
        elif diff > 0.1:
            return f"Above historical average (+{diff:.2f})"
        elif diff > -0.1:
            return f"Consistent with historical average ({diff:.2f})"
        elif diff > -0.5:
            return f"Below historical average ({diff:.2f})"
        else:
            return f"Significantly below historical average ({diff:.2f})"


class HistoricalComparator:
    """
    Compares current solutions with historical data.

    Usage:
        comparator = HistoricalComparator(
            metrics_collector,
            metrics_analyzer
        )

        # Compare current with historical
        report = await comparator.compare_with_historical(
            artifact_id="art_123",
            artifact_type="solution",
            lookback_days=30
        )
    """

    def __init__(
        self,
        metrics_collector: EvaluationMetricsCollector,
        metrics_analyzer: MetricsAnalyzer
    ):
        """
        Initialize historical comparator.

        Args:
            metrics_collector: Metrics collector
            metrics_analyzer: Metrics analyzer
        """
        self.metrics_collector = metrics_collector
        self.metrics_analyzer = metrics_analyzer

        logger.info("HistoricalComparator initialized")

    async def compare_with_historical(
        self,
        artifact_id: str,
        artifact_type: str,
        lookback_days: int = 30,
        limit: int = 50
    ) -> Optional[ComparisonReport]:
        """
        Compare current artifact with historical artifacts.

        Args:
            artifact_id: Current artifact ID
            artifact_type: Type of artifact
            lookback_days: Days to look back for historical data
            limit: Maximum historical artifacts to compare

        Returns:
            Comparison report or None
        """
        # Get current artifact metrics
        current_report = await self.metrics_analyzer.analyze_artifact(artifact_id)

        if not current_report:
            logger.warning(f"No metrics found for {artifact_id}")
            return None

        # Get historical metrics
        historical_metrics = await self.metrics_collector.get_historical_metrics(
            artifact_type=artifact_type,
            limit=limit
        )

        # Filter by lookback period
        import time
        cutoff_time = time.time() - (lookback_days * 24 * 3600)

        historical_metrics = [
            hm for hm in historical_metrics
            if hm.timestamp > cutoff_time and hm.artifact_id != artifact_id
        ]

        if not historical_metrics:
            logger.warning("No historical data available")
            return None

        # Analyze historical artifacts
        historical_scores = []

        for hm in historical_metrics[:20]:  # Limit to 20 for performance
            report = await self.metrics_analyzer.analyze_artifact(hm.artifact_id)
            if report:
                historical_scores.append(report.overall_score)

        if not historical_scores:
            return None

        # Compare metrics
        comparison_metrics = await self._compare_metrics(
            current_report,
            historical_metrics
        )

        # Generate insights
        insights = await self._generate_insights(
            current_report,
            historical_scores,
            comparison_metrics
        )

        # Calculate percentile rank
        percentile_rank = self._calculate_percentile_rank(
            current_report.overall_score,
            historical_scores
        )

        return ComparisonReport(
            artifact_id=artifact_id,
            comparison_type=ComparisonType.CURRENT_VS_HISTORICAL,
            current_score=current_report.overall_score,
            historical_scores=historical_scores,
            metrics=comparison_metrics,
            insights=insights,
            percentile_rank=percentile_rank,
            metadata={
                "artifact_type": artifact_type,
                "lookback_days": lookback_days,
                "historical_count": len(historical_scores),
                "average_historical": sum(historical_scores) / len(historical_scores)
            }
        )

    async def compare_peers(
        self,
        artifact_id: str,
        sub_problem_id: str
    ) -> Optional[ComparisonReport]:
        """
        Compare artifact with peer artifacts (same sub-problem).

        Args:
            artifact_id: Artifact to compare
            sub_problem_id: Sub-problem ID

        Returns:
            Comparison report or None
        """
        # Get current artifact
        current_report = await self.metrics_analyzer.analyze_artifact(artifact_id)

        if not current_report:
            return None

        # Get peer artifacts
        peer_metrics = await self.metrics_collector.get_metrics_by_subproblem(
            sub_problem_id
        )

        peer_metrics = [pm for pm in peer_metrics if pm.artifact_id != artifact_id]

        if not peer_metrics:
            return None

        # Analyze peers
        peer_scores = []

        for pm in peer_metrics:
            report = await self.metrics_analyzer.analyze_artifact(pm.artifact_id)
            if report:
                peer_scores.append(report.overall_score)

        if not peer_scores:
            return None

        # Generate comparison metrics
        comparison_metrics = await self._compare_metrics(
            current_report,
            peer_metrics
        )

        # Generate insights
        insights = await self._generate_peer_insights(
            current_report,
            peer_scores,
            comparison_metrics
        )

        # Calculate percentile rank
        percentile_rank = self._calculate_percentile_rank(
            current_report.overall_score,
            peer_scores
        )

        return ComparisonReport(
            artifact_id=artifact_id,
            comparison_type=ComparisonType.PEER_COMPARISON,
            current_score=current_report.overall_score,
            historical_scores=peer_scores,
            metrics=comparison_metrics,
            insights=insights,
            percentile_rank=percentile_rank,
            metadata={
                "sub_problem_id": sub_problem_id,
                "peer_count": len(peer_scores)
            }
        )

    async def analyze_trends(
        self,
        artifact_type: str,
        metric_category: Optional[MetricCategory] = None,
        window_size: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze trends in artifacts over time.

        Args:
            artifact_type: Type of artifact to analyze
            metric_category: Optional category to focus on
            window_size: Number of recent artifacts to analyze

        Returns:
            Trend analysis results
        """
        # Get historical metrics
        historical_metrics = await self.metrics_collector.get_historical_metrics(
            artifact_type=artifact_type,
            limit=window_size
        )

        if not historical_metrics:
            return {"error": "No historical data available"}

        # Sort by timestamp
        historical_metrics.sort(key=lambda x: x.timestamp)

        # Extract scores over time
        scores = []
        timestamps = []

        for hm in historical_metrics:
            report = await self.metrics_analyzer.analyze_artifact(hm.artifact_id)
            if report:
                if metric_category:
                    # Get category-specific score
                    cat_score = next(
                        (cs.score for cs in report.category_scores
                         if cs.category == metric_category),
                        None
                    )
                    if cat_score is not None:
                        scores.append(cat_score)
                        timestamps.append(hm.timestamp)
                else:
                    scores.append(report.overall_score)
                    timestamps.append(hm.timestamp)

        if not scores:
            return {"error": "No scores available for analysis"}

        # Calculate trend
        trend_analysis = self._calculate_trend(scores, timestamps)

        return {
            "artifact_type": artifact_type,
            "metric_category": metric_category.value if metric_category else "overall",
            "data_points": len(scores),
            "trend": trend_analysis,
            "scores": scores,
            "timestamps": timestamps
        }

    async def _compare_metrics(
        self,
        current_report: AnalysisReport,
        historical_metrics: List[MetricSet]
    ) -> List[ComparisonMetric]:
        """Compare current metrics with historical"""
        comparison_metrics = []

        # Analyze historical to get category scores
        historical_category_scores = {cat: [] for cat in MetricCategory}

        for hm in historical_metrics[:20]:
            report = await self.metrics_analyzer.analyze_artifact(hm.artifact_id)
            if report:
                for cs in report.category_scores:
                    historical_category_scores[cs.category].append(cs.score)

        # Compare each category
        for cs in current_report.category_scores:
            hist_scores = historical_category_scores.get(cs.category, [])

            if hist_scores:
                avg_hist = sum(hist_scores) / len(hist_scores)

                comparison_metrics.append(ComparisonMetric(
                    metric_name=f"{cs.category.value}_score",
                    current_value=cs.score,
                    historical_value=avg_hist,
                    difference=cs.score - avg_hist,
                    percent_change=((cs.score - avg_hist) / avg_hist * 100)
                                  if avg_hist > 0 else 0,
                    trend=self._determine_trend(cs.score, avg_hist),
                    significance=self._determine_significance(cs.score, hist_scores)
                ))

        return comparison_metrics

    async def _generate_insights(
        self,
        current_report: AnalysisReport,
        historical_scores: List[float],
        comparison_metrics: List[ComparisonMetric]
    ) -> List[ComparisonInsight]:
        """Generate insights from comparison"""
        insights = []

        avg_historical = sum(historical_scores) / len(historical_scores)
        diff = current_report.overall_score - avg_historical

        # Performance insight
        if diff > 0.3:
            insights.append(ComparisonInsight(
                insight_type="improvement",
                description=f"Solution performs {diff:.2f} points above historical average",
                metrics=["overall_score"],
                confidence=0.9,
                recommendation="Consider documenting the factors contributing to this improvement"
            ))
        elif diff < -0.3:
            insights.append(ComparisonInsight(
                insight_type="regression",
                description=f"Solution performs {abs(diff):.2f} points below historical average",
                metrics=["overall_score"],
                confidence=0.9,
                recommendation="Review and address factors causing below-average performance"
            ))

        # Category-specific insights
        for cm in comparison_metrics:
            if cm.percent_change > 20:
                insights.append(ComparisonInsight(
                    insight_type="improvement",
                    description=f"Strong improvement in {cm.metric_name}",
                    metrics=[cm.metric_name],
                    confidence=0.8
                ))
            elif cm.percent_change < -20:
                insights.append(ComparisonInsight(
                    insight_type="regression",
                    description=f"Significant decline in {cm.metric_name}",
                    metrics=[cm.metric_name],
                    confidence=0.8,
                    recommendation=f"Investigate and address {cm.metric_name} decline"
                ))

        return insights

    async def _generate_peer_insights(
        self,
        current_report: AnalysisReport,
        peer_scores: List[float],
        comparison_metrics: List[ComparisonMetric]
    ) -> List[ComparisonInsight]:
        """Generate insights from peer comparison"""
        insights = []

        avg_peer = sum(peer_scores) / len(peer_scores)
        rank = sorted(peer_scores + [current_report.overall_score], reverse=True).index(
            current_report.overall_score
        ) + 1
        percentile = (len(peer_scores) - rank + 1) / len(peer_scores) * 100

        # Ranking insight
        if percentile >= 80:
            insights.append(ComparisonInsight(
                insight_type="improvement",
                description=f"Solution ranks in top {100-percentile:.0f}% of peer solutions",
                metrics=["overall_score"],
                confidence=0.95
            ))
        elif percentile <= 20:
            insights.append(ComparisonInsight(
                insight_type="regression",
                description=f"Solution ranks in bottom {percentile:.0f}% of peer solutions",
                metrics=["overall_score"],
                confidence=0.95,
                recommendation="Study top-performing peer solutions for improvement opportunities"
            ))

        return insights

    def _determine_trend(
        self,
        current: float,
        historical_avg: float
    ) -> str:
        """Determine trend direction"""
        if current > historical_avg * 1.05:
            return "improving"
        elif current < historical_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _determine_significance(
        self,
        current: float,
        historical_scores: List[float]
    ) -> str:
        """Determine statistical significance"""
        if not historical_scores:
            return "minimal"

        import statistics

        avg_hist = sum(historical_scores) / len(historical_scores)

        try:
            std_dev = statistics.stdev(historical_scores)

            if std_dev == 0:
                return "minimal"

            z_score = abs(current - avg_hist) / std_dev

            if z_score > 2:
                return "significant"
            elif z_score > 1:
                return "moderate"
            else:
                return "minimal"
        except statistics.StatisticsError:
            return "minimal"

    def _calculate_percentile_rank(
        self,
        current_score: float,
        historical_scores: List[float]
    ) -> float:
        """Calculate percentile rank"""
        if not historical_scores:
            return None

        rank = sorted(historical_scores + [current_score]).index(current_score)
        percentile = (rank / len(historical_scores + [current_score])) * 100

        return percentile

    def _calculate_trend(
        self,
        scores: List[float],
        timestamps: List[float]
    ) -> Dict[str, Any]:
        """Calculate trend statistics"""
        if len(scores) < 2:
            return {"direction": "insufficient_data"}

        # Simple linear regression to determine trend
        n = len(scores)
        x_values = list(range(n))

        sum_x = sum(x_values)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(x_values, scores))
        sum_x2 = sum(x ** 2 for x in x_values)

        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Determine direction
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": slope,
            "start_score": scores[0],
            "end_score": scores[-1],
            "change": scores[-1] - scores[0]
        }
